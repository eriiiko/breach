#version 330
// Lit-marine fragment shader — samples the ship's baked light field on the
// marine's TRUE mesh normals. Mirrors shaders/lighting.fs's field unpack +
// composite + tone-map, adapted to a Y-up 3D mesh with half-Lambert + rim.
in vec2 fragTexCoord;
in vec3 fragWorldPos;
in vec3 fragWorldNormal;

uniform sampler2D texture0;   // marine albedo (MATERIAL_MAP_ALBEDO)
uniform sampler2D texture1;   // light_tex_a: RGB incoming light, A = dir.x
uniform sampler2D texture2;   // light_tex_b: RGB smoke_glow, A = dir.y
uniform sampler2D texture3;   // P2 marine normal map (MATERIAL_MAP_ROUGHNESS
                              // slot — a FREE slot; light textures own 1 & 2)
uniform vec4  colDiffuse;     // group tint (draw color): marines / zombies

uniform vec3  u_ambient;       // ship's ambient floor (single source of truth)
uniform float u_light_gain;    // ship's render exposure
uniform float u_light_z;       // marine's OWN grazing key (lower than ship)
uniform float u_normal_y_sign; // P2 normal-map green-channel convention (marine's
                               // OWN copy of the ship's H-toggle)
uniform vec2  u_world_px;      // (world_px_w, world_px_h) for the field UV
uniform vec3  u_view_dir;      // direction toward the eye (ortho ~ (0,1,0))
uniform float u_rim_strength;
uniform float u_rim_power;
uniform int   u_srgb_decode;
uniform int   u_use_normal_marine; // P2 guard: 0 = inert (N unchanged), 1 = on
uniform float u_normal_strength;   // P2 perturbation strength (feel knob)

out vec4 finalColor;


// Cheap sRGB <-> linear (gamma 2.2), matching shaders/lighting.fs.
vec3 srgb_to_linear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 linear_to_srgb(vec3 c) { return pow(c, vec3(1.0 / 2.2)); }

// ACES filmic tone-map (Narkowicz), identical to shaders/lighting.fs.
vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}


// Tangent-free TBN from screen-space derivatives of world-pos + UV (Schüler,
// "Followup: Normal Mapping Without Precomputed Tangents", thetenthplanet.de).
// The Quaternius mesh has NO vertex tangents and CPU skinning would not skin
// them anyway, so we reconstruct a per-fragment cotangent frame from dFdx/dFdy.
// Known degeneracy: on surfaces edge-on to the camera (the marine's vertical
// sides / silhouette) the UV derivatives collapse and the frame is unreliable;
// at true top-down the visible surface is mostly top-facing, where it holds.
mat3 cotangent_frame(vec3 N, vec3 p, vec2 uv) {
    vec3 dp1 = dFdx(p);
    vec3 dp2 = dFdy(p);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    vec3 dp2perp = cross(dp2, N);
    vec3 dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    // Scale-invariant, NaN-guarded (degenerate/edge-on fragments -> ~identity).
    float invmax = inversesqrt(max(max(dot(T, T), dot(B, B)), 1e-8));
    return mat3(T * invmax, B * invmax, N);
}

void main() {
    // Foot-plane world UV: sample the baked field at the marine's XZ ground
    // position (height Y ignored). worldPos.x -> grid X, worldPos.z -> grid Y
    // (y-down). NO v-flip: the marine reads the same texture the ship reads
    // with the same world->grid mapping (numpy row 0 = grid y 0 = texture v 0
    // = screen top under both the RT quad and the top-down 3D camera), so the
    // marine's lighting is glued to the same tiles as the ship beside it.
    vec2 world_uv = fragWorldPos.xz / u_world_px;

    vec4 tex_a = texture(texture1, world_uv);
    vec4 tex_b = texture(texture2, world_uv);
    vec3 incoming_rgb = tex_a.rgb;              // total light colour at this tile
    vec2 light_dir_2d = vec2(tex_a.a, tex_b.a); // signed, already unit-length

    // 2D baked direction -> 3D in the marine's Y-up world frame: dir.x -> X,
    // u_light_z -> Y (up), dir.y -> Z. Same vector the ship builds as
    // vec3(dir, light_z), reordered because the ship's tangent frame is Z-up.
    vec3 L = normalize(vec3(light_dir_2d.x, u_light_z, light_dir_2d.y));
    vec3 N = normalize(fragWorldNormal);

    // P2 normal map (guarded — zero cost when off). Reconstruct a tangent frame
    // from screen-space derivatives, sample the map as LINEAR data (never sRGB-
    // decode a normal map — its RGB are directions), unpack, apply the marine's
    // own green-channel sign + strength, and perturb N BEFORE the N·L below.
    // Inert if no map is bound (a flat 1x1 default reads (0.5,0.5,1) -> +Z, i.e.
    // N unchanged after transform) or if u_normal_strength is 0.
    if (u_use_normal_marine == 1) {
        vec3 nmap = texture(texture3, fragTexCoord).rgb * 2.0 - 1.0;
        nmap.y *= u_normal_y_sign;
        nmap.xy *= u_normal_strength;
        mat3 TBN = cotangent_frame(N, fragWorldPos, fragTexCoord);
        N = normalize(TBN * nmap);
    }

    // Half-Lambert: softens the pre-normalized-direction instability that
    // side normals get under near-overhead lamps AND reads better on a small
    // top-down token (design v2 §Bugs).
    float ndotl = dot(N, L) * 0.5 + 0.5;

    vec3 albedo = texture(texture0, fragTexCoord).rgb;
    if (u_srgb_decode == 1) albedo = srgb_to_linear(albedo);  // else double-dark
    albedo *= colDiffuse.rgb;                                  // group identity

    vec3 lit = albedo * (u_ambient + incoming_rgb * u_light_gain * ndotl);

    // Rim: Fresnel-ish silhouette term, tinted by the LOCAL light so it never
    // brightens a marine the room around it can't (dark room -> faint ambient
    // rim only).
    float rim = pow(1.0 - max(0.0, dot(N, u_view_dir)), u_rim_power);
    lit += u_rim_strength * rim * (u_ambient + incoming_rgb * u_light_gain);

    lit = aces_tonemap(lit);
    if (u_srgb_decode == 1) lit = linear_to_srgb(lit);

    // alpha = 1.0: the world RT is blitted premultiplied; a translucent marine
    // would bleed the background through (design v2 §Bugs).
    finalColor = vec4(lit, 1.0);
}
