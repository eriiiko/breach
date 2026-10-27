#version 330
// Lit-marine fragment shader — samples the ship's baked light field on the
// marine's TRUE mesh normals. Mirrors shaders/lighting.fs's field unpack +
// composite, adapted to a Y-up 3D mesh with half-Lambert + rim. Writes linear
// HDR into the scene RT; the world resolve tone-maps (G0 P1).
in vec2 fragTexCoord;
in vec3 fragWorldPos;
in vec3 fragWorldNormal;

uniform sampler2D texture0;   // marine albedo (MATERIAL_MAP_ALBEDO)
uniform sampler2D texture1;   // light_tex_a: RGB incoming light, A = dir.x
uniform sampler2D texture2;   // light_tex_b: RGB smoke_glow, A = dir.y
uniform sampler2D texture3;   // P2 marine normal map (MATERIAL_MAP_ROUGHNESS
                              // slot — a FREE slot; light textures own 1 & 2)
uniform vec4  colDiffuse;     // draw colour (white since #33: looks are models)

uniform vec3  u_ambient;       // ship's ambient floor (single source of truth)
uniform float u_light_gain;    // ship's render exposure
uniform float u_light_z;       // marine's OWN grazing key (lower than ship)
uniform float u_normal_y_sign; // P2 normal-map green-channel convention (marine's
                               // OWN copy of the ship's H-toggle)
uniform vec2  u_world_px;      // (world_px_w, world_px_h) for the field UV
uniform vec3  u_view_dir;      // direction toward the eye (ortho ~ (0,1,0))
uniform float u_rim_strength;
uniform float u_rim_power;
uniform float u_rim_albedo;    // #33: rim colour = mix(white, albedo, this)
uniform float u_gloss_strength;  // #33: highlight strength (0 = no highlight)
uniform float u_gloss_shininess; // #33: highlight exponent at gloss 1
uniform int   u_srgb_decode;
uniform int   u_use_normal_marine; // P2 guard: 0 = inert (N unchanged), 1 = on
uniform float u_normal_strength;   // P2 perturbation strength (feel knob)
// #114 body light: 1 = the whole body lit from the unit's ONE light read
// (renderer/unit_shadow.py read_unit_light, set per unit before its draw);
// 0 = each fragment from the field tile under it (the pre-#114 A/B).
uniform int   u_body_light;
uniform vec3  u_body_dir_rgb;      // one-sided irradiance (x exposure)
uniform vec3  u_body_iso_rgb;      // balanced irradiance (x exposure)
uniform vec2  u_body_to_light;     // toward the light, unit; (0,0) = none

out vec4 finalColor;


// Cheap sRGB -> linear (gamma 2.2), matching shaders/lighting.fs. Output stays
// linear HDR: WorldComposite.resolve() tone-maps and encodes, once.
vec3 srgb_to_linear(vec3 c) { return pow(c, vec3(2.2)); }

// --- smoke cast shadow (graphics G2; renderer/lit3d.py _SMOKE_SHADOW_GLSL) ---
uniform sampler2D u_smoke_cover; // the relief's cast product: r,g = ceiling
                                 // (alpha, height), b,a = floor; tiles
uniform vec2  u_smoke_grid;      // (grid_w, grid_h), tiles
uniform vec4  u_smoke_cast;      // (ceiling strength, floor strength,
                                 //  parallax: run per tile of rise, ceiling lift)
uniform vec4  u_smoke_cast2;     // (tallest column, tiles; unused x3)
vec4 smokeCoverAt(vec2 world_uv) {
    // a RenderTexture's rows run bottom-up
    return texture(u_smoke_cover, vec2(world_uv.x, 1.0 - world_uv.y));
}
float smokeShadow(vec2 world_uv, vec2 to_light, float z_tiles) {
    if (u_smoke_cast.x + u_smoke_cast.y <= 0.0) return 1.0;
    vec2 step_uv = to_light / u_smoke_grid;      // world uv per tile of run
    float k = u_smoke_cast.z;
    float hmax = max(u_smoke_cast2.x, 1e-3);
    float shC = 0.0;
    float shF = 0.0;
    for (int i = 0; i < 3; ++i) {
        float f = (float(i) + 0.5) / 3.0;
        // ceiling slab [lift, lift + h]: the ray is in it for run s with
        // z + s / k in that band
        float sC = (max(u_smoke_cast.w - z_tiles, 0.0) + f * hmax) * k;
        vec4 c = smokeCoverAt(world_uv + step_uv * sC);
        float topC = u_smoke_cast.w + c.g - (z_tiles + sC / max(k, 1e-3));
        shC = max(shC, c.r * smoothstep(0.0, 0.25 * hmax, topC));
        // floor slab [0, h]
        float sF = f * max(hmax - z_tiles, 0.0) * k;
        vec4 q = smokeCoverAt(world_uv + step_uv * sF);
        float topF = q.a - (z_tiles + sF / max(k, 1e-3));
        shF = max(shF, q.b * smoothstep(0.0, 0.25 * hmax, topF));
    }
    return (1.0 - u_smoke_cast.x * shC) * (1.0 - u_smoke_cast.y * shF);
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

    // Graphics G2: the smoke's cast shadow on this fragment, at its own
    // height (world px -> tiles), through the one shared GLSL above.
    incoming_rgb *= smokeShadow(world_uv, light_dir_2d,
                                fragWorldPos.y * u_smoke_grid.x / u_world_px.x);

    // #114 body light. The unit is stamped opaque into the light sweep (its
    // floor shadow), so the field under the body holds light only in the
    // footprint row facing the lamp. In "unit" mode the body's light is the
    // unit's own read instead -- the same read its projected shadow takes --
    // so the whole body is lit and its lit side agrees with the shadow. The
    // smoke's cast shadow stays PER FRAGMENT, toward the read's light.
    // Both parts are already in exposure units (x u_light_gain).
    vec3 light_dir_rgb = incoming_rgb * u_light_gain;
    vec3 light_iso_rgb = vec3(0.0);
    if (u_body_light == 1) {
        float smoke_k = smokeShadow(world_uv, u_body_to_light,
                                    fragWorldPos.y * u_smoke_grid.x / u_world_px.x);
        light_dir_rgb = u_body_dir_rgb * smoke_k;
        light_iso_rgb = u_body_iso_rgb * smoke_k;
        L = normalize(vec3(u_body_to_light.x, u_light_z, u_body_to_light.y));
    }

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
    // #114: the balanced share's weight -- the same half-Lambert averaged over
    // every horizontal direction at the key's height (exact: it is linear in
    // L), so a ring of equal lamps lights every side alike and the top most.
    float ndotl_iso = 0.5 + 0.5 * N.y * u_light_z / sqrt(1.0 + u_light_z * u_light_z);
    vec3 light_rgb = light_dir_rgb + light_iso_rgb;   // all of it, x exposure

    vec4 albedo_gloss = texture(texture0, fragTexCoord);
    vec3 albedo = albedo_gloss.rgb;
    float gloss = albedo_gloss.a;  // #33: the bake's gloss mask (0 matte .. 1 glossiest)
    if (u_srgb_decode == 1) albedo = srgb_to_linear(albedo);  // else double-dark
    albedo *= colDiffuse.rgb;                                  // draw colour

    vec3 lit = albedo * (u_ambient + light_dir_rgb * ndotl
                         + light_iso_rgb * ndotl_iso);

    // #33 highlight (Blinn-Phong): the LOCAL light's own colour, never the
    // ambient (a dark room puts no shine on a visor), scaled AND tightened by
    // the texel's gloss -- a visor gets a small bright spot, a plate a broad
    // faint sheen, cloth nothing. Faded out where the surface turns away from
    // the light (dot(N, L) < 0) so the lobe never lights a back face.
    vec3 H = normalize(L + u_view_dir);
    float shininess = mix(4.0, u_gloss_shininess, gloss);
    float spec = u_gloss_strength * gloss
               * pow(max(dot(N, H), 0.0), shininess)
               * clamp(dot(N, L) * 4.0, 0.0, 1.0);
    lit += spec * light_rgb;

    // Rim: Fresnel-ish silhouette term, tinted by the LOCAL light so it never
    // brightens a marine the room around it can't (dark room -> faint ambient
    // rim only), and by the surface's own colour (#33, u_rim_albedo) so dark
    // cloth keeps a dark edge instead of reading light grey from above.
    float rim = pow(1.0 - max(0.0, dot(N, u_view_dir)), u_rim_power);
    vec3 rim_col = mix(vec3(1.0), albedo, u_rim_albedo);
    lit += u_rim_strength * rim * rim_col * (u_ambient + light_rgb);

    // Linear HDR out (G0 P1): the world resolve tone-maps and encodes, once.

    // alpha = 1.0: the world RT is blitted premultiplied; a translucent marine
    // would bleed the background through (design v2 §Bugs). The texture's
    // alpha is the gloss mask (#33), never coverage.
    finalColor = vec4(lit, 1.0);
}
