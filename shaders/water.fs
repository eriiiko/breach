#version 330
// Water surface optics pass for Breach (graphics/water_rendering.md §2, §7
// steps 1+2 — the CORE look: perturbed-ripple normal + Schlick Fresnel + GGX
// glints (reusing the light buffer) + normal-driven refraction × depth +
// Beer-Lambert depth tint). Caustics / foam / chromatic-aberration / matcap
// are the NEXT step (§7 step 3/4) and are deliberately ABSENT here.
//
// This is a SEPARATE pass from lighting.fs (§5): it draws a full-RT quad in
// the WaterFieldOverlay compose slot, AFTER the lit ship, BEFORE units, and
// REPLACES the CPU-tinted WaterFieldOverlay placeholder.
//
// DORMANT-SAFE (the hard requirement): the pass outputs a PREMULTIPLIED vec4
// with alpha = 0 on every DRY tile (water_depth == 0). The world RT composites
// premultiplied (out = rt.rgb + bg*(1-rt.a)); alpha 0 + rgb 0 is a no-op, so a
// ship with no standing water renders bit-identically to no pass at all.
//
// Input textures (all reused from the lighting pass except u_water):
//   u_diffuse  - RGB UNLIT ship art (sRGB PNG), sampled at fragTexCoord
//                (raylib binds texture0 here implicitly).
//   u_light_a  - RGBA16F light field A: RGB = incoming light colour,
//                A = light_dir.x (signed). (lighting.fs:100,104)
//   u_light_b  - RGBA16F light field B: RGB = smoke_glow (unused here),
//                A = light_dir.y (signed). (lighting.fs:101,104)
//   u_water    - RGBA16F water field (renderer/water.py packing):
//                R = ripple height (m), G = ripple_v (reserved, unused here),
//                B = water_depth (m), A = foam/agitation (reserved, unused).
//
// Uniforms ([graphics.water] config block, bound in renderer/water.py):
//   u_roughness_base       GGX roughness floor (still puddles glint sharp)
//   u_roughness_agitation  added roughness per unit local ripple energy
//   u_fog_density          Beer-Lambert extinction (depth tint rate)
//   u_refract_strength     normal-driven floor-warp magnitude (UV units)
//   u_r0                   Fresnel reflectance at normal incidence (~0.02)
//   u_water_color          deep-water tint colour (the volume colour at depth)
//   u_light_z              light-direction z (shared with lighting.fs)
//   u_srgb_decode          1 = decode sRGB diffuse to linear, re-encode out
//   u_ripple_scale         metres-of-ripple -> normal-slope gain
//   u_texel                1/grid (x,y): neighbour tap offset for the gradient
//   u_art_uv_rect          art-UV subrect (matches lighting.fs; default 0,0,1,1)
//   u_time                 render animation clock (s) for the ambient sines
//   u_glint_strength       ADDITIVE GGX-glint HDR multiplier (light off surface)
//   u_alpha_scale          depth->opacity rate (alpha = clamp(depth*scale,...))
//   u_alpha_min            shoreline alpha floor   (transparency dial)
//   u_alpha_max            deep-water alpha ceiling (transparency dial)

in vec2 fragTexCoord;
in vec4 fragColor;

uniform sampler2D u_diffuse;
uniform sampler2D u_light_a;
uniform sampler2D u_light_b;
uniform sampler2D u_water;

uniform float u_roughness_base;
uniform float u_roughness_agitation;
uniform float u_fog_density;
uniform float u_refract_strength;
uniform float u_r0;
uniform vec3  u_water_color;
uniform float u_light_z;
uniform int   u_srgb_decode;
uniform float u_ripple_scale;
uniform vec2  u_texel;
uniform vec4  u_art_uv_rect;
uniform float u_time;
// Glint strength: the ADDITIVE HDR multiplier on the GGX highlight. The glint
// is light reflecting OFF the surface, so it is added on top of the (see-
// through, premultiplied) base — NOT Fresnel-blended at ~2% nor alpha-
// attenuated. Already gated by × lightRGB (only under a source) and × NdotL
// (no back-lit glint). Default ~2.0 makes glints clearly visible on a flashlit
// rippled puddle. ([graphics.water] glint_strength)
uniform float u_glint_strength;
// Alpha (transparency) ramp dials: alpha = clamp(depth*u_alpha_scale,
//   u_alpha_min, u_alpha_max). u_alpha_scale = depth->opacity rate (old 6.0),
//   u_alpha_min = shoreline floor (old 0.15), u_alpha_max = deep ceiling
//   (old 0.95). The glint does NOT raise alpha — it survives the premultiplied
//   blit as additive light regardless of how transparent the water is.
uniform float u_alpha_scale;
uniform float u_alpha_min;
uniform float u_alpha_max;

out vec4 finalColor;

vec3 srgb_to_linear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 linear_to_srgb(vec3 c) { return pow(c, vec3(1.0 / 2.2)); }

// Ambient sine lattice (ported from overlays.py:183-185 _WAVES) — three
// directions so the idle shimmer never reads as stripes. Returns a height
// offset in the same (m-ish) units as ripple, so it feeds the same gradient.
//   (kx, ky, omega) rad/tile spatial, rad/s temporal.
float ambientSine(vec2 grid_xy) {
    float s = 0.0;
    s += sin( 0.55 * grid_xy.x + 0.25 * grid_xy.y + 1.3 * u_time);
    s += sin(-0.35 * grid_xy.x + 0.45 * grid_xy.y + 0.9 * u_time);
    s += sin( 0.20 * grid_xy.x - 0.60 * grid_xy.y + 1.9 * u_time);
    return s * (1.0 / 3.0);
}

// Sample the ripple height (R) of the water texture at a world UV, plus the
// ambient sine lattice scaled by the local idle amplitude. The sine lattice
// gives standing water idle motion even where ripple == 0 (overlays.py §3).
float surfaceHeight(vec2 uv, float amb_amp) {
    float ripple = texture(u_water, uv).r;
    // grid coords for the sine phases (rad/tile): uv * grid = uv / u_texel.
    vec2 grid_xy = uv / max(u_texel, vec2(1e-6));
    return ripple + amb_amp * ambientSine(grid_xy);
}

void main() {
    // World UV (matches lighting.fs:78). Default art_uv_rect (0,0,1,1) makes
    // this bit-exactly fragTexCoord. Grid-resolution samplers (light, water)
    // read at world_uv; the diffuse reads at fragTexCoord (art space).
    vec2 world_uv = (fragTexCoord - u_art_uv_rect.xy) / u_art_uv_rect.zw;

    // --- depth gate (E) — the DORMANT-SAFE early-out ---------------------
    // water_depth (B) is the only thing that says "there is water here". On a
    // dry tile we emit a fully-transparent premultiplied fragment, so the pass
    // is a no-op over the composite. This is the gate the safety property
    // checks: no standing water -> identical render.
    float depth = texture(u_water, world_uv).b;
    if (depth <= 0.0) {
        finalColor = vec4(0.0);
        return;
    }

    // Local ripple energy -> ambient idle amplitude + roughness agitation.
    // (|ripple| + |ripple_v|, the overlays.py energy heuristic, channels R+G.)
    vec4 wtex = texture(u_water, world_uv);
    float energy = abs(wtex.r) + abs(wtex.g);
    float amb_amp = clamp(0.06 + 2.0 * energy, 0.0, 0.40); // overlays.py _AMB_*

    // --- A. perturbed surface normal -------------------------------------
    // Height-field normal from neighbour taps of the ripple texture (+ the
    // ambient sine lattice). N ≈ normalize(-∂h/∂x, -∂h/∂y, 1) (height-field
    // convention, §3). u_ripple_scale converts metres-of-height to a slope
    // gain that reads on screen (the ripples are millimetric vs a 1-unit UV).
    vec2 tx = vec2(u_texel.x, 0.0);
    vec2 ty = vec2(0.0, u_texel.y);
    float hL = surfaceHeight(world_uv - tx, amb_amp);
    float hR = surfaceHeight(world_uv + tx, amb_amp);
    float hD = surfaceHeight(world_uv - ty, amb_amp);
    float hU = surfaceHeight(world_uv + ty, amb_amp);
    float dhdx = (hR - hL) * 0.5 * u_ripple_scale;
    float dhdy = (hU - hD) * 0.5 * u_ripple_scale;
    vec3 N = normalize(vec3(-dhdx, -dhdy, 1.0));

    // --- D. view direction — top-down constant ---------------------------
    vec3 V = vec3(0.0, 0.0, 1.0);

    // --- B/C. reused light buffer: colour + 3D direction -----------------
    vec4 tex_a = texture(u_light_a, world_uv);
    vec4 tex_b = texture(u_light_b, world_uv);
    vec3 lightRGB = tex_a.rgb;                       // incoming light colour (B)
    vec2 light_dir_2d = vec2(tex_a.a, tex_b.a);      // signed (C)
    vec3 L = normalize(vec3(light_dir_2d, u_light_z));

    // --- REFRACTION branch (head-on; dominates top-down) -----------------
    // Warp the floor sample by the surface tilt, scaled by depth so the
    // shoreline (depth->0) doesn't swim. Re-light the UNLIT diffuse in-shader
    // with the reused light buffer at the refracted UV (§4-F) — NEVER sample
    // the world RT this pass draws into (read-while-write feedback).
    vec2 off = N.xy * u_refract_strength * clamp(depth, 0.0, 1.0);
    vec3 floorDiffuse = texture(u_diffuse, fragTexCoord + off).rgb;
    if (u_srgb_decode == 1) {
        floorDiffuse = srgb_to_linear(floorDiffuse);
    }
    vec3 floorLight = texture(u_light_a, world_uv + off).rgb;
    vec3 floorC = floorDiffuse * floorLight;

    // Beer-Lambert depth tint: the floor fades toward the deep-water colour
    // with depth, reading as a true volume rather than a flat blue wash.
    float fog = exp2(-u_fog_density * depth);
    vec3 refr = mix(u_water_color, floorC, fog);

    // --- REFLECTION branch (the GGX GLINT — light OFF the surface) -------
    // GGX specular, cheap dot(L,H) form (§2). roughness drives α = roughness²;
    // still puddles (low energy) glint sharp, sloshing water (high energy)
    // shimmers broad. The glint is ADDITIVE light reflecting off the surface,
    // so it does NOT go through the Fresnel mix (which top-down on near-flat
    // water sits at ~R0 ≈ 0.02 and crushes the highlight to ~2%); it is added
    // on top of the premultiplied base instead. Gates kept: × lightRGB (only
    // appears under the flashlight/fire) and × NdotL (no back-lit glint).
    float roughness = clamp(u_roughness_base + u_roughness_agitation * energy,
                            0.02, 1.0);
    float a = roughness * roughness;
    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float a2 = a * a;
    float d = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    float ggx = a2 / max(3.14159265 * d * d, 1e-6);
    float NdotL = max(dot(N, L), 0.0);
    vec3 glint = ggx * NdotL * lightRGB * u_glint_strength;

    // --- BASE: the see-through refraction + depth tint -------------------
    // The base is the refraction/Beer-Lambert term — the floor seen through
    // the water. Add a small Fresnel reflection of the ambient light back into
    // the base (grazing ripples pick up a touch of the source colour) — this
    // is the SUBTLE, alpha-bound part; the bright sparkle is the additive
    // glint above, NOT this.
    float NdotV = max(dot(N, V), 0.0);
    float F = u_r0 + (1.0 - u_r0) * pow(1.0 - NdotV, 5.0);
    vec3 base = mix(refr, refr + lightRGB * F, F);  // ≈ refr + small ambient sheen

    // --- output: PREMULTIPLIED base + ADDITIVE glint ---------------------
    // alpha ramps in over the first ~few cm of depth so the wet/dry shoreline
    // reads as a soft edge rather than a hard step; it never reaches 0 here
    // (depth > 0 already passed the gate), and is 0 on dry tiles (returned
    // above). The base is premultiplied by alpha (see-through, transparency-
    // bound); the GLINT is added on top WITHOUT × alpha, so it survives the
    // premultiplied blit as additive light — a bright highlight visible
    // regardless of how transparent the water is. The glint does NOT raise
    // alpha (a specular highlight does not make the water more opaque).
    float alpha = clamp(depth * u_alpha_scale, u_alpha_min, u_alpha_max);
    if (u_srgb_decode == 1) {
        base  = linear_to_srgb(base);
        glint = linear_to_srgb(glint);
    }
    finalColor = vec4(base * alpha + glint, alpha);
}
