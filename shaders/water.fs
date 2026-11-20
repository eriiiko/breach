#version 330
// Water surface optics pass for Breach (graphics/water_rendering.md §2, §7).
// CORE look (steps 1+2): perturbed-ripple normal + Schlick Fresnel + GGX glints
// (reusing the light buffer) + normal-driven refraction × depth + Beer-Lambert
// depth tint. MOOD PASS (step 3, phase 2, ADDED below the core and dormant-
// safe): real-surface caustics × light (§3, SIGN load-bearing), foam port
// (§4-foam, composited last), chromatic aberration. Matcap (§6) + SSR remain
// out.
//
// WATER W1 (#97, docs/water_look_design_2026-11-19.md): THE FINE SURFACE
// THAT FLOWS replaced the ambient sine lattice. Its law and spec is the numpy
// twin renderer/water_surface.py::detail_slope -- a change lands there first,
// then here, and tests/gl_water_flow_check.py holds the two equal. Credits:
//   - Alex Vlachos, "Water Flow in Portal 2", SIGGRAPH 2010 (the two-phase
//     flow map; the crossfade is renderer/advected_noise.py's, spliced in at
//     the ADVECTION marker below);
//   - Morten S. Mikkelsen, "Practical Real-Time Hex-Tiling", JCGT 11(3),
//     2022 (hexSlope: the anti-tiling lookup).
// WATER W2 (#97): FOAM WITH A LIFE. The foam amount (its own image, stepped
// by shaders/water_foam.fs; the twin renderer/water_foam.py) is drawn as a
// THRESHOLD on a bubble raft at two sizes riding the same flow map as the
// fine surface (water_foam.draw_cover: a sheet, then lace, then nothing --
// Yingst, Alford & Parberry, "Very Fast Real-Time Ocean Wave Foam Rendering
// Using Halftoning", GAMEON-NA 2011), OVER the water, lit as the floor is;
// it darkens the floor beneath it and hides the glint. It replaced the old
// gradient foam (foam_threshold / foam_intensity, deleted).
// The light this pass reads gets the smoke's cast shadow and the light curve
// through lit3d's ONE copy, spliced at the SMOKE_SHADOW marker by
// renderer/water.py::load_water_fs (raw, without the splices, this file does
// not compile).
//
// This is a SEPARATE pass from lighting.fs (§5): it draws over the view x
// the wet box, AFTER the lit ship, BEFORE the floor haze and the units.
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
//                B = water_depth (m), A = the foam's relative agitation
//                (W2; water_foam.relative_agitation, read by the foam step).
//   u_foam     - the foam amount image (W2; foam cells, bilinear), unit 8
//   u_bubbles  - the bubble bake (noise_bake.bake_foam_rgba, R uniform on
//                [0, 1]; REPEAT, mipmapped), unit 9
//
// Uniforms ([graphics.water] config block, bound in renderer/water.py):
//   u_roughness_base       GGX roughness floor (still puddles glint sharp)
//   u_roughness_agitation  added roughness per unit local ripple energy
//   u_fog_density          Beer-Lambert extinction (depth tint rate)
//   u_refract_strength     normal-driven floor-warp magnitude (UV units)
//   u_r0                   Fresnel reflectance at normal incidence (~0.02)
//   u_water_color          deep-water tint colour (the volume colour at depth)
//   u_light_z              light-direction z (shared with lighting.fs)
//   u_srgb_decode          1 = decode sRGB diffuse to linear (output stays
//                          linear HDR: the world resolve encodes, G0 P1)
//   u_ripple_scale         metres-of-ripple -> normal-slope gain
//   u_texel                1/grid (x,y): neighbour tap offset for the gradient
//   u_art_uv_rect          art-UV subrect (matches lighting.fs; default 0,0,1,1)
//   u_time                 the sim clock (s), for the caustic detail
//   u_light_gain           the ship's exposure (lighting.fs's u_light_gain)
//   u_grid_m               the grid's size in metres (w, h)
//   u_detail               (texture period m, strength still, strength flow,
//                           zoom fade)
//   u_flow_ph              (crossfade phase, cycle s, drift u, drift v)
//   u_flow_lim             (max slide m, edge depth m, phase noise m,
//                           speed ref m/s)
//   u_agit_ref             |ripple_v| (m/s) at which the strength saturates
//   u_debug_view           0 = the picture, 1 = (slope x, slope y, k, 1),
//                          2 = (foam amount, foam cover, 0, 1)
//   u_foam_on              1 once the foam image exists (W2)
//   u_foam_a               (small bubble period m, large period m,
//                           foam_cover, foam_shadow)
//   u_foam_b               (lace edge, foam albedo, large offset u, v)
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
// W1: the second grid texture (renderer/water_surface.py::pack_flow), RGBA16F
// bilinear: (vx, vy) m/s, solid (1), previous depth m (W2's), on unit 6;
// the ripple bake (noise_bake.bake_ripple_rgba: slope xy, height), RGBA32F,
// REPEAT, mipmapped + trilinear, on unit 7.
uniform sampler2D u_flow;
uniform sampler2D u_ripple;
// W2: the foam amount (unit 8) and the bubble bake (unit 9).
uniform sampler2D u_foam;
uniform sampler2D u_bubbles;
uniform int   u_foam_on;
uniform vec4  u_foam_a;
uniform vec4  u_foam_b;
// Optional per-pixel floor HEIGHTMAP (greyscale relief, 0..1), sampled in art
// space (fragTexCoord, like u_diffuse). ADJUSTS THE PER-PIXEL DEPTH the renderer
// sees: depth = tileDepth - relief*u_height_scale, so raised floor relief makes
// the local water shallower. ALL depth-driven effects (refraction offset,
// Beer-Lambert depth tint, the alpha ramp) compute off this adjusted depth, and
// where it goes <= 0 the floor pokes above the surface (the feature protrudes,
// the water laps around it with a soft u_height_edge shoreline). u_has_height = 0
// (no level heightmap) or u_height_scale == 0 forces relief = 0 so the per-pixel
// depth == the per-tile depth EXACTLY — bit-identical, the dormant default.
uniform sampler2D u_height;

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
// The SIM clock (s), SimClock.t_noise (W1: was wall time) -- only the
// caustic's procedural detail still reads it (W4 replaces that term).
uniform float u_time;
uniform float u_light_gain;
uniform vec2  u_grid_m;
uniform vec4  u_detail;
uniform vec4  u_flow_ph;
uniform vec4  u_flow_lim;
uniform float u_agit_ref;
uniform int   u_debug_view;
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
// --- Phase 2 (mood pass) uniforms ------------------------------------------
// Caustics (§3): focused light on the floor where the surface is concave-up.
//   caustic ~ +laplacian(ripple) (clamp negatives to 0) * lightRGB * strength,
//   ADDED into the refraction/floor term (rides the floor, × alpha). The SIGN
//   is load-bearing: +laplacian focuses (bright in dips), -div(N) ≡ same; a +
//   div(N) would invert it (bright on bumps). u_caustic_scale drives an
//   optional high-frequency procedural detail modulated by surface energy.
uniform float u_caustic_strength;
uniform float u_caustic_scale;
// Chromatic aberration (subtle polish): the floor refraction r/g/b are sampled
// at slightly different offsets (scaled 1 ± u_ca_amount per channel). Tiny.
uniform float u_ca_amount;
// Global ambient light (matches lighting.fs's u_ambient). The dry ship is lit
// by `diffuse * (ambient + sources)`; the water body must be lit the same way,
// else the refracted floor goes black outside any raycast source and shallow
// water reads as a void only visible inside the flashlight beam. Pushed every
// frame from the LightingPass's ambient (renderer/game_renderer.py) so the
// demo's ambient sliders drive the water ambient too.
uniform vec3  u_ambient;
// Heightmap depth-adjust (graphics/water_rendering.md §2 §8). The per-pixel
// relief = texture(u_height, fragTexCoord).r raises the floor, REDUCING the
// per-pixel water depth the whole shader sees:
//   depth = tileDepth - relief * u_height_scale
// so refraction/tint/alpha all vary per-pixel with it, and where depth <= 0 the
// feature protrudes (transparent) with a soft u_height_edge-wide lap shoreline.
//   u_has_height     1 if the level supplied a heightmap, else 0 (-> relief = 0)
//   u_height_scale   relief metres of floor lift subtracted from depth (0
//                    disables -> relief_term = 0); ~0.4 so furniture pokes
//                    through a ~0.3-0.5 m puddle
//   u_height_edge    soft protrusion-shoreline width (m) — the smoothstep ramp
//                    over the first metres of positive depth (~0.1; min-clamped)
//   u_height_floor   the heightmap value treated as floor level ("level 0"
//                    baseline), subtracted from relief BEFORE scaling so a floor
//                    pixel lifts ~0 and water shows on the floor immediately
//                    (no over-fill). Inert on the no-heightmap path.
uniform int   u_has_height;
uniform float u_height_scale;
uniform float u_height_edge;
// Heightmap BASELINE (the "level 0" floor relief). The art heightmap is a
// RELATIVE depth map: the floor itself sits at a nonzero relief value, so the
// raw `relief * u_height_scale` subtracts a constant everywhere and water must
// over-fill before it clears the floor. u_height_floor is the heightmap value
// treated as floor level — subtracted from relief BEFORE scaling, so a floor
// pixel (relief ≈ u_height_floor) reads ~0 lift and water shows on the floor
// immediately; a raised crate (relief > floor) protrudes; a dip (relief < floor)
// reads slightly deeper. ONLY bites on the heightmap path: with u_has_height==0
// or u_height_scale==0 the relief_term is forced to 0 (depth == tileDepth EXACTLY
// — the no-heightmap path stays bit-identical and u_height_floor cannot leak in).
uniform float u_height_floor;

out vec4 finalColor;

vec3 srgb_to_linear(vec3 c) { return pow(c, vec3(2.2)); }

// <SMOKE_SHADOW>
// <ADVECTION>

// --- the hex-tiled ripple lookup (water_surface.py::hex_slope) ---------------
const float HEX_SCALE = 3.46410161514;     // 2 sqrt 3
const float HEX_EXP = 7.0;
const float HEX_FALLOFF_CONTRAST = 0.6;
const float HEX_ROT_STRENGTH = 1.0;

uint hexHash(ivec2 v, uint salt) {
    uint x = (uint(v.x) * 0x8DA6B343u) ^ (uint(v.y) * 0xD8163841u)
           ^ (salt * 0xCB1AB31Fu);
    x ^= x >> 16u; x *= 0x7FEB352Du;
    x ^= x >> 15u; x *= 0x846CA68Bu;
    x ^= x >> 16u;
    return x;
}
float hexHash01(ivec2 v, uint salt) {
    return float(hexHash(v, salt) >> 8u) * (1.0 / 16777216.0);
}

// The slope at st (texture units); dx, dy = st's screen derivatives (the
// taps jump between vertices, so the lookup takes explicit gradients).
vec2 hexSlope(vec2 st, vec2 dx, vec2 dy) {
    vec2 sts = st * HEX_SCALE;
    vec2 sk = vec2(sts.x - 0.57735027 * sts.y, 1.15470054 * sts.y);
    vec2 b = floor(sk);
    vec2 f = sk - b;
    float fz = 1.0 - f.x - f.y;
    float s = (-fz >= 0.0) ? 1.0 : 0.0;
    float s2 = 2.0 * s - 1.0;
    vec3 w = max(vec3(-fz * s2, s - f.y * s2, s - f.x * s2), vec3(0.0));
    ivec2 bi = ivec2(b);
    int si = int(s);
    ivec2 verts[3] = ivec2[3](bi + ivec2(si, si), bi + ivec2(si, 1 - si),
                              bi + ivec2(1 - si, si));
    vec2 acc = vec2(0.0);
    float wsum = 0.0;
    for (int i = 0; i < 3; ++i) {
        ivec2 v = verts[i];
        vec2 cen = vec2(float(v.x) + 0.5 * float(v.y),
                        float(v.y) / 1.15470054) / HEX_SCALE;
        float th = 6.28318530718 * HEX_ROT_STRENGTH
                 * (hexHash01(v, 3u) - 0.5);
        float c = cos(th);
        float sn = sin(th);
        vec2 d = st - cen;
        vec2 q = vec2(c * d.x - sn * d.y, sn * d.x + c * d.y) + cen
               + vec2(hexHash01(v, 1u), hexHash01(v, 2u));
        mat2 R = mat2(c, sn, -sn, c);          // R * d, as above
        vec2 sm = textureGrad(u_ripple, q, R * dx, R * dy).rg;
        vec2 g = vec2(c * sm.x + sn * sm.y, -sn * sm.x + c * sm.y);
        float wi = (1.0 - HEX_FALLOFF_CONTRAST
                    + HEX_FALLOFF_CONTRAST * length(sm))
                 * pow(w[i], HEX_EXP);
        acc += wi * g;
        wsum += wi;
    }
    return acc / wsum;
}

vec2 capSlide(vec2 d, float cap) {
    float ln = length(d);
    return ln > cap ? d * (cap / ln) : d;
}

// THE flow map's two layers at a pixel (water_surface.py::layer_slides): the
// position in metres, the effective flow, the two slides (m) and the
// crossfade weight -- the one flow map the fine surface and the foam ride.
struct FlowLayers { vec2 p_m; vec2 v; vec2 d0; vec2 d1; float w0; };

FlowLayers flowLayers(vec2 world_uv, float tileDepth) {
    FlowLayers L;
    L.p_m = world_uv * u_grid_m;
    L.v = texture(u_flow, world_uv).xy
        * smoothstep(0.0, u_flow_lim.y, tileDepth);
    vec3 lay = advLayers(u_flow_ph.x, advPhaseOffset(L.p_m, u_flow_lim.z));
    L.d0 = capSlide(L.v * (lay.x * u_flow_ph.y), u_flow_lim.x);
    L.d1 = capSlide(L.v * (lay.y * u_flow_ph.y), u_flow_lim.x);
    L.w0 = lay.z;
    return L;
}

// THE fine surface (water_surface.py::detail_slope): its slope (y south) in
// .xy and its strength k in .z.
vec3 detailSlope(FlowLayers L, float rippleV) {
    float P = u_detail.x;
    vec2 q0 = (L.p_m - L.d0) / P - u_flow_ph.zw;
    vec2 q1 = (L.p_m - L.d1) / P - u_flow_ph.zw;
    vec2 dx = dFdx(L.p_m) / P;
    vec2 dy = dFdy(L.p_m) / P;
    vec2 n = mix(hexSlope(q1, dx, dy), hexSlope(q0, dx, dy), L.w0);
    float sat = clamp(max(length(L.v) / u_flow_lim.w,
                          abs(rippleV) / u_agit_ref), 0.0, 1.0);
    float k = (u_detail.y + (u_detail.z - u_detail.y) * sat) * u_detail.w;
    return vec3(n * k, k);
}

// --- the foam's draw (W2; water_foam.py::draw_cover) ------------------------
// The threshold the mean of two uniform bubble reads exceeds with
// probability f, widened by the lace edge at both ends.
float laceThreshold(float f) {
    float q = 1.0 - clamp(f, 0.0, 1.0);
    float x = q < 0.5 ? sqrt(q * 0.5) : 1.0 - sqrt((1.0 - q) * 0.5);
    return x * (1.0 + 2.0 * u_foam_b.x) - u_foam_b.x;
}

float lace(float b, float t) {
    return smoothstep(0.0, 1.0, (b - (t - u_foam_b.x)) / (2.0 * u_foam_b.x));
}

// The mean of the bubble raft at its two sizes at p (metres, slid).
float bubbleAt(vec2 p) {
    return 0.5 * (texture(u_bubbles, p / u_foam_a.x).r
                  + texture(u_bubbles, p / u_foam_a.y + u_foam_b.zw).r);
}

// The drawn foam cover in [0, 1] for the foam amount here.
float foamCover(FlowLayers L, float amount) {
    float f = clamp(amount / u_foam_a.z, 0.0, 1.0);
    float t = laceThreshold(f);
    vec2 drift = u_flow_ph.zw * u_detail.x;       // the drift in metres
    float c0 = lace(bubbleAt(L.p_m - L.d0 - drift), t);
    float c1 = lace(bubbleAt(L.p_m - L.d1 - drift), t);
    return mix(f, mix(c1, c0, L.w0), u_detail.w);
}

// The ripple height (R) of the water texture at a world UV.
float surfaceHeight(vec2 uv) {
    return texture(u_water, uv).r;
}

void main() {
    // World UV (matches lighting.fs:78). Default art_uv_rect (0,0,1,1) makes
    // this bit-exactly fragTexCoord. Grid-resolution samplers (light, water)
    // read at world_uv; the diffuse reads at fragTexCoord (art space).
    vec2 world_uv = (fragTexCoord - u_art_uv_rect.xy) / u_art_uv_rect.zw;

    // --- depth gate (E) — the DORMANT-SAFE early-out ---------------------
    // water_depth (B) is the per-TILE depth — the only thing that says "there
    // is standing water on this tile". On a dry tile we emit a fully-transparent
    // premultiplied fragment, so the pass is a no-op over the composite. This is
    // the gate the safety property checks: no standing water -> identical render.
    // (UNCHANGED from the alpha-only version — the dormant-safe tile gate.)
    float tileDepth = texture(u_water, world_uv).b;
    if (tileDepth <= 0.0) {
        finalColor = vec4(0.0);
        return;
    }

    // --- PER-PIXEL ADJUSTED DEPTH from the floor heightmap relief --------
    // The per-pixel art heightmap RAISES THE FLOOR locally: high relief subtracts
    // from the tile's water depth, so `depth` is the actual water column ABOVE
    // the floor at THIS pixel. Everything depth-driven below computes off this
    // adjusted `depth` (via depthPos, the >= 0 clamp): the refraction offset, the
    // Beer-Lambert tint, and the alpha ramp — a raised crate refracts/tints/laps
    // as the shallower water it sits in, and where the floor pokes ABOVE the
    // surface (depth <= 0) the feature protrudes (handled at output). (The foam
    // wet/dry shoreline still reads the per-TILE depth field gradient — the
    // tile-level water edge — and caustics read ripple curvature; neither is a
    // per-pixel-depth term, so they are intentionally left on the tile field.)
    // The raw relief is measured against u_height_floor (the heightmap value
    // treated as floor level) so the floor itself lifts ~0 and water clears it
    // immediately (no over-fill). When there is no heightmap (u_has_height == 0)
    // or the gain is off (u_height_scale == 0), the relief_term is forced to 0 so
    // `depth == tileDepth` EXACTLY (bit-identical, the no-regression / dormant-
    // safe guarantee the default level relies on; u_height_floor cannot leak in).
    // relief_term is the per-pixel floor LIFT above the baseline floor level:
    // (heightmap relief - u_height_floor). Subtracting u_height_floor before
    // scaling means a floor pixel (relief ≈ u_height_floor) lifts ~0 (water on
    // the floor immediately), a crate (relief > floor) lifts positive (shallower
    // / protrudes), a dip (relief < floor) lifts negative (slightly deeper). With
    // no heightmap or the gain off, relief_term is forced to 0 so depth ==
    // tileDepth EXACTLY (bit-identical; u_height_floor never touches this path).
    float relief_term = (u_has_height == 1 && u_height_scale > 0.0)
                      ? (texture(u_height, fragTexCoord).r - u_height_floor) : 0.0;
    float depth = tileDepth - relief_term * u_height_scale;
    // Effect inputs must never see a negative depth (a protruding floor): clamp
    // to >= 0 for the refraction/fog/alpha math; the protrusion itself is decided
    // by the raw `depth <= 0` test at output.
    float depthPos = max(depth, 0.0);

    // Local ripple energy -> roughness agitation (|ripple| + |ripple_v|,
    // the overlays.py energy heuristic, channels R+G).
    vec4 wtex = texture(u_water, world_uv);
    float energy = abs(wtex.r) + abs(wtex.g);

    // W1: the fine surface that flows (the twin's detail_slope).
    FlowLayers flayers = flowLayers(world_uv, tileDepth);
    vec3 detail = detailSlope(flayers, wtex.g);
    if (u_debug_view == 1) {
        finalColor = vec4(detail.xy, detail.z, 1.0);
        return;
    }
    // W2: the foam here (its amount; its drawn cover).
    float foamAmt = u_foam_on == 1 ? texture(u_foam, world_uv).r : 0.0;
    float foamC = foamAmt > 0.0 ? foamCover(flayers, foamAmt) : 0.0;
    if (u_debug_view == 2) {
        finalColor = vec4(foamAmt, foamC, 0.0, 1.0);
        return;
    }

    // --- A. perturbed surface normal -------------------------------------
    // Height-field normal from neighbour taps of the simulated ripple (the
    // tile-scale waves) plus the fine surface's slope (W1).
    // N ≈ normalize(-∂h/∂x, -∂h/∂y, 1) (height-field convention, §3).
    // u_ripple_scale converts metres-of-height to a slope gain that reads on
    // screen (the ripples are millimetric vs a 1-unit UV).
    vec2 tx = vec2(u_texel.x, 0.0);
    vec2 ty = vec2(0.0, u_texel.y);
    float hL = surfaceHeight(world_uv - tx);
    float hR = surfaceHeight(world_uv + tx);
    float hD = surfaceHeight(world_uv - ty);
    float hU = surfaceHeight(world_uv + ty);
    float dhdx = (hR - hL) * 0.5 * u_ripple_scale + detail.x;
    float dhdy = (hU - hD) * 0.5 * u_ripple_scale + detail.y;
    vec3 N = normalize(vec3(-dhdx, -dhdy, 1.0));

    // --- D. view direction — top-down constant ---------------------------
    vec3 V = vec3(0.0, 0.0, 1.0);

    // --- B/C. reused light buffer: colour + 3D direction -----------------
    vec4 tex_a = texture(u_light_a, world_uv);
    vec4 tex_b = texture(u_light_b, world_uv);
    vec2 light_dir_2d = vec2(tex_a.a, tex_b.a);      // signed (C)
    // The light as the floor takes it (lighting.fs): the exposure, the
    // smoke's cast shadow (G2) and the light curve (G5), through lit3d's
    // one copy. The refracted floor below shares this point's shadow.
    float smoke_sh = smokeShadow(world_uv, light_dir_2d, 0.0);
    vec3 lightRGB = tex_a.rgb * u_light_gain * smoke_sh;  // incoming (B)
    lightRGB *= lightCurveGain(lightRGB);
    // The light comes from u_light_z's elevation along the field's flux
    // direction. The direction is a UNIT vector per tile (light_field's
    // flux_dir), but bilinear filtering shortens it where two tiles' light
    // runs opposite ways (the line between two sources): left as it is, L
    // there turned vertical -- an overhead light no source is -- and the
    // near-mirror still surface (W1) lit a tile-stepped staircase of GGX
    // glints along that line. The filtered direction is renormalised, so no
    // seam ever reads as an overhead light (W1, 2026-11-20).
    float dir_len = length(light_dir_2d);
    vec2 light_dir_unit = dir_len > 1e-4 ? light_dir_2d / dir_len : vec2(0.0);
    vec3 L = normalize(vec3(light_dir_unit, u_light_z));

    // --- REFRACTION branch (head-on; dominates top-down) -----------------
    // Warp the floor sample by the surface tilt, scaled by depth so the
    // shoreline (depth->0) doesn't swim. Re-light the UNLIT diffuse in-shader
    // with the reused light buffer at the refracted UV (§4-F) — NEVER sample
    // the world RT this pass draws into (read-while-write feedback).
    vec2 off = N.xy * u_refract_strength * clamp(depthPos, 0.0, 1.0);
    // --- chromatic aberration (subtle polish, phase 2) ------------------
    // Sample the floor diffuse r/g/b at slightly different refraction offsets
    // (the offset scaled by 1 ± u_ca_amount per channel) — a faint prismatic
    // fringe on the refracted floor. u_ca_amount default is tiny; 0 collapses
    // to the single-sample core (the three offsets coincide).
    vec2 offR = off * (1.0 + u_ca_amount);
    vec2 offB = off * (1.0 - u_ca_amount);
    vec3 floorDiffuse = vec3(
        texture(u_diffuse, fragTexCoord + offR).r,
        texture(u_diffuse, fragTexCoord + off ).g,
        texture(u_diffuse, fragTexCoord + offB).b);
    if (u_srgb_decode == 1) {
        floorDiffuse = srgb_to_linear(floorDiffuse);
    }
    // Light the refracted floor by ambient + source, exactly as lighting.fs
    // lights the dry floor (`diffuse * (ambient + sources)`, the light
    // exposed, shadowed by the smoke and curved). Without the ambient term
    // the floor is black wherever no raycast source reaches, so shallow
    // water read as a black void only visible inside the flashlight.
    vec3 floorL = texture(u_light_a, world_uv + off).rgb * u_light_gain
                * smoke_sh;
    floorL *= lightCurveGain(floorL);
    vec3 floorC = lightCurveAlbedo(floorDiffuse, floorL) * (u_ambient + floorL);
    // W2: the foam's soft shadow on the floor beneath it (its amount where
    // the refracted floor point is; water_foam.shadow_factor).
    float foamShade = 1.0;
    if (u_foam_on == 1) {
        foamShade = 1.0 - u_foam_a.w * clamp(
            texture(u_foam, world_uv + off).r / u_foam_a.z, 0.0, 1.0);
    }
    floorC *= foamShade;

    // Beer-Lambert depth tint: the floor fades toward the deep-water colour
    // with depth, reading as a true volume rather than a flat blue wash. Uses
    // the per-pixel ADJUSTED depth — shallower water over a raised feature tints
    // less (the protrusion reads near the floor colour).
    float fog = exp2(-u_fog_density * depthPos);
    vec3 refr = mix(u_water_color, floorC, fog);

    // --- caustics from the REAL surface (§3; SIGN IS LOAD-BEARING) -------
    // Caustics = where the wavy surface focuses the downward light onto the
    // floor. Intensity ∝ +laplacian(ripple): a CONCAVE-UP (focusing) patch is
    // BRIGHT, a convex bump is dark. With the height-field normal
    // N ≈ (-∂r/∂x, -∂r/∂y, 1), div(N_xy) = -laplacian(ripple), so
    //   caustic ~ +laplacian(ripple) ≡ -div(N_xy)   (CLAMP NEGATIVES TO 0).
    // Brightening by +div(N) would INVERT it (bright on bumps) — do NOT.
    // 5-point Laplacian of the ripple texture R channel (the same neighbour
    // taps the normal uses, but the raw R — no ambient sine, no ripple_scale):
    float rC = texture(u_water, world_uv).r;
    float rL = texture(u_water, world_uv - tx).r;
    float rR = texture(u_water, world_uv + tx).r;
    float rD = texture(u_water, world_uv - ty).r;
    float rU = texture(u_water, world_uv + ty).r;
    float lap = (rL + rR + rD + rU - 4.0 * rC) * u_ripple_scale;
    float caustic = max(lap, 0.0);              // focusing only; bumps -> 0
    // Optional subtle hybrid: a touch of high-frequency procedural detail
    // (modulated by surface energy) for crispness — the Laplacian × light is
    // the core; u_caustic_scale drives the detail (0 = pure surface curvature).
    if (u_caustic_scale > 0.0) {
        vec2 g = world_uv / max(u_texel, vec2(1e-6));
        float hf = 0.5 + 0.5 * sin(u_caustic_scale * (g.x + g.y) + 2.7 * u_time)
                              * sin(u_caustic_scale * (g.x - g.y) - 1.9 * u_time);
        caustic *= mix(1.0, hf, clamp(2.0 * energy, 0.0, 1.0));
    }
    // × lightRGB so caustics only appear under a flashlight/fire (correct —
    // focused light needs light); × strength; ride the floor/base (× alpha
    // below, NOT the surface-additive glint).
    refr += caustic * u_caustic_strength * lightRGB * foamShade;

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
    // the water. Add a small Fresnel reflection back into the base (grazing
    // ripples pick up a touch of reflected light) — this is the SUBTLE, alpha-
    // bound part; the bright sparkle is the additive glint above, NOT this.
    // The reflected colour is ambient + source so the surface has a faint
    // sheen (presence) OUTSIDE any source, not only under the flashlight. This
    // ambient sheen is the placeholder for the deferred matcap/environment
    // reflection — keep it subtle. (The glint stays source-only; ambient gives
    // a flat sheen, not a sharp specular.)
    float NdotV = max(dot(N, V), 0.0);
    float F = u_r0 + (1.0 - u_r0) * pow(1.0 - NdotV, 5.0);
    vec3 sheenRGB = u_ambient + lightRGB;
    vec3 base = mix(refr, refr + sheenRGB * F, F);  // ≈ refr + small ambient sheen

    // --- output: PREMULTIPLIED base + ADDITIVE glint ---------------------
    // alpha ramps in over the first ~few cm of depth so the wet/dry shoreline
    // reads as a soft edge rather than a hard step; it never reaches 0 here
    // (depth > 0 already passed the gate), and is 0 on dry tiles (returned
    // above). The base is premultiplied by alpha (see-through, transparency-
    // bound); the GLINT is added on top WITHOUT × alpha, so it survives the
    // premultiplied blit as additive light — a bright highlight visible
    // regardless of how transparent the water is. The glint does NOT raise
    // alpha (a specular highlight does not make the water more opaque).
    float alpha = clamp(depthPos * u_alpha_scale, u_alpha_min, u_alpha_max);
    // --- PROTRUSION + SOFT LAP EDGE (per-pixel depth) -------------------
    // The heightmap adjusted the floor per-pixel above (subtracting relief from
    // the tile depth). Where the adjusted `depth` has gone <= 0 the floor pokes
    // ABOVE the surface — the feature protrudes, so emit a fully-transparent
    // fragment (the crate/console/debris shows through; the water laps around
    // it). For a SOFT shoreline (not a hard cut) we fade the alpha to 0 over the
    // first u_height_edge metres of positive depth: smoothstep is ~0 right at
    // the protrusion edge and ramps to 1 over height_edge, OVERRIDING the
    // alpha_min floor so the waterline against the feature reads as a soft lap
    // rather than a step. With NO heightmap relief == 0 -> depth == tileDepth > 0
    // (passed the gate), and smoothstep(0, edge, depth) == 1 for any depth beyond
    // a sub-cm edge, so the alpha is bit-identical to the pre-heightmap render
    // (and to a no-water ship). This SUBSUMES the b5e7bdb alpha-only block:
    // where depth <= 0 the feature is dry; the soft edge replaces the old wet=...
    // ramp but now drives every effect through the adjusted depth, not just alpha.
    float lapEdge = smoothstep(0.0, max(u_height_edge, 1e-4), depth);
    alpha *= lapEdge;
    if (depth <= 0.0) {
        finalColor = vec4(0.0);
        return;
    }
    // --- W2: the FOAM, over the water (opaque where it covers) -----------
    // Froth is diffuse surface scatter lit as the floor is (ambient + the
    // curved source light, the light curve's albedo); where it covers, it
    // hides the floor, the water's tint and the glint (premultiplied over).
    float cf = foamC * lapEdge;
    vec3 foamRGB = lightCurveAlbedo(vec3(u_foam_b.y), lightRGB)
                 * (u_ambient + lightRGB);
    // Linear HDR out (G0 P1): the world resolve tone-maps and encodes, once,
    // for the water and the floor together -- before G0 the water encoded
    // here but skipped the floor's ACES, so the two disagreed on exposure.
    finalColor = vec4((base * alpha + glint) * (1.0 - cf) + foamRGB * cf,
                      alpha * (1.0 - cf) + cf);
}
