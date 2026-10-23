#version 330
// fire_glow.fs -- THE FIRE'S OWN LIGHT in the world picture (graphics arc G3
// "Fire", patch F3; docs/g3_fire_design_2026-10-23.md §3.5, §3.6, §13.4,
// §13.7, §13.8). RENDER-ONLY. Composed in Python (renderer/fire_vfx.py
// compose_glow_source): the smoke cast product's shared GLSL
// (renderer/lit3d.py _SMOKE_SHADOW_GLSL, smokeCoverAt) is spliced in at the
// marker line below.
//
// One shader, three STAGES (u_stage), drawn by GameRenderer.compose_world.
// Every stage keeps the destination alpha (the blend's alpha factors are
// ZERO, ONE).
//
//   1 FLAME  -- SCENE (linear HDR, before the one tone-map), under the
//               units: the flame VFX as an EMITTER COVER with opacity tied
//               to its emission (§13.4), premultiplied over:
//               rgb = Le(T_vfx) E, a = eps = E w(T_vfx), w = the ramp's
//               intensity capped at 1 (BlackbodyRamp.cover_weight).
//   2 COMPENSATION -- SCENE, right after the ceiling soot: the flame again,
//               ADDED with weight alpha_ceil (1 - glow_through_ceiling),
//               alpha_ceil = the ceiling layer's own alpha from the cast
//               product -- so the soot hides the flame below it by
//               glow_through_ceiling only (Erik's 0.2), never by its full
//               alpha (§13.7).
//   3 HEAT   -- DISPLAY space, after the resolve: the black-body heat
//               layer, G0's named exception (§13.8 FALLBACK: the hue gate
//               failed on its linear move, tools/fire_look_gate.py), its
//               ACES-shaped display colour ADDED, weighted by G2's
//               glow-through (only the share below the smoke is dimmed:
//               t = (1 - gt.x cov.r)(1 - gt.y cov.b), mixed by the below
//               fraction) and by (1 - eps) of the flame over it -- the flame
//               COVERS its burning tile, nothing is counted twice (M4).
//               Colour passes through; only the weight is scaled.
//
// THE FLAME (§3.5): the flame field (R, R*T at m cells per tile) is
// reconstructed with a cubic B-spline (no texel blocks at the world RT's
// 48 px per tile), its edge torn by the FLAME NOISE -- a grain
// (renderer/noise_bake.py flame_noise_field) streaked ALONG the flame
// velocity by a short line integral on a two-phase flow map on the VFX
// clock -- then L = Le(T_vfx) E(R) with Le from the blackbody GPU LUT (THE
// ramp, never a second T -> colour map) and E the A/B law: A "mask"
// smoothstep(r0, r1, R), B "thickness" 1 - exp(-kappa R), clipped by the
// emission threshold e_min (fire_vfx_ref.emission_clip).
//
// Credit (repo rule): cubic B-spline via 4 bilinear taps -- Sigg & Hadwiger,
// "Fast Third-Order Texture Filtering", GPU Gems 2 ch. 20 (2005); line
// integral convolution -- Cabral & Leedom, "Imaging Vector Fields Using Line
// Integral Convolution", SIGGRAPH 1993; two-phase flow map -- Vlachos,
// "Water Flow in Portal 2", SIGGRAPH 2010 Advances in Real-Time Rendering.

in vec2 fragTexCoord;      // world uv: (0, 0) = the map's north-west corner
in vec4 fragColor;

uniform sampler2D texture0;     // the quad's carrier (unused)
uniform int   u_stage;          // 1 flame, 2 compensation, 3 heat (display)
uniform vec2  u_grid;           // the map, tiles (W, H)

// the heat layer (per tile, display; renderer/overlays.py HeatFieldOverlay)
uniform sampler2D u_heat;       // rgb = display hue (peak 1), a = weight
uniform sampler2D u_below;      // r = the share of the glow BELOW the smoke
uniform vec2  u_glow_through;   // (ceiling, floor), 0 when no cast product

// flame (renderer/fire_vfx.py fields; cells, north row first)
uniform sampler2D u_flame;      // x = R, y = R * T_vfx   (linear filter)
uniform sampler2D u_vel;        // xy = u_vfx, tiles/s     (linear filter)
uniform sampler2D u_noise;      // the flame noise grain (REPEAT)
uniform sampler2D u_lut;        // blackbody GPU LUT: rgb chroma, a intensity
uniform float u_m;              // flame cells per tile
uniform int   u_flame_on;
uniform vec2  u_lut_k;          // the LUT's Kelvin domain (floor, ceil)
uniform float u_lut_n;          // its size
uniform int   u_emission;       // 0 = A "mask", 1 = B "thickness"
uniform vec4  u_law;            // (r0, r1, kappa, e_min)
uniform vec4  u_noise_a;        // (amp, grain tiles, streak s, speed cap tiles/s)
uniform vec4  u_noise_ph;       // (phase 0, phase 1, cycle s, contrast)
uniform vec4  u_noise_off;      // per-phase grain offsets (xy, zw), tiles
uniform vec4  u_fboxes[4];      // the flame's active boxes, tiles (x0, y0, x1, y1)
uniform vec2  u_box_fade;       // at a box's edge, tiles: (e-fold of the
                                // brightness ceiling, ragged window width)

out vec4 finalColor;

// <SMOKE_SHADOW>

// --- cubic B-spline reconstruction (Sigg-Hadwiger; as shaders/gas_relief.fs)
vec4 cubicWeights(float v) {
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0 / 6.0);
}

vec4 textureBicubic(sampler2D tex, vec2 texCoords, vec2 texSize) {
    vec2 invTexSize = 1.0 / texSize;
    texCoords = texCoords * texSize - 0.5;
    vec2 fxy = fract(texCoords);
    texCoords -= fxy;
    vec4 xcubic = cubicWeights(fxy.x);
    vec4 ycubic = cubicWeights(fxy.y);
    vec4 c = texCoords.xxyy + vec2(-0.5, 1.5).xyxy;
    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4(xcubic.yw, ycubic.yw) / s;
    offset *= invTexSize.xxyy;
    vec4 sample0 = texture(tex, offset.xz);
    vec4 sample1 = texture(tex, offset.yz);
    vec4 sample2 = texture(tex, offset.xw);
    vec4 sample3 = texture(tex, offset.yw);
    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);
    return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}

// --- the one ramp -----------------------------------------------------------
vec4 lut(float kelvin) {                    // rgb chroma, a intensity
    float x = clamp((kelvin - u_lut_k.x) / (u_lut_k.y - u_lut_k.x), 0.0, 1.0);
    return texture(u_lut, vec2((x * (u_lut_n - 1.0) + 0.5) / u_lut_n, 0.5));
}

float emission_clip(float E) {              // fire_vfx_ref.emission_clip
    float e = u_law.w;
    return e > 0.0 ? smoothstep(e, 2.0 * e, E) : 1.0;
}

// --- the flame noise: the grain streaked along the flow ---------------------
float flame_noise(vec2 p, vec2 u) {
    float sp = length(u);
    // the flow map's drift: the flame velocity, capped (a 15 tiles/s jet
    // would smear the grain into a blur within one phase)
    vec2 v = u * min(1.0, u_noise_a.w / max(sp, 1e-4));
    // the streak: a line integral along the flow, as long as the flow
    // carries the grain in u_noise_a.z seconds; a still flame (the core)
    // keeps the round grain
    vec2 st = v * u_noise_a.z;
    float n = 0.0;
    for (int k = 0; k < 2; ++k) {
        float ph = (k == 0) ? u_noise_ph.x : u_noise_ph.y;
        float w = 1.0 - abs(2.0 * ph - 1.0);           // triangle: sums to 1
        vec2 off = (k == 0) ? u_noise_off.xy : u_noise_off.zw;
        vec2 base = p - v * (ph * u_noise_ph.z) + off;
        float acc = 0.0;
        for (int j = 0; j < 4; ++j) {
            vec2 q = (base - st * (float(j) / 3.0 - 0.5)) / u_noise_a.y;
            vec4 g = texture(u_noise, q);
            acc += (k == 0) ? g.r : g.g;
        }
        n += w * acc * 0.25;
    }
    // the crossfade and the integral both pull toward the mean: restore
    return clamp(0.5 + (n - 0.55) * u_noise_ph.w, 0.0, 1.0);
}

// The distance (tiles) to the edge of the active box holding p, 0 outside
// every box. The box is the flame's fluid domain (an open boundary): flame
// still standing there must not end in a straight cut (F3).
float box_dist(vec2 p) {
    for (int i = 0; i < 4; ++i) {
        vec4 b = u_fboxes[i];
        if (p.x >= b.x && p.y >= b.y && p.x < b.z && p.y < b.w)
            return min(min(p.x - b.x, b.z - p.x), min(p.y - b.y, b.w - p.y));
    }
    return 0.0;
}

// The flame's fade toward its box edge, as a factor on E: a RAGGED window
// (the flame noise moves its start by +-half its width, so faint tongues
// end at different depths, not on one line) times a BRIGHTNESS CEILING
// exp(d / e-fold) - 1 on the intensity L = I E, 0 at the edge and rising
// exponentially -- a white-hot flame (intensity ~60) fades over its own
// tone-mapped range instead of saturating to a straight cut.
float edge_fade(float d, float n, float L) {
    float win = smoothstep(0.0, 1.0, (d + (n - 0.5) * u_box_fade.y)
                                     / u_box_fade.y);
    float cap = exp(min(d / u_box_fade.x, 30.0)) - 1.0;
    return win * min(1.0, cap / max(L, 1e-6));
}

// (premultiplied flame radiance Le E, its cover eps = E w(T)) at world uv
vec4 flame_at(vec2 uv) {
    if (u_flame_on == 0) return vec4(0.0);
    vec4 f = textureBicubic(u_flame, uv, u_grid * u_m);
    float R = max(f.x, 0.0);
    if (R <= 1e-4) return vec4(0.0);
    float T = max(f.y, 0.0) / R;
    vec2 u = texture(u_vel, uv).xy;
    float n = flame_noise(uv * u_grid, u);
    // erode the EDGE (low R), leave the core: R' = R (1 + amp (2n - 1) edge)
    float edge = 1.0 - smoothstep(0.0, u_law.y, R);
    float Re = R * max(0.0, 1.0 + u_noise_a.x * (2.0 * n - 1.0) * edge);
    float E = (u_emission == 0) ? smoothstep(u_law.x, u_law.y, Re)
                                : 1.0 - exp(-u_law.z * Re);
    E *= emission_clip(E);
    // Le = chroma * intensity; the cover weight is the intensity capped at
    // 1 (blackbody.BlackbodyRamp.cover_weight): tied to the emission
    vec4 c = lut(T);
    E *= edge_fade(box_dist(uv * u_grid), n, c.a * E);
    return vec4(c.rgb * c.a * E, E * min(c.a, 1.0));
}

void main() {
    vec2 uv = fragTexCoord;
    if (u_stage == 1) {
        finalColor = flame_at(uv);
    } else if (u_stage == 2) {
        vec4 cov = smokeCoverAt(uv);
        float wgt = clamp(cov.r, 0.0, 1.0) * (1.0 - u_glow_through.x);
        finalColor = wgt > 0.0 ? vec4(flame_at(uv).rgb * wgt, 0.0)
                               : vec4(0.0);
    } else {
        vec4 c = texture(u_heat, uv);
        vec4 cov = smokeCoverAt(uv);
        // Only the glow BELOW the smoke is hidden: a body's (a burning
        // crate, a hot wall) beyond what the hot gas around it emits by
        // itself. A hot GAS tile's glow is the gas itself -- an emitting
        // slab is not hidden by its own depth (#95's question).
        float below = clamp(texture(u_below, uv).r, 0.0, 1.0);
        float t = (1.0 - u_glow_through.x * clamp(cov.r, 0.0, 1.0))
                * (1.0 - u_glow_through.y * clamp(cov.b, 0.0, 1.0));
        float wgt = c.a * mix(1.0, t, below) * (1.0 - flame_at(uv).a);
        finalColor = vec4(c.rgb * wgt, 0.0);
    }
}
