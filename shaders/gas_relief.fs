#version 330
// Smoke RELIEF shader for Breach (graphics arc G1,
// docs/graphics_direction_2026-10-18.md §2.1, §2.2, §2.4). RENDER-ONLY.
//
// Smoke seen from above is a relief: a height field with a lit surface. Per
// pixel of the world RT this builds the surface of ONE species group (the
// ceiling soot, or the floor haze) from the sim's tile fields and lights it
// with the sweep's light (renderer/gas_relief.py owns the pass):
//
//   coverage  C = smoothstep over ln(extinction depth), pushed by the billow
//               field B -> a narrow, billowed edge
//   height    h = H sqrt(sat(tau / tau_h)) (1 - a + a B) C, tiles: thick
//               columns stand tall, the billows bulge their tops
//
// OUTPUT: premultiplied LINEAR HDR (inscatter, 1 - T) into the RGBA16F scene
// RT, blended out = src.rgb + dst * (1 - src.a). Never a tone-map here: the
// ONE tone-map is shaders/world_resolve.fs (tests/test_one_tonemap.py).
//
// CLOCK = SIM TICK: u_phase / u_phase_hot are frac(tick / tau_ticks) from
// renderer/advected_noise.py (float64 on the CPU), never wall time.
//
// Credit (repo rule -- cite what a file implements):
//   - Andrew Schneider & Nathan Vos, "The Real-Time Volumetric Cloudscapes
//     of Horizon Zero Dawn", SIGGRAPH 2015 Advances in Real-Time Rendering
//     -- coverage remap of a billow (Perlin-Worley) field.
//   - Alex Vlachos, "Water Flow in Portal 2", SIGGRAPH 2010 -- the two-layer
//     flow-map crossfade; Fabrice Neyret, "Advected Textures", SCA 2003.
//   - Christian Sigg & Markus Hadwiger, "Fast Third-Order Texture
//     Filtering", GPU Gems 2 ch. 20 -- cubic B-spline via 4 bilinear taps.

in vec2 fragTexCoord;
in vec4 fragColor;

uniform sampler2D texture0;   // the group's MEDIUM: rgb = glow share, a = tau' (occlusion)
uniform sampler2D u_cover;    // rgb = surface albedo, a = extinction depth
uniform sampler2D u_dyn;      // rg = tamed wind (tiles/tick), b = boil weight
uniform sampler2D u_light_a;  // rgb = incoming light, a = light_dir.x (toward the light)
uniform sampler2D u_light_b;  // rgb = glow (unused here), a = light_dir.y
uniform sampler2D u_billow;   // R = billow B, GB = warp, A = fine domes (REPEAT)

uniform vec2  u_grid;         // (grid_w, grid_h): uv * u_grid = tiles
uniform float u_phase;        // base clock phase [0,1)
uniform float u_tau_ticks;    // base clock cycle, ticks
uniform float u_phase_hot;    // hot (boiling) clock phase [0,1)
uniform float u_tau_ticks_hot;
uniform float u_adv_gain;     // wind -> noise advection gain
uniform float u_drift_cap;    // most a layer may drift in one cycle, tiles
uniform float u_billow_tiles; // billow texture period, tiles
uniform float u_billow_mean;  // mean of B over the bake (centres the push)
uniform float u_warp_tiles;   // domain warp, tiles
uniform float u_jitter_wl;    // phase-jitter wavelength, tiles (SMOOTH: a
                              // per-pixel jitter turns the crossfade into grain)
uniform float u_cov_tau;      // extinction depth at the coverage edge
uniform float u_cov_edge;     // edge width where the field has a gradient, tiles
uniform float u_cov_soft;     // least edge width, e-folds (uniform haze stays soft)
uniform float u_cov_push;     // billow displacement of the edge, tiles
uniform float u_haze;         // share of the depth kept OUTSIDE the coverage
uniform float u_height;       // tallest column, tiles
uniform float u_height_tau;   // depth of a full-height column
uniform float u_billow_amp;   // share of the height the billows carve
uniform float u_lamp_z;       // lamp elevation (z of the unnormalised L)
uniform float u_wrap;         // wrap lighting floor
uniform float u_light_gain;   // the one exposure ([render.lighting] master_gain)
uniform vec3  u_ambient;      // the flat floor term (LightingPass.ambient)
uniform float u_shadow;
uniform float u_shadow_len;
uniform float u_shadow_soft;
uniform float u_powder;
uniform float u_ao;
uniform float u_rim;
uniform float u_rim_g;
uniform int   u_debug;

out vec4 finalColor;

const float TWO_PI = 6.28318530718;

// --- cubic B-spline reconstruction (Sigg-Hadwiger) ------------------------
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

// --- the billow field, advected on the two clocks ----------------------------
// Set once per pixel in main(): the layers' advection offsets (in billow
// texture units) and crossfade weights, the warp and the boil weight. The
// neighbour taps of the height field reuse them (they lie within a tile).
vec2 g_o0, g_o1, g_h0, g_h1, g_warp;
float g_w0, g_wh0, g_boil;

float billowRaw(vec2 p) {                     // p in tiles -> B in [0,1]
    vec2 q = (p + g_warp) / u_billow_tiles;
    float b = mix(texture(u_billow, q + g_o1 + vec2(0.5)).r,
                  texture(u_billow, q + g_o0).r, g_w0);
    if (g_boil > 0.004) {
        float bh = mix(texture(u_billow, q + g_h1 + vec2(0.25, 0.75)).r,
                       texture(u_billow, q + g_h0 + vec2(0.75, 0.25)).r, g_wh0);
        b = mix(b, bh, g_boil);
    }
    return b;
}

// Coverage as a DISPLACED LEVEL SET of ln(extinction depth): the edge is
// the contour ln(tau / u_cov_tau) = 0, moved along its normal by the billow
// field u_cov_push tiles at most -- q gains |grad ln tau| * displacement.
// Where the field has a gradient (a plume's edge) the transition is
// u_cov_edge tiles wide: sharp, billowed. Where it is flat (a room of thin
// uniform haze at the threshold) the gradient -> 0, so the billows displace
// nothing and the width falls back to u_cov_soft e-folds: an even haze, never
// a leopard of flecks (G1's first plume shots cut exactly that).
const float GRAD_MAX = 8.0;          // |grad ln tau| clamp, e-folds per tile
float lnTauAt(vec2 p) {
    return log(max(textureBicubic(u_cover, p / u_grid, u_grid).a, 1e-6));
}
float coverageOf(float ln_tau, float grad, float b) {
    float q = ln_tau - log(u_cov_tau) + u_cov_push * grad * (b - u_billow_mean);
    float w = max(u_cov_edge * grad, u_cov_soft);
    return smoothstep(-0.5 * w, 0.5 * w, q);
}

// The RELIEF HEIGHT at p (tiles), given the local |grad ln tau| (taken at
// the pixel: the field is tile-scale, the taps sub-tile).
float heightAt(vec2 p, float grad) {
    float t = max(textureBicubic(u_cover, p / u_grid, u_grid).a, 0.0);
    float b = billowRaw(p);
    float c = coverageOf(log(max(t, 1e-6)), grad, b);
    return u_height * sqrt(clamp(t / u_height_tau, 0.0, 1.0))
         * (1.0 - u_billow_amp + u_billow_amp * b) * c;
}

void main() {
    vec2 uv = fragTexCoord;
    vec2 tp = uv * u_grid;

    vec4 cov = textureBicubic(u_cover, uv, u_grid);
    float tau_ext = max(cov.a, 0.0);
    // Early out: no smoke at all (the empty ship costs one bicubic tap).
    if (tau_ext <= 1e-5) discard;
    float ln_tau = log(max(tau_ext, 1e-6));

    // --- per-pixel clock state (Vlachos two-layer crossfade, two clocks) ---
    vec4 dyn = texture(u_dyn, uv);
    vec2 wind = dyn.rg;                                   // tiles / tick
    g_boil = clamp(dyn.b, 0.0, 1.0);
    // Phase jitter: a smooth periodic fBm (the bake's B channel) at a coarse
    // scale, stretched over two cycles -- neighbouring pixels share nearly
    // the same phase, distant ones do not (no whole-screen pulse, no grain).
    float jph = 2.0 * texture(u_billow, tp / max(u_jitter_wl, 1e-3)
                                        + vec2(0.37, 0.71)).b;
    // The DRIFT CAP: a layer's whole-cycle displacement |wind| * tau is
    // limited to u_drift_cap tiles. Uncapped, a blast's wind (up to ~1.4
    // tiles/tick on the tamed scale) carries a layer ~80 tiles in one 60-tick
    // cycle, and any spatial change in wind or phase shears the noise into
    // streaks and rings (seen in G1's first plume shots). Capped, slow air
    // drifts at its own speed and fast air drifts at the cap; the sim's
    // plume still moves at the real speed underneath.
    vec2 v = wind * u_adv_gain;
    vec2 v_base = v * min(1.0, u_drift_cap / max(length(v) * u_tau_ticks, 1e-6));
    vec2 v_hot = v * min(1.0, u_drift_cap / max(length(v) * u_tau_ticks_hot, 1e-6));
    float ph = fract(u_phase + jph);
    g_w0 = 0.5 - 0.5 * cos(TWO_PI * ph);
    g_o0 = -v_base * (ph * u_tau_ticks) / u_billow_tiles;
    g_o1 = -v_base * (fract(ph + 0.5) * u_tau_ticks) / u_billow_tiles;
    float phh = fract(u_phase_hot + jph);
    g_wh0 = 0.5 - 0.5 * cos(TWO_PI * phh);
    g_h0 = -v_hot * (phh * u_tau_ticks_hot) / u_billow_tiles;
    g_h1 = -v_hot * (fract(phh + 0.5) * u_tau_ticks_hot) / u_billow_tiles;
    g_warp = (texture(u_billow, tp / (u_billow_tiles * 2.0)).gb - 0.5)
           * 2.0 * u_warp_tiles;

    // --- coverage: the sharp, billowed edge --------------------------------
    const float GE = 0.25;           // gradient step, tiles (tile-scale field)
    vec2 g2 = vec2(lnTauAt(tp + vec2(GE, 0.0)) - lnTauAt(tp - vec2(GE, 0.0)),
                   lnTauAt(tp + vec2(0.0, GE)) - lnTauAt(tp - vec2(0.0, GE)))
            / (2.0 * GE);
    float grad = min(length(g2), GRAD_MAX);
    float B = billowRaw(tp);
    float C = coverageOf(ln_tau, grad, B);

    vec4 med = textureBicubic(texture0, uv, u_grid);
    float tau_abs = max(med.a, 0.0);
    vec3 glow = max(med.rgb, vec3(0.0));

    // The cut keeps a HAZE share of the depth outside the edge: thin smoke
    // around a plume (and a room of it) still dims the floor, and the edge
    // reads as the step from haze to body.
    float K = u_haze + (1.0 - u_haze) * C;
    float alpha = 1.0 - exp(-tau_abs * K);
    vec3 rgb = glow * K;

    if (u_debug == 1) { finalColor = vec4(vec3(C), 1.0); return; }

    // --- the relief height ---------------------------------------------------
    float h = u_height * sqrt(clamp(tau_ext / u_height_tau, 0.0, 1.0))
            * (1.0 - u_billow_amp + u_billow_amp * B) * C;
    if (u_debug == 2) {
        finalColor = vec4(vec3(h / max(u_height, 1e-6)), 1.0); return;
    }

    finalColor = vec4(max(rgb, vec3(0.0)), clamp(alpha, 0.0, 1.0));
}
