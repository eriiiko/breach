#version 330
// Sub-tile gas DETAIL shader for Breach (Fire & Heat Beauty arc, B2 P3).
// Turns the P2 gas-medium layer (renderer/gas_medium.py — ONE premultiplied
// RGBA where RGB = additive inscatter, A = Beer-Lambert occlusion) into wispy,
// advected smoke WITHOUT touching the sim: the macro shape stays 100% the sim's
// field; this only adds sub-tile noise erosion + bicubic reconstruction on top.
// RENDER-ONLY, determinism-exempt.
//
// It draws the P2 layer through a full-quad pass in the smoke compose slot
// (renderer/gas_detail.py), premultiplied — exactly the P2 plain draw's blend
// (BLEND_ALPHA_PREMULTIPLY, out = src.rgb + dst.rgb*(1-src.a)). With the detail
// dials at their honest defaults it is a gently-eroded, bicubic-smoothed P2; the
// enabled=false path bypasses this shader entirely and draws the plain P2 layer
// (renderer/game_renderer.py), so "detail off" is byte-for-byte the P2 look.
//
// THE RECIPE (design §4), transcribed:
//  - TWO noise layers advected by the sampled wind, phase-offset by tau/2 and
//    crossfaded so each layer's periodic UV reset happens under zero weight
//    (Vlachos ping-pong). A per-pixel phase jitter kills the whole-screen pulse;
//    a half-texture UV offset between layers kills repetition.
//  - The noise ERODES optical depth, STRONGEST WHERE DENSITY IS LOW (Nubis-style
//    remap): wispy ragged edges, solid cores. The erosion scales the WHOLE
//    premultiplied contribution (RGB inscatter AND A occlusion by the same
//    coverage), so a wispy gap loses both its glow and its shadow — never alpha
//    alone.
//  - A few-texel DOMAIN WARP of the sampling UV (from the same noise) hides the
//    tile lattice.
//  - BICUBIC reconstruction (cubic B-spline via 4 bilinear taps, Sigg-Hadwiger)
//    of the low-res layer + density textures kills the bilinear diamond-stars.
//  - DITHER of the thin-gradient range against 8-bit banding (jitter texture is
//    the dither source), gated on u_dither_on, masked to where the layer has
//    content so vacuum tiles stay clean.
//
// CLOCK = SIM TICK (u_phase is frac(tick / tau_ticks), computed CPU-side in
// float64 — never a wall clock), so replays/spectators render identical smoke.
//
// WIND UNITS (critique finding): u_dynamics.rg is the wind ALREADY converted to
// tiles-per-tick on the CPU: dequantize(gmap.wind Q16.16) * advection_rate * dt
// (advection_rate = 900 == [physics].advection_rate, the SAME coefficient +
// convention SmokeDynamics::step / the gas-T solver use). A layer's age is in
// TICKS (phase * tau_ticks), so wind[tiles/tick] * age[ticks] is a tile
// displacement that tracks the plume's own drift BY CONSTRUCTION.
//
// Credit (repo rule — cite what a file implements):
//   - Alex Vlachos, "Water Flow in Portal 2", SIGGRAPH 2010 — two-layer flow-map
//     crossfade.
//   - Fabrice Neyret, "Advected Textures", SCA 2003 — advecting noise by a
//     velocity field with periodic regeneration.
//   - Ken Perlin & Fabrice Neyret, "Flow Noise", SIGGRAPH 2001 — animated noise.
//   - Christian Sigg & Markus Hadwiger, "Fast Third-Order Texture Filtering",
//     GPU Gems 2 ch. 20 — cubic B-spline reconstruction via 4 bilinear taps.
//   Links live in docs/research/smoke_render_litsearch_2026-07-21.md §4.

in vec2 fragTexCoord;
in vec4 fragColor;

// texture0: the P2 premultiplied layer (RGBA8, BILINEAR+CLAMP). raylib binds the
// draw texture here implicitly; we also bind it by name for the bicubic taps.
uniform sampler2D u_layer;
// Per-frame gas DYNAMICS (RGBA16F, grid res, BILINEAR+CLAMP):
//   R,G = wind in TILES/TICK (dequantized * advection_rate * dt)
//   B   = density SOLIDITY in [0,1] (saturate of the pre-curve optical depth) —
//         the erosion-weight field: 1 = solid core (untouched), 0 = wispy edge.
uniform sampler2D u_dynamics;
// Tiling fBm noise (RGBA8, REPEAT+BILINEAR): R = coverage noise, GB = warp vec.
uniform sampler2D u_fbm;
// Static white-noise jitter (RGBA8, REPEAT+POINT): R = phase jitter, G = dither.
uniform sampler2D u_jitter;

uniform vec2  u_grid;         // (grid_w, grid_h) — fragTexCoord * u_grid = tiles
uniform float u_noise_wl;     // noise wavelength in tiles (noise_wavelength_tiles)
uniform float u_adv_gain;     // k_adv (adv_gain): wind -> UV advection rate
uniform float u_phase;        // crossfade phase in [0,1) (frac(tick/tau_ticks))
uniform float u_tau_ticks;    // cycle length in ticks (cycle_seconds * tps)
uniform float u_erode;        // erode_strength: low-density erosion depth [0,1]
uniform float u_warp_tiles;   // domain-warp magnitude in tiles (warp_px / 24)
uniform int   u_dither_on;    // 1 = dither the thin-gradient range
uniform float u_dither_amp;   // dither amplitude (~1.5/255)
uniform float u_jitter_wl;    // phase-jitter sample wavelength in tiles (coarse)
uniform float u_dither_scale; // dither sample scale (fine, per-pixel-ish)

out vec4 finalColor;

const float TWO_PI = 6.28318530718;

// --- cubic B-spline weights (Sigg-Hadwiger, GPU Gems 2 ch. 20) --------------
vec4 cubicWeights(float v) {
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0 / 6.0);
}

// Bicubic reconstruction via 4 BILINEAR taps (tex MUST be bilinear-filtered).
// Smooth cubic B-spline upsampling of the low-res grid texture -> no diamond
// stars. texSize is the texture's resolution in texels (== u_grid here).
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

void main() {
    vec2 uv = fragTexCoord;
    vec2 tile_pos = uv * u_grid;                 // world position in tiles

    // --- BICUBIC layer + density (kills the low-res bilinear diamonds) -------
    vec4 layer = textureBicubic(u_layer, uv, u_grid);       // premult RGBA
    float density = clamp(textureBicubic(u_dynamics, uv, u_grid).b, 0.0, 1.0);
    // Wind stays a plain bilinear read (already smooth; no bicubic overshoot).
    vec2 wind = texture(u_dynamics, uv).rg;                  // tiles/tick

    // --- DOMAIN WARP of the sampling position (hide the tile lattice) --------
    vec2 warpN = texture(u_fbm, tile_pos / (u_noise_wl * 2.0)).gb; // low-freq
    vec2 tp = tile_pos + (warpN - 0.5) * 2.0 * u_warp_tiles;

    // --- per-pixel PHASE JITTER (coarse -> spatially smooth) -----------------
    // Desynchronise the crossfade across the screen so it never pulses in unison
    // (the design's "whole-screen pulse" killer), while staying smooth enough
    // that the advection age varies gently between neighbours (no shimmer).
    float jphase = texture(u_jitter, tile_pos / max(u_jitter_wl, 1e-3)).r;
    float ph = fract(u_phase + jphase);

    // --- TWO advected layers, crossfaded (Vlachos ping-pong) -----------------
    // Age in TICKS = phase * tau_ticks; wind is tiles/tick, so the displacement
    // wind*age tracks the sim's per-tick advection. Layer 1 is phase-offset by
    // tau/2 and half-texture-shifted (kills repetition). w0 hides each reset.
    float age0 = ph * u_tau_ticks;
    float age1 = fract(ph + 0.5) * u_tau_ticks;
    float w0 = 0.5 - 0.5 * cos(TWO_PI * ph);
    vec2 uv0 = (tp - wind * age0 * u_adv_gain) / u_noise_wl;
    vec2 uv1 = (tp - wind * age1 * u_adv_gain) / u_noise_wl + vec2(0.5);
    float n0 = texture(u_fbm, uv0).r;
    float n1 = texture(u_fbm, uv1).r;
    float n = mix(n1, n0, w0);                    // crossfaded coverage noise

    // --- NOISE ERODES optical depth, STRONGEST WHERE DENSITY IS LOW ----------
    // Nubis-style remap: cores (density -> 1) keep coverage 1; wispy edges
    // (density -> 0) get carved down to (1 - erode) where the noise dips. The
    // sim's macro shape is untouched; only the sub-tile fringe becomes ragged.
    float erosion = u_erode * (1.0 - density);
    float coverage = clamp(1.0 - erosion * (1.0 - n), 0.0, 1.0);

    // --- premult-CONSISTENT application: scale the WHOLE premult vec4 ---------
    // Both the additive inscatter (RGB) and the occlusion (A) scale by the same
    // coverage, so an eroded gap loses its glow AND its shadow together.
    vec4 outc = layer * coverage;

    // --- DITHER the thin-gradient range (break 8-bit banding) ----------------
    // Masked by the layer's content (alpha or glow) so fully-empty vacuum tiles
    // stay exactly transparent-black (no tint leak).
    if (u_dither_on == 1) {
        float lum = max(layer.a, max(layer.r, max(layer.g, layer.b)));
        float mask = smoothstep(0.0, 0.02, lum);
        float d = (texture(u_jitter, uv * u_dither_scale).g - 0.5) * u_dither_amp;
        outc += vec4(d) * mask;
    }

    finalColor = clamp(outc, 0.0, 1.0);
}
