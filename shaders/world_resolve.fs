#version 330
// World RESOLVE — the ONE tone-map of the world (graphics arc G0 P1,
// docs/graphics_direction_2026-10-18.md §6). RENDER-ONLY.
//
// Every world layer (lit ship, gas medium, black-body heat, water, 3D units
// and props) writes LINEAR, un-tone-mapped HDR into the RGBA16F scene RT.
// WorldComposite.resolve() draws that RT through this shader into the RGBA8
// display RT: exposure is already in the layers (u_light_gain, the single
// exposure dial), so this pass is exactly
//
//     display = encode( ACES(linear) ) + dither
//
// and nothing else in the world may tone-map, encode or dither. A layer that
// tone-maps itself is the bug this pass exists to retire (tests/
// test_one_tonemap.py).
//
// THE DISTORTION BLOCK (graphics G3 F4a, design §6.2, §10(b)): every
// screen-space distortion of the world is a TERM of distortion_px() below --
// heat shimmer now (renderer/heat_shimmer.py, its spec), the G4 shockwave
// next -- never a second pass over the scene. The block shifts WHERE this
// pass reads the scene RT, so nothing reads the target it writes, the HUD /
// UI (drawn after the resolve) are never distorted, and the tone-map below
// stays the one tone-map. Where the summed offset is exactly zero the read
// is the plain point fetch, bit for bit; a shifted read does its own
// bilinear tap (the scene RT is point-filtered, §13.12).
//
// The RT is PREMULTIPLIED (vacuum tiles are cleared to alpha 0 so the screen
// backdrop shows through the premultiplied blit; a smoke layer over vacuum
// leaves alpha < 1). RGB is tone-mapped as the light it is -- the additive,
// premultiplied radiance of the pixel -- and alpha passes through untouched.
//
// Credit (repo rule):
//  - ACES filmic tone-map, Narkowicz approximation: Krzysztof Narkowicz,
//    "ACES Filmic Tone Mapping Curve" (2016),
//    knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
//    (the same constants every layer used before G0).
//  - Interleaved gradient noise (the dither): Jorge Jimenez, "Next Generation
//    Post Processing in Call of Duty: Advanced Warfare", SIGGRAPH 2014
//    Advances in Real-Time Rendering.
//  - The shimmer's advected noise: Alex Vlachos, "Water Flow in Portal 2",
//    SIGGRAPH 2010 (two-phase flow-map crossfade); Fabrice Neyret, "Advected
//    Textures", SCA 2003.

in vec2 fragTexCoord;
in vec4 fragColor;

uniform sampler2D texture0;   // the scene RT (RGBA16F, premultiplied, linear)
uniform int   u_encode_srgb;  // 1 = gamma-2.2 encode (the G toggle's output half)
uniform float u_dither_amp;   // dither amplitude in 8-bit LSBs (1.0 = +-0.5 LSB)

// --- the distortion block's inputs -------------------------------------------
// heat shimmer (renderer/heat_shimmer.py HeatShimmer.uniforms)
uniform int   u_shimmer_on;       // 0 = the term is off (no fetch at all)
uniform sampler2D u_shimmer_heat; // per tile, bilinear: r = strength 0..1,
                                  // gb = tamed wind, tiles/s
uniform sampler2D u_shimmer_noise;// fBm, REPEAT + bilinear: gb = the vector pair
uniform vec2  u_shimmer_tiles;    // the map in tiles (w, h)
uniform vec4  u_shimmer_a;        // amp px (zoom-faded), grain tiles, px/tile, cycle s
uniform vec4  u_shimmer_ph;       // phase0, phase1, w0 (layer 0's weight), boil tiles/s
uniform vec4  u_shimmer_off;      // per-layer grain offsets (noise uv)
uniform vec4  u_shimmer_boil;     // per-layer boil directions (unit)
uniform vec4  u_shimmer_box;      // the hot tiles' bbox + 1 tile (x0, y0, x1, y1)

out vec4 finalColor;

vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

float interleaved_gradient_noise(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// HEAT SHIMMER: the floor seen through hot gas wobbles. Offset in world px
// (x east, y south) = noise * smoothstep(S) * amp, |offset| <= amp; S = the
// per-tile hot-gas strength upsampled bilinearly, so a pixel whose four
// nearest tile centres are cold gets exactly zero.
vec2 shimmer_term(vec2 wpos) {
    if (u_shimmer_on == 0) return vec2(0.0);
    vec2 tile = wpos / u_shimmer_a.z;
    // outside the hot area: no fetch at all (the cost follows the heat)
    if (any(lessThan(tile, u_shimmer_box.xy))
        || any(greaterThanEqual(tile, u_shimmer_box.zw))) return vec2(0.0);
    vec4 h = texture(u_shimmer_heat, tile / u_shimmer_tiles);
    float s = clamp(h.r, 0.0, 1.0);
    if (s <= 0.0) return vec2(0.0);
    s = s * s * (3.0 - 2.0 * s);
    // two layers advected along the wind + their own boil drift, each
    // restarting (with a fresh grain offset) under zero weight
    float age0 = u_shimmer_ph.x * u_shimmer_a.w;
    float age1 = u_shimmer_ph.y * u_shimmer_a.w;
    vec2 v0 = h.gb + u_shimmer_boil.xy * u_shimmer_ph.w;
    vec2 v1 = h.gb + u_shimmer_boil.zw * u_shimmer_ph.w;
    vec2 n0 = texture(u_shimmer_noise,
                      (tile - v0 * age0) / u_shimmer_a.y + u_shimmer_off.xy).gb;
    vec2 n1 = texture(u_shimmer_noise,
                      (tile - v1 * age1) / u_shimmer_a.y + u_shimmer_off.zw).gb;
    vec2 n = mix(n1, n0, u_shimmer_ph.z) * 2.0 - 1.0;
    vec2 d = n * (s * u_shimmer_a.x);
    float ln = length(d);
    return ln > u_shimmer_a.x ? d * (u_shimmer_a.x / ln) : d;
}

// THE DISTORTION BLOCK: the sum of every screen-space distortion term, in
// world px. A new distortion is one more line here, never a second pass.
vec2 distortion_px(vec2 wpos) {
    vec2 d = vec2(0.0);
    d += shimmer_term(wpos);
    return d;
}

// The scene RT at GL pixel position p (centres at .5), bilinear, by hand
// (the RT itself is point-filtered: §13.12).
vec4 scene_bilinear(vec2 p) {
    ivec2 sz = textureSize(texture0, 0);
    vec2 q = p - 0.5;
    ivec2 i0 = ivec2(floor(q));
    vec2 f = q - vec2(i0);
    ivec2 mx = sz - 1;
    ivec2 a = clamp(i0, ivec2(0), mx);
    ivec2 b = clamp(i0 + ivec2(1), ivec2(0), mx);
    vec4 t00 = texelFetch(texture0, ivec2(a.x, a.y), 0);
    vec4 t10 = texelFetch(texture0, ivec2(b.x, a.y), 0);
    vec4 t01 = texelFetch(texture0, ivec2(a.x, b.y), 0);
    vec4 t11 = texelFetch(texture0, ivec2(b.x, b.y), 0);
    return mix(mix(t00, t10, f.x), mix(t01, t11, f.x), f.y);
}

void main() {
    // fragTexCoord.y runs UP the RT (the resolve's flipped read): world px
    // (y down) = (u W, (1 - v) H)
    vec2 sz = vec2(textureSize(texture0, 0));
    vec2 wpos = vec2(fragTexCoord.x, 1.0 - fragTexCoord.y) * sz;
    vec2 d = distortion_px(wpos);
    vec4 hdr = (d == vec2(0.0))
        ? texture(texture0, fragTexCoord)
        : scene_bilinear(fragTexCoord * sz + vec2(d.x, -d.y));
    // max(): a negative premult value can only be float noise; never let it
    // reach the curve (ACES of a negative is not a colour).
    vec3 c = aces_tonemap(max(hdr.rgb, vec3(0.0)));
    if (u_encode_srgb == 1) {
        c = pow(c, vec3(1.0 / 2.2));      // same gamma-2.2 the layers used
    }
    // Dither in the 8-bit output domain, scaled by coverage so an empty
    // (alpha 0) vacuum pixel stays exactly transparent black. Static per
    // pixel (no time term): a shot renders the same on every run.
    float a = clamp(hdr.a, 0.0, 1.0);
    float n = interleaved_gradient_noise(gl_FragCoord.xy) - 0.5;
    c += vec3(n * u_dither_amp / 255.0) * a;
    finalColor = vec4(clamp(c, 0.0, 1.0), a);
}
