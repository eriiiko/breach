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

in vec2 fragTexCoord;
in vec4 fragColor;

uniform sampler2D texture0;   // the scene RT (RGBA16F, premultiplied, linear)
uniform int   u_encode_srgb;  // 1 = gamma-2.2 encode (the G toggle's output half)
uniform float u_dither_amp;   // dither amplitude in 8-bit LSBs (1.0 = +-0.5 LSB)

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

void main() {
    vec4 hdr = texture(texture0, fragTexCoord);
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
