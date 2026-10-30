#version 330
// particle_sprite.fs -- THE particle primitive's sprite (graphics arc G3
// F4a, renderer/particles.py ParticleDraw). RENDER-ONLY, LINEAR HDR into the
// scene RT, ADDITIVE (blend ONE, ONE on RGB; the destination alpha is kept,
// ZERO, ONE) -- the one tone-map (world_resolve.fs) does the rest.
//
// An ember's colour is ONLY the blackbody GPU LUT (FireVFXPass.lut_tex: rgb
// = the ramp's chroma, a = its HDR intensity, on the ramp's own Kelvin
// domain) -- never a second T -> colour map. Its spec is
// particles.ember_radiance: Le(T) * weight at the core.
//
// Shape: a soft capsule around the trail's core segment. Motion blur
// conserves the spark's light: the trail's radiance is scaled by r / (r +
// seg) so a fast spark is a longer, dimmer streak, not a brighter one; the
// head is brighter than the tail (the trail is where it WAS).
//
// Two stages, the flame's (shaders/fire_glow.fs, design §13.7):
//   u_stage 0  in the scene, after the flame, UNDER the units and the
//              ceiling soot;
//   u_stage 1  the COMPENSATION after the ceiling soot: the soot just hid the
//              sparks by its full alpha; add back alpha_ceil (1 -
//              glow_through_ceiling) of them, so their net occlusion is Erik's
//              glow_through_ceiling (0.2), exactly the flame's -- a fire's
//              sparks never vanish into its own smoke. The cover is read
//              through the lit3d block (smokeCoverAt), spliced in below.

in vec2 v_local;
in float v_half_seg;
in float v_radius;
in float v_temp;
in float v_weight;
in vec2 v_world_uv;

uniform sampler2D u_lut;      // blackbody GPU LUT (1024 x 1, bilinear)
uniform vec2 u_lut_k;         // the LUT's Kelvin domain (floor, ceil)
uniform float u_lut_n;        // its entry count
uniform int   u_stage;        // 0 = under the soot, 1 = compensation
uniform float u_glow_through_ceil;  // glow_through_ceiling (stage 1 only)

// <SMOKE_SHADOW>

out vec4 finalColor;

vec4 lut_at(float k) {
    float x = clamp((k - u_lut_k.x) / (u_lut_k.y - u_lut_k.x), 0.0, 1.0);
    return texture(u_lut, vec2((x * (u_lut_n - 1.0) + 0.5) / u_lut_n, 0.5));
}

void main() {
    float r = v_radius;
    float seg = v_half_seg;
    // distance to the core segment [-seg, +seg] along x, in radii
    float dx = max(abs(v_local.x) - seg, 0.0);
    float d = length(vec2(dx, v_local.y)) / r;
    if (d >= 1.0) discard;
    float core = 1.0 - d;
    core *= core;                              // a bright core, soft rim
    // head (x = +seg + r) bright, tail dimmer
    float along = clamp((v_local.x + seg + r) / (2.0 * (seg + r)), 0.0, 1.0);
    float taper = mix(0.35, 1.0, along);
    float energy = r / (r + seg);
    vec4 le = lut_at(v_temp);
    vec3 c = le.rgb * le.a * v_weight * core * taper * energy;
    if (u_stage == 1) {
        vec4 cov = smokeCoverAt(v_world_uv);
        c *= clamp(cov.r, 0.0, 1.0) * (1.0 - u_glow_through_ceil);
    }
    finalColor = vec4(c, 0.0);
}
