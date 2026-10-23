#version 330
// fire_vfx_follow_wind.fs -- flame VFX stage 1b (G3, design §3.4 step 1,
// §13.2): relax the flame-scale velocity toward the TAMED wind,
// u += (1 - exp(-k_follow dt)) (u_wind - u), with u_wind = tamed * tps in
// tiles/s, bilinear between tile centres, 0 in solid and vacuum tiles.
// Spec: renderer/fire_vfx_ref.py follow_wind + prepare_tick's u_wind.
// Target: the velocity field (xy = u tiles/s, z = p, w unused); z and w pass
// through unchanged. Texel space: see fire_vfx_common.glsl.

// @include fire_vfx_common.glsl

uniform sampler2D u_self;     // the velocity field, front buffer
uniform float u_k_follow;     // 1/s
uniform float u_dt;           // s

out vec4 finalColor;

void main() {
    ivec2 c = cell_here();
    vec4 v = texelFetch(u_self, c, 0);
    int fl = tile_flags(tile_of(c));
    vec2 uw = vec2(0.0);
    if ((fl & (FLAG_SOLID | FLAG_VACUUM)) == 0)
        uw = tile_bilinear(u_wind, (vec2(c) + 0.5) / u_m).xy;
    float a = 1.0 - exp(-u_k_follow * u_dt);
    v.xy += a * (uw - v.xy);
    finalColor = v;
}
