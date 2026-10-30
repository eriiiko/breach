#version 330
// fire_vfx_vel_advect.fs -- flame VFX pass "vel_advect" (G3 F2, design §3.4
// step 1, §13.2) = twin advect_velocity -> follow_wind:
//   1a. semi-Lagrangian self-advection of u (Stam 1999): u(x - u(x) dt), the
//       four taps clamped to the own box (the twin's 'clamp' mode);
//   1b. relax toward the tamed wind, u += (1 - exp(-k_follow dt)) (u_wind -
//       u), u_wind = tamed * tps in tiles/s, bilinear between tile centres,
//       0 in solid and vacuum tiles.
// Target: the velocity field (xy = u tiles/s, z = p, w = rhs); z and w pass
// through. Spec: renderer/fire_vfx_ref.py advect_velocity, follow_wind,
// prepare_tick's u_wind. Texel space and domain: fire_vfx_common.glsl.

// @include fire_vfx_common.glsl

uniform sampler2D u_vel;      // the velocity field, front buffer
uniform float u_k_follow;     // 1/s
uniform float u_dt;           // s

out vec4 finalColor;

vec2 U(ivec2 c) { return texelFetch(u_vel, clamp_box(c), 0).xy; }

void main() {
    ivec2 c = cell_here();
    find_box(c);
    vec4 v = texelFetch(u_vel, c, 0);
    vec2 pos = vec2(c) - v.xy * (u_dt * u_m);
    vec2 f0 = floor(pos);
    vec2 fr = pos - f0;
    ivec2 i0 = ivec2(clamp(f0, vec2(-4.0), u_grid * u_m + 4.0));
    vec2 ua = mix(mix(U(i0), U(i0 + ivec2(1, 0)), fr.x),
                  mix(U(i0 + ivec2(0, 1)), U(i0 + ivec2(1, 1)), fr.x), fr.y);
    vec2 uw = vec2(0.0);
    if (!cell_blocked(c))
        uw = tile_bilinear(u_wind, cell_centre_tiles(c)).xy;
    float a = 1.0 - exp(-u_k_follow * u_dt);
    finalColor = vec4(ua + a * (uw - ua), v.z, v.w);
}
