#version 330
// fire_vfx_flame.fs -- flame VFX pass "flame" (G3 F2, design §3.4 steps 4-5,
// §13.5, §13.6) = twin advect_flame -> inject -> decay_cool, on the
// projected velocity:
//   4a. advect R and the premultiplied R*T: MacCormack (Selle, Fedkiw, Kim,
//       Liu & Rossignac 2008) -- forward SL, backward SL of the forward
//       field, corrected by half the round-trip error, reverted to the SL
//       value where it leaves the min/max of the forward trace's four taps;
//       taps outside the own box read 0 (nothing flows in from beyond the
//       domain). Then the TEMPERATURE limiter: R*T <= R * (hottest tap T),
//       which keeps T_vfx <= T_src. The forward field at the backward taps
//       is recomputed here (one pass, no intermediate target);
//   4b. inject s * puff * dt at T_src (premultiplied), through the cover rule;
//   5.  R e-folds over lifetime_s; T_vfx relaxes toward the local game gas T
//       but never rises; R < r_cut -> exactly 0; R = 0 in solid and vacuum.
// Target: the flame field (x = R, y = R*T_vfx in K, zw = 0).

// @include fire_vfx_common.glsl

uniform sampler2D u_flame;    // the flame field, front buffer
uniform sampler2D u_vel;      // the PROJECTED velocity (cells of the same grid)
uniform float u_dt;
uniform float u_tau;          // t - t_base, s (the puffing clock)
uniform float u_strip;
uniform float u_puff_amp;
uniform float u_puff_wobble;
uniform float u_lifetime;
uniform float u_cool_rate;
uniform float u_r_eps;
uniform float u_r_cut;
uniform int u_maccormack;

out vec4 finalColor;

vec2 F(ivec2 n) { return in_box(n) ? texelFetch(u_flame, n, 0).xy : vec2(0.0); }

float t_of(vec2 f) { return f.x > u_r_eps ? f.y / max(f.x, u_r_eps) : 0.0; }

ivec2 tap0(vec2 pos, out vec2 fr) {
    vec2 f0 = floor(pos);
    fr = pos - f0;
    return ivec2(clamp(f0, vec2(-4.0), u_grid * u_m + 4.0));
}

vec2 bilin(vec2 a, vec2 b, vec2 c, vec2 d, vec2 fr) {
    return mix(mix(a, b, fr.x), mix(c, d, fr.x), fr.y);
}

// the forward SL value of (R, RT) at texel y (inside the box)
vec2 fwd_at(ivec2 y) {
    vec2 u = texelFetch(u_vel, y, 0).xy;
    vec2 fr;
    ivec2 i0 = tap0(vec2(y) - u * (u_dt * u_m), fr);
    return bilin(F(i0), F(i0 + ivec2(1, 0)), F(i0 + ivec2(0, 1)),
                 F(i0 + ivec2(1, 1)), fr);
}

void main() {
    ivec2 c = cell_here();
    find_box(c);
    vec2 u = texelFetch(u_vel, c, 0).xy;
    vec2 fr;
    ivec2 i0 = tap0(vec2(c) - u * (u_dt * u_m), fr);
    vec2 f00 = F(i0), f10 = F(i0 + ivec2(1, 0));
    vec2 f01 = F(i0 + ivec2(0, 1)), f11 = F(i0 + ivec2(1, 1));
    vec2 fwd = bilin(f00, f10, f01, f11, fr);
    vec2 lo = min(min(f00, f10), min(f01, f11));
    vec2 hi = max(max(f00, f10), max(f01, f11));
    float t_max = max(max(t_of(f00), t_of(f10)), max(t_of(f01), t_of(f11)));
    if (u_maccormack != 0) {
        vec2 fb;
        ivec2 j0 = tap0(vec2(c) + u * (u_dt * u_m), fb);
        ivec2 y00 = j0, y10 = j0 + ivec2(1, 0), y01 = j0 + ivec2(0, 1),
              y11 = j0 + ivec2(1, 1);
        vec2 b00 = in_box(y00) ? fwd_at(y00) : vec2(0.0);
        vec2 b10 = in_box(y10) ? fwd_at(y10) : vec2(0.0);
        vec2 b01 = in_box(y01) ? fwd_at(y01) : vec2(0.0);
        vec2 b11 = in_box(y11) ? fwd_at(y11) : vec2(0.0);
        vec2 back = bilin(b00, b10, b01, b11, fb);
        vec2 mc = fwd + 0.5 * (F(c) - back);
        fwd.x = (mc.x < lo.x || mc.x > hi.x) ? fwd.x : mc.x;
        fwd.y = (mc.y < lo.y || mc.y > hi.y) ? fwd.y : mc.y;
    }
    float R = max(fwd.x, 0.0);
    float RT = min(max(fwd.y, 0.0), R * t_max);
    // 4b. inject
    vec3 src = cell_source(c, u_tau, u_strip, u_puff_amp, u_puff_wobble);
    float add = src.x * u_dt;
    R += add;
    RT += add * src.z;
    // 5. decay, cool (never heating), clear
    float f = exp(-u_dt / u_lifetime);
    R *= f;
    float T = R > u_r_eps ? RT * f / max(R, u_r_eps) : 0.0;
    float a = 1.0 - exp(-u_cool_rate * u_dt);
    float t_gas = tile_fetch(u_src_b, tile_of(c)).z;
    T = min(T, T + a * (t_gas - T));
    bool dead = cell_blocked(c) || R < u_r_cut;
    finalColor = dead ? vec4(0.0) : vec4(R, R * T, 0.0, 0.0);
}
