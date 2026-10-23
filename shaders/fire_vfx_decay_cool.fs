#version 330
// fire_vfx_decay_cool.fs -- flame VFX stage 5 (G3, design §3.4 step 5,
// §13.5, §13.6): R e-folds over lifetime_s; T_vfx relaxes toward the local
// game gas temperature but NEVER rises (T <- min(T, relax(T, T_gas)));
// R below r_cut is cleared to exactly 0 (no stale flame); R = 0 in solid and
// vacuum tiles. Spec: renderer/fire_vfx_ref.py decay_cool.
// Target: the flame field (x = R, y = R*T_vfx in K, zw = 0).

// @include fire_vfx_common.glsl

uniform sampler2D u_self;     // the flame field, front buffer
uniform float u_dt;           // s
uniform float u_lifetime;     // s
uniform float u_cool_rate;    // 1/s
uniform float u_r_eps;
uniform float u_r_cut;

out vec4 finalColor;

void main() {
    ivec2 c = cell_here();
    vec4 v = texelFetch(u_self, c, 0);
    ivec2 t = tile_of(c);
    vec4 b = tile_fetch(u_src_b, t);
    float f = exp(-u_dt / u_lifetime);
    float R = v.x * f;
    float T = R > u_r_eps ? v.y * f / max(R, u_r_eps) : 0.0;
    float a = 1.0 - exp(-u_cool_rate * u_dt);
    T = min(T, T + a * (b.z - T));
    int fl = int(b.w + 0.5);
    bool dead = (fl & (FLAG_SOLID | FLAG_VACUUM)) != 0 || R < u_r_cut;
    finalColor = dead ? vec4(0.0) : vec4(R, R * T, 0.0, 0.0);
}
