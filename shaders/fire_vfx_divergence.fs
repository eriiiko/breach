#version 330
// fire_vfx_divergence.fs -- flame VFX pass "divergence" (G3 F2, design §3.3,
// §3.4 step 3) = twin pressure_rhs: rhs = div(u*) - q * puff, stored in the
// velocity texel's w for the Jacobi sweeps. Central differences (Harris
// 2004); a solid neighbour mirrors the normal component (the face between
// them carries no flow); beyond the box the velocity extends (open
// outflow); 0 divergence in a solid cell. The expansion q (Nguyen, Fedkiw &
// Jensen 2002) is IN the solve, so it survives the projection.
// Target: the velocity field (xy = u, z = p pass through; w = rhs).

// @include fire_vfx_common.glsl

uniform sampler2D u_vel;
uniform float u_tau;          // t - t_base, s (the puffing clock)
uniform float u_strip;
uniform float u_puff_amp;
uniform float u_puff_wobble;

out vec4 finalColor;

vec2 U(ivec2 c) { return texelFetch(u_vel, c, 0).xy; }

void main() {
    ivec2 c = cell_here();
    find_box(c);
    vec4 v = texelFetch(u_vel, c, 0);
    float d = 0.0;
    if (!cell_solid(c)) {
        float ue = nb_solid(c + D_E) ? -v.x : U(nbr(c, D_E)).x;
        float uw = nb_solid(c + D_W) ? -v.x : U(nbr(c, D_W)).x;
        float vs = nb_solid(c + D_S) ? -v.y : U(nbr(c, D_S)).y;
        float vn = nb_solid(c + D_N) ? -v.y : U(nbr(c, D_N)).y;
        d = ((ue - uw) + (vs - vn)) * (u_m * 0.5);
    }
    float q = cell_source(c, u_tau, u_strip, u_puff_amp, u_puff_wobble).y;
    finalColor = vec4(v.xyz, d - q);
}
