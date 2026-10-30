#version 330
// fire_vfx_jacobi.fs -- flame VFX pass "jacobi" (G3 F2, design §13.1) = twin
// jacobi_sweep: ONE Jacobi sweep of the SCREENED Poisson problem
//     (lap - 1/l^2) p = rhs
// p = 0 beyond the box edge and in vacuum (open: nothing to push against);
// a solid neighbour is Neumann (drops out of the stencil); p = 0 in solid
// and vacuum cells. The screening (the plume leaving the plane upward)
// makes every enclosed region's problem unique and bounded, so the
// warm-started p of a sealed room cannot grow without limit.
// Target: the velocity field (xy = u, w = rhs pass through; z = p).

// @include fire_vfx_common.glsl

uniform sampler2D u_vel;
uniform float u_screen;       // h^2 / l^2 (cells)

out vec4 finalColor;

float tap(ivec2 n, inout float n_open) {
    if (!in_box(n)) {
        n_open += 1.0;            // beyond the edge: open, p = 0
        return 0.0;
    }
    if (cell_solid(n)) return 0.0;
    n_open += 1.0;
    return texelFetch(u_vel, n, 0).z;
}

void main() {
    ivec2 c = cell_here();
    find_box(c);
    vec4 v = texelFetch(u_vel, c, 0);
    if (cell_blocked(c)) {
        finalColor = vec4(v.xy, 0.0, v.w);
        return;
    }
    float n_open = 0.0;
    float acc = tap(c + D_E, n_open) + tap(c + D_W, n_open)
              + tap(c + D_S, n_open) + tap(c + D_N, n_open);
    float hc2 = 1.0 / (u_m * u_m);
    finalColor = vec4(v.xy, (acc - hc2 * v.w) / (n_open + u_screen), v.w);
}
