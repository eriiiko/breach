#version 330
// fire_vfx_project.fs -- flame VFX pass "project" (G3 F2, design §3.4 step 3)
// = twin project: subtract grad p (central; a solid neighbour mirrors p, p =
// 0 beyond the box edge), then NO-THROUGH: no velocity component points
// into a solid neighbour; solid cells at rest.
// Target: the velocity field (xy = u; z = p, w = rhs pass through).

// @include fire_vfx_common.glsl

uniform sampler2D u_vel;

out vec4 finalColor;

float pn(ivec2 c, ivec2 d, float p) {
    ivec2 n = c + d;
    if (!in_box(n)) return 0.0;
    if (cell_solid(n)) return p;
    return texelFetch(u_vel, n, 0).z;
}

void main() {
    ivec2 c = cell_here();
    find_box(c);
    vec4 v = texelFetch(u_vel, c, 0);
    if (cell_solid(c)) {
        finalColor = vec4(0.0, 0.0, v.z, v.w);
        return;
    }
    float h2 = u_m * 0.5;
    float ux = v.x - (pn(c, D_E, v.z) - pn(c, D_W, v.z)) * h2;
    float uy = v.y - (pn(c, D_S, v.z) - pn(c, D_N, v.z)) * h2;
    if (nb_solid(c + D_E)) ux = min(ux, 0.0);
    if (nb_solid(c + D_W)) ux = max(ux, 0.0);
    if (nb_solid(c + D_S)) uy = min(uy, 0.0);
    if (nb_solid(c + D_N)) uy = max(uy, 0.0);
    finalColor = vec4(ux, uy, v.z, v.w);
}
