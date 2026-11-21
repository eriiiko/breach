#version 330
// water_scatter.vs -- PROTOTYPE (#97 reassessment, branch 97-reassess): THE
// SCATTER of the caustic, Wallace's own form (Evan Wallace, "WebGL Water",
// 2011, MIT; madebyevan.com/webgl-water; archived
// docs/papers/wallace2011_webgl_water_renderer.js): a grid over the water
// surface, each sample moved by the refraction map x -> x + a grad h
// (a = depth x (1 - 1/1.333), the gather's own map) onto the floor, each
// grid triangle rasterised ADDITIVELY into the light image with the ratio of
// its area before to after -- so every bit of light lands somewhere. The
// twin is renderer/water_surface.py::scatter_light (the same grid, offset,
// split and cap).
//
// No vertex attributes: draw 6 x (quads) vertices with the empty VAO (the
// PingPong's direct_pass). Quad q of the rect u_rect (x0, y0, w, h in cells)
// has corner samples (i, j), (i+1, j), (i, j+1), (i+1, j+1), split along its
// anti-diagonal; one row and column past the rect read the last sample
// (clamped), so every cell centre has a quad over it. Each vertex fetches its
// triangle's three samples and hands the triangle's density ratio flat.

uniform sampler2D u_slope;    // (dh/dx, dh/dy, depth m, 1) per sample
uniform vec4 u_rect;          // (x0, y0, w, h) cells
uniform vec2 u_target;        // the image's size in texels
uniform vec4 u_move;          // (CAUSTIC_K / cell m, sample offset, ratio cap, 0)

flat out float v_ratio;

const ivec2 CORN[6] = ivec2[6](ivec2(0, 0), ivec2(1, 0), ivec2(0, 1),
                               ivec2(1, 0), ivec2(1, 1), ivec2(0, 1));

vec2 samplePos(ivec2 v) {
    ivec2 r0 = ivec2(u_rect.xy);
    ivec2 r1 = r0 + ivec2(u_rect.zw) - 1;
    vec4 s = texelFetch(u_slope, clamp(v, r0, r1), 0);
    return vec2(v) + u_move.y + s.xy * (s.z * u_move.x);
}

void main() {
    int w = int(u_rect.z);
    int q = gl_VertexID / 6;
    int c = gl_VertexID % 6;
    ivec2 base = ivec2(u_rect.xy) + ivec2(q % w, q / w);
    int t0 = (c / 3) * 3;
    vec2 p0 = samplePos(base + CORN[t0]);
    vec2 p1 = samplePos(base + CORN[t0 + 1]);
    vec2 p2 = samplePos(base + CORN[t0 + 2]);
    vec2 e1 = p1 - p0;
    vec2 e2 = p2 - p0;
    float cr = e1.x * e2.y - e1.y * e2.x;
    // area before 1/2, after |cr|/2: the density this triangle carries
    v_ratio = min(1.0 / max(abs(cr), 1e-30), u_move.z);
    int k = c - t0;
    vec2 me = k == 0 ? p0 : (k == 1 ? p1 : p2);
    gl_Position = vec4(me / u_target * 2.0 - 1.0, 0.0, 1.0);
}
