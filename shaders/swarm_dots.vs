#version 330
// swarm_dots.vs -- THE swarm dot view (arc #63 P4, renderer/swarm_dots.py
// SwarmDotDraw). RENDER-ONLY, a diagnostic.
//
// No vertex attributes: draw 6 * k vertices with an empty VAO; quad i is
// sprite i of the data texture u_parts ((cap, 2) RGBA32F, texelFetch):
//   row 0 = (x, y, tick dx, tick dy)   world px, y DOWN; the tick's tip from
//                                      the centre (zero: no tick)
//   row 1 = (r, g, b linear, alpha)
// The quad is the box round the dot and its tick, grown by a pixel for the
// fragment's edge coverage. World px -> NDC as raylib's texture mode projects
// (ndc.y = 1 - 2 y / H), the particle draw's convention.

uniform sampler2D u_parts;
uniform vec2 u_world_px;      // the scene RT's size in world px
uniform float u_radius_px;    // the dot's radius

out vec2 v_px;                // this fragment, world px
flat out vec2 v_centre;
flat out vec2 v_tip;
flat out vec4 v_rgba;

const vec2 CORNER[6] = vec2[6](vec2(0.0, 0.0), vec2(0.0, 1.0),
                               vec2(1.0, 1.0), vec2(0.0, 0.0),
                               vec2(1.0, 1.0), vec2(1.0, 0.0));

void main() {
    int i = gl_VertexID / 6;
    vec4 a = texelFetch(u_parts, ivec2(i, 0), 0);
    vec4 b = texelFetch(u_parts, ivec2(i, 1), 0);
    vec2 c = a.xy;
    vec2 tip = a.xy + a.zw;
    float m = u_radius_px + 1.0;
    vec2 lo = min(c, tip) - vec2(m);
    vec2 hi = max(c, tip) + vec2(m);
    vec2 p = mix(lo, hi, CORNER[gl_VertexID % 6]);
    v_px = p;
    v_centre = c;
    v_tip = tip;
    v_rgba = b;
    gl_Position = vec4(p.x / u_world_px.x * 2.0 - 1.0,
                       1.0 - p.y / u_world_px.y * 2.0, 0.0, 1.0);
}
