#version 330
// pass_quad.vs -- the vertex stage of every DIRECT-GL simulation pass
// (renderer/core.py PingPong.direct_pass, load_pass_shader; graphics arc G3
// F2). No vertex attributes: draw 6 * n vertices with an empty VAO and
// quad k covers the texel rect u_boxes[k] = (x0, y0, x1, y1), end-exclusive,
// in the TARGET's texel space (GL row j = data row j: no Y flip anywhere,
// the PingPong convention). Fragment stages address texels with
// gl_FragCoord only.

uniform vec4 u_boxes[4];      // texel rects of this pass (also read by the fs)
uniform vec2 u_target;        // the target's size in texels

const vec2 CORNER[6] = vec2[6](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
                               vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));

void main() {
    vec4 r = u_boxes[gl_VertexID / 6];
    vec2 p = mix(r.xy, r.zw, CORNER[gl_VertexID % 6]);
    gl_Position = vec4(p / u_target * 2.0 - 1.0, 0.0, 1.0);
}
