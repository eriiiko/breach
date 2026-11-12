#version 330
// muzzle_flame.vs -- THE muzzle flame's quads (#87, renderer/muzzle_flame.py
// MuzzleFlameDraw). RENDER-ONLY.
//
// No vertex attributes: draw 6 * k vertices with an empty VAO; quad i is
// flash i of the data texture u_data ((cap, 9) RGBA32F, texelFetch):
//   row 0 = (x, y, dir x, dir y)   muzzle in world px (y DOWN, the scene
//                                  RT's raylib space), the barrel's unit
//                                  direction on the floor
//   row 1 = (age, noise u, noise v, px per metre)
//   row 3 = (T core, T edge, tear, half m)   -- half m = the square's half side
// The quad is that square, laid along the barrel. World px -> NDC exactly as
// the particle sprites (ndc.y = 1 - 2 y / H).

uniform sampler2D u_data;
uniform vec2 u_world_px;      // the scene RT's size in world px

out vec2 v_local;             // METRES from the muzzle: x along the barrel
out vec2 v_world_uv;          // world uv (0..1, y down) for the smoke cover
flat out int v_index;

// Wound counter-clockwise AFTER the y-down -> NDC flip (raylib culls back
// faces); (dir, nrm) is a proper rotation, so every direction keeps it.
const vec2 CORNER[6] = vec2[6](vec2(-1.0, -1.0), vec2(-1.0, 1.0),
                               vec2(1.0, 1.0), vec2(-1.0, -1.0),
                               vec2(1.0, 1.0), vec2(1.0, -1.0));

void main() {
    int i = gl_VertexID / 6;
    vec4 a = texelFetch(u_data, ivec2(i, 0), 0);
    vec4 b = texelFetch(u_data, ivec2(i, 1), 0);
    float half_m = texelFetch(u_data, ivec2(i, 3), 0).w;
    vec2 dir = a.zw;
    vec2 nrm = vec2(-dir.y, dir.x);
    vec2 loc = CORNER[gl_VertexID % 6] * half_m;
    vec2 p = a.xy + (dir * loc.x + nrm * loc.y) * b.w;
    v_local = loc;
    v_index = i;
    v_world_uv = p / u_world_px;
    gl_Position = vec4(p.x / u_world_px.x * 2.0 - 1.0,
                       1.0 - p.y / u_world_px.y * 2.0, 0.0, 1.0);
}
