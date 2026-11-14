#version 330
// particle_sprite.vs -- THE particle primitive's sprite quads (graphics arc
// G3 F4a, renderer/particles.py ParticleDraw). RENDER-ONLY.
//
// No vertex attributes: draw 6 * k vertices with an empty VAO; quad i is
// particle i of the data texture u_parts ((cap, 2) RGBA32F, texelFetch):
//   row 0 = (x, y, vx, vy)  world px (y DOWN, the scene RT's raylib space)
//                           and px / s
//   row 1 = (T K, weight, radius px, streak s)
// The quad is a capsule's bounding box laid along the velocity: a trail of
// |v| * streak behind the particle (motion stretch), radius r around it.
// World px -> NDC exactly as raylib's texture mode projects (ortho 0..W,
// 0..H with y down: ndc.y = 1 - 2 y / H), so the sprites land where every
// other scene layer draws.

uniform sampler2D u_parts;
uniform vec2 u_world_px;      // the scene RT's size in world px

out vec2 v_local;             // px from the trail's centre: x along v, y across
out float v_half_seg;         // half length of the trail's core segment, px
out float v_radius;
out float v_temp;
out float v_weight;
out vec2 v_world_uv;          // world uv (0..1, y down) for the smoke cover

// Wound counter-clockwise AFTER the y-down -> NDC flip (raylib keeps
// GL_CULL_FACE on with back faces culled: a clockwise quad never draws).
// (dir, nrm) is a proper rotation, so every direction keeps the winding.
const vec2 CORNER[6] = vec2[6](vec2(-1.0, -1.0), vec2(-1.0, 1.0),
                               vec2(1.0, 1.0), vec2(-1.0, -1.0),
                               vec2(1.0, 1.0), vec2(1.0, -1.0));

void main() {
    int i = gl_VertexID / 6;
    vec4 a = texelFetch(u_parts, ivec2(i, 0), 0);
    vec4 b = texelFetch(u_parts, ivec2(i, 1), 0);
    vec2 v = a.zw;
    float sp = length(v);
    vec2 dir = sp > 1e-3 ? v / sp : vec2(1.0, 0.0);
    vec2 nrm = vec2(-dir.y, dir.x);
    float r = max(b.z, 0.5);
    float seg = 0.5 * sp * b.w;               // half the trail length
    vec2 centre = a.xy - dir * seg;           // the trail lies BEHIND it
    vec2 c = CORNER[gl_VertexID % 6];
    vec2 loc = vec2(c.x * (seg + r), c.y * r);
    vec2 p = centre + dir * loc.x + nrm * loc.y;
    v_local = loc;
    v_half_seg = seg;
    v_radius = r;
    v_temp = b.x;
    v_weight = b.y;
    v_world_uv = p / u_world_px;
    gl_Position = vec4(p.x / u_world_px.x * 2.0 - 1.0,
                       1.0 - p.y / u_world_px.y * 2.0, 0.0, 1.0);
}
