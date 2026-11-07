#version 330
// particle_debris.vs -- THE particle primitive's DEBRIS stage (graphics G4
// P4, renderer/particles.py ParticleDraw.draw_debris). RENDER-ONLY.
//
// No vertex attributes: draw 6 * k vertices with an empty VAO; quad i is
// chip i of the data texture u_parts ((cap, 2) RGBA32F, texelFetch):
//   row 0 = (x, y, rotation rad, half-width px)  world px, y DOWN
//   row 1 = (opacity, 0, 0, 0)
// A chip is a small square rotated by its hashed angle, grown by half a
// pixel for the fragment's edge coverage. World px -> NDC exactly as
// raylib's texture mode projects (ndc.y = 1 - 2 y / H), as the sprite does.

uniform sampler2D u_parts;
uniform vec2 u_world_px;      // the scene RT's size in world px

out vec2 v_local;             // px from the chip's centre, in its own frame
out float v_half;             // its half-width, px
out float v_alpha;
out vec2 v_world_px;          // world px (y down) for the light field read

// Wound counter-clockwise AFTER the y-down -> NDC flip (raylib culls back
// faces), the sprite's corner order; (dir, nrm) is a proper rotation, so
// every angle keeps the winding.
const vec2 CORNER[6] = vec2[6](vec2(-1.0, -1.0), vec2(-1.0, 1.0),
                               vec2(1.0, 1.0), vec2(-1.0, -1.0),
                               vec2(1.0, 1.0), vec2(1.0, -1.0));

void main() {
    int i = gl_VertexID / 6;
    vec4 a = texelFetch(u_parts, ivec2(i, 0), 0);
    vec4 b = texelFetch(u_parts, ivec2(i, 1), 0);
    float hs = max(a.w, 0.5);
    vec2 dir = vec2(cos(a.z), sin(a.z));
    vec2 nrm = vec2(-dir.y, dir.x);
    vec2 loc = CORNER[gl_VertexID % 6] * (hs + 0.5);
    vec2 p = a.xy + dir * loc.x + nrm * loc.y;
    v_local = loc;
    v_half = hs;
    v_alpha = b.x;
    v_world_px = p;
    gl_Position = vec4(p.x / u_world_px.x * 2.0 - 1.0,
                       1.0 - p.y / u_world_px.y * 2.0, 0.0, 1.0);
}
