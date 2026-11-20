#version 330
// larva_body.vs -- THE swarm larvae's bodies (arc #63,
// renderer/larva_bodies.py LarvaBodyDraw). RENDER-ONLY.
//
// No vertex attributes: ONE glDrawArraysInstanced over an empty VAO.
// gl_VertexID indexes THE grub mesh (propgen.generate_larva) in u_mesh
// ((n, 2) RGBA32F: row 0 = position xyz metres (x along the body, tail to
// head; y up; z lateral) + s along the body; row 1 = normal xyz + shade);
// gl_InstanceID indexes the larva in u_inst (instance i at column i % W,
// rows 2 (i / W) and + 1):
//   row 0 = (x, y world px (y down), heading rad, wriggle phase rad)
//   row 1 = (limp 0..1, alpha 0..1, coma 0/1, size factor)
//
// The wriggle: ONE travelling wave per larva, argument
// a = 2 pi waves s - phase. Peristalsis moves each point along the body by
// P L sin(a) / (2 pi waves), so the local stretch is 1 + P cos(a) (P < 0.9:
// never folds), and the cross-section scales by 1 / sqrt(stretch) (a
// squeezed segment bulges). The sway moves it sideways by A L sin(a). Both
// fade with limp; a limp body flattens to u_dead.x and spreads. The phase
// is the instance's own (distance walked): no clock here.
//
// World frame: the lit-3D block's Y-up world px (X = x, Y = height,
// Z = y down), so _FIELD_SAMPLE_GLSL reads fragWorldPos / fragWorldNormal
// as for a prop. World px -> NDC as raylib's texture mode projects
// (ndc.y = 1 - 2 y / H), the debris' convention; z = 0.

uniform sampler2D u_inst;
uniform sampler2D u_mesh;
uniform int   u_inst_w;
uniform vec2  u_world_px;     // the scene RT's size, world px
uniform float u_m_to_px;      // world px per metre
uniform float u_len_m;        // the template's length, metres
uniform vec4  u_wave;         // (peristalsis, sway, waves along the body, -)
uniform vec2  u_dead;         // (dead_flatten, -)

out vec3 fragWorldPos;
out vec3 v_normal;          // the bent normal, world
out vec3 v_along;             // the body's axis at this point, world
out float v_s;                // along the body, 0 tail .. 1 head
out float v_shade;            // the mesh's baked occlusion
out float v_limp;
out float v_alpha;
out float v_coma;

const float TWO_PI = 6.28318530718;

void main() {
    int i = gl_InstanceID;
    ivec2 at = ivec2(i % u_inst_w, 2 * (i / u_inst_w));
    vec4 a = texelFetch(u_inst, at, 0);
    vec4 b = texelFetch(u_inst, at + ivec2(0, 1), 0);
    vec4 m0 = texelFetch(u_mesh, ivec2(gl_VertexID, 0), 0);
    vec4 m1 = texelFetch(u_mesh, ivec2(gl_VertexID, 1), 0);

    float limp = clamp(b.x, 0.0, 1.0);
    float live = 1.0 - limp;
    float s = m0.w;
    float len_m = u_len_m;                  // the template's length
    vec3 p = m0.xyz;
    vec3 n = m1.xyz;

    float waves = max(u_wave.z, 1e-3);
    float arg = TWO_PI * waves * s - a.w;
    float P = u_wave.x * live;
    float A = u_wave.y * live;
    float stretch = 1.0 + P * cos(arg);
    float dx = len_m * P * sin(arg) / (TWO_PI * waves);
    float dz = len_m * A * sin(arg);
    // the axis' slope: d(dz)/d(x0) over the stretch
    float slope = TWO_PI * waves * A * cos(arg) / stretch;
    float bulge = inversesqrt(max(stretch, 0.1));
    float flat_y = mix(1.0, u_dead.x, limp);
    float widen = mix(1.0, 1.0 + 0.5 * (1.0 - u_dead.x), limp);

    vec3 q = vec3(p.x + dx, p.y * bulge * flat_y, p.z * bulge * widen + dz);
    // normals: the inverse-transpose of the scale, then the axis' bend
    vec3 nn = normalize(vec3(n.x, n.y / flat_y, n.z / widen));
    vec2 t = normalize(vec2(1.0, slope));   // the bent axis in (x, z)
    nn = vec3(nn.x * t.x - nn.z * t.y, nn.y, nn.x * t.y + nn.z * t.x);

    // place: rotate (x, z) by the heading, scale to world px, translate
    float sc = u_m_to_px * b.w;
    float ch = cos(a.z), sh = sin(a.z);
    vec2 xz = vec2(q.x * ch - q.z * sh, q.x * sh + q.z * ch) * sc;
    vec3 world = vec3(a.x + xz.x, q.y * sc, a.y + xz.y);
    fragWorldPos = world;
    v_normal = vec3(nn.x * ch - nn.z * sh, nn.y, nn.x * sh + nn.z * ch);
    vec2 ax = vec2(t.x * ch - t.y * sh, t.x * sh + t.y * ch);
    v_along = vec3(ax.x, 0.0, ax.y);
    v_s = s;
    v_shade = m1.w;
    v_limp = limp;
    v_alpha = clamp(b.y, 0.0, 1.0);
    v_coma = b.z;
    gl_Position = vec4(world.x / u_world_px.x * 2.0 - 1.0,
                       1.0 - world.z / u_world_px.y * 2.0, 0.0, 1.0);
}
