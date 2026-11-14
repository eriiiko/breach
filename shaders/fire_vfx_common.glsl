// fire_vfx_common.glsl -- shared GLSL of the flame VFX passes (graphics arc
// G3 "Fire", docs/g3_fire_design_2026-10-23.md). NOT a shader on its own:
// renderer/fire_vfx.py splices it into every shaders/fire_vfx_*.fs at the
// line "// @include fire_vfx_common.glsl".
//
// THE TEXEL-SPACE CONVENTION (one for every field, renderer/core.py
// PingPong): GL texel (i, j) of a field texture is cell column i from the
// WEST and cell row j from the NORTH (the tile grid's row direction, +y =
// south). A per-tile texture uploaded from a numpy (h, w, 4) array has the
// same orientation (numpy row 0 = GL row 0 = north). Passes address texels
// with gl_FragCoord only -- never fragTexCoord -- so the Y flip of drawing
// into a render texture never enters the data. Linear filtering is never
// relied on: bilinear taps are done by hand (texelFetch), so the GL result
// matches the numpy spec twin (renderer/fire_vfx_ref.py) to float32.
//
// THE DOMAIN (F2). A pass runs on up to 4 disjoint active boxes
// (``u_boxes``, in cells of this grid, end-exclusive -- the same uniform
// pass_quad.vs draws). Each box is its own fluid domain, exactly the twin's
// whole grid: "beyond the grid edge" in the twin is "outside MY box" here
// (velocity taps clamp to the box, flame taps outside it read 0, pressure
// is 0 outside it, an outside neighbour is never solid). Per-TILE fields
// (sources, wind, gas T, local I) clamp at the MAP edge, as the twin's tile
// upsampling clamps at its grid edge.
//
// Render-only, linear quantities in physical units (tiles/s, Kelvin, 1/s).
// No tone-map, encode or dither here (the resolve owns those).

uniform vec2 u_grid;          // the map in tiles (W, H)
uniform float u_m;            // cells per tile of THIS pass's target grid
uniform vec4 u_boxes[4];      // the active boxes, cells (x0, y0, x1, y1); unused = 0
uniform sampler2D u_src_a;    // per tile: s (1/s), q (1/s), T_src (K), cover code
uniform sampler2D u_src_b;    // per tile: puff wobble rates g1, g2 (Hz), T_gas (K), flags
uniform sampler2D u_puff;     // per tile: puff Hz, phases theta0..2 (rad) at t_base
uniform sampler2D u_wind;     // per tile: u_wind (tiles/s) x, y; local I; 0

const int FLAG_SOLID = 1;     // flags bits (renderer/fire_vfx.py FLAG_*)
const int FLAG_VACUUM = 2;

const int COVER_FULL = 16;    // renderer/fire_vfx_ref.py cover code bits
const int FACE_N = 1;
const int FACE_E = 2;
const int FACE_S = 4;
const int FACE_W = 8;

const float TWO_PI = 6.283185307179586;

const ivec2 D_E = ivec2(1, 0);    // +x, east
const ivec2 D_W = ivec2(-1, 0);
const ivec2 D_S = ivec2(0, 1);    // +y, south (data row = GL row)
const ivec2 D_N = ivec2(0, -1);

ivec2 cell_here() { return ivec2(gl_FragCoord.xy); }

ivec2 tile_of(ivec2 cell) { return cell / int(u_m + 0.5); }

vec4 tile_fetch(sampler2D t, ivec2 tile) {
    return texelFetch(t, clamp(tile, ivec2(0), ivec2(u_grid) - 1), 0);
}

int tile_flags(ivec2 tile) { return int(tile_fetch(u_src_b, tile).w + 0.5); }

int cell_flags(ivec2 cell) { return tile_flags(tile_of(cell)); }

bool cell_solid(ivec2 cell) { return (cell_flags(cell) & FLAG_SOLID) != 0; }

bool cell_blocked(ivec2 cell) {
    return (cell_flags(cell) & (FLAG_SOLID | FLAG_VACUUM)) != 0;
}

// ---- the own box ---------------------------------------------------------
ivec4 g_box = ivec4(0);

void find_box(ivec2 c) {
    for (int i = 0; i < 4; ++i) {
        ivec4 b = ivec4(u_boxes[i]);
        if (c.x >= b.x && c.y >= b.y && c.x < b.z && c.y < b.w) {
            g_box = b;
            return;
        }
    }
}

bool in_box(ivec2 c) {
    return c.x >= g_box.x && c.y >= g_box.y && c.x < g_box.z && c.y < g_box.w;
}

ivec2 clamp_box(ivec2 c) { return clamp(c, g_box.xy, g_box.zw - 1); }

// the neighbour, or the cell itself beyond the box edge (twin 'replicate')
ivec2 nbr(ivec2 c, ivec2 d) { ivec2 n = c + d; return in_box(n) ? n : c; }

// twin _solid_nb: a solid neighbour INSIDE the box (outside = open)
bool nb_solid(ivec2 n) { return in_box(n) && cell_solid(n); }

// Bilinear between tile CENTRES with clamped edges at a position in tiles
// (a tile's centre is at k + 0.5) -- fire_vfx_ref._upsample_bilinear.
vec4 tile_bilinear(sampler2D t, vec2 p_tiles) {
    vec2 q = p_tiles - 0.5;
    vec2 f0 = floor(q);
    vec2 fr = q - f0;
    ivec2 i0 = ivec2(f0);
    vec4 v00 = tile_fetch(t, i0);
    vec4 v10 = tile_fetch(t, i0 + ivec2(1, 0));
    vec4 v01 = tile_fetch(t, i0 + ivec2(0, 1));
    vec4 v11 = tile_fetch(t, i0 + ivec2(1, 1));
    return mix(mix(v00, v10, fr.x), mix(v01, v11, fr.x), fr.y);
}

vec2 cell_centre_tiles(ivec2 c) { return (vec2(c) + 0.5) / u_m; }

// The cell's position inside its tile, (0, 1) per axis, north-west origin.
vec2 tile_frac(ivec2 cell) {
    int m = int(u_m + 0.5);
    return (vec2(cell - tile_of(cell) * m) + 0.5) / u_m;
}

// fire_vfx_ref.rasterize_sources: does the tile's cover code fill this cell?
bool source_covers(int cover, vec2 frac, float strip) {
    if ((cover & COVER_FULL) != 0) return true;
    if ((cover & FACE_N) != 0 && frac.y < strip) return true;
    if ((cover & FACE_S) != 0 && frac.y > 1.0 - strip) return true;
    if ((cover & FACE_W) != 0 && frac.x < strip) return true;
    if ((cover & FACE_E) != 0 && frac.x > 1.0 - strip) return true;
    return false;
}

// fire_vfx_ref.puff_factor_rebased: the pool fire's puffing (Cetegen &
// Ahmed 1993) with the per-cluster phase wobble (the flicker no-sync rule,
// §6.3), evaluated at tau = t - t_base from phases the CPU wrapped to
// [0, 2 pi) at t_base in float64 -- so the float32 argument stays small
// however long the game runs (F1 open item 3).
float puff_at(ivec2 tile, float tau, float amp, float wobble) {
    vec4 pf = tile_fetch(u_puff, tile);       // hz, theta0, theta1, theta2
    if (pf.x <= 0.0) return 1.0;
    vec4 b = tile_fetch(u_src_b, tile);       // g1, g2
    float theta = pf.y + TWO_PI * pf.x * tau
        + wobble * (sin(pf.z + TWO_PI * b.x * tau)
                    + sin(pf.w + TWO_PI * b.y * tau));
    return 1.0 + amp * sin(theta);
}

// The cell's source this substep: (s * puff, q * puff, T_src), 0 where the
// tile's cover rule does not fill the cell.
vec3 cell_source(ivec2 c, float tau, float strip, float amp, float wobble) {
    ivec2 t = tile_of(c);
    vec4 a = tile_fetch(u_src_a, t);
    if (!source_covers(int(a.w + 0.5), tile_frac(c), strip)) return vec3(0.0);
    float pf = puff_at(t, tau, amp, wobble);
    return vec3(a.x * pf, a.y * pf, a.z);
}
