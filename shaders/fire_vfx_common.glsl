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
// Render-only, linear quantities in physical units (tiles/s, Kelvin, 1/s).
// No tone-map, encode or dither here (the resolve owns those).

uniform vec2 u_grid;          // the map in tiles (W, H)
uniform float u_m;            // cells per tile of THIS pass's target grid
uniform sampler2D u_src_a;    // per tile: s (1/s), q (1/s), T_src (K), cover code
uniform sampler2D u_src_b;    // per tile: puff Hz, puff phase, T_gas (K), flags
uniform sampler2D u_wind;     // per tile: u_wind (tiles/s) x, y; local I; 0

const int FLAG_SOLID = 1;     // flags bits (renderer/fire_vfx.py FLAG_*)
const int FLAG_VACUUM = 2;
const int FLAG_DOMAIN = 4;    // the tile is inside an active box this frame

const int COVER_FULL = 16;    // renderer/fire_vfx_ref.py cover code bits
const int FACE_N = 1;
const int FACE_E = 2;
const int FACE_S = 4;
const int FACE_W = 8;

const float TWO_PI = 6.283185307179586;

ivec2 cell_here() { return ivec2(gl_FragCoord.xy); }

ivec2 tile_of(ivec2 cell) { return cell / int(u_m + 0.5); }

vec4 tile_fetch(sampler2D t, ivec2 tile) {
    return texelFetch(t, clamp(tile, ivec2(0), ivec2(u_grid) - 1), 0);
}

int tile_flags(ivec2 tile) { return int(tile_fetch(u_src_b, tile).w + 0.5); }

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

// fire_vfx_ref.puff_factor: the pool fire's puffing (Cetegen & Ahmed 1993)
// with the per-cluster phase wobble (the flicker no-sync rule, §6.3).
float puff_factor(float hz, float phase, float t, float amp, float wobble) {
    if (hz <= 0.0) return 1.0;
    float u = phase / TWO_PI;
    float g1 = hz * (0.17 + 0.22 * fract(u * 13.0));
    float g2 = hz * (0.41 + 0.22 * fract(u * 29.0));
    float theta = TWO_PI * hz * t + phase
        + wobble * (sin(TWO_PI * g1 * t + 1.7 * phase)
                    + sin(TWO_PI * g2 * t + 2.9 * phase));
    return 1.0 + amp * sin(theta);
}
