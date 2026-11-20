#version 330
// water_foam.fs -- THE water foam's step (water look arc #97, patch W2;
// docs/water_look_design_2026-11-19.md W2 + critique M9-M11). One SUBSTEP of
// renderer/water_foam.py::substep, its numpy twin -- a change lands there
// first, then here, and tests/gl_water_foam_check.py holds the two equal:
//   1. ADVECT  MacCormack + clamp (Selle, Fedkiw, Kim, Liu & Rossignac,
//      J. Sci. Comput. 35, 2008): a forward semi-Lagrangian trace, a backward
//      trace of its result, corrected by half the round-trip error, reverted
//      to the forward value where it leaves the min / max of the forward
//      trace's four taps. A tap on a BLOCKED cell (dry or solid tile) or off
//      the grid reads the traced cell's OWN value (the backtrace clamped to
//      wet, non-solid cells). The forward value at the backward taps is
//      recomputed here (one pass, no intermediate target; the flame pass's
//      way).
//   2. SOURCE  + rate * dt (whitewater by Froude number, W1's strength term,
//      the advancing edge, walls -- the twin's ``sources``; W3's splats --
//      the twin's ``splat_rate``).
//   3. DECAY   * exp(-dt / foam_decay_s) (handed in as u_decay_f).
//   4. CLAMP   to [0, 1]; 0 on a blocked cell.
// Cell (i, j) of the foam grid (u_m cells per tile) has its centre at
// ((j + .5) / m, (i + .5) / m) tiles. The tile textures (RGBA16F, the
// water pass's per-tick upload) are read with texelFetch and a hand-made
// tile-centred bilinear (CLAMP) -- the twin's heat_shimmer.bilinear_tiles --
// so the GPU computes the twin's numbers, whatever the sampler's filter.
// Texels addressed by gl_FragCoord, GL row = data row (the PingPong rule).
// Target: the foam image (r = amount, gba = 0).

uniform sampler2D u_water;    // (ripple, ripple_v, depth m, relative agitation)
uniform sampler2D u_flow;     // (vx, vy m/s, solid, previous depth m)
uniform sampler2D u_foam;     // the foam amount, front buffer (cells)
uniform vec2 u_grid;          // the map in tiles (w, h)
uniform float u_m;            // foam cells per tile
uniform float u_dt;           // this substep, s
uniform float u_cell_k;       // cells moved per (m/s) this substep
uniform float u_decay_f;      // exp(-dt / foam_decay_s)
// (gain_froude, froude, gain_steep, gain_edge)
uniform vec4 u_src_a;
// (gain_wall, flow_speed_ref, agitation_ref, flow_edge_depth_m)
uniform vec4 u_src_b;
// W3: the frame's splats (the twin's splat_rate / frame_splats): a ring, a
// wake, a landing droplet's dab -- capsule sources, tiles and amount/s
const int SPLAT_MAX = 32;
uniform int  u_splat_n;
uniform vec4 u_splat_a[SPLAT_MAX];   // (ax, ay, bx, by) the segment, tiles
uniform vec4 u_splat_b[SPLAT_MAX];   // (r_c, hw tiles, rate amount/s, 0)

out vec4 finalColor;

const float G_MPS2 = 9.81;
const float CFL_MAX_CELLS = 1.0;

ivec2 gridTiles() { return ivec2(u_grid + 0.5); }
ivec2 gridCells() { return ivec2(u_grid * u_m + 0.5); }

vec4 tileAt(sampler2D t, ivec2 ij) {
    return texelFetch(t, clamp(ij, ivec2(0), gridTiles() - 1), 0);
}

// The GPU's CLAMP bilinear of a per-tile texture at p (tiles), by hand.
vec4 tileBilinear(sampler2D t, vec2 p) {
    vec2 f = p - 0.5;
    vec2 b = floor(f);
    vec2 a = f - b;
    ivec2 i = ivec2(b);
    vec4 top = tileAt(t, i) * (1.0 - a.x) + tileAt(t, i + ivec2(1, 0)) * a.x;
    vec4 bot = tileAt(t, i + ivec2(0, 1)) * (1.0 - a.x)
             + tileAt(t, i + ivec2(1, 1)) * a.x;
    return top * (1.0 - a.y) + bot * a.y;
}

vec2 cellPos(ivec2 c) { return (vec2(c) + 0.5) / u_m; }

// W3: the splats' rate at p (tiles): rate x max(0, 1 - |d - r_c| / hw),
// d = p's distance to the splat's segment (the twin's splat_rate).
float splatRate(vec2 p) {
    float r = 0.0;
    for (int k = 0; k < SPLAT_MAX; ++k) {
        if (k >= u_splat_n) break;
        vec4 a = u_splat_a[k];
        vec4 b = u_splat_b[k];
        vec2 e = a.zw - a.xy;
        vec2 q = p - a.xy;
        float ee = dot(e, e);
        float t = ee > 0.0 ? clamp(dot(q, e) / ee, 0.0, 1.0) : 0.0;
        float d = length(q - t * e);
        r += b.z * max(1.0 - abs(d - b.x) / max(b.y, 1e-9), 0.0);
    }
    return r;
}

bool blocked(ivec2 c) {
    ivec2 n = gridCells();
    if (c.x < 0 || c.y < 0 || c.x >= n.x || c.y >= n.y) return true;
    ivec2 tile = c / int(u_m + 0.5);
    return tileAt(u_water, tile).b <= 0.0 || tileAt(u_flow, tile).b > 0.5;
}

// W1's effective flow (water_surface.edge_flow) at a cell centre, m/s.
vec2 effFlow(ivec2 c, out float depth) {
    vec2 p = cellPos(c);
    depth = tileBilinear(u_water, p).b;
    float e = clamp(depth / max(u_src_b.w, 1e-12), 0.0, 1.0);
    return tileBilinear(u_flow, p).xy * (e * e * (3.0 - 2.0 * e));
}

// This substep's displacement in cells, its length clamped to one cell.
vec2 disp(ivec2 c) {
    float depth;
    vec2 d = effFlow(c, depth) * u_cell_k;
    float ln = length(d);
    return ln > CFL_MAX_CELLS ? d * (CFL_MAX_CELLS / max(ln, 1e-30)) : d;
}

float F(ivec2 c) { return texelFetch(u_foam, c, 0).r; }

float tap(ivec2 c, float own) { return blocked(c) ? own : F(c); }

// Bilinear read of the foam at cell coordinates pos (centres at integers),
// blocked taps reading own; lo / hi the four taps' range.
float traced(vec2 pos, float own, out float lo, out float hi) {
    vec2 b = floor(pos);
    vec2 a = pos - b;
    ivec2 i = ivec2(b);
    float v00 = tap(i, own), v10 = tap(i + ivec2(1, 0), own);
    float v01 = tap(i + ivec2(0, 1), own), v11 = tap(i + ivec2(1, 1), own);
    lo = min(min(v00, v10), min(v01, v11));
    hi = max(max(v00, v10), max(v01, v11));
    return (v00 * (1.0 - a.x) + v10 * a.x) * (1.0 - a.y)
         + (v01 * (1.0 - a.x) + v11 * a.x) * a.y;
}

// The forward value at cell y (its own trace).
float fwdAt(ivec2 y) {
    float lo, hi;
    return traced(vec2(y) - disp(y), F(y), lo, hi);
}

float backTap(ivec2 y, float own) { return blocked(y) ? own : fwdAt(y); }

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    if (blocked(c)) {
        finalColor = vec4(0.0);
        return;
    }
    float Fc = F(c);
    vec2 d = disp(c);
    float lo, hi;
    float fwd = traced(vec2(c) - d, Fc, lo, hi);
    vec2 pb = vec2(c) + d;
    vec2 bb = floor(pb);
    vec2 ab = pb - bb;
    ivec2 j = ivec2(bb);
    float b00 = backTap(j, fwd), b10 = backTap(j + ivec2(1, 0), fwd);
    float b01 = backTap(j + ivec2(0, 1), fwd);
    float b11 = backTap(j + ivec2(1, 1), fwd);
    float back = (b00 * (1.0 - ab.x) + b10 * ab.x) * (1.0 - ab.y)
               + (b01 * (1.0 - ab.x) + b11 * ab.x) * ab.y;
    float mc = fwd + 0.5 * (Fc - back);
    float A = (mc < lo || mc > hi) ? fwd : mc;

    // 2. the sources (the twin's sources)
    vec2 p = cellPos(c);
    vec4 W = tileBilinear(u_water, p);
    vec4 Fl = tileBilinear(u_flow, p);
    float depth = W.b;
    float e = clamp(depth / max(u_src_b.w, 1e-12), 0.0, 1.0);
    vec2 v = Fl.xy * (e * e * (3.0 - 2.0 * e));
    float speed = length(v);
    float fr = speed / sqrt(G_MPS2 * max(depth, u_src_b.w));
    float s_fr = u_src_a.x * clamp(fr - u_src_a.y, 0.0, 1.0);
    float ratio = max(speed / u_src_b.y, W.a / u_src_b.z);
    float steep = clamp(ratio - 1.0, 0.0, 1.0);
    float s_steep = u_src_a.z * steep * steep;
    float rise = max(depth - Fl.a, 0.0);
    float frac = clamp(rise / max(depth, 1e-6), 0.0, 1.0);
    float s_edge = u_src_a.w * frac * frac;
    float sv = clamp(speed / u_src_b.y, 0.0, 1.0);
    float s_wall = u_src_b.x * Fl.b * sv * sv;
    float s_splat = u_splat_n > 0 ? splatRate(p) : 0.0;
    A += (s_fr + s_steep + s_edge + s_wall + s_splat) * u_dt;
    // 3. decay, 4. clamp
    A = clamp(A * u_decay_f, 0.0, 1.0);
    finalColor = vec4(A, 0.0, 0.0, 0.0);
}
