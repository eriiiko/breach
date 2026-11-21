// ==========================================================================
// #29 P8.6b (part 1): THE ORDER on the card (see cuda_obs.h), Class T. The
// twin of obs_spec_v1's order block of build_vector (cols 84-95) and the far
// window's route / place planes of build_windows (spec 1.7, 3.8), every env on
// its decision tick with its ended latch clear, on THE tick stream, no sync.
// The host builders are the reference; the list's order columns (47-51) are
// the list's own (cuda_obs_list.cu's order_tail, called after mate_entry).
//
// WHAT IS READ: the soldier's order is the store's four order columns (the
// conductor's ORDER table writes them at slot 1b', its expiry clears them at
// 10'), the place is THE ROOM GRAPH (device) -- the place plane (obs_spec_v1.
// place_plane), the room grid, the openings, staged per env at the build and
// at a reset -- and the place table (the centres, P8.8's).
//
// THE ROUTE (obs_spec_v1.route), k_route, ONE BLOCK PER ENV:
//   the tiles     horde.walkable_tiles (the mobility table over the resident
//                 material plane) OR the rectangle of every opening
//                 room_graph.opening_states calls openable NOW: not every tile
//                 walkable, and every unwalkable tile in the span of a living
//                 CLOSED door that is not wired and not a hull door (the door
//                 tables, the wired flags, obs_rows::door_hull);
//   the keys      the distinct (place, footprint) of the env's valid rows
//                 holding an order, in row order (thread 0);
//   per key       THE flood (horde_dev::flood_block, the one device flood of
//                 horde.flood) seeded at every passable anchor whose block
//                 centre lies in the place (the place plane at anchor + fp // 2,
//                 NO_PLACE past the map's last fp // 2 rows / columns), then its
//                 far block minima (blocks(field, UNREACHED).min) and, per row
//                 of the key, the field at the soldier's anchor (Unit.tile_y /
//                 tile_x: horde.field_at) and the compass's step
//                 (horde.candidates: the in-map 4-neighbours strictly below the
//                 cell, lowest first, ties in the cell's parity order
//                 horde_dev::DESC_DY / DESC_DX).
// The flood's distances are the BFS's, which no visiting order changes, so the
// result equals the host's whatever the frontier's order. RECOMPUTED every
// decision, never cached: the host's cache (keyed on the tile serial and the
// openings' states) holds exactly this value.
//
// THE ARITHMETIC: integers where the reference is integer; the place
// direction / distance and route lengths float64 through the kit (unfused),
// lengths as sqrt(dx*dx + dy*dy), d_c's log1p and the squash's log CUDA's;
// one cast per value (float32 for the vector, the hand double -> float16 for
// the route plane).
// ==========================================================================
#include "cuda_obs.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "cuda_batch.cuh"          // tstream(): THE tick stream (blocking)
#include "cuda_obs_float.cuh"
#include "cuda_obs_rows.cuh"       // env_due, row_unit, door_hull
#include "cuda_obs_sight.cuh"      // tile_of, centre_x / _y, floordiv
#include "cuda_unit_events.h"      // UnitRecordList (the door tables)
#include "cuda_unit_horde.cuh"     // horde_dev::flood_block, Shared, DESC_DY / DESC_DX
#include "cuda_unit_lib.cuh"       // MobilityMap, mobility_at
#include "cuda_unit_store.h"       // UnitTablesDevice
#include "obs_layout.h"
#include "object_tables_layout.h"
#include "unit_draws_layout.h"
#include "unit_store_layout.h"
#include "unit_tables_layout.h"

namespace breach_cuda {

using namespace obs_layout;

static_assert(OBS_ROUTE_UNREACHED == horde_dev::UNREACHED, "horde.UNREACHED is horde_dev's");

namespace {

using namespace obs_f64;
using namespace unit_store;
using obs_rows::env_due;
using obs_rows::row_unit;
using obs_sight::floordiv;
using obs_sight::tile_of;

constexpr int CELLS = OBS_WIN * OBS_WIN;

// The route's seeds (obs_spec_v1.route): every passable anchor whose block
// centre (anchor + fp // 2) lies in the place `where` on the place plane; the
// anchors past the map's last half rows / columns read NO_PLACE. flood_block
// reads `pass` after it built it and skips a seed whose y is -1.
struct RouteSeeds {
    const uint8_t* pass;
    const int32_t* place;   // (h, w) of this env
    int h, w, half;
    int64_t where;
    __device__ int count() const { return h * w; }
    __device__ bool seed(int s) const {
        const int y = s / w, x = s - (s / w) * w;
        if (!pass[s] || y >= h - half || x >= w - half) return false;
        return (int64_t)place[(size_t)(y + half) * w + (x + half)] == where;
    }
    __device__ int64_t y(int s) const { return seed(s) ? (int64_t)(s / w) : -1; }
    __device__ int64_t x(int s) const { return (int64_t)(s - (s / w) * w); }
};

// ONE block per env (HORDE_BT threads): the tiles, the keys, per key the flood,
// its block minima and its rows' answers.
__global__ void k_route(const int64_t* store, int U, const int64_t* tables, const int32_t* rowmap,
                        int n_rows, int n_slots, const int64_t* trow, const int64_t* params,
                        const uint8_t* ended, const int8_t* material, const bool* vac,
                        const bool* amb, int h, int w, const int64_t* doors,
                        const int64_t* door_count, int door_cap, const int64_t* door_tiles,
                        int tile_cap, const uint8_t* wired, int wired_cap,
                        const int32_t* has_graph, const int32_t* graph_place,
                        const int64_t* openings, int open_cap, const int32_t* open_count,
                        uint8_t* open_prev, uint8_t* route_ok, uint8_t* route_pass,
                        int32_t* route_dist, int32_t* route_q, uint8_t* route_door,
                        int64_t* route_keys, int32_t* route_row, int32_t* route_blocks, int hb,
                        int wb, int32_t* faults, int brk) {
    using namespace object_tables;
    __shared__ horde_dev::Shared S;
    __shared__ int n_keys;
    __shared__ int bad;
    const int e = blockIdx.y;
    if (!env_due(e, trow, params, ended)) return;      // uniform over the block
    const size_t n = (size_t)h * w;
    uint8_t* ok = route_ok + e * n;
    uint8_t* pass = route_pass + e * n;
    int32_t* dist = route_dist + e * n;
    int32_t* q0 = route_q + 2 * e * n;
    int32_t* q1 = q0 + n;
    const int8_t* mat = material + e * n;
    const bool* vac_e = vac + e * n;
    const bool* amb_e = amb + e * n;
    const int32_t* place = graph_place + e * n;
    const int64_t* tab = tables + (size_t)e * unit_tables::UNIT_TABLES_W;
    const unit_lib::MobilityMap mm{mat, h, w, tab + unit_tables::UNIT_TAB_MATERIAL_MOBILITY,
                         unit_tables::UNIT_TAB_MATERIAL_MOBILITY_W};
    const int64_t* drows = doors + (size_t)e * door_cap * OBJ_DOOR_W;
    const int64_t* dtiles = door_tiles + (size_t)e * tile_cap * OBJ_DOOR_TILE_W;
    const int n_doors = (int)(door_count[e] < door_cap ? door_count[e] : door_cap);
    uint8_t* delig = route_door + (size_t)e * door_cap;
    int64_t* keys = route_keys + (size_t)e * n_rows * 2;
    // -- the keys (thread 0, row order) and every row's answers reset --------
    if (threadIdx.x == 0) {
        bad = 0;
        int nk = 0;
        for (int r = 0; r < n_rows; ++r) {
            int32_t* rr = route_row + ((size_t)e * n_rows + r) * OBS_ROUTE_ROW_W;
            rr[OBS_ROUTE_ROW_SLOT] = -1;
            rr[OBS_ROUTE_ROW_HERE] = OBS_ROUTE_UNREACHED;
            rr[OBS_ROUTE_ROW_CDX] = 0;
            rr[OBS_ROUTE_ROW_CDY] = 0;
            const int64_t* u = row_unit(store, U, rowmap, e, n_rows, r, n_slots);
            if (!u || !has_graph[e] || u[UNIT_COL_ORDER_WHAT] == OBS_ORDER_NONE) continue;
            const int64_t where = u[UNIT_COL_ORDER_WHERE], fp = u[UNIT_COL_FOOTPRINT];
            int k = 0;
            while (k < nk && !(keys[2 * k] == where && keys[2 * k + 1] == fp)) ++k;
            if (k == nk) {
                keys[2 * k] = where;
                keys[2 * k + 1] = fp;
                ++nk;
            }
            rr[OBS_ROUTE_ROW_SLOT] = k;
        }
        n_keys = nk;
    }
    // -- the walkable tiles (horde.walkable_tiles) ------------------------------
    for (int i = threadIdx.x; i < (int)n; i += blockDim.x) {
        bool b = false;
        ok[i] = unit_lib::mobility_at(mm, i / w, i % w, &b) > 0 ? 1 : 0;
        if (b) atomicOr(&bad, 1);
    }
    // -- the doors a unit could open (opening_states' door_tile set): living,
    // CLOSED, not wired, not a hull door ----------------------------------------
    for (int k = threadIdx.x; k < n_doors; k += blockDim.x) {
        const int64_t* row = drows + (size_t)k * OBJ_DOOR_W;
        const int64_t a = row[OBJ_DOOR_COL_TILE_START], c = row[OBJ_DOOR_COL_TILE_COUNT];
        uint8_t el = 0;
        if (row[OBJ_DOOR_COL_ALIVE] && row[OBJ_DOOR_COL_STATE] == OBS_DOOR_STATE_CLOSED &&
            !(k < wired_cap && wired[k])) {
            if (a < 0 || c < 0 || a + c > tile_cap) {
                atomicOr(&bad, 1);
            } else if (!obs_rows::door_hull(dtiles, a, c, vac_e, amb_e, h, w)) {
                el = 1;
            }
        }
        delig[k] = el;
    }
    __syncthreads();
    // -- the openings (room_graph.opening_states): openable ones open their
    // rectangle for the route (ord_prev: the flags of the env's previous build)
    const int n_open = open_count[e];
    for (int o = threadIdx.x; o < n_open; o += blockDim.x) {
        const int64_t* op = openings + ((size_t)e * open_cap + o) * OBS_OPEN_W;
        const int64_t ox = op[OBS_OPEN_X], oy = op[OBS_OPEN_Y];
        const int64_t ow = op[OBS_OPEN_WIDTH], oh = op[OBS_OPEN_HEIGHT];
        bool passable = true, openable = true, b = false;
        for (int64_t y = oy; y < oy + oh; ++y)
            for (int64_t x = ox; x < ox + ow; ++x) {
                if (unit_lib::mobility_at(mm, y, x, &b) > 0) continue;
                passable = false;
                bool in_door = false;
                for (int k = 0; k < n_doors && !in_door; ++k) {
                    if (!delig[k]) continue;
                    const int64_t* row = drows + (size_t)k * OBJ_DOOR_W;
                    const int64_t a = row[OBJ_DOOR_COL_TILE_START];
                    const int64_t c = row[OBJ_DOOR_COL_TILE_COUNT];
                    for (int64_t i = a; i < a + c && !in_door; ++i)
                        in_door = dtiles[i * OBJ_DOOR_TILE_W + OBJ_DOOR_TILE_COL_TY] == y &&
                                  dtiles[i * OBJ_DOOR_TILE_W + OBJ_DOOR_TILE_COL_TX] == x;
                }
                if (!in_door) openable = false;
            }
        if (b) atomicOr(&bad, 1);
        openable = openable && !passable;
        uint8_t* prev = open_prev + (size_t)e * open_cap + o;
        const bool use = brk == OBS_ORDER_BREAK_ORD_PREV ? (*prev != 0) : openable;
        *prev = openable ? 1 : 0;
        if (use)
            for (int64_t y = oy; y < oy + oh; ++y)
                for (int64_t x = ox; x < ox + ow; ++x) ok[(size_t)y * w + (size_t)x] = 1;
    }
    __syncthreads();
    // -- per key: the flood, its block minima, its rows -------------------------
    const int nb = hb * wb;
    for (int s = 0; s < n_keys; ++s) {
        const int64_t where = keys[2 * s], fp = keys[2 * s + 1];
        const RouteSeeds sd{pass, place, h, w, (int)floordiv(fp, 2), where};
        horde_dev::flood_block(ok, h, w, (int)fp, dist, pass, q0, q1, sd, S);
        int32_t* blk = route_blocks + ((size_t)e * n_rows + s) * nb;
        for (int c = threadIdx.x; c < nb; c += blockDim.x) {
            const int bi = c / wb, bj = c % wb;
            int32_t m = OBS_ROUTE_UNREACHED;
            for (int y = bi * OBS_BLOCK; y < bi * OBS_BLOCK + OBS_BLOCK && y < h; ++y)
                for (int x = bj * OBS_BLOCK; x < bj * OBS_BLOCK + OBS_BLOCK && x < w; ++x) {
                    const int32_t d = dist[(size_t)y * w + x];
                    m = d < m ? d : m;
                }
            blk[c] = m;
        }
        for (int r = threadIdx.x; r < n_rows; r += blockDim.x) {
            int32_t* rr = route_row + ((size_t)e * n_rows + r) * OBS_ROUTE_ROW_W;
            if (rr[OBS_ROUTE_ROW_SLOT] != s) continue;
            const int64_t* u = row_unit(store, U, rowmap, e, n_rows, r, n_slots);
            const int64_t ay = tile_of(u[UNIT_COL_Y_Q]), ax = tile_of(u[UNIT_COL_X_Q]);
            const int32_t cur = (ay >= 0 && ay < h && ax >= 0 && ax < w)
                                    ? dist[(size_t)ay * w + (size_t)ax]
                                    : OBS_ROUTE_UNREACHED;
            rr[OBS_ROUTE_ROW_HERE] = cur;
            const int par = (int)((ax + ay) & 1);
            int best = -1;
            int32_t bv = 0;
            for (int k = 0; k < 4; ++k) {
                const int64_t ny = ay + horde_dev::DESC_DY[par][k];
                const int64_t nx = ax + horde_dev::DESC_DX[par][k];
                if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                const int32_t v = dist[(size_t)ny * w + (size_t)nx];
                if (v >= cur) continue;
                // (value, order) lowest first; ord_tie: the LAST of equal values
                if (best < 0 || v < bv || (v == bv && brk == OBS_ORDER_BREAK_ORD_TIE)) {
                    best = k;
                    bv = v;
                }
            }
            if (best >= 0) {
                rr[OBS_ROUTE_ROW_CDX] = horde_dev::DESC_DX[par][best];
                rr[OBS_ROUTE_ROW_CDY] = horde_dev::DESC_DY[par][best];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && bad) atomicOr(faults + e, OBS_ORDER_FAULT_RANGE);
}

__device__ __forceinline__ void put_v(float* v, int col, double x) { v[col] = to_f32(x); }

// One thread per (env, row): the vector's order block (build_vector's 1.5
// tail, cols 84-95); a row that is not valid reads 0 (observe_rows zeroes it).
__global__ void k_order_vector(const int64_t* store, int U, const int32_t* rowmap, int n_rows,
                               int n_slots, const int64_t* otab, const int64_t* trow,
                               const int64_t* params, const uint8_t* ended, int h, int w,
                               const int32_t* has_graph, const int32_t* graph_place,
                               const int64_t* places, int place_cap, const int32_t* place_count,
                               const int32_t* route_row, float* vector, int brk) {
    const int e = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows || !env_due(e, trow, params, ended)) return;
    const size_t er = (size_t)e * n_rows + r;
    float* v = vector + er * OBS_V;
    const int cols[] = {OBS_V_ORDER_WHAT, OBS_V_ORDER_HAS_LIMIT, OBS_V_ORDER_LEFT, OBS_V_INSIDE,
                        OBS_V_PLACE_KIND, OBS_V_PLACE_DX, OBS_V_PLACE_DY, OBS_V_PLACE_DIST_C,
                        OBS_V_COMPASS_DX, OBS_V_COMPASS_DY, OBS_V_ROUTE_LEN, OBS_V_HAS_TARGET};
    for (int c : cols) v[c] = 0.0f;
    const int64_t* u = row_unit(store, U, rowmap, e, n_rows, r, n_slots);
    if (!u) return;
    const int64_t* ot = otab + (size_t)e * OBS_TAB_W;
    const double cap = of_bits(OBS_SQUASH_CAP_BITS);
    const int64_t lref = ot[OBS_TAB_L_REF];
    const int64_t tick = trow[(size_t)e * unit_draws::TICK_ROW_W + unit_draws::TICK_ROW_TICK];
    const int64_t what = u[UNIT_COL_ORDER_WHAT];
    put_v(v, OBS_V_ORDER_WHAT, (double)what);
    if (what == OBS_ORDER_NONE) {
        put_v(v, OBS_V_ROUTE_LEN, -1.0);
        return;
    }
    const int64_t cx = obs_sight::centre_x(u), cy = obs_sight::centre_y(u);
    const int64_t where = u[UNIT_COL_ORDER_WHERE], until = u[UNIT_COL_ORDER_UNTIL];
    put_v(v, OBS_V_ORDER_HAS_LIMIT, until != OBS_NO_LIMIT ? 1.0 : 0.0);
    if (until != OBS_NO_LIMIT) put_v(v, OBS_V_ORDER_LEFT, ratio(until - tick, ot[OBS_TAB_LIMIT_TICKS]));
    if (has_graph[e]) {
        // room_graph.place_at at the centre tile: the place plane, NO_PLACE off the map
        const int64_t here_place = (cy >= 0 && cy < h && cx >= 0 && cx < w)
                                       ? (int64_t)graph_place[(size_t)e * h * w + cy * w + cx]
                                       : (int64_t)OBS_NO_PLACE;
        put_v(v, OBS_V_INSIDE, here_place == where ? 1.0 : 0.0);
        put_v(v, OBS_V_PLACE_KIND, where < OBS_ORDER_OPENING_BASE ? 1.0 : 2.0);
        // _place_centre: the place table's first row of the code (ord_place: the next row)
        const int64_t* pl = places + (size_t)e * place_cap * OBS_PLACE_W;
        const int np = place_count[e];
        int p = 0;
        while (p < np && pl[(size_t)p * OBS_PLACE_W + OBS_PLACE_WHERE] != where) ++p;
        if (p < np && brk == OBS_ORDER_BREAK_ORD_PLACE) p = (p + 1) % np;
        if (p < np) {
            const int64_t pdx = pl[(size_t)p * OBS_PLACE_W + OBS_PLACE_X] - cx;
            const int64_t pdy = pl[(size_t)p * OBS_PLACE_W + OBS_PLACE_Y] - cy;
            put_v(v, OBS_V_PLACE_DX, squash(ratio(pdx, lref), cap));
            put_v(v, OBS_V_PLACE_DY, squash(ratio(pdy, lref), cap));
            const double fx = (double)pdx, fy = (double)pdy;
            const double d = __dsqrt_rn(add(mul(fx, fx), mul(fy, fy)));   // obs_spec_v1.length
            put_v(v, OBS_V_PLACE_DIST_C,
                  squash(div(log1p(d > 0.0 ? d : 0.0), log1p((double)lref)), cap));   // d_c
        }
        const int32_t* rr = route_row + er * OBS_ROUTE_ROW_W;
        put_v(v, OBS_V_COMPASS_DX, (double)rr[OBS_ROUTE_ROW_CDX]);
        put_v(v, OBS_V_COMPASS_DY, (double)rr[OBS_ROUTE_ROW_CDY]);
        const int32_t here = rr[OBS_ROUTE_ROW_HERE];
        put_v(v, OBS_V_ROUTE_LEN, here == OBS_ROUTE_UNREACHED ? -1.0 : squash(ratio(here, lref), cap));
    } else {
        put_v(v, OBS_V_ROUTE_LEN, -1.0);
    }
    put_v(v, OBS_V_HAS_TARGET, u[UNIT_COL_ORDER_TARGET] != OBS_NO_TARGET ? 1.0 : 0.0);
}

// One thread per (env, row, cell): the far window's route plane (the flood's
// block minimum over L_ref, squashed; -1 unreached, off the block grid and
// without an order) and place plane (1 on a block holding a tile of the place:
// a room's grid tiles, an opening's rectangle); a row that is not valid reads 0.
__global__ void k_order_planes(const int64_t* store, int U, const int32_t* rowmap, int n_rows,
                               int n_slots, const int64_t* otab, const int64_t* trow,
                               const int64_t* params, const uint8_t* ended, int h, int w,
                               const int32_t* has_graph, const int32_t* graph_grid,
                               const int64_t* openings, int open_cap, const int32_t* open_count,
                               const int32_t* route_row, const int32_t* route_blocks, int hb,
                               int wb, uint16_t* far_f, uint8_t* far_u) {
    const int e = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * CELLS || !env_due(e, trow, params, ended)) return;
    const int r = idx / CELLS, cell = idx % CELLS;
    const int i = cell / OBS_WIN, j = cell % OBS_WIN;
    const size_t er = (size_t)e * n_rows + r;
    uint16_t* rp = far_f + (er * OBS_FAR_F_PLANES + OBS_FAR_F_ROUTE) * CELLS + cell;
    uint8_t* pp = far_u + (er * OBS_FAR_U_PLANES + OBS_FAR_U_PLACE) * CELLS + cell;
    const int64_t* u = row_unit(store, U, rowmap, e, n_rows, r, n_slots);
    if (!u) {
        *rp = 0;
        *pp = 0;
        return;
    }
    *rp = (uint16_t)OBS_ROUTE_OUTSIDE_HALF;
    *pp = 0;
    if (u[UNIT_COL_ORDER_WHAT] == OBS_ORDER_NONE || !has_graph[e]) return;
    const int64_t cx = obs_sight::centre_x(u), cy = obs_sight::centre_y(u);
    const int64_t bi = floordiv(cy, OBS_BLOCK) - OBS_HALF + i;
    const int64_t bj = floordiv(cx, OBS_BLOCK) - OBS_HALF + j;
    if (bi < 0 || bi >= hb || bj < 0 || bj >= wb) return;
    const int64_t* ot = otab + (size_t)e * OBS_TAB_W;
    const int slot = route_row[er * OBS_ROUTE_ROW_W + OBS_ROUTE_ROW_SLOT];
    const int32_t fb = route_blocks[(((size_t)e * n_rows + slot) * hb + bi) * wb + bj];
    if (fb != OBS_ROUTE_UNREACHED)
        *rp = to_half_bits(squash(ratio(fb, ot[OBS_TAB_L_REF]), of_bits(OBS_SQUASH_CAP_BITS)));
    // place_blocks: the block's in-map tiles
    const int64_t where = u[UNIT_COL_ORDER_WHERE];
    const int64_t y0 = bi * OBS_BLOCK, x0 = bj * OBS_BLOCK;
    const int64_t y1 = y0 + OBS_BLOCK < h ? y0 + OBS_BLOCK : h;
    const int64_t x1 = x0 + OBS_BLOCK < w ? x0 + OBS_BLOCK : w;
    bool hit = false;
    if (where < OBS_ORDER_OPENING_BASE) {
        const int32_t* g = graph_grid + (size_t)e * h * w;
        for (int64_t y = y0; y < y1 && !hit; ++y)
            for (int64_t x = x0; x < x1 && !hit; ++x) hit = (int64_t)g[y * w + x] == where;
    } else {
        const int no = open_count[e];
        for (int o = 0; o < no && !hit; ++o) {
            const int64_t* op = openings + ((size_t)e * open_cap + o) * OBS_OPEN_W;
            if (OBS_ORDER_OPENING_BASE + op[OBS_OPEN_ID] != where) continue;
            const int64_t ox0 = op[OBS_OPEN_X], oy0 = op[OBS_OPEN_Y];
            const int64_t ox1 = ox0 + op[OBS_OPEN_WIDTH], oy1 = oy0 + op[OBS_OPEN_HEIGHT];
            hit = ox0 < x1 && x0 < ox1 && oy0 < y1 && y0 < oy1;
        }
    }
    *pp = hit ? 1 : 0;
}

void need(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("obs_orders: ") + what);
}

}  // namespace

void obs_orders(int n_env, std::uintptr_t store, int n_units, int store_w,
                const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
                int n_teams, std::uintptr_t obs_table, int table_w, std::uintptr_t tick_row,
                std::uintptr_t params, std::uintptr_t ended, std::uintptr_t material,
                std::uintptr_t is_vacuum, std::uintptr_t is_ambient, int h, int w,
                UnitRecordList& doors, UnitRecordList& door_tiles, std::uintptr_t wired,
                int wired_cap, std::uintptr_t has_graph, std::uintptr_t graph_place,
                std::uintptr_t graph_grid, std::uintptr_t openings, int open_cap, int open_w,
                std::uintptr_t open_count, std::uintptr_t open_prev, std::uintptr_t places,
                int place_cap, int place_w, std::uintptr_t place_count, std::uintptr_t route_ok,
                std::uintptr_t route_pass, std::uintptr_t route_dist, std::uintptr_t route_q,
                std::uintptr_t route_door, std::uintptr_t route_keys, std::uintptr_t route_row,
                std::uintptr_t route_blocks, int hb, int wb, std::uintptr_t vector,
                std::uintptr_t far_f, std::uintptr_t far_u, std::uintptr_t faults, int brk) {
    need(n_env >= 1 && n_units >= 1 && n_rows >= 1 && n_slots >= 1 && n_teams >= 1 && h >= 1 &&
         w >= 1 && wired_cap >= 1 && open_cap >= 1 && place_cap >= 1, "a non-positive size");
    need(n_rows == n_teams * n_slots, "n_rows is not n_teams x n_slots");
    need(store_w == UNIT_STORE_W, "the store's width is not UNIT_STORE_W");
    need(table_w == OBS_TAB_W, "the obs table's width is not OBS_TAB_W (regenerate obs_layout.h)");
    need(open_w == OBS_OPEN_W, "the openings' width is not OBS_OPEN_W");
    need(place_w == OBS_PLACE_W, "the place table's width is not OBS_PLACE_W");
    need(tables.n_env() == n_env && tables.width() == unit_tables::UNIT_TABLES_W,
         "the device tables' shape");
    need(doors.n_env() == n_env && door_tiles.n_env() == n_env, "the door tables' env count");
    need(doors.width() == object_tables::OBJ_DOOR_W &&
         door_tiles.width() == object_tables::OBJ_DOOR_TILE_W, "the door tables' widths");
    need(hb == (h + OBS_BLOCK - 1) / OBS_BLOCK && wb == (w + OBS_BLOCK - 1) / OBS_BLOCK,
         "the block grid's shape");
    need(store && rowmap && obs_table && tick_row && params && ended && material && is_vacuum &&
         is_ambient && wired && has_graph && graph_place && graph_grid && openings &&
         open_count && open_prev && places && place_count && route_ok && route_pass &&
         route_dist && route_q && route_door && route_keys && route_row && route_blocks &&
         vector && far_f && far_u && faults, "a null device pointer");
    cudaStream_t s = tstream();
    const int64_t* st = reinterpret_cast<const int64_t*>(store);
    const int32_t* rm = reinterpret_cast<const int32_t*>(rowmap);
    const int64_t* ot = reinterpret_cast<const int64_t*>(obs_table);
    const int64_t* trow = reinterpret_cast<const int64_t*>(tick_row);
    const int64_t* prm = reinterpret_cast<const int64_t*>(params);
    const uint8_t* end = reinterpret_cast<const uint8_t*>(ended);
    const int32_t* hg = reinterpret_cast<const int32_t*>(has_graph);
    const int64_t* ops = reinterpret_cast<const int64_t*>(openings);
    const int32_t* oc = reinterpret_cast<const int32_t*>(open_count);
    int32_t* rrow = reinterpret_cast<int32_t*>(route_row);
    int32_t* rblk = reinterpret_cast<int32_t*>(route_blocks);
    k_route<<<dim3(1, n_env), horde_dev::HORDE_BT, 0, s>>>(
        st, n_units, tables.values(), rm, n_rows, n_slots, trow, prm, end,
        reinterpret_cast<const int8_t*>(material), reinterpret_cast<const bool*>(is_vacuum),
        reinterpret_cast<const bool*>(is_ambient), h, w, doors.records_dev(), doors.count_dev(),
        doors.cap(), door_tiles.records_dev(), door_tiles.cap(),
        reinterpret_cast<const uint8_t*>(wired), wired_cap, hg,
        reinterpret_cast<const int32_t*>(graph_place), ops, open_cap, oc,
        reinterpret_cast<uint8_t*>(open_prev), reinterpret_cast<uint8_t*>(route_ok),
        reinterpret_cast<uint8_t*>(route_pass), reinterpret_cast<int32_t*>(route_dist),
        reinterpret_cast<int32_t*>(route_q), reinterpret_cast<uint8_t*>(route_door),
        reinterpret_cast<int64_t*>(route_keys), rrow, rblk, hb, wb,
        reinterpret_cast<int32_t*>(faults), brk);
    k_order_vector<<<dim3((n_rows + 63) / 64, n_env), 64, 0, s>>>(
        st, n_units, rm, n_rows, n_slots, ot, trow, prm, end, h, w, hg,
        reinterpret_cast<const int32_t*>(graph_place), reinterpret_cast<const int64_t*>(places),
        place_cap, reinterpret_cast<const int32_t*>(place_count), rrow,
        reinterpret_cast<float*>(vector), brk);
    const int cells = n_rows * CELLS;
    k_order_planes<<<dim3((cells + 255) / 256, n_env), 256, 0, s>>>(
        st, n_units, rm, n_rows, n_slots, ot, trow, prm, end, h, w, hg,
        reinterpret_cast<const int32_t*>(graph_grid), ops, open_cap, oc, rrow, rblk, hb, wb,
        reinterpret_cast<uint16_t*>(far_f), reinterpret_cast<uint8_t*>(far_u));
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("obs_orders: launch failed: ") +
                                 cudaGetErrorString(err));
}

}  // namespace breach_cuda
