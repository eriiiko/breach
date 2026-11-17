#pragma once
// ============================================================================
// #29 P7c.2 -- THE BLAST'S WALK, ITS AIR READ AND ITS SOURCE on the device
// (residency design 4.4c, the cut table's P7c.2 row): device functions of the
// per-env serial applier's PAYLOAD function (cuda_unit_payload.cuh) and of its
// SOURCES stage, integer for integer:
//
//   reach        blast_expansion.reach_tiles (walk_seeds + _walk): the open
//                tiles (open_mask: never solid, vacuum or the ambient ring)
//                with squared distance from the centre under r2, walked
//                4-neighbour from the seeds, a seed always in, ROW-MAJOR. The
//                set is a connected component, so the visiting order cannot
//                change it: the walk marks, then the box around the centre is
//                read row-major (the reference's sorted()) and the marks
//                cleared.
//   air read     blast_air.read_blast_air over a tile list (the walked tiles
//                or disc_tiles' disc): per tile g (the pressure ramp) and b
//                (the O2 mole fraction's ramp), its smoke factor g (1 +
//                starved (1 - b)) and its burn g b; the heat factor s + (1 -
//                s) mean(burn), the mean floored; `all_one` when every smoke
//                factor is exactly ONE (the reference's smoke plane is None).
//                The smoke factor of a tile is a pure function of the planes
//                as they stand at the read (air_smoke_q); nothing between the
//                read and the smoke edit writes a plane it reads.
//   add_source   blast_expansion.add_source: s_q added to the pending plane
//                on each tile in order, saturating at S_MAX_Q, each cut
//                counted (div_source_sat_hits, div_source_cut_q).
//   take_sources the SOURCES stage: blast_expansion.take_source_plane -- the
//                pending plane handed out (`out`, the taken flag) and zeroed,
//                every row of the blast-source table added in row order and
//                its ticks counted down, then the dead dropped by the SPAN
//                COMPACTION (the table's stable compaction with its side
//                table: a survivor's tiles moved down to the running cursor,
//                its start rewritten), the reference's filter.
//
// THE ORACLE is src/simulation/blast_expansion.py and blast_air.py: an
// arithmetic change lands there first, then here, IN THE SAME PATCH; the
// payloads gate (tests/cuda_unit_payloads_check.py, parts WALK, SOURCES,
// HANDOVER) is the guard (the corpus detonates one frag in 500 ticks).
//
// Device-only header; CLASS T: nothing in the played tick includes it.
#include <cstdint>

#include "applier_layout.h"
#include "cuda_topology.cuh"
#include "cuda_unit_lib.cuh"
#include "object_tables_layout.h"
#include "unit_tables_layout.h"

namespace breach_cuda {
namespace unit_blast {

using namespace unit_tables;
using namespace object_tables;

constexpr int64_t ONE = 65536;

// The walk's per-env scratch: a mark per cell (0 between walks), the BFS
// queue, and the two walked lists of one PAYLOAD (the heat walk, the source
// walk), each of n cells.
struct Scratch {
    uint8_t* mark;
    int32_t* queue;
    int32_t* heat;
    int32_t* src;
};

// open_mask: the tiles the pressure solve acts on.
__device__ __forceinline__ bool open_at(const topo::TopoEnv& E, int i) {
    return !(E.solid[i] || E.is_vacuum[i] || E.is_ambient[i]);
}

__device__ __forceinline__ bool in_grid(const topo::TopoEnv& E, int64_t y, int64_t x) {
    return 0 <= y && y < E.h && 0 <= x && x < E.w;
}

// walk_seeds -> up to 4 cells into `seeds`; returns their count.
__device__ inline int walk_seeds(const topo::TopoEnv& E, int64_t fy, int64_t fx,
                                 bool has_appr, int64_t ay, int64_t ax, int32_t* seeds) {
    if (in_grid(E, fy, fx) && open_at(E, (int)(fy * E.w + fx))) {
        seeds[0] = (int32_t)(fy * E.w + fx);
        return 1;
    }
    // the 4-neighbours in sorted (y, x) order
    const int64_t ny[4] = {fy - 1, fy, fy, fy + 1};
    const int64_t nx[4] = {fx, fx - 1, fx + 1, fx};
    int32_t near[4];
    int64_t nyy[4], nxx[4];
    int nn = 0;
    for (int k = 0; k < 4; ++k)
        if (in_grid(E, ny[k], nx[k]) && open_at(E, (int)(ny[k] * E.w + nx[k]))) {
            near[nn] = (int32_t)(ny[k] * E.w + nx[k]);
            nyy[nn] = ny[k];
            nxx[nn] = nx[k];
            ++nn;
        }
    if (!has_appr || nn == 0) {
        for (int k = 0; k < nn; ++k) seeds[k] = near[k];
        return nn;
    }
    const int64_t dy = ay - fy < 0 ? fy - ay : ay - fy;
    const int64_t dx = ax - fx < 0 ? fx - ax : ax - fx;
    if ((dy > dx ? dy : dx) == 1 && in_grid(E, ay, ax) && open_at(E, (int)(ay * E.w + ax))) {
        seeds[0] = (int32_t)(ay * E.w + ax);
        return 1;
    }
    int best = 0;
    int64_t bd = -1;
    for (int k = 0; k < nn; ++k) {         // min(): the first of the least
        const int64_t d = (nyy[k] - ay) * (nyy[k] - ay) + (nxx[k] - ax) * (nxx[k] - ax);
        if (bd < 0 || d < bd) {
            bd = d;
            best = k;
        }
    }
    seeds[0] = near[best];
    return 1;
}

// reach_tiles(open_mask, fy, fx, r2, approach) into `out` (row-major cell
// indices); returns the count. `S.mark` is all-zero on entry and on exit.
__device__ inline int reach(const topo::TopoEnv& E, int64_t fy, int64_t fx, int64_t r2,
                            bool has_appr, int64_t ay, int64_t ax, const Scratch& S,
                            int32_t* out) {
    int32_t seeds[4];
    const int ns = walk_seeds(E, fy, fx, has_appr, ay, ax, seeds);
    int head = 0, tail = 0;
    for (int k = 0; k < ns; ++k) {
        S.mark[seeds[k]] = 1;
        S.queue[tail++] = seeds[k];
    }
    while (head < tail) {
        const int32_t c = S.queue[head++];
        const int64_t y = c / E.w, x = c % E.w;
        const int64_t qy[4] = {y - 1, y + 1, y, y};
        const int64_t qx[4] = {x, x, x - 1, x + 1};
        for (int k = 0; k < 4; ++k) {
            if (!in_grid(E, qy[k], qx[k])) continue;
            const int j = (int)(qy[k] * E.w + qx[k]);
            if (S.mark[j] || !open_at(E, j)) continue;
            if ((qy[k] - fy) * (qy[k] - fy) + (qx[k] - fx) * (qx[k] - fx) >= r2) continue;
            S.mark[j] = 1;
            S.queue[tail++] = j;
        }
    }
    // the reached set lies in the box: a seed touches the centre (Chebyshev
    // 1), any other tile has d2 < r2 <= R^2
    int64_t R = 1;
    while (R * R < r2) ++R;
    int n = 0;
    const int64_t y0 = fy - R > 0 ? fy - R : 0, y1 = fy + R < E.h - 1 ? fy + R : E.h - 1;
    const int64_t x0 = fx - R > 0 ? fx - R : 0, x1 = fx + R < E.w - 1 ? fx + R : E.w - 1;
    for (int64_t y = y0; y <= y1; ++y)
        for (int64_t x = x0; x <= x1; ++x) {
            const int i = (int)(y * E.w + x);
            if (S.mark[i]) {
                out[n++] = i;
                S.mark[i] = 0;
            }
        }
    return n;
}

// blast_air._ramp_q of one value (lo < hi: the table row's proved range).
__device__ __forceinline__ int64_t ramp_q(int64_t x, int64_t lo, int64_t hi) {
    if (x >= hi) return ONE;
    const int64_t d = x - lo > 0 ? x - lo : 0;
    const int64_t r = (d * ONE) / (hi - lo);
    return r < ONE ? r : ONE;
}

// One tile's (smoke factor, burn) of the air read.
__device__ __forceinline__ void air_tile(const topo::TopoEnv& E, const int64_t* tab, int i,
                                         int64_t* smoke, int64_t* burn) {
    const int64_t* s = tab + UNIT_TAB_BLAST_AIR_Q;
    const int64_t o2 = (int64_t)E.g(E.o2, i);
    const int64_t bulk = o2 + (int64_t)E.g(E.n2, i);
    const int64_t x_q =
        bulk > 0 ? fixedpoint::floordiv_q(o2 * ONE, bulk) : 0;   // numpy's floor
    const int64_t p_q = (int64_t)E.atmosphere[i];
    const int64_t b_q = ramp_q(x_q, s[0], s[1]);
    const int64_t g_q = ramp_q(p_q, s[4], s[5]);
    *smoke = (g_q * (ONE + ((s[2] * (ONE - b_q)) >> 16))) >> 16;
    *burn = (g_q * b_q) >> 16;
}

__device__ __forceinline__ int64_t air_smoke_q(const topo::TopoEnv& E, const int64_t* tab,
                                               int i) {
    int64_t sm, bu;
    air_tile(E, tab, i, &sm, &bu);
    return sm;
}

// read_blast_air over `n` tiles: the heat factor; *all_one the smoke plane's
// None. No tile: (ONE, true).
__device__ inline int64_t air_read(const topo::TopoEnv& E, const int64_t* tab,
                                   const int32_t* tiles, int n, bool* all_one) {
    *all_one = true;
    if (n <= 0) return ONE;
    int64_t sum = 0;
    for (int k = 0; k < n; ++k) {
        int64_t sm, bu;
        air_tile(E, tab, tiles[k], &sm, &bu);
        sum += bu;
        if (sm != ONE) *all_one = false;
    }
    const int64_t det = tab[UNIT_TAB_BLAST_AIR_Q + 3];
    const int64_t mean = sum / n;                       // burn >= 0: // == /
    return det + (((ONE - det) * mean) >> 16);
}

// blast_air.disc_tiles(gmap, fy, fx, radius): the open (non-solid) tiles of
// the disc d2 <= r^2, row-major, into `out`; returns the count.
__device__ inline int disc_tiles(const topo::TopoEnv& E, int64_t fy, int64_t fx, int64_t r,
                                 int32_t* out) {
    int n = 0;
    const int64_t y0 = fy - r > 0 ? fy - r : 0, y1 = fy + r + 1 < E.h ? fy + r + 1 : E.h;
    const int64_t x0 = fx - r > 0 ? fx - r : 0, x1 = fx + r + 1 < E.w ? fx + r + 1 : E.w;
    for (int64_t y = y0; y < y1; ++y)
        for (int64_t x = x0; x < x1; ++x) {
            if ((y - fy) * (y - fy) + (x - fx) * (x - fx) > r * r) continue;
            const int i = (int)(y * E.w + x);
            if (!E.solid[i]) out[n++] = i;
        }
    return n;
}

// add_source on `n` cells (blast_expansion.add_source): the pending plane
// plus s_q, saturating at S_MAX_Q, each cut counted.
__device__ inline void add_source(const topo::TopoEnv& E, const int64_t* tab, const int32_t* tiles,
                                  int n, int64_t s_q, int64_t* B) {
    const int64_t smax = tab[UNIT_TAB_BLAST_S_MAX_Q];
    for (int k = 0; k < n; ++k) {
        int64_t v = (int64_t)E.div_source[tiles[k]] + s_q;
        if (v > smax) {
            B[applier::APPLIER_BOOK_DIV_SOURCE_SAT_HITS] += 1;
            B[applier::APPLIER_BOOK_DIV_SOURCE_CUT_Q] += v - smax;
            v = smax;
        }
        E.div_source[tiles[k]] = (int32_t)v;
    }
}

// The SOURCES stage of one env (see the header): `out` the env's (h * w)
// hand-out plane (nullptr: not handed out), the blast-source table (rows,
// count) and its tile side table. False on a malformed table (a FAULT,
// counted by the caller; nothing written).
__device__ inline bool take_sources(const topo::TopoEnv& E, const int64_t* tab, int32_t* out,
                                    int64_t* rows, int64_t* n_rows, int64_t* tiles,
                                    int64_t* n_tiles, int64_t* B) {
    const int64_t nr = *n_rows, nt = *n_tiles;
    // the operands, before any write: every span inside the side table and in
    // order (the span compaction moves tiles only downwards), every tile in
    // the grid
    int64_t run = 0;
    for (int64_t r = 0; r < nr; ++r) {
        const int64_t* row = rows + r * OBJ_SOURCE_W;
        const int64_t a = row[OBJ_SOURCE_COL_TILE_START], c = row[OBJ_SOURCE_COL_TILE_COUNT];
        if (a < run || c < 0 || a + c > nt) return false;
        for (int64_t k = a; k < a + c; ++k) {
            const int64_t* t = tiles + k * OBJ_SOURCE_TILE_W;
            if (!in_grid(E, t[OBJ_SOURCE_TILE_COL_Y], t[OBJ_SOURCE_TILE_COL_X])) return false;
        }
        run = a + c;
    }
    // 1. the pending plane handed out, then zeroed
    bool any = false;
    for (int i = 0; i < E.n; ++i) any = any || E.div_source[i] != 0;
    if (any) {
        B[applier::APPLIER_BOOK_SOURCE_TAKEN] += 1;
        for (int i = 0; i < E.n; ++i) {
            if (out != nullptr) out[i] = E.div_source[i];
            E.div_source[i] = 0;
        }
    } else if (out != nullptr) {
        for (int i = 0; i < E.n; ++i) out[i] = 0;
    }
    // 2. every source added in row order, its ticks counted down; 3. the span
    // compaction (survivors in order, their tiles moved down to the cursor)
    const int64_t smax = tab[UNIT_TAB_BLAST_S_MAX_Q];
    int64_t keep = 0, cursor = 0;
    for (int64_t r = 0; r < nr; ++r) {
        int64_t* row = rows + r * OBJ_SOURCE_W;
        const int64_t a = row[OBJ_SOURCE_COL_TILE_START], c = row[OBJ_SOURCE_COL_TILE_COUNT];
        const int64_t s_q = row[OBJ_SOURCE_COL_S_Q];
        for (int64_t k = a; k < a + c; ++k) {
            const int64_t* t = tiles + k * OBJ_SOURCE_TILE_W;
            const int i = (int)(t[OBJ_SOURCE_TILE_COL_Y] * E.w + t[OBJ_SOURCE_TILE_COL_X]);
            int64_t v = (int64_t)E.div_source[i] + s_q;
            if (v > smax) {
                B[applier::APPLIER_BOOK_DIV_SOURCE_SAT_HITS] += 1;
                B[applier::APPLIER_BOOK_DIV_SOURCE_CUT_Q] += v - smax;
                v = smax;
            }
            E.div_source[i] = (int32_t)v;
        }
        row[OBJ_SOURCE_COL_TICKS] -= 1;
        row[OBJ_SOURCE_COL_LIVE] = row[OBJ_SOURCE_COL_TICKS] > 0 ? 1 : 0;
        if (!row[OBJ_SOURCE_COL_LIVE]) continue;
        for (int64_t k = 0; k < c; ++k)
            for (int q = 0; q < OBJ_SOURCE_TILE_W; ++q)
                tiles[(cursor + k) * OBJ_SOURCE_TILE_W + q] = tiles[(a + k) * OBJ_SOURCE_TILE_W + q];
        row[OBJ_SOURCE_COL_TILE_START] = cursor;
        cursor += c;
        if (keep != r)
            for (int q = 0; q < OBJ_SOURCE_W; ++q) rows[keep * OBJ_SOURCE_W + q] = row[q];
        ++keep;
    }
    for (int64_t r = keep; r < nr; ++r)
        for (int q = 0; q < OBJ_SOURCE_W; ++q) rows[r * OBJ_SOURCE_W + q] = 0;
    for (int64_t k = cursor; k < nt; ++k)
        for (int q = 0; q < OBJ_SOURCE_TILE_W; ++q) tiles[k * OBJ_SOURCE_TILE_W + q] = 0;
    *n_rows = keep;
    *n_tiles = cursor;
    return true;
}

static_assert(OBJ_SOURCE_W == 5 && OBJ_SOURCE_TILE_W == 3,
              "the blast-source tables' columns moved: update the stage");

}  // namespace unit_blast
}  // namespace breach_cuda
