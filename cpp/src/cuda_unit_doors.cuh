#pragma once
// ============================================================================
// #29 P6b.1 -- THE DOOR SWEEP as a device function (residency design 4.4c,
// the cut table's P6b.1 row; decision 1: slot 9e is a STAGE of the one
// applier kernel walking the entity tables in the reference's order, each
// entity applied AT ONCE). src/simulation/door_system.sweep_doors integer for
// integer, on ONE env: its door table (unit_schema.DOOR_TABLE, row = the
// door's place in Simulation._doors, ordinal order) and the door tile side
// table (DOOR_TILE_TABLE: each door's span, row-major, with the tile's hp),
// the env's planes (cuda_topology.cuh) and its unit rows (the occupancy).
//
// Per door, in row order:
//   1. a dead door is skipped (its latch is dead);
//   2. EXTERNAL DESTRUCTION: a CLOSED door any of whose span tiles is no
//      longer MAT_DOOR_CLOSED (a blast, a chew, a burn-through destroyed it)
//      dies whole -- alive 0, state DESTROYED, every hp 0 -- and each
//      REMAINING closed tile is destroyed through topo::destroy_wall in span
//      order (its render-only DoorDestroyedEvent has no device form);
//   3. CLOSED and want_open: the hp copied from wall_hp (the panel remembers
//      its damage), then topo::unseal_tiles; state OPEN;
//   4. OPEN and not want_open: when no LIVING unit's footprint holds a span
//      tile (corpses never block) and topo::can_seal says yes,
//      topo::seal_tiles(span, MAT_DOOR_CLOSED), then each tile's wall_hp
//      restamped from the door's own hp; state CLOSED. Otherwise nothing: the
//      latch is kept, the close retried next tick.
// The door's latch (want_open) is read as the host left it: slot 1b's USE
// writes it (cuda_unit_doors.cu's door_uses), 9e(c)'s signal inputs are
// P6b.3's.
//
// A FAULT (counted in the books' `faults`; the Python face refuses the run):
// a span outside the side table or the grid, an unseal or a seal the Python
// would have raised on (a bug there: rider 3), a destroy_wall fault, a
// footprint outside [1, FOOTPRINT_MAX]. A door that faults is left as it was.
// Every transition (open, close, death) counts in `applied`.
//
// Device-only header; included by cuda_applier.cu only. CLASS T.
#include "cuda_topology.cuh"
#include "cuda_unit_lib.cuh"
#include "object_tables_layout.h"

namespace breach_cuda {
namespace doors {

using namespace object_tables;
using namespace unit_tables;
using namespace applier;

// door_system._occupancy_clear: no LIVING valid row's footprint (the
// fp x fp square at its tile, footprint_rowmajor) holds a span tile. *bad:
// a footprint outside [1, FOOTPRINT_MAX].
__device__ inline bool occupancy_clear(const int64_t* store, int n_units, const int64_t* span,
                                       int stride, int n, bool* bad) {
    for (int r = 0; r < n_units; ++r) {
        const int64_t* row = unit_lib::unit_row(const_cast<int64_t*>(store), n_units, r);
        if (!unit_lib::row_valid(row) || !unit_lib::row_alive(row)) continue;
        const int64_t fp = row[UNIT_COL_FOOTPRINT];
        if (fp < 1 || fp > unit_lib::FOOTPRINT_MAX) {
            *bad = true;
            return false;
        }
        const int64_t ax = unit_lib::tile_of(row[UNIT_COL_X_Q]);
        const int64_t ay = unit_lib::tile_of(row[UNIT_COL_Y_Q]);
        for (int i = 0; i < n; ++i) {
            const int64_t ty = span[(size_t)i * stride], tx = span[(size_t)i * stride + 1];
            if (ax <= tx && tx < ax + fp && ay <= ty && ty < ay + fp) return false;
        }
    }
    return true;
}

// The door sweep over this block env's door table (`drows` n_doors rows of OBJ_DOOR_W)
// and tile table (`trows` n_tiles rows of OBJ_DOOR_TILE_W).
__device__ inline void sweep(const topo::TopoEnv& E, const int64_t* tab, int64_t* drows,
                             int64_t n_doors, int64_t* trows, int64_t n_tiles,
                             const int64_t* store /* (N, U, W): unit_row rebases */, int n_units, int64_t* B) {
    const int64_t CLOSED = tab[UNIT_TAB_DOOR_STATES + 0];
    const int64_t OPEN = tab[UNIT_TAB_DOOR_STATES + 1];
    const int64_t DESTROYED = tab[UNIT_TAB_DOOR_STATES + 2];
    const int64_t mat_closed = tab[UNIT_TAB_DOOR_MAT_CLOSED];
    const int stride = OBJ_DOOR_TILE_W;
    static_assert(OBJ_DOOR_TILE_COL_TX == OBJ_DOOR_TILE_COL_TY + 1,
                  "a span is read as (ty, tx) pairs: the tile table's ty then tx");
    for (int64_t k = 0; k < n_doors; ++k) {
        int64_t* d = drows + (size_t)k * OBJ_DOOR_W;
        if (d[OBJ_DOOR_COL_ALIVE] == 0) continue;
        const int64_t a = d[OBJ_DOOR_COL_TILE_START], n = d[OBJ_DOOR_COL_TILE_COUNT];
        if (a < 0 || n < 0 || a + n > n_tiles || n > topo::SPAN_MAX) {
            B[APPLIER_BOOK_FAULTS] += 1;
            continue;
        }
        int64_t* T = trows + (size_t)a * stride;
        const int64_t* span = T + OBJ_DOOR_TILE_COL_TY;
        bool inside = true;
        for (int64_t i = 0; i < n; ++i)
            if (!topo::in_grid64(E, span[i * stride], span[i * stride + 1])) inside = false;
        if (!inside) {
            B[APPLIER_BOOK_FAULTS] += 1;
            continue;
        }
        const int64_t state = d[OBJ_DOOR_COL_STATE];
        // 1. external destruction (the whole-door rule)
        if (state == CLOSED) {
            bool destroyed_any = false;
            for (int64_t i = 0; i < n; ++i) {
                const int c = (int)span[i * stride] * E.w + (int)span[i * stride + 1];
                if ((int64_t)E.material[c] != mat_closed) destroyed_any = true;
            }
            if (destroyed_any) {
                d[OBJ_DOOR_COL_ALIVE] = 0;
                d[OBJ_DOOR_COL_STATE] = DESTROYED;
                for (int64_t i = 0; i < n; ++i) T[i * stride + OBJ_DOOR_TILE_COL_HP] = 0;
                for (int64_t i = 0; i < n; ++i) {
                    const int fy = (int)span[i * stride], fx = (int)span[i * stride + 1];
                    if ((int64_t)E.material[fy * E.w + fx] == mat_closed)
                        topo::destroy_wall(E, tab, fy, fx, B);   // books its own fault
                }
                B[APPLIER_BOOK_APPLIED] += 1;
                continue;
            }
        }
        // 2. the latch
        if (state == CLOSED && d[OBJ_DOOR_COL_WANT_OPEN] != 0) {
            int32_t hp[topo::SPAN_MAX];
            for (int64_t i = 0; i < n; ++i)
                hp[i] = E.wall_hp[(int)span[i * stride] * E.w + (int)span[i * stride + 1]];
            if (topo::unseal_tiles(E, tab, span, stride, (int)n, B) != 0) {
                B[APPLIER_BOOK_FAULTS] += 1;               // the reference would raise
                continue;
            }
            for (int64_t i = 0; i < n; ++i) T[i * stride + OBJ_DOOR_TILE_COL_HP] = hp[i];
            d[OBJ_DOOR_COL_STATE] = OPEN;
            B[APPLIER_BOOK_APPLIED] += 1;
        } else if (state == OPEN && d[OBJ_DOOR_COL_WANT_OPEN] == 0) {
            bool bad = false;
            const bool clear = occupancy_clear(store, n_units, span, stride, (int)n, &bad);
            if (bad) {
                B[APPLIER_BOOK_FAULTS] += 1;
                continue;
            }
            if (!clear || topo::can_seal(E, tab, span, stride, (int)n) != 0) continue;
            bool hp_ok = true;                            // the restamp's int32 store
            for (int64_t i = 0; i < n; ++i) {
                const int64_t v = T[i * stride + OBJ_DOOR_TILE_COL_HP];
                if (v < -topo::I32_LIMIT || v >= topo::I32_LIMIT) hp_ok = false;
            }
            if (!hp_ok) {
                B[APPLIER_BOOK_FAULTS] += 1;
                continue;
            }
            if (topo::seal_tiles(E, tab, span, stride, (int)n, mat_closed, B) != 0) {
                B[APPLIER_BOOK_FAULTS] += 1;               // can_seal said yes: a fault
                continue;
            }
            for (int64_t i = 0; i < n; ++i)               // the restamp: the door's own hp
                E.wall_hp[(int)span[i * stride] * E.w + (int)span[i * stride + 1]] =
                    (int32_t)T[i * stride + OBJ_DOOR_TILE_COL_HP];
            d[OBJ_DOOR_COL_STATE] = CLOSED;
            B[APPLIER_BOOK_APPLIED] += 1;
        }
    }
}

}  // namespace doors
}  // namespace breach_cuda
