// ============================================================================
// #29 P6.2m -- SLOT 3, MOVEMENT, on the device unit store (the CONTINUOUS
// path's slot: Simulation._update_player_movement). See cuda_unit_kernels.h;
// residency design §4.4b.
//
// Per valid LIVING row, integer for integer the Python law:
//   - a non-zombie holding a live MOVE_DIR (Simulation._step_move_dir): it
//     WALKS while its statuses let it move (composed can_move) and its
//     reaction delay has run (unit_impulse.walk_allowed) -- the [speeds]
//     table's walk and full cost (Q16 ticks per tile) each reduced by
//     movement.default_speed over the footprint's mobility at its
//     (tile_y, tile_x) (the WALK / SPRINT hook inside that one rounding, the
//     effort scale unit_impulse.effort_scale_q16 after it), then the direction
//     rule and the throttle as ONE rational rounded half up once
//     (movement.move_speed_q16, cos theta from movement.facing_cos_q16 against
//     this tick's facing), the walk step each axis truncated
//     (unit_impulse.walk_step_q); then the walk and its pushed velocity move
//     it as ONE displacement (unit_impulse.move); a walker holding no aim makes
//     a non-zero travel direction its facing TARGET (turning.set_target);
//   - any other row with a pushed velocity moves by it alone (unit_impulse.
//     move), a zombie included (Python ignores a zombie's MOVE_DIR).
// unit_impulse.move: k = max(1, ceil(max(|dx|, |dy|) / SUBSTEP_Q)) substeps of
// the truncated part (the remainder on the last), each the slide rule through
// GameMap.is_passable_block (the full step; else x alone -- y is the wall; else
// y alone -- x is the wall; else the move ends); the axis into a wall zeroes
// pv's component there and drops the rest of the move on that axis.
//
// What has NO device form (reference-only, never read by the continuous law):
// the momentum books' WALL_X / WALL_Y entries (unit_impulse.move books the
// zeroed pv there; Unit.impulse_books is REFERENCE_ONLY) and the legacy WEGO
// path step after the loop (Unit.move_path / path_tick_offset, REFERENCE_ONLY;
// on the continuous path move_path is always empty, so the step only bumps
// path_tick_offset of a held non-zombie with no MOVE_DIR). OnePhaseWEGO
// (Simulation._displace_pushed + its timeline) and TwoPhaseWEGO's path branch
// are LEGACY and not ported.
//
// The map: the env's GameMap.material plane (int8, the resident field, (N, h,
// w)) through the device tables' material_mobility row (the materials table's
// own mobility column) -- the projection is_passable_block and
// footprint_mobility both make; no derived plane.
//
// WIDTHS. Every product is proved to fit int64 from the tables' doors
// (unit_tables.ROWS: costs <= 2^26, direction factors <= ONE, mobility
// |m| <= 2^20) and the library's guards (cuda_unit_lib.cuh: a footprint of
// at most 8 x 8 tiles, a cost of at most 2^29 after each reduction step, a hook
// of at most 2^20): the speed's numerator < 2^62. A state operand past its
// bound -- a move component past 2^31, a walk speed past 2^31 Q16 tiles a
// tick, a pushed velocity past 2^40, more than SUBSTEPS_MAX substeps -- is a
// FAULT: the row is left as it was and the driver refuses the result. No game
// reaches one (a walk is under a tile a tick, pv is clamped to 4 tiles).
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_unit_lib.cuh"

namespace breach_cuda {

namespace {

using namespace unit_lib;

//: unit_impulse.SUBSTEP_Q: just under one tile per axis per substep.
constexpr int64_t SUBSTEP_Q = ONE - 1;
//: The substep bound: a displacement of more than 64 tiles a tick faults.
constexpr int64_t SUBSTEPS_MAX = 64;
//: Operand bounds (see the header).
constexpr int64_t MOVE_COMPONENT_MAX = 2147483647LL;
constexpr int64_t WALK_SPEED_MAX = 2147483647LL;
constexpr int64_t PV_OPERAND_MAX = (int64_t)1 << 40;

__device__ __forceinline__ int64_t iabs(int64_t v) { return v < 0 ? -v : v; }

// unit_impulse.move on (x, y, pvx, pvy): the walk (wdx, wdy) plus pv as ONE
// displacement through the slide rule.
__device__ inline void displace(const MobilityMap& m, int64_t fp, int64_t wdx, int64_t wdy,
                                int64_t& x, int64_t& y, int64_t& pvx, int64_t& pvy,
                                bool* bad) {
    if (iabs(pvx) > PV_OPERAND_MAX || iabs(pvy) > PV_OPERAND_MAX) {
        *bad = true;
        return;
    }
    const int64_t dx = wdx + pvx;
    const int64_t dy = wdy + pvy;
    if (dx == 0 && dy == 0) return;
    const int64_t mag = iabs(dx) > iabs(dy) ? iabs(dx) : iabs(dy);
    int64_t k = mag / SUBSTEP_Q + ((mag % SUBSTEP_Q) != 0 ? 1 : 0);
    if (k < 1) k = 1;
    if (k > SUBSTEPS_MAX || fp < 1 || fp > FOOTPRINT_MAX) {
        *bad = true;
        return;
    }
    const int64_t px = dx / k, py = dy / k;          // div_trunc: C's /
    bool x_free = true, y_free = true;
    int64_t cx = x, cy = y;
    for (int64_t i = 0; i < k; ++i) {
        const bool last = (i == k - 1);
        const int64_t ex = x_free ? (last ? dx - px * (k - 1) : px) : 0;
        const int64_t ey = y_free ? (last ? dy - py * (k - 1) : py) : 0;
        if (ex == 0 && ey == 0) continue;
        const int64_t nx = cx + ex, ny = cy + ey;
        // Python's >> floors; nvcc's >> on a signed int64 is arithmetic.
        if (passable_block(m, ny >> 16, nx >> 16, fp, bad)) {
            cx = nx;
            cy = ny;
        } else if (passable_block(m, cy >> 16, nx >> 16, fp, bad)) {
            cx = nx;                                  // slide along x: y is the wall
            if (ey) {
                pvy = 0;
                y_free = false;
            }
        } else if (passable_block(m, ny >> 16, cx >> 16, fp, bad)) {
            cy = ny;                                  // slide along y: x is the wall
            if (ex) {
                pvx = 0;
                x_free = false;
            }
        } else {                                      // boxed in: the move ends
            if (ex) pvx = 0;
            if (ey) pvy = 0;
            break;
        }
        if (*bad) return;
    }
    x = cx;
    y = cy;
}

// Slot 3: one thread per unit row of one env.
__global__ void k_movement(int64_t* store, int n_units, const int64_t* tables,
                           const int8_t* material, int h, int w,
                           unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    if (!row_valid(row) || !row_alive(row)) return;
    const int64_t* tab = env_tables(tables);
    MobilityMap m{bptr(material, (size_t)h * w), h, w, tab + UNIT_TAB_MATERIAL_MOBILITY,
                  UNIT_TAB_MATERIAL_MOBILITY_W};
    bool bad = false;

    int64_t x = row[UNIT_COL_X_Q], y = row[UNIT_COL_Y_Q];
    int64_t pvx = row[UNIT_COL_PVX], pvy = row[UNIT_COL_PVY];
    int64_t target = row[UNIT_COL_FACING_TARGET_Q];
    const int64_t fp = row[UNIT_COL_FOOTPRINT];
    const int64_t* md = row + UNIT_COL_LIVE_MOVE_DIR;   // (present, dx, dy, throttle)

    if (!row[UNIT_COL_IS_ZOMBIE] && md[0]) {
        // ---- Simulation._step_move_dir ---------------------------------------
        const bool walking = composed_flags(row, tab, false, &bad).can_move &&
                             walk_allowed(row);
        int64_t wdx = 0, wdy = 0;
        if (walking && !bad) {
            const int64_t dx_q = md[1], dy_q = md[2];
            if (fp < 1 || fp > FOOTPRINT_MAX || iabs(dx_q) > MOVE_COMPONENT_MAX ||
                iabs(dy_q) > MOVE_COMPONENT_MAX) {
                bad = true;
            }
            int64_t n = 0, total = 0;
            if (!bad) footprint_mobility(m, tile_of(y), tile_of(x), fp, &n, &total, &bad);
            const int64_t es = effort_scale_q16(row[UNIT_COL_EFFORT],
                                                tab[UNIT_TAB_IMPULSE_EFFORT_EDGE_Q]);
            int64_t cw = 0, cf = 0, cos_q = 0, v = 0;
            if (!bad) cw = default_speed_q(n, total, tab[UNIT_TAB_SPEED_WALK_COST_Q],
                                           hook(row, UNIT_HOOK_WALK), es, &bad);
            if (!bad) cf = default_speed_q(n, total, tab[UNIT_TAB_SPEED_FULL_COST_Q],
                                           hook(row, UNIT_HOOK_SPRINT), es, &bad);
            if (!bad) cos_q = facing_cos_q16(dx_q, dy_q, row[UNIT_COL_FACING_Q], &bad);
            if (!bad) v = move_speed_q16(cw, cf, tab[UNIT_TAB_SPEED_SIDE_Q],
                                         tab[UNIT_TAB_SPEED_BACK_Q], md[3], cos_q, &bad);
            if (!bad && v > WALK_SPEED_MAX) bad = true;
            if (!bad) {
                wdx = fixedpoint::shr_trunc_q(dx_q * v, 16);
                wdy = fixedpoint::shr_trunc_q(dy_q * v, 16);
            }
        }
        if (!bad) displace(m, fp, wdx, wdy, x, y, pvx, pvy, &bad);
        // Facing: a held aim wins; otherwise the travel is the facing TARGET.
        if (!bad && walking && !row[UNIT_COL_LIVE_AIM] && (md[1] != 0 || md[2] != 0)) {
            target = normalize_q16(fixedpoint::atan2_q16((q16)(-md[2]), (q16)md[1]));
        }
    } else if (pvx || pvy) {
        // ---- no walk of its own: the pushed velocity alone ------------------
        displace(m, fp, 0, 0, x, y, pvx, pvy, &bad);
    }

    if (bad) {
        count_fault(cnt);
        return;                                   // the row as it was
    }
    bool changed = false;
    put(row, UNIT_COL_X_Q, x, changed);
    put(row, UNIT_COL_Y_Q, y, changed);
    put(row, UNIT_COL_PVX, pvx, changed);
    put(row, UNIT_COL_PVY, pvy, changed);
    put(row, UNIT_COL_FACING_TARGET_Q, target, changed);
    if (changed) count_changed(cnt);
}

}  // namespace

void unit_movement(UnitStoreDevice& store, const UnitTablesDevice& tables,
                   std::uintptr_t material, int h, int w, const UnitKernelCounts& out) {
    unit_check_pair(store, tables, "unit_movement");
    if (!material || h < 1 || w < 1)
        throw std::invalid_argument("unit_movement: the material plane and its shape are "
                                    "required");
    store.begin_counts();
    k_movement<<<bgrid(store.n_units(), store.n_env()), 128, 0, tstream()>>>(
        store.rows(), store.n_units(), tables.values(),
        reinterpret_cast<const int8_t*>(material), h, w, store.counts_dev());
    unit_finish(store, out, "unit_movement");
}

}  // namespace breach_cuda
