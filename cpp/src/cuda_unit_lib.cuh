#pragma once
// ============================================================================
// #29 P6.2 -- THE DEVICE UNIT LIBRARY (residency design §4.4b, "Re-check at
// P6.2's start", item 3): the __device__ functions several unit slot kernels
// share, over the GENERATED layouts (unit_store_layout.h: a unit's row;
// unit_tables_layout.h: an env's table vector, the hooks' and the status
// kinds' indices). Integer for integer the Python it names; an arithmetic
// change lands in the Python first, then here, and the gate
// (tests/cuda_unit_kernels_check.py, part LIB) holds the two equal at tol 0.
//
//   the launch      unit_row / env_tables (the (N, U) rebase), row_valid
//   the life gate   row_alive (life_state ALIVE)
//   the counts      count_changed / count_fault / count_refused, put() (a
//                   write that marks the row changed only when it changes)
//   the hooks       hook, scaled_by, scaled          (unit_traits.hook /
//                   scaled_by / scaled: round(class value x hook), ONE
//                   rounding, skipped at ONE)
//   the statuses    composed_flags (after_sweep), apply_status (the stack
//                   cap's refusal), sweep_statuses     (status.composed_flags /
//                   apply_status / tick_statuses' sweep)
//   the angles      wrap_q16, normalize_q16, facing_cos_q16
//                   (turning.wrap_q16 / normalize_q16, movement.facing_cos_q16)
//   the turn        top_speed_within, turn_law_q16  (turning.top_speed_within /
//                   law_q16)
//   the operands    sq2 (a squared 2-vector with the range guard every
//                   kernel's squares use)
//
// A row is a unit of an env (`int64_t*` into the store); a kernel computes
// it with unit_row(), reads and writes its columns by the generated offsets
// (unit_store::UNIT_COL_*), and reads its env's tables with env_tables()
// (unit_tables::UNIT_TAB_*). A row's operand outside the range a kernel was
// proved for is a FAULT (count_fault): the row is left as it was and the
// driver refuses the kernel's result (the Python names the kernel and env).
//
// Device-only header (CUDA types); included by the unit slot TUs only.
// CLASS T: nothing in the played tick includes it.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>

#include "cuda_batch.cuh"          // benv, bptr
#include "cuda_unit_kernels.h"     // UnitKernelCounts, the store, the tables
#include "fixed_point.h"           // the kit (FP_HD)
#include "unit_store_layout.h"     // GENERATED: the row layout
#include "unit_tables_layout.h"    // GENERATED: the table layout

namespace breach_cuda {
namespace unit_lib {

using namespace unit_store;
using namespace unit_tables;

constexpr int64_t ONE = 65536;
//: unit_schema's life_state encoding ("alive", "dead").
constexpr int64_t LIFE_ALIVE = 0;
//: The status block's rows: (kind, magnitude_q16, remaining_ticks, source).
constexpr int STATUS_ROWS = UNIT_COL_STATUSES_W / 4;
//: turning.PI_Q16 / TWO_PI_Q16 / OMEGA_MAX_Q16 / TURN_TRIALS_MAX.
constexpr int64_t PI_Q16 = 205887;
constexpr int64_t TWO_PI_Q16 = 2 * PI_Q16;
constexpr int64_t OMEGA_MAX_Q16 = PI_Q16 - 1;
constexpr int TURN_TRIALS_MAX = 32;
//: The bound on a component entering a squared 2-vector: |x|, |y| <= 2^31 - 1
//: keeps x^2 + y^2 < 2^63 (int64), every root < 2^31.5.
constexpr int64_t SQ_COMPONENT_MAX = 2147483647LL;

// ---- the launch ---------------------------------------------------------------
// Row r of this block's env: the store rebased by the env's (U, W) stride.
__device__ __forceinline__ int64_t* unit_row(int64_t* store, int n_units, int r) {
    return bptr(store, (size_t)n_units * UNIT_STORE_W) + (size_t)r * UNIT_STORE_W;
}
// This block's env's table vector.
__device__ __forceinline__ const int64_t* env_tables(const int64_t* tab) {
    return bptr(tab, (size_t)UNIT_TABLES_W);
}
__device__ __forceinline__ bool row_valid(const int64_t* row) {
    return row[UNIT_COL_VALID] != 0;
}

// ---- the life gate ---------------------------------------------------------
__device__ __forceinline__ bool row_alive(const int64_t* row) {
    return row[UNIT_COL_LIFE_STATE] == LIFE_ALIVE;
}

// ---- the counts (UnitStoreDevice::counts_dev: 3 x N) -------------------------
__device__ __forceinline__ void count_changed(unsigned long long* cnt) {
    atomicAdd(&cnt[benv()], 1ULL);
}
__device__ __forceinline__ void count_fault(unsigned long long* cnt) {
    atomicAdd(&cnt[gridDim.y + benv()], 1ULL);
}
__device__ __forceinline__ void count_refused(unsigned long long* cnt) {
    atomicAdd(&cnt[2 * gridDim.y + benv()], 1ULL);
}
// Write column `col` and mark the row changed only when the value changes.
__device__ __forceinline__ void put(int64_t* row, int col, int64_t v, bool& changed) {
    if (row[col] != v) {
        row[col] = v;
        changed = true;
    }
}

// ---- the hooks (unit_traits) -------------------------------------------------
__device__ __forceinline__ int64_t hook(const int64_t* row, int h) {
    return row[UNIT_COL_TRAITS + h];
}
// unit_traits.scaled_by: class_value x hq rounded ONCE half away from zero,
// the class value itself at ONE.
__device__ __forceinline__ int64_t scaled_by(int64_t hq, int64_t class_value) {
    return hq == ONE ? class_value : fixedpoint::mul_q16_round_q(class_value, hq);
}
__device__ __forceinline__ int64_t scaled(const int64_t* row, int h, int64_t class_value) {
    return scaled_by(hook(row, h), class_value);
}

// ---- the statuses (status.py) -------------------------------------------------
struct Flags {
    bool can_move, can_act, can_aim, is_prone;
};

// status.composed_flags: AND of the suppressions, OR of prone, over the
// block's rows in list order up to the first empty row (kind < 0, the
// decoder's end). after_sweep: only the rows the coming sweep keeps
// (remaining_ticks > 0). A kind outside the registry sets *bad.
__device__ inline Flags composed_flags(const int64_t* row, const int64_t* tab,
                                       bool after_sweep, bool* bad) {
    Flags f{true, true, true, false};
    const int64_t* s = row + UNIT_COL_STATUSES;
    for (int i = 0; i < STATUS_ROWS; ++i) {
        const int64_t kind = s[4 * i];
        if (kind < 0) break;
        if (kind >= UNIT_N_STATUS_KINDS) {
            if (bad) *bad = true;
            break;
        }
        if (after_sweep && !(s[4 * i + 2] > 0)) continue;
        const int64_t fl = tab[UNIT_TAB_STATUS_FLAGS + kind];
        if (fl & UNIT_FLAG_NO_MOVE) f.can_move = false;
        if (fl & UNIT_FLAG_NO_ACT) f.can_act = false;
        if (fl & UNIT_FLAG_NO_AIM) f.can_aim = false;
        if (fl & UNIT_FLAG_PRONE) f.is_prone = true;
    }
    return f;
}

// status.apply_status on the block (magnitude already a Q16 count; source -1
// = None). Returns APPLIED, REFUSED (a stacking kind at the stack cap: nothing
// changes -- the caller counts it, as status.OVERFLOW_REFUSED does) or BAD (a
// duration < 1 or a kind outside the registry, where Python raises; or a
// block with no free row, where the encoder would raise: nothing changes).
enum ApplyResult { APPLY_APPLIED = 0, APPLY_REFUSED = 1, APPLY_BAD = 2 };

__device__ inline int apply_status(int64_t* row, const int64_t* tab, int64_t kind,
                                   int64_t mag_q16, int64_t duration, int64_t source,
                                   bool& changed) {
    if (duration < 1 || kind < 0 || kind >= UNIT_N_STATUS_KINDS) return APPLY_BAD;
    int64_t* s = row + UNIT_COL_STATUSES;
    const int64_t stacking = tab[UNIT_TAB_STATUS_STACKING + kind];
    int n = 0;                                // the list's length
    while (n < STATUS_ROWS && s[4 * n] >= 0) ++n;
    if (stacking != UNIT_STACK_STACK) {
        for (int i = 0; i < n; ++i) {         // list order: the first owns
            if (s[4 * i] != kind) continue;
            if (stacking == UNIT_STACK_REFRESH) {
                put(s, 4 * i + 1, mag_q16, changed);
                put(s, 4 * i + 2, duration, changed);
                put(s, 4 * i + 3, source, changed);
            } else {                          // STACK_MAX
                if (mag_q16 > s[4 * i + 1]) {
                    put(s, 4 * i + 1, mag_q16, changed);
                    put(s, 4 * i + 3, source, changed);
                }
                if (duration > s[4 * i + 2]) put(s, 4 * i + 2, duration, changed);
            }
            return APPLY_APPLIED;
        }
    } else {
        int held = 0;
        for (int i = 0; i < n; ++i) held += (s[4 * i] == kind) ? 1 : 0;
        if (held >= tab[UNIT_TAB_STATUS_STACK_CAP]) return APPLY_REFUSED;
    }
    if (n >= STATUS_ROWS) return APPLY_BAD;
    put(s, 4 * n + 0, kind, changed);
    put(s, 4 * n + 1, mag_q16, changed);
    put(s, 4 * n + 2, duration, changed);
    put(s, 4 * n + 3, source, changed);
    return APPLY_APPLIED;
}

// tick_statuses' sweep: drop the rows the previous pass zeroed
// (status.survives_sweep: remaining_ticks > 0 survives), keeping list order;
// the freed rows become (-1, 0, 0, -1).
__device__ inline void sweep_statuses(int64_t* row, bool& changed) {
    int64_t* s = row + UNIT_COL_STATUSES;
    int n = 0;
    while (n < STATUS_ROWS && s[4 * n] >= 0) ++n;
    int k = 0;
    for (int i = 0; i < n; ++i) {
        if (!(s[4 * i + 2] > 0)) continue;
        if (k != i) {
            for (int j = 0; j < 4; ++j) put(s, 4 * k + j, s[4 * i + j], changed);
        }
        ++k;
    }
    for (int i = k; i < n; ++i) {
        put(s, 4 * i + 0, -1, changed);
        put(s, 4 * i + 1, 0, changed);
        put(s, 4 * i + 2, 0, changed);
        put(s, 4 * i + 3, -1, changed);
    }
}

// ---- the angles (turning.py, movement.py) ------------------------------------
// turning.wrap_q16: an angle difference folded into (-PI_Q16, PI_Q16] --
// Python's % floors, so the residue is taken non-negative first.
__device__ __forceinline__ int64_t wrap_q16(int64_t d) {
    int64_t m = d % TWO_PI_Q16;
    if (m < 0) m += TWO_PI_Q16;
    return m > PI_Q16 ? m - TWO_PI_Q16 : m;
}
__device__ __forceinline__ int64_t normalize_q16(int64_t a) { return wrap_q16(a); }

// True when v fits the kit's int32 trig arguments (pybind refuses otherwise:
// the Python law raises there, the kernel faults).
__device__ __forceinline__ bool fits_i32(int64_t v) {
    return v >= -2147483648LL && v <= 2147483647LL;
}

// movement.facing_cos_q16: cos of the angle between the move (dx, dy) (world
// axes, y down) and the facing, through the kit, clamped to [-ONE, ONE].
// *bad when an argument leaves the kit's int32.
__device__ inline int64_t facing_cos_q16(int64_t dx, int64_t dy, int64_t facing_q,
                                         bool* bad) {
    if (!fits_i32(-dy) || !fits_i32(dx)) {
        *bad = true;
        return 0;
    }
    const int64_t move_q = fixedpoint::atan2_q16((q16)(-dy), (q16)dx);
    const int64_t theta = wrap_q16(move_q - facing_q);
    const int64_t c = fixedpoint::cos_q16((q16)theta);
    return c < -ONE ? -ONE : (c > ONE ? ONE : c);
}

// ---- the turn (turning.py) ----------------------------------------------------
// turning.top_speed_within: the largest speed in [0, w] whose braking
// distance is at most e >= 0; at most TURN_TRIALS_MAX trials (the door's bound:
// one more sets *bad).
__device__ inline int64_t top_speed_within(int64_t e, int64_t a, int64_t w, bool* bad) {
    int64_t best = 0;
    int64_t n = 0;
    while (n * a < w) {
        if (n >= TURN_TRIALS_MAX) {
            *bad = true;
            return best;
        }
        const int64_t lo = n * a;
        int64_t hi = (n + 1) * a;
        if (hi > w) hi = w;
        int64_t x = (e + a * ((n * (n + 1)) / 2)) / (n + 1);   // e, a, n >= 0: floor
        if (x > hi) x = hi;
        if (x > lo) {
            best = x;
        } else {
            break;
        }
        ++n;
    }
    return best;
}

// turning.law_q16: one tick of the turning law -> (facing', omega').
__device__ inline void turn_law_q16(int64_t facing_q, int64_t omega_q, int64_t target_q,
                                    int64_t a, int64_t w, int64_t* facing_out,
                                    int64_t* omega_out, bool* bad) {
    int64_t omega = omega_q;
    if (omega > OMEGA_MAX_Q16) omega = OMEGA_MAX_Q16;
    if (omega < -OMEGA_MAX_Q16) omega = -OMEGA_MAX_Q16;
    if (w > PI_Q16) w = PI_Q16;
    const int64_t e = wrap_q16(target_q - facing_q);
    const int64_t s = top_speed_within(e < 0 ? -e : e, a, w, bad);
    const int64_t want = e >= 0 ? s : -s;
    int64_t hi = omega + a;
    if (want < hi) hi = want;
    omega = (omega - a) > hi ? (omega - a) : hi;     // max(omega - a, min(omega + a, want))
    *facing_out = normalize_q16(facing_q + omega);
    *omega_out = omega;
}

// ---- the operands ----------------------------------------------------------
// x^2 + y^2 of a 2-vector whose components are within SQ_COMPONENT_MAX (else
// *bad, returns 0).
__device__ __forceinline__ int64_t sq2(int64_t x, int64_t y, bool* bad) {
    if (x > SQ_COMPONENT_MAX || x < -SQ_COMPONENT_MAX ||
        y > SQ_COMPONENT_MAX || y < -SQ_COMPONENT_MAX) {
        *bad = true;
        return 0;
    }
    return x * x + y * y;
}

}  // namespace unit_lib

// ---- the host side every slot driver shares ------------------------------------
// The store and the tables must describe the same envs.
inline void unit_check_pair(const UnitStoreDevice& store, const UnitTablesDevice& tables,
                            const char* who) {
    if (tables.n_env() != store.n_env()) {
        std::ostringstream os;
        os << who << ": the tables hold " << tables.n_env() << " envs, the store "
           << store.n_env();
        throw std::invalid_argument(os.str());
    }
}

// After a slot's launch: the launch error, then the counts (one D2H + sync).
inline void unit_finish(UnitStoreDevice& store, const UnitKernelCounts& out, const char* who) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in " << who << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
    store.end_counts(out.coverage, out.faults, out.refused);
}

}  // namespace breach_cuda

// ============================================================================
// #29 P6.2m -- the walk's shared functions (the footprint walk, the speed
// reduction, the direction rule, the walk ban): slot 3 is their first user;
// the zombie walk (slot 5) and the onephase / WEGO walkers read the same
// Python functions, so they live here, not in the slot TU.
// ============================================================================
namespace breach_cuda {
namespace unit_lib {

//: The proved ranges of the walk (cuda_unit_movement.cu's header): a
//: footprint of at most FOOTPRINT_MAX^2 tiles, a Q16 tick cost of at most
//: COST_MAX after each reduction step, a hook at most HOOK_OPERAND_MAX.
constexpr int64_t FOOTPRINT_MAX = 8;
constexpr int64_t COST_MAX = (int64_t)1 << 29;
constexpr int64_t HOOK_OPERAND_MAX = (int64_t)1 << 20;
//: movement.MOBILITY_ONE.
constexpr int64_t MOBILITY_ONE = 1000;

// Unit.tile_x / tile_y: int(x) of a Q16 position (truncation toward zero).
__device__ __forceinline__ int64_t tile_of(int64_t q) {
    return fixedpoint::shr_trunc_q(q, 16);
}

// unit_impulse.walk_allowed: no walk while the reaction delay runs.
__device__ __forceinline__ bool walk_allowed(const int64_t* row) {
    return row[UNIT_COL_REACT_TICKS] == 0;
}

// unit_impulse.effort_scale_q16: ONE up to the slow edge e0, linear down to
// half at an effort of ONE, half above (floored; every operand non-negative
// on the division's branch).
__device__ __forceinline__ int64_t effort_scale_q16(int64_t e, int64_t e0) {
    if (e <= e0) return ONE;
    if (e >= ONE) return ONE / 2;
    return ONE - ((ONE / 2) * (e - e0)) / (ONE - e0);
}

// The terrain a walk reads: an env's material plane (GameMap.material, int8,
// the resident field) projected through the materials table's mobility
// column (the device tables' material_mobility row).
struct MobilityMap {
    const int8_t* mat;      // (h, w) of this env
    int h, w;
    const int64_t* mob;     // the mobility column, n_mat entries
    int n_mat;
};

// A material id outside the table: Python's numpy index wraps a negative id
// and raises past the end; the kernel refuses both (*bad).
__device__ __forceinline__ int64_t mobility_at(const MobilityMap& m, int64_t y, int64_t x,
                                               bool* bad) {
    const int id = (int)m.mat[(size_t)y * m.w + (size_t)x];
    if (id < 0 || id >= m.n_mat) {
        *bad = true;
        return 0;
    }
    return m.mob[id];
}

// GameMap.is_passable_block: the fp x fp block at (fy, fx) in bounds and every
// tile's mobility > 0 (every tile read, so a bad id refuses as numpy would).
__device__ inline bool passable_block(const MobilityMap& m, int64_t fy, int64_t fx,
                                      int64_t fp, bool* bad) {
    if (fy < 0 || fx < 0 || fy + fp > m.h || fx + fp > m.w) return false;
    bool all = true;
    for (int64_t y = fy; y < fy + fp; ++y)
        for (int64_t x = fx; x < fx + fp; ++x)
            if (!(mobility_at(m, y, x, bad) > 0)) all = false;
    return all;
}

// GameMap.footprint_mobility reduced as default_speed reduces it: the tile
// count n and the mobility sum over the block clamped to the map. A block
// whose clamped stop is negative (numpy would count from the end) refuses.
__device__ inline void footprint_mobility(const MobilityMap& m, int64_t fy, int64_t fx,
                                          int64_t fp, int64_t* n, int64_t* total,
                                          bool* bad) {
    const int64_t y0 = fy > 0 ? fy : 0, x0 = fx > 0 ? fx : 0;
    const int64_t y1 = (fy + fp) < m.h ? (fy + fp) : m.h;
    const int64_t x1 = (fx + fp) < m.w ? (fx + fp) : m.w;
    *n = 0;
    *total = 0;
    if (y1 < 0 || x1 < 0) {
        *bad = true;
        return;
    }
    for (int64_t y = y0; y < y1; ++y)
        for (int64_t x = x0; x < x1; ++x) {
            *total += mobility_at(m, y, x, bad);
            *n += 1;
        }
}

// movement.default_speed over (n, total): the class cost (Q16 ticks per
// tile) times n * MOBILITY_ONE over the mobility sum, the speed hook inside
// that one rounding (skipped at ONE), then the effort scale (skipped at ONE),
// at least 1. Python raises on a sum <= 0, a hook or scale <= 0 (*bad); a
// cost past COST_MAX leaves the proved range (*bad).
__device__ inline int64_t default_speed_q(int64_t n, int64_t total, int64_t speed_class,
                                          int64_t hook_q16, int64_t scale_q16, bool* bad) {
    if (total <= 0 || n < 1 || n > FOOTPRINT_MAX * FOOTPRINT_MAX || speed_class < 0 ||
        speed_class > COST_MAX || hook_q16 <= 0 || hook_q16 > HOOK_OPERAND_MAX ||
        scale_q16 <= 0) {
        *bad = true;
        return 0;
    }
    int64_t cost;
    if (hook_q16 == ONE) {
        cost = fixedpoint::half_up_q(speed_class * n * MOBILITY_ONE, total);
    } else {
        cost = fixedpoint::half_up_q(speed_class * n * MOBILITY_ONE * ONE, total * hook_q16);
    }
    if (cost > COST_MAX) {
        *bad = true;
        return 0;
    }
    if (scale_q16 != ONE) cost = fixedpoint::half_up_q(cost * ONE, scale_q16);
    if (cost > COST_MAX) {
        *bad = true;
        return 0;
    }
    return cost >= 1 ? cost : 1;
}

// movement.move_speed_q16 -- THE direction rule and the throttle: the exact
// rational (F * Cf + t * c+ * (Cw - Cf)) / (Cw * Cf), F = f(theta) in Q32,
// rounded half up ONCE to Q16 tiles per tick. Python refuses a throttle
// outside [0, ONE], a cos outside [-ONE, ONE], a cost below 1 (*bad); a cost
// past COST_MAX or a direction factor outside [0, ONE] leaves the proved range
// (*bad).
__device__ inline int64_t move_speed_q16(int64_t cw, int64_t cf, int64_t side, int64_t back,
                                         int64_t t, int64_t c, bool* bad) {
    if (t < 0 || t > ONE || c < -ONE || c > ONE || cw < 1 || cf < 1 || cw > COST_MAX ||
        cf > COST_MAX || side < 0 || side > ONE || back < 0 || back > ONE) {
        *bad = true;
        return 0;
    }
    int64_t f32, c_plus;
    if (c >= 0) {
        f32 = side * ONE + (ONE - side) * c;
        c_plus = c;
    } else {
        f32 = side * ONE + (side - back) * c;
        c_plus = 0;
    }
    return fixedpoint::half_up_q(f32 * cf + t * c_plus * (cw - cf), cw * cf);
}

}  // namespace unit_lib
}  // namespace breach_cuda

// ============================================================================
// #29 P6.4a -- THE HAND VERBS' GATES (verb_gates.py) and what they read: the
// 1b envelope (acts), the HANDLING durations (unit_traits._handling), the
// magazine as mag_status reads it (combat.magazines_carried / mag_status),
// the shove's cooldown (unit_impulse.shove_cooled). Slot 1b is their first
// user; P8b's mask kernel calls the same functions at the tick's start. And
// the RECORD LIST's view and row sink (cuda_unit_events.h's UnitRecordList:
// the request list of slot 1b; the event ring is the other instance).
// ============================================================================
namespace breach_cuda {

// What a kernel needs of a record list (built on the host from a
// UnitRecordList): any record width / row cap.
struct RecordListView {
    int64_t* recs = nullptr;                // (N, cap, width)
    int64_t* count = nullptr;               // (N,)
    unsigned long long* dropped = nullptr;  // (N,)
    int64_t* stage = nullptr;               // (N, U, row_cap, width)
    int64_t* stage_count = nullptr;         // (N, U)
    int cap = 0, width = 0, row_cap = 0;
};

inline RecordListView list_view(UnitRecordList& l) {
    RecordListView v;
    v.recs = l.records_dev();
    v.count = l.count_dev();
    v.dropped = l.dropped_dev();
    v.stage = l.stage_dev();
    v.stage_count = l.stage_count_dev();
    v.cap = l.cap();
    v.width = l.width();
    v.row_cap = l.row_cap();
    return v;
}

namespace unit_lib {

// A record sink of any width: a row's staging (gathered in row order) or the
// env's list itself (a serial per-env producer).
struct RecordSink {
    int64_t* recs;
    int64_t cap;
    int64_t* count;
    unsigned long long* dropped;
    int width;
};
__device__ __forceinline__ RecordSink list_row_sink(const RecordListView& v, int n_units, int r) {
    const size_t rr = (size_t)benv() * n_units + r;
    return RecordSink{v.stage + rr * (size_t)v.row_cap * v.width, v.row_cap,
                      v.stage_count + rr, v.dropped + benv(), v.width};
}
// Append one record (width int64); past the capacity: dropped and counted.
__device__ inline void emit_record(RecordSink& s, const int64_t* rec) {
    const int64_t n = *s.count;
    if (n >= s.cap) {
        atomicAdd(s.dropped, 1ULL);
        return;
    }
    int64_t* d = s.recs + n * s.width;
    for (int f = 0; f < s.width; ++f) d[f] = rec[f];
    *s.count = n + 1;
}

//: unit_schema's sentinels and combat.SPARES_UNTRACKED.
constexpr int64_t MAG_UNBOUND = -1;
constexpr int64_t SPARES_UNTRACKED = -1;
constexpr int64_t NO_ROW = -1;
constexpr int64_t NO_PLANT = -1;
constexpr int64_t THROW_NO_REACH = -2147483648LL;
//: The proved range of a tick and of a duration added to it (a tick plus a
//: HANDLING-scaled duration stays far inside int64).
constexpr int64_t TICK_MAX = (int64_t)1 << 40;

// verb_gates._acts: the 1b envelope's unit terms -- living and not a zombie.
__device__ __forceinline__ bool acts(const int64_t* row) {
    return row_alive(row) && row[UNIT_COL_IS_ZOMBIE] == 0;
}

// The statuses' can_act (composed_flags, not after the sweep).
__device__ __forceinline__ bool can_act(const int64_t* row, const int64_t* tab, bool* bad) {
    return composed_flags(row, tab, false, bad).can_act;
}

// unit_traits._handling: a duration's class ticks times the HANDLING hook,
// at least one tick when the class value is (a class value <= 0 is itself).
__device__ __forceinline__ int64_t handling_ticks(const int64_t* row, int64_t base) {
    if (base <= 0) return base;
    const int64_t t = scaled(row, UNIT_HOOK_HANDLING, base);
    return t >= 1 ? t : 1;
}

// verb_gates.can_throw for hand-grenade kind `kind`: not winding up, not
// planting, one of the kind carried, able to act. A kind outside the family
// sets *bad (the Python refuses it by name).
__device__ inline bool can_throw(const int64_t* row, const int64_t* tab, int64_t kind,
                                 bool* bad) {
    if (kind < 0 || kind >= UNIT_COL_GRENADE_COUNTS_W) {
        *bad = true;
        return false;
    }
    return acts(row) && row[UNIT_COL_THROW_RELEASE_TICK] < 0 &&
           row[UNIT_COL_PLANT_DONE_TICK] < 0 && row[UNIT_COL_GRENADE_COUNTS + kind] > 0 &&
           can_act(row, tab, bad);
}

// verb_gates.can_plant at `tick`: not planting, no wind-up still running at
// the plant's consumption (throw_release_tick <= tick), a charge carried, able
// to act.
__device__ inline bool can_plant(const int64_t* row, const int64_t* tab, int64_t tick,
                                 bool* bad) {
    return acts(row) && row[UNIT_COL_PLANT_DONE_TICK] < 0 &&
           row[UNIT_COL_THROW_RELEASE_TICK] <= tick && row[UNIT_COL_HAS_EXPLOSIVE] > 0 &&
           row[UNIT_COL_CHARGE_AMMO] != NO_ROW && can_act(row, tab, bad);
}

// The weapon row the unit holds: its weapons-table index, or -1 for none
// (weapon_id ""); an index past the table sets *bad (no such row).
__device__ __forceinline__ int64_t weapon_row(const int64_t* row, bool* bad) {
    const int64_t w = row[UNIT_COL_WEAPON_ID];
    if (w == NO_ROW) return -1;
    if (w < 0 || w >= UNIT_TAB_WEAPON_MAG_SIZE_W) {
        *bad = true;
        return -1;
    }
    return w;
}

// combat.magazines_carried: the loadout's magazine count when it belongs to
// the weapon in the hands (magazines_weapon), else 0. The column decodes "no
// row" to "" (never None: Unit.__init__ writes "" too), and "" is not None
// in the Python's test, so a weapon in the hands with no loadout weapon is
// "another weapon": the encoded indices compare as the names do.
__device__ __forceinline__ int64_t magazines_carried(const int64_t* row) {
    const int64_t total = row[UNIT_COL_MAGAZINES];
    if (total > 0 && row[UNIT_COL_WEAPON_ID] != row[UNIT_COL_MAGAZINES_WEAPON]) return 0;
    return total;
}

// combat.mag_status: the magazine as mag_settle would leave it at `tick`,
// without writing. tracked false = untracked (no weapon row or mag_size 0,
// Python's None); spares -1 = untracked spares (Python's None).
struct MagStatus {
    bool tracked;
    int64_t rounds, spares;
    bool reloading;
};
__device__ inline MagStatus mag_status(const int64_t* row, const int64_t* tab, int64_t tick,
                                       bool* bad) {
    MagStatus m{false, 0, -1, false};
    const int64_t w = weapon_row(row, bad);
    if (w < 0) return m;
    const int64_t full = tab[UNIT_TAB_WEAPON_MAG_SIZE + w];
    if (full <= 0) return m;
    m.tracked = true;
    if (row[UNIT_COL_CURRENT_MAG] == MAG_UNBOUND) {
        const int64_t total = magazines_carried(row);
        m.rounds = full;
        m.spares = total <= 0 ? -1 : total - 1;
        return m;
    }
    const int64_t spare = row[UNIT_COL_SPARE_MAGS];
    m.spares = spare < 0 ? -1 : spare;
    const int64_t done = row[UNIT_COL_RELOAD_DONE_TICK];
    if (done >= 0) {
        if (tick < done) {
            m.rounds = row[UNIT_COL_CURRENT_MAG];
            m.reloading = true;
        } else {
            m.rounds = full;
        }
        return m;
    }
    m.rounds = row[UNIT_COL_CURRENT_MAG];
    return m;
}

// verb_gates.can_reload: a tracked magazine, no stall running, not full, a
// spare (counted or untracked). No can_act, wind-up or plant term (the
// oversight, kept).
__device__ inline bool can_reload(const int64_t* row, const int64_t* tab, int64_t tick,
                                  bool* bad) {
    if (!acts(row)) return false;
    const MagStatus m = mag_status(row, tab, tick, bad);
    if (!m.tracked) return false;
    const int64_t full = tab[UNIT_TAB_WEAPON_MAG_SIZE + row[UNIT_COL_WEAPON_ID]];
    return !m.reloading && m.rounds < full && m.spares != 0;
}

// unit_impulse.shove_cooled: the shove's cooldown has run at `tick`.
__device__ __forceinline__ bool shove_cooled(const int64_t* row, const int64_t* tab,
                                             int64_t tick) {
    const int64_t last = row[UNIT_COL_LAST_SHOVE_TICK];
    return !(last >= 0 && tick - last < tab[UNIT_TAB_SHOVE_COOLDOWN_TICKS]);
}

// verb_gates.can_shove: able to act and off cooldown (whoever is in reach
// is no term).
__device__ inline bool can_shove(const int64_t* row, const int64_t* tab, int64_t tick,
                                 bool* bad) {
    return acts(row) && can_act(row, tab, bad) && shove_cooled(row, tab, tick);
}

}  // namespace unit_lib
}  // namespace breach_cuda

// ============================================================================
// #29 P7a.1 -- THE PUSH and THE MELEE VERB'S GATE AND TARGET: the impulse add
// (unit_impulse.add with its angular part angular_q -- slot 1b's shove and
// the serial applier's PUSH effect both need it), the footprint centre
// (unit_impulse.centre_q), verb_gates.can_melee in both of its moments (slot
// 4a's own read and the tick-start read P8b's mask kernel calls), the
// adjacency predicate (combat.melee_adjacent) and the arc target
// (combat.melee_arc_target). Integer for integer the Python; held equal on
// synthetic rows by the verbs probe (cuda_unit_verbs_probe.cu).
// ============================================================================
namespace breach_cuda {
namespace unit_lib {

// unit_impulse.centre_q on one axis: the footprint's centre, Q16 tiles.
__device__ __forceinline__ int64_t centre_q(int64_t pos_q, int64_t fp) {
    return pos_q + (fp << 15);
}

__device__ __forceinline__ bool add_i64(int64_t a, int64_t b, int64_t* out) {
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) return false;
    *out = a + b;
    return true;
}

// unit_impulse.angular_q: the lever CLAMPED to the body radius (per input),
// times the push's magnitude (isqrt of the squared magnitude, taken in 128
// bits), times twist_q16 (skipped at ONE), each product truncated toward
// zero. Python's integers are unbounded: a product past int64 sets *bad.
__device__ inline int64_t angular_q(const int64_t* tab, int64_t jx, int64_t jy, int64_t lever,
                                    int64_t twist_q16, bool* bad) {
    const int64_t rb = tab[UNIT_TAB_IMPULSE_R_BODY_Q];
    const int64_t lev = lever > rb ? rb : (lever < -rb ? -rb : lever);
    if (lev == 0) return 0;
    if (jx == INT64_MIN || jy == INT64_MIN) {
        *bad = true;
        return 0;
    }
    const uint64_t ax = (uint64_t)(jx < 0 ? -jx : jx), ay = (uint64_t)(jy < 0 ? -jy : jy);
    uint64_t h1, l1, h2, l2;
    fixedpoint::umul64_wide(ax, ax, &h1, &l1);
    fixedpoint::umul64_wide(ay, ay, &h2, &l2);
    const uint64_t lo = l1 + l2;
    const uint64_t hi = h1 + h2 + (lo < l1 ? 1u : 0u);
    const uint64_t mag = fixedpoint::isqrt_u128_q(hi, lo);
    const int64_t alev = lev < 0 ? -lev : lev;
    if (mag > (uint64_t)(INT64_MAX / alev)) {
        *bad = true;
        return 0;
    }
    int64_t l_q = fixedpoint::shr_trunc_q((int64_t)mag * lev, 16);
    if (twist_q16 != ONE) {
        const int64_t at = twist_q16 < 0 ? -twist_q16 : twist_q16;
        const int64_t al = l_q < 0 ? -l_q : l_q;
        if (twist_q16 == INT64_MIN || (at != 0 && al > INT64_MAX / at)) {
            *bad = true;
            return 0;
        }
        l_q = fixedpoint::shr_trunc_q(l_q * twist_q16, 16);
    }
    return l_q;
}

// unit_impulse.add on a row: the linear impulse into imp_x / imp_y and, when
// the lever is non-zero, its angular part into imp_l. Every operand is read
// and every sum checked before the first write (all or nothing). The
// REFERENCE_ONLY momentum books' IN entries have no device form.
__device__ inline void impulse_add(int64_t* row, const int64_t* tab, int64_t jx, int64_t jy,
                                   int64_t lever, int64_t twist_q16, bool& changed, bool* bad) {
    int64_t ix, iy, il = row[UNIT_COL_IMP_L];
    if (!add_i64(row[UNIT_COL_IMP_X], jx, &ix) || !add_i64(row[UNIT_COL_IMP_Y], jy, &iy)) {
        *bad = true;
        return;
    }
    if (lever != 0) {
        bool b = false;
        const int64_t dl = angular_q(tab, jx, jy, lever, twist_q16, &b);
        if (b || !add_i64(il, dl, &il)) {
            *bad = true;
            return;
        }
    }
    put(row, UNIT_COL_IMP_X, ix, changed);
    put(row, UNIT_COL_IMP_Y, iy, changed);
    put(row, UNIT_COL_IMP_L, il, changed);
}

// The melee row a unit holds: its weapons-table index, -1 for none
// (melee_id ""); an index past the table sets *bad.
__device__ __forceinline__ int64_t melee_row(const int64_t* row, bool* bad) {
    const int64_t m = row[UNIT_COL_MELEE_ID];
    if (m == NO_ROW) return -1;
    if (m < 0 || m >= UNIT_TAB_WEAPON_ROF_TICKS_W) {
        *bad = true;
        return -1;
    }
    return m;
}

// verb_gates.can_melee: a melee row, able to act, no wind-up, no plant, the
// row's cadence run. at_tick_start false is slot 4a's own read (the verb);
// true the same decision read at the start of the tick (the mask: the
// statuses after the coming sweep; a wind-up / plant that slot 1b keeps only
// while not past and the unit can still act).
__device__ inline bool can_melee(const int64_t* row, const int64_t* tab, int64_t tick,
                                 bool at_tick_start, bool* bad) {
    if (!acts(row)) return false;
    const int64_t m = melee_row(row, bad);
    if (m < 0) return false;
    if (at_tick_start) {
        if (!composed_flags(row, tab, true, bad).can_act) return false;
        const bool holds = composed_flags(row, tab, false, bad).can_act;
        if (holds && (row[UNIT_COL_THROW_RELEASE_TICK] > tick ||
                      row[UNIT_COL_PLANT_DONE_TICK] > tick))
            return false;
    } else {
        if (!composed_flags(row, tab, false, bad).can_act ||
            row[UNIT_COL_THROW_RELEASE_TICK] >= 0 || row[UNIT_COL_PLANT_DONE_TICK] >= 0)
            return false;
    }
    const int64_t last = row[UNIT_COL_LAST_BLOW_TICK];
    return last < 0 || tick - last >= tab[UNIT_TAB_WEAPON_ROF_TICKS + m];
}

// combat.melee_adjacent for the row-major square footprints (the tables'
// footprint_rowmajor row): some occupied tile of one within Chebyshev 1 of
// some occupied tile of the other -- the two anchor squares, one dilated by
// one, intersect. A footprint outside [1, FOOTPRINT_MAX] sets *bad.
__device__ inline bool melee_adjacent(const int64_t* a, const int64_t* t, bool* bad) {
    const int64_t fa = a[UNIT_COL_FOOTPRINT], ft = t[UNIT_COL_FOOTPRINT];
    if (fa < 1 || fa > FOOTPRINT_MAX || ft < 1 || ft > FOOTPRINT_MAX) {
        *bad = true;
        return false;
    }
    const int64_t ax = tile_of(a[UNIT_COL_X_Q]), ay = tile_of(a[UNIT_COL_Y_Q]);
    const int64_t tx = tile_of(t[UNIT_COL_X_Q]), ty = tile_of(t[UNIT_COL_Y_Q]);
    return ax - tx <= ft && tx - ax <= fa && ay - ty <= ft && ty - ay <= fa;
}

// combat.melee_arc_target: among the env's OTHER living valid rows of a
// different team that are melee_adjacent to row r (`loc` its slot-start
// copy) and whose bearing from r's centre lies within the melee row m's
// half-arc of r's facing, the one nearest that facing, ties to the lower id;
// a centre that coincides counts as dead ahead. -1 = none. Reads the
// slot-start rows (slot 4a writes no position, footprint, team, life state
// or id).
__device__ inline int melee_arc_target(int64_t* store, int n_units, int r, const int64_t* loc,
                                       const int64_t* tab, int64_t m, bool* bad) {
    const int64_t fp = loc[UNIT_COL_FOOTPRINT];
    const int64_t cx = centre_q(loc[UNIT_COL_X_Q], fp), cy = centre_q(loc[UNIT_COL_Y_Q], fp);
    const int64_t facing = loc[UNIT_COL_FACING_Q];
    const int64_t arc = tab[UNIT_TAB_WEAPON_MELEE_ARC_HALF_Q + m];
    int best = -1;
    int64_t best_off = 0, best_id = 0;
    for (int t = 0; t < n_units; ++t) {
        if (t == r) continue;
        const int64_t* tr = unit_row(store, n_units, t);
        if (!row_valid(tr) || !row_alive(tr) || tr[UNIT_COL_TEAM] == loc[UNIT_COL_TEAM]) continue;
        if (!melee_adjacent(loc, tr, bad)) continue;
        const int64_t dx = centre_q(tr[UNIT_COL_X_Q], tr[UNIT_COL_FOOTPRINT]) - cx;
        const int64_t dy = centre_q(tr[UNIT_COL_Y_Q], tr[UNIT_COL_FOOTPRINT]) - cy;
        int64_t off = 0;
        if (dx != 0 || dy != 0) {
            if (!fits_i32(-dy) || !fits_i32(dx)) {
                *bad = true;
                return -1;
            }
            off = wrap_q16(fixedpoint::atan2_q16((q16)(-dy), (q16)dx) - facing);
            if (off < 0) off = -off;
        }
        if (off > arc) continue;
        const int64_t id = tr[UNIT_COL_ID];
        if (best < 0 || off < best_off || (off == best_off && id < best_id)) {
            best = t;
            best_off = off;
            best_id = id;
        }
    }
    return best;
}

}  // namespace unit_lib
}  // namespace breach_cuda

// ============================================================================
// #29 P7a.3 -- a SERIAL per-env producer's sink on a record list: env e's
// list itself (the cell buffer, the claims list, the edit door's rows, the
// effect list when one thread per env appends). P7b's two private copies
// (cuda_unit_objects.cu env_list_sink, cuda_object_table.cu table_sink) are
// this function and may fold onto it.
// ============================================================================
namespace breach_cuda {
namespace unit_lib {
__device__ __forceinline__ RecordSink list_env_sink(const RecordListView& v, int e) {
    return RecordSink{v.recs + (size_t)e * v.cap * v.width, v.cap, v.count + e, v.dropped + e,
                      v.width};
}
}  // namespace unit_lib
}  // namespace breach_cuda
