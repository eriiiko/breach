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
