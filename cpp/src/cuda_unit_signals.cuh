#pragma once
// ============================================================================
// #29 P6b.3 -- THE SIGNALS as device functions (residency design 4.4c, the
// cut table's P6b.3 row; decision 1: slot 9e's sub-steps are STAGES of the one
// applier kernel walking the entity tables in the reference's order, each
// entity applied AT ONCE). On ONE env, integer for integer:
//
//   emit         (a) Simulation._signal_emit (every bus row in slot order: a
//                door's is_open / alive from the door table, a constant alive)
//                then sensor_system.sample_sensors (every sensor in ordinal
//                order; a dead one publishes 0): field sensors through
//                sample_raw (EntityFieldAccessor.sample / .area: the RAW
//                Q16 integer of the channel, smoke converted from the trace
//                currency PER SAMPLED CELL, the area mean floored over the
//                currently non-solid disc tiles), the clock (tick // period),
//                motion plates over the unit rows with unit_sight::has_los;
//   logic        (b) logic_nodes.sweep_logic_nodes: every node in ordinal
//                order reads pub, writes stg (the predicate a CODE, the
//                filter's ema in int64 with its proved range, the airlock
//                controller's Moore machine);
//   door_inputs  (c) Simulation._resolve_door_inputs: each wired door's
//                want_open (close beats open; neither keeps the latch);
//   pumps        (d) pump_system.sweep_pumps: every pump in ordinal order, its
//                flux through the topology seam's gas-N primitives
//                (transfer_gas_n / extract_gas_n_vec / inject_gas_n, extract
//                beats inject), then the latched Schmitt at_target from the
//                port's atmosphere, published to its slot;
//   swap         (e) SignalBus.swap_node_signals: pub <- stg on the node slots.
//
// THE FILTER'S PROVED RANGE (Python's ema is unbounded): with |ema| <= 2^60,
// |x| <= 2^60 and 0 <= k <= 62 (half <= 2^61), ema + half, the floored value
// (|v| <= 2^61), ema + x - v (|.| <= 3 2^60 + 2^61 < 2^63) and the output's
// ema' + half (< 5 2^60 + 2^61 < 2^63) are exact in int64. The stage's head
// checks every filter of the env against it and, past it, REFUSES the env's
// LOGIC stage (the signal_refused book; nothing written; the Python face
// raises applier.SignalRangeRefused naming the node). The shifts of possibly
// negative values go through the kit's floordiv_q (Python's >> is a floor).
// An airlock controller in interior mode reads three pressures: each must lie
// within +-2^61 (a FAULT past it, nothing of the env written).
//
// A FAULT (the `faults` book; the Python face refuses the run): a slot index
// outside the bus, a door row outside the door table, a span outside its side
// table, a sample tile outside the grid, a gas slice outside the planes, a
// clock period < 1, a footprint outside [1, FOOTPRINT_MAX], a sight endpoint
// past its range, a seam call the reference would raise on (a pump's port off
// the grid, an overflow) or past its proved range. `signal_writes` counts
// every write that changed a value (a pub / stg slot, want_open, a node's
// ema / state / dwell, a pump's at_target).
//
// Device-only header; included by cuda_applier.cu only. CLASS T.
#include "cuda_topology.cuh"
#include "cuda_unit_lib.cuh"
#include "cuda_unit_sight.cuh"
#include "object_tables_layout.h"

namespace breach_cuda {
namespace signals {

using namespace object_tables;
using namespace unit_store;
using namespace unit_tables;
using namespace applier;

//: The filter's and the airlock's proved operand bound.
constexpr int64_t FILTER_LIM = (int64_t)1 << 60;
constexpr int64_t FILTER_K_MAX = 62;
constexpr int64_t PRESS_LIM = (int64_t)1 << 61;

// One env's signal tables (rows and counts; null when the stage reads none).
struct Tables {
    int64_t* bus = nullptr;        int64_t n_bus = 0;
    int64_t* sensors = nullptr;    int64_t n_sensors = 0;
    int64_t* stiles = nullptr;     int64_t n_stiles = 0;
    int64_t* nodes = nullptr;      int64_t n_nodes = 0;
    int64_t* nins = nullptr;       int64_t n_nins = 0;
    int64_t* pumps = nullptr;      int64_t n_pumps = 0;
    int64_t* pins = nullptr;       int64_t n_pins = 0;
    int64_t* drives = nullptr;     int64_t n_drives = 0;
    int64_t* doors = nullptr;      int64_t n_doors = 0;
};

__device__ __forceinline__ int64_t* bus_row(const Tables& T, int64_t s) {
    return T.bus + (size_t)s * OBJ_SIGNAL_W;
}
__device__ __forceinline__ bool slot_ok(const Tables& T, int64_t s) {
    return s >= 0 && s < T.n_bus;
}
__device__ __forceinline__ int64_t pub(const Tables& T, int64_t s) {
    return bus_row(T, s)[OBJ_SIGNAL_COL_PUB];
}
// A write that counts in signal_writes when it changes the value.
__device__ __forceinline__ void put(int64_t* cell, int64_t v, int64_t* B) {
    if (*cell != v) {
        *cell = v;
        B[APPLIER_BOOK_SIGNAL_WRITES] += 1;
    }
}
__device__ __forceinline__ void set_pub(const Tables& T, int64_t s, int64_t v, int64_t* B) {
    put(bus_row(T, s) + OBJ_SIGNAL_COL_PUB, v, B);
}
__device__ __forceinline__ void set_stg(const Tables& T, int64_t s, int64_t v, int64_t* B) {
    put(bus_row(T, s) + OBJ_SIGNAL_COL_STG, v, B);
}

// ----------------------------------------------------------------------------
// The accessor's RAW sample (EntityFieldAccessor._channel_array / .sample):
// channel `ch` (SIGNAL_CHANNEL_*) at cell i; `gas` the channel's gas slice
// (smoke, o2). False: a slice outside the planes or an unknown channel.
// ----------------------------------------------------------------------------
__device__ inline bool sample_raw(const topo::TopoEnv& E, int64_t ch, int64_t gas, int i,
                                  int64_t* v) {
    if (ch == SIGNAL_CHANNEL_PRESSURE) { *v = E.atmosphere[i]; return true; }
    if (ch == SIGNAL_CHANNEL_SMOKE) {
        if (gas < 0 || gas >= E.n_gases) return false;
        // gas_fixed.trace_to_q16: the trace currency to Q16, floor (>> TRACE_FINE_BITS)
        *v = fixedpoint::floordiv_q((int64_t)E.g((int)gas, i),
                                    (int64_t)1 << fixedpoint::TRACE_FINE_BITS);
        return true;
    }
    if (ch == SIGNAL_CHANNEL_WATER_DEPTH) { *v = E.water_depth[i]; return true; }
    if (ch == SIGNAL_CHANNEL_TEMPERATURE) { *v = E.temperature[i]; return true; }
    if (ch == SIGNAL_CHANNEL_FIRE) { *v = E.fire[i]; return true; }
    if (ch == SIGNAL_CHANNEL_O2) {
        if (gas < 0 || gas >= E.n_gases) return false;
        *v = E.g((int)gas, i);
        return true;
    }
    if (ch == SIGNAL_CHANNEL_SOLID) { *v = E.solid[i] ? 1 : 0; return true; }
    return false;
}

// A field sensor's value: the single-tile sample, or the floored area mean
// over the disc's currently NON-SOLID tiles (0 when none). False: a fault.
__device__ inline bool field_value(const topo::TopoEnv& E, const Tables& T, const int64_t* s,
                                   int64_t* v) {
    const int64_t ch = s[OBJ_SENSOR_COL_CHANNEL], gas = s[OBJ_SENSOR_COL_GAS];
    if (s[OBJ_SENSOR_COL_HAS_DISC] == 0) {
        const int64_t y = s[OBJ_SENSOR_COL_SY], x = s[OBJ_SENSOR_COL_SX];
        if (!topo::in_grid64(E, y, x)) return false;
        return sample_raw(E, ch, gas, (int)y * E.w + (int)x, v);
    }
    const int64_t a = s[OBJ_SENSOR_COL_TILE_START], n = s[OBJ_SENSOR_COL_TILE_COUNT];
    if (a < 0 || n < 0 || a + n > T.n_stiles) return false;
    int64_t total = 0, cnt = 0;
    for (int64_t j = 0; j < n; ++j) {
        const int64_t* t = T.stiles + (size_t)(a + j) * OBJ_SENSOR_TILE_W;
        const int64_t y = t[OBJ_SENSOR_TILE_COL_TY], x = t[OBJ_SENSOR_TILE_COL_TX];
        if (!topo::in_grid64(E, y, x)) return false;
        const int i = (int)y * E.w + (int)x;
        if (E.solid[i]) continue;                 // currently-solid tiles excluded (D6)
        int64_t c = 0;
        if (!sample_raw(E, ch, gas, i, &c)) return false;
        total += c;                               // |c| < 2^31, n <= 1024: exact
        ++cnt;
    }
    *v = cnt > 0 ? fixedpoint::floordiv_q(total, cnt) : 0;
    return true;
}

// A motion plate (_MotionSensorRuntime.evaluate): the living rows within r2
// of the plate (the anchor corner), of its faction (-1 any), at least its
// footprint, in line of sight from the plate when it needs one. A count, so
// the rows' order (sim.units' id order) does not change it. False: a fault.
__device__ inline bool motion_value(const topo::TopoEnv& E, const int64_t* s,
                                    const int64_t* store, int n_units, int64_t* v) {
    const int64_t sy = s[OBJ_SENSOR_COL_SY], sx = s[OBJ_SENSOR_COL_SX];
    const int64_t r2 = s[OBJ_SENSOR_COL_R2], fac = s[OBJ_SENSOR_COL_FACTION];
    const int64_t fp_min = s[OBJ_SENSOR_COL_MIN_FOOTPRINT];
    const bool los = s[OBJ_SENSOR_COL_NEEDS_LOS] != 0;
    if (sy < -unit_sight::COORD_MAX || sy > unit_sight::COORD_MAX ||
        sx < -unit_sight::COORD_MAX || sx > unit_sight::COORD_MAX)
        return false;
    int64_t count = 0;
    for (int r = 0; r < n_units; ++r) {
        const int64_t* row = unit_lib::unit_row(const_cast<int64_t*>(store), n_units, r);
        if (!unit_lib::row_valid(row) || !unit_lib::row_alive(row)) continue;
        const int64_t uy = unit_lib::tile_of(row[UNIT_COL_Y_Q]);
        const int64_t ux = unit_lib::tile_of(row[UNIT_COL_X_Q]);
        const int64_t dy = uy - sy, dx = ux - sx;
        if (dy * dy + dx * dx > r2) continue;
        if (fac != -1 && row[UNIT_COL_TEAM] != fac) continue;
        if (row[UNIT_COL_FOOTPRINT] < fp_min) continue;
        if (los) {
            bool bad = false;
            const bool seen = unit_sight::has_los(
                reinterpret_cast<const uint8_t*>(E.solid),
                reinterpret_cast<const uint8_t*>(E.sight_screen), E.h, E.w, sy, sx, uy, ux,
                &bad);
            if (bad) return false;
            if (!seen) continue;                 // the plate is the Bresenham origin
        }
        ++count;
    }
    *v = count;
    return true;
}

// (a) the emit, then the sensors.
__device__ inline void emit(const topo::TopoEnv& E, const int64_t* tab, const Tables& T,
                            int64_t tick, const int64_t* store, int n_units, int64_t* B) {
    const int64_t OPEN = tab[UNIT_TAB_DOOR_STATES + 1];
    for (int64_t j = 0; j < T.n_bus; ++j) {
        const int64_t* r = bus_row(T, j);
        const int64_t code = r[OBJ_SIGNAL_COL_EMIT], ref = r[OBJ_SIGNAL_COL_EMIT_REF];
        int64_t v = 0;
        if (code == SIGNAL_EMIT_NONE) {
            continue;
        } else if (code == SIGNAL_EMIT_DOOR_OPEN || code == SIGNAL_EMIT_DOOR_ALIVE) {
            if (ref < 0 || ref >= T.n_doors) {
                B[APPLIER_BOOK_FAULTS] += 1;
                continue;
            }
            const int64_t* d = T.doors + (size_t)ref * OBJ_DOOR_W;
            v = code == SIGNAL_EMIT_DOOR_OPEN ? (d[OBJ_DOOR_COL_STATE] == OPEN ? 1 : 0)
                                              : (d[OBJ_DOOR_COL_ALIVE] != 0 ? 1 : 0);
        } else if (code == SIGNAL_EMIT_CONSTANT) {
            v = ref;
        } else {
            B[APPLIER_BOOK_FAULTS] += 1;
            continue;
        }
        set_pub(T, j, v, B);
    }
    for (int64_t k = 0; k < T.n_sensors; ++k) {
        const int64_t* s = T.sensors + (size_t)k * OBJ_SENSOR_W;
        const int64_t slot = s[OBJ_SENSOR_COL_SLOT];
        if (!slot_ok(T, slot)) {
            B[APPLIER_BOOK_FAULTS] += 1;
            continue;
        }
        int64_t v = 0;
        bool ok = true;
        if (s[OBJ_SENSOR_COL_ALIVE] != 0) {      // a dead sensor publishes 0 (D13)
            const int64_t kind = s[OBJ_SENSOR_COL_KIND];
            if (kind == SIGNAL_SENSOR_FIELD) {
                ok = field_value(E, T, s, &v);
            } else if (kind == SIGNAL_SENSOR_CLOCK) {
                const int64_t p = s[OBJ_SENSOR_COL_PERIOD];
                ok = p >= 1;
                if (ok) v = fixedpoint::floordiv_q(tick, p);
            } else if (kind == SIGNAL_SENSOR_MOTION) {
                ok = motion_value(E, s, store, n_units, &v);
            } else {
                ok = false;
            }
        }
        if (!ok) {
            B[APPLIER_BOOK_FAULTS] += 1;
            continue;
        }
        set_pub(T, slot, v, B);
    }
}

// The driving slots of node k's `in` (its side span); false: outside.
__device__ __forceinline__ bool node_ins(const Tables& T, const int64_t* n, const int64_t** ins,
                                         int64_t* cnt) {
    const int64_t a = n[OBJ_NODE_COL_IN_START], c = n[OBJ_NODE_COL_IN_COUNT];
    if (a < 0 || c < 0 || a + c > T.n_nins) return false;
    *ins = T.nins + (size_t)a * OBJ_NODE_IN_W;
    *cnt = c;
    return true;
}
__device__ __forceinline__ int64_t in_slot(const int64_t* ins, int64_t j) {
    return ins[(size_t)j * OBJ_NODE_IN_W + OBJ_NODE_IN_COL_SLOT];
}

__device__ __forceinline__ bool compare(int64_t code, int64_t a, int64_t b, bool* bad) {
    if (code == SIGNAL_CMP_GT) return a > b;
    if (code == SIGNAL_CMP_GE) return a >= b;
    if (code == SIGNAL_CMP_LT) return a < b;
    if (code == SIGNAL_CMP_LE) return a <= b;
    if (code == SIGNAL_CMP_EQ) return a == b;
    if (code == SIGNAL_CMP_NE) return a != b;
    *bad = true;
    return false;
}

// The head of the LOGIC stage: every slot a node names inside the bus, every
// span inside its side table, every filter and interior airlock inside the
// proved range. Returns 0 ok, 1 a fault, 2 a range refusal.
__device__ inline int logic_head(const Tables& T) {
    for (int64_t k = 0; k < T.n_nodes; ++k) {
        const int64_t* n = T.nodes + (size_t)k * OBJ_NODE_W;
        const int64_t kind = n[OBJ_NODE_COL_KIND];
        const int64_t* ins = nullptr;
        int64_t c = 0;
        if (!node_ins(T, n, &ins, &c)) return 1;
        for (int64_t j = 0; j < c; ++j)
            if (!slot_ok(T, in_slot(ins, j))) return 1;
        if (kind == SIGNAL_NODE_AIRLOCK_CONTROLLER) {
            for (int j = 0; j < OBJ_NODE_COL_A_OUT_W; ++j)
                if (!slot_ok(T, n[OBJ_NODE_COL_A_OUT + j])) return 1;
            for (int j = 0; j < OBJ_NODE_COL_A_IN_W; ++j) {
                const int64_t s = n[OBJ_NODE_COL_A_IN + j];
                if (s != -1 && !slot_ok(T, s)) return 1;
            }
            if (n[OBJ_NODE_COL_INTERIOR] != 0) {
                for (int j = 7; j < 10; ++j) {       // chamber_p, inner_side_p, outer_side_p
                    const int64_t s = n[OBJ_NODE_COL_A_IN + j];
                    const int64_t p = s == -1 ? 0 : pub(T, s);
                    if (p < -PRESS_LIM || p > PRESS_LIM) return 1;
                }
                const int64_t band = n[OBJ_NODE_COL_BAND];
                if (band < -PRESS_LIM || band > PRESS_LIM) return 1;
            }
            continue;
        }
        if (!slot_ok(T, n[OBJ_NODE_COL_OUT])) return 1;
        if (kind == SIGNAL_NODE_DECIDER && n[OBJ_NODE_COL_REQUIRE_ALIVE] != 0) {
            const int64_t a = n[OBJ_NODE_COL_ALIVE_SLOT];
            if (a != -1 && !slot_ok(T, a)) return 1;
        }
        if (kind == SIGNAL_NODE_FILTER) {
            const int64_t kk = n[OBJ_NODE_COL_K], ema = n[OBJ_NODE_COL_EMA];
            const int64_t x = c > 0 ? pub(T, in_slot(ins, 0)) : 0;
            if (kk < 0 || kk > FILTER_K_MAX || ema < -FILTER_LIM || ema > FILTER_LIM ||
                x < -FILTER_LIM || x > FILTER_LIM)
                return 2;
        }
    }
    return 0;
}

__device__ __forceinline__ void pump_dir(int64_t cmd, int64_t* inj, int64_t* ext) {
    *inj = cmd > 0 ? 1 : 0;
    *ext = cmd < 0 ? 1 : 0;
}

// AirlockControllerRuntime._next_state.
__device__ inline int64_t next_state(int64_t s, int64_t presence, int64_t inner_open,
                                     int64_t outer_open, int64_t inner_alive,
                                     int64_t outer_alive, int64_t at_far, int64_t at_near,
                                     int64_t far_open, int64_t far_alive) {
    if (s == SIGNAL_AIRLOCK_IDLE)
        return presence >= 1 ? SIGNAL_AIRLOCK_CLOSING : SIGNAL_AIRLOCK_IDLE;
    if (s == SIGNAL_AIRLOCK_CLOSING) {
        if (inner_alive == 0 || outer_alive == 0) return SIGNAL_AIRLOCK_FAULT;
        if (inner_open == 0 && outer_open == 0) return SIGNAL_AIRLOCK_EQUALIZE;
        return SIGNAL_AIRLOCK_CLOSING;
    }
    if (s == SIGNAL_AIRLOCK_EQUALIZE)
        return at_far != 0 ? SIGNAL_AIRLOCK_OPEN_FAR : SIGNAL_AIRLOCK_EQUALIZE;
    if (s == SIGNAL_AIRLOCK_OPEN_FAR)
        return presence == 0 ? SIGNAL_AIRLOCK_RESEAL : SIGNAL_AIRLOCK_OPEN_FAR;
    if (s == SIGNAL_AIRLOCK_RESEAL) {
        if (far_alive == 0) return SIGNAL_AIRLOCK_FAULT;
        return far_open == 0 ? SIGNAL_AIRLOCK_REPRESSURIZE : SIGNAL_AIRLOCK_RESEAL;
    }
    if (s == SIGNAL_AIRLOCK_REPRESSURIZE)
        return at_near != 0 ? SIGNAL_AIRLOCK_IDLE : SIGNAL_AIRLOCK_REPRESSURIZE;
    return SIGNAL_AIRLOCK_FAULT;                 // FAULT is latched until reset
}

// AirlockControllerRuntime.evaluate on node row n.
__device__ inline void airlock(const Tables& T, int64_t* n, int64_t* B) {
    const int64_t* ai = n + OBJ_NODE_COL_A_IN;
    const int64_t* ao = n + OBJ_NODE_COL_A_OUT;
    auto rd = [&](int j) -> int64_t { return ai[j] == -1 ? 0 : pub(T, ai[j]); };
    const int64_t presence = rd(0), inner_open = rd(1), outer_open = rd(2);
    const int64_t inner_alive = rd(3), outer_alive = rd(4);
    int64_t at_far = rd(5), at_near = rd(6);
    const bool far_outer = n[OBJ_NODE_COL_FAR_IS_OUTER] != 0;
    const bool interior = n[OBJ_NODE_COL_INTERIOR] != 0;
    const int64_t far_open = far_outer ? outer_open : inner_open;
    const int64_t far_alive = far_outer ? outer_alive : inner_alive;
    int64_t far_dir = 0, near_dir = 0;
    if (interior) {
        const int64_t p_ch = rd(7), p_in = rd(8), p_out = rd(9);
        const int64_t p_far = far_outer ? p_out : p_in, p_near = far_outer ? p_in : p_out;
        const int64_t d_far = p_far - p_ch, d_near = p_near - p_ch;
        const int64_t band = n[OBJ_NODE_COL_BAND];
        at_far = (d_far < 0 ? -d_far : d_far) <= band ? 1 : 0;
        at_near = (d_near < 0 ? -d_near : d_near) <= band ? 1 : 0;
        far_dir = at_far ? 0 : (d_far > 0 ? 1 : -1);
        near_dir = at_near ? 0 : (d_near > 0 ? 1 : -1);
    }
    const int64_t prev = n[OBJ_NODE_COL_STATE];
    const int64_t nxt = next_state(prev, presence, inner_open, outer_open, inner_alive,
                                   outer_alive, at_far, at_near, far_open, far_alive);
    put(n + OBJ_NODE_COL_DWELL, nxt == prev ? n[OBJ_NODE_COL_DWELL] + 1 : 0, B);
    put(n + OBJ_NODE_COL_STATE, nxt, B);
    int64_t far_open_cmd = 0, far_close = 0, near_open_cmd = 0, near_close = 0;
    int64_t inject = 0, extract = 0, busy = 0, near_inject = 0, near_extract = 0;
    const int64_t eq = n[OBJ_NODE_COL_EQUALIZE_CMD];
    if (nxt == SIGNAL_AIRLOCK_IDLE) {
        near_open_cmd = 1;
        far_close = 1;
    } else if (nxt == SIGNAL_AIRLOCK_CLOSING || nxt == SIGNAL_AIRLOCK_RESEAL) {
        near_close = far_close = busy = 1;
    } else if (nxt == SIGNAL_AIRLOCK_EQUALIZE) {
        near_close = far_close = busy = 1;
        pump_dir(interior ? far_dir : eq, &inject, &extract);
    } else if (nxt == SIGNAL_AIRLOCK_OPEN_FAR) {
        near_close = far_open_cmd = busy = 1;
    } else if (nxt == SIGNAL_AIRLOCK_REPRESSURIZE) {
        near_close = far_close = busy = 1;
        if (interior)
            pump_dir(near_dir, &near_inject, &near_extract);
        else
            pump_dir(-eq, &inject, &extract);
    }
    const int64_t inner_open_o = far_outer ? near_open_cmd : far_open_cmd;
    const int64_t inner_close_o = far_outer ? near_close : far_close;
    const int64_t outer_open_o = far_outer ? far_open_cmd : near_open_cmd;
    const int64_t outer_close_o = far_outer ? far_close : near_close;
    // AIRLOCK_OUTPUTS' order
    set_stg(T, ao[0], inner_close_o, B);
    set_stg(T, ao[1], inner_open_o, B);
    set_stg(T, ao[2], outer_close_o, B);
    set_stg(T, ao[3], outer_open_o, B);
    set_stg(T, ao[4], inject, B);
    set_stg(T, ao[5], extract, B);
    set_stg(T, ao[6], busy, B);
    set_stg(T, ao[7], near_inject, B);
    set_stg(T, ao[8], near_extract, B);
}

// (b) the node sweep.
__device__ inline void logic(const Tables& T, int64_t* B) {
    const int head = logic_head(T);
    if (head == 1) {
        B[APPLIER_BOOK_FAULTS] += 1;
        return;
    }
    if (head == 2) {
        B[APPLIER_BOOK_SIGNAL_REFUSED] += 1;
        return;
    }
    for (int64_t k = 0; k < T.n_nodes; ++k) {
        int64_t* n = T.nodes + (size_t)k * OBJ_NODE_W;
        const int64_t kind = n[OBJ_NODE_COL_KIND];
        if (kind == SIGNAL_NODE_AIRLOCK_CONTROLLER) {
            airlock(T, n, B);
            continue;
        }
        const int64_t* ins = nullptr;
        int64_t c = 0;
        node_ins(T, n, &ins, &c);
        const int64_t out = n[OBJ_NODE_COL_OUT];
        const int64_t single = c > 0 ? pub(T, in_slot(ins, 0)) : 0;   // INPUT_SINGLE
        if (kind == SIGNAL_NODE_DECIDER) {
            bool bad = false;
            int64_t res = compare(n[OBJ_NODE_COL_CMP], single, n[OBJ_NODE_COL_THRESHOLD], &bad)
                              ? 1 : 0;
            if (bad) {
                B[APPLIER_BOOK_FAULTS] += 1;
                continue;
            }
            if (n[OBJ_NODE_COL_REQUIRE_ALIVE] != 0) {   // fail-passive (D4)
                const int64_t a = n[OBJ_NODE_COL_ALIVE_SLOT];
                if (a == -1 || pub(T, a) == 0) res = 0;
            }
            set_stg(T, out, res, B);
        } else if (kind == SIGNAL_NODE_GATE_AND) {
            int64_t res = c > 0 ? 1 : 0;              // INPUT_AND: empty => 0
            for (int64_t j = 0; j < c; ++j)
                if (pub(T, in_slot(ins, j)) == 0) res = 0;
            set_stg(T, out, res, B);
        } else if (kind == SIGNAL_NODE_GATE_OR) {
            int64_t res = 0;                          // INPUT_HELD
            for (int64_t j = 0; j < c; ++j)
                if (pub(T, in_slot(ins, j)) != 0) res = 1;
            set_stg(T, out, res, B);
        } else if (kind == SIGNAL_NODE_GATE_NOT) {
            set_stg(T, out, single != 0 ? 0 : 1, B);
        } else if (kind == SIGNAL_NODE_FILTER) {
            const int64_t kk = n[OBJ_NODE_COL_K];
            const int64_t half = kk >= 1 ? ((int64_t)1 << (kk - 1)) : 0;
            const int64_t d = (int64_t)1 << kk;
            const int64_t ema = n[OBJ_NODE_COL_EMA];
            const int64_t value = fixedpoint::floordiv_q(ema + half, d);
            const int64_t ema2 = ema + single - value;
            put(n + OBJ_NODE_COL_EMA, ema2, B);
            set_stg(T, out, fixedpoint::floordiv_q(ema2 + half, d), B);
        } else {
            B[APPLIER_BOOK_FAULTS] += 1;
        }
    }
}

// (c) the wired doors' want_open (close beats open; neither keeps the latch).
__device__ inline void door_inputs(const Tables& T, int64_t* B) {
    for (int64_t j = 0; j < T.n_drives; ++j) {
        const int64_t* r = T.drives + (size_t)j * OBJ_DOOR_DRIVE_W;
        if (r[OBJ_DOOR_DRIVE_COL_DOOR] < 0 || r[OBJ_DOOR_DRIVE_COL_DOOR] >= T.n_doors ||
            !slot_ok(T, r[OBJ_DOOR_DRIVE_COL_SLOT])) {
            B[APPLIER_BOOK_FAULTS] += 1;
            return;                                  // nothing of the env written
        }
    }
    for (int64_t k = 0; k < T.n_doors; ++k) {
        bool wired = false, close_on = false, open_on = false;
        for (int64_t j = 0; j < T.n_drives; ++j) {
            const int64_t* r = T.drives + (size_t)j * OBJ_DOOR_DRIVE_W;
            if (r[OBJ_DOOR_DRIVE_COL_DOOR] != k) continue;
            wired = true;
            const bool on = pub(T, r[OBJ_DOOR_DRIVE_COL_SLOT]) != 0;
            if (r[OBJ_DOOR_DRIVE_COL_CLOSE] != 0) close_on = close_on || on;
            else open_on = open_on || on;
        }
        if (!wired) continue;                        // unwired door: its own latch (D3)
        int64_t* d = T.doors + (size_t)k * OBJ_DOOR_W;
        if (close_on)
            put(d + OBJ_DOOR_COL_WANT_OPEN, 0, B);
        else if (open_on)
            put(d + OBJ_DOOR_COL_WANT_OPEN, 1, B);
    }
}

// (d) the pump sweep: the flux through the seam, then the latch.
__device__ inline void pumps(const topo::TopoEnv& E, const int64_t* tab, const Tables& T,
                             int64_t* B) {
    for (int64_t k = 0; k < T.n_pumps; ++k) {
        int64_t* p = T.pumps + (size_t)k * OBJ_PUMP_W;
        if (p[OBJ_PUMP_COL_ALIVE] == 0) continue;    // a dead pump's latch is frozen
        const int64_t a = p[OBJ_PUMP_COL_IN_START], c = p[OBJ_PUMP_COL_IN_COUNT];
        const int64_t at_slot = p[OBJ_PUMP_COL_AT_SLOT];
        if (a < 0 || c < 0 || a + c > T.n_pins || (at_slot != -1 && !slot_ok(T, at_slot))) {
            B[APPLIER_BOOK_FAULTS] += 1;
            continue;
        }
        bool inject = false, extract = false, bad = false;
        for (int64_t j = 0; j < c; ++j) {
            const int64_t* r = T.pins + (size_t)(a + j) * OBJ_PUMP_IN_W;
            const int64_t s = r[OBJ_PUMP_IN_COL_SLOT];
            if (!slot_ok(T, s)) { bad = true; break; }
            const bool on = pub(T, s) != 0;
            if (r[OBJ_PUMP_IN_COL_EXTRACT] != 0) extract = extract || on;
            else inject = inject || on;
        }
        const int64_t py = p[OBJ_PUMP_COL_PORT_Y], px = p[OBJ_PUMP_COL_PORT_X];
        if (bad || !topo::in_grid64(E, py, px)) {
            B[APPLIER_BOOK_FAULTS] += 1;
            continue;
        }
        const int64_t dn = p[OBJ_PUMP_COL_DELTA_N_Q];
        int64_t res = 0;
        if (p[OBJ_PUMP_COL_HAS_SOURCE] != 0) {       // feed = "transfer": a move
            const int64_t sy = p[OBJ_PUMP_COL_SRC_Y], sx = p[OBJ_PUMP_COL_SRC_X];
            int64_t moved = 0;
            if (extract)                              // extract beats inject (safe state)
                res = topo::transfer_gas_n(E, tab, py, px, sy, sx, dn, &moved, B);
            else if (inject)
                res = topo::transfer_gas_n(E, tab, sy, sx, py, px, dn, &moved, B);
        } else if (extract) {
            int64_t remove[topo::GAS_MAX];
            int64_t taken = 0;
            res = topo::extract_gas_n_vec(E, tab, py, px, dn, remove, &taken, B);
        } else if (inject) {
            res = topo::inject_gas_n(E, tab, py, px, dn, B);
        }
        if (res != 0) {
            B[APPLIER_BOOK_FAULTS] += 1;             // the reference would raise
            continue;
        }
        if (inject || extract) B[APPLIER_BOOK_APPLIED] += 1;
        // at_target: the port pressure against the target, latched Schmitt
        int64_t err = (int64_t)E.atmosphere[(int)py * E.w + (int)px] - p[OBJ_PUMP_COL_TARGET];
        if (err < 0) err = -err;
        int64_t at = p[OBJ_PUMP_COL_AT_TARGET];
        if (at != 0) {
            if (err > p[OBJ_PUMP_COL_RELEASE]) at = 0;
        } else {
            if (err <= p[OBJ_PUMP_COL_BAND]) at = 1;
        }
        put(p + OBJ_PUMP_COL_AT_TARGET, at, B);
        if (at_slot != -1) set_pub(T, at_slot, at, B);
    }
}

// (e) pub <- stg on the node slots.
__device__ inline void swap(const Tables& T, int64_t* B) {
    for (int64_t j = 0; j < T.n_bus; ++j) {
        int64_t* r = bus_row(T, j);
        if (r[OBJ_SIGNAL_COL_NODE] != 0) put(r + OBJ_SIGNAL_COL_PUB, r[OBJ_SIGNAL_COL_STG], B);
    }
}

}  // namespace signals
}  // namespace breach_cuda
