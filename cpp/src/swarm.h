#pragma once
// ============================================================================
// swarm.h -- THE SWARM SHARED KIT (arc #63 P3, docs/swarm_P3_impl.md §3, §4;
// executes docs/architecture/engine/17_swarm_units.md §4 and §9b's "written
// once" row).
// ============================================================================
//
// Species-agnostic: the store's C twins, the heading wrap, the field-generic
// tile reader and head-sweep sampler, the side preference, the turn draw, the
// wall probe, the store / launch types, the CPU loop template and the
// species-law registry type. A species law (swarm_larva.h) is ONE FP_HD
// per-slot function wrapped in a functor; its CPU entry, its per-call CUDA
// entry and its resident CUDA entry are loops around that one functor
// (host_loop below, swarm_launch.cuh), never a copy.
//
// Rules this header keeps (the Swarm shared kit's, .claude/rules/swarm.md):
// - pure integer: Q16.16 through cpp/src/fixed_point.h, draws only through
//   the vendored Philox (philox32.h), no line names a non-integer type;
// - no std:: helper in an FP_HD function (nvcc runs without
//   --expt-relaxed-constexpr), hand-written ternaries as fixed_point.h does;
// - a blocked sweep point (solid or off the grid) is never a reading;
// - every move goes through probe_move (out-of-grid and solid = blocked, no
//   sliding, step < 1 tile, so a move never skips a tile).
//
// Credit: the head sweep after Luo L, Gershow M, Rosenzweig M, Kang K,
// Fang-Yen C, Garrity PA, Samuel ADT, "Navigational decision making in
// Drosophila thermotaxis", J Neurosci 2010;30(12):4261-72,
// doi:10.1523/JNEUROSCI.4090-09.2010 (the two-point comparison is a discrete
// stand-in for one head sweep, not a claim that larvae compute a spatial
// gradient). Philox-4x32-10: Salmon, Moraes, Dror, Shaw, SC'11,
// doi:10.1145/2063384.2063405 (philox32.h).
#include <cstdint>
#include <vector>

#include "fixed_point.h"
#include "philox32.h"

namespace swarm {

// ---- the store's twins (src/simulation/swarm.py; identity-tested) ---------
constexpr int32_t AI_DEAD = 0, AI_RUN = 1, AI_EAT = 2, AI_COMA = 3;
constexpr int32_t FIRST_UNIT_ID = 1;
// The kit's narrowed Q16.16 pi (swarm_fixed.PI_Q16).
constexpr int32_t PI_Q16 = 205887;
static_assert(PI_Q16 == ((fixedpoint::PI_Q30 + (1 << 13)) >> 14),
              "PI_Q16 must be the trig kit's narrowed pi");
// The heading WRAP period, 2 * PI_Q16 = 411774 -- deliberately NOT
// PHILOX32_TWO_PI_Q16 (411775, the draw range): draw, then wrap.
constexpr int64_t HEADING_PERIOD_Q16 = 2 * (int64_t)PI_Q16;

// The ROSTER, column for column (swarm.ROSTER order).
constexpr int N_COLUMNS = 9;
using pos_x_t = int32_t;
using pos_y_t = int32_t;
using heading_t = int32_t;
using hp_t = int32_t;
using ai_state_t = int32_t;
using dist_walked_t = uint32_t;
using unit_id_t = int32_t;
using faction_t = int32_t;
using valid_t = uint8_t;
static_assert(sizeof(pos_x_t) == 4 && sizeof(pos_y_t) == 4 && sizeof(heading_t) == 4 &&
              sizeof(hp_t) == 4 && sizeof(ai_state_t) == 4 && sizeof(dist_walked_t) == 4 &&
              sizeof(unit_id_t) == 4 && sizeof(faction_t) == 4 && sizeof(valid_t) == 1,
              "the swarm roster's C types must match swarm.ROSTER's itemsizes");

// The (N, ENV_WORDS) env block Python passes: per env (draws.tick & M32,
// key lo, key hi).
enum : int { ENV_TICK = 0, ENV_SEED_LO = 1, ENV_SEED_HI = 2, ENV_WORDS = 3 };
// The per-env launch block the engine composes (C++-internal).
enum : int { LAUNCH_TICK = 0, LAUNCH_SEED_LO = 1, LAUNCH_SEED_HI = 2, LAUNCH_HW = 3,
             LAUNCH_FIXED_WORDS = 4 };
// One per-slot death record (zeros = no death this launch).
enum : int { DEATH_UNIT_ID = 0, DEATH_POS_X = 1, DEATH_POS_Y = 2, DEATH_HEADING = 3,
             DEATH_WORDS = 4 };
// One compacted death row, (species, env, slot) order.
enum : int { ROW_SPECIES = 0, ROW_ENV = 1, ROW_SLOT = 2, ROW_UNIT_ID = 3, ROW_POS_X = 4,
             ROW_POS_Y = 5, ROW_HEADING = 6, ROW_WORDS = 7 };

// ---- the heading wrap (swarm_fixed.wrap_heading_q16's twin) ---------------
// Floor-mod in int64, because C++ `%` truncates. `neg` (nullable) reports the
// negative-remainder branch (the law's branch tally).
FP_HD inline int32_t wrap_heading_q16(int64_t h, bool* neg = nullptr) {
    int64_t r = ((int64_t)PI_Q16 - h) % HEADING_PERIOD_Q16;
    const bool is_neg = r < 0;
    if (is_neg) r += HEADING_PERIOD_Q16;
    if (neg) *neg = is_neg;
    return (int32_t)((int64_t)PI_Q16 - r);
}

FP_HD inline int64_t abs64(int64_t v) { return (v < 0) ? -v : v; }

// ---- the tile lookup (never reads) ----------------------------------------
// tx = x >> 16, ty = y >> 16 (an arithmetic shift: floor, so -0.5 is -1).
FP_HD inline bool tile_of(int32_t x_q, int32_t y_q, int h, int w, int* idx) {
    const int32_t tx = x_q >> 16;
    const int32_t ty = y_q >> 16;
    const bool in = tx >= 0 && tx < w && ty >= 0 && ty < h;
    *idx = in ? ty * w + tx : 0;
    return in;
}

// ---- the field reader (RAW, ruling 11), field-generic ----------------------
// Any int32 per-tile field (temperature now, food at P6). Precondition: the
// point is in the grid (the own position by the kernel contract, an open
// sweep point by tile_of).
struct TileReader {
    const int32_t* f;
    int w;
    FP_HD int32_t operator()(int32_t x_q, int32_t y_q) const {
        return f[(y_q >> 16) * w + (x_q >> 16)];
    }
};

enum : int32_t { SIDE_MINUS = -1, SIDE_NONE = 0, SIDE_PLUS = 1 };

struct SweepSample {
    int32_t here;                  // R(px, py)
    bool open_plus, open_minus;    // in the grid and not solid
    bool in_plus, in_minus;        // in the grid (the branch tally's solid / edge split)
    int32_t val_plus, val_minus;   // R(point) iff open, else 0 (never read)
};

// One head sweep: "here" and two points at distance `dist_q` ahead, at
// heading +- angle (PLUS first, then MINUS). The adds stay int32: a canonical
// |heading| <= PI_Q16 and angle <= PI_Q16 / 2 + 1 keep |a| far inside the
// trig kit's pinned +-4 pi band.
template <class R>
FP_HD inline SweepSample head_sweep(const R& read, const bool* solid, int h, int w,
                                    int32_t px, int32_t py, int32_t heading,
                                    int32_t angle_q, int32_t dist_q) {
    SweepSample s;
    s.here = read(px, py);
    {
        const int32_t a = heading + angle_q;
        const int32_t sx = px + fixedpoint::narrow_round_signed(
                                    fixedpoint::mul_wide(fixedpoint::cos_q16(a), dist_q));
        const int32_t sy = py + fixedpoint::narrow_round_signed(
                                    fixedpoint::mul_wide(fixedpoint::sin_q16(a), dist_q));
        int idx;
        s.in_plus = tile_of(sx, sy, h, w, &idx);
        s.open_plus = s.in_plus && !solid[idx];
        s.val_plus = s.open_plus ? read(sx, sy) : 0;
    }
    {
        const int32_t a = heading - angle_q;
        const int32_t sx = px + fixedpoint::narrow_round_signed(
                                    fixedpoint::mul_wide(fixedpoint::cos_q16(a), dist_q));
        const int32_t sy = py + fixedpoint::narrow_round_signed(
                                    fixedpoint::mul_wide(fixedpoint::sin_q16(a), dist_q));
        int idx;
        s.in_minus = tile_of(sx, sy, h, w, &idx);
        s.open_minus = s.in_minus && !solid[idx];
        s.val_minus = s.open_minus ? read(sx, sy) : 0;
    }
    return s;
}

// Comfort distances: smaller is better. Open beats blocked; two blocked or
// two equal = no preference.
FP_HD inline int32_t side_preference(bool op, int64_t dp, bool om, int64_t dm) {
    if (op && !om) return SIDE_PLUS;
    if (!op && om) return SIDE_MINUS;
    if (!op && !om) return SIDE_NONE;
    return (dp < dm) ? SIDE_PLUS : (dm < dp) ? SIDE_MINUS : SIDE_NONE;
}

// How a turn chose its side (the branch tally): by the coin (no preference),
// toward the preferred side, away from it.
enum : int { TURN_BY_COIN = 0, TURN_TOWARD = 1, TURN_AWAY = 2 };

// The turn draw (tumble AND wall turn). w_side, w_mag: two Philox words. The
// side: a fair coin with no preference, else the preferred side with
// probability bias_q / 65536. The magnitude: uniform in [turn_min, turn_max]
// through philox32_below (never `%`). `how` and `wrap_neg` are nullable.
FP_HD inline int32_t turn_heading(int32_t heading, int32_t pref, uint32_t w_side,
                                  uint32_t w_mag, int32_t bias_q,
                                  int32_t turn_min_q, int32_t turn_max_q,
                                  int* how = nullptr, bool* wrap_neg = nullptr) {
    const uint32_t u = philox32_uniform_q16(w_side);            // 0..65535
    int32_t side;
    int k;
    if (pref == SIDE_NONE) {
        side = (u < 32768u) ? SIDE_PLUS : SIDE_MINUS;
        k = TURN_BY_COIN;
    } else if (u < (uint32_t)bias_q) {
        side = pref;
        k = TURN_TOWARD;
    } else {
        side = -pref;
        k = TURN_AWAY;
    }
    if (how) *how = k;
    const uint32_t span = (uint32_t)(turn_max_q - turn_min_q) + 1u;   // >= 1 by rule R4
    const int32_t m = turn_min_q + (int32_t)philox32_below(w_mag, span);
    return wrap_heading_q16((int64_t)heading + (int64_t)side * (int64_t)m, wrap_neg);
}

// The wall probe (engine/17 §4.1 steps 2-4): the target is the own tile or
// one of its 8 neighbours (|dx|, |dy| <= step_q < 1 tile); out of the grid or
// solid is blocked. No sliding.
struct Probe {
    bool blocked;
    bool in;          // the target is in the grid (the tally's solid / edge split)
    int32_t nx, ny;
};
FP_HD inline Probe probe_move(const bool* solid, int h, int w, int32_t px, int32_t py,
                              int32_t heading, int32_t step_q) {
    const int32_t dx = fixedpoint::narrow_round_signed(
        fixedpoint::mul_wide(fixedpoint::cos_q16(heading), step_q));
    const int32_t dy = fixedpoint::narrow_round_signed(
        fixedpoint::mul_wide(fixedpoint::sin_q16(heading), step_q));
    Probe p;
    p.nx = px + dx;
    p.ny = py + dy;
    int idx;
    p.in = tile_of(p.nx, p.ny, h, w, &idx);
    p.blocked = !p.in || solid[idx];
    return p;
}

// ---- the store / launch types ---------------------------------------------
// One species' (n_env, max_units) SoA, host OR device pointers.
struct StoreView {
    int32_t* pos_x = nullptr;
    int32_t* pos_y = nullptr;
    int32_t* heading = nullptr;
    int32_t* hp = nullptr;
    int32_t* ai_state = nullptr;
    uint32_t* dist_walked = nullptr;
    int32_t* unit_id = nullptr;
    int32_t* faction = nullptr;
    uint8_t* valid = nullptr;
};

// What one slot step reads and writes (host or device pointers). Every env's
// planes are (n_env, h, w); slot s of env e is flat index e * max_units + s.
struct SlotIO {
    StoreView st;
    int max_units = 0, h = 0, w = 0;
    const int32_t* T = nullptr;          // the field the law senses, (n_env, h, w)
    const bool* solid = nullptr;         // (n_env, h, w)
    const uint32_t* launch = nullptr;    // (n_env, LAUNCH_FIXED_WORDS)
    const int64_t* tables = nullptr;     // (n_env, table_width): the species' table rows
    int table_width = 0;
    int32_t* death = nullptr;            // (n_env, max_units, DEATH_WORDS)
    uint64_t* tally = nullptr;           // the CPU loop's branch tally; nullptr on device
};

// One law invocation. `launch_host` is the host copy of the launch block
// (the extents); on the CPU path every pointer is host.
struct LawIO {
    SlotIO io;
    int n_env = 0;
    const uint32_t* launch_host = nullptr;
};

// The resident launch core's invocation (device pointers in `io`).
struct ResidentLaunch {
    SlotIO io;
    int n_env = 0;
    int launch_hw = 0;                   // max_e hw[e]
    unsigned long long* counters = nullptr;   // (n_env, SWARM_CNT_SLOTS) in the books arena
    void* stream = nullptr;              // the tick stream
};

// The resident stage's per-env counters (a region of the books arena).
enum : int { SWARM_CNT_DEATHS = 0, SWARM_CNT_STEPPED = 1, SWARM_CNT_SLOTS = 2 };

// One species law: its name, its table width, the field + reduction it senses
// through (the coupling rows' labels), and its three entries.
struct SpeciesLaw {
    const char* name;
    int table_width;
    const char* field;
    const char* field_reduction;
    const char* const* branch_names;     // the CPU tally's branch names
    int n_branches;
    uint64_t* tally;                      // the process-global CPU tally
    int (*cpu)(const LawIO&);
    int (*cuda_percall)(const LawIO&);              // nullptr on the CPU build
    void (*cuda_resident)(const ResidentLaunch&);   // nullptr on the CPU build
};
extern const SpeciesLaw SPECIES_LAWS[];
extern const int N_SPECIES_LAWS;
// The registry index of `name`, or -1.
int species_law_index(const char* name);

// The CPU loop: for each env, slots [0, hw[e]); returns the death count.
template <class Law>
int host_loop(const Law& law, const LawIO& L) {
    int deaths = 0;
    for (int e = 0; e < L.n_env; ++e) {
        const int hw = (int)L.launch_host[e * LAUNCH_FIXED_WORDS + LAUNCH_HW];
        for (int s = 0; s < hw; ++s) deaths += law.step_slot(L.io, e, s);
    }
    return deaths;
}

// ---- the engine's host path (PhysicsEngine::step_swarm's body) ------------
// One species' host-path inputs: its (n_env, max_units) store, (n_env,) hw,
// (n_env, table_width) rows and the host-dirty flag the step sets after a
// launch (the host store moved, so any device copy is stale).
struct SpeciesStep {
    int law = -1;                // registry index
    StoreView st;
    int max_units = 0;
    const int32_t* hw = nullptr;
    const int64_t* tables = nullptr;
    int table_width = 0;
    uint8_t* host_dirty = nullptr;
};

// One species' resident inputs (ResidentTickArgs' swarm block): the device
// addresses of the batch's (N, max_units) columns (env 0's address is the
// base), the host hw and table rows.
struct ResidentSpecies {
    int law = -1;
    StoreView st;                // DEVICE pointers
    int max_units = 0;
    const int32_t* hw = nullptr;          // host (N,)
    const int64_t* tables = nullptr;      // host (N, table_width)
    int table_width = 0;
};

// The engine's per-call scratch (records, launch block), never freed.
struct StepScratch {
    std::vector<int32_t> records;
    std::vector<uint32_t> launch;
};

// The host-path swarm step: dormant (no species present in any env) returns
// at once with no launch and no scratch growth; else, per present species,
// the CPU loop (or the per-call CUDA wrapper while set_swarm_backend is on),
// the host-dirty flag set, and the death rows compacted in (species, env,
// slot) order into `rows` (flat, ROW_WORDS per row).
void step_all(const SpeciesStep* packs, int n_packs, const int32_t* T, const bool* solid,
              int h, int w, const uint32_t* env, int n_env, StepScratch& scratch,
              std::vector<int32_t>& rows);

// Compose the per-env launch block from the env words and hw.
void compose_launch(const uint32_t* env, const int32_t* hw, int n_env, uint32_t* out);

// The counted wrappers (process-global telemetry, once per invocation with
// launch_hw > 0): the CPU loop and the per-call CUDA wrapper.
int run_cpu(int law, const LawIO& L);
int run_cuda_percall(int law, const LawIO& L);
long long swarm_cpu_calls();
long long swarm_cuda_calls();
// The per-call CUDA backend flag (residency is the runner's switch).
void set_swarm_backend(bool use_cuda);
bool get_swarm_backend();

// Out of line in a strict TU, for the binding (bindings.cpp keeps the global fast-math flag and
// never calls an FP_HD law function inline).
int32_t wrap_heading_q16_host(int64_t h);
int32_t felt_tile_host(const int32_t* T, int h, int w, int32_t x_q, int32_t y_q);

#ifdef BREACH_HAS_CUDA
// swarm.cu: the per-call staging and the resident stage's helpers.
void note_cuda_call();
// Resident stage, per species: stage the launch block and the table rows on
// the tick stream, launch the core, and copy each env's [0, hw[e]) records
// into the species' pinned host buffer (before the tail's books sync).
// Returns the launch_hw (0: nothing launched).
int resident_species(int law, const StoreView& st_dev, int max_units, int n_env, int h,
                     int w, const int32_t* d_T, const bool* d_solid,
                     const uint32_t* launch_host, const int64_t* tables_host,
                     int table_width, unsigned long long* d_counters, void* stream);
// The species' pinned host records (n_env, max_units, DEATH_WORDS), valid
// after the tick's books sync for [0, hw[e]) per env.
const int32_t* resident_records_host(int law);
// TEST-ONLY (swarm_probe.cu): the resident launch core on caller-owned host
// arrays copied into fresh device arrays (padding and garbage included), then
// copied back WHOLE; returns the death count.
int resident_probe(int law, const LawIO& L);
#endif

}  // namespace swarm
