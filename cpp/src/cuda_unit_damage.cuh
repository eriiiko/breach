#pragma once
// ============================================================================
// #29 P6.3 -- THE DEVICE DAMAGE PIPELINE and what its slots share, beside the
// device unit library (cuda_unit_lib.cuh): damage.apply_packet (the packet
// door's mitigation, the exposure scale between the snap's halves, the hit
// points, the kill exactly once, every death clear, the exposure raise), the
// event sinks of the ring (cuda_unit_events.h), the felt temperature
// (exchange.felt_temperature_q -- the heat row's AND, from P8b, the
// observation's ONE read), the credit claims (damage.TickCredit as a per-tick
// table), the footprint reduction (exchange.reduce_max). Integer for integer
// the Python it names; a change lands in the Python first.
//
//   apply_packet(row, tab, packet, sink, changed, &killed, &bad) -> applied Q16
//
// works on ANY row the caller hands it (a slot of this patch hands the
// thread's own row; P6.4 / P7's serial applier hands the target's) and appends
// its events through the sink it is given: a per-row kernel's ROW sink (the
// staging, gathered in row order) or a serial producer's ENV sink (the ring
// itself, in its own order). It never writes outside the row and the sink.
// A packet or a row operand outside the range proved here sets *bad and
// writes NOTHING (the caller faults the row).
//
// Device-only header (CUDA types). CLASS T.
#include "cuda_unit_lib.cuh"
#include "cuda_unit_events.h"
#include "unit_events_layout.h"    // GENERATED: the ring's records, the vocabulary

namespace breach_cuda {

// What a kernel needs of the ring (built on the host from a UnitEventRing).
struct UnitRingView {
    int64_t* ring = nullptr;                // (N, E_max, F)
    int64_t* count = nullptr;               // (N,)
    unsigned long long* dropped = nullptr;  // (N,)
    int64_t* stage = nullptr;               // (N, U, ROW_CAP, F)
    int64_t* stage_count = nullptr;         // (N, U)
    int e_max = 0;
};

inline UnitRingView ring_view(UnitEventRing& r) {
    UnitRingView v;
    v.ring = r.ring_dev();
    v.count = r.count_dev();
    v.dropped = r.dropped_dev();
    v.stage = r.stage_dev();
    v.stage_count = r.stage_count_dev();
    v.e_max = r.e_max();
    return v;
}

namespace unit_lib {

using namespace unit_events;

//: unit_schema's life_state encoding of "dead".
constexpr int64_t LIFE_DEAD = 1;
constexpr int N_DT = UNIT_N_DAMAGE_TYPES;
//: The proved ranges: a packet's amount, a row's exposure level, a packet's
//: expose and armor penetration (Q16). Mitigation's product is then below
//: 2^66 / 2^16; the exposure gain below 2^62; the scale is taken in 128 bits.
constexpr int64_t AMOUNT_MAX = (int64_t)1 << 46;
constexpr int64_t EXPOSURE_MAX = (int64_t)1 << 40;
constexpr int64_t EXPOSE_MAX = (int64_t)1 << 40;
constexpr int64_t AP_MAX = (int64_t)1 << 40;
//: The largest footprint a reduction walks: the unit library's FOOTPRINT_MAX
//: (cuda_unit_lib.cuh, P6.2m), the one bound.

// ---- the event sinks ------------------------------------------------------------
struct EventSink {
    int64_t* recs;                 // the sink's first record
    int64_t cap;                   // its capacity (records)
    int64_t* count;                // its count (one writer)
    unsigned long long* dropped;   // the env's dropped count
};

// Row r of this block's env: its staging (gathered in row order).
__device__ __forceinline__ EventSink row_sink(const UnitRingView& v, int n_units, int r) {
    const size_t rr = (size_t)benv() * n_units + r;
    return EventSink{v.stage + rr * UNIT_EVENT_ROW_CAP * UNIT_EVENT_F, UNIT_EVENT_ROW_CAP,
                     v.stage_count + rr, v.dropped + benv()};
}
// This block's env's ring itself (a serial per-env producer: one writer).
__device__ __forceinline__ EventSink env_sink(const UnitRingView& v) {
    return EventSink{v.ring + (size_t)benv() * v.e_max * UNIT_EVENT_F, v.e_max,
                     v.count + benv(), v.dropped + benv()};
}

__device__ inline void emit(EventSink& s, int64_t kind, int64_t uid, int64_t dmg_q,
                            int64_t cause, int64_t attacker, int64_t dx, int64_t dy) {
    const int64_t n = *s.count;
    if (n >= s.cap) {
        atomicAdd(s.dropped, 1ULL);
        return;
    }
    int64_t* r = s.recs + n * UNIT_EVENT_F;
    r[UNIT_EV_KIND] = kind;
    r[UNIT_EV_UNIT_ID] = uid;
    r[UNIT_EV_DAMAGE_Q] = dmg_q;
    r[UNIT_EV_CAUSE] = cause;
    r[UNIT_EV_ATTACKER_ID] = attacker;
    r[UNIT_EV_DIR_X_Q] = dx;
    r[UNIT_EV_DIR_Y_Q] = dy;
    *s.count = n + 1;
}

// ---- the packet ------------------------------------------------------------------
// damage.DamagePacket at the pipeline's door: the amount already in Q16
// (DamagePacket.q()), the armor penetration already quantized (mitigate_q's
// door), the cause as its index in events.CAUSE_NAMES (apply_packet's
// `source`), the attacker as given (attacker_value normalizes it).
struct Packet {
    int64_t amount_q = 0;
    int64_t dtype = 0;
    int64_t attacker = -1;
    bool indirect = false;            // credit only (recent_attackers: host only)
    int64_t expose = 0;
    int64_t dir_x_q = 0, dir_y_q = 0; // render-only: rides the hit event
    int64_t ap_q = 0;
    int64_t cause = 0;
    bool mark_killed_by_zombie = false;
};

// events.attacker_value: a negative id is the environment.
__device__ __forceinline__ int64_t attacker_value(int64_t a) { return a >= 0 ? a : -1; }

// damage.mitigation_for: the zombie state's tables, else the species'.
__device__ inline bool mitigation_of(const int64_t* row, const int64_t* tab,
                                     const int64_t** armor, const int64_t** resist) {
    if (row[UNIT_COL_IS_ZOMBIE]) {
        *armor = tab + UNIT_TAB_MITIG_ARMOR_ZOMBIE_Q;
        *resist = tab + UNIT_TAB_MITIG_RESIST_ZOMBIE_Q;
        return true;
    }
    const int64_t sp = row[UNIT_COL_SPECIES_ID];
    if (sp < 0 || sp >= UNIT_TAB_MITIG_ARMOR_SPECIES_Q_W / N_DT) return false;
    *armor = tab + UNIT_TAB_MITIG_ARMOR_SPECIES_Q + sp * N_DT;
    *resist = tab + UNIT_TAB_MITIG_RESIST_SPECIES_Q + sp * N_DT;
    return true;
}

// damage.mitigate_q: max(0, a - max(0, armor - ap)) x resist, rounded once
// half away from zero; HEAL the identity.
__device__ inline int64_t mitigate_q(int64_t a, int64_t dtype, const int64_t* armor,
                                     const int64_t* resist, int64_t ap_q) {
    if (dtype == UNIT_DTYPE_HEAL) return a;
    int64_t eff = armor[dtype] - ap_q;
    if (eff < 0) eff = 0;
    a -= eff;
    if (a <= 0) return 0;
    return fixedpoint::mul_q16_round_q(a, resist[dtype]);
}

// exposure.resist_for: the zombie state's block, else the unit's environment
// (its env profile's, else its species').
__device__ inline const int64_t* expose_resist_of(const int64_t* row, const int64_t* tab) {
    if (row[UNIT_COL_IS_ZOMBIE]) return tab + UNIT_TAB_EXPOSE_RESIST_ZOMBIE_Q;
    const int64_t p = row[UNIT_COL_ENV_PROFILE_NAME];
    if (p >= 0) {
        if (p >= UNIT_TAB_EXPOSE_RESIST_PROFILE_Q_W / N_DT) return nullptr;
        return tab + UNIT_TAB_EXPOSE_RESIST_PROFILE_Q + p * N_DT;
    }
    const int64_t sp = row[UNIT_COL_SPECIES_ID];
    if (sp < 0 || sp >= UNIT_TAB_EXPOSE_RESIST_SPECIES_Q_W / N_DT) return nullptr;
    return tab + UNIT_TAB_EXPOSE_RESIST_SPECIES_Q + sp * N_DT;
}

// exposure.scale_q: dmg x (ONE + (k x >> 16)) >> 16 -- the identity at
// exposure 0, k 0, for HEAL and for dmg <= 0. The product in 128 bits; a
// result past int64 sets *bad.
__device__ inline int64_t scale_q(int64_t dmg, const int64_t* row, int64_t dtype,
                                  const int64_t* tab, bool* bad) {
    if (dtype == UNIT_DTYPE_HEAL || dmg <= 0) return dmg;
    const int64_t x = row[UNIT_COL_EXPOSURE + dtype];
    if (x <= 0) return dmg;
    const int64_t gain = (tab[UNIT_TAB_EXPOSE_K_Q] * x) >> 16;   // k <= 2^22, x <= 2^40
    if (gain <= 0) return dmg;
    uint64_t hi, lo;
    fixedpoint::umul64_wide((uint64_t)dmg, (uint64_t)(ONE + gain), &hi, &lo);
    const uint64_t q = (hi << 48) | (lo >> 16);
    if ((hi >> 16) != 0 || (q >> 63) != 0) {
        *bad = true;
        return 0;
    }
    return (int64_t)q;
}

// #29 P7a.1 -- THE EXPOSURE LANDING (exposure.ExposureLanding; design 4.4c
// decision 2): an attack slot's resolve holds every hit's exposure RAISE and
// every death's ZEROING back, so each packet of the slot reads its target's
// slot-start exposure, and lands them together after the last effect:
// raises first (each skipped when its unit is dead at the landing; each
// min(x_max, x + add) in record order, as _raise does), then the dead
// zeroed. Per env: `buf` (cap, LANDING_W) int64 -- a raise in columns
// (row, dtype, add) of slot n_raise, a death's row in column 3 of slot
// n_dead; a HIT adds at most one of each, so cap = the effect list's.
constexpr int LANDING_W = 4;
struct ExposureLanding {
    int64_t* buf = nullptr;
    int64_t cap = 0;
    int64_t n_raise = 0, n_dead = 0;
};

// exposure.raise_amount: expose x (ONE - resist[T]) >> 16, 0 while k_expose
// is 0 or for HEAL (a level that scales nothing is not kept).
__device__ __forceinline__ int64_t raise_amount(const int64_t* tab, const int64_t* xres,
                                                const Packet& p) {
    const int64_t k = tab[UNIT_TAB_EXPOSE_K_Q];
    if (p.expose <= 0 || p.dtype == UNIT_DTYPE_HEAL || k == 0) return 0;
    return (p.expose * (ONE - xres[p.dtype])) >> 16;
}

// damage.apply_packet on `row`: returns the APPLIED Q16 damage; *killed on
// the alive -> dead transition. Writes (all or nothing): the hp; the hit
// event; on the kill life_state, the death clears (exposure.on_death,
// unit_impulse.clear, horde.clear, unit_orders.clear, poison.clear; the
// conversion flag when the packet marks it) and the kill event; on a living
// target the exposure raise (exposure.on_hit). `landing` null: the raise
// and the death's zeroing at once (every caller but a resolve stage's HIT);
// non-null: both held in the landing for the walk's end (`r` the row's
// index in its env), which the CALLER chooses per (stage, kind) -- a packet
// never picks its mode.
__device__ inline int64_t apply_packet(int64_t* row, const int64_t* tab, const Packet& p,
                                       EventSink& sink, bool& changed, bool* killed,
                                       bool* bad, ExposureLanding* landing, int64_t r) {
    *killed = false;
    if (landing != nullptr &&
        (landing->n_raise >= landing->cap || landing->n_dead >= landing->cap)) {
        *bad = true;          // never with cap = the list's (one of each per HIT)
        return 0;
    }
    // -- the door: every operand checked before anything is written ----------
    if (p.dtype < 0 || p.dtype >= N_DT || p.cause < 0 || p.cause >= UNIT_N_CAUSES ||
        p.amount_q > AMOUNT_MAX || p.amount_q < -AMOUNT_MAX || p.expose < 0 ||
        p.expose > EXPOSE_MAX || p.ap_q < -AP_MAX || p.ap_q > AP_MAX) {
        *bad = true;
        return 0;
    }
    const int64_t x = row[UNIT_COL_EXPOSURE + p.dtype];
    if (x > EXPOSURE_MAX || x < -EXPOSURE_MAX) {
        *bad = true;
        return 0;
    }
    const int64_t* armor;
    const int64_t* resist;
    const int64_t* xres = expose_resist_of(row, tab);
    if (!mitigation_of(row, tab, &armor, &resist) || xres == nullptr) {
        *bad = true;
        return 0;
    }
    const bool was_alive = row_alive(row);
    const int64_t attacker = attacker_value(p.attacker);
    const int64_t amount = mitigate_q(p.amount_q, p.dtype, armor, resist, p.ap_q);
    bool b = false;
    const int64_t dmg = scale_q(amount, row, p.dtype, tab, &b);
    const int64_t hp = row[UNIT_COL_CURRENT_HP];
    if (b || (dmg > 0 && hp < INT64_MIN + dmg) || (dmg < 0 && hp > INT64_MAX + dmg)) {
        *bad = true;
        return 0;
    }
    // -- the writes ------------------------------------------------------------
    const int64_t new_hp = hp - dmg;
    put(row, UNIT_COL_CURRENT_HP, new_hp, changed);
    const int64_t uid = row[UNIT_COL_ID];
    emit(sink, UNIT_EVKIND_HIT, uid, dmg, p.cause, attacker, p.dir_x_q, p.dir_y_q);
    if (was_alive && new_hp <= 0) {
        put(row, UNIT_COL_LIFE_STATE, LIFE_DEAD, changed);
        if (landing != nullptr) {                                  // exposure.on_death
            landing->buf[landing->n_dead * LANDING_W + 3] = r;     //  (the landing's)
            landing->n_dead += 1;
        } else {
            for (int t = 0; t < UNIT_COL_EXPOSURE_W; ++t)          // exposure.clear
                put(row, UNIT_COL_EXPOSURE + t, 0, changed);
        }
        const int imp[] = {UNIT_COL_IMP_X, UNIT_COL_IMP_Y, UNIT_COL_IMP_L, UNIT_COL_STAND_X,
                           UNIT_COL_STAND_Y, UNIT_COL_SWX, UNIT_COL_SWY, UNIT_COL_PVX,
                           UNIT_COL_PVY, UNIT_COL_EFFORT, UNIT_COL_REACT_TICKS};
        for (int c : imp) put(row, c, 0, changed);                 // unit_impulse.clear
        put(row, UNIT_COL_ZOMBIE_NOISE_Y, -1, changed);            // horde.clear
        put(row, UNIT_COL_ZOMBIE_NOISE_X, -1, changed);
        put(row, UNIT_COL_ZOMBIE_NOISE_TICKS, 0, changed);
        put(row, UNIT_COL_ZOMBIE_ANCHOR_Y, -1, changed);
        put(row, UNIT_COL_ZOMBIE_ANCHOR_X, -1, changed);
        put(row, UNIT_COL_ORDER_WHAT, 0, changed);                 // unit_orders.clear
        put(row, UNIT_COL_ORDER_WHERE, -1, changed);
        put(row, UNIT_COL_ORDER_UNTIL, -1, changed);
        put(row, UNIT_COL_ORDER_TARGET, -1, changed);
        put(row, UNIT_COL_POISON_X0, 0, changed);                  // poison.clear
        put(row, UNIT_COL_POISON_X1, 0, changed);
        put(row, UNIT_COL_POISON_SRC, -1, changed);
        if (p.mark_killed_by_zombie) put(row, UNIT_COL_KILLED_BY_ZOMBIE, 1, changed);
        emit(sink, UNIT_EVKIND_KILLED, uid, 0, p.cause, attacker, 0, 0);
        *killed = true;
    }
    if (was_alive && landing != nullptr) {                         // exposure.on_hit
        const int64_t add = raise_amount(tab, xres, p);            //  (the landing's:
        if (add > 0) {                                             //  recorded for a
            int64_t* q = landing->buf + landing->n_raise * LANDING_W;   // unit this
            q[0] = r;                                              //  packet killed too,
            q[1] = p.dtype;                                        //  skipped at land)
            q[2] = add;
            landing->n_raise += 1;
        }
    } else if (was_alive && row_alive(row)) {                      // exposure.on_hit
        const int64_t add = raise_amount(tab, xres, p);
        if (add > 0) {
            const int64_t xm = tab[UNIT_TAB_EXPOSE_X_MAX_Q];
            const int64_t nx = x + add;
            put(row, UNIT_COL_EXPOSURE + p.dtype, nx < xm ? nx : xm, changed);
        }
    }
    return dmg;
}

// The immediate form (every caller outside a resolve stage).
__device__ inline int64_t apply_packet(int64_t* row, const int64_t* tab, const Packet& p,
                                       EventSink& sink, bool& changed, bool* killed,
                                       bool* bad) {
    return apply_packet(row, tab, p, sink, changed, killed, bad, nullptr, -1);
}

// ExposureLanding.land() over this block's env's rows (`store` (N, U, W),
// unit_row's rebase): the raises (a unit dead at the landing skipped), then
// the dead zeroed. `touched` (U,) marks each row a write changed.
__device__ inline void land_exposure(const ExposureLanding& L, int64_t* store, int n_units,
                                     const int64_t* tab, uint8_t* touched) {
    const int64_t xm = tab[UNIT_TAB_EXPOSE_X_MAX_Q];
    for (int64_t i = 0; i < L.n_raise; ++i) {
        const int64_t* q = L.buf + i * LANDING_W;
        int64_t* row = unit_row(store, n_units, (int)q[0]);
        if (!row_alive(row)) continue;
        bool changed = false;
        const int64_t nx = row[UNIT_COL_EXPOSURE + q[1]] + q[2];
        put(row, UNIT_COL_EXPOSURE + (int)q[1], nx < xm ? nx : xm, changed);
        if (changed) touched[q[0]] = 1;
    }
    for (int64_t i = 0; i < L.n_dead; ++i) {
        const int64_t r = L.buf[i * LANDING_W + 3];
        int64_t* row = unit_row(store, n_units, (int)r);
        bool changed = false;
        for (int t = 0; t < UNIT_COL_EXPOSURE_W; ++t) put(row, UNIT_COL_EXPOSURE + t, 0, changed);
        if (changed) touched[r] = 1;
    }
}

// ---- the footprint -----------------------------------------------------------------
// The unit's anchor tile (Unit.tile_x / tile_y: int(x), truncation toward
// zero -- C's division) and its square footprint, row-major (the tables'
// footprint_rowmajor row holds every species to it). *bad on a footprint
// outside [1, FOOTPRINT_MAX].
struct Footprint {
    int64_t ax, ay, fp;
};
__device__ inline Footprint footprint_of(const int64_t* row, bool* bad) {
    Footprint f{tile_of(row[UNIT_COL_X_Q]), tile_of(row[UNIT_COL_Y_Q]), row[UNIT_COL_FOOTPRINT]};
    if (f.fp < 1 || f.fp > FOOTPRINT_MAX) *bad = true;
    return f;
}

// exchange.reduce_max over a (h, w) int32 plane: the true maximum over the
// in-bounds footprint tiles, 0 when none is in bounds.
__device__ inline int64_t reduce_max(const int32_t* plane, int h, int w, const Footprint& f) {
    bool any = false;
    int64_t best = 0;
    for (int64_t dy = 0; dy < f.fp; ++dy) {
        const int64_t ty = f.ay + dy;
        if (ty < 0 || ty >= h) continue;
        for (int64_t dx = 0; dx < f.fp; ++dx) {
            const int64_t tx = f.ax + dx;
            if (tx < 0 || tx >= w) continue;
            const int64_t v = plane[(size_t)ty * w + tx];
            if (!any || v > best) best = v;
            any = true;
        }
    }
    return best;
}

// ---- the credit claims (damage.TickCredit) -----------------------------------------
// The claims table of this block's env: (n_claims, 4) rows (channel, y, x,
// attacker) -- channel UNIT_CREDIT_HEAT for the heat buffer, a gas slice's
// index for a gas deposit; every key once (the first claim kept, as the
// host's dict does). The claimant of (channel, y, x), else -1.
__device__ inline int64_t credit_at(const int64_t* claims, int n_claims, int64_t channel,
                                    int64_t y, int64_t x) {
    if (claims == nullptr) return -1;
    const int64_t* c = bptr(claims, (size_t)n_claims * 4);
    for (int i = 0; i < n_claims; ++i) {
        if (c[4 * i] == channel && c[4 * i + 1] == y && c[4 * i + 2] == x)
            return c[4 * i + 3];
    }
    return -1;
}

// ---- the felt temperature (exchange.felt_temperature_q) ----------------------------
//: The largest exposure (Q16 heat units) the felt-temperature product is
//: proved for: gain_q <= 2^36 (the tables) x 2^42 >> 16 < 2^63.
constexpr int64_t HEAT_PEAK_MAX = (int64_t)1 << 42;

struct FeltTemp {
    int64_t t_felt_q;
    int64_t peak_y, peak_x;
    bool has_peak;     // a tile reached the body (peak_raw > 0)
    bool from_heat;    // its exposure was the heat buffer (the credit question)
};

// The max over the in-bounds footprint tiles of max(heat, fine_heat_shr(
// rad_flux_sweep, 0, fine_bits)) -- the first strictly hottest tile, row-major
// -- then ambient + round(gain x peak / 65536). `heat` / `rad` are this env's
// (h, w) planes.
__device__ inline FeltTemp felt_temperature_q(const int64_t* row, const int64_t* tab,
                                              const int32_t* heat, const int64_t* rad, int h,
                                              int w, bool* bad) {
    FeltTemp ft{tab[UNIT_TAB_HEAT_AMBIENT_Q], -1, -1, false, false};
    const Footprint f = footprint_of(row, bad);
    if (*bad) return ft;
    const int fb = (int)tab[UNIT_TAB_HEAT_FINE_BITS];
    int64_t peak = 0;
    for (int64_t dy = 0; dy < f.fp; ++dy) {
        const int64_t ty = f.ay + dy;
        if (ty < 0 || ty >= h) continue;
        for (int64_t dx = 0; dx < f.fp; ++dx) {
            const int64_t tx = f.ax + dx;
            if (tx < 0 || tx >= w) continue;
            const size_t i = (size_t)ty * w + tx;
            int64_t v = heat[i];
            bool from_heat = true;
            if (rad != nullptr) {
                const int64_t fv = fixedpoint::fine_heat_shr(rad[i], 0, fb);
                if (fv > v) {
                    v = fv;
                    from_heat = false;
                }
            }
            if (v > peak) {
                peak = v;
                ft.peak_y = ty;
                ft.peak_x = tx;
                ft.from_heat = from_heat;
            }
        }
    }
    if (peak <= 0) {
        ft.peak_y = ft.peak_x = -1;
        ft.from_heat = false;
        return ft;
    }
    if (peak > HEAT_PEAK_MAX) {
        *bad = true;
        return ft;
    }
    ft.has_peak = true;
    ft.t_felt_q = tab[UNIT_TAB_HEAT_AMBIENT_Q] +
                  fixedpoint::mul_q16_round_q(tab[UNIT_TAB_HEAT_GAIN_Q], peak);
    return ft;
}

// ---- a slot kernel's row (the local copy) ------------------------------------------
// A slot kernel works on a LOCAL copy of its row and finishes here: a faulted
// row is left as it was in the store and its staged events are dropped (the
// driver refuses the run); else the copy is written back and the row counted
// changed when any column moved.
__device__ inline void finish_row(int64_t* row, const int64_t* loc, EventSink& sink, bool bad,
                                  unsigned long long* cnt) {
    if (bad) {
        *sink.count = 0;
        count_fault(cnt);
        return;
    }
    bool changed = false;
    for (int c = 0; c < UNIT_STORE_W; ++c) {
        if (row[c] != loc[c]) {
            row[c] = loc[c];
            changed = true;
        }
    }
    if (changed) count_changed(cnt);
}

}  // namespace unit_lib

// ---- the host side every damage slot driver shares ------------------------------
// The ring must describe the store's envs and rows.
inline void unit_check_ring(const UnitStoreDevice& store, const UnitEventRing& ring,
                            const char* who) {
    if (ring.n_env() != store.n_env() || ring.n_units() != store.n_units()) {
        std::ostringstream os;
        os << who << ": the event ring holds (" << ring.n_env() << " envs, " << ring.n_units()
           << " rows), the store (" << store.n_env() << ", " << store.n_units() << ")";
        throw std::invalid_argument(os.str());
    }
}

// After a damage slot's launch: the launch error, the ring's gather (the
// staged rows appended in row order), the counts, then each env's dropped
// count (one more D2H + sync).
inline void unit_finish_events(UnitStoreDevice& store, UnitEventRing& ring,
                               const UnitDamageCounts& out, const char* who) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in " << who << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
    ring.gather();
    store.end_counts(out.coverage, out.faults, out.refused);
    if (out.dropped) ring.download(nullptr, nullptr, out.dropped);
}

}  // namespace breach_cuda
