// ============================================================================
// #29 P6.3 -- THE EXCHANGE ROWS THAT HURT, on the device unit store: slot 9c
// (the heat row) and slot 9c3 (the gas rows and the poison law). See
// cuda_unit_kernels.h; residency design §4.4b.
//
// Slot 9c is exchange.apply_environmental_damage: per valid LIVING row, the
// body temperature (#215, body_temp.advance: the felt temperature the flow
// in, the air on the footprint and the radiated share the loss) read by the
// band, the felt temperature (cuda_unit_damage.cuh's felt_temperature_q: the footprint's
// hottest tile of max(heat, rad_flux_sweep), the sweep plane converted ONCE
// through the kit's fine_heat_shr), and above the unit's environment band
// (its env profile's temperature_max, else its species') ONE HEAT packet of
//     round(rate x (ONE + round(overtemp x over / ONE)) / (ONE x tps))
// through the device apply_packet (indirect; cause heat), credited to the
// heat-channel claimant of the hottest tile only when its exposure was the
// heat buffer (damage.TickCredit, a per-tick table).
//
// Slot 9c3 is its label's three callables fused per row -- each unit's three
// steps read only the planes and its own row, so the per-row order is the
// reference's: exchange.apply_teargas_blind (the footprint max of the teargas
// plane at or above the threshold applies BLINDED, refresh, no source),
// exchange.apply_poison_dose (the footprint max of the poison plane at or
// above its threshold, the unit not immune -- mitigation resist 0 --, doses
// round(gas_q x density / (TRACE_ONE x tps)) through poison.dose, credited to
// the poison-channel claimant of the first densest footprint tile; a dose past
// the body's store is clipped and COUNTED in `refused`), then poison.tick (the
// two levels: m0 depot -> blood, m1 out of the blood with the flush only once
// the depot is empty, the TOX hooks through the library's scaled, ONE POISON
// packet of round(m1 x harm) credited to the dose's claimant, indirectly;
// poison_src back to -1 when both levels reach 0), then (#215) stress.tick (the
// level less clearing_drain at its 2^-24 rate). The trace planes are read in
// their own (TRACE) currency against thresholds quantized in it.
//
// Planes: device pointers to (N, ...) arrays in the resident fields' dtypes
// (heat (N, h, w) int32, rad_flux_sweep (N, h, w) int64, gas (N, G, h, w)
// int32) -- ResidentBatch.buffers as they are.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_unit_damage.cuh"

namespace breach_cuda {

namespace {

using namespace unit_lib;

// The unit's environment row index: (true, profile p) or (false, species).
__device__ inline bool env_select(const int64_t* row, int n_profiles, int n_species,
                                  bool* is_profile, int64_t* idx) {
    const int64_t p = row[UNIT_COL_ENV_PROFILE_NAME];
    if (p >= 0) {
        *is_profile = true;
        *idx = p;
        return p < n_profiles;
    }
    *is_profile = false;
    *idx = row[UNIT_COL_SPECIES_ID];
    return *idx >= 0 && *idx < n_species;
}

// ---- slot 9c ----------------------------------------------------------------------
__global__ void k_heat_damage(int64_t* store, int n_units, const int64_t* tables,
                              const int32_t* heat, const int64_t* rad, const int32_t* atm,
                              const int32_t* temp, const bool* solid, int h, int w,
                              const int64_t* claims, int n_claims, UnitRingView ring,
                              unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    if (!row_valid(row) || !row_alive(row)) return;
    const int64_t* tab = env_tables(tables);
    const size_t plane = (size_t)h * w;
    bool bad = false;
    const FeltTemp ft = felt_temperature_q(row, tab, bptr(heat, plane), bptr(rad, plane), h, w,
                                           &bad);
    EventSink sink = row_sink(ring, n_units, r);
    int64_t loc[UNIT_STORE_W];
    for (int c = 0; c < UNIT_STORE_W; ++c) loc[c] = row[c];
    if (!bad) {
        bool is_profile;
        int64_t idx;
        if (!env_select(loc, UNIT_TAB_HEAT_TEMP_MAX_PROFILE_Q_W,
                        UNIT_TAB_HEAT_TEMP_MAX_SPECIES_Q_W, &is_profile, &idx)) {
            bad = true;
        } else {
            const int64_t temp_max = is_profile ? tab[UNIT_TAB_HEAT_TEMP_MAX_PROFILE_Q + idx]
                                                : tab[UNIT_TAB_HEAT_TEMP_MAX_SPECIES_Q + idx];
            const int64_t rate = is_profile ? tab[UNIT_TAB_HEAT_RATE_PROFILE_Q + idx]
                                            : tab[UNIT_TAB_HEAT_RATE_SPECIES_Q + idx];
            // #215: body_temp.advance -- the felt temperature is the flow into
            // the body; the band reads the body.
            const int64_t a = is_profile ? tab[UNIT_TAB_BODY_A_PROFILE_Q + idx]
                                         : tab[UNIT_TAB_BODY_A_SPECIES_Q + idx];
            const int64_t a_rad = is_profile ? tab[UNIT_TAB_BODY_A_RAD_PROFILE_Q + idx]
                                             : tab[UNIT_TAB_BODY_A_RAD_SPECIES_Q + idx];
            const BodyAir air = body_air(loc, tab, bptr(atm, plane), bptr(temp, plane),
                                         bptr(solid, plane), h, w, &bad);
            const int64_t body = bad ? 0 : body_step(loc[UNIT_COL_BODY_TEMP_Q], ft.t_felt_q, air,
                                                     a, a_rad, tab[UNIT_TAB_HEAT_AMBIENT_Q], &bad);
            if (!bad) {
                bool ch = false;
                put(loc, UNIT_COL_BODY_TEMP_Q, body, ch);
            }
            const int64_t over = bad ? 0 : body - temp_max;
            if (over > 0) {                      // above the band
                if (over > ((int64_t)1 << 46)) {
                    bad = true;
                } else {
                    const int64_t g = ONE + fixedpoint::mul_q16_round_q(
                                                tab[UNIT_TAB_HEAT_OVERTEMP_Q], over);
                    uint64_t hi, lo;
                    fixedpoint::umul64_wide((uint64_t)rate, (uint64_t)g, &hi, &lo);
                    if (hi != 0 || (lo >> 63) != 0) {
                        bad = true;
                    } else {
                        const int64_t dmg = fixedpoint::div_round_q(
                            (int64_t)lo, ONE * tab[UNIT_TAB_HEAT_TPS]);
                        Packet p;
                        p.amount_q = dmg;
                        p.dtype = UNIT_DTYPE_HEAT;
                        p.attacker = ft.from_heat ? credit_at(claims, n_claims, UNIT_CREDIT_HEAT,
                                                              ft.peak_y, ft.peak_x)
                                                  : -1;
                        p.indirect = true;
                        p.cause = UNIT_CAUSE_HEAT;
                        bool changed = false, killed = false;
                        apply_packet(loc, tab, p, sink, changed, &killed, &bad);
                    }
                }
            }
        }
    }
    finish_row(row, loc, sink, bad, cnt);
}

// ---- slot 9c3 ---------------------------------------------------------------------
// poison.step: what leaves a level x at the Q16 fraction a -- the shared
// clearing step (unit_lib::clearing_drain, #215), the flush only when asked.
__device__ __forceinline__ int64_t poison_step(int64_t x, int64_t a, bool flush,
                                               int64_t flush_q) {
    return clearing_drain(x, a, 16, flush ? flush_q : 0);
}

//: clearing.RATE_SHIFT: the stress rate's currency (2^-24 per tick).
constexpr int STRESS_RATE_SHIFT = 24;

//: The levels the law's products are proved for (x * a < 2^56).
constexpr int64_t POISON_LEVEL_MAX = (int64_t)1 << 40;

__global__ void k_gas_rows(int64_t* store, int n_units, const int64_t* tables, const int32_t* gas,
                           int n_gases, int h, int w, const int64_t* claims, int n_claims,
                           UnitRingView ring, unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    if (!row_valid(row)) return;
    const bool alive = row_alive(row);
    if (!alive && row[UNIT_COL_POISON_X0] == 0 && row[UNIT_COL_POISON_X1] == 0 &&
        row[UNIT_COL_STRESS_Q] == 0) return;
    const int64_t* tab = env_tables(tables);
    const size_t plane = (size_t)h * w;
    const int32_t* g = bptr(gas, (size_t)n_gases * plane);
    int64_t loc[UNIT_STORE_W];
    for (int c = 0; c < UNIT_STORE_W; ++c) loc[c] = row[c];
    EventSink sink = row_sink(ring, n_units, r);
    bool bad = false, changed = false;
    const int64_t gi_t = tab[UNIT_TAB_GAS_TEARGAS], gi_p = tab[UNIT_TAB_GAS_POISON];
    if (gi_t >= n_gases || gi_p >= n_gases) bad = true;

    if (alive && !bad) {
        const Footprint f = footprint_of(loc, &bad);
        // -- exchange.apply_teargas_blind --------------------------------------
        if (!bad) {
            const int64_t dens = reduce_max(g + (size_t)gi_t * plane, h, w, f);
            if (dens >= tab[UNIT_TAB_TEARGAS_THRESH_Q]) {
                const int res = apply_status(loc, tab, UNIT_STATUS_BLINDED, 0,
                                             tab[UNIT_TAB_TEARGAS_BLIND_TICKS], -1, changed);
                if (res == APPLY_REFUSED) count_refused(cnt);
                if (res == APPLY_BAD) bad = true;
            }
        }
        // -- exchange.apply_poison_dose -> poison.dose -------------------------
        if (!bad) {
            const int32_t* pp = g + (size_t)gi_p * plane;
            const int64_t dens = reduce_max(pp, h, w, f);
            const int64_t* armor;
            const int64_t* resist;
            if (dens >= tab[UNIT_TAB_POISON_THRESH_Q]) {
                if (!mitigation_of(loc, tab, &armor, &resist)) {
                    bad = true;
                } else if (resist[UNIT_DTYPE_POISON] != 0) {
                    const int64_t dose = fixedpoint::div_round_q(
                        tab[UNIT_TAB_POISON_GAS_Q] * dens, tab[UNIT_TAB_POISON_DOSE_DEN]);
                    // the claimant of the first densest footprint tile, row-major
                    int64_t attacker = -1;
                    bool found = false;
                    for (int64_t dy = 0; dy < f.fp && !found; ++dy) {
                        const int64_t ty = f.ay + dy;
                        if (ty < 0 || ty >= h) continue;
                        for (int64_t dx = 0; dx < f.fp; ++dx) {
                            const int64_t tx = f.ax + dx;
                            if (tx < 0 || tx >= w) continue;
                            if ((int64_t)pp[(size_t)ty * w + tx] == dens) {
                                attacker = credit_at(claims, n_claims, gi_p, ty, tx);
                                found = true;
                                break;
                            }
                        }
                    }
                    // poison.dose: a zombie or a corpse takes nothing (both
                    // already excluded here); the body's store is X_MAX.
                    const int64_t x0 = loc[UNIT_COL_POISON_X0], x1 = loc[UNIT_COL_POISON_X1];
                    if (x0 < 0 || x1 < 0 || x0 > POISON_LEVEL_MAX || x1 > POISON_LEVEL_MAX) {
                        bad = true;
                    } else if (dose > 0 && !loc[UNIT_COL_IS_ZOMBIE]) {
                        const int64_t room = tab[UNIT_TAB_POISON_X_MAX] - x0 - x1;
                        const int64_t take = dose <= room ? dose : (room > 0 ? room : 0);
                        if (take < dose) count_refused(cnt);     // the clip, counted
                        if (take > 0) {
                            put(loc, UNIT_COL_POISON_X0, x0 + take, changed);
                            put(loc, UNIT_COL_POISON_SRC, attacker_value(attacker), changed);
                        }
                    }
                }
            }
        }
    }
    // -- poison.tick ---------------------------------------------------------------
    int64_t x0 = loc[UNIT_COL_POISON_X0], x1 = loc[UNIT_COL_POISON_X1];
    if (!bad && (x0 != 0 || x1 != 0)) {
        if (!row_alive(loc) || loc[UNIT_COL_IS_ZOMBIE]) {
            put(loc, UNIT_COL_POISON_X0, 0, changed);            // poison.clear
            put(loc, UNIT_COL_POISON_X1, 0, changed);
            put(loc, UNIT_COL_POISON_SRC, -1, changed);
        } else if (x0 < 0 || x1 < 0 || x0 > POISON_LEVEL_MAX || x1 > POISON_LEVEL_MAX) {
            bad = true;
        } else {
            const int64_t src = loc[UNIT_COL_POISON_SRC];
            int64_t onset = scaled(loc, UNIT_HOOK_TOX_ONSET, tab[UNIT_TAB_POISON_ONSET_CLASS_Q]);
            int64_t decay = scaled(loc, UNIT_HOOK_TOX_DECAY, tab[UNIT_TAB_POISON_DECAY_CLASS_Q]);
            if (onset > ONE) onset = ONE;                        // poison._fraction
            if (decay > ONE) decay = ONE;
            const int64_t fq = tab[UNIT_TAB_POISON_FLUSH_Q];
            const int64_t m0 = poison_step(x0, onset, false, fq);
            x0 -= m0;
            x1 += m0;
            const int64_t m1 = poison_step(x1, decay, x0 == 0, fq);
            x1 -= m1;
            put(loc, UNIT_COL_POISON_X0, x0, changed);
            put(loc, UNIT_COL_POISON_X1, x1, changed);
            if (x0 == 0 && x1 == 0) put(loc, UNIT_COL_POISON_SRC, -1, changed);
            if (m1 > 0) {
                const int64_t harm = scaled(loc, UNIT_HOOK_TOX_HARM,
                                            tab[UNIT_TAB_POISON_HARM_CLASS_Q]);
                const int64_t dmg = fixedpoint::mul_q16_round_q(m1, harm);
                if (dmg > 0) {
                    Packet p;
                    p.amount_q = dmg;
                    p.dtype = UNIT_DTYPE_POISON;
                    p.attacker = src;
                    p.indirect = true;
                    p.cause = UNIT_CAUSE_POISON_GAS;
                    bool killed = false;
                    apply_packet(loc, tab, p, sink, changed, &killed, &bad);
                }
            }
        }
    }
    // -- stress.tick (#215, slot 9c3'': after the poison law fed it) -------------
    const int64_t sx = loc[UNIT_COL_STRESS_Q];
    if (!bad && sx != 0) {
        if (!row_alive(loc) || loc[UNIT_COL_IS_ZOMBIE]) {
            put(loc, UNIT_COL_STRESS_Q, 0, changed);             // stress.clear
        } else if (sx < 0 || sx > tab[UNIT_TAB_STRESS_MAX]) {
            bad = true;
        } else {
            put(loc, UNIT_COL_STRESS_Q,
                sx - clearing_drain(sx, tab[UNIT_TAB_STRESS_RATE_Q], STRESS_RATE_SHIFT,
                                    tab[UNIT_TAB_STRESS_FLUSH_Q]), changed);
        }
    }
    finish_row(row, loc, sink, bad, cnt);
}

void check_claims(const int64_t* claims, int n_claims, const char* who) {
    if (n_claims < 0 || (n_claims > 0 && claims == nullptr)) {
        std::ostringstream os;
        os << who << ": the claims table must be (n_env, n_claims, 4) int64";
        throw std::invalid_argument(os.str());
    }
}

}  // namespace

void unit_heat_damage(UnitStoreDevice& store, const UnitTablesDevice& tables, UnitEventRing& ring,
                      const UnitDamagePlanes& planes, const int64_t* claims, int n_claims,
                      const UnitDamageCounts& out) {
    unit_check_pair(store, tables, "unit_heat_damage");
    unit_check_ring(store, ring, "unit_heat_damage");
    check_claims(claims, n_claims, "unit_heat_damage");
    if (!planes.heat || !planes.rad_flux || !planes.atmosphere || !planes.temperature ||
        !planes.solid || planes.h < 1 || planes.w < 1)
        throw std::invalid_argument("unit_heat_damage: the heat, rad_flux_sweep, atmosphere, "
                                    "temperature and solid planes and their shape are required");
    const int64_t* d_claims = n_claims > 0
        ? store.stage_input(1, claims, (size_t)store.n_env() * n_claims * 4) : nullptr;
    store.begin_counts();
    k_heat_damage<<<bgrid(store.n_units(), store.n_env()), 128, 0, tstream()>>>(
        store.rows(), store.n_units(), tables.values(),
        reinterpret_cast<const int32_t*>(planes.heat),
        reinterpret_cast<const int64_t*>(planes.rad_flux),
        reinterpret_cast<const int32_t*>(planes.atmosphere),
        reinterpret_cast<const int32_t*>(planes.temperature),
        reinterpret_cast<const bool*>(planes.solid), planes.h, planes.w, d_claims,
        n_claims, ring_view(ring), store.counts_dev());
    unit_finish_events(store, ring, out, "unit_heat_damage");
}

void unit_gas_rows(UnitStoreDevice& store, const UnitTablesDevice& tables, UnitEventRing& ring,
                   const UnitDamagePlanes& planes, const int64_t* claims, int n_claims,
                   const UnitDamageCounts& out) {
    unit_check_pair(store, tables, "unit_gas_rows");
    unit_check_ring(store, ring, "unit_gas_rows");
    check_claims(claims, n_claims, "unit_gas_rows");
    if (!planes.gas || planes.n_gases < 1 || planes.h < 1 || planes.w < 1)
        throw std::invalid_argument("unit_gas_rows: the gas planes and their shape are required");
    const int64_t* d_claims = n_claims > 0
        ? store.stage_input(1, claims, (size_t)store.n_env() * n_claims * 4) : nullptr;
    store.begin_counts();
    k_gas_rows<<<bgrid(store.n_units(), store.n_env()), 128, 0, tstream()>>>(
        store.rows(), store.n_units(), tables.values(),
        reinterpret_cast<const int32_t*>(planes.gas), planes.n_gases, planes.h, planes.w,
        d_claims, n_claims, ring_view(ring), store.counts_dev());
    unit_finish_events(store, ring, out, "unit_gas_rows");
}

}  // namespace breach_cuda
