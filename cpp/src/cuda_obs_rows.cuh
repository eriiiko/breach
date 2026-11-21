#pragma once
// ==========================================================================
// #29 P8.4b: THE DEVICE OBSERVATION's row predicates (Class T), the ONE copy
// every observation TU reads (cuda_obs.cu, cuda_obs_picture.cu,
// cuda_obs_list.cu): an env's decision tick, a store row that is a living
// mate of a team (TeamKnowledge.mates), a store row a team sees now and holds
// a record of (TeamKnowledge.seen). The observation's own header, never the
// unit library.
// ==========================================================================
#include <cstdint>

#include <cuda_runtime.h>

#include "cuda_obs_float.cuh"
#include "cuda_obs_picture.cuh"
#include "cuda_unit_lib.cuh"
#include "fixed_point.h"
#include "obs_layout.h"
#include "unit_tables_layout.h"
#include "unit_draws_layout.h"
#include "unit_store_layout.h"

namespace breach_cuda {
namespace obs_rows {

using namespace obs_layout;

// The env's decision tick: (tick - episode start) a multiple of the
// decimation, and its ended latch clear.
__device__ __forceinline__ bool env_due(int e, const int64_t* trow, const int64_t* params,
                                        const uint8_t* ended) {
    if (ended[e]) return false;
    const int64_t tick = trow[(size_t)e * unit_draws::TICK_ROW_W + unit_draws::TICK_ROW_TICK];
    const int64_t t0 = params[(size_t)e * OBS_PARAM_W + OBS_PARAM_EPISODE_TICK];
    const int64_t dec = params[(size_t)e * OBS_PARAM_W + OBS_PARAM_DECIMATION];
    const int64_t d = tick - t0;
    return dec > 0 && d >= 0 && d % dec == 0;
}

// A store row that is a living mate of team t (TeamKnowledge.mates).
__device__ __forceinline__ bool is_mate(const int64_t* m, int64_t t) {
    using namespace unit_store;
    return m[UNIT_COL_VALID] && m[UNIT_COL_LIFE_STATE] == OBS_LIFE_ALIVE &&
           !m[UNIT_COL_IS_ZOMBIE] && m[UNIT_COL_TEAM] == t;
}

// A store row team t sees now and holds a record of (TeamKnowledge.seen:
// visible_by_team[t] restricted to the records).
__device__ __forceinline__ bool is_seen(const uint8_t* vis_t, const int64_t* blk,
                                        const obs_pic::PicLayout& L, int U, int t, int q) {
    return vis_t[q] && obs_pic::rec_of(blk, L, U, t, q)[OBS_PIC_REC_PRESENT];
}

// #29 P8.6a: a soldier's own turning dials as the observation reads them
// (the _Aimer's a / w / dw: Simulation.unit_turn_accel_q16 /
// unit_turn_rate_q16 through the TURN hook, unit_impulse.pending_dw =
// imp_l / inertia, truncating) -- ticks_to_within's (P8.4a) and
// on_next_tick's (P8.6a) one copy.
__device__ __forceinline__ void turn_dials(const int64_t* me, const int64_t* tab, int64_t* a,
                                           int64_t* w, int64_t* dw) {
    using namespace unit_store;
    *a = unit_lib::scaled(me, unit_tables::UNIT_HOOK_TURN,
                          tab[unit_tables::UNIT_TAB_TURN_ACCEL_CLASS_Q16]);
    *w = unit_lib::scaled(me, unit_tables::UNIT_HOOK_TURN,
                          tab[unit_tables::UNIT_TAB_TURN_RATE_CLASS_Q16]);
    const int64_t l = me[UNIT_COL_IMP_L];
    *dw = l ? l / tab[unit_tables::UNIT_TAB_IMPULSE_INERTIA_G] : 0;   // div_trunc
}

// ---------------------------------------------------------------------------
// #29 P8.8: the reads the list (P8.4a / b) and the full view share, lifted
// here from cuda_obs.cu's TU so both call ONE copy.
// ---------------------------------------------------------------------------

// _cone_of: a zombie the reference kit's melee arc; a ranged weapon (range >
// 0) its shot cone (the SPREAD hook over the cone door of spread_deg); else
// its melee row's arc; else the reference kit's melee arc.
__device__ __forceinline__ int64_t aim_cone(const int64_t* u, const int64_t* ot,
                                            const int64_t* tab) {
    using namespace unit_store;
    if (u[UNIT_COL_IS_ZOMBIE]) return ot[OBS_TAB_REF_MELEE_ARC_Q];
    const int64_t wr = u[UNIT_COL_WEAPON_ID];
    if (wr >= 0) {
        const int64_t* W = ot + OBS_TAB_WEAPON + wr * OBS_WEAPON_W;
        if (W[OBS_WEAPON_RANGE_TILES] > 0)
            return unit_lib::scaled(u, unit_tables::UNIT_HOOK_SPREAD, W[OBS_WEAPON_CONE_AIM_Q]);
    }
    const int64_t mr = u[UNIT_COL_MELEE_ID];
    if (mr >= 0) return tab[unit_tables::UNIT_TAB_WEAPON_MELEE_ARC_HALF_Q + mr];
    return ot[OBS_TAB_REF_MELEE_ARC_Q];
}

// The kit's atan2_q16(-dy << 16, dx << 16) (the bearing of an integer tile
// offset); *bad when an argument leaves the kit's int32 (pybind would refuse).
__device__ __forceinline__ int64_t bearing_q16(int64_t dx, int64_t dy, bool* bad) {
    const int64_t y = -dy * 65536, x = dx * 65536;
    if (!unit_lib::fits_i32(y) || !unit_lib::fits_i32(x)) {
        *bad = true;
        return 0;
    }
    return fixedpoint::atan2_q16((int32_t)y, (int32_t)x);
}

// turning.ticks_to_within: the ticks until |wrap(target - facing)| <= tol for
// the unit turning under its own a / w from its facing and omega + dw, by
// ITERATING the unit library's law (never a copy); cap when not reached.
__device__ inline int64_t ticks_to_within(int64_t facing, int64_t omega, int64_t target,
                                          int64_t a, int64_t w, int64_t tol, int64_t cap,
                                          bool* bad) {
    int64_t n = 0;
    while (n < cap) {
        const int64_t d = unit_lib::wrap_q16(target - facing);
        if ((d < 0 ? -d : d) <= tol) return n;
        int64_t f2, o2;
        unit_lib::turn_law_q16(facing, omega, target, a, w, &f2, &o2, bad);
        facing = f2;
        omega = o2;
        ++n;
    }
    return cap;
}

// _enemy_geometry's ticks (col 27): the soldier `me` turning toward the
// offset (dx, dy) under its own dials, within its weapon's shot cone (0
// without a weapon), capped at 4 x the half turn. *bad: the bearing's range;
// *law_bad: the law's.
__device__ inline int64_t ticks_toward(const int64_t* me, const int64_t* ot, const int64_t* tab,
                                       int64_t dx, int64_t dy, bool* bad, bool* law_bad) {
    using namespace unit_store;
    const int64_t want = bearing_q16(dx, dy, bad);
    int64_t a, wq, dw;
    turn_dials(me, tab, &a, &wq, &dw);
    const int64_t wr = me[UNIT_COL_WEAPON_ID];
    const int64_t spread = wr >= 0 ? unit_lib::scaled(me, unit_tables::UNIT_HOOK_SPREAD,
                                                      ot[OBS_TAB_WEAPON + wr * OBS_WEAPON_W +
                                                         OBS_WEAPON_CONE_AIM_Q])
                                   : 0;
    if (*bad) return 0;
    return ticks_to_within(me[UNIT_COL_FACING_Q], me[UNIT_COL_OMEGA_Q] + dw, want, a, wq, spread,
                           4 * ot[OBS_TAB_HALF_TURN_TICKS], law_bad);
}

// _status_left: per status kind the longest remaining_ticks of the list.
__device__ __forceinline__ int64_t status_left(const int64_t* u, int kind) {
    using namespace unit_store;
    int64_t best = 0;
    const int64_t* s = u + UNIT_COL_STATUSES;
    for (int i = 0; i < unit_lib::STATUS_ROWS; ++i) {
        const int64_t k = s[4 * i];
        if (k < 0) break;
        if (k == kind && s[4 * i + 2] > best) best = s[4 * i + 2];
    }
    return best;
}

// _kit_values (the vector's cols 24-26, 29.., 44, 49, 51 and the full view's
// extra columns: ONE reader, so the two agree column for column), float64.
struct Kit {
    double rounds, mags, untracked, reloading, reload_left;
    double grenades[OBS_HAND_GRENADE_SLOTS];
    double charges, melee_cool, shove_cool;
};
__device__ inline Kit kit_of(const int64_t* u, const int64_t* ot, const int64_t* tab, int64_t tick,
                             double cap, bool* bad) {
    using namespace unit_store;
    using namespace obs_f64;
    Kit k{};
    const int64_t w = u[UNIT_COL_WEAPON_ID];                 // NO_ROW: no weapon
    const int64_t lo = u[UNIT_COL_LOADOUT_NAME];             // NO_ROW: no loadout
    const int64_t* L = lo >= 0 ? ot + OBS_TAB_LOADOUT + lo * OBS_LOADOUT_W : nullptr;
    const int64_t* W = w >= 0 ? ot + OBS_TAB_WEAPON + w * OBS_WEAPON_W : nullptr;
    const int64_t mags_total = L ? L[OBS_LOADOUT_MAGAZINES] : 0;
    k.rounds = k.mags = k.untracked = 1.0;
    k.reloading = k.reload_left = 0.0;
    const unit_lib::MagStatus m = W ? unit_lib::mag_status(u, tab, tick, bad)
                                    : unit_lib::MagStatus{false, 0, -1, false};
    if (m.tracked) {
        k.rounds = ratio(m.rounds, W[OBS_WEAPON_MAG_SIZE]);
        if (m.spares < 0) {
            k.mags = k.untracked = 1.0;
        } else {
            k.mags = squash(ratio(m.spares, mags_total > 1 ? mags_total : 1), cap);
            k.untracked = 0.0;
        }
        k.reloading = m.reloading ? 1.0 : 0.0;
        if (m.reloading)
            k.reload_left = ratio(u[UNIT_COL_RELOAD_DONE_TICK] - tick,
                                  unit_lib::handling_ticks(u, W[OBS_WEAPON_RELOAD_TICKS]));
    }
    for (int g = 0; g < OBS_HAND_GRENADE_SLOTS; ++g) {
        const int64_t have = L ? L[OBS_LOADOUT_GRENADES + g] : 0;
        k.grenades[g] = have > 0 ? ratio(u[UNIT_COL_GRENADE_COUNTS + g], have) : 0.0;
    }
    k.charges = ratio(u[UNIT_COL_HAS_EXPLOSIVE], L ? L[OBS_LOADOUT_EXPLOSIVES] : 0);
    k.melee_cool = 0.0;
    const int64_t mr = u[UNIT_COL_MELEE_ID];
    if (mr >= 0 && u[UNIT_COL_LAST_BLOW_TICK] >= 0) {
        const int64_t rof = ot[OBS_TAB_WEAPON + mr * OBS_WEAPON_W + OBS_WEAPON_ROF_TICKS];
        const int64_t left = u[UNIT_COL_LAST_BLOW_TICK] + rof - tick;
        k.melee_cool = ratio(left > 0 ? left : 0, rof);
    }
    k.shove_cool = 0.0;
    if (u[UNIT_COL_LAST_SHOVE_TICK] >= 0) {
        const int64_t cd = ot[OBS_TAB_SHOVE_COOLDOWN_TICKS];
        const int64_t left = u[UNIT_COL_LAST_SHOVE_TICK] + cd - tick;
        k.shove_cool = ratio(left > 0 ? left : 0, cd);
    }
    return k;
}

}  // namespace obs_rows
}  // namespace breach_cuda
