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

#include "cuda_obs_picture.cuh"
#include "cuda_unit_lib.cuh"
#include "obs_layout.h"
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

}  // namespace obs_rows
}  // namespace breach_cuda
