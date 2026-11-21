// ==========================================================================
// #29 P8.6a: THE DEVICE DECODER (see cuda_obs.h; src/simulation/
// device_decoder.py), Class T. The twin of the `policy` controller's action
// spec v1 path (control_policy.AgentPolicy.set_decisions / _decode,
// issue_row, VERB_RULES; action_spec_v1.check_actions / masked_choices /
// aim_vector / onto_reach_q), every env in one launch per step, on THE tick
// stream. The host decoder is the reference.
//
//   obs_dec_check   once per DECISION (one thread per env): the range check
//                   (a per-env fault word: the count, per head the first row
//                   and its value), the masked count per row against the
//                   SERVED masks, the pointer -> game id against the list_uid
//                   handed in (the observation the decision answers), the
//                   no-sim fallbacks; the held throw aim un-aliased from the
//                   decision it was taken from.
//   obs_decode      EVERY TICK at the conductor's tick head (one thread per
//                   (env, row) of a decoder-driven team): _decode line for
//                   line, the issued intents written into their latch columns
//                   in the store's encodings (the hold kinds every tick, the
//                   latch kinds on the decision's first tick), the slot state,
//                   DeviceObservation.aimed; the fault words deferred into the
//                   conductor's one books read.
//   obs_dec_horde   the zombie controller's latch (every living zombie at the
//                   tick's start: SET + per-row flags, EMPTY when none).
//   obs_dec_obs     on a decision tick after the list and the windows: the
//                   vector's prev block, the list's prev_aim / on_next_tick.
//   obs_dec_probe   TEST-ONLY: aim_vector and onto_reach_q over offsets.
//
// THE FLOATS: aim_vector (quantize_stick_direction: sqrt, the reciprocal,
// two products, quantize_scalar) and onto_reach_q (quantize_scalar of the
// IEEE sqrt) in float64 through the kit (cuda_obs_float.cuh: never fused),
// __dsqrt_rn correctly rounded like math.sqrt; their integer results exact.
// ==========================================================================
#include "cuda_obs.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "cuda_batch.cuh"          // tstream(): THE tick stream (blocking)
#include "cuda_obs_float.cuh"
#include "cuda_obs_picture.cuh"
#include "cuda_obs_rows.cuh"
#include "cuda_obs_sight.cuh"
#include "cuda_unit_books.h"       // books_defer: the fault words ride the one books read
#include "cuda_unit_lib.cuh"       // the unit library's integer reads (used, never added to)
#include "cuda_unit_store.h"       // UnitTablesDevice
#include "fixed_point.h"
#include "obs_layout.h"
#include "unit_draws_layout.h"
#include "unit_store_layout.h"
#include "unit_tables_layout.h"

namespace breach_cuda {

using namespace obs_layout;

namespace {

using namespace obs_f64;
using namespace unit_store;
using obs_pic::PicLayout;
using obs_pic::pic_layout;
using obs_rows::env_due;

__constant__ int64_t c_compass[2 * OBS_DEC_N_DIRS] = OBS_DEC_COMPASS_INIT;

void need(bool ok, const char* who, const char* what) {
    if (!ok) throw std::invalid_argument(std::string(who) + ": " + what);
}

void launched(const char* who) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(who) + ": launch failed: " + cudaGetErrorString(err));
}

// unit_fixed.quantize_scalar: v * 65536, round half away from zero.
__device__ __forceinline__ int64_t quantize_scalar(double u) {
    const double v = mul(u, 65536.0);
    return v >= 0.0 ? (int64_t)floor(add(v, 0.5)) : (int64_t)ceil(sub(v, 0.5));
}

// action_spec_v1.aim_vector over the integer offset (dx, dy) from the unit's
// centre tile: quantize_stick_direction(float(dx), float(dy), 0.0); false at
// a zero offset (mag <= the deadzone 0.0). `div_brk`: THE BREAK dec_div.
__device__ __forceinline__ bool aim_vector(int64_t dx, int64_t dy, bool div_brk, int64_t* qx,
                                           int64_t* qy) {
    const double fx = (double)dx, fy = (double)dy;
    const double mag = __dsqrt_rn(add(mul(fx, fx), mul(fy, fy)));
    if (mag <= 0.0) return false;
    double ux, uy;
    if (div_brk) {
        ux = div(fx, mag);
        uy = div(fy, mag);
    } else {
        const double inv = div(1.0, mag);
        ux = mul(fx, inv);
        uy = mul(fy, inv);
    }
    *qx = quantize_scalar(ux);
    *qy = quantize_scalar(uy);
    return true;
}

// action_spec_v1.onto_reach_q: quantize_scalar(sqrt(float(dx*dx + dy*dy))).
__device__ __forceinline__ int64_t onto_reach_q(int64_t dx, int64_t dy) {
    return quantize_scalar(__dsqrt_rn((double)(dx * dx + dy * dy)));
}

// The team t's record of unit `uid` in the picture block (records(team).get):
// its store row, or -1.
__device__ int record_row(const int64_t* blk, const PicLayout& L, int U, int t, int64_t uid) {
    for (int q = 0; q < U; ++q) {
        const int64_t* rc = obs_pic::rec_of(blk, L, U, t, q);
        if (rc[OBS_PIC_REC_PRESENT] && rc[OBS_PIC_REC_UID] == uid) return q;
    }
    return -1;
}

// AgentPolicy._point over an aim's fields A = (kind, value, lpv, lpx, lpy):
// the record's centre refreshes last_point; returns whether a point exists.
__device__ bool point_of(int64_t* A, const int64_t* blk, const PicLayout& L, int U, int t) {
    const int q = record_row(blk, L, U, t, A[1]);
    if (q >= 0) {
        const int64_t* rc = obs_pic::rec_of(blk, L, U, t, q);
        A[2] = 1;
        A[3] = rc[OBS_PIC_REC_CENTRE_X];
        A[4] = rc[OBS_PIC_REC_CENTRE_Y];
    }
    return A[2] != 0;
}

// The fields of the decision's aim and of the held throw aim (contiguous).
static_assert(OBS_DEC_S_D_VAL == OBS_DEC_S_D_KIND + 1 && OBS_DEC_S_D_LPV == OBS_DEC_S_D_KIND + 2 &&
                  OBS_DEC_S_D_LPX == OBS_DEC_S_D_KIND + 3 && OBS_DEC_S_D_LPY == OBS_DEC_S_D_KIND + 4,
              "the decision aim's fields are contiguous");
static_assert(OBS_DEC_S_TA_VAL == OBS_DEC_S_TA_KIND + 1 &&
                  OBS_DEC_S_TA_LPV == OBS_DEC_S_TA_KIND + 2 &&
                  OBS_DEC_S_TA_LPX == OBS_DEC_S_TA_KIND + 3 &&
                  OBS_DEC_S_TA_LPY == OBS_DEC_S_TA_KIND + 4,
              "the throw aim's fields are contiguous");

// ---------------------------------------------------------------------------
// THE DECISION: one thread per env (rows are few), every row of a due env.
// ---------------------------------------------------------------------------
__global__ void k_check(int n_env, int n_rows, const int32_t* dec, const uint8_t* due,
                        const int32_t* list_uid, int n_entries, const uint8_t* m_feet,
                        const uint8_t* m_aim, const uint8_t* m_trig, const uint8_t* m_hand,
                        int n_kinds, const int64_t* dtab, int64_t* state, uint8_t* masked,
                        int64_t* faults) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_env || !due[e]) return;
    const int64_t* tb = dtab + (size_t)e * OBS_DEC_TAB_W;
    int64_t* F = faults + (size_t)e * OBS_DEC_FAULT_W;
    F[0] = 0;
    for (int h = 0; h < OBS_DEC_N_HEADS; ++h) {
        F[1 + 2 * h] = -1;
        F[2 + 2 * h] = 0;
    }
    const int n_aim = OBS_MASK_AIM_TOKENS + n_entries;
    const int n_hand = OBS_MASK_HAND_THROW0 + n_kinds;
    for (int r = 0; r < n_rows; ++r) {
        const size_t er = (size_t)e * n_rows + r;
        const int32_t* a = dec + er * OBS_DEC_N_HEADS;
        int64_t* st = state + er * OBS_DEC_S_W;
        bool bad = false;
        for (int h = 0; h < OBS_DEC_N_HEADS; ++h) {
            const int64_t v = a[h];
            if (v < 0 || v >= tb[OBS_DEC_TAB_SIZE0 + h]) {
                bad = true;
                if (F[1 + 2 * h] < 0) {
                    F[1 + 2 * h] = r;
                    F[2 + 2 * h] = v;
                }
            }
        }
        // a held throw aim that IS the old decision's aim object keeps that
        // object (its kind, value, last point) when the decision is replaced
        if (st[OBS_DEC_S_TA] && st[OBS_DEC_S_TA_ALIAS]) {
            for (int k = 0; k < 5; ++k) st[OBS_DEC_S_TA_KIND + k] = st[OBS_DEC_S_D_KIND + k];
            st[OBS_DEC_S_TA_ALIAS] = 0;
        }
        st[OBS_DEC_S_HAVE] = 1;
        st[OBS_DEC_S_FIRST] = 1;
        st[OBS_DEC_S_BAD] = bad ? 1 : 0;
        if (bad) {                       // refused at the books read; decodes as nothing
            ++F[0];
            masked[er] = 0;
            st[OBS_DEC_S_D_FEET] = OBS_MASK_FEET_STAND;
            st[OBS_DEC_S_D_KIND] = OBS_DEC_AIM_WALK;
            st[OBS_DEC_S_D_VAL] = -1;
            st[OBS_DEC_S_D_LPV] = 0;
            st[OBS_DEC_S_D_IDX] = OBS_MASK_AIM_WALK;
            st[OBS_DEC_S_D_TRIG] = 0;
            st[OBS_DEC_S_D_HAND] = OBS_MASK_HAND_NOTHING;
            st[OBS_DEC_S_D_ARG] = 0;
            continue;
        }
        const int64_t feet = a[0], aim = a[1], trig = a[2], arg = a[4];
        int64_t hand = a[3];
        // -- masked_choices: the four served masks and the realised arg_mask
        int cnt = 0;
        cnt += m_feet[er * OBS_MASK_N_FEET + feet] == 0;
        cnt += m_aim[er * n_aim + aim] == 0;
        cnt += m_trig[er * OBS_MASK_N_TRIGGER + trig] == 0;
        cnt += m_hand[er * n_hand + hand] == 0;
        {
            const bool throw_ = hand >= OBS_MASK_HAND_THROW0;
            const bool four = throw_ || hand == OBS_MASK_HAND_PLANT;
            bool legal;
            if (arg < OBS_DEC_N_FRACTIONS) legal = four || arg == 0;
            else legal = throw_ && aim >= OBS_MASK_AIM_TOKENS;     // arg == ARG_ONTO
            cnt += legal ? 0 : 1;
        }
        masked[er] = (uint8_t)cnt;
        // -- set_decisions: the aim, the pointer -> a game id, the fallbacks
        st[OBS_DEC_S_D_FEET] = feet;
        st[OBS_DEC_S_D_TRIG] = trig;
        st[OBS_DEC_S_D_ARG] = arg;
        st[OBS_DEC_S_D_LPV] = 0;
        st[OBS_DEC_S_D_LPX] = 0;
        st[OBS_DEC_S_D_LPY] = 0;
        if (aim < OBS_MASK_AIM_WALK) {
            st[OBS_DEC_S_D_KIND] = OBS_DEC_AIM_TOK;
            st[OBS_DEC_S_D_VAL] = aim;
            st[OBS_DEC_S_D_IDX] = aim;
        } else if (aim == OBS_MASK_AIM_KEEP) {
            st[OBS_DEC_S_D_KIND] = OBS_DEC_AIM_KEEP;
            st[OBS_DEC_S_D_VAL] = -1;
            st[OBS_DEC_S_D_IDX] = OBS_MASK_AIM_KEEP;
        } else {
            st[OBS_DEC_S_D_KIND] = OBS_DEC_AIM_WALK;
            st[OBS_DEC_S_D_VAL] = -1;
            st[OBS_DEC_S_D_IDX] = OBS_MASK_AIM_WALK;
            const int64_t i = aim - OBS_MASK_AIM_TOKENS;
            if (i >= 0) {
                const int64_t uid = list_uid[er * n_entries + i];
                if (uid >= 0 && tb[OBS_DEC_TAB_AIM_LO] <= i && i < tb[OBS_DEC_TAB_AIM_HI]) {
                    st[OBS_DEC_S_D_KIND] = OBS_DEC_AIM_UNIT;
                    st[OBS_DEC_S_D_VAL] = uid;
                    st[OBS_DEC_S_D_IDX] = OBS_MASK_AIM_TOKENS;
                }
            }
        }
        if (hand == OBS_MASK_HAND_FLASHLIGHT) {
            hand = OBS_MASK_HAND_NOTHING;
        } else if (hand >= OBS_MASK_HAND_THROW0) {
            if (arg == OBS_DEC_ARG_ONTO && st[OBS_DEC_S_D_KIND] != OBS_DEC_AIM_UNIT)
                hand = OBS_MASK_HAND_NOTHING;
        } else if (hand == OBS_MASK_HAND_PLANT && arg == OBS_DEC_ARG_ONTO) {
            hand = OBS_MASK_HAND_NOTHING;
        }
        st[OBS_DEC_S_D_HAND] = hand;
    }
}

// ---------------------------------------------------------------------------
// THE DECODE: one thread per (env, row) of a decoder-driven team.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void put_move(int64_t* m, bool set, int64_t dx, int64_t dy,
                                         int64_t thr) {
    int64_t* c = m + UNIT_COL_LIVE_MOVE_DIR;          // (set, dx_q, dy_q, throttle_q)
    c[0] = set ? 1 : 0;
    c[1] = set ? dx : 0;
    c[2] = set ? dy : 0;
    c[3] = set ? thr : 0;
}

__global__ void k_decode(int n_env, int64_t* store, int U, int n_rows, int S, int T,
                         const int32_t* rowmap, const int64_t* trow, const uint8_t* ended,
                         const int64_t* pic, int W, const int64_t* dtab, const uint8_t* drive,
                         int64_t* state, int32_t* aimed, int brk) {
    const int e = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const int t = r / S;
    if (ended[e] || !drive[(size_t)e * T + t]) return;
    const size_t er = (size_t)e * n_rows + r;
    int64_t* st = state + er * OBS_DEC_S_W;
    if (!st[OBS_DEC_S_HAVE]) return;                 // no decision handed over yet
    // the policy's first-tick flag is consumed on its next decoded tick,
    // whichever of its units live
    const bool first = st[OBS_DEC_S_FIRST] != 0 || brk == OBS_DEC_BREAK_DEC_EVERY;
    st[OBS_DEC_S_FIRST] = 0;
    const int sr = rowmap[er];
    if (sr < 0 || sr >= U) return;
    int64_t* me = store + ((size_t)e * U + sr) * UNIT_STORE_W;
    // a dead or turned row issues NOTHING, ever
    if (me[UNIT_COL_LIFE_STATE] != OBS_LIFE_ALIVE || me[UNIT_COL_TEAM] != (int64_t)t) return;
    st[OBS_DEC_S_EXISTS] = 1;
    st[OBS_DEC_S_UID] = me[UNIT_COL_ID];
    const int64_t tick = trow[(size_t)e * unit_draws::TICK_ROW_W + unit_draws::TICK_ROW_TICK];
    const int64_t* tb = dtab + (size_t)e * OBS_DEC_TAB_W;
    const int64_t* blk = pic + (size_t)e * W;
    const PicLayout L = pic_layout(T, U);
    const bool div_brk = brk == OBS_DEC_BREAK_DEC_DIV;
    const int64_t release = me[UNIT_COL_THROW_RELEASE_TICK];
    // -- the aim held through a wind-up: the game's own state decides
    if (st[OBS_DEC_S_TA] && release < tick) st[OBS_DEC_S_TA] = 0;
    int64_t hand = st[OBS_DEC_S_D_HAND];
    const int64_t dkind = st[OBS_DEC_S_D_KIND];
    const bool dec_is_d = dkind != OBS_DEC_AIM_KEEP || st[OBS_DEC_S_TA];   // else _WALK
    if (first && hand >= OBS_MASK_HAND_THROW0 && release < 0) {
        st[OBS_DEC_S_TA] = 1;
        if (dec_is_d) {
            st[OBS_DEC_S_TA_ALIAS] = 1;                // the decision's own aim object
        } else {
            st[OBS_DEC_S_TA_ALIAS] = 0;                // _WALK
            st[OBS_DEC_S_TA_KIND] = OBS_DEC_AIM_WALK;
            st[OBS_DEC_S_TA_VAL] = -1;
            st[OBS_DEC_S_TA_LPV] = 0;
        }
    }
    int64_t walk[5] = {OBS_DEC_AIM_WALK, -1, 0, 0, 0};
    int64_t* A;
    if (st[OBS_DEC_S_TA])
        A = st[OBS_DEC_S_TA_ALIAS] ? st + OBS_DEC_S_D_KIND : st + OBS_DEC_S_TA_KIND;
    else
        A = dec_is_d ? st + OBS_DEC_S_D_KIND : walk;
    if (A[0] == OBS_DEC_AIM_KEEP) A = walk;
    // -- feet
    const int64_t feet = st[OBS_DEC_S_D_FEET];
    const bool stand = feet == OBS_MASK_FEET_STAND;
    const int64_t dir = stand ? 0 : (feet - 1) / OBS_DEC_N_SPEEDS;
    const int64_t spd = stand ? 0 : (feet - 1) % OBS_DEC_N_SPEEDS;
    if (stand) {
        put_move(me, false, 0, 0, 0);                  // clear_move_dir
    } else {
        int64_t thr = tb[OBS_DEC_TAB_THROTTLE0 + spd];  // set_move_dir clamps to [0, ONE]
        thr = thr < 0 ? 0 : (thr > OBS_THROTTLE_ONE ? OBS_THROTTLE_ONE : thr);
        put_move(me, true, c_compass[2 * dir], c_compass[2 * dir + 1], thr);
    }
    // -- aim
    int64_t vx = 0, vy = 0, aim_uid = -1;
    const int64_t cx = obs_sight::centre_x(me), cy = obs_sight::centre_y(me);
    if (A[0] == OBS_DEC_AIM_TOK) {
        vx = c_compass[2 * A[1]];
        vy = c_compass[2 * A[1] + 1];
    } else if (A[0] == OBS_DEC_AIM_UNIT) {
        int64_t qx, qy;
        if (point_of(A, blk, L, U, t) && aim_vector(A[3] - cx, A[4] - cy, div_brk, &qx, &qy)) {
            vx = qx;
            vy = qy;
        } else {
            vx = st[OBS_DEC_S_LVX];
            vy = st[OBS_DEC_S_LVY];
        }
        aim_uid = A[1];
    }
    st[OBS_DEC_S_LVX] = vx;
    st[OBS_DEC_S_LVY] = vy;
    st[OBS_DEC_S_AIMED] = aim_uid;
    aimed[er] = (int32_t)aim_uid;
    int64_t* la = me + UNIT_COL_LIVE_AIM;              // set_aim: (1, dx_q, dy_q)
    la[0] = 1;
    la[1] = vx;
    la[2] = vy;
    me[UNIT_COL_LIVE_TRIGGER] = st[OBS_DEC_S_D_TRIG] ? 1 : 0;
    me[UNIT_COL_FLASHLIGHT_ON] = 1;                    // set_flashlight(off=False)
    // -- hand: a latch, issued on the decision's first tick only
    if (!first) return;
    int64_t arg_used = -1;
    const int64_t arg = st[OBS_DEC_S_D_ARG];
    if (hand == OBS_MASK_HAND_RELOAD) {
        me[UNIT_COL_PENDING_RELOAD] = 1;
    } else if (hand == OBS_MASK_HAND_MELEE) {
        me[UNIT_COL_PENDING_MELEE] = 1;
    } else if (hand == OBS_MASK_HAND_SHOVE) {
        me[UNIT_COL_PENDING_SHOVE] = 1;
    } else if (hand == OBS_MASK_HAND_USE) {
        me[UNIT_COL_PENDING_USE] = 1;
    } else if (hand == OBS_MASK_HAND_PLANT) {
        const int64_t timer = tb[OBS_DEC_TAB_TIMER0 + arg];
        if (timer != 0) {                              // the 'nonzero' gate, then the clamp
            const int64_t lo = tb[OBS_DEC_TAB_PLANT_LO], hi = tb[OBS_DEC_TAB_PLANT_HI];
            me[UNIT_COL_PENDING_PLANT] = timer < lo ? lo : (timer > hi ? hi : timer);
        }
        arg_used = arg;
    } else if (hand >= OBS_MASK_HAND_THROW0) {
        bool have = true;
        int64_t reach;
        if (arg == OBS_DEC_ARG_ONTO) {
            int64_t* D = st + OBS_DEC_S_D_KIND;        // d.aim, the decision's own aim
            have = point_of(D, blk, L, U, t);
            reach = have ? onto_reach_q(D[3] - cx, D[4] - cy) : 0;
        } else {
            reach = tb[OBS_DEC_TAB_REACH0 + arg];
        }
        if (!have) {
            hand = OBS_MASK_HAND_NOTHING;
        } else {
            const int64_t f = me[UNIT_COL_FACING_Q];
            const int64_t fx = fixedpoint::cos_q16((int32_t)f), fy = -fixedpoint::sin_q16((int32_t)f);
            if (fx != 0 || fy != 0) {                  // the 'vector' gate
                int64_t* pt = me + UNIT_COL_PENDING_THROW;   // (1, dx, dy, fuse, reach, kind)
                pt[0] = 1;
                pt[1] = fx;
                pt[2] = fy;
                pt[3] = tb[OBS_DEC_TAB_FUSE_TICKS];
                pt[4] = reach;
                pt[5] = hand - OBS_MASK_HAND_THROW0;
            }
            arg_used = arg;
        }
    }
    int64_t aim_idx = st[OBS_DEC_S_D_IDX];
    if (aim_idx == OBS_MASK_AIM_KEEP && dkind == OBS_DEC_AIM_KEEP && !dec_is_d)
        aim_idx = OBS_MASK_AIM_WALK;
    st[OBS_DEC_S_PREV0] = stand ? OBS_DEC_N_DIRS : dir;
    st[OBS_DEC_S_PREV1] = stand ? -1 : spd;
    st[OBS_DEC_S_PREV2] = aim_idx;
    st[OBS_DEC_S_PREV3] = st[OBS_DEC_S_D_TRIG] ? 1 : 0;
    st[OBS_DEC_S_PREV4] = hand;
    st[OBS_DEC_S_PREV5] = arg_used;
    st[OBS_DEC_S_HAS_PREV] = 1;
}

// ---------------------------------------------------------------------------
// THE HORDE: ZombieControl's request set, per env the card serves.
// ---------------------------------------------------------------------------
__global__ void k_horde(int n_env, const int64_t* store, int U, const int32_t* mode,
                        int64_t* rows, int row_w) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_env) return;
    const int m = mode[e];
    if (m != OBS_DEC_HORDE_DEVICE && m != OBS_DEC_HORDE_IDLE) return;
    int64_t* R = rows + (size_t)e * row_w;
    bool any = false;
    for (int q = 0; q < U && OBS_DEC_HORDE_BASE + q < row_w; ++q) {
        const int64_t* u = store + ((size_t)e * U + q) * UNIT_STORE_W;
        const bool z = m == OBS_DEC_HORDE_DEVICE && u[UNIT_COL_VALID] &&
                       u[UNIT_COL_LIFE_STATE] == OBS_LIFE_ALIVE && u[UNIT_COL_IS_ZOMBIE];
        R[OBS_DEC_HORDE_BASE + q] = z ? 1 : 0;
        any = any || z;
    }
    for (int q = OBS_DEC_HORDE_BASE + U; q < row_w; ++q) R[q] = 0;
    R[OBS_DEC_HORDE_MODE] = any ? OBS_DEC_HORDE_SET : OBS_DEC_HORDE_EMPTY;
}

// ---------------------------------------------------------------------------
// THE OBSERVATION'S COLUMNS: one thread per (env, row), on a decision tick.
// ---------------------------------------------------------------------------
__global__ void k_dec_obs(const int64_t* store, int U, const int64_t* tables, const int32_t* rowmap,
                          int n_rows, int S, int T, const int64_t* otab, const int64_t* trow,
                          const int64_t* params, const uint8_t* ended, const int64_t* pic, int W,
                          const int64_t* state, const int32_t* aimed, const uint8_t* list_mask,
                          const int32_t* list_uid, int n_entries, float* list, float* vector) {
    const int e = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows || !env_due(e, trow, params, ended)) return;
    const size_t er = (size_t)e * n_rows + r;
    float* V = vector + er * OBS_V;
    const int t = r / S;
    const int sr = rowmap[er];
    const int64_t* rows = store + (size_t)e * U * UNIT_STORE_W;
    const bool valid = sr >= 0 && sr < U &&
                       rows[(size_t)sr * UNIT_STORE_W + UNIT_COL_LIFE_STATE] == OBS_LIFE_ALIVE &&
                       rows[(size_t)sr * UNIT_STORE_W + UNIT_COL_TEAM] == (int64_t)t;
    const int64_t* st = state + er * OBS_DEC_S_W;
    // -- the vector's prev block (cols 71-76): -1 before the first decision
    for (int i = 0; i < 6; ++i)
        V[OBS_V_PREV_0 + i] = !valid ? 0.0f
                              : (st[OBS_DEC_S_EXISTS] && st[OBS_DEC_S_HAS_PREV])
                                    ? (float)st[OBS_DEC_S_PREV0 + i] : -1.0f;
    if (!valid) return;
    // -- the aimed entry: prev_aim (col 66) and on_next_tick (col 28)
    const int64_t want_uid = aimed[er];
    if (want_uid < 0) return;
    const int caps0 = S - 1;
    const int units_end = caps0 + OBS_CAP_SEEN + OBS_CAP_RECORD;
    const uint8_t* M = list_mask + er * n_entries;
    const int32_t* UID = list_uid + er * n_entries;
    int k = -1;
    for (int i = 0; i < units_end && i < n_entries; ++i)
        if (M[i] != 0 && (int64_t)UID[i] == want_uid) {
            k = i;
            break;
        }
    if (k < 0) return;
    float* E = list + (er * n_entries + k) * OBS_F;
    E[OBS_L_PREV_AIM] = 1.0f;
    if (k < caps0) return;                            // a mate's place: no record read
    const int64_t* blk = pic + (size_t)e * W;
    const PicLayout L = pic_layout(T, U);
    const int q = record_row(blk, L, U, t, want_uid);
    if (q < 0) return;
    const int64_t* rc = obs_pic::rec_of(blk, L, U, t, q);
    const int64_t* me = rows + (size_t)sr * UNIT_STORE_W;
    int64_t vx, vy;
    if (!aim_vector(rc[OBS_PIC_REC_CENTRE_X] - obs_sight::centre_x(me),
                    rc[OBS_PIC_REC_CENTRE_Y] - obs_sight::centre_y(me), false, &vx, &vy))
        return;
    // the bots' own test (control_bots._on_target): the facing after this
    // tick's turn toward the aim (turning.predict: the law at the unit's
    // facing, omega + its pending spin, its own a / w) within its shot cone
    const int64_t* ot = otab + (size_t)e * OBS_TAB_W;
    const int64_t* tab = tables + (size_t)e * unit_tables::UNIT_TABLES_W;
    const int64_t want = fixedpoint::atan2_q16((int32_t)(-vy), (int32_t)vx);
    int64_t a, w, dw;
    obs_rows::turn_dials(me, tab, &a, &w, &dw);
    int64_t after, o2;
    bool bad = false;
    unit_lib::turn_law_q16(me[UNIT_COL_FACING_Q], me[UNIT_COL_OMEGA_Q] + dw, want, a, w, &after,
                           &o2, &bad);
    const int64_t wr = me[UNIT_COL_WEAPON_ID];
    const int64_t spread = wr >= 0 ? unit_lib::scaled(me, unit_tables::UNIT_HOOK_SPREAD,
                                                      ot[OBS_TAB_WEAPON + wr * OBS_WEAPON_W +
                                                         OBS_WEAPON_CONE_AIM_Q])
                                   : 0;
    const int64_t d = unit_lib::wrap_q16(want - after);
    E[OBS_L_ON_NEXT_TICK] = (d < 0 ? -d : d) <= spread ? 1.0f : 0.0f;
}

__global__ void k_probe(int n, const int64_t* dx, const int64_t* dy, int64_t* out, int div_brk) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t qx = 0, qy = 0;
    const bool ok = aim_vector(dx[i], dy[i], div_brk != 0, &qx, &qy);
    out[4 * i + 0] = qx;
    out[4 * i + 1] = qy;
    out[4 * i + 2] = ok ? 0 : 1;
    out[4 * i + 3] = onto_reach_q(dx[i], dy[i]);
}

}  // namespace

void obs_dec_check(int n_env, int n_rows, int n_slots, std::uintptr_t decisions,
                   std::uintptr_t due, std::uintptr_t list_uid, int n_entries,
                   std::uintptr_t mask_feet, std::uintptr_t mask_aim, std::uintptr_t mask_trigger,
                   std::uintptr_t mask_hand, int n_kinds, std::uintptr_t dec_table,
                   std::uintptr_t state, std::uintptr_t masked, std::uintptr_t faults) {
    const char* who = "obs_dec_check";
    need(n_env >= 1 && n_rows >= 1 && n_slots >= 1 && n_kinds >= 0, who, "a non-positive size");
    need(n_rows % n_slots == 0, who, "n_rows is not a multiple of n_slots");
    need(n_entries == n_slots + OBS_ENTRIES_BEYOND_SLOTS, who,
         "n_entries is not n_slots + OBS_ENTRIES_BEYOND_SLOTS");
    need(decisions && due && list_uid && mask_feet && mask_aim && mask_trigger && mask_hand &&
             dec_table && state && masked && faults, who, "a null device pointer");
    k_check<<<(n_env + 63) / 64, 64, 0, tstream()>>>(
        n_env, n_rows, reinterpret_cast<const int32_t*>(decisions),
        reinterpret_cast<const uint8_t*>(due), reinterpret_cast<const int32_t*>(list_uid),
        n_entries, reinterpret_cast<const uint8_t*>(mask_feet),
        reinterpret_cast<const uint8_t*>(mask_aim), reinterpret_cast<const uint8_t*>(mask_trigger),
        reinterpret_cast<const uint8_t*>(mask_hand), n_kinds,
        reinterpret_cast<const int64_t*>(dec_table), reinterpret_cast<int64_t*>(state),
        reinterpret_cast<uint8_t*>(masked), reinterpret_cast<int64_t*>(faults));
    launched(who);
}

bool obs_decode(int n_env, std::uintptr_t store, int n_units, int store_w,
                const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
                int n_teams, std::uintptr_t obs_table, int table_w, std::uintptr_t tick_row,
                std::uintptr_t ended, std::uintptr_t pic_state, int pic_w, std::uintptr_t dec_table,
                std::uintptr_t drive, std::uintptr_t state, std::uintptr_t aimed,
                std::uintptr_t faults, int brk) {
    const char* who = "obs_decode";
    need(n_env >= 1 && n_units >= 1 && n_rows >= 1 && n_slots >= 1 && n_teams >= 1, who,
         "a non-positive size");
    need(n_rows == n_teams * n_slots, who, "n_rows is not n_teams x n_slots");
    need(store_w == UNIT_STORE_W, who, "the store's width is not UNIT_STORE_W");
    need(table_w == OBS_TAB_W, who, "the obs table's width is not OBS_TAB_W");
    need(pic_w == pic_layout(n_teams, n_units).w, who, "the picture block's width");
    need(tables.n_env() == n_env && tables.width() == unit_tables::UNIT_TABLES_W, who,
         "the device tables' shape");
    need(store && rowmap && obs_table && tick_row && ended && pic_state && dec_table && drive &&
             state && aimed && faults, who, "a null device pointer");
    k_decode<<<dim3((n_rows + 63) / 64, n_env), 64, 0, tstream()>>>(
        n_env, reinterpret_cast<int64_t*>(store), n_units, n_rows, n_slots, n_teams,
        reinterpret_cast<const int32_t*>(rowmap), reinterpret_cast<const int64_t*>(tick_row),
        reinterpret_cast<const uint8_t*>(ended), reinterpret_cast<const int64_t*>(pic_state),
        pic_w, reinterpret_cast<const int64_t*>(dec_table),
        reinterpret_cast<const uint8_t*>(drive), reinterpret_cast<int64_t*>(state),
        reinterpret_cast<int32_t*>(aimed), brk);
    launched(who);
    // the out-of-range words ride the conductor's one books read
    return books_defer("decoder_faults", reinterpret_cast<const void*>(faults),
                       (size_t)n_env * OBS_DEC_FAULT_W);
}

void obs_dec_horde(int n_env, std::uintptr_t store, int n_units, int store_w,
                   std::uintptr_t mode, std::uintptr_t rows, int row_w) {
    const char* who = "obs_dec_horde";
    need(n_env >= 1 && n_units >= 1 && row_w >= OBS_DEC_HORDE_BASE + n_units, who,
         "a size (the latch row holds the mode and a flag per store row)");
    need(store_w == UNIT_STORE_W, who, "the store's width is not UNIT_STORE_W");
    need(store && mode && rows, who, "a null device pointer");
    k_horde<<<(n_env + 63) / 64, 64, 0, tstream()>>>(
        n_env, reinterpret_cast<const int64_t*>(store), n_units,
        reinterpret_cast<const int32_t*>(mode), reinterpret_cast<int64_t*>(rows), row_w);
    launched(who);
}

void obs_dec_obs(int n_env, std::uintptr_t store, int n_units, int store_w,
                 const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
                 int n_teams, std::uintptr_t obs_table, int table_w, std::uintptr_t tick_row,
                 std::uintptr_t params, std::uintptr_t ended, std::uintptr_t pic_state, int pic_w,
                 std::uintptr_t state, std::uintptr_t aimed, std::uintptr_t list_mask,
                 std::uintptr_t list_uid, int n_entries, std::uintptr_t list,
                 std::uintptr_t vector) {
    const char* who = "obs_dec_obs";
    need(n_env >= 1 && n_units >= 1 && n_rows >= 1 && n_slots >= 1 && n_teams >= 1, who,
         "a non-positive size");
    need(n_rows == n_teams * n_slots, who, "n_rows is not n_teams x n_slots");
    need(n_entries == n_slots + OBS_ENTRIES_BEYOND_SLOTS, who,
         "n_entries is not n_slots + OBS_ENTRIES_BEYOND_SLOTS");
    need(store_w == UNIT_STORE_W, who, "the store's width is not UNIT_STORE_W");
    need(table_w == OBS_TAB_W, who, "the obs table's width is not OBS_TAB_W");
    need(pic_w == pic_layout(n_teams, n_units).w, who, "the picture block's width");
    need(tables.n_env() == n_env && tables.width() == unit_tables::UNIT_TABLES_W, who,
         "the device tables' shape");
    need(store && rowmap && obs_table && tick_row && params && ended && pic_state && state &&
             aimed && list_mask && list_uid && list && vector, who, "a null device pointer");
    k_dec_obs<<<dim3((n_rows + 63) / 64, n_env), 64, 0, tstream()>>>(
        reinterpret_cast<const int64_t*>(store), n_units, tables.values(),
        reinterpret_cast<const int32_t*>(rowmap), n_rows, n_slots, n_teams,
        reinterpret_cast<const int64_t*>(obs_table), reinterpret_cast<const int64_t*>(tick_row),
        reinterpret_cast<const int64_t*>(params), reinterpret_cast<const uint8_t*>(ended),
        reinterpret_cast<const int64_t*>(pic_state), pic_w, reinterpret_cast<const int64_t*>(state),
        reinterpret_cast<const int32_t*>(aimed), reinterpret_cast<const uint8_t*>(list_mask),
        reinterpret_cast<const int32_t*>(list_uid), n_entries, reinterpret_cast<float*>(list),
        reinterpret_cast<float*>(vector));
    launched(who);
}

void obs_dec_probe(int n, std::uintptr_t dx, std::uintptr_t dy, std::uintptr_t out, int div_brk) {
    const char* who = "obs_dec_probe";
    need(n >= 0 && (n == 0 || (dx && dy && out)), who, "a null device pointer");
    if (n == 0) return;
    k_probe<<<(n + 255) / 256, 256, 0, tstream()>>>(
        n, reinterpret_cast<const int64_t*>(dx), reinterpret_cast<const int64_t*>(dy),
        reinterpret_cast<int64_t*>(out), div_brk);
    launched(who);
}

}  // namespace breach_cuda
