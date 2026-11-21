// ==========================================================================
// #29 P8.4b: THE LIST (see cuda_obs.h), Class T. The twin of
// obs_spec_v1.build_list (spec 2.2) line for line, every env on its decision
// tick with its ended latch clear, in ONE launch on THE tick stream: one
// thread per (env, row, block). The host builder is the reference.
//
// THE FIVE BLOCKS at fixed places of the row's E = S + OBS_ENTRIES_BEYOND_SLOTS
// entries: team-mates (cap S - 1), enemies seen now (OBS_CAP_SEEN), last-seen
// records (OBS_CAP_RECORD), grenades and charges (OBS_CAP_OBJECT), doors
// (OBS_CAP_DOOR). Each block's candidates (_candidates):
//   mates    the living non-zombie units of the team but the soldier (live
//            centres), key (d2, uid);
//   seen     the team's records of units it sees now (TeamKnowledge.seen),
//            key (d2 of the RECORDED centre, uid);
//   records  the team's other records, the same key;
//   objects  the team's object tracker's seen_now (the picture's SEEN list;
//            the object found on the live grenade / charge rows by its
//            serial), key (d2 of its tile, its uid);
//   doors    the living doors of the door table, key (d2 of the span's
//            middle tile, ordinal).
// THE SORT: every key is two integers and the second is unique within a block
// (a unit id, an object uid, a door ordinal), so the order is TOTAL; the card
// takes it by successive minima (the k-th entry is the least key greater than
// the (k-1)-th), which is Python's sort of the same tuples, and stops at the
// block's capacity: the host's [:cap] after the sort. No float, no tie left.
//
// THE ARITHMETIC: integers where the reference is integer; every float column
// float64 in the reference's order through the kit (cuda_obs_float.cuh: never
// fused), cast ONCE to float32; d_c's log1p and the squash's log are CUDA's
// (design "P8 RE-CUT, VERSION 2": the gate compares served raw bits).
// The answers P8.4a computed (aim_of, the line of fire, ticks_to_within, the
// own lanes) are read from their buffers, never recomputed.
//
// Columns owned by later patches stay 0 here: on_next_tick / prev_aim (the
// decoder's, P8.6a). The team-mates' order block 47-51 (P8.6b) is
// order_tail, the full view's own function, called after mate_entry.
//
// #29 P8.4c: THE FOUR MASKS (obs_masks, k_masks below) live in this TU too,
// one launch after the list's.
//
// #29 P8.8: THE FULL VIEW (obs_full: k_full_aim, k_full_list, k_full_watched,
// k_full_match below) lives here too, beside the entry builders it calls.
// ==========================================================================
#include "cuda_obs.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "cuda_batch.cuh"          // tstream(): THE tick stream (blocking)
#include "cuda_object_table.h"     // ObjectTable (the grenade and charge tables)
#include "cuda_obs_float.cuh"
#include "cuda_obs_picture.cuh"
#include "cuda_obs_rows.cuh"
#include "cuda_obs_sight.cuh"
#include "cuda_unit_events.h"      // UnitRecordList (the door tables)
#include "cuda_unit_lib.cuh"       // the unit library's integer reads (used, never added to)
#include "cuda_unit_store.h"       // UnitTablesDevice
#include "fixed_point.h"
#include "obs_layout.h"
#include "object_tables_layout.h"
#include "unit_draws_layout.h"
#include "unit_store_layout.h"
#include "unit_tables_layout.h"

namespace breach_cuda {

using namespace obs_layout;

static_assert(OBS_LIST_OBJECT_UID_BASE == OBS_OBJECT_UID_BASE, "one object uid base");
static_assert(OBS_NO_TIMER == -1, "the charge table's NO_TIMER sentinel");

namespace {

using namespace obs_f64;
using namespace unit_store;
using obs_pic::PicLayout;
using obs_pic::pic_layout;
using obs_rows::env_due;
using obs_rows::is_mate;
using obs_rows::is_seen;
using obs_sight::floordiv;
using obs_sight::tile_of;

// A block's sort key: (squared centre distance, id). `rev`: THE BREAK
// list_tie (ties by the higher id), never in a real run.
struct Key {
    int64_t d2, id;
};
__device__ __forceinline__ bool key_less(const Key& a, const Key& b, bool rev) {
    if (a.d2 != b.d2) return a.d2 < b.d2;
    return rev ? a.id > b.id : a.id < b.id;
}

// The least key of the candidates c in [0, n) that is greater than `prev`
// (when `have_prev`): returns its candidate (or -1), its key into *out.
template <class KeyOf>
__device__ int next_min(int n, KeyOf key_of, bool have_prev, const Key& prev, bool rev,
                        Key* out) {
    int best = -1;
    Key bk{0, 0};
    for (int c = 0; c < n; ++c) {
        Key k;
        if (!key_of(c, &k)) continue;
        if (have_prev && !key_less(prev, k, rev)) continue;
        if (best >= 0 && !key_less(k, bk, rev)) continue;
        best = c;
        bk = k;
    }
    *out = bk;
    return best;
}

__device__ __forceinline__ void put(float* E, int col, double x) { E[col] = to_f32(x); }

// float(x) / float(d), 0 when d is 0 (obs_spec_v1.ratio over float operands).
__device__ __forceinline__ double fratio(double x, double d) { return d == 0.0 ? 0.0 : div(x, d); }

// One soldier's view (the row's): what every block reads.
struct View {
    const int64_t* rows;   // the env's store rows
    int U, T, t, n_rows, r, e;
    const int64_t* me;     // the soldier's store row
    int sr;
    int64_t me_id, cx, cy, tick, lref, n_t;
    const int64_t* ot;     // the env's obs table
    const int64_t* tab;    // the env's device tables
    const int64_t* blk;    // the env's picture block
    PicLayout L;
    const uint8_t* vis_t;  // team t's visible row (U)
    double cap;            // the squash cap
};

// _entry_common: type, dx / dy over L_ref (squashed), d_c of the length.
__device__ void entry_common(float* E, int type, int64_t dx, int64_t dy, const View& v) {
    put(E, OBS_L_TYPE, (double)type);
    put(E, OBS_L_DX, squash(ratio(dx, v.lref), v.cap));
    put(E, OBS_L_DY, squash(ratio(dy, v.lref), v.cap));
    const double fx = (double)dx, fy = (double)dy;
    const double d = __dsqrt_rn(add(mul(fx, fx), mul(fy, fy)));        // obs_spec_v1.length
    put(E, OBS_L_DIST_C, squash(div(log1p(d > 0.0 ? d : 0.0), log1p((double)v.lref)), v.cap));
}

// cos / sin of a Q16 angle over FP_ONE (the kit's integer cos / sin).
__device__ __forceinline__ void put_angle(float* E, int col_c, int col_s, int64_t f) {
    put(E, col_c, div((double)fixedpoint::cos_q16((int32_t)f), (double)OBS_FP_ONE));
    put(E, col_s, div((double)fixedpoint::sin_q16((int32_t)f), (double)OBS_FP_ONE));
}

// _unit_columns: cols 11-23 of a living unit (a mate or a seen enemy).
__device__ void unit_columns(float* E, const int64_t* it, const View& v) {
    const int64_t* ot = v.ot;
    put(E, OBS_L_EXPOSURE, ratio(it[UNIT_COL_EXPOSURE + OBS_KINETIC], ot[OBS_TAB_X_MAX_Q]));
    put(E, OBS_L_FIRING, it[UNIT_COL_LIVE_TRIGGER] != 0 ? 1.0 : 0.0);
    bool bad = false;
    const int64_t w = it[UNIT_COL_WEAPON_ID];
    if (w >= 0) {
        const unit_lib::MagStatus m = unit_lib::mag_status(it, v.tab, v.tick, &bad);
        if (m.tracked && m.reloading) {
            put(E, OBS_L_RELOADING, 1.0);
            put(E, OBS_L_RELOAD_LEFT,
                ratio(it[UNIT_COL_RELOAD_DONE_TICK] - v.tick,
                      unit_lib::handling_ticks(
                          it, ot[OBS_TAB_WEAPON + w * OBS_WEAPON_W + OBS_WEAPON_RELOAD_TICKS])));
        }
    }
    const int64_t release = it[UNIT_COL_THROW_RELEASE_TICK];
    if (release >= 0) {
        put(E, OBS_L_WINDING, 1.0);
        put(E, OBS_L_THROW_LEFT, ratio(release - v.tick,
                                       unit_lib::handling_ticks(it, ot[OBS_TAB_HAND_WINDUP_TICKS])));
        put(E, OBS_L_THROW_KIND, (double)it[UNIT_COL_THROW_KIND]);
    } else {
        put(E, OBS_L_THROW_KIND, -1.0);
    }
    const int64_t pdone = it[UNIT_COL_PLANT_DONE_TICK];
    if (pdone >= 0) {
        put(E, OBS_L_PLANTING, 1.0);
        put(E, OBS_L_PLANT_LEFT,
            ratio(pdone - v.tick, unit_lib::handling_ticks(it, ot[OBS_TAB_PLANT_TICKS])));
        put(E, OBS_L_PLANT_TIMER, ratio(it[UNIT_COL_PLANT_FUSE_TICKS], ot[OBS_TAB_C4_FUSE_MAX_TICKS]));
    }
    const unit_lib::Flags fl = unit_lib::composed_flags(it, v.tab, false, &bad);
    put(E, OBS_L_STUNNED, fl.can_act ? 0.0 : 1.0);
    put(E, OBS_L_BLINDED, fl.can_aim ? 0.0 : 1.0);
    put(E, OBS_L_DOWN, fl.is_prone ? 1.0 : 0.0);
    (void)bad;   // the store's encoder refuses what would set it (as the vector's)
}

// Type 0: the team-mate of store row k (cols 0-23, 39-46, 52-53).
// `aim_is_unit`: the caller's aim answer for it (the list's aim_of, the full
// view's every-unit aim); `lanes`: the lane pair is written (the full view's
// own entry, the soldier itself, marks nothing: a CHOICE of the reference).
__device__ void mate_entry(float* E, int k, const View& v, bool aim_is_unit, bool lanes,
                           const uint8_t* lane_own, int gh, int gw) {
    const int64_t* m = v.rows + (size_t)k * UNIT_STORE_W;
    const int64_t mx = obs_sight::centre_x(m), my = obs_sight::centre_y(m);
    entry_common(E, OBS_TYPE_MATE, mx - v.cx, my - v.cy, v);
    put_angle(E, OBS_L_COS, OBS_L_SIN, m[UNIT_COL_FACING_Q]);
    const int64_t* tr = obs_pic::unit_of(v.blk, v.L, k);
    int64_t vx, vy;
    obs_pic::motion_of(tr, &vx, &vy);
    put(E, OBS_L_VEL_X, squash(ratio(vx, v.ot[OBS_TAB_V_REF_Q]), v.cap));
    put(E, OBS_L_VEL_Y, squash(ratio(vy, v.ot[OBS_TAB_V_REF_Q]), v.cap));
    put(E, OBS_L_SIDE, (double)m[UNIT_COL_TEAM]);
    put(E, OBS_L_CLASS, (double)m[UNIT_COL_LOADOUT_NAME]);      // NO_ROW == _row_index's -1
    put(E, OBS_L_HP, obs_pic::hp_over_bar(m[UNIT_COL_CURRENT_HP], tr, v.cap));
    unit_columns(E, m, v);
    // -- _mate_tail: its facing target, its aim, its feet, where its throw lands
    const int64_t ft = m[UNIT_COL_FACING_TARGET_Q];
    put_angle(E, OBS_L_AIM_COS, OBS_L_AIM_SIN, ft);
    put(E, OBS_L_AIM_IS_UNIT, aim_is_unit ? 1.0 : 0.0);
    const int64_t* mv = m + UNIT_COL_LIVE_MOVE_DIR;            // (set, dx_q, dy_q, throttle_q)
    if (mv[0]) {
        put(E, OBS_L_FEET_DX, div((double)mv[1], (double)OBS_FP_ONE));
        put(E, OBS_L_FEET_DY, div((double)mv[2], (double)OBS_FP_ONE));
        put(E, OBS_L_THROTTLE, ratio(mv[3], OBS_THROTTLE_ONE));
    }
    if (m[UNIT_COL_THROW_RELEASE_TICK] >= 0) {
        const int64_t rq = m[UNIT_COL_THROW_REACH_Q];
        const double reach = rq == OBS_THROW_NO_REACH ? 6.0 : div((double)rq, (double)OBS_FP_ONE);
        const double c = div((double)fixedpoint::cos_q16((int32_t)ft), (double)OBS_FP_ONE);
        const double s = div((double)fixedpoint::sin_q16((int32_t)ft), (double)OBS_FP_ONE);
        const double lx = sub(add((double)mx, mul(reach, c)), (double)v.cx);
        const double ly = sub(sub((double)my, mul(reach, s)), (double)v.cy);
        put(E, OBS_L_LAND_DX, squash(fratio(lx, (double)v.lref), v.cap));
        put(E, OBS_L_LAND_DY, squash(fratio(ly, (double)v.lref), v.cap));
    }
    // -- the lane pair: each owner's OWN rasterised lane (P8.4a's lane_own)
    if (!lanes) return;
    const int64_t cell = v.ot[OBS_TAB_LANE_CELL];
    const size_t cells = (size_t)gh * gw;
    const int64_t gy0 = floordiv(v.cy, cell), gx0 = floordiv(v.cx, cell);
    const int64_t gy1 = floordiv(my, cell), gx1 = floordiv(mx, cell);
    const uint8_t* own_m = lane_own + ((size_t)v.e * v.U + k) * cells;
    const uint8_t* own_me = lane_own + ((size_t)v.e * v.U + v.sr) * cells;
    const bool in0 = 0 <= gy0 && gy0 < gh && 0 <= gx0 && gx0 < gw;
    const bool in1 = 0 <= gy1 && gy1 < gh && 0 <= gx1 && gx1 < gw;
    put(E, OBS_L_ME_IN_ITS_LANE, in0 && own_m[gy0 * gw + gx0] ? 1.0 : 0.0);
    put(E, OBS_L_IT_IN_MY_LANE, in1 && own_me[gy1 * gw + gx1] ? 1.0 : 0.0);
}

// The enemy columns both views share (cols 27, 29-38) of a unit at (ex, ey),
// offset (dx, dy) from the soldier: the ticks to bring its aim on (`ticks`,
// turning.ticks_to_within's answer), the reach flags, the aim category of the
// unit's own aim answer `hit` (0 nobody, 1 me, 2 a living mate, 3 another),
// the zombie state, the other mates aiming at it and firing (`mate_aim(k)`:
// mate row k's aim answer), its damage over its bar (`tr` its trackers), the
// priority flag, the mates in the frag radius.
template <class MateAim>
__device__ void enemy_tail(float* E, const View& v, const int64_t* it, const int64_t* tr,
                           int64_t dx, int64_t dy, int64_t ex, int64_t ey, int64_t euid,
                           int64_t ticks, int32_t hit, MateAim mate_aim) {
    // -- _enemy_geometry: the ticks over the half turn, the reach flags
    put(E, OBS_L_TICKS_TO_ON, squash(ratio(ticks, v.ot[OBS_TAB_HALF_TURN_TICKS]), v.cap));
    const int64_t d2 = dx * dx + dy * dy;
    const int64_t wr = v.me[UNIT_COL_WEAPON_ID];
    int64_t range2 = -1;
    if (wr >= 0) {
        const int64_t rt = v.ot[OBS_TAB_WEAPON + wr * OBS_WEAPON_W + OBS_WEAPON_RANGE_TILES];
        range2 = rt * rt;
    }
    const int64_t thr = v.ot[OBS_TAB_MAX_THROW_RANGE];
    put(E, OBS_L_IN_RANGE, d2 <= range2 ? 1.0 : 0.0);
    put(E, OBS_L_IN_THROW_RANGE, d2 <= thr * thr ? 1.0 : 0.0);
    // -- aim_of's category: 0 nobody, 1 me, 2 a living mate, 3 another
    int cat = 0;
    if (hit >= 0) {
        if ((int64_t)hit == v.me_id) {
            cat = 1;
        } else {
            cat = 3;
            for (int k = 0; k < v.U; ++k) {
                const int64_t* m = v.rows + (size_t)k * UNIT_STORE_W;
                if (is_mate(m, v.t) && m[UNIT_COL_ID] == (int64_t)hit) {
                    cat = 2;
                    break;
                }
            }
        }
    }
    put(E, OBS_L_FACING_ON_ME, cat == 1 ? 1.0 : 0.0);
    put(E, OBS_L_AIMS_AT, (double)cat);
    if (it[UNIT_COL_IS_ZOMBIE])
        put(E, OBS_L_ZOMBIE_STATE, it[UNIT_COL_ZOMBIE_ACTIVATED] ? 2.0
                                   : it[UNIT_COL_ZOMBIE_NOISE_TICKS] > 0 ? 3.0 : 1.0);
    // -- the other mates aiming at it (and firing); every mate in the frag radius
    const int64_t fr = v.ot[OBS_TAB_FRAG_RADIUS];
    int64_t aiming = 0, firing = 0, catch_n = 0;
    for (int k = 0; k < v.U; ++k) {
        const int64_t* m = v.rows + (size_t)k * UNIT_STORE_W;
        if (!is_mate(m, v.t)) continue;
        const int64_t fx = obs_sight::centre_x(m) - ex, fy = obs_sight::centre_y(m) - ey;
        if (fx * fx + fy * fy <= fr * fr) ++catch_n;
        if (m[UNIT_COL_ID] == v.me_id) continue;
        if ((int64_t)mate_aim(k) == euid) {
            ++aiming;
            if (m[UNIT_COL_LIVE_TRIGGER]) ++firing;
        }
    }
    put(E, OBS_L_MATES_AIMING, ratio(aiming, v.n_t));
    put(E, OBS_L_MATES_FIRING, ratio(firing, v.n_t));
    put(E, OBS_L_DMG_LAST_S,
        squash(fratio(obs_pic::damage_sum(v.blk, v.L, v.t, euid, false), obs_pic::bar_of(tr)),
               v.cap));
    put(E, OBS_L_PRIORITY, v.me[UNIT_COL_ORDER_TARGET] == euid ? 1.0 : 0.0);
    put(E, OBS_L_FRAG_CATCH, ratio(catch_n, v.n_t));
}

// Type 1: the enemy of store row q the team sees now (cols 0-27, 29-38): its
// RECORDED centre, live facing and body, recorded velocity, team, class, the
// P8.4a answers. `aim_e` (T, U), `sees_me` (U: who the soldier sees),
// `lof_r` / `ttw_r` (U: the soldier's row).
__device__ void seen_entry(float* E, int q, const View& v, const int32_t* aim_e,
                           const uint8_t* sees_me, const uint8_t* lof_r, const int32_t* ttw_r) {
    const int64_t* it = v.rows + (size_t)q * UNIT_STORE_W;
    const int64_t* rc = obs_pic::rec_of(v.blk, v.L, v.U, v.t, q);
    const int64_t ex = rc[OBS_PIC_REC_CENTRE_X], ey = rc[OBS_PIC_REC_CENTRE_Y];
    const int64_t euid = rc[OBS_PIC_REC_UID];
    const int64_t dx = ex - v.cx, dy = ey - v.cy;
    entry_common(E, OBS_TYPE_SEEN, dx, dy, v);
    put_angle(E, OBS_L_COS, OBS_L_SIN, it[UNIT_COL_FACING_Q]);
    put(E, OBS_L_VEL_X, squash(ratio(rc[OBS_PIC_REC_VEL_X], v.ot[OBS_TAB_V_REF_Q]), v.cap));
    put(E, OBS_L_VEL_Y, squash(ratio(rc[OBS_PIC_REC_VEL_Y], v.ot[OBS_TAB_V_REF_Q]), v.cap));
    put(E, OBS_L_SIDE, (double)rc[OBS_PIC_REC_TEAM]);
    put(E, OBS_L_CLASS, rc[OBS_PIC_REC_ZOMBIE] ? (double)v.ot[OBS_TAB_ZOMBIE_CLASS]
                                                : (double)rc[OBS_PIC_REC_LOADOUT]);
    const int64_t* tr = obs_pic::unit_of(v.blk, v.L, q);
    put(E, OBS_L_HP, obs_pic::hp_over_bar(it[UNIT_COL_CURRENT_HP], tr, v.cap));
    unit_columns(E, it, v);
    put(E, OBS_L_SEEN_BY_ME, sees_me[q] ? 1.0 : 0.0);
    const uint8_t lf = lof_r[q];
    put(E, OBS_L_LOF_SOLID, (lf & OBS_LOF_SOLID) ? 1.0 : 0.0);
    put(E, OBS_L_LOF_MATE, (lf & OBS_LOF_MATE) ? 1.0 : 0.0);
    enemy_tail(E, v, it, tr, dx, dy, ex, ey, euid, ttw_r[q], aim_e[(size_t)v.t * v.U + q],
               [&](int k) { return aim_e[(size_t)v.t * v.U + k]; });
}

// Type 2: the team's record of store row q, not seen now (cols 0-9, 54).
__device__ void record_entry(float* E, int q, const View& v) {
    const int64_t* rc = obs_pic::rec_of(v.blk, v.L, v.U, v.t, q);
    entry_common(E, OBS_TYPE_RECORD, rc[OBS_PIC_REC_CENTRE_X] - v.cx,
                 rc[OBS_PIC_REC_CENTRE_Y] - v.cy, v);
    put_angle(E, OBS_L_COS, OBS_L_SIN, rc[OBS_PIC_REC_FACING]);
    put(E, OBS_L_VEL_X, squash(ratio(rc[OBS_PIC_REC_VEL_X], v.ot[OBS_TAB_V_REF_Q]), v.cap));
    put(E, OBS_L_VEL_Y, squash(ratio(rc[OBS_PIC_REC_VEL_Y], v.ot[OBS_TAB_V_REF_Q]), v.cap));
    put(E, OBS_L_SIDE, (double)rc[OBS_PIC_REC_TEAM]);
    put(E, OBS_L_CLASS, rc[OBS_PIC_REC_ZOMBIE] ? (double)v.ot[OBS_TAB_ZOMBIE_CLASS]
                                                : (double)rc[OBS_PIC_REC_LOADOUT]);
    put(E, OBS_L_AGE, ratio(v.tick - rc[OBS_PIC_REC_TICK], v.ot[OBS_TAB_MEMORY_TICKS]));
}

// The live grenade / charge row holding the tracker key (serial) `key`:
// kind 0 a grenade row, 1 a charge row; false when none (a fault).
struct Objs {
    const int64_t* g;      // the env's grenade rows
    int64_t gc;
    const int64_t* c;      // the env's charge rows
    int64_t cc;
};
__device__ bool object_row(const Objs& o, const View& v, int64_t key, int* kind, int64_t* i) {
    using namespace object_tables;
    for (int64_t k = 0; k < o.gc; ++k)
        if (o.g[k * OBJ_GRENADE_W + OBJ_GRENADE_LIVE] && v.blk[v.L.side_g + k] == key) {
            *kind = 0;
            *i = k;
            return true;
        }
    for (int64_t k = 0; k < o.cc; ++k)
        if (o.c[k * OBJ_CHARGE_W + OBJ_CHARGE_LIVE] && v.blk[v.L.side_c + k] == key) {
            *kind = 1;
            *i = k;
            return true;
        }
    return false;
}

// team_picture.object_tile as (x, y): a grenade at its position's tile, a
// charge at (x, y).
__device__ void object_tile(const Objs& o, int kind, int64_t i, int64_t* ox, int64_t* oy) {
    using namespace object_tables;
    if (kind == 0) {
        const int64_t* g = o.g + i * OBJ_GRENADE_W;
        *ox = tile_of(g[OBJ_GRENADE_COL_POS_X_Q]);
        *oy = tile_of(g[OBJ_GRENADE_COL_POS_Y_Q]);
    } else {
        const int64_t* c = o.c + i * OBJ_CHARGE_W;
        *ox = c[OBJ_CHARGE_COL_X];
        *oy = c[OBJ_CHARGE_COL_Y];
    }
}

// Type 3: _object_entry (cols 0-9, 55-61). `roster` (S): the team's roster
// store rows (-1: empty).
__device__ void object_entry(float* E, const Objs& o, int kind, int64_t i, const View& v,
                             const int32_t* roster, int S, const int64_t* rows_all) {
    using namespace object_tables;
    int64_t ox, oy;
    object_tile(o, kind, i, &ox, &oy);
    const int64_t dx = ox - v.cx, dy = oy - v.cy;
    entry_common(E, OBS_TYPE_OBJECT, dx, dy, v);
    const int64_t* g = kind == 0 ? o.g + i * OBJ_GRENADE_W : nullptr;
    const int64_t* c = kind == 1 ? o.c + i * OBJ_CHARGE_W : nullptr;
    const int64_t ammo = g ? g[OBJ_GRENADE_COL_AMMO_NAME] : c[OBJ_CHARGE_COL_AMMO_NAME];
    const int64_t* A = (ammo >= 0 && ammo < OBS_TAB_N_AMMO)
                           ? v.ot + OBS_TAB_AMMO + ammo * OBS_AMMO_W : nullptr;
    bool landed = false;
    if (g) {
        const double ddx = div((double)(g[OBJ_GRENADE_COL_TARGET_X_Q] - g[OBJ_GRENADE_COL_START_X_Q]),
                               65536.0);   // target_fx - start_fx: exact on the Q16 grid
        const double ddy = div((double)(g[OBJ_GRENADE_COL_TARGET_Y_Q] - g[OBJ_GRENADE_COL_START_Y_Q]),
                               65536.0);
        const double dist = __dsqrt_rn(add(mul(ddx, ddx), mul(ddy, ddy)));
        landed = g[OBJ_GRENADE_COL_POS_X_Q] == g[OBJ_GRENADE_COL_TARGET_X_Q] &&
                 g[OBJ_GRENADE_COL_POS_Y_Q] == g[OBJ_GRENADE_COL_TARGET_Y_Q];
        if (dist >= 0.1 && !landed) {
            const double ts = A ? of_bits(A[OBS_AMMO_TRAVEL_SPEED_BITS]) : 0.0;
            const double vt = div(div(ts, (double)v.ot[OBS_TAB_TPS]), dist);
            const double vref = div((double)v.ot[OBS_TAB_V_REF_Q], (double)OBS_FP_ONE);
            put(E, OBS_L_VEL_X, squash(fratio(mul(ddx, vt), vref), v.cap));
            put(E, OBS_L_VEL_Y, squash(fratio(mul(ddy, vt), vref), v.cap));
        }
    }
    put(E, OBS_L_SIDE, -1.0);
    put(E, OBS_L_CLASS, -1.0);
    put(E, OBS_L_OBJ_KIND, A ? (double)A[OBS_AMMO_OBJ_KIND] : -1.0);
    const int64_t owner = g ? g[OBJ_GRENADE_COL_OWNER_ID] : c[OBJ_CHARGE_COL_OWNER_ID];
    double whose = owner >= 0 ? 2.0 : 3.0;
    if (owner == v.me_id) {
        whose = 0.0;
    } else {
        for (int s = 0; s < S; ++s) {
            const int32_t rr = roster[s];
            if (rr >= 0 && rr < v.U && rows_all[(size_t)rr * UNIT_STORE_W + UNIT_COL_ID] == owner) {
                whose = 1.0;
                break;
            }
        }
    }
    put(E, OBS_L_WHOSE, whose);
    if (c) {
        const int64_t det = c[OBJ_CHARGE_COL_DET_TICK];
        if (det == OBS_NO_TIMER)
            put(E, OBS_L_REMOTE, 1.0);
        else
            put(E, OBS_L_TIME_LEFT,
                squash(ratio(det - v.tick, v.ot[OBS_TAB_C4_FUSE_MAX_TICKS]), v.cap));
        put(E, OBS_L_ON_FLOOR, 1.0);
    } else {
        // Projectile.get_detonate_tick: thrown + fuse_ticks, else thrown +
        // int(fuse_seconds * tps) (fuse_seconds an exact Q16 value)
        const int64_t ft = g[OBJ_GRENADE_COL_FUSE_TICKS];
        const int64_t det = g[OBJ_GRENADE_COL_THROWN_TICK] +
            (ft != OBS_NO_TIMER ? ft
                                : (int64_t)mul(div((double)g[OBJ_GRENADE_COL_FUSE_SECONDS], 65536.0),
                                               (double)v.ot[OBS_TAB_TPS]));
        put(E, OBS_L_TIME_LEFT, squash(ratio(det - v.tick, v.ot[OBS_TAB_HAND_FUSE_MAX_TICKS]), v.cap));
        put(E, OBS_L_ON_FLOOR, landed ? 1.0 : 0.0);
    }
    const int64_t rad = A ? A[OBS_AMMO_RADIUS] : 0;
    put(E, OBS_L_BLAST_R, squash(ratio(rad, v.lref), v.cap));
    put(E, OBS_L_IN_BLAST, dx * dx + dy * dy <= rad * rad ? 1.0 : 0.0);
}

// The door table's rows and span tiles of one env, the planes the hull reads.
struct Doors {
    const int64_t* d;      // (cap, OBJ_DOOR_W)
    int64_t n;
    const int64_t* tiles;  // (tile_cap, OBJ_DOOR_TILE_W)
    int64_t tile_cap;
    const bool* vac;       // (h, w) is_vacuum
    const bool* amb;       // (h, w) is_ambient
    int h, w;
    const uint8_t* wired;  // (wired_cap,)
    int wired_cap;
};

__device__ __forceinline__ void span_tile(const Doors& D, int64_t ti, int64_t* ty, int64_t* tx) {
    using namespace object_tables;
    const int64_t* t = D.tiles + ti * OBJ_DOOR_TILE_W;
    *ty = t[OBJ_DOOR_TILE_COL_TY];
    *tx = t[OBJ_DOOR_TILE_COL_TX];
}

// _door_centre (x, y): the span's middle tile (upper-middle of an even span);
// false for an empty span or one off the tile table.
__device__ bool door_centre(const Doors& D, const int64_t* row, int64_t* x, int64_t* y) {
    using namespace object_tables;
    const int64_t a = row[OBJ_DOOR_COL_TILE_START], n = row[OBJ_DOOR_COL_TILE_COUNT];
    if (n <= 0 || a < 0 || a + n > D.tile_cap) return false;
    span_tile(D, a + n / 2, y, x);
    return true;
}

// Type 4: _door_entry (cols 0-9, 62-65) of door table row k: open, in the use
// verb's reach of the soldier (verb_gates.door_in_reach: a span tile on its
// footprint or its one-tile ring), a hull door (horde.is_hull_door: a span
// tile 4-bordering, outside the span, the map's edge, vacuum or the ambient
// ring), wired. The reach is the unit library's unit_lib::reach_key (the use
// verb's own geometry, P8.4c), never restated here.
__device__ void door_entry(float* E, int k, const Doors& D, const View& v) {
    using namespace object_tables;
    const int64_t* row = D.d + (size_t)k * OBJ_DOOR_W;
    int64_t x, y;
    door_centre(D, row, &x, &y);
    entry_common(E, OBS_TYPE_DOOR, x - v.cx, y - v.cy, v);
    put(E, OBS_L_SIDE, -1.0);
    put(E, OBS_L_CLASS, -1.0);
    put(E, OBS_L_OPEN, row[OBJ_DOOR_COL_STATE] == OBS_DOOR_STATE_OPEN ? 1.0 : 0.0);
    const int64_t a = row[OBJ_DOOR_COL_TILE_START], n = row[OBJ_DOOR_COL_TILE_COUNT];
    bool reach = false;
    for (int64_t i = a; i < a + n; ++i) {
        int64_t ty, tx;
        span_tile(D, i, &ty, &tx);
        if (unit_lib::reach_key(v.me, ty, tx) >= 0) reach = true;
    }
    // horde.is_hull_door: the observation's one copy (obs_rows::door_hull,
    // P8.6b's route reads it too)
    const bool hull = obs_rows::door_hull(D.tiles, a, n, D.vac, D.amb, D.h, D.w);
    put(E, OBS_L_IN_REACH, reach ? 1.0 : 0.0);
    put(E, OBS_L_HULL, hull ? 1.0 : 0.0);
    put(E, OBS_L_WIRED, (k < D.wired_cap && D.wired[k]) ? 1.0 : 0.0);
}

// _mate_tail's order block (cols 47-51) of team-mate store row m: its order's
// kind, the place's centre from me (_place_centre over the env's place table:
// rooms by id, openings by ORDER_OPENING_BASE + id), the same place as my
// order's, the order's clock.
__device__ void order_tail(float* E, const int64_t* m, const View& v, const int64_t* places,
                           int n_places) {
    const int64_t what = m[UNIT_COL_ORDER_WHAT];
    put(E, OBS_L_ORDER_WHAT, (double)what);
    if (!what) return;
    const int64_t where = m[UNIT_COL_ORDER_WHERE];
    for (int p = 0; p < n_places; ++p) {
        const int64_t* pl = places + (size_t)p * OBS_PLACE_W;
        if (pl[OBS_PLACE_WHERE] != where) continue;
        put(E, OBS_L_ORDER_PLACE_DX, squash(ratio(pl[OBS_PLACE_X] - v.cx, v.lref), v.cap));
        put(E, OBS_L_ORDER_PLACE_DY, squash(ratio(pl[OBS_PLACE_Y] - v.cy, v.lref), v.cap));
        break;
    }
    const int64_t my_where = v.me[UNIT_COL_ORDER_WHAT] ? v.me[UNIT_COL_ORDER_WHERE] : OBS_NO_PLACE;
    put(E, OBS_L_ORDER_SAME_PLACE, (my_where != OBS_NO_PLACE && where == my_where) ? 1.0 : 0.0);
    const int64_t until = m[UNIT_COL_ORDER_UNTIL];
    if (until != OBS_NO_LIMIT)
        put(E, OBS_L_ORDER_LEFT, ratio(until - v.tick, v.ot[OBS_TAB_LIMIT_TICKS]));
}

// One thread per (env, row r, block b): block b's entries of row r, zeroed
// first (observe_rows' out[...] = 0, mask 0, uid -1), then, for a valid row,
// its candidates in key order up to the block's capacity.
__global__ void k_list(const int64_t* store, int U, const int64_t* tables, const int32_t* rowmap,
                       int n_rows, int S, int T, const int64_t* otab, const int64_t* teams,
                       const int64_t* trow, const int64_t* params, const uint8_t* ended,
                       const uint8_t* visible, const uint8_t* sees, const int64_t* state, int W,
                       const int32_t* aim, const uint8_t* lof, const int32_t* ttw,
                       const uint8_t* lane_own, int gh, int gw, const int64_t* grenades,
                       const int64_t* g_count, const int64_t* charges, const int64_t* c_count,
                       const int64_t* doors, const int64_t* door_count, int door_cap,
                       const int64_t* door_tiles, int tile_cap, const bool* vac, const bool* amb,
                       int h, int w, const uint8_t* wired, int wired_cap,
                       const int64_t* places, int place_cap, const int32_t* place_count,
                       int n_entries, float* list, uint8_t* list_mask, int32_t* list_uid,
                       int32_t* faults, int brk) {
    const int e = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * 5 || !env_due(e, trow, params, ended)) return;
    const int r = idx / 5, b = idx % 5;
    const int caps[5] = {S - 1, OBS_CAP_SEEN, OBS_CAP_RECORD, OBS_CAP_OBJECT, OBS_CAP_DOOR};
    int start = 0;
    for (int k = 0; k < b; ++k) start += caps[k];
    const size_t er = (size_t)e * n_rows + r;
    float* LST = list + er * n_entries * OBS_F;
    uint8_t* MSK = list_mask + er * n_entries;
    int32_t* UID = list_uid + er * n_entries;
    for (int i = start; i < start + caps[b]; ++i) {
        for (int f = 0; f < OBS_F; ++f) LST[(size_t)i * OBS_F + f] = 0.0f;
        MSK[i] = 0;
        UID[i] = -1;
    }
    const int t = r / S;
    const int sr = rowmap[er];
    const int64_t* rows = store + (size_t)e * U * UNIT_STORE_W;
    if (sr < 0 || sr >= U) return;
    const int64_t* me = rows + (size_t)sr * UNIT_STORE_W;
    if (me[UNIT_COL_LIFE_STATE] != OBS_LIFE_ALIVE || me[UNIT_COL_TEAM] != (int64_t)t) return;
    View v;
    v.rows = rows;
    v.U = U;
    v.T = T;
    v.t = t;
    v.n_rows = n_rows;
    v.r = r;
    v.e = e;
    v.me = me;
    v.sr = sr;
    v.me_id = me[UNIT_COL_ID];
    v.cx = obs_sight::centre_x(me);
    v.cy = obs_sight::centre_y(me);
    v.tick = trow[(size_t)e * unit_draws::TICK_ROW_W + unit_draws::TICK_ROW_TICK];
    v.ot = otab + (size_t)e * OBS_TAB_W;
    v.lref = v.ot[OBS_TAB_L_REF];
    v.n_t = teams[((size_t)e * T + t) * OBS_TEAM_W + OBS_TEAM_N];
    v.tab = tables + (size_t)e * unit_tables::UNIT_TABLES_W;
    v.blk = state + (size_t)e * W;
    v.L = pic_layout(T, U);
    v.vis_t = visible + ((size_t)e * (T + 1) + t) * U;
    v.cap = of_bits(OBS_SQUASH_CAP_BITS);
    const bool rev = brk == OBS_LIST_BREAK_LIST_TIE;
    int cap_b = caps[b];
    if (b == 4 && brk == OBS_LIST_BREAK_LIST_CAP) cap_b -= 1;
    const int32_t* aim_e = aim + (size_t)e * T * U;
    Key prev{0, 0};
    bool have = false;
    auto d2_of = [&](int64_t x, int64_t y) {
        const int64_t dx = x - v.cx, dy = y - v.cy;
        return dx * dx + dy * dy;
    };
    for (int i = 0; i < cap_b; ++i) {
        float* E = LST + (size_t)(start + i) * OBS_F;
        Key k;
        int c = -1;
        int64_t uid = -1;
        if (b == 0) {                                          // team-mates
            c = next_min(U, [&](int q, Key* o) {
                const int64_t* m = rows + (size_t)q * UNIT_STORE_W;
                if (!is_mate(m, t) || m[UNIT_COL_ID] == v.me_id) return false;
                *o = Key{d2_of(obs_sight::centre_x(m), obs_sight::centre_y(m)), m[UNIT_COL_ID]};
                return true;
            }, have, prev, rev, &k);
            if (c < 0) break;
            mate_entry(E, c, v, aim_e[(size_t)t * U + c] != OBS_AIM_NONE, true, lane_own, gh, gw);
            // P8.6b: _mate_tail's order block, the full view's own function
            order_tail(E, rows + (size_t)c * UNIT_STORE_W, v,
                       places + (size_t)e * place_cap * OBS_PLACE_W, place_count[e]);
            uid = k.id;
        } else if (b == 1 || b == 2) {                         // seen enemies / records
            const bool want_seen = b == 1;
            c = next_min(U, [&](int q, Key* o) {
                const int64_t* rc = obs_pic::rec_of(v.blk, v.L, U, t, q);
                if (!rc[OBS_PIC_REC_PRESENT]) return false;
                if ((v.vis_t[q] != 0) != want_seen) return false;
                *o = Key{d2_of(rc[OBS_PIC_REC_CENTRE_X], rc[OBS_PIC_REC_CENTRE_Y]),
                         rc[OBS_PIC_REC_UID]};
                return true;
            }, have, prev, rev, &k);
            if (c < 0) break;
            if (want_seen)
                seen_entry(E, c, v, aim_e, sees + ((size_t)e * U + sr) * U,
                           lof + er * U, ttw + er * U);
            else
                record_entry(E, c, v);
            uid = k.id;
        } else if (b == 3) {                                   // grenades and charges
            using namespace object_tables;
            Objs o{grenades + (size_t)e * OBJ_GRENADE_CAP * OBJ_GRENADE_W, g_count[e],
                   charges + (size_t)e * OBJ_CHARGE_CAP * OBJ_CHARGE_W, c_count[e]};
            const int64_t* seen = v.blk + v.L.seen + (int64_t)t * OBS_PIC_OBJ_CAP * 2;
            const int n_seen = (int)v.blk[v.L.seen_n + t];
            c = next_min(n_seen, [&](int s, Key* out) {
                int kind;
                int64_t row;
                if (!object_row(o, v, seen[2 * s], &kind, &row)) {
                    atomicOr(faults + e, OBS_LIST_FAULT_OBJECT);
                    return false;
                }
                int64_t ox, oy;
                object_tile(o, kind, row, &ox, &oy);
                *out = Key{d2_of(ox, oy), seen[2 * s + 1]};
                return true;
            }, have, prev, rev, &k);
            if (c < 0) break;
            int kind;
            int64_t row;
            object_row(o, v, seen[2 * c], &kind, &row);
            object_entry(E, o, kind, row, v, rowmap + (size_t)e * n_rows + (size_t)t * S, S, rows);
            uid = k.id;
        } else {                                               // doors
            using namespace object_tables;
            Doors D{doors + (size_t)e * door_cap * OBJ_DOOR_W, door_count[e],
                    door_tiles + (size_t)e * tile_cap * OBJ_DOOR_TILE_W, tile_cap,
                    vac + (size_t)e * h * w, amb + (size_t)e * h * w, h, w,
                    wired + (size_t)e * wired_cap, wired_cap};
            c = next_min((int)(D.n < door_cap ? D.n : door_cap), [&](int q, Key* o) {
                const int64_t* row = D.d + (size_t)q * OBJ_DOOR_W;
                if (!row[OBJ_DOOR_COL_ALIVE]) return false;
                int64_t x, y;
                if (!door_centre(D, row, &x, &y)) return false;
                *o = Key{d2_of(x, y), row[OBJ_DOOR_COL_ORDINAL]};
                return true;
            }, have, prev, rev, &k);
            if (c < 0) break;
            door_entry(E, c, D, v);
            uid = (int64_t)OBS_DOOR_UID_BASE + k.id;
        }
        MSK[start + i] = 1;
        UID[start + i] = (int32_t)uid;
        prev = k;
        have = true;
    }
}


// ==========================================================================
// #29 P8.4c: THE FOUR MASKS (action_spec_v1.build_masks), one thread per
// (env, row), every env on its decision tick with its ended latch clear, read
// AFTER the list of the same tick (the aim mask reads its type and mask). The
// state is the decision tick's start (the walk's end of tick T - 1 is the
// reference's start of tick T: the tick row already holds T). A row that is
// not valid (no roster unit, dead, turned) reads default_masks: stand,
// "where I walk", release, nothing. Each hand bit is ONE unit-library gate
// (unit_lib::can_reload, can_melee at the tick's start, can_shove, can_plant,
// can_throw per kind; use: acts() and a live door with a span tile at
// unit_lib::reach_key >= 0 -- verb_gates.can_use, whose use_door returns a
// door iff one is in reach); never a copy of a decision.
// ==========================================================================

// verb_gates.can_use's question on the door table: a live door (row order =
// sim._doors) whose span holds a tile in the unit's reach. A live door whose
// span leaves the door-tile table sets *bad.
__device__ bool door_in_reach_any(const int64_t* me, const int64_t* drows, int64_t n_doors,
                                  const int64_t* tiles, int64_t n_tiles, bool* bad) {
    using namespace object_tables;
    for (int64_t q = 0; q < n_doors; ++q) {
        const int64_t* row = drows + (size_t)q * OBJ_DOOR_W;
        if (!row[OBJ_DOOR_COL_ALIVE]) continue;
        const int64_t a = row[OBJ_DOOR_COL_TILE_START], c = row[OBJ_DOOR_COL_TILE_COUNT];
        if (a < 0 || c < 0 || a + c > n_tiles) {
            *bad = true;
            return false;
        }
        for (int64_t i = a; i < a + c; ++i) {
            const int64_t* t = tiles + (size_t)i * OBJ_DOOR_TILE_W;
            if (unit_lib::reach_key(me, t[OBJ_DOOR_TILE_COL_TY], t[OBJ_DOOR_TILE_COL_TX]) >= 0)
                return true;
        }
    }
    return false;
}

__global__ void k_masks(const int64_t* store, int U, const int64_t* tables, const int32_t* rowmap,
                        int n_rows, int S, const int64_t* trow, const int64_t* params,
                        const uint8_t* ended, const float* list, const uint8_t* list_mask,
                        int n_entries, const int64_t* doors, const int64_t* door_count,
                        int door_cap, const int64_t* door_tiles, const int64_t* tile_count,
                        int tile_cap, int n_kinds, uint8_t* m_feet, uint8_t* m_aim,
                        uint8_t* m_trig, uint8_t* m_hand, int32_t* faults, int brk) {
    const int e = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows || !env_due(e, trow, params, ended)) return;
    const size_t er = (size_t)e * n_rows + r;
    const int n_aim = OBS_MASK_AIM_TOKENS + n_entries;
    const int n_hand = OBS_MASK_HAND_THROW0 + n_kinds;
    uint8_t* feet = m_feet + er * OBS_MASK_N_FEET;
    uint8_t* aim = m_aim + er * n_aim;
    uint8_t* trig = m_trig + er * OBS_MASK_N_TRIGGER;
    uint8_t* hand = m_hand + er * n_hand;
    for (int i = 0; i < OBS_MASK_N_FEET; ++i) feet[i] = 0;
    for (int i = 0; i < n_aim; ++i) aim[i] = 0;
    for (int i = 0; i < OBS_MASK_N_TRIGGER; ++i) trig[i] = 0;
    for (int i = 0; i < n_hand; ++i) hand[i] = 0;
    const int t = r / S;
    const int sr = rowmap[er];
    const int64_t* me = store + ((size_t)e * U + (sr < 0 ? 0 : sr)) * UNIT_STORE_W;
    if (sr < 0 || sr >= U || me[UNIT_COL_LIFE_STATE] != OBS_LIFE_ALIVE ||
        me[UNIT_COL_TEAM] != (int64_t)t) {
        // default_masks: exactly one legal option per head
        feet[OBS_MASK_FEET_STAND] = 1;
        aim[OBS_MASK_AIM_WALK] = 1;
        trig[0] = 1;
        hand[OBS_MASK_HAND_NOTHING] = 1;
        return;
    }
    const int64_t* tab = tables + (size_t)e * unit_tables::UNIT_TABLES_W;
    const int64_t tick = trow[(size_t)e * unit_draws::TICK_ROW_W + unit_draws::TICK_ROW_TICK];
    const int64_t dec = params[(size_t)e * OBS_PARAM_W + OBS_PARAM_DECIMATION];
    for (int i = 0; i < OBS_MASK_N_FEET; ++i) feet[i] = 1;
    for (int i = 0; i < OBS_MASK_N_TRIGGER; ++i) trig[i] = 1;
    // the aim: the wind-up holds it over the coming decision's every tick
    if (me[UNIT_COL_THROW_RELEASE_TICK] >= tick + dec - 1) {
        aim[OBS_MASK_AIM_KEEP] = 1;
    } else {
        for (int i = 0; i < OBS_MASK_AIM_KEEP; ++i) aim[i] = 1;
        const float* L = list + er * (size_t)n_entries * OBS_F;
        const uint8_t* M = list_mask + er * (size_t)n_entries;
        for (int i = 0; i < n_entries; ++i) {
            const int64_t ty = (int64_t)L[(size_t)i * OBS_F + OBS_L_TYPE];   // astype: truncation
            const bool aimable = ty >= 0 && ty < 31 && ((OBS_MASK_AIMABLE_BITS >> ty) & 1);
            aim[OBS_MASK_AIM_TOKENS + i] = (M[i] != 0 && aimable) ? 1 : 0;
        }
    }
    // the hand: ONE gate per bit
    bool bad = false;
    const bool reload = unit_lib::can_reload(me, tab, tick, &bad);
    const bool shove = unit_lib::can_shove(me, tab, tick, &bad);
    const bool swap = brk == OBS_MASK_BREAK_MASK_SWAP;     // THE BREAK: two gates swapped
    hand[OBS_MASK_HAND_NOTHING] = 1;
    hand[OBS_MASK_HAND_RELOAD] = (swap ? shove : reload) ? 1 : 0;
    hand[OBS_MASK_HAND_MELEE] = unit_lib::can_melee(me, tab, tick, true, &bad) ? 1 : 0;
    hand[OBS_MASK_HAND_SHOVE] = (swap ? reload : shove) ? 1 : 0;
    {
        using namespace object_tables;
        int64_t wide[UNIT_STORE_W];
        const int64_t* who = me;
        if (brk == OBS_MASK_BREAK_MASK_REACH) {            // THE BREAK: the reach one tile wider
            for (int c = 0; c < UNIT_STORE_W; ++c) wide[c] = me[c];
            wide[UNIT_COL_FOOTPRINT] += 1;
            who = wide;
        }
        const int64_t nd = door_count[e], nt = tile_count[e];
        bool dbad = nd < 0 || nd > door_cap || nt < 0 || nt > tile_cap;
        const bool use = !dbad && unit_lib::acts(me) &&
                         door_in_reach_any(who, doors + (size_t)e * door_cap * OBJ_DOOR_W, nd,
                                           door_tiles + (size_t)e * tile_cap * OBJ_DOOR_TILE_W,
                                           nt, &dbad);
        if (dbad) atomicOr(faults + e, OBS_MASK_FAULT_DOOR);
        hand[OBS_MASK_HAND_USE] = (use && !dbad) ? 1 : 0;
    }
    hand[OBS_MASK_HAND_PLANT] = unit_lib::can_plant(me, tab, tick, &bad) ? 1 : 0;
    hand[OBS_MASK_HAND_FLASHLIGHT] = 0;                    // reserved (design 2.6)
    for (int k = 0; k < n_kinds; ++k)
        hand[OBS_MASK_HAND_THROW0 + k] = unit_lib::can_throw(me, tab, k, &bad) ? 1 : 0;
    if (bad) atomicOr(faults + e, OBS_MASK_FAULT_GATE);
}


// ==========================================================================
// #29 P8.8: THE FULL VIEW (obs_spec_v1.FullView / build_full_list /
// build_full_watched / FullView.match, spec 5), the value network's view, the
// OTHER SIDE OF THE WALL: every living unit where it truly is. Every env on
// its decision tick with its ended latch clear, on THE tick stream, no sync.
// The entry builders are the list's own (entry_common, unit_columns,
// mate_entry, enemy_tail, object_entry, door_entry: called, never copied);
// the full view's differences are its own: which units (EVERY living unit,
// never the team's visible set), their LIVE centres and motion, the order
// (the marines by (team, slot, id), the zombies, objects and doors nearest
// first by (d2, id)), the capacities (FULL_ENTRIES, the farthest doors, then
// zombies, then objects dropped, never a marine), the every-unit aim
// (FullView.aim over every living unit, its own buffer `full_aim`), the
// extra columns (knowledge, kit, four statuses, order) and the order block of
// a mate entry (cols 47-51, the place table).
//   full_aim (N, 2, U) int32: FullView.aim of store row q along its facing
//     (0) / its facing TARGET (1): the nearest other living unit within q's
//     cone of that ray, by (d2, uid, cx, cy), to which the sight line is
//     clear; -1 nobody (and every row that is not a living unit).
// ==========================================================================

// FullView.aim for (env, ray kind, store row q).
__global__ void k_full_aim(const int64_t* store, int U, const int64_t* otab, const int64_t* tables,
                           const uint8_t* solid, const uint8_t* screen, int h, int w,
                           const int64_t* trow, const int64_t* params, const uint8_t* ended,
                           int32_t* full_aim, int32_t* faults) {
    const int e = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 * U || !env_due(e, trow, params, ended)) return;
    const int kind = idx / U, q = idx % U;
    int32_t* out = full_aim + ((size_t)e * 2 + kind) * U + q;
    const int64_t* rows = store + (size_t)e * U * UNIT_STORE_W;
    const int64_t* u = rows + (size_t)q * UNIT_STORE_W;
    *out = OBS_AIM_NONE;
    if (!u[UNIT_COL_VALID] || u[UNIT_COL_LIFE_STATE] != OBS_LIFE_ALIVE) return;
    const int64_t* ot = otab + (size_t)e * OBS_TAB_W;
    const int64_t* tab = tables + (size_t)e * unit_tables::UNIT_TABLES_W;
    const int64_t ray = kind ? u[UNIT_COL_FACING_TARGET_Q] : u[UNIT_COL_FACING_Q];
    const int64_t cone = obs_rows::aim_cone(u, ot, tab);
    const int64_t ux = obs_sight::centre_x(u), uy = obs_sight::centre_y(u);
    const int64_t uid_u = u[UNIT_COL_ID];
    const uint8_t* so = solid + (size_t)e * h * w;
    const uint8_t* sc = screen + (size_t)e * h * w;
    bool bad = false;
    obs_sight::AimKey prev{};
    bool have_prev = false;
    int32_t hit = OBS_AIM_NONE;
    for (;;) {
        obs_sight::AimKey best{};
        bool have_best = false;
        for (int c = 0; c < U; ++c) {                     // every living unit, live centres
            const int64_t* m = rows + (size_t)c * UNIT_STORE_W;
            if (!m[UNIT_COL_VALID] || m[UNIT_COL_LIFE_STATE] != OBS_LIFE_ALIVE ||
                m[UNIT_COL_ID] == uid_u)
                continue;
            obs_sight::AimKey key;
            key.cx = obs_sight::centre_x(m);
            key.cy = obs_sight::centre_y(m);
            key.uid = m[UNIT_COL_ID];
            const int64_t dx = key.cx - ux, dy = key.cy - uy;
            key.d2 = dx * dx + dy * dy;
            if (have_prev && !obs_sight::aim_less(prev, key)) continue;   // tried already
            if (have_best && !obs_sight::aim_less(key, best)) continue;
            const int64_t b = obs_rows::bearing_q16(dx, dy, &bad);
            const int64_t off = unit_lib::wrap_q16(b - ray);
            if ((off < 0 ? -off : off) > cone) continue;
            best = key;
            have_best = true;
        }
        if (!have_best || bad) break;
        if (obs_sight::line_clear(so, sc, h, w, false, uy, ux, best.cy, best.cx, &bad)) {
            hit = (int32_t)best.uid;
            break;
        }
        prev = best;
        have_prev = true;
    }
    if (bad) atomicOr(faults + e, OBS_FULL_FAULT_RANGE);
    *out = hit;
}

// The full view's extra columns of a UNIT entry (spec 5.3 cols F+1 .. F+17,
// _full_extra): its kit (the vector's own reader, obs_rows::kit_of), four
// statuses, its order's kind. Col F (the team's knowledge of it) is the row's.
__device__ void full_extra(float* E, const int64_t* it, const View& v, bool* bad) {
    const obs_rows::Kit k = obs_rows::kit_of(it, v.ot, v.tab, v.tick, v.cap, bad);
    put(E, OBS_FX_ROUNDS, k.rounds);
    put(E, OBS_FX_MAGS, k.mags);
    put(E, OBS_FX_AMMO_UNTRACKED, k.untracked);
    for (int g = 0; g < OBS_HAND_GRENADE_SLOTS; ++g) put(E, OBS_FX_GRENADE0 + g, k.grenades[g]);
    put(E, OBS_FX_CHARGES, k.charges);
    put(E, OBS_FX_MELEE_COOL, k.melee_cool);
    put(E, OBS_FX_SHOVE_COOL, k.shove_cool);
    const int kinds[OBS_FULL_N_STATUS] = {OBS_FULL_STATUS_KIND_0, OBS_FULL_STATUS_KIND_1,
                                          OBS_FULL_STATUS_KIND_2, OBS_FULL_STATUS_KIND_3};
    for (int j = 0; j < OBS_FULL_N_STATUS; ++j)
        put(E, OBS_FX_STATUS0 + j,
            squash(ratio(obs_rows::status_left(it, kinds[j]),
                         v.ot[OBS_TAB_STATUS_SCALE + kinds[j]]), v.cap));
    put(E, OBS_FX_ORDER_WHAT, (double)it[UNIT_COL_ORDER_WHAT]);
}

// What the full view's rows read beside the list's View.
struct Full {
    const int32_t* aim;        // (2, U) of the env: [0] facing ray, [1] facing target
    const uint8_t* sees_me;    // (U,) the soldier's sees row
    const uint8_t* solid;      // the env's (h, w)
    int h, w;
    const int64_t* covers;     // the env's cover rows
    int n_cover;
    const int64_t* places;     // the env's place table
    int n_places;
    const uint8_t* lane_own;
    int gh, gw;
};

// A team-mate's full entry (type 0; the soldier itself when `k == sr`): the
// list's mate entry with the every-unit aim along its facing target, the
// order block, the extra columns, knowledge SEEN.
__device__ void full_mate(float* E, int k, const View& v, const Full& F, bool* bad) {
    const int64_t* m = v.rows + (size_t)k * UNIT_STORE_W;
    mate_entry(E, k, v, F.aim[v.U + k] >= 0, k != v.sr, F.lane_own, F.gh, F.gw);
    order_tail(E, m, v, F.places, F.n_places);
    full_extra(E, m, v, bad);
    put(E, OBS_FX_KNOWLEDGE, (double)OBS_KNOW_SEEN);
}

// Any other living unit's full entry (type 1, enemy_entry): its LIVE centre,
// facing, motion, team and class, the line of fire to its live centre
// (obs_sight::solid_in_way over the cover table), a mate on the line, the
// ticks, the every-unit aim along its facing, the shared enemy tail, the
// extra columns, the team's knowledge of it (seen / record / unknown).
__device__ void full_enemy(float* E, int q, const View& v, const Full& F, bool* bad,
                           bool* law_bad) {
    const int64_t* it = v.rows + (size_t)q * UNIT_STORE_W;
    const int64_t ex = obs_sight::centre_x(it), ey = obs_sight::centre_y(it);
    const int64_t euid = it[UNIT_COL_ID];
    const int64_t dx = ex - v.cx, dy = ey - v.cy;
    entry_common(E, OBS_TYPE_SEEN, dx, dy, v);
    put_angle(E, OBS_L_COS, OBS_L_SIN, it[UNIT_COL_FACING_Q]);
    const int64_t* tr = obs_pic::unit_of(v.blk, v.L, q);
    int64_t vx, vy;
    obs_pic::motion_of(tr, &vx, &vy);                  // the full view knows every unit's truth
    put(E, OBS_L_VEL_X, squash(ratio(vx, v.ot[OBS_TAB_V_REF_Q]), v.cap));
    put(E, OBS_L_VEL_Y, squash(ratio(vy, v.ot[OBS_TAB_V_REF_Q]), v.cap));
    put(E, OBS_L_SIDE, (double)it[UNIT_COL_TEAM]);
    put(E, OBS_L_CLASS, it[UNIT_COL_IS_ZOMBIE] ? (double)v.ot[OBS_TAB_ZOMBIE_CLASS]
                                               : (double)it[UNIT_COL_LOADOUT_NAME]);
    put(E, OBS_L_HP, obs_pic::hp_over_bar(it[UNIT_COL_CURRENT_HP], tr, v.cap));
    unit_columns(E, it, v);
    put(E, OBS_L_SEEN_BY_ME, F.sees_me[q] ? 1.0 : 0.0);
    put(E, OBS_L_LOF_SOLID, obs_sight::solid_in_way(F.solid, F.h, F.w, F.covers, F.n_cover, v.cx,
                                                    v.cy, ex, ey, bad) ? 1.0 : 0.0);
    bool mate_way = false;
    {
        const int64_t dd = dx * dx + dy * dy;
        if (dd != 0) {
            const int64_t cr = floordiv(v.me[UNIT_COL_FOOTPRINT], 2) + v.ot[OBS_TAB_LINE_CLEAR_TILES];
            for (int k = 0; k < v.U && !mate_way; ++k) {
                const int64_t* m = v.rows + (size_t)k * UNIT_STORE_W;
                if (!is_mate(m, v.t) || m[UNIT_COL_ID] == v.me_id) continue;
                mate_way = obs_sight::mate_near_segment(v.cx, v.cy, dx, dy, dd, cr,
                                                        obs_sight::centre_x(m),
                                                        obs_sight::centre_y(m));
            }
        }
    }
    put(E, OBS_L_LOF_MATE, mate_way ? 1.0 : 0.0);
    const int64_t ticks = obs_rows::ticks_toward(v.me, v.ot, v.tab, dx, dy, bad, law_bad);
    const int32_t* aim_t = F.aim + v.U;                // the mates' aims: along the facing target
    enemy_tail(E, v, it, tr, dx, dy, ex, ey, euid, ticks, F.aim[q],
               [&](int k) { return aim_t[k]; });
    full_extra(E, it, v, bad);
    const int know = v.vis_t[q] ? OBS_KNOW_SEEN
                   : obs_pic::rec_of(v.blk, v.L, v.U, v.t, q)[OBS_PIC_REC_PRESENT] ? OBS_KNOW_RECORD
                                                                                  : OBS_KNOW_UNKNOWN;
    put(E, OBS_FX_KNOWLEDGE, (double)know);
}

// The candidates of the full list's blocks (spec 5.3, _full_entries): a
// living non-zombie unit other than the soldier (`brk_vis`, THE BREAK
// full_vis: only those the team sees or its own), a living zombie, a live
// object (grenade rows then charge rows: live_objects' order; c < gc a
// grenade row), a living door with a span.
__device__ __forceinline__ bool full_marine(const int64_t* m, const View& v, bool brk_vis, int q) {
    if (!m[UNIT_COL_VALID] || m[UNIT_COL_LIFE_STATE] != OBS_LIFE_ALIVE || m[UNIT_COL_IS_ZOMBIE] ||
        m[UNIT_COL_ID] == v.me_id)
        return false;
    return !brk_vis || m[UNIT_COL_TEAM] == (int64_t)v.t || v.vis_t[q];
}
__device__ __forceinline__ bool full_zombie(const int64_t* m, const View& v, bool brk_vis, int q) {
    if (!m[UNIT_COL_VALID] || m[UNIT_COL_LIFE_STATE] != OBS_LIFE_ALIVE || !m[UNIT_COL_IS_ZOMBIE])
        return false;
    return !brk_vis || v.vis_t[q];
}

// One thread per (env, row r, block b): b = 0 the soldier itself, 1 the other
// living marines, 2 the zombies, 3 the objects, 4 the doors. Each block's
// place follows from the counts (the entry order stays marines, zombies,
// objects, doors; the room is cut objects first, then zombies, then doors);
// each thread zeroes its own places (observe_full_rows' out[...] = 0, mask 0,
// uid -1; the doors' thread to the end of the row), then fills them.
__global__ void k_full_list(const int64_t* store, int U, const int64_t* tables,
                            const int32_t* rowmap, int n_rows, int S, int T, const int64_t* otab,
                            const int64_t* teams, const int64_t* trow, const int64_t* params,
                            const uint8_t* ended, const uint8_t* visible, const uint8_t* sees,
                            const int64_t* state, int W, const int32_t* full_aim,
                            const uint8_t* lane_own, int gh, int gw, const int64_t* grenades,
                            const int64_t* g_count, const int64_t* charges,
                            const int64_t* c_count, const int64_t* doors,
                            const int64_t* door_count, int door_cap, const int64_t* door_tiles,
                            int tile_cap, const bool* vac, const bool* amb, int h, int w,
                            const uint8_t* wired, int wired_cap, const uint8_t* solid,
                            const int64_t* covers, int cover_cap, const int32_t* cover_count,
                            const int64_t* places, int place_cap, const int32_t* place_count,
                            float* full_list, uint8_t* full_mask, int32_t* full_uid,
                            int32_t* faults, int brk) {
    using namespace object_tables;
    const int e = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * 5 || !env_due(e, trow, params, ended)) return;
    const int r = idx / 5, b = idx % 5;
    const size_t er = (size_t)e * n_rows + r;
    float* LST = full_list + er * OBS_FULL_ENTRIES * OBS_FULL_F;
    uint8_t* MSK = full_mask + er * OBS_FULL_ENTRIES;
    int32_t* UID = full_uid + er * OBS_FULL_ENTRIES;
    auto clear = [&](int a, int z) {
        for (int i = a; i < z; ++i) {
            for (int f = 0; f < OBS_FULL_F; ++f) LST[(size_t)i * OBS_FULL_F + f] = 0.0f;
            MSK[i] = 0;
            UID[i] = -1;
        }
    };
    const int t = r / S;
    const int sr = rowmap[er];
    const int64_t* rows = store + (size_t)e * U * UNIT_STORE_W;
    if (sr < 0 || sr >= U) {                          // an empty roster place: zeros
        if (b == 0) clear(0, OBS_FULL_ENTRIES);
        return;
    }
    const int64_t* me = rows + (size_t)sr * UNIT_STORE_W;
    const bool alive = me[UNIT_COL_LIFE_STATE] == OBS_LIFE_ALIVE && me[UNIT_COL_TEAM] == (int64_t)t;
    View v;
    v.rows = rows;
    v.U = U;
    v.T = T;
    v.t = t;
    v.n_rows = n_rows;
    v.r = r;
    v.e = e;
    v.me = me;
    v.sr = sr;
    v.me_id = me[UNIT_COL_ID];
    v.cx = obs_sight::centre_x(me);
    v.cy = obs_sight::centre_y(me);
    v.tick = trow[(size_t)e * unit_draws::TICK_ROW_W + unit_draws::TICK_ROW_TICK];
    v.ot = otab + (size_t)e * OBS_TAB_W;
    v.lref = v.ot[OBS_TAB_L_REF];
    v.n_t = teams[((size_t)e * T + t) * OBS_TEAM_W + OBS_TEAM_N];
    v.tab = tables + (size_t)e * unit_tables::UNIT_TABLES_W;
    v.blk = state + (size_t)e * W;
    v.L = pic_layout(T, U);
    v.vis_t = visible + ((size_t)e * (T + 1) + t) * U;
    v.cap = of_bits(OBS_SQUASH_CAP_BITS);
    const bool brk_vis = brk == OBS_FULL_BREAK_FULL_VIS;
    const bool rev = brk == OBS_FULL_BREAK_FULL_TIE;
    Objs o{grenades + (size_t)e * OBJ_GRENADE_CAP * OBJ_GRENADE_W, g_count[e],
           charges + (size_t)e * OBJ_CHARGE_CAP * OBJ_CHARGE_W, c_count[e]};
    Doors D{doors + (size_t)e * door_cap * OBJ_DOOR_W, door_count[e],
            door_tiles + (size_t)e * tile_cap * OBJ_DOOR_TILE_W, tile_cap,
            vac + (size_t)e * h * w, amb + (size_t)e * h * w, h, w,
            wired + (size_t)e * wired_cap, wired_cap};
    const int n_doors = (int)(D.n < door_cap ? D.n : door_cap);
    auto door_ok = [&](int q) {
        const int64_t* row = D.d + (size_t)q * OBJ_DOOR_W;
        int64_t x, y;
        return row[OBJ_DOOR_COL_ALIVE] && door_centre(D, row, &x, &y);
    };
    auto obj_live = [&](int c) {
        return c < o.gc ? o.g[(size_t)c * OBJ_GRENADE_W + OBJ_GRENADE_LIVE] != 0
                        : o.c[(size_t)(c - o.gc) * OBJ_CHARGE_W + OBJ_CHARGE_LIVE] != 0;
    };
    // -- the counts and the places (_full_entries' cut) --
    int M = 0, Z = 0, O = 0, DN = 0;
    for (int q = 0; q < U; ++q) {
        const int64_t* m = rows + (size_t)q * UNIT_STORE_W;
        if (full_marine(m, v, brk_vis, q)) ++M;
        if (full_zombie(m, v, brk_vis, q)) ++Z;
    }
    const int n_obj = (int)(o.gc + o.cc);
    for (int c = 0; c < n_obj; ++c) O += obj_live(c) ? 1 : 0;
    for (int q = 0; q < n_doors; ++q) DN += door_ok(q) ? 1 : 0;
    if (1 + M > OBS_FULL_ENTRIES) {                     // never a marine dropped: refused
        atomicOr(faults + e, OBS_FULL_FAULT_OVERFLOW);
        M = OBS_FULL_ENTRIES - 1;
    }
    int room = OBS_FULL_ENTRIES - 1 - M - (brk == OBS_FULL_BREAK_FULL_CAP ? 1 : 0);
    room = room > 0 ? room : 0;
    const int Ok = O < room ? O : room;                 // the units' room first ...
    room -= Ok;
    const int Zk = Z < room ? Z : room;
    room -= Zk;
    const int Dk = DN < room ? DN : room;               // ... the doors fill what is left
    const int s1 = 1, s2 = s1 + M, s3 = s2 + Zk, s4 = s3 + Ok;
    const int starts[5] = {0, s1, s2, s3, s4};
    const int ends[5] = {1, s2, s3, s4, OBS_FULL_ENTRIES};
    const int caps[5] = {1, M, Zk, Ok, Dk};
    clear(starts[b], ends[b]);
    Full F{full_aim + (size_t)e * 2 * U, sees + ((size_t)e * U + sr) * U, solid + (size_t)e * h * w,
           h, w, covers + (size_t)e * cover_cap * OBS_COVER_W, cover_count[e],
           places + (size_t)e * place_cap * OBS_PLACE_W, place_count[e], lane_own, gh, gw};
    bool bad = false, law_bad = false;
    auto d2_of = [&](int64_t x, int64_t y) {
        const int64_t dx = x - v.cx, dy = y - v.cy;
        return dx * dx + dy * dy;
    };
    if (b == 0) {                                       // -- the soldier itself (5.3; 5.6)
        float* E = LST;
        if (alive) {
            full_mate(E, sr, v, F, &bad);
        } else {                                        // the corpse: type 0, hp 0, age
            entry_common(E, OBS_TYPE_MATE, 0, 0, v);
            put_angle(E, OBS_L_COS, OBS_L_SIN, me[UNIT_COL_FACING_Q]);
            put(E, OBS_L_SIDE, (double)t);
            put(E, OBS_L_CLASS, (double)me[UNIT_COL_LOADOUT_NAME]);
            const int64_t* tr = obs_pic::unit_of(v.blk, v.L, sr);
            const int64_t died = tr[OBS_PIC_UNIT_DEATH_SET] ? tr[OBS_PIC_UNIT_DEATH_TICK] : v.tick;
            put(E, OBS_L_AGE, ratio(v.tick - died, v.ot[OBS_TAB_LIMIT_TICKS]));
        }
        MSK[0] = 1;
        UID[0] = (int32_t)v.me_id;
    } else {
        Key prev{0, 0}, k{0, 0};
        bool have = false;
        for (int i = 0; i < caps[b]; ++i) {
            float* E = LST + (size_t)(starts[b] + i) * OBS_FULL_F;
            int c = -1;
            int64_t uid = -1;
            if (b == 1) {                               // the marines: (team, slot, id)
                c = next_min(U, [&](int q, Key* out) {
                    const int64_t* m = rows + (size_t)q * UNIT_STORE_W;
                    if (!full_marine(m, v, brk_vis, q)) return false;
                    *out = Key{(m[UNIT_COL_TEAM] << 20) + m[UNIT_COL_SLOT], m[UNIT_COL_ID]};
                    return true;
                }, have, prev, rev, &k);
                if (c < 0) break;
                if (rows[(size_t)c * UNIT_STORE_W + UNIT_COL_TEAM] == (int64_t)t)
                    full_mate(E, c, v, F, &bad);
                else
                    full_enemy(E, c, v, F, &bad, &law_bad);
                uid = k.id;
            } else if (b == 2) {                        // the zombies, nearest first
                c = next_min(U, [&](int q, Key* out) {
                    const int64_t* m = rows + (size_t)q * UNIT_STORE_W;
                    if (!full_zombie(m, v, brk_vis, q)) return false;
                    *out = Key{d2_of(obs_sight::centre_x(m), obs_sight::centre_y(m)),
                               m[UNIT_COL_ID]};
                    return true;
                }, have, prev, rev, &k);
                if (c < 0) break;
                full_enemy(E, c, v, F, &bad, &law_bad);
                uid = k.id;
            } else if (b == 3) {                        // the objects, nearest first
                c = next_min(n_obj, [&](int q, Key* out) {
                    if (!obj_live(q)) return false;
                    int64_t ox, oy;
                    object_tile(o, q < o.gc ? 0 : 1, q < o.gc ? q : q - o.gc, &ox, &oy);
                    *out = Key{d2_of(ox, oy), q};       // q: live_objects' order
                    return true;
                }, have, prev, rev, &k);
                if (c < 0) break;
                const int kind = c < o.gc ? 0 : 1;
                const int64_t row = c < o.gc ? c : c - o.gc;
                object_entry(E, o, kind, row, v, rowmap + (size_t)e * n_rows + (size_t)t * S, S,
                             rows);
                // the team's knowledge: seen now, ever seen (the tracker), unknown
                const int64_t serial = kind == 0 ? v.blk[v.L.side_g + row] : v.blk[v.L.side_c + row];
                int know = OBS_KNOW_UNKNOWN;
                const int64_t* seen = v.blk + v.L.seen + (int64_t)t * OBS_PIC_OBJ_CAP * 2;
                for (int64_t s = 0; s < v.blk[v.L.seen_n + t]; ++s)
                    if (seen[2 * s] == serial) know = OBS_KNOW_SEEN;
                if (know != OBS_KNOW_SEEN) {
                    const int64_t* ob = v.blk + v.L.obj + (int64_t)t * OBS_PIC_OBJ_CAP * 2;
                    for (int64_t s = 0; s < v.blk[v.L.obj_n + t]; ++s)
                        if (ob[2 * s] == serial) know = OBS_KNOW_RECORD;
                }
                put(E, OBS_FX_KNOWLEDGE, (double)know);
                uid = (int64_t)OBS_OBJECT_UID_BASE + i;  // its place in the kept block
            } else {                                    // the doors, nearest first
                c = next_min(n_doors, [&](int q, Key* out) {
                    if (!door_ok(q)) return false;
                    const int64_t* row = D.d + (size_t)q * OBJ_DOOR_W;
                    int64_t x, y;
                    door_centre(D, row, &x, &y);
                    *out = Key{d2_of(x, y), row[OBJ_DOOR_COL_ORDINAL]};
                    return true;
                }, have, prev, rev, &k);
                if (c < 0) break;
                door_entry(E, c, D, v);
                uid = (int64_t)OBS_DOOR_UID_BASE + k.id;
            }
            MSK[starts[b] + i] = 1;
            UID[starts[b] + i] = (int32_t)uid;
            prev = k;
            have = true;
        }
    }
    if (bad) atomicOr(faults + e, OBS_FULL_FAULT_RANGE);
    if (law_bad) atomicOr(faults + e, OBS_FULL_FAULT_LAW);
}

// One thread per (env, row, window cell): full_watched (spec 5.5) -- the
// window cell seen FROM any living enemy's LIVE centre (every living unit but
// the row team's own living marines; zombies included, dormant or not), the
// one device window_cell, OR-ed. 0 on an empty roster place.
__global__ void k_full_watched(const int64_t* store, int U, const int32_t* rowmap, int n_rows,
                               int S, const uint8_t* solid, const uint8_t* screen, int h, int w,
                               const int64_t* trow, const int64_t* params, const uint8_t* ended,
                               uint8_t* full_watched, int32_t* faults) {
    constexpr int CELLS = OBS_WIN * OBS_WIN;
    const int e = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * CELLS || !env_due(e, trow, params, ended)) return;
    const int r = idx / CELLS, cell = idx % CELLS;
    const size_t er = (size_t)e * n_rows + r;
    uint8_t* out = full_watched + er * CELLS + cell;
    const int sr = rowmap[er];
    if (sr < 0 || sr >= U) {
        *out = 0;
        return;
    }
    const int64_t t = r / S;
    const int64_t* rows = store + (size_t)e * U * UNIT_STORE_W;
    const int64_t* me = rows + (size_t)sr * UNIT_STORE_W;
    const int64_t y0 = obs_sight::centre_y(me) - OBS_HALF, x0 = obs_sight::centre_x(me) - OBS_HALF;
    const uint8_t* so = solid + (size_t)e * h * w;
    const uint8_t* sc = screen + (size_t)e * h * w;
    bool bad = false, watched = false;
    for (int q = 0; q < U && !watched; ++q) {
        const int64_t* m = rows + (size_t)q * UNIT_STORE_W;
        if (!m[UNIT_COL_VALID] || m[UNIT_COL_LIFE_STATE] != OBS_LIFE_ALIVE) continue;
        if (m[UNIT_COL_TEAM] == t && !m[UNIT_COL_IS_ZOMBIE]) continue;
        watched = obs_sight::window_cell(so, sc, h, w, false, false, obs_sight::centre_y(m),
                                         obs_sight::centre_x(m), y0, x0, OBS_WIN, cell, &bad);
    }
    *out = watched ? 1 : 0;
    if (bad) atomicOr(faults + e, OBS_FULL_FAULT_RANGE);
}

// One thread per (env, row): full_match (spec 5.4, FullView.match) -- per team
// t its living count over n_t and its standing H(t) over n_t squashed
// (reward_v1.standing over the full view's marines: per team the bars of its
// living units summed IN (slot, id) ORDER, the full view's marine order, from
// 0.0 through the kit's unfused add), the horde awake / dormant over the
// match's zombies squashed, time_left. 0 on an empty roster place.
__global__ void k_full_match(const int64_t* store, int U, const int32_t* rowmap, int n_rows, int S,
                             int T, const int64_t* otab, const int64_t* teams, const int64_t* trow,
                             const int64_t* params, const uint8_t* ended, const int64_t* state,
                             int W, float* full_match) {
    const int e = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows || !env_due(e, trow, params, ended)) return;
    const size_t er = (size_t)e * n_rows + r;
    const int M = 2 * T + 3;
    float* out = full_match + er * M;
    for (int i = 0; i < M; ++i) out[i] = 0.0f;
    if (rowmap[er] < 0 || rowmap[er] >= U) return;
    const int64_t* rows = store + (size_t)e * U * UNIT_STORE_W;
    const int64_t* ot = otab + (size_t)e * OBS_TAB_W;
    const int64_t* blk = state + (size_t)e * W;
    const PicLayout L = pic_layout(T, U);
    const double cap = of_bits(OBS_SQUASH_CAP_BITS);
    const int64_t tick = trow[(size_t)e * unit_draws::TICK_ROW_W + unit_draws::TICK_ROW_TICK];
    for (int t = 0; t < T; ++t) {
        const int64_t n_t = teams[((size_t)e * T + t) * OBS_TEAM_W + OBS_TEAM_N];
        int64_t count = 0;
        double hsum = 0.0;
        Key prev{0, 0}, k{0, 0};
        bool have = false;
        for (;;) {                                      // (slot, id) order: successive minima
            const int c = next_min(U, [&](int q, Key* o) {
                const int64_t* m = rows + (size_t)q * UNIT_STORE_W;
                if (!is_mate(m, t)) return false;
                *o = Key{m[UNIT_COL_SLOT], m[UNIT_COL_ID]};
                return true;
            }, have, prev, false, &k);
            if (c < 0) break;
            const int64_t* tr = obs_pic::unit_of(blk, L, c);
            const int64_t bar = tr[OBS_PIC_UNIT_BAR_SET] ? tr[OBS_PIC_UNIT_BAR_Q] : 0;
            hsum = add(hsum, obs_pic::bar_ratio(rows[(size_t)c * UNIT_STORE_W + UNIT_COL_CURRENT_HP],
                                                bar));
            ++count;
            prev = k;
            have = true;
        }
        out[2 * t] = to_f32(ratio(count, n_t));
        out[2 * t + 1] = to_f32(squash(fratio(hsum, (double)n_t), cap));
    }
    int64_t awake = 0, nz_live = 0;
    for (int q = 0; q < U; ++q) {
        const int64_t* m = rows + (size_t)q * UNIT_STORE_W;
        if (!m[UNIT_COL_VALID] || m[UNIT_COL_LIFE_STATE] != OBS_LIFE_ALIVE || !m[UNIT_COL_IS_ZOMBIE])
            continue;
        ++nz_live;
        if (m[UNIT_COL_ZOMBIE_ACTIVATED]) ++awake;
    }
    const int64_t nz = ot[OBS_TAB_ZOMBIES];
    const int kk = 2 * T;
    out[kk] = to_f32(squash(ratio(awake, nz), cap));
    out[kk + 1] = to_f32(squash(ratio(nz_live - awake, nz), cap));
    const int64_t limit = ot[OBS_TAB_LIMIT_TICKS];
    out[kk + 2] = to_f32(ratio(limit - tick, limit));
}

}  // namespace

void obs_list(int n_env, std::uintptr_t store, int n_units, int store_w,
              const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
              int n_teams, std::uintptr_t obs_table, int table_w, std::uintptr_t teams,
              std::uintptr_t tick_row, std::uintptr_t params, std::uintptr_t ended,
              std::uintptr_t visible, std::uintptr_t sees, std::uintptr_t state, int state_w,
              std::uintptr_t aim, std::uintptr_t lof, std::uintptr_t ttw, std::uintptr_t lane_own,
              int gh, int gw, ObjectTable& grenades, ObjectTable& charges, UnitRecordList& doors,
              UnitRecordList& door_tiles, std::uintptr_t is_vacuum, std::uintptr_t is_ambient,
              int h, int w, std::uintptr_t wired, int wired_cap, std::uintptr_t places,
              int place_cap, int place_w, std::uintptr_t place_count, int n_entries,
              std::uintptr_t list, std::uintptr_t list_mask, std::uintptr_t list_uid,
              std::uintptr_t faults, int brk) {
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::invalid_argument(std::string("obs_list: ") + what);
    };
    need(n_env >= 1 && n_units >= 1 && n_rows >= 1 && n_slots >= 1 && n_teams >= 1 && h >= 1 &&
         w >= 1 && gh >= 1 && gw >= 1 && wired_cap >= 1 && place_cap >= 1,
         "a non-positive size");
    need(place_w == OBS_PLACE_W, "the place table's width is not OBS_PLACE_W");
    need(places && place_count, "a null place table");
    need(n_rows == n_teams * n_slots, "n_rows is not n_teams x n_slots");
    need(n_entries == n_slots + OBS_ENTRIES_BEYOND_SLOTS,
         "n_entries is not n_slots + OBS_ENTRIES_BEYOND_SLOTS");
    need(store_w == UNIT_STORE_W, "the store's width is not UNIT_STORE_W");
    need(table_w == OBS_TAB_W, "the obs table's width is not OBS_TAB_W (regenerate obs_layout.h)");
    need(state_w == pic_layout(n_teams, n_units).w, "the picture block's width");
    need(tables.n_env() == n_env && tables.width() == unit_tables::UNIT_TABLES_W,
         "the device tables' shape");
    need(grenades.n_env() == n_env && charges.n_env() == n_env && doors.n_env() == n_env &&
         door_tiles.n_env() == n_env, "the object / door tables' env count");
    need(grenades.cap() == object_tables::OBJ_GRENADE_CAP &&
         grenades.width() == object_tables::OBJ_GRENADE_W &&
         charges.cap() == object_tables::OBJ_CHARGE_CAP &&
         charges.width() == object_tables::OBJ_CHARGE_W, "the object tables' layout");
    need(doors.width() == object_tables::OBJ_DOOR_W &&
         door_tiles.width() == object_tables::OBJ_DOOR_TILE_W, "the door tables' widths");
    need(store && rowmap && obs_table && teams && tick_row && params && ended && visible &&
         sees && state && aim && lof && ttw && lane_own && is_vacuum && is_ambient && wired &&
         list && list_mask && list_uid && faults, "a null device pointer");
    cudaStream_t s = tstream();
    const int n = n_rows * 5;
    k_list<<<dim3((n + 63) / 64, n_env), 64, 0, s>>>(
        reinterpret_cast<const int64_t*>(store), n_units, tables.values(),
        reinterpret_cast<const int32_t*>(rowmap), n_rows, n_slots, n_teams,
        reinterpret_cast<const int64_t*>(obs_table), reinterpret_cast<const int64_t*>(teams),
        reinterpret_cast<const int64_t*>(tick_row), reinterpret_cast<const int64_t*>(params),
        reinterpret_cast<const uint8_t*>(ended), reinterpret_cast<const uint8_t*>(visible),
        reinterpret_cast<const uint8_t*>(sees), reinterpret_cast<const int64_t*>(state), state_w,
        reinterpret_cast<const int32_t*>(aim), reinterpret_cast<const uint8_t*>(lof),
        reinterpret_cast<const int32_t*>(ttw), reinterpret_cast<const uint8_t*>(lane_own), gh, gw,
        grenades.records_dev(), grenades.count_dev(), charges.records_dev(), charges.count_dev(),
        doors.records_dev(), doors.count_dev(), doors.cap(), door_tiles.records_dev(),
        door_tiles.cap(), reinterpret_cast<const bool*>(is_vacuum),
        reinterpret_cast<const bool*>(is_ambient), h, w, reinterpret_cast<const uint8_t*>(wired),
        wired_cap, reinterpret_cast<const int64_t*>(places), place_cap,
        reinterpret_cast<const int32_t*>(place_count), n_entries, reinterpret_cast<float*>(list),
        reinterpret_cast<uint8_t*>(list_mask), reinterpret_cast<int32_t*>(list_uid),
        reinterpret_cast<int32_t*>(faults), brk);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("obs_list: launch failed: ") + cudaGetErrorString(err));
}

void obs_masks(int n_env, std::uintptr_t store, int n_units, int store_w,
               const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
               std::uintptr_t tick_row, std::uintptr_t params, std::uintptr_t ended,
               std::uintptr_t list, std::uintptr_t list_mask, int n_entries,
               UnitRecordList& doors, UnitRecordList& door_tiles, int n_kinds,
               std::uintptr_t mask_feet, std::uintptr_t mask_aim, std::uintptr_t mask_trigger,
               std::uintptr_t mask_hand, std::uintptr_t faults, int brk) {
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::invalid_argument(std::string("obs_masks: ") + what);
    };
    need(n_env >= 1 && n_units >= 1 && n_rows >= 1 && n_slots >= 1 && n_kinds >= 0,
         "a non-positive size");
    need(n_rows % n_slots == 0, "n_rows is not a multiple of n_slots");
    need(n_entries == n_slots + OBS_ENTRIES_BEYOND_SLOTS,
         "n_entries is not n_slots + OBS_ENTRIES_BEYOND_SLOTS");
    need(n_kinds <= UNIT_COL_GRENADE_COUNTS_W, "more throw kinds than the store's grenade counts");
    need(store_w == UNIT_STORE_W, "the store's width is not UNIT_STORE_W");
    need(tables.n_env() == n_env && tables.width() == unit_tables::UNIT_TABLES_W,
         "the device tables' shape");
    need(doors.n_env() == n_env && door_tiles.n_env() == n_env, "the door tables' env count");
    need(doors.width() == object_tables::OBJ_DOOR_W &&
         door_tiles.width() == object_tables::OBJ_DOOR_TILE_W, "the door tables' widths");
    need(store && rowmap && tick_row && params && ended && list && list_mask && mask_feet &&
         mask_aim && mask_trigger && mask_hand && faults, "a null device pointer");
    cudaStream_t s = tstream();
    k_masks<<<dim3((n_rows + 63) / 64, n_env), 64, 0, s>>>(
        reinterpret_cast<const int64_t*>(store), n_units, tables.values(),
        reinterpret_cast<const int32_t*>(rowmap), n_rows, n_slots,
        reinterpret_cast<const int64_t*>(tick_row), reinterpret_cast<const int64_t*>(params),
        reinterpret_cast<const uint8_t*>(ended), reinterpret_cast<const float*>(list),
        reinterpret_cast<const uint8_t*>(list_mask), n_entries, doors.records_dev(),
        doors.count_dev(), doors.cap(), door_tiles.records_dev(), door_tiles.count_dev(),
        door_tiles.cap(), n_kinds, reinterpret_cast<uint8_t*>(mask_feet),
        reinterpret_cast<uint8_t*>(mask_aim), reinterpret_cast<uint8_t*>(mask_trigger),
        reinterpret_cast<uint8_t*>(mask_hand), reinterpret_cast<int32_t*>(faults), brk);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("obs_masks: launch failed: ") +
                                 cudaGetErrorString(err));
}

void obs_full(int n_env, std::uintptr_t store, int n_units, int store_w,
              const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
              int n_teams, std::uintptr_t obs_table, int table_w, std::uintptr_t teams,
              std::uintptr_t tick_row, std::uintptr_t params, std::uintptr_t ended,
              std::uintptr_t visible, std::uintptr_t sees, std::uintptr_t state, int state_w,
              std::uintptr_t lane_own, int gh, int gw, ObjectTable& grenades,
              ObjectTable& charges, UnitRecordList& doors, UnitRecordList& door_tiles,
              std::uintptr_t is_vacuum, std::uintptr_t is_ambient, std::uintptr_t solid,
              std::uintptr_t screen, int h, int w, std::uintptr_t wired, int wired_cap,
              std::uintptr_t covers, int cover_cap, int cover_w, std::uintptr_t cover_count,
              std::uintptr_t places, int place_cap, int place_w, std::uintptr_t place_count,
              std::uintptr_t full_aim, std::uintptr_t full_list, std::uintptr_t full_mask,
              std::uintptr_t full_uid, std::uintptr_t full_match, std::uintptr_t full_watched,
              std::uintptr_t faults, int brk) {
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::invalid_argument(std::string("obs_full: ") + what);
    };
    need(n_env >= 1 && n_units >= 1 && n_rows >= 1 && n_slots >= 1 && n_teams >= 1 && h >= 1 &&
         w >= 1 && gh >= 1 && gw >= 1 && wired_cap >= 1 && cover_cap >= 1 && place_cap >= 1,
         "a non-positive size");
    need(n_rows == n_teams * n_slots, "n_rows is not n_teams x n_slots");
    need(store_w == UNIT_STORE_W, "the store's width is not UNIT_STORE_W");
    need(table_w == OBS_TAB_W, "the obs table's width is not OBS_TAB_W (regenerate obs_layout.h)");
    need(cover_w == OBS_COVER_W, "the cover table's width is not OBS_COVER_W");
    need(place_w == OBS_PLACE_W, "the place table's width is not OBS_PLACE_W");
    need(state_w == pic_layout(n_teams, n_units).w, "the picture block's width");
    need(tables.n_env() == n_env && tables.width() == unit_tables::UNIT_TABLES_W,
         "the device tables' shape");
    need(grenades.n_env() == n_env && charges.n_env() == n_env && doors.n_env() == n_env &&
         door_tiles.n_env() == n_env, "the object / door tables' env count");
    need(grenades.cap() == object_tables::OBJ_GRENADE_CAP &&
         grenades.width() == object_tables::OBJ_GRENADE_W &&
         charges.cap() == object_tables::OBJ_CHARGE_CAP &&
         charges.width() == object_tables::OBJ_CHARGE_W, "the object tables' layout");
    need(doors.width() == object_tables::OBJ_DOOR_W &&
         door_tiles.width() == object_tables::OBJ_DOOR_TILE_W, "the door tables' widths");
    need(store && rowmap && obs_table && teams && tick_row && params && ended && visible &&
         sees && state && lane_own && is_vacuum && is_ambient && solid && screen && wired &&
         covers && cover_count && places && place_count && full_aim && full_list &&
         full_mask && full_uid && full_match && full_watched && faults,
         "a null device pointer");
    cudaStream_t s = tstream();
    const int64_t* st = reinterpret_cast<const int64_t*>(store);
    const int32_t* rm = reinterpret_cast<const int32_t*>(rowmap);
    const int64_t* ot = reinterpret_cast<const int64_t*>(obs_table);
    const int64_t* trow = reinterpret_cast<const int64_t*>(tick_row);
    const int64_t* prm = reinterpret_cast<const int64_t*>(params);
    const uint8_t* end = reinterpret_cast<const uint8_t*>(ended);
    const uint8_t* so = reinterpret_cast<const uint8_t*>(solid);
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(screen);
    const int64_t* pic = reinterpret_cast<const int64_t*>(state);
    int32_t* aim = reinterpret_cast<int32_t*>(full_aim);
    int32_t* flt = reinterpret_cast<int32_t*>(faults);
    k_full_aim<<<dim3((2 * n_units + 63) / 64, n_env), 64, 0, s>>>(
        st, n_units, ot, tables.values(), so, sc, h, w, trow, prm, end, aim, flt);
    const int n = n_rows * 5;
    k_full_list<<<dim3((n + 63) / 64, n_env), 64, 0, s>>>(
        st, n_units, tables.values(), rm, n_rows, n_slots, n_teams, ot,
        reinterpret_cast<const int64_t*>(teams), trow, prm, end,
        reinterpret_cast<const uint8_t*>(visible), reinterpret_cast<const uint8_t*>(sees), pic,
        state_w, aim, reinterpret_cast<const uint8_t*>(lane_own), gh, gw, grenades.records_dev(),
        grenades.count_dev(), charges.records_dev(), charges.count_dev(), doors.records_dev(),
        doors.count_dev(), doors.cap(), door_tiles.records_dev(), door_tiles.cap(),
        reinterpret_cast<const bool*>(is_vacuum), reinterpret_cast<const bool*>(is_ambient), h, w,
        reinterpret_cast<const uint8_t*>(wired), wired_cap, so,
        reinterpret_cast<const int64_t*>(covers), cover_cap,
        reinterpret_cast<const int32_t*>(cover_count), reinterpret_cast<const int64_t*>(places),
        place_cap, reinterpret_cast<const int32_t*>(place_count),
        reinterpret_cast<float*>(full_list), reinterpret_cast<uint8_t*>(full_mask),
        reinterpret_cast<int32_t*>(full_uid), flt, brk);
    const int cells = n_rows * OBS_WIN * OBS_WIN;
    k_full_watched<<<dim3((cells + 255) / 256, n_env), 256, 0, s>>>(
        st, n_units, rm, n_rows, n_slots, so, sc, h, w, trow, prm, end,
        reinterpret_cast<uint8_t*>(full_watched), flt);
    k_full_match<<<dim3((n_rows + 63) / 64, n_env), 64, 0, s>>>(
        st, n_units, rm, n_rows, n_slots, n_teams, ot, reinterpret_cast<const int64_t*>(teams),
        trow, prm, end, pic, state_w, reinterpret_cast<float*>(full_match));
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("obs_full: launch failed: ") + cudaGetErrorString(err));
}

}  // namespace breach_cuda
