#pragma once
// ============================================================================
// #29 P7a.2 -- THE MARCH (residency design 4.4c, the P7a.2 row): the device
// form of combat.BulletInFlight.advance on the CONTINUOUS path, ONE function
// that slot 4's shooting (the rounds marched in the tick they are fired) and
// slot 2''s rounds (every round still in flight) both call -- never a copy.
// Integer for integer the Python; held equal on synthetic rounds by
// tests/cuda_unit_shooting_check.py part MARCH and on live scenes by the
// shadow lockstep.
//
//   march(m, rd, fx, bad) -> still flying
//     m    MarchMap: this env's unit rows (row order = sim.units' order), its
//          device tables, its solid / material planes (h, w), its draw key,
//          the tick (sim.tick: the door refuses a WEGO tick base)
//     rd   ONE row of the round table (object_tables_layout.h OBJ_ROUND_*),
//          read and written in place: the budget, the position, the last
//          tile crossed, the steps left
//     fx   the record sink the march's effects go to (a row's staging in
//          slot 4, the env's effect list in slot 2'), in the reference's
//          order: a CHEW (a wall stop, or a cover tile's absorption), or a
//          HIT then a PUSH (the hit unit), then a payload round's PAYLOAD at
//          its stop (recorded; the applier refuses it until P7c.1)
//
// The law, per step: the kit step added; out of the grid or into a solid tile
// stops (the solid tile chewed); the FIRST VALID ROW in row order whose
// footprint holds the tile, the shooter excluded BY ITS ID (a converted
// shooter keeps it) and the dead skipped, is the hit; the material of the
// tile crossed just before reads its cover exposure and, below ONE, ONE lazy
// draw (Salt.EXPOSURE, draws.round_index(age, ordinal)) decides absorption
// (the tile chewed, the round back on its pre-step position). A hit with the
// round's authored damage > 0 is a HIT (the zombie multiplier truncated; the
// lazy crit draw Salt.SHOT_CRIT against crit_chance x the arc multiplier when
// the row has a crit; the crit multiplier rounded half away); every hit is a
// PUSH of the round's row momentum along its flight with its lever arm.
//
// NOT PORTED (WEGO only, refused by the round table's encoder by name):
// physical cover (march_cover) and the facing defense. A round whose operands
// leave the proved range (a position past POS_MAX, a step vector past ONE, a
// speed <= 0 or past SPEED_MAX, a step count outside [0, STEPS_MAX], an age or ordinal
// outside draws.round_index's 16 bits, a tile or material outside its table)
// sets *bad: the caller faults.
//
// Device-only header; CLASS T: nothing in the played tick includes it.
// ============================================================================
#include <cstdint>

#include "applier_layout.h"         // GENERATED: the effect record
#include "cuda_unit_draws.cuh"      // the device draws
#include "cuda_unit_lib.cuh"        // the rows, the tables, the record sinks
#include "object_tables_layout.h"   // GENERATED: the round table's columns
#include "unit_events_layout.h"     // GENERATED: the causes

namespace breach_cuda {
namespace unit_march {

using namespace unit_lib;
using namespace applier;
using namespace object_tables;

constexpr int64_t POS_MAX = (int64_t)1 << 40;
constexpr int64_t SPEED_MAX = (int64_t)1 << 30;
constexpr int64_t STEPS_MAX = (int64_t)1 << 24;
constexpr int64_t ROUND_FIELD_MAX = 0xFFFF;     // draws.ROUND_FIELD_BITS = 16

struct MarchMap {
    int64_t* store;                 // the store (the env's rows via unit_row)
    int n_units;
    const int64_t* tab;             // this env's table vector
    const bool* solid;              // this env's (h, w) solid plane
    const int8_t* material;         // this env's (h, w) material plane
    int64_t h, w;
    unit_draws_dev::DrawKey key;
    int64_t tick;
};

// The env's effect list itself as a sink (a serial per-env producer).
__device__ __forceinline__ RecordSink list_env_sink(const RecordListView& v) {
    const int e = benv();
    return RecordSink{v.recs + (size_t)e * v.cap * v.width, v.cap, v.count + e,
                      v.dropped + e, v.width};
}

__device__ __forceinline__ void blank(int64_t* rec, int64_t kind) {
    for (int f = 0; f < APPLIER_F; ++f) rec[f] = 0;
    rec[APPLIER_FX_KIND] = kind;
}

// BulletInFlight._draw: (salt, shooter id, round_index(tick - spawn, ordinal)).
__device__ inline uint32_t round_draw(const MarchMap& m, const int64_t* rd, uint32_t salt,
                                      bool* bad) {
    const int64_t age = m.tick - rd[OBJ_ROUND_COL_SPAWN_TICK];
    const int64_t ord = rd[OBJ_ROUND_COL_ORDINAL];
    if (age < 0 || age > ROUND_FIELD_MAX || ord < 0 || ord > ROUND_FIELD_MAX) {
        *bad = true;
        return 0u;
    }
    return unit_draws_dev::draw_at(m.key, salt, rd[OBJ_ROUND_COL_SHOOTER_ID],
                                   (age << 16) | ord, bad);
}

// attack_resolver.arc_multiplier: the crit multiplier of a hit travelling
// along angle_q (screen convention) on the target row tr (its environment:
// the env profile, else the species').
__device__ inline int64_t arc_multiplier(const int64_t* tab, int64_t angle_q, const int64_t* tr,
                                         bool* bad) {
    int64_t front, behind;
    const int64_t p = tr[UNIT_COL_ENV_PROFILE_NAME];
    if (p >= 0) {
        if (p >= UNIT_TAB_ARC_FRONT_HALF_PROFILE_Q_W) {
            *bad = true;
            return 0;
        }
        front = tab[UNIT_TAB_ARC_FRONT_HALF_PROFILE_Q + p];
        behind = tab[UNIT_TAB_ARC_BEHIND_HALF_PROFILE_Q + p];
    } else {
        const int64_t sp = tr[UNIT_COL_SPECIES_ID];
        if (sp < 0 || sp >= UNIT_TAB_ARC_FRONT_HALF_SPECIES_Q_W) {
            *bad = true;
            return 0;
        }
        front = tab[UNIT_TAB_ARC_FRONT_HALF_SPECIES_Q + sp];
        behind = tab[UNIT_TAB_ARC_BEHIND_HALF_SPECIES_Q + sp];
    }
    const int64_t incoming = -angle_q + PI_Q16;
    int64_t ad = wrap_q16(incoming - tr[UNIT_COL_FACING_Q]);
    if (ad < 0) ad = -ad;
    if (ad <= front) return tab[UNIT_TAB_CRIT_FRONT_MULT_Q];
    if (ad >= PI_Q16 - behind) return tab[UNIT_TAB_CRIT_BEHIND_MULT_Q];
    return tab[UNIT_TAB_CRIT_FLANK_MULT_Q];
}

__device__ inline void record_chew(RecordSink& fx, const int64_t* tab, int64_t a, int64_t y,
                                   int64_t x) {
    int64_t rec[APPLIER_F];
    blank(rec, APPLIER_KIND_CHEW);
    rec[APPLIER_CHEW_Y] = y;
    rec[APPLIER_CHEW_X] = x;
    rec[APPLIER_CHEW_DAMAGE_Q] = tab[UNIT_TAB_AMMO_WALL_DAMAGE_Q + a];
    rec[APPLIER_CHEW_LIVE] = tab[UNIT_TAB_AMMO_WALL_LIVE + a];
    emit_record(fx, rec);
}

// One round's march for this tick (see the header). Returns still_flying.
__device__ inline bool march(const MarchMap& m, int64_t* rd, RecordSink& fx, bool* bad) {
    const int64_t* tab = m.tab;
    const int64_t a = rd[OBJ_ROUND_COL_AMMO];
    const int64_t speed = rd[OBJ_ROUND_COL_SPEED_Q16];
    int64_t rx = rd[OBJ_ROUND_COL_RX_Q], ry = rd[OBJ_ROUND_COL_RY_Q];
    const int64_t sx = rd[OBJ_ROUND_COL_SX_Q], sy = rd[OBJ_ROUND_COL_SY_Q];
    int64_t rem = rd[OBJ_ROUND_COL_REMAINING_STEPS];
    int64_t budget = rd[OBJ_ROUND_COL_BUDGET_Q16];
    if (a < 0 || a >= UNIT_TAB_AMMO_SPEED_Q16_W || speed <= 0 || speed > SPEED_MAX ||
        rem < 0 || rem > STEPS_MAX || budget < 0 || budget >= ONE || sx > ONE || sx < -ONE ||
        sy > ONE || sy < -ONE || rx > POS_MAX || rx < -POS_MAX || ry > POS_MAX ||
        ry < -POS_MAX) {
        *bad = true;
        return false;
    }
    // the integer step budget: whole tiles this tick, the fraction carries
    budget += speed;
    const int64_t n_move = budget >> 16;
    budget -= n_move << 16;
    const int64_t n_steps = n_move < rem ? n_move : rem;   // <= STEPS_MAX: |r| < 2^41
    int64_t prev_ix = rd[OBJ_ROUND_COL_PREV_IX], prev_iy = rd[OBJ_ROUND_COL_PREV_IY];
    const int64_t shooter = rd[OBJ_ROUND_COL_SHOOTER_ID];
    int hit = -1;
    bool stopped = false;
    for (int64_t step = 0; step < n_steps; ++step) {
        const int64_t px = rx, py = ry;      // pre-step (absorption rollback)
        rx += sx;
        ry += sy;
        const int64_t ix = fixedpoint::shr_trunc_q(rx, 16);
        const int64_t iy = fixedpoint::shr_trunc_q(ry, 16);
        if (0 <= iy && iy < m.h && 0 <= ix && ix < m.w) {
            if (m.solid[iy * m.w + ix]) {    // missed shots chew walls
                record_chew(fx, tab, a, iy, ix);
                stopped = true;
                break;
            }
        } else {
            stopped = true;
            break;
        }
        // the first unit in LIST order whose footprint holds the tile
        for (int t = 0; t < m.n_units; ++t) {
            const int64_t* tr = unit_row(m.store, m.n_units, t);
            if (!row_valid(tr) || tr[UNIT_COL_ID] == shooter || !row_alive(tr)) continue;
            const int64_t fp = tr[UNIT_COL_FOOTPRINT];
            const int64_t tx = tile_of(tr[UNIT_COL_X_Q]), ty = tile_of(tr[UNIT_COL_Y_Q]);
            if (tx <= ix && ix < tx + fp && ty <= iy && iy < ty + fp) {
                hit = t;
                break;
            }
        }
        if (hit >= 0) {
            // exposure vs cover: the tile crossed just before, LAZY
            if (prev_iy < 0 || prev_iy >= m.h || prev_ix < 0 || prev_ix >= m.w) {
                *bad = true;
                return false;
            }
            const int64_t mat = m.material[prev_iy * m.w + prev_ix];
            if (mat < 0 || mat >= UNIT_TAB_MAT_COVER_EXPOSURE_Q_W) {
                *bad = true;
                return false;
            }
            const int64_t expo = tab[UNIT_TAB_MAT_COVER_EXPOSURE_Q + mat];
            if (expo < ONE) {
                const uint32_t word = round_draw(m, rd, unit_draws::SALT_EXPOSURE, bad);
                if (!unit_draws_dev::roll_q16(word, expo)) {
                    record_chew(fx, tab, a, prev_iy, prev_ix);   // absorbed by the cover
                    rx = px;
                    ry = py;
                    hit = -1;
                    stopped = true;
                }
            }
            break;
        }
        prev_ix = ix;
        prev_iy = iy;
    }
    rem -= n_steps;
    const int64_t uid = shooter;
    if (hit >= 0) {
        const int64_t* tr = unit_row(m.store, m.n_units, hit);
        const int64_t dmg0 = tab[UNIT_TAB_AMMO_DAMAGE + a];
        if (dmg0 > 0) {
            int64_t dmg = dmg0;
            if (tr[UNIT_COL_IS_ZOMBIE]) dmg = (dmg * tab[UNIT_TAB_ZOMBIE_BULLET_MULT_Q]) >> 16;
            const int64_t w = rd[OBJ_ROUND_COL_WEAPON];
            if (w < 0 || w >= UNIT_TAB_WEAPON_CRIT_ON_W) {
                *bad = true;
                return false;
            }
            if (tab[UNIT_TAB_WEAPON_CRIT_ON + w]) {
                const int64_t mult = arc_multiplier(tab, rd[OBJ_ROUND_COL_ANGLE_Q], tr, bad);
                const uint32_t word = round_draw(m, rd, unit_draws::SALT_SHOT_CRIT, bad);
                const int64_t p_q =
                    fixedpoint::mul_q16_round_q(tab[UNIT_TAB_WEAPON_CRIT_CHANCE_Q + w], mult);
                if (unit_draws_dev::roll_q16(word, p_q))
                    dmg = fixedpoint::mul_q16_round_q(dmg, tab[UNIT_TAB_WEAPON_CRIT_MULT_Q + w]);
            }
            int64_t rec[APPLIER_F];
            blank(rec, APPLIER_KIND_HIT);
            rec[APPLIER_HIT_ROW] = hit;
            rec[APPLIER_HIT_AMOUNT_Q] = dmg << 16;
            rec[APPLIER_HIT_DTYPE] = tab[UNIT_TAB_AMMO_DTYPE + a];
            rec[APPLIER_HIT_ATTACKER] = uid >= 0 ? uid : -1;
            rec[APPLIER_HIT_INDIRECT] = 0;
            rec[APPLIER_HIT_EXPOSE] = tab[UNIT_TAB_AMMO_EXPOSE_Q + a];
            rec[APPLIER_HIT_DIR_X_Q] = sx;
            rec[APPLIER_HIT_DIR_Y_Q] = sy;
            rec[APPLIER_HIT_AP_Q] = tab[UNIT_TAB_AMMO_AP_Q + a];
            rec[APPLIER_HIT_CAUSE] = unit_events::UNIT_CAUSE_BULLET;
            emit_record(fx, rec);
        }
        // the round that stopped on a unit pushes it (push_on), lever and all
        const int64_t j = tab[UNIT_TAB_AMMO_PUSH_Q + a];
        if (j != 0) {
            const int64_t jx = fixedpoint::shr_trunc_q(j * sx, 16);
            const int64_t jy = fixedpoint::shr_trunc_q(j * sy, 16);
            const int64_t fp = tr[UNIT_COL_FOOTPRINT];
            const int64_t cx = centre_q(tr[UNIT_COL_X_Q], fp), cy = centre_q(tr[UNIT_COL_Y_Q], fp);
            const int64_t dx = cx - rd[OBJ_ROUND_COL_OX_Q], dy = cy - rd[OBJ_ROUND_COL_OY_Q];
            if (dx > 2 * POS_MAX || dx < -2 * POS_MAX || dy > 2 * POS_MAX || dy < -2 * POS_MAX) {
                *bad = true;
                return false;
            }
            const int64_t lever = fixedpoint::shr_trunc_q(dx * sy - dy * sx, 16);
            if (jx != 0 || jy != 0) {
                int64_t rec[APPLIER_F];
                blank(rec, APPLIER_KIND_PUSH);
                rec[APPLIER_PUSH_ROW] = hit;
                rec[APPLIER_PUSH_JX_Q] = jx;
                rec[APPLIER_PUSH_JY_Q] = jy;
                rec[APPLIER_PUSH_LEVER_Q] = lever;
                rec[APPLIER_PUSH_TWIST_Q16] = tab[UNIT_TAB_IMPULSE_TWIST_Q16];
                emit_record(fx, rec);
            }
        }
    }
    const bool flying = !stopped && hit < 0 && rem > 0;
    // a payload round detonates at its stop (recorded; refused at the apply)
    const int64_t pay = rd[OBJ_ROUND_COL_PAYLOAD];
    if (pay >= 0 && !flying) {
        int64_t dy = fixedpoint::shr_trunc_q(ry, 16), dx = fixedpoint::shr_trunc_q(rx, 16);
        if (!(0 <= dy && dy < m.h && 0 <= dx && dx < m.w)) {
            dy = prev_iy;
            dx = prev_ix;
        }
        int64_t rec[APPLIER_F];
        blank(rec, APPLIER_KIND_PAYLOAD);
        rec[APPLIER_PAYLOAD_Y] = dy;
        rec[APPLIER_PAYLOAD_X] = dx;
        rec[APPLIER_PAYLOAD_PAYLOAD] = pay;
        rec[APPLIER_PAYLOAD_OWNER] = uid;
        rec[APPLIER_PAYLOAD_APPROACH_Y] = prev_iy;
        rec[APPLIER_PAYLOAD_APPROACH_X] = prev_ix;
        rec[APPLIER_PAYLOAD_HAS_APPROACH] = 1;
        rec[APPLIER_PAYLOAD_KIND] = APPLIER_PAYLOAD_KIND_SHELL;
        emit_record(fx, rec);
    }
    rd[OBJ_ROUND_COL_RX_Q] = rx;
    rd[OBJ_ROUND_COL_RY_Q] = ry;
    rd[OBJ_ROUND_COL_PREV_IX] = prev_ix;
    rd[OBJ_ROUND_COL_PREV_IY] = prev_iy;
    rd[OBJ_ROUND_COL_REMAINING_STEPS] = rem;
    rd[OBJ_ROUND_COL_BUDGET_Q16] = budget;
    rd[OBJ_ROUND_COL_LIVE] = flying ? 1 : 0;
    return flying;
}

}  // namespace unit_march
}  // namespace breach_cuda
