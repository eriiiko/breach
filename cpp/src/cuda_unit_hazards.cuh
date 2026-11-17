#pragma once
// ============================================================================
// #29 P6b.4 -- THE TIMED CHARGES and THE HEAT SOURCES as device functions
// (residency design 4.4c, the cut table's P6b.4 row; decision 1: slot 9e is a
// set of STAGES of the one applier kernel walking the entity tables in the
// reference's order, each entity applied AT ONCE).
//
// timed_sweep: src/simulation/timed_charge_system.sweep_timed_charges integer
// for integer, on ONE env's timed-charge table (unit_schema.TIMED_CHARGE_TABLE,
// row = the charge's place in Simulation._timed_charges, ordinal order). Per
// row in order:
//   1. a heat charge whose latch is open: a disabled one is skipped; else its
//      tile's Q16 temperature MIRROR below trigger_temp_q16 skips it; at or
//      above it the latch closes (fired 1) and the fuse loads (ticks_to_fire =
//      fuse_ticks) -- the same sweep then counts it;
//   2. ticks_to_fire < 0: skipped (spent, disabled, waiting);
//   3. ticks_to_fire == 0: the charge's PAYLOAD (its tile, its row, no owner,
//      no approach, kind timed_charge) applied AT ONCE through the one payload
//      function (`apply`), then ticks_to_fire = period - 1, or -1 for a
//      one-shot;
//   4. else ticks_to_fire - 1.
// AT ONCE, not recorded for a later walk: step 1 reads the temperature mirror,
// which an earlier charge's blast writes in the same sweep (destroy_wall sets
// a destroyed tile's T to 0; bulk gas refreshes a cell's mirror).
//
// heat_sweep: src/simulation/heat_source_system.sweep_heat_sources integer for
// integer, on ONE env's heat-source table (HEAT_SOURCE_TABLE, ordinal order)
// and its blast log (unit_schema.BLAST_LOG_TABLE: every radius > 0 PAYLOAD of
// the tick so far, slots 2, 4c, 9e(d') included -- THE seam):
//   1. THE CHAIN: the log's discs in order, the list GROWING while it is read:
//      per disc, per living source in row order, the wall law of
//      apply_explosion at the source's tile (falloff_round(wall_q, d2, r^2)
//      inside d2 <= r^2, 0 for r <= 0); a positive damage lowers hp; at hp <= 0
//      the source dies (alive 0, enabled 0, hp 0) and its death payload (kind
//      heat_source, no owner, no approach) is applied AT ONCE, appending its
//      own disc to the log, which this loop then reads;
//   2. the log is emptied (the reference's `log.clear()`);
//   3. THE DEPOSIT: every living, enabled source whose cap is 0 or whose tile's
//      temperature mirror reads below the cap enqueues deposit_heat -- the
//      heat claims of no owner (none) and ONE heat DISC ADD LINEAR edit of
//      heat_q16 at radius heat_radius -- through the device edit door.
//
// THE BOUND (proved; the head refuses past it). Each source dies at most once
// (a dead source is skipped) and a death applies at most one PAYLOAD, which
// appends at most one disc (only a row with radius > 0 appends). So the discs
// the loop reads number at most D0 + S_d: D0 the log's count at the head, S_d
// the living sources whose death payload's row has radius > 0. The loop ends
// after at most D0 + S_d discs, and the log never holds more than that. The
// head checks D0 + S_d <= the log's capacity; past it the env is refused BY
// NAME (`heat_refused`, applier.HeatChainRefused) with nothing written.
//
// A FAULT (counted in `faults`; the Python face refuses the run): a row whose
// tile is off the grid, a payload index outside the table, a falloff operand
// past the kit's range (wall_q >= 2^31), an hp past +-2^62, a PAYLOAD the
// payload function faults or refuses, a deposit the edit door cannot hold. The
// sweep stops at the faulting row (the run is refused either way).
//
// Device-only header; included by cuda_applier.cu only. CLASS T.
#include "cuda_topology.cuh"
#include "cuda_unit_lib.cuh"
#include "cuda_unit_payload.cuh"
#include "object_tables_layout.h"
#include "applier_layout.h"
#include "unit_tables_layout.h"

namespace breach_cuda {
namespace hazards {

using namespace object_tables;
using namespace unit_tables;
using namespace applier;

constexpr int64_t HP_LIM = (int64_t)1 << 62;

// A PAYLOAD record of the call `execute_payload(gmap, queue, units, fy, fx,
// payload, draws, events=..., kind=<kind>)` (no owner, no approach).
__device__ inline void payload_record(int64_t* rec, int64_t fy, int64_t fx, int64_t p,
                                      int64_t kind) {
    for (int k = 0; k < APPLIER_F; ++k) rec[k] = 0;
    rec[APPLIER_FX_KIND] = APPLIER_KIND_PAYLOAD;
    rec[APPLIER_PAYLOAD_Y] = fy;
    rec[APPLIER_PAYLOAD_X] = fx;
    rec[APPLIER_PAYLOAD_PAYLOAD] = p;
    rec[APPLIER_PAYLOAD_OWNER] = -1;            // events.NO_ATTACKER
    rec[APPLIER_PAYLOAD_APPROACH_Y] = -1;
    rec[APPLIER_PAYLOAD_APPROACH_X] = -1;
    rec[APPLIER_PAYLOAD_HAS_APPROACH] = 0;
    rec[APPLIER_PAYLOAD_KIND] = kind;
}

// The tile mirror's temperature (an int32 Q16 plane) of an in-grid tile.
__device__ __forceinline__ int64_t tile_t(const topo::TopoEnv& E, int64_t y, int64_t x) {
    return (int64_t)E.temperature[(size_t)y * E.w + x];
}

__device__ __forceinline__ bool in_grid(const topo::TopoEnv& E, int64_t y, int64_t x) {
    return y >= 0 && y < E.h && x >= 0 && x < E.w;
}

// timed_charge_system.sweep_timed_charges (see the header). `apply(rec)` is the
// one PAYLOAD function on the stage's reach (unit_payload::PayloadResult);
// `own` (may be null) receives each applied record.
template <class ApplyFn>
__device__ inline void timed_sweep(const topo::TopoEnv& E, const int64_t* tab, int64_t* rows,
                                   int64_t n, ApplyFn&& apply, unit_lib::RecordSink* own,
                                   int64_t* B) {
    for (int64_t k = 0; k < n; ++k) {
        int64_t* c = rows + (size_t)k * OBJ_TIMED_CHARGE_W;
        const int64_t fy = c[OBJ_TIMED_CHARGE_COL_FY], fx = c[OBJ_TIMED_CHARGE_COL_FX];
        if (!in_grid(E, fy, fx)) {
            B[APPLIER_BOOK_FAULTS] += 1;
            return;
        }
        if (c[OBJ_TIMED_CHARGE_COL_HEAT] != 0 && c[OBJ_TIMED_CHARGE_COL_FIRED] == 0) {
            if (c[OBJ_TIMED_CHARGE_COL_ENABLED] == 0) continue;
            if (tile_t(E, fy, fx) < c[OBJ_TIMED_CHARGE_COL_TRIGGER_TEMP_Q16]) continue;
            c[OBJ_TIMED_CHARGE_COL_FIRED] = 1;
            c[OBJ_TIMED_CHARGE_COL_TICKS_TO_FIRE] = c[OBJ_TIMED_CHARGE_COL_FUSE_TICKS];
        }
        const int64_t t = c[OBJ_TIMED_CHARGE_COL_TICKS_TO_FIRE];
        if (t < 0) continue;
        if (t == 0) {
            const int64_t p = c[OBJ_TIMED_CHARGE_COL_PAYLOAD];
            if (p < 0 || p >= UNIT_TAB_PAYLOAD_REFUSE_W) {
                B[APPLIER_BOOK_FAULTS] += 1;
                return;
            }
            int64_t rec[APPLIER_F];
            payload_record(rec, fy, fx, p, APPLIER_PAYLOAD_KIND_TIMED_CHARGE);
            const int res = apply(rec);
            if (res != unit_payload::PAYLOAD_APPLIED) return;   // counted by `apply`
            if (own != nullptr) unit_lib::emit_record(*own, rec);
            B[APPLIER_BOOK_TIMED_FIRED] += 1;
            const int64_t per = c[OBJ_TIMED_CHARGE_COL_PERIOD_TICKS];
            c[OBJ_TIMED_CHARGE_COL_TICKS_TO_FIRE] = per > 0 ? per - 1 : -1;
        } else {
            c[OBJ_TIMED_CHARGE_COL_TICKS_TO_FIRE] = t - 1;
        }
    }
}

// heat_source_system._blast_damage_q16: the wall law at the source's tile.
// Returns -1 past the kit's range (a fault).
__device__ inline int64_t blast_damage(const int64_t* s, int64_t fy, int64_t fx, int64_t r,
                                       int64_t wall_q) {
    if (r <= 0) return 0;
    if (r > 2147483647LL) return -1;
    const int64_t dy = s[OBJ_HEAT_SOURCE_COL_FY] - fy, dx = s[OBJ_HEAT_SOURCE_COL_FX] - fx;
    if (dy > 2147483647LL || dy < -2147483647LL || dx > 2147483647LL || dx < -2147483647LL)
        return -1;
    const int64_t d2 = dy * dy + dx * dx;      // < 2^63
    const int64_t r2 = r * r;
    if (d2 > r2) return 0;
    return fixedpoint::falloff_round_q(wall_q, d2, r2);
}

// heat_source_system.sweep_heat_sources (see the header). `blast` is the env's
// blast log (its count grows as death payloads append); `door` / `claims` the
// deposit's edit door and claims list.
template <class ApplyFn>
__device__ inline void heat_sweep(const topo::TopoEnv& E, const int64_t* tab, int e,
                                  int64_t* rows, int64_t n, unit_lib::RecordSink& blast,
                                  const edit_door::DoorView* door, unit_lib::RecordSink* claims,
                                  ApplyFn&& apply, unit_lib::RecordSink* own, int64_t* B) {
    // ---- the head: every operand, then the chain's bound --------------------
    int64_t sd = 0;
    for (int64_t k = 0; k < n; ++k) {
        const int64_t* s = rows + (size_t)k * OBJ_HEAT_SOURCE_W;
        const int64_t p = s[OBJ_HEAT_SOURCE_COL_DEATH_PAYLOAD];
        const int64_t hp = s[OBJ_HEAT_SOURCE_COL_HP_Q16];
        if (!in_grid(E, s[OBJ_HEAT_SOURCE_COL_FY], s[OBJ_HEAT_SOURCE_COL_FX]) ||
            p < -1 || p >= UNIT_TAB_PAYLOAD_REFUSE_W || hp > HP_LIM || hp < -HP_LIM ||
            s[OBJ_HEAT_SOURCE_COL_HEAT_RADIUS] < 0 ||
            s[OBJ_HEAT_SOURCE_COL_HEAT_RADIUS] > edit_door::DISC_CENTRE_MAX) {
            B[APPLIER_BOOK_FAULTS] += 1;
            return;
        }
        if (s[OBJ_HEAT_SOURCE_COL_ALIVE] != 0 && p >= 0 && tab[UNIT_TAB_PAYLOAD_RADIUS + p] > 0)
            sd += 1;
    }
    if (*blast.count + sd > blast.cap) {
        B[APPLIER_BOOK_HEAT_REFUSED] += 1;
        return;
    }
    if (n > 0 && (door == nullptr || claims == nullptr)) {
        B[APPLIER_BOOK_FAULTS] += 1;
        return;
    }
    // ---- 1. the chain over the growing log -----------------------------------
    for (int64_t i = 0; i < *blast.count; ++i) {
        const int64_t* d = blast.recs + (size_t)i * blast.width;
        const int64_t fy = d[OBJ_BLAST_COL_FY], fx = d[OBJ_BLAST_COL_FX];
        const int64_t radius = d[OBJ_BLAST_COL_RADIUS], wall_q = d[OBJ_BLAST_COL_WALL_Q];
        for (int64_t k = 0; k < n; ++k) {
            int64_t* s = rows + (size_t)k * OBJ_HEAT_SOURCE_W;
            if (s[OBJ_HEAT_SOURCE_COL_ALIVE] == 0) continue;
            const int64_t dmg = blast_damage(s, fy, fx, radius, wall_q);
            if (dmg < 0) {
                B[APPLIER_BOOK_FAULTS] += 1;
                return;
            }
            if (dmg == 0) continue;
            s[OBJ_HEAT_SOURCE_COL_HP_Q16] -= dmg;
            if (s[OBJ_HEAT_SOURCE_COL_HP_Q16] > 0) continue;
            s[OBJ_HEAT_SOURCE_COL_ALIVE] = 0;            // _die
            s[OBJ_HEAT_SOURCE_COL_ENABLED] = 0;
            s[OBJ_HEAT_SOURCE_COL_HP_Q16] = 0;
            B[APPLIER_BOOK_HEAT_DEATHS] += 1;
            const int64_t p = s[OBJ_HEAT_SOURCE_COL_DEATH_PAYLOAD];
            if (p >= 0) {
                int64_t rec[APPLIER_F];
                payload_record(rec, s[OBJ_HEAT_SOURCE_COL_FY], s[OBJ_HEAT_SOURCE_COL_FX], p,
                               APPLIER_PAYLOAD_KIND_HEAT_SOURCE);
                const int res = apply(rec);
                if (res != unit_payload::PAYLOAD_APPLIED) return;   // counted by `apply`
                if (own != nullptr) unit_lib::emit_record(*own, rec);
            }
        }
    }
    // ---- 2. the log emptied ---------------------------------------------------
    *blast.count = 0;
    // ---- 3. the per-tick deposits ---------------------------------------------
    for (int64_t k = 0; k < n; ++k) {
        const int64_t* s = rows + (size_t)k * OBJ_HEAT_SOURCE_W;
        if (s[OBJ_HEAT_SOURCE_COL_ALIVE] == 0 || s[OBJ_HEAT_SOURCE_COL_ENABLED] == 0) continue;
        const int64_t fy = s[OBJ_HEAT_SOURCE_COL_FY], fx = s[OBJ_HEAT_SOURCE_COL_FX];
        const int64_t cap = s[OBJ_HEAT_SOURCE_COL_MAX_TEMP_Q16];
        if (cap != 0 && tile_t(E, fy, fx) >= cap) continue;
        const int64_t r_q = s[OBJ_HEAT_SOURCE_COL_HEAT_RADIUS] * edit_door::ONE_Q16;
        edit_door::disc_claims(*claims, unit_events::UNIT_CREDIT_HEAT, fy, fx, r_q, -1, E.h,
                               E.w);
        if (!unit_payload::disc_edit(*door, e, tab, EDQ_F_HEAT, -1, unit_payload::MODE_ADD,
                                     s[OBJ_HEAT_SOURCE_COL_HEAT_Q16],
                                     tab[UNIT_TAB_PAYLOAD_SRC_HEAT], fy, fx, r_q,
                                     edit_door::FALLOFF_LINEAR, E.h, E.w)) {
            B[APPLIER_BOOK_FAULTS] += 1;
            return;
        }
        B[APPLIER_BOOK_HEAT_DEPOSITS] += 1;
    }
}

}  // namespace hazards
}  // namespace breach_cuda
