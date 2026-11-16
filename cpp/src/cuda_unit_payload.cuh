#pragma once
// ============================================================================
// #29 P7c.1 -- THE PAYLOAD FUNCTION, the disc path (residency design 4.4c, the
// cut table's P7c.1 row): payloads.execute_payload as a device function of the
// per-env serial applier. #29 P7c.2 -- and every other row: the strength path
// (THE generic explosion), the air read, the blast smoke, the multi-tick
// source and bulk gas (the cut table's P7c.2 row); a row whose payload_refuse
// is not 0 is REFUSED by name (applier.PAYLOAD_REFUSALS: none ships), nothing
// written.
//
// THE ORACLE is src/simulation/payloads.py (execute_payload), physics.py
// (apply_explosion), exchange.py (apply_blast_damage), flash_blind.py
// (apply_flash_blind) and noise.py (emit); an arithmetic change lands there
// first, then here, IN THE SAME PATCH: the corpus detonates almost nothing,
// so the twin ledger does not trace this law; tests/cuda_unit_payloads_check.py
// is the guard.
//
// THE SUB-STEPS, in the reference's order, integer for integer:
//   1. radius > 0 -- apply_explosion: the atmosphere DISC edit (LINEAR, the
//      bulk currency), the smoke clear (a FLAT REMOVE DISC over 2/5 of the
//      radius) when its radius is > 0, then the disc d2 <= r^2 walked SERIALLY
//      in row-major order: a blast-wall material's hp less
//      falloff_round(wall_q, d2, r^2), destroy_wall through the topology seam
//      AT ONCE at <= 0; then, on a tile neither solid nor vacuum AS IT STANDS
//      NOW (an earlier tile of the same disc may have been destroyed), the 3x3
//      wave stencil's TILE edits (each neighbour neither solid nor vacuum as
//      it stands) of div_round(falloff_round(press_q * kw, d2, r^2), 16);
//      then the blast log's record (fy, fx, radius, wall_q).
//   2. the noise (noise.emit: radius 0 records nothing) on the noise list.
//   3. unit_damage > 0 -- apply_blast_damage: every valid, living row in row
//      order whose centre tile lies within the Q16 radius takes, at or above
//      the threshold, ONE BLAST packet of falloff_floor(max_d, d2, r_q^2)
//      through the IMMEDIATE apply_packet (no exposure landing in any stage:
//      the reference's), indirect, credited to the owner, its direction the
//      victim's centre less the detonation tile.
//   4. blind_radius_q > 0 -- apply_flash_blind: every living row within the
//      integer bound that SEES the tile (unit_sight::has_los, the sight form
//      over solid and sight_screen) gets BLINDED for floor(ticks * f * g)
//      unless it already holds a BLINDED at least that long (never shortened),
//      through apply_status.
//   4'. #29 P7c.2 THE AIR READ AND THE SOURCE, here: after the wall damage and
//      before any later edit or deposit, so a tile the blast opened is open
//      (cuda_unit_blast.cuh). A strength row walks its heat tiles
//      (reach_tiles over heat_r2, the approach seed rule), reads the air over
//      them, walks its source tiles (source_r2) and adds its one-tick source
//      (start_source: the walk's count divides the row's total). A row with
//      an explosive reads the air over blast_air.disc_tiles. A comparison row
//      (expansion_m3, no strength) walks expansion_r2 with no approach, adds
//      its first tick and appends its later ticks to the BLAST-SOURCE TABLE
//      (its tiles first, into the side table: start_expansion).
//   4''. the blast smoke: a strength row's FIXED-TOTAL TILES deposit
//      (deposit_over_tiles: the soot's units over the walked tiles' linear
//      weights, noisy, scaled by the air read); else emit_blast_smoke's
//      add_explosion_smoke (blast_smoke_peak_q's peak over the LINEAR disc of
//      its non-solid tiles; noisy, scaled by the disc read).
//   5. gas_species -- emit_gas: the disc's claims (channel = the slice), then
//      the LINEAR ADD DISC edit into that gas slice.
//   5'. bulk_gas -- inject_bulk_gas: every tile of the FLAT bulk_radius disc
//      not aperture_blocked, through topo::inject_gas_n_vec with its own book
//      pair (payload_bulk_gas / _rail); a tile whose slice would pass int32
//      refuses the WHOLE deposit by name (BulkGasOverflow; the reference
//      raises OverflowError), checked before its first write.
//   6. a strength row: the heat claims over the walked tiles, then the
//      FIXED-TOTAL TILES heat (heat_base_q x the air read's heat factor);
//      else ignite_radius > 0 -- ignite_ring: the flame's LINEAR MAX DISC edit
//      (IGNITE_CLAMP), the heat claims, the FLAT heat DISC edit (IGNITE_HEAT).
//   7. heat_amount > 0 -- deposit_heat: the heat claims, the LINEAR ADD DISC
//      (its amount times the heat factor when the row read the air).
//   (ExplosionEvent is render-only: no device form, dropped as elsewhere.)
// Every edit goes through THE DEVICE EDIT DOOR (cuda_edit_door.cuh), every
// claim onto THE CLAIMS LIST; the walk is thread 0's. #29 P7c.2: the walks
// and the deposits stay SERIAL -- the fan-out over the block's threads was
// left out (the measured cost: docs/residency_benchmark_2026-11-07.md).
//
// FAULTS: the record's operands (the payload row, the centre within
// +-2^20, the lists and the ring present) and every disc wall's new hp
// (inside int32) are checked BEFORE the first write: such a fault writes
// nothing. A fault met later (a row operand apply_packet / apply_status /
// destroy_wall refuses, a sight end) stops the effect where it is, as a
// CHEW's does; the run is refused by name either way.
//
// Device-only header (CUDA types); included by the applier's TU only.
// CLASS T: nothing in the played tick includes it.
#include <cstdint>

#include "applier_layout.h"         // GENERATED: PAYLOAD's fields, the books
#include "cuda_edit_door.cuh"       // the door's DISC / TILE forms, the claims
#include "cuda_topology.cuh"        // destroy_wall, TopoEnv
#include "cuda_unit_blast.cuh"      // #29 P7c.2: the walk, the air read, the source
#include "cuda_unit_damage.cuh"     // apply_packet, the event sinks
#include "cuda_unit_sight.cuh"      // has_los
#include "object_tables_layout.h"   // GENERATED: the noise list, the blast log
#include "unit_events_layout.h"     // GENERATED: causes, damage types, CREDIT_HEAT
#include "unit_tables_layout.h"     // GENERATED: the payload rows

namespace breach_cuda {
namespace unit_payload {

using namespace unit_lib;
using namespace unit_tables;
using namespace object_tables;
using namespace unit_events;

//: field_edit.EditMode.
constexpr int64_t MODE_ADD = 0, MODE_REMOVE = 1, MODE_MAX = 2;
//: The result of apply_payload.
enum PayloadResult { PAYLOAD_APPLIED = 0, PAYLOAD_FAULT = 1, PAYLOAD_REFUSED = 2,
                     PAYLOAD_BULK_REFUSED = 3 };

//: The 3x3 wave stencil (physics.apply_explosion): (ky, kx, kw) of entry k,
//: in its order (0, 0, 4), (-1, 0, 2), (1, 0, 2), (0, -1, 2), (0, 1, 2),
//: (-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1).
__device__ __forceinline__ void wave_k(int k, int64_t* ky, int64_t* kx, int64_t* kw) {
    const int8_t K[9][3] = {{0, 0, 4},   {-1, 0, 2},  {1, 0, 2},  {0, -1, 2}, {0, 1, 2},
                            {-1, -1, 1}, {-1, 1, 1}, {1, -1, 1}, {1, 1, 1}};
    *ky = K[k][0];
    *kx = K[k][1];
    *kw = K[k][2];
}

// Whether material id m takes the blast's wall damage (physics.
// _BLAST_WALL_MATERIALS: `material in` the tuple; an id outside the table
// never does).
__device__ __forceinline__ bool blast_wall(const int64_t* tab, int m) {
    return m >= 0 && m < UNIT_TAB_BLAST_WALL_MAT_W && tab[UNIT_TAB_BLAST_WALL_MAT + m] != 0;
}

// What one PAYLOAD effect reaches besides the env's planes and tables.
struct PayloadReach {
    int64_t* store;              // (N, U, W)
    int n_units;
    uint8_t* touched;            // the env's (U,) rows-changed flags
    EventSink* sink;             // the env's ring (nullptr: none -> a fault)
    const edit_door::DoorView* door;   // nullptr: none -> a fault
    RecordSink* claims;          // nullptr: none -> a fault
    RecordSink* noise;
    RecordSink* blast;
    // #29 P7c.2: the walks' scratch (nullptr: a row that walks, reads the air
    // or deposits bulk gas FAULTS), the blast-source table and its tile side
    // table (nullptr: a comparison row of more than one tick FAULTS)
    const unit_blast::Scratch* scratch = nullptr;
    RecordSink* sources = nullptr;
    RecordSink* source_tiles = nullptr;
};

__device__ __forceinline__ int64_t centre_tile(int64_t pos_q, int64_t fp) {
    return tile_of(pos_q) + fp / 2;      // Unit.center_tile_*: fp >= 1
}

__device__ __forceinline__ int64_t abs64(int64_t v) { return v < 0 ? -v : v; }

// The policy clamp of a field ordinal (the edit door's rows).
__device__ __forceinline__ void policy_clamp(const int64_t* tab, int field, int64_t* f,
                                             int64_t* lo, int64_t* hi) {
    *f = tab[UNIT_TAB_EDIT_CLAMP_FLAGS + field];
    *lo = tab[UNIT_TAB_EDIT_CLAMP_LO + field];
    *hi = tab[UNIT_TAB_EDIT_CLAMP_HI + field];
}

// A policy-clamped DISC edit (emit_gas, deposit_heat, the atmosphere, the clear).
__device__ __forceinline__ bool disc_edit(const edit_door::DoorView& D, int e,
                                          const int64_t* tab, int field, int64_t slice,
                                          int64_t mode, int64_t amount_q, int64_t source,
                                          int64_t cy, int64_t cx, int64_t r_q,
                                          int64_t falloff, int h, int w) {
    int64_t cf, lo, hi;
    policy_clamp(tab, field, &cf, &lo, &hi);
    return edit_door::door_disc_edit(D, e, field, slice, mode, amount_q, source, cy, cx, r_q,
                                     falloff, cf, lo, hi, h, w);
}

// payloads.execute_payload of one PAYLOAD record on env e (see the header).
__device__ inline int apply_payload(const int64_t* rec, const topo::TopoEnv& E,
                                    const int64_t* tab, int e, const PayloadReach& R,
                                    int64_t* B) {
    using namespace applier;
    const int h = E.h, w = E.w;
    const int64_t fy = rec[APPLIER_PAYLOAD_Y], fx = rec[APPLIER_PAYLOAD_X];
    const int64_t p = rec[APPLIER_PAYLOAD_PAYLOAD];
    const int64_t owner = rec[APPLIER_PAYLOAD_OWNER];
    // ---- the operands, before any write --------------------------------------
    if (p < 0 || p >= UNIT_TAB_PAYLOAD_REFUSE_W) return PAYLOAD_FAULT;
    if (tab[UNIT_TAB_PAYLOAD_REFUSE + p] != 0) return PAYLOAD_REFUSED;
    if (R.sink == nullptr || R.door == nullptr || R.claims == nullptr || R.noise == nullptr ||
        R.blast == nullptr)
        return PAYLOAD_FAULT;
    if (abs64(fy) > edit_door::DISC_CENTRE_MAX || abs64(fx) > edit_door::DISC_CENTRE_MAX)
        return PAYLOAD_FAULT;
    // #29 P7c.2: what the strength path, the air read and the source need
    const bool generic = tab[UNIT_TAB_PAYLOAD_GENERIC + p] != 0;
    const int64_t air_mode = tab[UNIT_TAB_PAYLOAD_AIR_MODE + p];
    const bool exp_live = tab[UNIT_TAB_PAYLOAD_EXPANSION_LIVE + p] != 0;
    const int64_t exp_ticks = tab[UNIT_TAB_PAYLOAD_EXPANSION_TICKS + p];
    if ((generic || air_mode != 0 || exp_live || tab[UNIT_TAB_PAYLOAD_BULK_SLICE + p] >= 0) &&
        R.scratch == nullptr)
        return PAYLOAD_FAULT;
    if (exp_live && exp_ticks > 1 && (R.sources == nullptr || R.source_tiles == nullptr))
        return PAYLOAD_FAULT;
    const bool has_appr = rec[APPLIER_PAYLOAD_HAS_APPROACH] != 0;
    const int64_t ay = rec[APPLIER_PAYLOAD_APPROACH_Y], ax = rec[APPLIER_PAYLOAD_APPROACH_X];
    if (has_appr &&
        (abs64(ay) > edit_door::DISC_CENTRE_MAX || abs64(ax) > edit_door::DISC_CENTRE_MAX))
        return PAYLOAD_FAULT;
    const int64_t radius = tab[UNIT_TAB_PAYLOAD_RADIUS + p];
    const int64_t r2 = radius * radius;
    const int64_t wall_q = tab[UNIT_TAB_PAYLOAD_WALL_Q + p];
    if (radius > 0) {
        // every disc wall's new hp inside int32 (each tile is decremented once,
        // and a destroy changes only its own tile's hp and material)
        for (int64_t dy = -radius; dy <= radius; ++dy)
            for (int64_t dx = -radius; dx <= radius; ++dx) {
                const int64_t ny = fy + dy, nx = fx + dx;
                if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                const int64_t d2 = dy * dy + dx * dx;
                if (d2 > r2) continue;
                const int i = (int)ny * w + (int)nx;
                if (!blast_wall(tab, (int)E.material[i])) continue;
                const int64_t d = fixedpoint::falloff_round_q(wall_q, d2, r2);
                const int64_t hp = (int64_t)E.wall_hp[i] - d;
                if (d < 0 || hp < -2147483648LL || hp > 2147483647LL) return PAYLOAD_FAULT;
            }
    }
    const edit_door::DoorView& D = *R.door;
    const int64_t src_x = tab[UNIT_TAB_PAYLOAD_SRC_EXPLOSION];
    const int64_t att = owner < 0 ? -1 : owner;      // _claim_tiles: the environment's
    // ---- 1. apply_explosion ----------------------------------------------------
    if (radius > 0) {
        if (!disc_edit(D, e, tab, EDQ_F_ATMOSPHERE, -1, MODE_ADD,
                       tab[UNIT_TAB_PAYLOAD_ATM_Q + p], src_x, fy, fx,
                       tab[UNIT_TAB_PAYLOAD_RADIUS_Q + p], edit_door::FALLOFF_LINEAR, h, w))
            return PAYLOAD_FAULT;
        const int64_t inner_q = tab[UNIT_TAB_PAYLOAD_INNER_Q + p];
        if (inner_q > 0 &&
            !disc_edit(D, e, tab, EDQ_F_SMOKE, tab[UNIT_TAB_PAYLOAD_SMOKE_SLICE], MODE_REMOVE,
                       tab[UNIT_TAB_PAYLOAD_SMOKE_CLEAR_Q], src_x, fy, fx, inner_q,
                       edit_door::FALLOFF_FLAT, h, w))
            return PAYLOAD_FAULT;
        const int64_t press_q = tab[UNIT_TAB_PAYLOAD_PRESS_Q + p];
        for (int64_t dy = -radius; dy <= radius; ++dy)
            for (int64_t dx = -radius; dx <= radius; ++dx) {
                const int64_t ny = fy + dy, nx = fx + dx;
                if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                const int64_t d2 = dy * dy + dx * dx;
                if (d2 > r2) continue;
                const int i = (int)ny * w + (int)nx;
                // structural wall damage, IMMEDIATE (the blast-wall materials)
                if (blast_wall(tab, (int)E.material[i])) {
                    const int64_t hp =
                        (int64_t)E.wall_hp[i] - fixedpoint::falloff_round_q(wall_q, d2, r2);
                    E.wall_hp[i] = (int32_t)hp;
                    if (hp <= 0) topo::destroy_wall(E, tab, (int)ny, (int)nx, B);
                }
                // the wave stencil, on the planes AS THEY STAND NOW
                if (!E.solid[i] && !E.is_vacuum[i]) {
                    for (int k = 0; k < 9; ++k) {
                        int64_t ky, kx, kw;
                        wave_k(k, &ky, &kx, &kw);
                        const int64_t sy = ny + ky, sx = nx + kx;
                        if (sy < 0 || sy >= h || sx < 0 || sx >= w) continue;
                        const int j = (int)sy * w + (int)sx;
                        if (E.solid[j] || E.is_vacuum[j]) continue;
                        const int64_t a_q = fixedpoint::div_round_q(
                            fixedpoint::falloff_round_q(press_q * kw, d2, r2), 16);
                        edit_door::door_tile_edit(D, e, tab, EDQ_F_WAVE_SOURCE, -1, MODE_ADD,
                                                  a_q, src_x, sy, sx, h, w);
                    }
                }
            }
        const int64_t bl[OBJ_BLAST_W] = {fy, fx, radius, wall_q, 1};
        static_assert(OBJ_BLAST_W == 5 && OBJ_BLAST_COL_WALL_Q == 3 && OBJ_BLAST_COL_LIVE == 4,
                      "the blast log's columns moved: update the record");
        emit_record(*R.blast, bl);
    }
    // ---- 2. the noise ------------------------------------------------------------
    const int64_t loud = tab[UNIT_TAB_PAYLOAD_LOUDNESS_TILES + p];
    if (loud > 0) {
        const int64_t nr[OBJ_NOISE_W] = {fy, fx, loud, 1};
        static_assert(OBJ_NOISE_W == 4 && OBJ_NOISE_COL_LIVE == 3,
                      "the noise list's columns moved: update the record");
        emit_record(*R.noise, nr);
    }
    // ---- 3. apply_blast_damage -------------------------------------------------
    if (tab[UNIT_TAB_PAYLOAD_BLAST_LIVE + p] != 0) {
        const int64_t r_q = tab[UNIT_TAB_PAYLOAD_RADIUS_Q + p];
        const int64_t r2_q = r_q * r_q;
        const int64_t max_d = tab[UNIT_TAB_PAYLOAD_UNIT_DAMAGE + p];
        const int64_t thr = tab[UNIT_TAB_BLAST_DAMAGE_THRESHOLD];
        for (int r = 0; r < R.n_units; ++r) {
            int64_t* row = unit_row(R.store, R.n_units, r);
            if (!row_valid(row) || !row_alive(row)) continue;
            const int64_t fp = row[UNIT_COL_FOOTPRINT];
            const int64_t dx = centre_tile(row[UNIT_COL_X_Q], fp) - fx;
            const int64_t dy = centre_tile(row[UNIT_COL_Y_Q], fp) - fy;
            if (r_q <= 0 || abs64(dx) > radius || abs64(dy) > radius) continue;
            const int64_t dx_q = dx << 16, dy_q = dy << 16;
            const int64_t d2 = dx_q * dx_q + dy_q * dy_q;           // Q32
            if (d2 > r2_q) continue;
            const int64_t dmg = fixedpoint::falloff_floor_q(max_d, d2, r2_q);
            if (dmg < 0) return PAYLOAD_FAULT;
            if (dmg < thr) continue;
            Packet pk;
            pk.amount_q = dmg << 16;
            pk.dtype = UNIT_DTYPE_BLAST;
            pk.attacker = attacker_value(owner);
            pk.indirect = true;
            pk.expose = 0;
            pk.dir_x_q = dx_q;
            pk.dir_y_q = dy_q;
            pk.ap_q = 0;
            pk.cause = UNIT_CAUSE_EXPLOSION;
            pk.mark_killed_by_zombie = false;
            bool changed = false, killed = false, bad = false;
            apply_packet(row, tab, pk, *R.sink, changed, &killed, &bad);   // IMMEDIATE
            if (bad) return PAYLOAD_FAULT;
            if (changed) R.touched[r] = 1;
        }
    }
    // ---- 4. apply_flash_blind ----------------------------------------------------
    const int64_t brq = tab[UNIT_TAB_PAYLOAD_BLIND_RADIUS_Q + p];
    if (brq > 0) {
        const int64_t r_tiles = (brq >> 16) + 1;
        const int64_t r2_max = r_tiles * r_tiles;
        const int64_t bticks = tab[UNIT_TAB_PAYLOAD_BLIND_TICKS + p];
        const int64_t away_q = tab[UNIT_TAB_PAYLOAD_BLIND_AWAY_Q + p];
        const int64_t front = tab[UNIT_TAB_FLASH_FRONT_Q16];
        const int64_t src = owner < 0 ? -1 : owner;
        for (int r = 0; r < R.n_units; ++r) {
            int64_t* row = unit_row(R.store, R.n_units, r);
            if (!row_valid(row) || !row_alive(row)) continue;
            const int64_t fp = row[UNIT_COL_FOOTPRINT];
            const int64_t ux = centre_tile(row[UNIT_COL_X_Q], fp);
            const int64_t uy = centre_tile(row[UNIT_COL_Y_Q], fp);
            const int64_t dx = fx - ux, dy = fy - uy;
            if (abs64(dx) > r_tiles || abs64(dy) > r_tiles) continue;
            const int64_t d2 = dx * dx + dy * dy;
            if (d2 > r2_max) continue;
            bool bad = false;
            const bool sees = unit_sight::has_los(
                reinterpret_cast<const uint8_t*>(E.solid),
                reinterpret_cast<const uint8_t*>(E.sight_screen), h, w, uy, ux, fy, fx, &bad);
            if (bad) return PAYLOAD_FAULT;
            if (!sees) continue;
            // distance_factor_q16
            const int64_t dist_q = (int64_t)fixedpoint::isqrt_u64_q((uint64_t)d2 << 32);
            if (dist_q >= brq) continue;
            const int64_t g = ONE - (dist_q * ONE) / brq;
            // facing_factor_q16
            int64_t f = ONE;
            if (dx != 0 || dy != 0) {
                const int64_t to_flash =
                    fixedpoint::atan2_q16((q16)(-dy * ONE), (q16)(dx * ONE));
                int64_t diff = wrap_q16(to_flash - row[UNIT_COL_FACING_Q]);
                diff = diff < 0 ? -diff : diff;
                if (diff > front)
                    f = ONE - ((ONE - away_q) * (diff - front)) / (PI_Q16 - front);
            }
            const int64_t ticks = (bticks * f * g) >> 32;
            if (ticks < 1) continue;
            const int64_t* s = row + UNIT_COL_STATUSES;
            bool longer = false;
            for (int i = 0; i < STATUS_ROWS && s[4 * i] >= 0; ++i)
                if (s[4 * i] == UNIT_STATUS_BLINDED && s[4 * i + 2] >= ticks) {
                    longer = true;                     // already blinded longer
                    break;
                }
            if (longer) continue;
            bool changed = false;
            if (apply_status(row, tab, UNIT_STATUS_BLINDED, 0, ticks, src, changed) ==
                APPLY_BAD)
                return PAYLOAD_FAULT;
            if (changed) R.touched[r] = 1;
        }
    }
    // ---- 4'. the air read and the source (cuda_unit_blast.cuh) ------------------
    const unit_blast::Scratch* S = R.scratch;
    int nh = 0;                         // the walked heat tiles (a strength row)
    int64_t heat_scale = ONE;           // read_blast_air's heat factor
    bool all_one = true;                // read_blast_air's smoke plane is None
    const int64_t r2h = tab[UNIT_TAB_PAYLOAD_HEAT_R2 + p];
    if (generic) {
        nh = unit_blast::reach(E, fy, fx, r2h, has_appr, ay, ax, *S, S->heat);
        heat_scale = unit_blast::air_read(E, tab, S->heat, nh, &all_one);
        const int ns = unit_blast::reach(E, fy, fx, tab[UNIT_TAB_PAYLOAD_SOURCE_R2 + p],
                                         has_appr, ay, ax, *S, S->src);
        if (ns > 0)                                                   // start_source
            unit_blast::add_source(
                E, tab, S->src, ns,
                fixedpoint::div_round_q(tab[UNIT_TAB_PAYLOAD_SOURCE_TOTAL_Q + p], ns), B);
    } else if (air_mode == 1) {
        const int nd =
            unit_blast::disc_tiles(E, fy, fx, tab[UNIT_TAB_PAYLOAD_AIR_DISC_R + p], S->heat);
        heat_scale = unit_blast::air_read(E, tab, S->heat, nd, &all_one);
    }
    if (exp_live) {                                                   // start_expansion
        const int ns = unit_blast::reach(E, fy, fx, tab[UNIT_TAB_PAYLOAD_SOURCE_R2 + p], false,
                                         0, 0, *S, S->src);
        if (ns > 0) {
            const int64_t s_q =
                fixedpoint::div_round_q(tab[UNIT_TAB_PAYLOAD_SOURCE_TOTAL_Q + p], ns);
            unit_blast::add_source(E, tab, S->src, ns, s_q, B);
            if (exp_ticks > 1) {
                RecordSink& st = *R.source_tiles;
                RecordSink& sr = *R.sources;
                const bool fit_t = *st.count + ns <= st.cap, fit_r = *sr.count + 1 <= sr.cap;
                if (!fit_t || !fit_r) {                 // all or nothing, counted
                    if (!fit_t) atomicAdd(st.dropped, 1ULL);
                    if (!fit_r) atomicAdd(sr.dropped, 1ULL);
                } else {
                    const int64_t start = *st.count;
                    for (int k = 0; k < ns; ++k) {
                        const int64_t t[OBJ_SOURCE_TILE_W] = {S->src[k] / w, S->src[k] % w, 1};
                        emit_record(st, t);
                    }
                    int64_t row[OBJ_SOURCE_W];
                    row[OBJ_SOURCE_COL_S_Q] = s_q;
                    row[OBJ_SOURCE_COL_TICKS] = exp_ticks - 1;
                    row[OBJ_SOURCE_COL_TILE_START] = start;
                    row[OBJ_SOURCE_COL_TILE_COUNT] = ns;
                    row[OBJ_SOURCE_COL_LIVE] = 1;
                    emit_record(sr, row);
                }
            }
        }
    }
    // ---- 4''. the blast smoke -----------------------------------------------------
    const int64_t noise_q = tab[UNIT_TAB_BLAST_SMOKE_NOISE_Q];
    const int64_t smoke_slice = tab[UNIT_TAB_PAYLOAD_SMOKE_SLICE];
    const int64_t src_smoke = tab[UNIT_TAB_PAYLOAD_SRC_EXPLOSION_SMOKE];
    const int32_t* HT = S != nullptr ? S->heat : nullptr;
    auto lin_w = [&](int k) -> int64_t {                // linear_weights
        const int64_t y = HT[k] / w - fy, x = HT[k] % w - fx;
        return fixedpoint::falloff_round_q(ONE, y * y + x * x, r2h);
    };
    int64_t norm = 0;
    if (generic) {
        for (int k = 0; k < nh; ++k) {
            const int64_t wk = lin_w(k);
            if (wk < 0) return PAYLOAD_FAULT;           // never in the proved range
            norm += wk;
        }
        const int64_t total = tab[UNIT_TAB_PAYLOAD_SOOT_UNITS_Q + p];
        if (total > 0 && nh > 0 && norm > 0) {          // deposit_over_tiles
            const int64_t amount_q =
                fixedpoint::div_round_q(total * 2 * ONE, 2 * ONE - noise_q);
            edit_door::door_tiles_edit(
                D, e, tab, EDQ_F_SMOKE, smoke_slice, MODE_ADD, amount_q, src_smoke, HT, nh, norm,
                noise_q, !all_one, lin_w,
                [&](int k) -> int64_t { return unit_blast::air_smoke_q(E, tab, HT[k]); });
        }
    } else if (tab[UNIT_TAB_PAYLOAD_SMOKE_LIVE + p] != 0) {     // add_explosion_smoke
        const int64_t r_q = tab[UNIT_TAB_PAYLOAD_RADIUS_Q + p];
        const int64_t units = tab[UNIT_TAB_PAYLOAD_SOOT_UNITS_Q + p];
        if (radius > 0 && units > 0) {
            const edit_door::DiscBox b = edit_door::disc_box(fy, fx, r_q, h, w);
            int64_t wsum = 0;                           // blast_smoke_peak_q
            for (int64_t r = b.r0; r <= b.r1; ++r)
                for (int64_t c = b.c0; c <= b.c1; ++c) {
                    const int64_t d2 = ((r - fy) * (r - fy) + (c - fx) * (c - fx)) << 32;
                    if (!(d2 < b.r2) || E.solid[(int)(r * w + c)]) continue;
                    const int64_t wq = fixedpoint::falloff_round_q(ONE, d2, b.r2);
                    if (wq < 0) return PAYLOAD_FAULT;
                    wsum += wq;
                }
            const int64_t peak =
                wsum > 0 ? fixedpoint::div_round_q(units * ONE * 2 * ONE,
                                                   wsum * (2 * ONE - noise_q))
                         : 0;
            if (peak > 0) {
                int64_t cf, lo, hi;
                policy_clamp(tab, EDQ_F_SMOKE, &cf, &lo, &hi);
                if (!edit_door::door_disc_edit_ns(
                        D, e, EDQ_F_SMOKE, smoke_slice, MODE_ADD, peak, src_smoke, fy, fx, r_q,
                        edit_door::FALLOFF_LINEAR, cf, lo, hi, noise_q,
                        air_mode != 0 && !all_one,
                        [&](int i) -> int64_t {
                            return E.solid[i] ? ONE : unit_blast::air_smoke_q(E, tab, i);
                        },
                        h, w))
                    return PAYLOAD_FAULT;
            }
        }
    }
    // ---- 5. emit_gas -------------------------------------------------------------
    const int64_t gas_id = tab[UNIT_TAB_PAYLOAD_GAS_ID + p];
    if (gas_id >= 0) {
        if (gas_id >= E.n_gases) return PAYLOAD_FAULT;
        const int64_t gr_q = tab[UNIT_TAB_PAYLOAD_GAS_RADIUS_Q + p];
        edit_door::disc_claims(*R.claims, gas_id, fy, fx, gr_q, att, h, w);
        if (!disc_edit(D, e, tab, EDQ_F_GAS, gas_id, MODE_ADD, tab[UNIT_TAB_PAYLOAD_GAS_Q + p],
                       tab[UNIT_TAB_PAYLOAD_SRC_GAS], fy, fx, gr_q, edit_door::FALLOFF_LINEAR,
                       h, w))
            return PAYLOAD_FAULT;
    }
    // ---- 5'. inject_bulk_gas ------------------------------------------------------
    const int64_t bulk_slice = tab[UNIT_TAB_PAYLOAD_BULK_SLICE + p];
    if (bulk_slice >= 0) {
        if (bulk_slice >= E.n_gases || E.n_gases > topo::GAS_MAX) return PAYLOAD_FAULT;
        const int64_t add_q = tab[UNIT_TAB_PAYLOAD_BULK_Q + p];
        if (add_q > 0) {
            int64_t add[topo::GAS_MAX];
            for (int g = 0; g < topo::GAS_MAX; ++g) add[g] = 0;
            add[bulk_slice] = add_q;
            // bulk_gas_tiles: the FLAT DISC minus _aperture_blocked, row-major
            const edit_door::DiscBox b =
                edit_door::disc_box(fy, fx, tab[UNIT_TAB_PAYLOAD_BULK_RADIUS_Q + p], h, w);
            int nb = 0;
            for (int64_t r = b.r0; r <= b.r1; ++r)
                for (int64_t c = b.c0; c <= b.c1; ++c) {
                    if (!((((r - fy) * (r - fy) + (c - fx) * (c - fx)) << 32) < b.r2)) continue;
                    if (topo::aperture_blocked(E, (int)r, (int)c)) continue;
                    S->src[nb++] = (int32_t)(r * w + c);
                }
            for (int k = 0; k < nb; ++k)                // the overflow, before any write
                if (!topo::inject_core_fits(E, tab, S->src[k], add)) return PAYLOAD_BULK_REFUSED;
            const topo::InjectVecBooks bk{APPLIER_BOOK_GE_PAYLOAD_BULK_GAS,
                                          APPLIER_BOOK_GE_PAYLOAD_BULK_GAS_RAIL};
            for (int k = 0; k < nb; ++k) {
                int64_t t_new = 0, rail = 0;
                if (topo::inject_gas_n_vec(E, tab, S->src[k] / w, S->src[k] % w, add,
                                           tab[UNIT_TAB_PAYLOAD_BULK_TEMP_Q + p],
                                           tab[UNIT_TAB_BULK_T_RAILS + 0],
                                           tab[UNIT_TAB_BULK_T_RAILS + 1], &t_new, &rail, B,
                                           bk) != 0)
                    return PAYLOAD_FAULT;
            }
        }
    }
    // ---- 6. a strength row's heat, else ignite_ring --------------------------------
    if (generic) {
        edit_door::tiles_claims(*R.claims, unit_events::UNIT_CREDIT_HEAT, HT, nh, att, w);
        const int64_t total =
            fixedpoint::div_round_q(tab[UNIT_TAB_PAYLOAD_HEAT_BASE_Q + p] * heat_scale, ONE);
        if (total > 0 && nh > 0 && norm > 0)            // deposit_over_tiles, no noise
            edit_door::door_tiles_edit(
                D, e, tab, EDQ_F_HEAT, -1, MODE_ADD,
                fixedpoint::div_round_q(total * 2 * ONE, 2 * ONE), tab[UNIT_TAB_PAYLOAD_SRC_HEAT],
                HT, nh, norm, 0, false, lin_w, [&](int) -> int64_t { return 1; });
    } else if (tab[UNIT_TAB_PAYLOAD_IGNITE_LIVE + p] != 0) {
        const int64_t ir_q = tab[UNIT_TAB_PAYLOAD_IGNITE_RADIUS_Q + p];
        const int64_t src_i = tab[UNIT_TAB_PAYLOAD_SRC_IGNITE];
        if (!edit_door::door_disc_edit(
                D, e, EDQ_F_FIRE, -1, MODE_MAX, tab[UNIT_TAB_PAYLOAD_IGNITE_Q + p], src_i, fy, fx,
                ir_q, edit_door::FALLOFF_LINEAR, tab[UNIT_TAB_PAYLOAD_IGNITE_CLAMP + 0],
                tab[UNIT_TAB_PAYLOAD_IGNITE_CLAMP + 1], tab[UNIT_TAB_PAYLOAD_IGNITE_CLAMP + 2], h,
                w))
            return PAYLOAD_FAULT;
        edit_door::disc_claims(*R.claims, unit_events::UNIT_CREDIT_HEAT, fy, fx, ir_q, att, h, w);
        if (!disc_edit(D, e, tab, EDQ_F_HEAT, -1, MODE_ADD, tab[UNIT_TAB_PAYLOAD_IGNITE_HEAT_Q],
                       src_i, fy, fx, ir_q, edit_door::FALLOFF_FLAT, h, w))
            return PAYLOAD_FAULT;
    }
    // ---- 7. deposit_heat ---------------------------------------------------------
    if (tab[UNIT_TAB_PAYLOAD_HEAT_LIVE + p] != 0) {
        const int64_t hr_q = tab[UNIT_TAB_PAYLOAD_HEAT_RADIUS_Q + p];
        int64_t heat_q = tab[UNIT_TAB_PAYLOAD_HEAT_Q + p];
        if (air_mode != 0) heat_q = fixedpoint::div_round_q(heat_q * heat_scale, ONE);
        edit_door::disc_claims(*R.claims, unit_events::UNIT_CREDIT_HEAT, fy, fx, hr_q, att, h, w);
        if (!disc_edit(D, e, tab, EDQ_F_HEAT, -1, MODE_ADD, heat_q,
                       tab[UNIT_TAB_PAYLOAD_SRC_HEAT], fy, fx, hr_q, edit_door::FALLOFF_LINEAR,
                       h, w))
            return PAYLOAD_FAULT;
    }
    return PAYLOAD_APPLIED;
}

static_assert(UNIT_TAB_PAYLOAD_IGNITE_CLAMP_W == 3, "the ring's clamp row is (flags, lo, hi)");
static_assert(UNIT_TAB_BLAST_WALL_MAT_W == UNIT_TAB_MAT_FLAMMABLE_W,
              "the blast-wall row is one per material");

}  // namespace unit_payload
}  // namespace breach_cuda
