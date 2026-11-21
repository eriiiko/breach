// ============================================================================
// #29 P6.4a -- SLOT 1b, THE HAND VERBS on the device unit store
// (Simulation._consume_direct_intents on the continuous path). See
// cuda_unit_kernels.h; residency design §4.3 row "1 events, 1b intents",
// §4.4b.
//
// Integer for integer the reference, per valid row in id order:
//   * a dead or turned (zombie) row DROPS a held trigger, a throw wind-up and
//     a plant under way (nothing leaves the hand, the counts were never spent);
//   * every living marine consumes ITS OWN columns -- no possession column
//     (design §4.4b "Settled 2026-11-14": a living marine outside the host's
//     possessed set holds nothing slot 1b consumes, tests/test_possession_
//     carries_nothing.py) -- in the reference's fixed order:
//       THROW  the latch -> can_throw -> the wind-up (HANDLING) or, at 0 ticks,
//              the release now along the intent's vector; then the wind-up's
//              advance (cannot act: dropped; the release tick: released along
//              the facing the unit has then);
//       PLANT  the latch -> can_plant -> the plant (HANDLING); its advance (the
//              last tick: the charge laid on the centre tile);
//       USE    the latch -> the live door in reach the unit faces (#215;
//              the door spans a per-tick table, in sim._doors order);
//       RELOAD the latch -> can_reload -> mag_settle -> reload_begin;
//       SHOVE  the latch -> can_shove -> the target (shove_target over the
//              env's other rows) -> the impulse (the PUSH hook) for the TARGET.
//
// What is not a write to the unit's own row:
//   * a grenade released, a charge laid, a door flipped are REQUEST records
//     (src/simulation/unit_requests.py; unit_requests_layout.h) staged per row
//     and gathered per env in row (= id) order -- the order slot 2 detonates
//     in and charges.place counts in;
//   * a shove's impulse lands on ANOTHER row: each shover stages (target row,
//     jx, jy, angular) in a per-row scratch, and a second kernel lets each
//     target sum the pushes aimed at it in shover-row order (unit_impulse.add:
//     integer sums of per-input terms -- the lever clamped per input -- so the
//     result never depends on the thread schedule);
//   * which plant a full charge table refuses (charges.place counts live
//     charges, in id order) is decided by ONE thread per env walking the
//     gathered list against the env's live-charge count (a per-tick input):
//     an accepted plant spends the planter's charge (has_explosive), a refused
//     one is counted in `refused`.
//
// Proved ranges (past them the row FAULTS, left as it was, its requests and
// shove dropped): a tick and a duration added to it below TICK_MAX; a throw
// vector component and a fuse below 2^40; a kind or a weapon index inside its
// table; the shove's impulse below 2^35 (its axis split below 2^51, its
// magnitude's square in 128 bits); an accumulator sum inside int64.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_object_table.h"
#include "cuda_unit_lib.cuh"
#include "object_tables_layout.h"   // GENERATED: the door and charge rows (#29 P7e.4)
#include "unit_requests_layout.h"   // GENERATED: the request records

namespace breach_cuda {

namespace {

using namespace unit_lib;
using namespace unit_requests;
using namespace object_tables;   // #29 P7e.4: the door and charge rows

//: The per-row shove scratch: target row (-1 none), jx, jy, angular, the
//: row's own "changed" flag from the first kernel.
constexpr int SC_W = 5;
constexpr int SC_TARGET = 0, SC_JX = 1, SC_JY = 2, SC_DL = 3, SC_CHANGED = 4;
constexpr int SPAN_W = 3;
constexpr int64_t COMPONENT_MAX = (int64_t)1 << 40;
constexpr int64_t PUSH_MAX = (int64_t)1 << 35;

// A grown, never shrunk device buffer (no allocation in a steady state).
struct GrowBuf {
    int64_t* d = nullptr;
    size_t cap = 0;
    int64_t* get(size_t n) {
        if (n > cap) {
            counted_sync(tstream(), "cuda_unit_verbs.cu:78");
            if (d) cudaFree(d);
            d = nullptr;
            if (cudaMalloc(reinterpret_cast<void**>(&d), n * sizeof(int64_t)) != cudaSuccess)
                throw std::runtime_error("unit_verbs: scratch allocation failed");
            cap = n;
        }
        return d;
    }
};
GrowBuf g_scratch;
GrowBuf g_inputs;   // #29 P7e.4: the spans and live counts built from the tables

__device__ __forceinline__ bool tick_ok(int64_t v) { return v >= -TICK_MAX && v <= TICK_MAX; }

__device__ __forceinline__ int64_t center_tile(int64_t q, int64_t fp) {
    return tile_of(q) + fp / 2;          // fp >= 1: floor == truncation
}

__device__ __forceinline__ void blank_record(int64_t* rec, int64_t kind, int64_t uid,
                                             int64_t tick) {
    for (int f = 0; f < UNIT_REQ_F; ++f) rec[f] = 0;
    rec[UNIT_RQ_KIND] = kind;
    rec[UNIT_RQ_UNIT_ID] = uid;
    rec[UNIT_RQ_TICK] = tick;
    rec[UNIT_RQ_DOOR] = UNIT_REQ_NO_DOOR;
}

// Simulation._release_grenade: kind `kind` leaves the hand along (dx, dy) at
// `reach` (THROW_NO_REACH = the fixed 6 tiles), its fuse clamped, one spent.
__device__ inline void release_grenade(int64_t* loc, const int64_t* tab, int64_t tick,
                                       int64_t kind, int64_t fuse, int64_t reach, int64_t dx,
                                       int64_t dy, const Plane& pl, RecordSink& sink,
                                       bool* bad) {
    if (kind < 0 || kind >= UNIT_COL_GRENADE_COUNTS_W) {
        *bad = true;
        return;
    }
    if (loc[UNIT_COL_GRENADE_COUNTS + kind] <= 0) return;
    if (dx > COMPONENT_MAX || dx < -COMPONENT_MAX || dy > COMPONENT_MAX ||
        dy < -COMPONENT_MAX || fuse > COMPONENT_MAX || fuse < -COMPONENT_MAX) {
        *bad = true;
        return;
    }
    const int64_t fp = loc[UNIT_COL_FOOTPRINT];
    const int64_t cx = center_tile(loc[UNIT_COL_X_Q], fp);
    const int64_t cy = center_tile(loc[UNIT_COL_Y_Q], fp);
    int64_t r;
    if (reach == THROW_NO_REACH) {
        r = (int64_t)6 << 16;
    } else {
        const int64_t max_q = tab[UNIT_TAB_HAND_MAX_REACH_Q];
        r = reach < 0 ? 0 : (reach > max_q ? max_q : reach);
    }
    const int64_t half = (int64_t)1 << 15;
    const int64_t lo = tab[UNIT_TAB_HAND_FUSE_LO_TICKS], hi = tab[UNIT_TAB_HAND_FUSE_HI_TICKS];
    int64_t f = fuse > lo ? fuse : lo;
    f = f < hi ? f : hi;
    int64_t rec[UNIT_REQ_F];
    blank_record(rec, UNIT_RQKIND_THROW, loc[UNIT_COL_ID], tick);
    // #215: the start is the HAND (combat.release_point over the flight plane)
    const int64_t xq = loc[UNIT_COL_X_Q], yq = loc[UNIT_COL_Y_Q];
    // the release point's proved range: a position within +-2^25 (512 tiles), the
    // throw vector within +-2^20, the footprint 0 .. 64 (products below 2^61)
    const int64_t RP_POS = (int64_t)1 << 25, RP_DIR = (int64_t)1 << 20;
    if (xq > RP_POS || xq < -RP_POS || yq > RP_POS || yq < -RP_POS || dx > RP_DIR ||
        dx < -RP_DIR || dy > RP_DIR || dy < -RP_DIR || fp < 0 || fp > 64) {
        *bad = true;
        return;
    }
    int64_t sx, sy;
    release_point(pl, xq, yq, fp, dx, dy, tab[UNIT_TAB_HAND_RELEASE_FORWARD_Q],
                  tab[UNIT_TAB_HAND_RELEASE_SIDE_Q], &sx, &sy);
    rec[UNIT_RQ_START_X_Q] = sx;
    rec[UNIT_RQ_START_Y_Q] = sy;
    rec[UNIT_RQ_TARGET_X_Q] = (cx << 16) + half + fixedpoint::mul_q16_round_q(dx, r);
    rec[UNIT_RQ_TARGET_Y_Q] = (cy << 16) + half + fixedpoint::mul_q16_round_q(dy, r);
    rec[UNIT_RQ_TICKS] = f;
    rec[UNIT_RQ_ITEM] = kind;
    rec[UNIT_RQ_ACCEPTED] = 1;
    emit_record(sink, rec);
    loc[UNIT_COL_GRENADE_COUNTS + kind] -= 1;
}

// Simulation._start_throw (the latch's columns already cleared).
__device__ inline void start_throw(int64_t* loc, const int64_t* tab, int64_t tick,
                                   const int64_t* pt, const Plane& pl, RecordSink& sink,
                                   bool* bad) {
    const int64_t kind = pt[5];
    if (!can_throw(loc, tab, kind, bad) || *bad) return;
    const int64_t ticks = handling_ticks(loc, tab[UNIT_TAB_HAND_WINDUP_CLASS_TICKS]);
    if (ticks == 0) {
        release_grenade(loc, tab, tick, kind, pt[3], pt[4], pt[1], pt[2], pl, sink, bad);
        return;
    }
    if (!tick_ok(tick + ticks) || pt[3] > COMPONENT_MAX || pt[3] < -COMPONENT_MAX) {
        *bad = true;
        return;
    }
    loc[UNIT_COL_THROW_RELEASE_TICK] = tick + ticks;
    loc[UNIT_COL_THROW_KIND] = kind;
    loc[UNIT_COL_THROW_FUSE_TICKS] = pt[3];
    loc[UNIT_COL_THROW_REACH_Q] = pt[4];
}

// Simulation._advance_throw.
__device__ inline void advance_throw(int64_t* loc, const int64_t* tab, int64_t tick,
                                     const Plane& pl, RecordSink& sink, bool* bad) {
    if (loc[UNIT_COL_THROW_RELEASE_TICK] < 0) return;
    if (!can_act(loc, tab, bad)) {
        loc[UNIT_COL_THROW_RELEASE_TICK] = -1;
        return;
    }
    if (tick < loc[UNIT_COL_THROW_RELEASE_TICK]) return;
    loc[UNIT_COL_THROW_RELEASE_TICK] = -1;
    const int64_t f = loc[UNIT_COL_FACING_Q];
    if (!fits_i32(f)) {
        *bad = true;
        return;
    }
    const int64_t c = fixedpoint::cos_q16((q16)f);
    const int64_t s = fixedpoint::sin_q16((q16)f);
    release_grenade(loc, tab, tick, loc[UNIT_COL_THROW_KIND], loc[UNIT_COL_THROW_FUSE_TICKS],
                    loc[UNIT_COL_THROW_REACH_Q], c, -s, pl, sink, bad);
}

// Simulation._start_plant (the latch already cleared).
__device__ inline void start_plant(int64_t* loc, const int64_t* tab, int64_t tick,
                                   int64_t fuse, bool* bad) {
    if (!can_plant(loc, tab, tick, bad) || *bad) return;
    const int64_t ticks = handling_ticks(loc, tab[UNIT_TAB_PLANT_CLASS_TICKS]);
    if (!tick_ok(tick + ticks) || fuse > COMPONENT_MAX || fuse < -COMPONENT_MAX) {
        *bad = true;
        return;
    }
    loc[UNIT_COL_PLANT_DONE_TICK] = tick + ticks;
    loc[UNIT_COL_PLANT_FUSE_TICKS] = fuse;
}

// Simulation._advance_plant: on the last tick the charge's PLANT request
// (accepted, and the charge spent, by the per-env pass).
__device__ inline void advance_plant(int64_t* loc, const int64_t* tab, int64_t tick,
                                     RecordSink& sink, bool* bad) {
    if (loc[UNIT_COL_PLANT_DONE_TICK] < 0) return;
    if (!can_act(loc, tab, bad)) {
        loc[UNIT_COL_PLANT_DONE_TICK] = -1;
        return;
    }
    if (tick < loc[UNIT_COL_PLANT_DONE_TICK]) return;
    loc[UNIT_COL_PLANT_DONE_TICK] = -1;
    const int64_t fp = loc[UNIT_COL_FOOTPRINT];
    const int64_t ty = center_tile(loc[UNIT_COL_Y_Q], fp);
    const int64_t tx = center_tile(loc[UNIT_COL_X_Q], fp);
    const int64_t fuse = loc[UNIT_COL_PLANT_FUSE_TICKS];
    if (fuse > COMPONENT_MAX || fuse < -COMPONENT_MAX) {
        *bad = true;
        return;
    }
    int64_t rec[UNIT_REQ_F];
    blank_record(rec, UNIT_RQKIND_PLANT, loc[UNIT_COL_ID], tick);
    rec[UNIT_RQ_TILE_X] = tx;
    rec[UNIT_RQ_TILE_Y] = ty;
    rec[UNIT_RQ_TICKS] = tick + fuse;
    rec[UNIT_RQ_ITEM] = loc[UNIT_COL_CHARGE_AMMO];
    rec[UNIT_RQ_APPROACH_Y] = ty;
    rec[UNIT_RQ_APPROACH_X] = tx;
    rec[UNIT_RQ_ACCEPTED] = 0;                    // the per-env pass decides
    emit_record(sink, rec);
}

// verb_gates.use_door (#215): among the live doors in reach (a span tile in
// the unit's footprint plus its one-tile ring: unit_lib::reach_key, the
// library's one statement of the reach since #29 P8.4c), the one the unit FACES --
// the smallest verb_gates.facing_off: |wrap_q16(atan2_q16(-dy, dx) - facing)|
// to the centre of the door's NEAREST span tile from the footprint centre
// (squared Q16 distance, ties to the earlier tile in span order; a tile
// centre on the unit's centre is 0 off) -- ties to the door holding the
// row-major first ring tile (the scan order before #215). ONE door in reach:
// that door, its facing never read. spans: this env's (n, 3) rows (door
// index, ty, tx) in door order, each door's rows consecutive in span order;
// index -1 a pad. -1 = none. An operand past the kit's int32 range faults.
__device__ inline int64_t use_door(const int64_t* loc, const int64_t* spans, int n_spans,
                                   bool* bad) {
    const int64_t fp = loc[UNIT_COL_FOOTPRINT];
    // pass 1: the candidates (doors with a ring tile), each door's reach key
    int n_cand = 0;
    int64_t first = -1, first_k = 0;
    for (int i = 0; i < n_spans;) {
        const int64_t door = spans[(size_t)i * SPAN_W];
        int j = i;
        int64_t k_door = -1;
        for (; j < n_spans && spans[(size_t)j * SPAN_W] == door; ++j) {
            const int64_t* s = spans + (size_t)j * SPAN_W;
            const int64_t k = reach_key(loc, s[1], s[2]);    // the library's reach (P8.4c)
            if (k < 0) continue;
            if (k_door < 0 || k < k_door) k_door = k;
        }
        if (door >= 0 && k_door >= 0) {
            ++n_cand;
            if (first < 0 || k_door < first_k) {
                first = door;
                first_k = k_door;
            }
        }
        i = j;
    }
    if (n_cand <= 1) return first;
    // pass 2: several in reach -> the smallest (facing off, reach key)
    const int64_t cx = centre_q(loc[UNIT_COL_X_Q], fp), cy = centre_q(loc[UNIT_COL_Y_Q], fp);
    const int64_t facing = loc[UNIT_COL_FACING_Q];
    const int64_t half = ONE >> 1;
    int64_t best = -1, best_off = 0, best_k = 0;
    for (int i = 0; i < n_spans;) {
        const int64_t door = spans[(size_t)i * SPAN_W];
        int j = i;
        int64_t k_door = -1, d2_min = -1, ndx = 0, ndy = 0;
        for (; j < n_spans && spans[(size_t)j * SPAN_W] == door; ++j) {
            const int64_t* s = spans + (size_t)j * SPAN_W;
            const int64_t k = reach_key(loc, s[1], s[2]);
            if (k >= 0 && (k_door < 0 || k < k_door)) k_door = k;
            const int64_t dx = (s[2] << 16) + half - cx, dy = (s[1] << 16) + half - cy;
            if (!fits_i32(dx) || !fits_i32(dy)) {
                *bad = true;
                return -1;
            }
            const int64_t d2 = dx * dx + dy * dy;
            if (d2_min < 0 || d2 < d2_min) {
                d2_min = d2;
                ndx = dx;
                ndy = dy;
            }
        }
        if (door >= 0 && k_door >= 0) {
            int64_t off = 0;
            if (ndx != 0 || ndy != 0) {
                off = wrap_q16(fixedpoint::atan2_q16((q16)(-ndy), (q16)ndx) - facing);
                if (off < 0) off = -off;
            }
            if (best < 0 || off < best_off || (off == best_off && k_door < best_k)) {
                best = door;
                best_off = off;
                best_k = k_door;
            }
        }
        i = j;
    }
    return best;
}

// Simulation._reload past its gate: combat.mag_settle, then reload_begin.
__device__ inline void reload(int64_t* loc, const int64_t* tab, int64_t tick, bool* bad) {
    // #29 P7a.2: the magazine law is the unit library's (slot 4 runs it too)
    const int64_t w = loc[UNIT_COL_WEAPON_ID];          // the gate proved it a row
    mag_settle(loc, tab, w, tick);
    reload_begin(loc, tab, w, tick, bad);
}

// unit_impulse.shove_target: among the env's OTHER living rows (any team),
// the adjacent one within the shove's cone whose bearing is nearest the
// facing, ties to the lower id; -1 = none. Reads the slot-start rows (1b
// writes no position, footprint, life state or id).
__device__ inline int shove_target(int64_t* store, int n_units, int r, const int64_t* loc,
                                   const int64_t* tab, bool* bad) {
    const int64_t fp = loc[UNIT_COL_FOOTPRINT];
    const int64_t cx = centre_q(loc[UNIT_COL_X_Q], fp), cy = centre_q(loc[UNIT_COL_Y_Q], fp);
    const int64_t facing = loc[UNIT_COL_FACING_Q];
    int best = -1;
    int64_t best_off = 0, best_id = 0;
    for (int t = 0; t < n_units; ++t) {
        if (t == r) continue;
        const int64_t* tr = unit_row(store, n_units, t);
        if (!row_valid(tr) || !row_alive(tr)) continue;
        const int64_t tfp = tr[UNIT_COL_FOOTPRINT];
        const int64_t dx = centre_q(tr[UNIT_COL_X_Q], tfp) - cx;
        const int64_t dy = centre_q(tr[UNIT_COL_Y_Q], tfp) - cy;
        const int64_t reach = ((fp + tfp) << 15) + ONE;
        if (dx > reach || dx < -reach || dy > reach || dy < -reach) continue;   // beyond: dx^2 > reach^2
        if (dx * dx + dy * dy > reach * reach) continue;
        if (dx == 0 && dy == 0) continue;
        if (!fits_i32(-dy) || !fits_i32(dx)) {
            *bad = true;
            return -1;
        }
        int64_t off = wrap_q16(fixedpoint::atan2_q16((q16)(-dy), (q16)dx) - facing);
        if (off < 0) off = -off;
        if (off > tab[UNIT_TAB_SHOVE_CONE_Q]) continue;
        const int64_t id = tr[UNIT_COL_ID];
        if (best < 0 || off < best_off || (off == best_off && id < best_id)) {
            best = t;
            best_off = off;
            best_id = id;
        }
    }
    return best;
}

// unit_impulse.shove past its cooldown and with a target: the impulse along
// the facing (the PUSH hook), its lever about the target's centre, the
// angular part angular_q's (the lever clamped to the body radius); staged
// for the target's row. Returns false when the row faults.
__device__ inline void stage_shove(const int64_t* loc, const int64_t* tr, const int64_t* tab,
                                   int t, int64_t* sc, bool* bad) {
    const int64_t f = loc[UNIT_COL_FACING_Q];
    if (!fits_i32(f)) {
        *bad = true;
        return;
    }
    const int64_t sx = fixedpoint::cos_q16((q16)f);
    const int64_t sy = -(int64_t)fixedpoint::sin_q16((q16)f);
    const int64_t push = scaled(loc, UNIT_HOOK_PUSH, tab[UNIT_TAB_SHOVE_Q]);
    if (push > PUSH_MAX || push < -PUSH_MAX) {
        *bad = true;
        return;
    }
    const int64_t jx = fixedpoint::shr_trunc_q(push * sx, 16);
    const int64_t jy = fixedpoint::shr_trunc_q(push * sy, 16);
    const int64_t fp = loc[UNIT_COL_FOOTPRINT], tfp = tr[UNIT_COL_FOOTPRINT];
    const int64_t ux = centre_q(loc[UNIT_COL_X_Q], fp), uy = centre_q(loc[UNIT_COL_Y_Q], fp);
    const int64_t tx = centre_q(tr[UNIT_COL_X_Q], tfp), ty = centre_q(tr[UNIT_COL_Y_Q], tfp);
    const int64_t lever = fixedpoint::shr_trunc_q((tx - ux) * sy - (ty - uy) * sx, 16);
    int64_t dl = 0;
    if (lever != 0)                                    // unit_impulse.add: `if lever:`
        dl = angular_q(tab, jx, jy, lever, ONE, bad);  // the library's (P7a.1)
    sc[SC_TARGET] = t;
    sc[SC_JX] = jx;
    sc[SC_JY] = jy;
    sc[SC_DL] = dl;
}

// Kernel 1: one thread per row -- every own-row write, the requests staged,
// the shove staged.
__global__ void k_verbs(int64_t* store, int n_units, const int64_t* tables, const int64_t* trow,
                        const int64_t* spans, int n_spans, RecordListView rq, int64_t* scratch,
                        const uint8_t* flight, int fh, int fw, unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* sc = scratch + ((size_t)benv() * n_units + r) * SC_W;
    sc[SC_TARGET] = -1;
    sc[SC_JX] = sc[SC_JY] = sc[SC_DL] = 0;
    sc[SC_CHANGED] = 0;
    int64_t* row = unit_row(store, n_units, r);
    if (!row_valid(row)) return;
    const int64_t* tab = env_tables(tables);
    const int64_t tick = unit_draws_dev::trow_tick(trow, benv());
    int64_t loc[UNIT_STORE_W];
    for (int c = 0; c < UNIT_STORE_W; ++c) loc[c] = row[c];
    RecordSink sink = list_row_sink(rq, n_units, r);
    bool bad = !tick_ok(tick);
    const Plane pl{flight + (size_t)benv() * fh * fw, fh, fw};   // #215: the flight plane

    if (!bad && !acts(loc)) {
        // the drop branch: a dead or turned row releases what it held
        loc[UNIT_COL_LIVE_TRIGGER] = 0;
        if (loc[UNIT_COL_THROW_RELEASE_TICK] >= 0) loc[UNIT_COL_THROW_RELEASE_TICK] = -1;
        if (loc[UNIT_COL_PLANT_DONE_TICK] >= 0) loc[UNIT_COL_PLANT_DONE_TICK] = -1;
    } else if (!bad) {
        // THROW -> start the wind-up (the latch); a wind-up under way advances
        int64_t pt[UNIT_COL_PENDING_THROW_W];
        for (int i = 0; i < UNIT_COL_PENDING_THROW_W; ++i) pt[i] = loc[UNIT_COL_PENDING_THROW + i];
        if (pt[0]) {
            for (int i = 0; i < UNIT_COL_PENDING_THROW_W; ++i) loc[UNIT_COL_PENDING_THROW + i] = 0;
            start_throw(loc, tab, tick, pt, pl, sink, &bad);
        }
        if (!bad) advance_throw(loc, tab, tick, pl, sink, &bad);
        // PLANT -> start the plant (the latch); a plant under way advances
        if (!bad) {
            const int64_t pp = loc[UNIT_COL_PENDING_PLANT];
            if (pp != NO_PLANT) {
                loc[UNIT_COL_PENDING_PLANT] = NO_PLANT;
                start_plant(loc, tab, tick, pp, &bad);
            }
        }
        if (!bad) advance_plant(loc, tab, tick, sink, &bad);
        // USE -> the door in reach flips (a request)
        if (!bad && loc[UNIT_COL_PENDING_USE]) {
            loc[UNIT_COL_PENDING_USE] = 0;
            const int64_t door = use_door(loc, bptr(spans, (size_t)n_spans * SPAN_W), n_spans,
                                         &bad);
            if (!bad && door >= 0) {
                int64_t rec[UNIT_REQ_F];
                blank_record(rec, UNIT_RQKIND_USE, loc[UNIT_COL_ID], tick);
                rec[UNIT_RQ_DOOR] = door;
                rec[UNIT_RQ_ACCEPTED] = 1;
                emit_record(sink, rec);
            }
        }
        // RELOAD -> discard the magazine, run the stall
        if (!bad && loc[UNIT_COL_PENDING_RELOAD]) {
            loc[UNIT_COL_PENDING_RELOAD] = 0;
            if (can_reload(loc, tab, tick, &bad) && !bad) reload(loc, tab, tick, &bad);
        }
        // SHOVE -> push the adjacent unit nearest the facing
        if (!bad && loc[UNIT_COL_PENDING_SHOVE]) {
            loc[UNIT_COL_PENDING_SHOVE] = 0;
            if (can_shove(loc, tab, tick, &bad) && !bad) {
                const int t = shove_target(store, n_units, r, loc, tab, &bad);
                if (!bad && t >= 0) {
                    stage_shove(loc, unit_row(store, n_units, t), tab, t, sc, &bad);
                    if (!bad) loc[UNIT_COL_LAST_SHOVE_TICK] = tick;
                }
            }
        }
    }
    if (bad) {                       // a fault: the row, its requests, its shove dropped
        *sink.count = 0;
        sc[SC_TARGET] = -1;
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
    sc[SC_CHANGED] = changed ? 1 : 0;
}

__device__ __forceinline__ bool add_ovf(int64_t a, int64_t b, int64_t* out) {
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) return true;
    *out = a + b;
    return false;
}

// Kernel 2: one thread per row -- the pushes aimed at it, summed in shover-row
// order (unit_impulse.add's three accumulators), and the row counted once.
__global__ void k_verbs_push(int64_t* store, int n_units, const int64_t* scratch,
                             unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    if (!row_valid(row)) return;
    const int64_t* sc = scratch + (size_t)benv() * n_units * SC_W;
    bool changed = sc[(size_t)r * SC_W + SC_CHANGED] != 0;
    int64_t ix = row[UNIT_COL_IMP_X], iy = row[UNIT_COL_IMP_Y], il = row[UNIT_COL_IMP_L];
    bool any = false, bad = false;
    for (int s = 0; s < n_units && !bad; ++s) {
        const int64_t* q = sc + (size_t)s * SC_W;
        if (q[SC_TARGET] != r) continue;
        any = true;
        bad = add_ovf(ix, q[SC_JX], &ix) || add_ovf(iy, q[SC_JY], &iy) ||
              add_ovf(il, q[SC_DL], &il);
    }
    if (bad) {
        count_fault(cnt);
        return;
    }
    if (any) {
        put(row, UNIT_COL_IMP_X, ix, changed);
        put(row, UNIT_COL_IMP_Y, iy, changed);
        put(row, UNIT_COL_IMP_L, il, changed);
    }
    if (changed) count_changed(cnt);
}

// Kernel 3 (after the gather): ONE thread per env walks the env's requests in
// list order and decides each plant against the charge table's capacity
// (charges.place: the live charges, those placed this slot included); an
// accepted plant spends its planter's charge.
__global__ void k_plant_accept(int64_t* store, int n_units, const int64_t* tables,
                               const int64_t* live, RecordListView rq, unsigned long long* cnt) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = (int)benv();
    const int64_t* tab = env_tables(tables);
    int64_t n_live = live[e];
    const int64_t cap = tab[UNIT_TAB_CHARGE_CAPACITY];
    int64_t* recs = rq.recs + (size_t)e * rq.cap * rq.width;
    const int64_t n = rq.count[e];
    for (int64_t i = 0; i < n; ++i) {
        int64_t* rec = recs + i * rq.width;
        if (rec[UNIT_RQ_KIND] != UNIT_RQKIND_PLANT) continue;
        if (n_live >= cap) {
            rec[UNIT_RQ_ACCEPTED] = 0;
            count_refused(cnt);
            continue;
        }
        rec[UNIT_RQ_ACCEPTED] = 1;
        ++n_live;
        for (int r = 0; r < n_units; ++r) {
            int64_t* row = unit_row(store, n_units, r);
            if (row_valid(row) && row[UNIT_COL_ID] == rec[UNIT_RQ_UNIT_ID]) {
                row[UNIT_COL_HAS_EXPLOSIVE] -= 1;
                break;
            }
        }
    }
}

}  // namespace

namespace {

void check_verbs_args(UnitStoreDevice& store, const UnitTablesDevice& tables,
                      UnitRecordList& requests, std::uintptr_t tick_row, std::uintptr_t flight,
                      int fh, int fw) {
    unit_check_pair(store, tables, "unit_verbs");
    if (requests.n_env() != store.n_env() || requests.n_units() != store.n_units() ||
        requests.width() != UNIT_REQ_F || requests.row_cap() != UNIT_REQ_ROW_CAP) {
        std::ostringstream os;
        os << "unit_verbs: the request list holds (" << requests.n_env() << " envs, "
           << requests.n_units() << " rows, width " << requests.width() << ", row cap "
           << requests.row_cap() << "), the store (" << store.n_env() << ", " << store.n_units()
           << ") and the layout (width " << UNIT_REQ_F << ", row cap " << UNIT_REQ_ROW_CAP << ")";
        throw std::invalid_argument(os.str());
    }
    if (tick_row == 0)
        throw std::invalid_argument("unit_verbs: the tick row");
    if (flight == 0 || fh <= 0 || fw <= 0)
        throw std::invalid_argument("unit_verbs: the flight plane (N, h, w)");
}

// The three kernels of slot 1b on device inputs (the counts begun by the
// caller): the per-row verbs, the shove's push, the gather, the plant pass.
void verbs_launch(UnitStoreDevice& store, const UnitTablesDevice& tables,
                  UnitRecordList& requests, std::uintptr_t tick_row, const int64_t* d_spans,
                  int n_spans, const int64_t* d_live, std::uintptr_t flight, int fh, int fw,
                  const UnitDamageCounts& out) {
    const int n = store.n_env(), U = store.n_units();
    const int64_t* d_trow = reinterpret_cast<const int64_t*>(tick_row);
    int64_t* sc = g_scratch.get((size_t)n * U * SC_W);
    const RecordListView v = list_view(requests);
    k_verbs<<<bgrid(U, n), 128, 0, tstream()>>>(store.rows(), U, tables.values(), d_trow,
                                                 d_spans, n_spans, v, sc,
                                                 reinterpret_cast<const uint8_t*>(flight), fh, fw,
                                                 store.counts_dev());
    k_verbs_push<<<bgrid(U, n), 128, 0, tstream()>>>(store.rows(), U, sc, store.counts_dev());
    requests.gather();
    k_plant_accept<<<dim3(1, n), 1, 0, tstream()>>>(store.rows(), U, tables.values(), d_live, v,
                                                    store.counts_dev());
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in unit_verbs: " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
    store.end_counts(out.coverage, out.faults, out.refused);
    if (out.dropped) requests.download(nullptr, nullptr, out.dropped);
}

// #29 P7e.4: slot 1b's two derived inputs built ON THE DEVICE from the
// tables (unit_requests.door_spans / charges_live): ONE thread per env walks
// the door table in row (= sim._doors) order and writes each LIVE door's span
// tiles (row index, ty, tx) in span order, the rest of the n_spans rows pads
// (index -1); and counts the charge table's live rows. A door whose span
// leaves the tile table, or more span tiles than n_spans, FAULTS the env (its
// spans are then all pads).
__global__ void k_verbs_inputs(const int64_t* doors, const int64_t* door_n, int door_cap,
                               const int64_t* tiles, const int64_t* tile_n, int tile_cap,
                               const int64_t* charges, const int64_t* charge_n, int charge_cap,
                               int n_spans, int64_t* spans, int64_t* live,
                               unsigned long long* cnt) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = (int)benv();
    const int64_t* dr = doors + (size_t)e * door_cap * OBJ_DOOR_W;
    const int64_t* tr = tiles + (size_t)e * tile_cap * OBJ_DOOR_TILE_W;
    const int64_t* cr = charges + (size_t)e * charge_cap * OBJ_CHARGE_W;
    int64_t* sp = spans + (size_t)e * n_spans * SPAN_W;
    const int64_t nd = door_n[e], nt = tile_n[e], nc = charge_n[e];
    int k = 0;
    bool bad = nd < 0 || nd > door_cap || nt < 0 || nt > tile_cap || nc < 0 || nc > charge_cap;
    for (int64_t d = 0; !bad && d < nd; ++d) {
        const int64_t* row = dr + d * OBJ_DOOR_W;
        if (!row[OBJ_DOOR_COL_ALIVE]) continue;
        const int64_t a = row[OBJ_DOOR_COL_TILE_START], c = row[OBJ_DOOR_COL_TILE_COUNT];
        if (a < 0 || c < 0 || a + c > nt || k + c > n_spans) {
            bad = true;
            break;
        }
        for (int64_t t = a; t < a + c; ++t, ++k) {
            sp[(size_t)k * SPAN_W + 0] = d;
            sp[(size_t)k * SPAN_W + 1] = tr[t * OBJ_DOOR_TILE_W + OBJ_DOOR_TILE_COL_TY];
            sp[(size_t)k * SPAN_W + 2] = tr[t * OBJ_DOOR_TILE_W + OBJ_DOOR_TILE_COL_TX];
        }
    }
    if (bad) {
        k = 0;
        count_fault(cnt);
    }
    for (; k < n_spans; ++k) {
        sp[(size_t)k * SPAN_W + 0] = -1;
        sp[(size_t)k * SPAN_W + 1] = 0;
        sp[(size_t)k * SPAN_W + 2] = 0;
    }
    int64_t n_live = 0;
    for (int64_t i = 0; !bad && i < nc; ++i)
        if (cr[i * OBJ_CHARGE_W + OBJ_CHARGE_COL_LIVE]) ++n_live;
    live[e] = n_live;
}

}  // namespace

void unit_verbs(UnitStoreDevice& store, const UnitTablesDevice& tables, UnitRecordList& requests,
                std::uintptr_t tick_row, const int64_t* spans, int n_spans, const int64_t* live,
                std::uintptr_t flight, int fh, int fw, const UnitDamageCounts& out) {
    check_verbs_args(store, tables, requests, tick_row, flight, fh, fw);
    if (spans == nullptr || live == nullptr || n_spans < 1)
        throw std::invalid_argument("unit_verbs: spans (N, n >= 1, 3), live (N,)");
    const int n = store.n_env();
    const int64_t* d_spans = store.stage_input(1, spans, (size_t)n * n_spans * SPAN_W);
    const int64_t* d_live = store.stage_input(2, live, (size_t)n);
    store.begin_counts();
    verbs_launch(store, tables, requests, tick_row, d_spans, n_spans, d_live, flight, fh, fw,
                 out);
}

void unit_verbs_tables(UnitStoreDevice& store, const UnitTablesDevice& tables,
                       UnitRecordList& requests, std::uintptr_t tick_row, ObjectTable& doors,
                       ObjectTable& door_tiles, ObjectTable& charges, std::uintptr_t flight,
                       int fh, int fw, const UnitDamageCounts& out) {
    check_verbs_args(store, tables, requests, tick_row, flight, fh, fw);
    const int n = store.n_env();
    if (doors.n_env() != n || door_tiles.n_env() != n || charges.n_env() != n ||
        doors.width() != OBJ_DOOR_W || door_tiles.width() != OBJ_DOOR_TILE_W ||
        charges.width() != OBJ_CHARGE_W)
        throw std::invalid_argument("unit_verbs_tables: the door, door tile and charge tables "
                                    "of the store's envs");
    const int n_spans = door_tiles.cap();
    int64_t* buf = g_inputs.get((size_t)n * n_spans * SPAN_W + (size_t)n);
    int64_t* d_spans = buf;
    int64_t* d_live = buf + (size_t)n * n_spans * SPAN_W;
    store.begin_counts();
    k_verbs_inputs<<<dim3(1, n), 1, 0, tstream()>>>(
        doors.records_dev(), doors.count_dev(), doors.cap(), door_tiles.records_dev(),
        door_tiles.count_dev(), door_tiles.cap(), charges.records_dev(), charges.count_dev(),
        charges.cap(), n_spans, d_spans, d_live, store.counts_dev());
    verbs_launch(store, tables, requests, tick_row, d_spans, n_spans, d_live, flight, fh, fw,
                 out);
}

}  // namespace breach_cuda
