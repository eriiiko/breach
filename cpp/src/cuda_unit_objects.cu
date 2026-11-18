// ============================================================================
// #29 P7b -- THE GRENADES AND THE PLANTED CHARGES on their device object
// tables (residency design 4.4c, the cut table's P7b row; decision 3: what
// flies and what is placed are tables). See cuda_unit_kernels.h; the tables
// are cuda_object_table.h's (GENERATED object_tables_layout.h: the grenade
// table `OBJ_GRENADE_*`, the charge table `OBJ_CHARGE_*`).
//
// Integer for integer the reference (src/simulation/simulation.py
// `_update_grenades`, combat.Projectile, src/simulation/charges.py):
//
//   unit_table_appends  slot 1b's THROW and PLANT requests (gathered in id
//       order, the plants decided by k_plant_accept) become rows: ONE thread
//       per env walks the request list in list order; a THROW appends a
//       grenade row (Simulation._release_grenade's Projectile: the hand
//       (#215 combat.release_point) as the Q16 start, the Q16 target, the clamped fuse, the thrown
//       tick, the kind's round and its travel speed), an ACCEPTED plant a
//       charge row (charges.place's PlantedCharge). Each table's order is
//       the host list's: survivors, then this tick's appends.
//   unit_grenades  slot 2 "2 grenades": the FLIGHT per row, one thread per
//       table row (#215: Projectile.fly -- the current leg's position, the
//       piece flown walked over the env's solid plane AS THE SLOT FOUND IT
//       (the host flies every grenade before any detonates), a bounce; on a
//       fuse-out Projectile.settle: order-free, wide); a row whose fuse ran
//       out (tick >= thrown + fuse) is marked dead (`detonated`) and, when
//       its proj_type is ORDER_GRENADE, flagged; then ONE thread per env walks
//       the table in ROW ORDER and appends each flagged row's PAYLOAD record
//       (execute_payload's call: combat.detonation_tile of where it rests, the
//       round's payload row, the owner, the current leg's start tile as the
//       approach, kind "grenade") to the effect list -- detonations are
//       serial and in list order, RECORDS walked by the DETONATE stage.
//   unit_charges_tick  slot 9e(d') `charges.tick`: ONE thread per env walks
//       the charge table in row order; a live charge whose det_tick has come
//       is marked dead and its PAYLOAD record appended (kind
//       "door_explosive"); then the table's stable compaction (the prune).
//   The slot-10 prune of the grenades (ContinuousRealtime.on_tick_end) is the
//   grenade table's stable compaction (ObjectTable::compact), no kernel here.
//
// Proved ranges (past them the row FAULTS and is left as it was, nothing
// appended for it): a Q16 coordinate within +-2^25 (512 tiles; a bounced
// leg's target past it faults), a bounce count and bounce_max in 0 .. 64, a
// restitution in 0 .. 1 (Q16), a travel
// speed in 0 .. 2^24, the elapsed ticks in 0 .. 2^31, a tick, a fuse and a
// det_tick within +-2^40, a request's tile within +-2^14, a kind or an ammo
// row inside its table, a round with a payload row. REFUSED (counted apart,
// the Python face names it): a grenade whose fuse is NO_TIMER -- the WEGO
// order path's float `fuse_seconds`, out of the continuous scope.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_object_table.h"
#include "cuda_unit_lib.cuh"
#include "applier_layout.h"         // GENERATED: the effect record, PAYLOAD's fields
#include "object_tables_layout.h"   // GENERATED: the grenade and charge rows
#include "unit_requests_layout.h"   // GENERATED: the request records

#include <vector>

namespace breach_cuda {

namespace {

using namespace unit_lib;
using namespace object_tables;
using namespace unit_requests;

constexpr int64_t COORD_MAX = (int64_t)1 << 25;
constexpr int64_t SPEED_MAX = (int64_t)1 << 24;
constexpr int64_t ELAPSED_MAX = (int64_t)1 << 31;
constexpr int64_t TIME_MAX = (int64_t)1 << 40;
constexpr int64_t TILE_MAX = (int64_t)1 << 14;
constexpr int64_t BOUNCE_CAP = 64;         // unit_tables' grenade_bounce_max range
constexpr int64_t LEG_MAX = COORD_MAX;     // a bounced leg's target, as a row's
constexpr int64_t K_NO_TIMER = -1;         // unit_schema.NO_TIMER
constexpr int64_t K_NO_TILE = -1;          // unit_schema.NO_TILE
//: The per-env counts: rows changed, faults, refused, effect records dropped.
constexpr int C_CHANGED = 0, C_FAULTS = 1, C_REFUSED = 2, C_DROPPED = 3, C_W = 4;

// A grown, never shrunk device buffer (no allocation in a steady state).
struct GrowBuf {
    int64_t* d = nullptr;
    size_t cap = 0;
    int64_t* get(size_t n) {
        if (n > cap) {
            cudaStreamSynchronize(tstream());
            if (d) cudaFree(d);
            d = nullptr;
            if (cudaMalloc(reinterpret_cast<void**>(&d), n * sizeof(int64_t)) != cudaSuccess)
                throw std::runtime_error("unit_objects: scratch allocation failed");
            cap = n;
        }
        return d;
    }
};
GrowBuf g_flags, g_counts, g_ticks;

inline void ob_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in unit_objects/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

__device__ __forceinline__ bool within(int64_t v, int64_t m) { return v >= -m && v <= m; }

// An env's list (a table or the effect list) as a serial record sink.
__device__ __forceinline__ RecordSink env_list_sink(const RecordListView& v, int e) {
    return RecordSink{v.recs + (size_t)e * v.cap * v.width, v.cap, v.count + e, v.dropped + e,
                      v.width};
}

__device__ __forceinline__ void bump(unsigned long long* c, int e, int k) {
    atomicAdd(c + (size_t)e * C_W + k, 1ULL);
}

// The payload row of an [ammo.*] row (payload_for_ammo); -1 when the row is
// outside the table or carries none (the host would raise).
__device__ __forceinline__ int64_t payload_of(const int64_t* tab, int64_t ammo) {
    if (ammo < 0 || ammo >= UNIT_TAB_AMMO_PAYLOAD_ROW_W) return -1;
    return tab[UNIT_TAB_AMMO_PAYLOAD_ROW + ammo];
}

// PAYLOAD record: execute_payload(gmap, queue, units, y, x, payload, ...,
// kind=, owner_id=, approach=) as applier.record_of makes it.
__device__ inline void payload_record(int64_t* rec, int64_t y, int64_t x, int64_t payload,
                                      int64_t owner, bool has_approach, int64_t ay, int64_t ax,
                                      int64_t kind) {
    for (int f = 0; f < applier::APPLIER_F; ++f) rec[f] = 0;
    rec[applier::APPLIER_FX_KIND] = applier::APPLIER_KIND_PAYLOAD;
    rec[applier::APPLIER_PAYLOAD_Y] = y;
    rec[applier::APPLIER_PAYLOAD_X] = x;
    rec[applier::APPLIER_PAYLOAD_PAYLOAD] = payload;
    rec[applier::APPLIER_PAYLOAD_OWNER] = owner;
    rec[applier::APPLIER_PAYLOAD_APPROACH_Y] = has_approach ? ay : -1;
    rec[applier::APPLIER_PAYLOAD_APPROACH_X] = has_approach ? ax : -1;
    rec[applier::APPLIER_PAYLOAD_HAS_APPROACH] = has_approach ? 1 : 0;
    rec[applier::APPLIER_PAYLOAD_KIND] = kind;
}

// ---- slot 1b's appends ------------------------------------------------------
__global__ void k_table_appends(const int64_t* tables, RecordListView rq, RecordListView gr,
                                RecordListView ch, unsigned long long* cnt) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = (int)benv();
    const int64_t* tab = env_tables(tables);
    const int64_t* recs = rq.recs + (size_t)e * rq.cap * rq.width;
    const int64_t n = rq.count[e];
    RecordSink gs = env_list_sink(gr, e), cs = env_list_sink(ch, e);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t* r = recs + i * rq.width;
        if (r[UNIT_RQ_KIND] == UNIT_RQKIND_THROW) {
            const int64_t kind = r[UNIT_RQ_ITEM];
            if (kind < 0 || kind >= UNIT_TAB_HAND_GRENADE_AMMO_ROW_W ||
                !within(r[UNIT_RQ_START_X_Q], COORD_MAX) || !within(r[UNIT_RQ_START_Y_Q], COORD_MAX) ||
                !within(r[UNIT_RQ_TICK], TIME_MAX) || !within(r[UNIT_RQ_TICKS], TIME_MAX) ||
                r[UNIT_RQ_TICKS] == K_NO_TIMER) {
                bump(cnt, e, C_FAULTS);
                continue;
            }
            int64_t g[OBJ_GRENADE_W];
            for (int f = 0; f < OBJ_GRENADE_W; ++f) g[f] = 0;
            const int64_t sx = r[UNIT_RQ_START_X_Q], sy = r[UNIT_RQ_START_Y_Q];   // #215: the hand
            g[OBJ_GRENADE_COL_PROJ_TYPE] = tab[UNIT_TAB_ORDER_GRENADE];
            g[OBJ_GRENADE_COL_OWNER_ID] = r[UNIT_RQ_UNIT_ID];
            g[OBJ_GRENADE_COL_START_X_Q] = sx;
            g[OBJ_GRENADE_COL_START_Y_Q] = sy;
            g[OBJ_GRENADE_COL_TARGET_X_Q] = r[UNIT_RQ_TARGET_X_Q];
            g[OBJ_GRENADE_COL_TARGET_Y_Q] = r[UNIT_RQ_TARGET_Y_Q];
            g[OBJ_GRENADE_COL_POS_X_Q] = sx;
            g[OBJ_GRENADE_COL_POS_Y_Q] = sy;
            g[OBJ_GRENADE_COL_FUSE_SECONDS] = 0;           // the continuous throw's 0.0
            g[OBJ_GRENADE_COL_FUSE_TICKS] = r[UNIT_RQ_TICKS];
            g[OBJ_GRENADE_COL_THROWN_TICK] = r[UNIT_RQ_TICK];
            g[OBJ_GRENADE_COL_AMMO_NAME] = tab[UNIT_TAB_HAND_GRENADE_AMMO_ROW + kind];
            g[OBJ_GRENADE_COL_TRAVEL_SPEED_Q] = tab[UNIT_TAB_HAND_GRENADE_SPEED_Q + kind];
            g[OBJ_GRENADE_COL_LEG_TICK] = r[UNIT_RQ_TICK];       // #215: the throw is leg 1
            g[OBJ_GRENADE_COL_BOUNCES] = 0;
            g[OBJ_GRENADE_LIVE] = 1;
            emit_record(gs, g);
            bump(cnt, e, C_CHANGED);
        } else if (r[UNIT_RQ_KIND] == UNIT_RQKIND_PLANT && r[UNIT_RQ_ACCEPTED]) {
            if (!within(r[UNIT_RQ_TICKS], TIME_MAX) || r[UNIT_RQ_TICKS] == K_NO_TIMER) {
                bump(cnt, e, C_FAULTS);
                continue;
            }
            int64_t c[OBJ_CHARGE_W];
            for (int f = 0; f < OBJ_CHARGE_W; ++f) c[f] = 0;
            c[OBJ_CHARGE_COL_X] = r[UNIT_RQ_TILE_X];
            c[OBJ_CHARGE_COL_Y] = r[UNIT_RQ_TILE_Y];
            c[OBJ_CHARGE_COL_DET_TICK] = r[UNIT_RQ_TICKS];
            c[OBJ_CHARGE_COL_AMMO_NAME] = r[UNIT_RQ_ITEM];
            c[OBJ_CHARGE_COL_OWNER_ID] = r[UNIT_RQ_UNIT_ID];
            c[OBJ_CHARGE_COL_APPROACH] = r[UNIT_RQ_APPROACH_Y];
            c[OBJ_CHARGE_COL_APPROACH + 1] = r[UNIT_RQ_APPROACH_X];
            c[OBJ_CHARGE_LIVE] = 1;
            emit_record(cs, c);
            bump(cnt, e, C_CHANGED);
        }
    }
}

// ---- slot 2: the flight, wide -------------------------------------------------
// #215: THE WALL WALK of a thrown grenade (combat.flight_walk, integer for
// integer; the grid traversal of J. Amanatides and A. Woo, "A Fast Voxel
// Traversal Algorithm for Ray Tracing", Eurographics 1987) over the env's
// flight plane as the slot found it (unit_lib::flight_walk, the device
// library's), and the leg law (Projectile.bounce / .fly / .settle,
// combat.detonation_tile).
// A grenade's moving state (the row's flight columns).
struct Leg {
    int64_t sx, sy, tx, ty, px, py, leg_tick, bounces;
};

__device__ __forceinline__ void leg_rest(Leg& g, int64_t px, int64_t py, int64_t tick) {
    g.sx = g.tx = g.px = px;
    g.sy = g.ty = g.py = py;
    g.leg_tick = tick;
}

// Projectile.bounce
__device__ void leg_bounce(Leg& g, const Hit& h, int64_t tick, int64_t rq, int64_t bmax,
                           int64_t min_q) {
    if (g.bounces >= bmax) {
        leg_rest(g, h.px, h.py, tick);
        return;
    }
    const int64_t ldx = g.tx - g.sx, ldy = g.ty - g.sy;
    const int64_t ldist = fixedpoint::isqrt_q(ldx * ldx + ldy * ldy);
    const int64_t rx = g.tx - h.px, ry = g.ty - h.py;
    const int64_t length = fixedpoint::mul_q16_round_q(fixedpoint::isqrt_q(rx * rx + ry * ry), rq);
    if (length < min_q || ldist <= 0) {
        leg_rest(g, h.px, h.py, tick);
        return;
    }
    const int64_t mx = h.flip_x ? -ldx : ldx, my = h.flip_y ? -ldy : ldy;
    g.sx = h.px;
    g.sy = h.py;
    g.tx = h.px + fixedpoint::div_round_q(mx * length, ldist);
    g.ty = h.py + fixedpoint::div_round_q(my * length, ldist);
    g.px = h.px;
    g.py = h.py;
    g.leg_tick = tick;
    g.bounces += 1;
}

// Projectile.update_position on the current leg
__device__ void leg_position(Leg& g, int64_t tick, int64_t speed, int64_t tps, int64_t eps) {
    const int64_t elapsed = tick - g.leg_tick;
    const int64_t dx = g.tx - g.sx, dy = g.ty - g.sy;
    const int64_t dist = fixedpoint::isqrt_q(dx * dx + dy * dy);
    if (dist < eps || elapsed * speed >= dist * tps) {
        g.px = g.tx;
        g.py = g.ty;
    } else {
        const int64_t num = elapsed * speed;
        const int64_t den = tps * dist;
        g.px = g.sx + fixedpoint::div_round_q(dx * num, den);
        g.py = g.sy + fixedpoint::div_round_q(dy * num, den);
    }
}

// combat.detonation_tile: the resting point's tile, else the first free
// 4-neighbour (north, south, west, east), else the tile itself.
__device__ void detonation_tile(const Plane& p, int64_t px, int64_t py, int64_t* fy,
                                int64_t* fx) {
    const int64_t y = fixedpoint::shr_trunc_q(py, 16), x = fixedpoint::shr_trunc_q(px, 16);
    *fy = y;
    *fx = x;
    if (!flight_blocked(p, y, x)) return;
    const int64_t ny[4] = {y - 1, y + 1, y, y}, nx[4] = {x, x, x - 1, x + 1};
    for (int k = 0; k < 4; ++k) {
        if (!flight_blocked(p, ny[k], nx[k])) {
            *fy = ny[k];
            *fx = nx[k];
            return;
        }
    }
}

__device__ __forceinline__ Plane env_plane(const uint8_t* solid, int h, int w, int e) {
    return Plane{solid + (size_t)e * h * w, h, w};
}

// flags[e, i]: 1 the row fused out and detonates (a PAYLOAD record), 0 not.
__global__ void k_grenade_flight(const int64_t* tables, RecordListView gr, const int64_t* ticks,
                                 const uint8_t* solid, int h, int w, int64_t* flags,
                                 unsigned long long* cnt) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int e = (int)benv();
    if (i >= gr.cap) return;
    int64_t* flag = flags + (size_t)e * gr.cap + i;
    *flag = 0;
    if (i >= gr.count[e]) return;
    int64_t* row = gr.recs + ((size_t)e * gr.cap + i) * gr.width;
    if (row[OBJ_GRENADE_LIVE] == 0) return;                 // `if proj.detonated: continue`
    const int64_t* tab = env_tables(tables);
    const int64_t tick = ticks[e];
    const int64_t fuse = row[OBJ_GRENADE_COL_FUSE_TICKS];
    if (fuse == K_NO_TIMER) {                                 // the WEGO order path's float fuse
        bump(cnt, e, C_REFUSED);
        return;
    }
    Leg g{row[OBJ_GRENADE_COL_START_X_Q], row[OBJ_GRENADE_COL_START_Y_Q],
          row[OBJ_GRENADE_COL_TARGET_X_Q], row[OBJ_GRENADE_COL_TARGET_Y_Q],
          row[OBJ_GRENADE_COL_POS_X_Q],   row[OBJ_GRENADE_COL_POS_Y_Q],
          row[OBJ_GRENADE_COL_LEG_TICK],  row[OBJ_GRENADE_COL_BOUNCES]};
    const int64_t speed = row[OBJ_GRENADE_COL_TRAVEL_SPEED_Q];
    const int64_t thrown = row[OBJ_GRENADE_COL_THROWN_TICK];
    const int64_t tps = tab[UNIT_TAB_CLOCK_TPS];
    const int64_t rq = tab[UNIT_TAB_GRENADE_BOUNCE_RESTITUTION_Q];
    const int64_t bmax = tab[UNIT_TAB_GRENADE_BOUNCE_MAX];
    const int64_t min_q = tab[UNIT_TAB_GRENADE_BOUNCE_MIN_Q];
    const int64_t eps = tab[UNIT_TAB_GRENADE_FLIGHT_EPS_Q];
    const int64_t elapsed = tick - g.leg_tick;
    if (!within(g.sx, COORD_MAX) || !within(g.sy, COORD_MAX) || !within(g.tx, COORD_MAX) ||
        !within(g.ty, COORD_MAX) || !within(g.px, COORD_MAX) || !within(g.py, COORD_MAX) ||
        speed < 0 || speed > SPEED_MAX || !within(tick, TIME_MAX) ||
        !within(thrown, TIME_MAX) || !within(fuse, TIME_MAX) || !within(g.leg_tick, TIME_MAX) ||
        elapsed < 0 || elapsed > ELAPSED_MAX || g.bounces < 0 || g.bounces > BOUNCE_CAP ||
        rq < 0 || rq > ONE || bmax < 0 || bmax > BOUNCE_CAP) {
        bump(cnt, e, C_FAULTS);
        return;
    }
    // the fuse-out (decided before any write: a row is all or nothing)
    const bool out = tick >= thrown + fuse;
    const bool lob = row[OBJ_GRENADE_COL_PROJ_TYPE] == tab[UNIT_TAB_ORDER_GRENADE];
    if (out && lob && payload_of(tab, row[OBJ_GRENADE_COL_AMMO_NAME]) < 0) {
        bump(cnt, e, C_FAULTS);
        return;
    }
    const Plane pl = env_plane(solid, h, w, e);
    // Projectile.fly: the leg's position, the piece flown walked, a bounce
    const int64_t ox = g.px, oy = g.py;
    leg_position(g, tick, speed, tps, eps);
    Hit hit;
    if (flight_walk(pl, ox, oy, g.px, g.py, &hit)) leg_bounce(g, hit, tick, rq, bmax, min_q);
    // Projectile.settle: a fuse-out's rest of the flight at once
    if (out) {
        for (int64_t k = 0; k < bmax + 2; ++k) {
            if (g.px == g.tx && g.py == g.ty) break;
            if (!flight_walk(pl, g.px, g.py, g.tx, g.ty, &hit)) {
                g.px = g.tx;
                g.py = g.ty;
                break;
            }
            leg_bounce(g, hit, tick, rq, bmax, min_q);
        }
    }
    if (!within(g.tx, LEG_MAX) || !within(g.ty, LEG_MAX)) {   // a bounced leg past the range
        bump(cnt, e, C_FAULTS);
        return;
    }
    bool changed = false;
    const int64_t cols[8] = {OBJ_GRENADE_COL_START_X_Q, OBJ_GRENADE_COL_START_Y_Q,
                             OBJ_GRENADE_COL_TARGET_X_Q, OBJ_GRENADE_COL_TARGET_Y_Q,
                             OBJ_GRENADE_COL_POS_X_Q,   OBJ_GRENADE_COL_POS_Y_Q,
                             OBJ_GRENADE_COL_LEG_TICK,  OBJ_GRENADE_COL_BOUNCES};
    const int64_t vals[8] = {g.sx, g.sy, g.tx, g.ty, g.px, g.py, g.leg_tick, g.bounces};
    for (int k = 0; k < 8; ++k) {
        if (row[cols[k]] != vals[k]) {
            row[cols[k]] = vals[k];
            changed = true;
        }
    }
    if (out) {
        row[OBJ_GRENADE_LIVE] = 0;                          // proj.detonated = True
        changed = true;
        if (lob) *flag = 1;
    }
    if (changed) bump(cnt, e, C_CHANGED);
}

// ---- slot 2: the detonations, serial in row order -----------------------------
// The tile is where the grenade rests (combat.detonation_tile over the plane
// the flight read: no write lands on it before the DETONATE stage), the
// approach the current leg's start (Projectile.approach_tile).
__global__ void k_grenade_fuse(const int64_t* tables, RecordListView gr, const int64_t* flags,
                               const uint8_t* solid, int h, int w, RecordListView fx) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = (int)benv();
    const int64_t* tab = env_tables(tables);
    const Plane pl = env_plane(solid, h, w, e);
    RecordSink s = env_list_sink(fx, e);
    const int64_t n = gr.count[e];
    for (int64_t i = 0; i < n; ++i) {
        if (flags[(size_t)e * gr.cap + i] != 1) continue;
        const int64_t* row = gr.recs + ((size_t)e * gr.cap + i) * gr.width;
        int64_t ty, tx;
        detonation_tile(pl, row[OBJ_GRENADE_COL_POS_X_Q], row[OBJ_GRENADE_COL_POS_Y_Q], &ty, &tx);
        int64_t rec[applier::APPLIER_F];
        payload_record(rec, ty, tx, payload_of(tab, row[OBJ_GRENADE_COL_AMMO_NAME]),
                       row[OBJ_GRENADE_COL_OWNER_ID], true,
                       fixedpoint::shr_trunc_q(row[OBJ_GRENADE_COL_START_Y_Q], 16),
                       fixedpoint::shr_trunc_q(row[OBJ_GRENADE_COL_START_X_Q], 16),
                       applier::APPLIER_PAYLOAD_KIND_GRENADE);
        emit_record(s, rec);
    }
}

// ---- slot 9e(d'): charges.tick, serial in row order ---------------------------
__global__ void k_charges_tick(const int64_t* tables, RecordListView ch, const int64_t* ticks,
                               RecordListView fx, unsigned long long* cnt) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = (int)benv();
    const int64_t* tab = env_tables(tables);
    const int64_t tick = ticks[e];
    RecordSink s = env_list_sink(fx, e);
    const int64_t n = ch.count[e];
    for (int64_t i = 0; i < n; ++i) {
        int64_t* row = ch.recs + ((size_t)e * ch.cap + i) * ch.width;
        const int64_t det = row[OBJ_CHARGE_COL_DET_TICK];
        // PlantedCharge.due: live, a timer, its moment come
        if (row[OBJ_CHARGE_LIVE] == 0 || det == K_NO_TIMER || tick < det) continue;
        const int64_t payload = payload_of(tab, row[OBJ_CHARGE_COL_AMMO_NAME]);
        if (payload < 0) {
            bump(cnt, e, C_FAULTS);
            continue;
        }
        row[OBJ_CHARGE_LIVE] = 0;                           // charges.fire: live = False
        const int64_t ay = row[OBJ_CHARGE_COL_APPROACH], ax = row[OBJ_CHARGE_COL_APPROACH + 1];
        const bool has = !(ay == K_NO_TILE && ax == K_NO_TILE);  // _dec_approach's None
        int64_t rec[applier::APPLIER_F];
        payload_record(rec, row[OBJ_CHARGE_COL_Y], row[OBJ_CHARGE_COL_X], payload,
                       row[OBJ_CHARGE_COL_OWNER_ID], has, ay, ax,
                       applier::APPLIER_PAYLOAD_KIND_DOOR_EXPLOSIVE);
        emit_record(s, rec);
        bump(cnt, e, C_CHANGED);
    }
}

void check_table(const ObjectTable& t, int n_env, int width, int live, const char* who,
                 const char* name) {
    if (t.n_env() != n_env || t.width() != width || t.live_col() != live) {
        std::ostringstream os;
        os << who << ": the " << name << " table holds (" << t.n_env() << " envs, width "
           << t.width() << ", live " << t.live_col() << "), want (" << n_env << ", " << width
           << ", " << live << ")";
        throw std::invalid_argument(os.str());
    }
}

void check_effects(UnitRecordList& fx, int n_env, const char* who) {
    if (fx.n_env() != n_env || fx.width() != applier::APPLIER_F) {
        std::ostringstream os;
        os << who << ": the effect list holds (" << fx.n_env() << " envs, width " << fx.width()
           << "), want (" << n_env << ", " << applier::APPLIER_F << ")";
        throw std::invalid_argument(os.str());
    }
}

unsigned long long* begin_counts(int n) {
    int64_t* c = g_counts.get((size_t)n * C_W);
    ob_check(cudaMemsetAsync(c, 0, (size_t)n * C_W * sizeof(int64_t), tstream()), "zero counts");
    return reinterpret_cast<unsigned long long*>(c);
}

const int64_t* stage_ticks(const int64_t* ticks, int n) {
    int64_t* d = g_ticks.get((size_t)n);
    ob_check(cudaMemcpyAsync(d, ticks, (size_t)n * sizeof(int64_t), cudaMemcpyHostToDevice,
                             tstream()),
             "H2D ticks");
    return d;
}

void end_counts(int n, const ObjectSlotCounts& out, UnitRecordList* dropped_of) {
    ob_check(cudaGetLastError(), "launch");
    std::vector<int64_t> h((size_t)n * C_W);
    ob_check(cudaMemcpyAsync(h.data(), g_counts.d, h.size() * sizeof(int64_t),
                             cudaMemcpyDeviceToHost, tstream()),
             "D2H counts");
    ob_check(cudaStreamSynchronize(tstream()), "counts sync");
    for (int e = 0; e < n; ++e) {
        if (out.changed) out.changed[e] = h[(size_t)e * C_W + C_CHANGED];
        if (out.faults) out.faults[e] = h[(size_t)e * C_W + C_FAULTS];
        if (out.refused) out.refused[e] = h[(size_t)e * C_W + C_REFUSED];
    }
    if (out.dropped) {
        if (dropped_of)
            dropped_of->download(nullptr, nullptr, out.dropped);
        else
            for (int e = 0; e < n; ++e) out.dropped[e] = 0;
    }
}

}  // namespace

void unit_table_appends(const UnitTablesDevice& tables, UnitRecordList& requests,
                        ObjectTable& grenades, ObjectTable& charges, const ObjectSlotCounts& out) {
    const int n = tables.n_env();
    if (requests.n_env() != n || requests.width() != UNIT_REQ_F)
        throw std::invalid_argument("unit_table_appends: the request list is not the tables' "
                                    "envs at the request width");
    check_table(grenades, n, OBJ_GRENADE_W, OBJ_GRENADE_LIVE, "unit_table_appends", "grenade");
    check_table(charges, n, OBJ_CHARGE_W, OBJ_CHARGE_LIVE, "unit_table_appends", "charge");
    unsigned long long* cnt = begin_counts(n);
    k_table_appends<<<dim3(1, n), 1, 0, tstream()>>>(tables.values(), list_view(requests),
                                                     list_view(grenades), list_view(charges),
                                                     cnt);
    end_counts(n, out, nullptr);     // the tables' overflow: their own counts
}

void unit_grenades(const UnitTablesDevice& tables, ObjectTable& grenades, UnitRecordList& effects,
                   const int64_t* ticks, std::uintptr_t solid, int h, int w,
                   const ObjectSlotCounts& out) {
    const int n = tables.n_env();
    check_table(grenades, n, OBJ_GRENADE_W, OBJ_GRENADE_LIVE, "unit_grenades", "grenade");
    check_effects(effects, n, "unit_grenades");
    if (ticks == nullptr) throw std::invalid_argument("unit_grenades: ticks (N,)");
    unsigned long long* cnt = begin_counts(n);
    const int64_t* d_ticks = stage_ticks(ticks, n);
    if (solid == 0 || h <= 0 || w <= 0)
        throw std::invalid_argument("unit_grenades: the solid plane (N, h, w)");
    const uint8_t* sp = reinterpret_cast<const uint8_t*>(solid);
    int64_t* flags = g_flags.get((size_t)n * grenades.cap());
    const RecordListView gv = list_view(grenades);
    k_grenade_flight<<<dim3((grenades.cap() + 63) / 64, n), 64, 0, tstream()>>>(
        tables.values(), gv, d_ticks, sp, h, w, flags, cnt);
    k_grenade_fuse<<<dim3(1, n), 1, 0, tstream()>>>(tables.values(), gv, flags, sp, h, w,
                                                    list_view(effects));
    end_counts(n, out, &effects);
}

void unit_charges_tick(const UnitTablesDevice& tables, ObjectTable& charges,
                       UnitRecordList& effects, const int64_t* ticks,
                       const ObjectSlotCounts& out) {
    const int n = tables.n_env();
    check_table(charges, n, OBJ_CHARGE_W, OBJ_CHARGE_LIVE, "unit_charges_tick", "charge");
    check_effects(effects, n, "unit_charges_tick");
    if (ticks == nullptr) throw std::invalid_argument("unit_charges_tick: ticks (N,)");
    unsigned long long* cnt = begin_counts(n);
    const int64_t* d_ticks = stage_ticks(ticks, n);
    k_charges_tick<<<dim3(1, n), 1, 0, tstream()>>>(tables.values(), list_view(charges), d_ticks,
                                                    list_view(effects), cnt);
    charges.compact();               // the prune: `[c for c in planted_charges if c.live]`
    end_counts(n, out, &effects);
}

}  // namespace breach_cuda
