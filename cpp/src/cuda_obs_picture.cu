// ==========================================================================
// #29 P8.3: THE DEVICE TEAM PICTURE (see cuda_obs.h, cuda_obs_picture.cuh and
// src/simulation/device_picture.py), Class T. The twin of
// simulation.team_picture.TeamPicture.update, line for line, every env in
// one launch per stage, on THE tick stream; the host picture is the
// reference. Nothing here writes a synced buffer; the fault bits ride the
// conductor's one books read (kind "picture_faults").
//
// THE UPDATE's order (TeamPicture.update):
//   k_picture_units (one thread per env): the tick check; the object serials
//     (rows KNOWN .. count of the grenade table, then of the charge table,
//     numbered from SERIAL_NEXT in row order); then _track_units over the
//     store rows in row order (= sim.units order): bars at the first update a
//     unit exists and again at the first that sees it a zombie, death_tick,
//     motion; then last_hit over the ring's hits with damage > 0, emission
//     order (the last wins);
//   k_picture_teams (one thread per (env, marine team)): the damage window
//     against the records as they stand BEFORE this update touches them
//     (the hits in emission order, then the entries older than the window
//     dropped from the FIFO's head); the seen deaths (a kill whose victim's
//     record was written at tick - 1); the memory (tick - rec.tick >
//     memory_ticks); this tick's sightings (the device vision's visible
//     table, the record from the live row, vel against a record of tick - 1);
//     the object tracker (a dead object forgotten, the sight gate over every
//     (mate centre, object tile) pair, a first sighting draws the team's next
//     id).
// The first update (start_team_picture's) reads no events.
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
#include "cuda_unit_books.h"       // books_defer: the faults ride the one books read
#include "cuda_unit_events.h"
#include "obs_layout.h"
#include "object_tables_layout.h"
#include "unit_draws_layout.h"
#include "unit_events_layout.h"
#include "unit_store_layout.h"

namespace breach_cuda {

using namespace obs_layout;
using obs_pic::PicLayout;
using obs_pic::pic_layout;

static_assert(OBS_PIC_GRENADE_CAP == object_tables::OBJ_GRENADE_CAP, "one grenade capacity");
static_assert(OBS_PIC_CHARGE_CAP == object_tables::OBJ_CHARGE_CAP, "one charge capacity");

namespace {

using namespace unit_store;

void need(bool ok, const char* who, const char* what) {
    if (!ok) throw std::invalid_argument(std::string(who) + ": " + what);
}

void launched(const char* who) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(who) + ": launch failed: " + cudaGetErrorString(err));
}

__device__ __forceinline__ int64_t tick_of(const int64_t* trow, int e) {
    return trow[(size_t)e * unit_draws::TICK_ROW_W + unit_draws::TICK_ROW_TICK];
}

// The store row holding unit id `uid` (-1: none). by_id of the reference.
__device__ __forceinline__ int row_of(const int64_t* rows, int U, int64_t uid) {
    for (int r = 0; r < U; ++r) {
        const int64_t* u = rows + (size_t)r * UNIT_STORE_W;
        if (u[UNIT_COL_VALID] && u[UNIT_COL_ID] == uid) return r;
    }
    return -1;
}

__device__ __forceinline__ bool alive(const int64_t* u) {
    return u[UNIT_COL_LIFE_STATE] == OBS_LIFE_ALIVE;
}

// ---------------------------------------------------------------------------
// Stage 1: the tick check, the serials, the per-unit trackers, last_hit.
// ---------------------------------------------------------------------------
__global__ void k_picture_units(const int64_t* store, int U, const int64_t* ring,
                                const int64_t* ring_count, int ring_cap,
                                const int64_t* g_count, const int64_t* c_count,
                                const int64_t* trow, int T, int64_t* state, int W,
                                int64_t* faults, int first) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = blockIdx.y;
    const PicLayout L = pic_layout(T, U);
    int64_t* blk = state + (size_t)e * W;
    const int64_t* rows = store + (size_t)e * U * UNIT_STORE_W;
    const int64_t tick = tick_of(trow, e);
    int64_t fault = 0;
    // update(): no-op when already there; a skipped / doubled tick is the bug
    const int64_t at = blk[L.tick];
    if (!first && at >= 0 && tick != at + 1) fault |= OBS_PIC_FAULT_SKIPPED;
    if (blk[L.param + OBS_PIC_PARAM_N_TEAMS] > T) fault |= OBS_PIC_FAULT_TEAMS;
    // -- the object serials: new rows are the table's tail (appends only ever
    // land after what it holds; the compaction carried KNOWN with the rows)
    {
        const int64_t gc = g_count[e], cc = c_count[e];
        int64_t kg = blk[L.known_g], kc = blk[L.known_c];
        if (kg > gc || kc > cc) {
            fault |= OBS_PIC_FAULT_SERIAL;
        } else {
            int64_t next = blk[L.serial_next];
            for (int64_t i = kg; i < gc; ++i) blk[L.side_g + i] = next++;
            for (int64_t i = kc; i < cc; ++i) blk[L.side_c + i] = next++;
            blk[L.serial_next] = next;
            blk[L.known_g] = gc;
            blk[L.known_c] = cc;
        }
    }
    // -- _track_units: sim.units order (store row order)
    for (int r = 0; r < U; ++r) {
        const int64_t* u = rows + (size_t)r * UNIT_STORE_W;
        if (!u[UNIT_COL_VALID]) continue;
        int64_t* m = blk + L.unit + (int64_t)r * OBS_PIC_UNIT_W;
        if (!m[OBS_PIC_UNIT_BAR_SET]) {                       // exists: its spawn
            m[OBS_PIC_UNIT_BAR_SET] = 1;
            m[OBS_PIC_UNIT_BAR_Q] = u[UNIT_COL_CURRENT_HP];
        }
        if (u[UNIT_COL_IS_ZOMBIE] && !m[OBS_PIC_UNIT_TURNED]) {
            m[OBS_PIC_UNIT_TURNED] = 1;
            m[OBS_PIC_UNIT_BAR_Q] = u[UNIT_COL_CURRENT_HP];  // at turning
            if (!m[OBS_PIC_UNIT_DEATH_SET] && m[OBS_PIC_UNIT_MOTION_SET]) {
                m[OBS_PIC_UNIT_DEATH_SET] = 1;               // turned (a marine)
                m[OBS_PIC_UNIT_DEATH_TICK] = tick;
            }
        }
        if (!alive(u)) {
            if (!m[OBS_PIC_UNIT_DEATH_SET]) {
                m[OBS_PIC_UNIT_DEATH_SET] = 1;
                m[OBS_PIC_UNIT_DEATH_TICK] = tick;
            }
            continue;
        }
        const int64_t nx = u[UNIT_COL_X_Q], ny = u[UNIT_COL_Y_Q];
        if (m[OBS_PIC_UNIT_MOTION_SET]) {
            m[OBS_PIC_UNIT_PREV_X] = m[OBS_PIC_UNIT_NOW_X];
            m[OBS_PIC_UNIT_PREV_Y] = m[OBS_PIC_UNIT_NOW_Y];
        } else {
            m[OBS_PIC_UNIT_PREV_X] = nx;
            m[OBS_PIC_UNIT_PREV_Y] = ny;
        }
        m[OBS_PIC_UNIT_NOW_X] = nx;
        m[OBS_PIC_UNIT_NOW_Y] = ny;
        m[OBS_PIC_UNIT_MOTION_SET] = 1;
    }
    // -- last_hit: the step's hits with damage > 0, emission order
    if (!first) {
        using namespace unit_events;
        const int64_t n = ring_count[e] < ring_cap ? ring_count[e] : ring_cap;
        for (int64_t k = 0; k < n; ++k) {
            const int64_t* ev = ring + ((size_t)e * ring_cap + k) * UNIT_EVENT_F;
            if (ev[UNIT_EV_KIND] != UNIT_EVKIND_HIT || ev[UNIT_EV_DAMAGE_Q] <= 0) continue;
            const int r = row_of(rows, U, ev[UNIT_EV_UNIT_ID]);
            if (r < 0) {
                fault |= OBS_PIC_FAULT_UNKNOWN_UID;
                continue;
            }
            int64_t* m = blk + L.unit + (int64_t)r * OBS_PIC_UNIT_W;
            m[OBS_PIC_UNIT_HIT_SET] = 1;
            m[OBS_PIC_UNIT_HIT_TICK] = tick;
            m[OBS_PIC_UNIT_HIT_DX] = ev[UNIT_EV_DIR_X_Q];
            m[OBS_PIC_UNIT_HIT_DY] = ev[UNIT_EV_DIR_Y_Q];
            m[OBS_PIC_UNIT_HIT_CAUSE] = ev[UNIT_EV_CAUSE];
        }
    }
    blk[L.tick] = tick;
    if (fault) atomicOr(reinterpret_cast<unsigned long long*>(faults + e),
                        (unsigned long long)fault);
}

// ---------------------------------------------------------------------------
// Stage 2: per (env, marine team): the damage window, the records, the
// object tracker.
// ---------------------------------------------------------------------------
// An object's key in the tracker: its serial (THE BREAK serial_row: the row).
__device__ __forceinline__ int64_t obj_key(const int64_t* blk, const PicLayout& L, int kind,
                                           int64_t i, int brk) {
    if (brk == OBS_PIC_BREAK_SERIAL_ROW) return kind == 0 ? i : OBS_PIC_GRENADE_CAP + i;
    return kind == 0 ? blk[L.side_g + i] : blk[L.side_c + i];
}

__global__ void k_picture_teams(const int64_t* store, int U, const int64_t* ring,
                                const int64_t* ring_count, int ring_cap, const int64_t* grenades,
                                const int64_t* g_count, const int64_t* charges,
                                const int64_t* c_count, const uint8_t* solid,
                                const uint8_t* screen, int h, int w, const int64_t* trow,
                                const uint8_t* visible, int T, int64_t* state, int W,
                                int64_t* faults, int first, int brk) {
    const int e = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const PicLayout L = pic_layout(T, U);
    int64_t* blk = state + (size_t)e * W;
    if (t >= T || t >= blk[L.param + OBS_PIC_PARAM_N_TEAMS]) return;
    const int64_t* rows = store + (size_t)e * U * UNIT_STORE_W;
    const int64_t tick = tick_of(trow, e);
    const int64_t mem = blk[L.param + OBS_PIC_PARAM_MEMORY_TICKS];
    const int64_t win = blk[L.param + OBS_PIC_PARAM_WINDOW_TICKS];
    int64_t fault = 0;
    using namespace unit_events;
    const int64_t n_ev = first ? 0 : (ring_count[e] < ring_cap ? ring_count[e] : ring_cap);
    const int64_t* evs = ring + (size_t)e * ring_cap * UNIT_EVENT_F;
    // -- the damage window (1.6), against the records BEFORE this update
    {
        int64_t* hd = blk + L.dmg_hd + 2 * t;
        int64_t head = hd[0], len = hd[1];
        for (int64_t k0 = 0; k0 < n_ev; ++k0) {
            const int64_t k = (brk == OBS_PIC_BREAK_DMG_ORDER) ? n_ev - 1 - k0 : k0;
            const int64_t* ev = evs + k * UNIT_EVENT_F;
            if (ev[UNIT_EV_KIND] != UNIT_EVKIND_HIT || ev[UNIT_EV_DAMAGE_Q] <= 0) continue;
            const int64_t v = ev[UNIT_EV_UNIT_ID];
            const int ar = row_of(rows, U, ev[UNIT_EV_ATTACKER_ID]);
            bool dealt = false;
            if (ar >= 0) {
                const int64_t* a = rows + (size_t)ar * UNIT_STORE_W;
                dealt = a[UNIT_COL_TEAM] == (int64_t)t && !a[UNIT_COL_IS_ZOMBIE];
            }
            bool seen_prev = false;
            const int vr = row_of(rows, U, v);
            if (vr >= 0) {
                const int64_t* rc = obs_pic::rec_of(blk, L, U, t, vr);
                seen_prev = rc[OBS_PIC_REC_PRESENT] && rc[OBS_PIC_REC_TICK] == tick - 1;
            }
            if (!(dealt || seen_prev)) continue;
            if (len >= OBS_PIC_DMG_CAP) {
                fault |= OBS_PIC_FAULT_DMG_OVERFLOW;
                continue;
            }
            int64_t* d = blk + L.dmg + ((int64_t)t * OBS_PIC_DMG_CAP +
                                        (head + len) % OBS_PIC_DMG_CAP) * OBS_PIC_DMG_W;
            d[OBS_PIC_DMG_VICTIM] = v;
            d[OBS_PIC_DMG_TICK] = tick;
            d[OBS_PIC_DMG_DQ] = ev[UNIT_EV_DAMAGE_Q];
            ++len;
        }
        const int64_t oldest = tick - win;
        while (len > 0) {
            const int64_t* d = blk + L.dmg + ((int64_t)t * OBS_PIC_DMG_CAP + head) * OBS_PIC_DMG_W;
            if (d[OBS_PIC_DMG_TICK] > oldest) break;
            head = (head + 1) % OBS_PIC_DMG_CAP;
            --len;
        }
        if (len == 0) head = 0;
        hd[0] = head;
        hd[1] = len;
    }
    // -- (1) a SEEN death: the victim's record written at tick - 1
    for (int64_t k = 0; k < n_ev; ++k) {
        const int64_t* ev = evs + k * UNIT_EVENT_F;
        if (ev[UNIT_EV_KIND] != UNIT_EVKIND_KILLED) continue;
        const int r = row_of(rows, U, ev[UNIT_EV_UNIT_ID]);
        if (r < 0) continue;
        int64_t* rc = obs_pic::rec_of(blk, L, U, t, r);
        if (rc[OBS_PIC_REC_PRESENT] && rc[OBS_PIC_REC_TICK] == tick - 1)
            for (int f = 0; f < OBS_PIC_REC_W; ++f) rc[f] = 0;
    }
    // -- (2) the memory: kept while tick - rec.tick <= memory_ticks
    for (int r = 0; r < U; ++r) {
        int64_t* rc = obs_pic::rec_of(blk, L, U, t, r);
        if (!rc[OBS_PIC_REC_PRESENT]) continue;
        const int64_t age = tick - rc[OBS_PIC_REC_TICK];
        const bool drop = (brk == OBS_PIC_BREAK_LIFETIME) ? age >= mem : age > mem;
        if (drop)
            for (int f = 0; f < OBS_PIC_REC_W; ++f) rc[f] = 0;
    }
    // -- (3) this tick's sightings (_sighting from the live row)
    const uint8_t* vis = visible + ((size_t)e * (T + 1) + t) * U;
    for (int r = 0; r < U; ++r) {
        if (!vis[r]) continue;
        const int64_t* u = rows + (size_t)r * UNIT_STORE_W;
        int64_t* rc = obs_pic::rec_of(blk, L, U, t, r);
        const int64_t px = u[UNIT_COL_X_Q], py = u[UNIT_COL_Y_Q];
        int64_t vx = 0, vy = 0;
        if (rc[OBS_PIC_REC_PRESENT] && rc[OBS_PIC_REC_TICK] == tick - 1) {
            vx = px - rc[OBS_PIC_REC_POS_X];
            vy = py - rc[OBS_PIC_REC_POS_Y];
        }
        rc[OBS_PIC_REC_PRESENT] = 1;
        rc[OBS_PIC_REC_UID] = u[UNIT_COL_ID];
        rc[OBS_PIC_REC_TEAM] = u[UNIT_COL_TEAM];
        rc[OBS_PIC_REC_ZOMBIE] = u[UNIT_COL_IS_ZOMBIE] ? 1 : 0;
        rc[OBS_PIC_REC_LOADOUT] = u[UNIT_COL_LOADOUT_NAME];
        rc[OBS_PIC_REC_TILE_X] = obs_sight::tile_of(px);
        rc[OBS_PIC_REC_TILE_Y] = obs_sight::tile_of(py);
        rc[OBS_PIC_REC_CENTRE_X] = obs_sight::centre_x(u);
        rc[OBS_PIC_REC_CENTRE_Y] = obs_sight::centre_y(u);
        rc[OBS_PIC_REC_POS_X] = px;
        rc[OBS_PIC_REC_POS_Y] = py;
        rc[OBS_PIC_REC_VEL_X] = vx;
        rc[OBS_PIC_REC_VEL_Y] = vy;
        rc[OBS_PIC_REC_TICK] = tick;
        rc[OBS_PIC_REC_FACING] = u[UNIT_COL_FACING_Q];
    }
    // -- the object tracker (_track_objects)
    {
        using namespace object_tables;
        const int64_t gc = g_count[e], cc = c_count[e];
        const int64_t* grows = grenades + (size_t)e * OBJ_GRENADE_CAP * OBJ_GRENADE_W;
        const int64_t* crows = charges + (size_t)e * OBJ_CHARGE_CAP * OBJ_CHARGE_W;
        int64_t* obj = blk + L.obj + (int64_t)t * OBS_PIC_OBJ_CAP * 2;
        int64_t n_obj = blk[L.obj_n + t];
        // a dead object is forgotten (its key not among the live rows')
        int64_t m = 0;
        for (int64_t k = 0; k < n_obj; ++k) {
            const int64_t key = obj[2 * k];
            bool live = false;
            for (int64_t i = 0; i < gc && !live; ++i)
                live = grows[i * OBJ_GRENADE_W + OBJ_GRENADE_LIVE] != 0 &&
                       obj_key(blk, L, 0, i, brk) == key;
            for (int64_t i = 0; i < cc && !live; ++i)
                live = crows[i * OBJ_CHARGE_W + OBJ_CHARGE_LIVE] != 0 &&
                       obj_key(blk, L, 1, i, brk) == key;
            if (!live) continue;
            obj[2 * m] = key;
            obj[2 * m + 1] = obj[2 * k + 1];
            ++m;
        }
        n_obj = m;
        int64_t* seen = blk + L.seen + (int64_t)t * OBS_PIC_OBJ_CAP * 2;
        int64_t n_seen = 0;
        const uint8_t* so = solid + (size_t)e * h * w;
        const uint8_t* sc = screen + (size_t)e * h * w;
        for (int kind = 0; kind < 2; ++kind) {
            const int64_t cnt = kind == 0 ? gc : cc;
            for (int64_t i = 0; i < cnt; ++i) {
                int64_t ty, tx;
                if (kind == 0) {
                    const int64_t* g = grows + i * OBJ_GRENADE_W;
                    if (g[OBJ_GRENADE_LIVE] == 0) continue;
                    ty = obs_sight::tile_of(g[OBJ_GRENADE_COL_POS_Y_Q]);
                    tx = obs_sight::tile_of(g[OBJ_GRENADE_COL_POS_X_Q]);
                } else {
                    const int64_t* c = crows + i * OBJ_CHARGE_W;
                    if (c[OBJ_CHARGE_LIVE] == 0) continue;
                    ty = c[OBJ_CHARGE_COL_Y];
                    tx = c[OBJ_CHARGE_COL_X];
                }
                // the sight gate: some living non-zombie mate has a line to it
                bool ok = false;
                for (int q = 0; q < U && !ok; ++q) {
                    const int64_t* mu = rows + (size_t)q * UNIT_STORE_W;
                    if (!mu[UNIT_COL_VALID] || !alive(mu) || mu[UNIT_COL_IS_ZOMBIE] ||
                        mu[UNIT_COL_TEAM] != (int64_t)t)
                        continue;
                    bool bad = false;
                    ok = obs_sight::line_clear(so, sc, h, w, false, obs_sight::centre_y(mu),
                                               obs_sight::centre_x(mu), ty, tx, &bad);
                    if (bad) {
                        fault |= OBS_PIC_FAULT_SIGHT;
                        ok = false;
                    }
                }
                if (!ok) continue;
                const int64_t key = obj_key(blk, L, kind, i, brk);
                int64_t uid = -1;
                for (int64_t k = 0; k < n_obj; ++k)
                    if (obj[2 * k] == key) uid = obj[2 * k + 1];
                if (uid < 0) {
                    if (n_obj >= OBS_PIC_OBJ_CAP) {
                        fault |= OBS_PIC_FAULT_OBJ_OVERFLOW;
                        continue;
                    }
                    uid = (int64_t)OBS_OBJECT_UID_BASE + blk[L.obj_next + t];
                    blk[L.obj_next + t] += 1;
                    obj[2 * n_obj] = key;
                    obj[2 * n_obj + 1] = uid;
                    ++n_obj;
                }
                if (n_seen >= OBS_PIC_OBJ_CAP) {
                    fault |= OBS_PIC_FAULT_OBJ_OVERFLOW;
                    continue;
                }
                seen[2 * n_seen] = key;
                seen[2 * n_seen + 1] = uid;
                ++n_seen;
            }
        }
        blk[L.obj_n + t] = n_obj;
        blk[L.seen_n + t] = n_seen;
    }
    if (fault) atomicOr(reinterpret_cast<unsigned long long*>(faults + e),
                        (unsigned long long)fault);
}

// ---------------------------------------------------------------------------
// The vector's P8.3 columns (obs_spec_v1.build_vector, line for line): hp,
// vel_x / vel_y, the five hit_* (1.4), the four record counts (1.5).
// ---------------------------------------------------------------------------
using obs_rows::env_due;   // the observation's one copy (cuda_obs_rows.cuh)

__device__ __forceinline__ void put_v(float* v, int col, double x) { v[col] = obs_f64::to_f32(x); }

__global__ void k_picture_vector(const int64_t* store, int U, const int32_t* rowmap, int n_rows,
                                 int n_slots, int T, const int64_t* otab, const int64_t* teams,
                                 const int64_t* trow, const int64_t* params, const uint8_t* ended,
                                 const uint8_t* visible, const int64_t* state, int W,
                                 float* vector) {
    using namespace obs_f64;
    const int e = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows || !env_due(e, trow, params, ended)) return;
    const size_t er = (size_t)e * n_rows + r;
    float* v = vector + er * OBS_V;
    const int team = r / n_slots;
    const int sr = rowmap[er];
    const int64_t* rows = store + (size_t)e * U * UNIT_STORE_W;
    const int64_t* u = nullptr;
    bool valid = false;
    if (sr >= 0 && sr < U) {
        u = rows + (size_t)sr * UNIT_STORE_W;
        valid = alive(u) && u[UNIT_COL_TEAM] == (int64_t)team;
    }
    const int owned[] = {OBS_V_HP, OBS_V_VEL_X, OBS_V_VEL_Y, OBS_V_HIT_RECENT, OBS_V_HIT_AGE,
                         OBS_V_HIT_FROM_COS, OBS_V_HIT_FROM_SIN, OBS_V_HIT_CAUSE,
                         OBS_V_ENEMIES_KNOWN, OBS_V_ENEMIES_SEEN, OBS_V_ZOMBIES_KNOWN,
                         OBS_V_ZOMBIES_NEAR};
    for (int c : owned) v[c] = 0.0f;
    if (!valid) return;
    const PicLayout L = pic_layout(T, U);
    const int64_t* blk = state + (size_t)e * W;
    const int64_t* ot = otab + (size_t)e * OBS_TAB_W;
    const int64_t tick = tick_of(trow, e);
    const double cap = of_bits(OBS_SQUASH_CAP_BITS);
    const int64_t* m = obs_pic::unit_of(blk, L, sr);
    // -- 1.1: hp over the bar (float(current_hp) / float(bar), exact Q16 views)
    put_v(v, OBS_V_HP, obs_pic::hp_over_bar(u[UNIT_COL_CURRENT_HP], m, cap));
    // -- the motion tracker's one-tick displacement over v_ref
    {
        int64_t vx, vy;
        obs_pic::motion_of(m, &vx, &vy);
        put_v(v, OBS_V_VEL_X, squash(ratio(vx, ot[OBS_TAB_V_REF_Q]), cap));
        put_v(v, OBS_V_VEL_Y, squash(ratio(vy, ot[OBS_TAB_V_REF_Q]), cap));
    }
    // -- 1.4: what just happened to it
    const int64_t mem = ot[OBS_TAB_MEMORY_TICKS];
    if (m[OBS_PIC_UNIT_HIT_SET]) {
        const int64_t age = tick - m[OBS_PIC_UNIT_HIT_TICK];
        put_v(v, OBS_V_HIT_RECENT, age <= mem ? 1.0 : 0.0);
        const double a = ratio(age, mem);
        put_v(v, OBS_V_HIT_AGE, a < 1.0 ? a : 1.0);                // min(1.0, a)
        const double dx = (double)m[OBS_PIC_UNIT_HIT_DX], dy = (double)m[OBS_PIC_UNIT_HIT_DY];
        const double n = __dsqrt_rn(add(mul(dx, dx), mul(dy, dy)));   // obs_spec_v1.length
        if (n != 0.0) {
            put_v(v, OBS_V_HIT_FROM_COS, div(-dx, n));
            put_v(v, OBS_V_HIT_FROM_SIN, div(dy, n));
        }
        put_v(v, OBS_V_HIT_CAUSE, (double)m[OBS_PIC_UNIT_HIT_CAUSE]);
    } else {
        put_v(v, OBS_V_HIT_AGE, 1.0);
        put_v(v, OBS_V_HIT_CAUSE, -1.0);
    }
    // -- 1.5: the team's records (marines / zombies), seen now, zombies near
    int64_t others = 0;
    for (int i = 0; i < T; ++i)
        if (i != team) others += teams[((size_t)e * T + i) * OBS_TEAM_W + OBS_TEAM_N];
    const uint8_t* vis = visible + ((size_t)e * (T + 1) + team) * U;
    const int64_t near = ot[OBS_TAB_ZOMBIE_NEAR_TILES];
    const int64_t near2 = near * near;
    int64_t marines = 0, seen = 0, zombies = 0, close = 0;
    for (int q = 0; q < U; ++q) {
        const int64_t* rc = obs_pic::rec_of(blk, L, U, team, q);
        if (!rc[OBS_PIC_REC_PRESENT]) continue;
        if (!rc[OBS_PIC_REC_ZOMBIE]) {
            ++marines;
            if (vis[q]) ++seen;
            continue;
        }
        ++zombies;
        const int64_t zx = rc[OBS_PIC_REC_CENTRE_X], zy = rc[OBS_PIC_REC_CENTRE_Y];
        for (int k = 0; k < U; ++k) {
            const int64_t* mu = rows + (size_t)k * UNIT_STORE_W;
            if (!mu[UNIT_COL_VALID] || !alive(mu) || mu[UNIT_COL_IS_ZOMBIE] ||
                mu[UNIT_COL_TEAM] != (int64_t)team)
                continue;
            const int64_t dx = zx - obs_sight::centre_x(mu), dy = zy - obs_sight::centre_y(mu);
            if (dx * dx + dy * dy <= near2) {
                ++close;
                break;
            }
        }
    }
    const int64_t nz = ot[OBS_TAB_ZOMBIES];
    put_v(v, OBS_V_ENEMIES_KNOWN, ratio(marines, others));
    put_v(v, OBS_V_ENEMIES_SEEN, ratio(seen, others));
    put_v(v, OBS_V_ZOMBIES_KNOWN, squash(ratio(zombies, nz), cap));
    put_v(v, OBS_V_ZOMBIES_NEAR, squash(ratio(close, nz), cap));
}

__global__ void k_picture_damage_sums(const int64_t* store, int U, int T, const int64_t* state,
                                      int W, double* out, int reverse) {
    const int e = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * U) return;
    const int t = idx / U, r = idx % U;
    const int64_t* u = store + ((size_t)e * U + r) * UNIT_STORE_W;
    const PicLayout L = pic_layout(T, U);
    const int64_t* blk = state + (size_t)e * W;
    out[((size_t)e * T + t) * U + r] =
        u[UNIT_COL_VALID] ? obs_pic::damage_sum(blk, L, t, u[UNIT_COL_ID], reverse != 0) : 0.0;
}

}  // namespace

bool obs_picture(int n_env, std::uintptr_t store, int n_units, int store_w, UnitEventRing& ring,
                 ObjectTable& grenades, ObjectTable& charges, std::uintptr_t solid,
                 std::uintptr_t screen, int h, int w, std::uintptr_t tick_row,
                 std::uintptr_t visible, int n_teams, std::uintptr_t state, int state_w,
                 std::uintptr_t faults, bool first, int brk) {
    const char* who = "obs_picture";
    need(n_env >= 1 && n_units >= 1 && n_teams >= 1 && h >= 1 && w >= 1, who,
         "a non-positive size");
    need(store_w == UNIT_STORE_W, who, "the store's width is not UNIT_STORE_W");
    need(state_w == pic_layout(n_teams, n_units).w, who,
         "the picture block's width is not the compiled layout's");
    need(store && solid && screen && tick_row && visible && state && faults, who,
         "a null device pointer");
    need(ring.n_env() == n_env && ring.width() == unit_events::UNIT_EVENT_F, who,
         "the event ring's shape");
    need(grenades.n_env() == n_env && grenades.width() == object_tables::OBJ_GRENADE_W &&
             grenades.cap() == object_tables::OBJ_GRENADE_CAP, who, "the grenade table's shape");
    need(charges.n_env() == n_env && charges.width() == object_tables::OBJ_CHARGE_W &&
             charges.cap() == object_tables::OBJ_CHARGE_CAP, who, "the charge table's shape");
    cudaStream_t s = tstream();
    int64_t* f = reinterpret_cast<int64_t*>(faults);
    cudaMemsetAsync(f, 0, sizeof(int64_t) * (size_t)n_env, s);
    const int64_t* st = reinterpret_cast<const int64_t*>(store);
    const int64_t* tr = reinterpret_cast<const int64_t*>(tick_row);
    int64_t* blk = reinterpret_cast<int64_t*>(state);
    k_picture_units<<<dim3(1, n_env), 1, 0, s>>>(
        st, n_units, ring.records_dev(), ring.count_dev(), ring.cap(), grenades.count_dev(),
        charges.count_dev(), tr, n_teams, blk, state_w, f, first ? 1 : 0);
    launched("obs_picture (units)");
    k_picture_teams<<<dim3((n_teams + 31) / 32, n_env), 32, 0, s>>>(
        st, n_units, ring.records_dev(), ring.count_dev(), ring.cap(), grenades.records_dev(),
        grenades.count_dev(), charges.records_dev(), charges.count_dev(),
        reinterpret_cast<const uint8_t*>(solid), reinterpret_cast<const uint8_t*>(screen), h, w,
        tr, reinterpret_cast<const uint8_t*>(visible), n_teams, blk, state_w, f,
        first ? 1 : 0, brk);
    launched("obs_picture (teams)");
    // the faults ride the conductor's one books read; outside a conductor
    // tick the caller reads them (the build)
    return books_defer("picture_faults", f, (size_t)n_env);
}

void obs_picture_vector(int n_env, std::uintptr_t store, int n_units, int store_w,
                        std::uintptr_t rowmap, int n_rows, int n_slots, int n_teams,
                        std::uintptr_t obs_table, int table_w, std::uintptr_t teams,
                        std::uintptr_t tick_row, std::uintptr_t params, std::uintptr_t ended,
                        std::uintptr_t visible, std::uintptr_t state, int state_w,
                        std::uintptr_t vector) {
    const char* who = "obs_picture_vector";
    need(n_env >= 1 && n_units >= 1 && n_rows >= 1 && n_slots >= 1 && n_teams >= 1, who,
         "a non-positive size");
    need(n_rows == n_teams * n_slots, who, "n_rows is not n_teams x n_slots");
    need(store_w == UNIT_STORE_W, who, "the store's width is not UNIT_STORE_W");
    need(table_w == OBS_TAB_W, who, "the obs table's width is not OBS_TAB_W");
    need(state_w == pic_layout(n_teams, n_units).w, who, "the picture block's width");
    need(store && rowmap && obs_table && teams && tick_row && params && ended && visible &&
             state && vector, who, "a null device pointer");
    k_picture_vector<<<dim3((n_rows + 63) / 64, n_env), 64, 0, tstream()>>>(
        reinterpret_cast<const int64_t*>(store), n_units, reinterpret_cast<const int32_t*>(rowmap),
        n_rows, n_slots, n_teams, reinterpret_cast<const int64_t*>(obs_table),
        reinterpret_cast<const int64_t*>(teams), reinterpret_cast<const int64_t*>(tick_row),
        reinterpret_cast<const int64_t*>(params), reinterpret_cast<const uint8_t*>(ended),
        reinterpret_cast<const uint8_t*>(visible), reinterpret_cast<const int64_t*>(state),
        state_w, reinterpret_cast<float*>(vector));
    launched(who);
}

void obs_picture_damage_sums(int n_env, std::uintptr_t store, int n_units, int store_w,
                             int n_teams, std::uintptr_t state, int state_w, std::uintptr_t out,
                             bool reverse) {
    const char* who = "obs_picture_damage_sums";
    need(n_env >= 1 && n_units >= 1 && n_teams >= 1, who, "a non-positive size");
    need(store_w == UNIT_STORE_W, who, "the store's width is not UNIT_STORE_W");
    need(state_w == pic_layout(n_teams, n_units).w, who, "the picture block's width");
    need(store && state && out, who, "a null device pointer");
    const int n = n_teams * n_units;
    k_picture_damage_sums<<<dim3((n + 63) / 64, n_env), 64, 0, tstream()>>>(
        reinterpret_cast<const int64_t*>(store), n_units, n_teams,
        reinterpret_cast<const int64_t*>(state), state_w, reinterpret_cast<double*>(out),
        reverse ? 1 : 0);
    launched(who);
}

std::vector<std::pair<std::string, int64_t>> obs_picture_layout(int n_teams, int n_units) {
    const PicLayout L = pic_layout(n_teams, n_units);
    return {{"REC", L.rec}, {"UNIT", L.unit}, {"DMG", L.dmg}, {"DMG_HD", L.dmg_hd},
            {"OBJ", L.obj}, {"OBJ_N", L.obj_n}, {"OBJ_NEXT", L.obj_next}, {"SEEN", L.seen},
            {"SEEN_N", L.seen_n}, {"SERIAL_NEXT", L.serial_next}, {"SIDE_G", L.side_g},
            {"KNOWN_G", L.known_g}, {"SIDE_C", L.side_c}, {"KNOWN_C", L.known_c},
            {"PARAM", L.param}, {"TICK", L.tick}, {"W", L.w}};
}

}  // namespace breach_cuda
