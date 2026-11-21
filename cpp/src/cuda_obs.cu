// ==========================================================================
// #29 P8.0: THE DEVICE OBSERVATION's kernels (see cuda_obs.h), Class T.
//
// The twin of obs_spec_v1.build_windows' structure planes (spec 3.1, 3.3,
// 3.4: near_u "wall" / "door", far_f "wall_share", far_u "door"),
// FieldPlanes' door plane, and EnvRunner.row_flags. Pure integer: the only
// float16 values written (k / 16) are bit patterns generated from numpy's
// own conversion (obs_layout.h). On the target-wide CUDA determinism floor
// (--fmad=false, the strict host floor) like every .cu.
//
// The reference's arithmetic, value for value:
//   tile_x = int(x_q / 65536) truncated toward zero (Unit.tile_x);
//   centre = tile + footprint // 2 (Unit.center_tile_x / _y);
//   near window top-left (cy - HALF, cx - HALF), cells outside the map read
//     wall OUTSIDE (1) and door 0 (crop);
//   far window top-left (cy // BLOCK - HALF, cx // BLOCK - HALF) in blocks
//     anchored at tile (0, 0) (Python floor division), a block's tiles past
//     the map padded wall 1 / door 0 (blocks(plane, fill)); a block outside
//     the block grid reads wall_share OUTSIDE (1.0) and door 0;
//   wall_share = (wall tiles of the block) / 16 as float16.
// ==========================================================================
#include "cuda_obs.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "cuda_batch.cuh"          // tstream(): THE tick stream (blocking)
#include "cuda_obs_float.cuh"      // #29 P8.1: the observation's float kit
#include "cuda_obs_sight.cuh"      // #29 P8.2: the observation's sight (floordiv, tile_of)
#include "cuda_unit_events.h"      // UnitRecordList (the door tables' buffers)
#include "cuda_unit_lib.cuh"       // #29 P8.1: the unit library's integer reads (used, never added to)
#include "cuda_unit_store.h"       // UnitTablesDevice
#include "fixed_point.h"           // cos_q16 / sin_q16 (the kit)
#include "obs_layout.h"
#include "object_tables_layout.h"
#include "unit_draws_layout.h"
#include "unit_store_layout.h"
#include "unit_tables_layout.h"

namespace breach_cuda {

using namespace obs_layout;

// The spec's constants the observation shares with the unit layouts.
static_assert(OBS_HAND_GRENADE_SLOTS == unit_store::UNIT_COL_GRENADE_COUNTS_W,
              "the vector's grenade slots are the store's grenade counts");
static_assert(OBS_TAB_N_GRENADES == OBS_HAND_GRENADE_SLOTS,
              "the loadout rows' grenade block is the vector's grenade slots");
static_assert(OBS_N_STATUS_KINDS == unit_tables::UNIT_N_STATUS_KINDS,
              "one status registry");
static_assert(OBS_PI_Q16 == unit_lib::PI_Q16, "one PI_Q16");
static_assert(OBS_NO_ROW == unit_lib::NO_ROW, "one NO_ROW");
static_assert(OBS_TAB_N_WEAPONS == unit_tables::UNIT_TAB_WEAPON_MAG_SIZE_W,
              "the obs table's weapon rows are the device tables' weapon rows");

namespace {

__constant__ uint16_t c_share_half[OBS_BLOCK * OBS_BLOCK + 1] = OBS_WALL_SHARE_HALF_INIT;

constexpr int CELLS = OBS_WIN * OBS_WIN;

// Python's floor division, Unit.tile_x / tile_y: the observation's one copy
// (cuda_obs_sight.cuh, #29 P8.2).
using obs_sight::floordiv;
using obs_sight::tile_of;

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

// FieldPlanes.door: per living door, its code (open / closed) at every span
// tile on the map, the max (order-free, so an atomic max is exact).
__global__ void k_door_plane(const int64_t* doors, const int64_t* door_count, int door_cap,
                             int door_w, const int64_t* tiles, int tile_cap, int tile_w,
                             int32_t* plane, int h, int w, const int64_t* trow,
                             const int64_t* params, const uint8_t* ended) {
    const int e = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= door_cap || d >= door_count[e] || !env_due(e, trow, params, ended)) return;
    const int64_t* row = doors + ((size_t)e * door_cap + d) * door_w;
    if (!row[object_tables::OBJ_DOOR_COL_ALIVE]) return;
    const int64_t st = row[object_tables::OBJ_DOOR_COL_STATE];
    const int code = st == OBS_DOOR_STATE_OPEN ? OBS_DOOR_CODE_OPEN
                   : st == OBS_DOOR_STATE_CLOSED ? OBS_DOOR_CODE_CLOSED : OBS_DOOR_CODE_NONE;
    if (code == OBS_DOOR_CODE_NONE) return;
    const int64_t t0 = row[object_tables::OBJ_DOOR_COL_TILE_START];
    const int64_t tn = row[object_tables::OBJ_DOOR_COL_TILE_COUNT];
    for (int64_t k = 0; k < tn; ++k) {
        const int64_t ti = t0 + k;
        if (ti < 0 || ti >= tile_cap) continue;
        const int64_t* t = tiles + ((size_t)e * tile_cap + ti) * tile_w;
        const int64_t ty = t[object_tables::OBJ_DOOR_TILE_COL_TY];
        const int64_t tx = t[object_tables::OBJ_DOOR_TILE_COL_TX];
        if (0 <= ty && ty < h && 0 <= tx && tx < w)
            atomicMax(plane + ((size_t)e * h + ty) * w + tx, code);
    }
}

// One thread per (env, row, window cell): valid / in_match (cell 0) and the
// four structure planes at that cell of both windows.
__global__ void k_obs_block(const int64_t* store, int n_units, const int32_t* rowmap,
                            int n_rows, int n_slots, const uint8_t* solid,
                            const int32_t* door_plane, int h, int w, const int64_t* trow,
                            const int64_t* params, const uint8_t* ended, uint8_t* near_u,
                            uint16_t* far_f, uint8_t* far_u, uint8_t* valid_out,
                            uint8_t* in_match_out) {
    const int e = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * CELLS || !env_due(e, trow, params, ended)) return;
    const int r = idx / CELLS, cell = idx % CELLS;
    const int i = cell / OBS_WIN, j = cell % OBS_WIN;
    const size_t er = (size_t)e * n_rows + r;
    const int sr = rowmap[er];
    bool valid = false;
    const int64_t* u = nullptr;
    if (sr >= 0 && sr < n_units) {
        u = store + ((size_t)e * n_units + sr) * unit_store::UNIT_STORE_W;
        valid = u[unit_store::UNIT_COL_LIFE_STATE] == OBS_LIFE_ALIVE &&
                u[unit_store::UNIT_COL_TEAM] == (int64_t)(r / n_slots);
    }
    if (cell == 0) {
        valid_out[er] = valid ? 1 : 0;
        in_match_out[er] = sr >= 0 ? 1 : 0;
    }
    uint8_t* nu = near_u + er * OBS_NEAR_U_PLANES * CELLS;
    uint16_t* ff = far_f + er * OBS_FAR_F_PLANES * CELLS;
    uint8_t* fu = far_u + er * OBS_FAR_U_PLANES * CELLS;
    if (!valid) {
        nu[OBS_NEAR_U_WALL * CELLS + cell] = 0;
        nu[OBS_NEAR_U_DOOR * CELLS + cell] = 0;
        ff[OBS_FAR_F_WALL_SHARE * CELLS + cell] = 0;
        fu[OBS_FAR_U_DOOR * CELLS + cell] = 0;
        return;
    }
    const int64_t fp = u[unit_store::UNIT_COL_FOOTPRINT];
    const int64_t cx = tile_of(u[unit_store::UNIT_COL_X_Q]) + floordiv(fp, 2);
    const int64_t cy = tile_of(u[unit_store::UNIT_COL_Y_Q]) + floordiv(fp, 2);
    const uint8_t* so = solid + (size_t)e * h * w;
    const int32_t* dp = door_plane + (size_t)e * h * w;
    // -- near (3.1, 3.4) --
    const int64_t y = cy - OBS_HALF + i, x = cx - OBS_HALF + j;
    const bool inside = 0 <= y && y < h && 0 <= x && x < w;
    nu[OBS_NEAR_U_WALL * CELLS + cell] =
        inside ? (so[y * w + x] ? 1 : 0) : (uint8_t)OBS_WALL_OUTSIDE;
    nu[OBS_NEAR_U_DOOR * CELLS + cell] = inside ? (uint8_t)dp[y * w + x] : 0;
    // -- far (3.3, 3.4): the block grid anchored at tile (0, 0) --
    const int64_t hb = (h + OBS_BLOCK - 1) / OBS_BLOCK, wb = (w + OBS_BLOCK - 1) / OBS_BLOCK;
    const int64_t bi = floordiv(cy, OBS_BLOCK) - OBS_HALF + i;
    const int64_t bj = floordiv(cx, OBS_BLOCK) - OBS_HALF + j;
    if (0 <= bi && bi < hb && 0 <= bj && bj < wb) {
        int walls = 0, door = 0;
        for (int a = 0; a < OBS_BLOCK; ++a) {
            const int64_t ty = bi * OBS_BLOCK + a;
            for (int b = 0; b < OBS_BLOCK; ++b) {
                const int64_t tx = bj * OBS_BLOCK + b;
                if (ty < h && tx < w) {
                    walls += so[ty * w + tx] ? 1 : 0;
                    const int dv = dp[ty * w + tx];
                    door = dv > door ? dv : door;
                } else {
                    walls += 1;                     // blocks(wall, 1): the pad is wall
                }
            }
        }
        ff[OBS_FAR_F_WALL_SHARE * CELLS + cell] = c_share_half[walls];
        fu[OBS_FAR_U_DOOR * CELLS + cell] = (uint8_t)door;
    } else {
        ff[OBS_FAR_F_WALL_SHARE * CELLS + cell] = (uint16_t)OBS_WALL_SHARE_OUTSIDE_HALF;
        fu[OBS_FAR_U_DOOR * CELLS + cell] = 0;
    }
}

// ---------------------------------------------------------------------------
// #29 P8.1: THE VECTOR's columns of the store and the tables (spec 1.1, 1.2,
// 1.5: obs_spec_v1.build_vector / _kit_values, line for line). One thread per
// (env, row); a row that is not valid reads 0 in every column this block owns
// (observe_rows zeroes the vector). Every value is float64 in the reference's
// order (obs_f64), cast ONCE to float32. The table values come from the obs
// table (`ot`, device_obs.table_values) and the team rows; the unit
// library's integer reads (mag_status, composed_flags, handling_ticks,
// scaled, wrap_q16) take the device tables (`tab`) as they were built for.
// ---------------------------------------------------------------------------
using namespace obs_f64;

__device__ __forceinline__ void put_v(float* v, int col, double x) { v[col] = to_f32(x); }

__global__ void k_obs_vector(const int64_t* store, int n_units, const int64_t* tables,
                             const int32_t* rowmap, int n_rows, int n_slots, int n_teams,
                             const int64_t* otab, const int64_t* teams, const int64_t* trow,
                             const int64_t* params, const uint8_t* ended, float* vector) {
    const int e = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows || !env_due(e, trow, params, ended)) return;
    const size_t er = (size_t)e * n_rows + r;
    float* v = vector + er * OBS_V;
    const int team = r / n_slots;
    const int sr = rowmap[er];
    const int64_t* env_rows = store + (size_t)e * n_units * unit_store::UNIT_STORE_W;
    const int64_t* u = nullptr;
    bool valid = false;
    if (sr >= 0 && sr < n_units) {
        u = env_rows + (size_t)sr * unit_store::UNIT_STORE_W;
        valid = u[unit_store::UNIT_COL_LIFE_STATE] == OBS_LIFE_ALIVE &&
                u[unit_store::UNIT_COL_TEAM] == (int64_t)team;
    }
    // the columns this block owns, zeroed first (the reference's out[...] = 0)
    const int owned[] = {
        OBS_V_FACING_COS, OBS_V_FACING_SIN, OBS_V_OMEGA, OBS_V_TURN_LEFT, OBS_V_PUSHED_X,
        OBS_V_PUSHED_Y, OBS_V_EFFORT, OBS_V_REACT, OBS_V_PRONE, OBS_V_CAN_MOVE, OBS_V_CAN_ACT,
        OBS_V_CAN_AIM, OBS_V_ROUNDS, OBS_V_MAGS, OBS_V_AMMO_UNTRACKED, OBS_V_RELOADING,
        OBS_V_RELOAD_LEFT, OBS_V_THROW_WINDUP, OBS_V_THROW_LEFT, OBS_V_THROW_KIND,
        OBS_V_CHARGES, OBS_V_HAS_CHARGE_KIND, OBS_V_PLANTING, OBS_V_PLANT_LEFT,
        OBS_V_PLANT_TIMER, OBS_V_MELEE_COOL, OBS_V_HAS_MELEE, OBS_V_SHOVE_COOL, OBS_V_CLASS,
        OBS_V_WEAPON, OBS_V_RANGE, OBS_V_ROF, OBS_V_SPREAD, OBS_V_TRIGGER, OBS_V_TIME_LEFT,
        OBS_V_MATES_ALIVE, OBS_V_TEAM_AMMO};
    for (int c : owned) v[c] = 0.0f;
    for (int k = 0; k < OBS_N_STATUS_KINDS; ++k) v[OBS_V_STATUS_LEFT_0 + k] = 0.0f;
    for (int k = 0; k < OBS_HAND_GRENADE_SLOTS; ++k) {
        v[OBS_V_GRENADE_0 + 2 * k] = 0.0f;
        v[OBS_V_GRENADE_KIND_OK_0 + 2 * k] = 0.0f;
    }
    if (!valid) return;
    const int64_t* ot = otab + (size_t)e * OBS_TAB_W;
    const int64_t* tab = tables + (size_t)e * unit_tables::UNIT_TABLES_W;
    const int64_t tick = trow[(size_t)e * unit_draws::TICK_ROW_W + unit_draws::TICK_ROW_TICK];
    const double cap = of_bits(OBS_SQUASH_CAP_BITS);
    bool bad = false;
    using namespace unit_store;
    // -- 1.1 body --
    const int64_t f = u[UNIT_COL_FACING_Q];
    put_v(v, OBS_V_FACING_COS, div((double)fixedpoint::cos_q16((int32_t)f), (double)OBS_FP_ONE));
    put_v(v, OBS_V_FACING_SIN, div((double)fixedpoint::sin_q16((int32_t)f), (double)OBS_FP_ONE));
    const int64_t turn = unit_lib::scaled(u, unit_tables::UNIT_HOOK_TURN,
                                          ot[OBS_TAB_TURN_RATE_CLASS_Q]);
    put_v(v, OBS_V_OMEGA, squash(ratio(u[UNIT_COL_OMEGA_Q], turn), cap));
    put_v(v, OBS_V_TURN_LEFT,
          div((double)unit_lib::wrap_q16(u[UNIT_COL_FACING_TARGET_Q] - f), (double)OBS_PI_Q16));
    put_v(v, OBS_V_PUSHED_X, squash(ratio(u[UNIT_COL_PVX], ot[OBS_TAB_V_FALL_Q]), cap));
    put_v(v, OBS_V_PUSHED_Y, squash(ratio(u[UNIT_COL_PVY], ot[OBS_TAB_V_FALL_Q]), cap));
    put_v(v, OBS_V_EFFORT, squash(ratio(u[UNIT_COL_EFFORT], OBS_FP_ONE), cap));
    put_v(v, OBS_V_REACT, ratio(u[UNIT_COL_REACT_TICKS], ot[OBS_TAB_REACT_TICKS]));
    const unit_lib::Flags fl = unit_lib::composed_flags(u, tab, false, &bad);
    put_v(v, OBS_V_PRONE, fl.is_prone ? 1.0 : 0.0);
    put_v(v, OBS_V_CAN_MOVE, fl.can_move ? 1.0 : 0.0);
    put_v(v, OBS_V_CAN_ACT, fl.can_act ? 1.0 : 0.0);
    put_v(v, OBS_V_CAN_AIM, fl.can_aim ? 1.0 : 0.0);
    {   // _status_left: per kind the longest remaining_ticks of the list
        int64_t best[OBS_N_STATUS_KINDS];
        for (int k = 0; k < OBS_N_STATUS_KINDS; ++k) best[k] = 0;
        const int64_t* s = u + UNIT_COL_STATUSES;
        for (int i = 0; i < unit_lib::STATUS_ROWS; ++i) {
            const int64_t kind = s[4 * i];
            if (kind < 0) break;
            if (kind < OBS_N_STATUS_KINDS && s[4 * i + 2] > best[kind]) best[kind] = s[4 * i + 2];
        }
        for (int k = 0; k < OBS_N_STATUS_KINDS; ++k)
            put_v(v, OBS_V_STATUS_LEFT_0 + k,
                  squash(ratio(best[k], ot[OBS_TAB_STATUS_SCALE + k]), cap));
    }
    // -- 1.2 weapon and kit (_kit_values) --
    const int64_t w = u[UNIT_COL_WEAPON_ID];                 // NO_ROW: no weapon
    const int64_t lo = u[UNIT_COL_LOADOUT_NAME];             // NO_ROW: no loadout
    const int64_t* L = lo >= 0 ? ot + OBS_TAB_LOADOUT + lo * OBS_LOADOUT_W : nullptr;
    const int64_t* W = w >= 0 ? ot + OBS_TAB_WEAPON + w * OBS_WEAPON_W : nullptr;
    {
        const int64_t mags_total = L ? L[OBS_LOADOUT_MAGAZINES] : 0;
        double rounds = 1.0, mags = 1.0, untracked = 1.0, reloading = 0.0, reload_left = 0.0;
        const unit_lib::MagStatus m = W ? unit_lib::mag_status(u, tab, tick, &bad)
                                        : unit_lib::MagStatus{false, 0, -1, false};
        if (m.tracked) {
            rounds = ratio(m.rounds, W[OBS_WEAPON_MAG_SIZE]);
            if (m.spares < 0) {
                mags = untracked = 1.0;
            } else {
                mags = squash(ratio(m.spares, mags_total > 1 ? mags_total : 1), cap);
                untracked = 0.0;
            }
            reloading = m.reloading ? 1.0 : 0.0;
            if (m.reloading)
                reload_left = ratio(u[UNIT_COL_RELOAD_DONE_TICK] - tick,
                                    unit_lib::handling_ticks(u, W[OBS_WEAPON_RELOAD_TICKS]));
        }
        put_v(v, OBS_V_ROUNDS, rounds);
        put_v(v, OBS_V_MAGS, mags);
        put_v(v, OBS_V_AMMO_UNTRACKED, untracked);
        put_v(v, OBS_V_RELOADING, reloading);
        put_v(v, OBS_V_RELOAD_LEFT, reload_left);
    }
    for (int k = 0; k < OBS_HAND_GRENADE_SLOTS; ++k) {
        const int64_t have = L ? L[OBS_LOADOUT_GRENADES + k] : 0;
        if (have > 0) {
            put_v(v, OBS_V_GRENADE_0 + 2 * k, ratio(u[UNIT_COL_GRENADE_COUNTS + k], have));
            put_v(v, OBS_V_GRENADE_KIND_OK_0 + 2 * k, 1.0);
        }
    }
    const int64_t release = u[UNIT_COL_THROW_RELEASE_TICK];
    if (release >= 0) {
        put_v(v, OBS_V_THROW_WINDUP, 1.0);
        put_v(v, OBS_V_THROW_LEFT, ratio(release - tick,
                                         unit_lib::handling_ticks(u, ot[OBS_TAB_HAND_WINDUP_TICKS])));
        put_v(v, OBS_V_THROW_KIND, (double)u[UNIT_COL_THROW_KIND]);
    } else {
        put_v(v, OBS_V_THROW_KIND, -1.0);
    }
    put_v(v, OBS_V_CHARGES, ratio(u[UNIT_COL_HAS_EXPLOSIVE], L ? L[OBS_LOADOUT_EXPLOSIVES] : 0));
    put_v(v, OBS_V_HAS_CHARGE_KIND, (L && L[OBS_LOADOUT_CHARGE]) ? 1.0 : 0.0);
    const int64_t pdone = u[UNIT_COL_PLANT_DONE_TICK];
    if (pdone >= 0) {
        put_v(v, OBS_V_PLANTING, 1.0);
        put_v(v, OBS_V_PLANT_LEFT,
              ratio(pdone - tick, unit_lib::handling_ticks(u, ot[OBS_TAB_PLANT_TICKS])));
        put_v(v, OBS_V_PLANT_TIMER,
              ratio(u[UNIT_COL_PLANT_FUSE_TICKS], ot[OBS_TAB_C4_FUSE_MAX_TICKS]));
    }
    const int64_t mr = u[UNIT_COL_MELEE_ID];
    if (mr >= 0 && u[UNIT_COL_LAST_BLOW_TICK] >= 0) {
        const int64_t rof = ot[OBS_TAB_WEAPON + mr * OBS_WEAPON_W + OBS_WEAPON_ROF_TICKS];
        const int64_t left = u[UNIT_COL_LAST_BLOW_TICK] + rof - tick;
        put_v(v, OBS_V_MELEE_COOL, ratio(left > 0 ? left : 0, rof));
    }
    put_v(v, OBS_V_HAS_MELEE, mr != OBS_NO_ROW ? 1.0 : 0.0);
    if (u[UNIT_COL_LAST_SHOVE_TICK] >= 0) {
        const int64_t cd = ot[OBS_TAB_SHOVE_COOLDOWN_TICKS];
        const int64_t left = u[UNIT_COL_LAST_SHOVE_TICK] + cd - tick;
        put_v(v, OBS_V_SHOVE_COOL, ratio(left > 0 ? left : 0, cd));
    }
    put_v(v, OBS_V_CLASS, (double)lo);
    put_v(v, OBS_V_WEAPON, (double)w);
    if (W) {
        put_v(v, OBS_V_RANGE, squash(ratio(W[OBS_WEAPON_RANGE_TILES], ot[OBS_TAB_L_REF]), cap));
        put_v(v, OBS_V_ROF, squash(ratio(W[OBS_WEAPON_ROF_TICKS], ot[OBS_TAB_REF_ROF_TICKS]), cap));
        put_v(v, OBS_V_SPREAD,
              squash(ratio(unit_lib::scaled(u, unit_tables::UNIT_HOOK_SPREAD, W[OBS_WEAPON_CONE_AIM_Q]),
                           ot[OBS_TAB_REF_CONE_Q]), cap));
    }
    put_v(v, OBS_V_TRIGGER, u[UNIT_COL_LIVE_TRIGGER] != 0 ? 1.0 : 0.0);
    // -- 1.5 the clock and the team --
    const int64_t limit = ot[OBS_TAB_LIMIT_TICKS];
    put_v(v, OBS_V_TIME_LEFT, ratio(limit - tick, limit));
    const int64_t* tm = teams + ((size_t)e * n_teams + team) * OBS_TEAM_W;
    const int64_t n_t = tm[OBS_TEAM_N];
    int64_t mates = 0, ammo = 0;
    for (int q = 0; q < n_units; ++q) {                     // TeamKnowledge.mates
        const int64_t* m = env_rows + (size_t)q * UNIT_STORE_W;
        if (!m[UNIT_COL_VALID] || m[UNIT_COL_LIFE_STATE] != OBS_LIFE_ALIVE ||
            m[UNIT_COL_IS_ZOMBIE] || m[UNIT_COL_TEAM] != (int64_t)team)
            continue;
        ++mates;
        const int64_t mw = m[UNIT_COL_WEAPON_ID];
        if (mw < 0) continue;
        const unit_lib::MagStatus ms = unit_lib::mag_status(m, tab, tick, &bad);
        if (!ms.tracked) continue;
        const int64_t ml = m[UNIT_COL_LOADOUT_NAME];
        int64_t s = ms.spares;
        if (s < 0) s = ml >= 0 ? ot[OBS_TAB_LOADOUT + ml * OBS_LOADOUT_W + OBS_LOADOUT_MAGAZINES] : 0;
        ammo += ms.rounds + s * ot[OBS_TAB_WEAPON + mw * OBS_WEAPON_W + OBS_WEAPON_MAG_SIZE];
    }
    put_v(v, OBS_V_MATES_ALIVE, ratio(mates, n_t));
    int64_t total = 0;
    const int64_t tl = tm[OBS_TEAM_LOADOUT];
    if (tl >= 0) {
        const int64_t* TL = ot + OBS_TAB_LOADOUT + tl * OBS_LOADOUT_W;
        const int64_t tw = TL[OBS_LOADOUT_WEAPON];
        if (tw >= 0)
            total = n_t * TL[OBS_LOADOUT_MAGAZINES] *
                    ot[OBS_TAB_WEAPON + tw * OBS_WEAPON_W + OBS_WEAPON_MAG_SIZE];
    }
    put_v(v, OBS_V_TEAM_AMMO, squash(ratio(ammo, total), cap));
    (void)bad;   // a status kind outside the registry / a weapon index past the table:
                 // the store's encoder refuses both, so a valid store never sets it
}

// ---------------------------------------------------------------------------
// #29 P8.1 TEST-ONLY probes of the float kit (part FLOATKIT of the check).
// ---------------------------------------------------------------------------
__global__ void k_probe_half(const double* x, uint16_t* out, int n, int via_f32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // via_f32: THE BREAK (double rounding through float32), never the kit
    out[i] = via_f32 ? to_half_bits((double)to_f32(x[i])) : to_half_bits(x[i]);
}
__global__ void k_probe_f32(const double* x, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = to_f32(x[i]);
}
__global__ void k_probe_muladd(const double* a, const double* b, const double* c, double* out,
                               int n, int fused) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // fused: THE BREAK (what contraction would build), never the kit
    out[i] = fused ? fma(a[i], b[i], c[i]) : add(mul(a[i], b[i]), c[i]);
}
__global__ void k_probe_squash(const double* x, double* out64, float* out32, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double s = squash(x[i], of_bits(OBS_SQUASH_CAP_BITS));
    out64[i] = s;
    out32[i] = to_f32(s);
}

void need(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("obs_block: ") + what);
}

}  // namespace

void obs_block(int n_env, std::uintptr_t store, int n_units, int store_w,
               std::uintptr_t rowmap, int n_rows, int n_slots,
               std::uintptr_t solid, int h, int w,
               UnitRecordList& doors, UnitRecordList& door_tiles, std::uintptr_t door_plane,
               std::uintptr_t tick_row, std::uintptr_t params, std::uintptr_t ended,
               std::uintptr_t near_u, std::uintptr_t far_f, std::uintptr_t far_u,
               std::uintptr_t valid, std::uintptr_t in_match) {
    need(n_env >= 1 && n_units >= 1 && n_rows >= 1 && n_slots >= 1 && h >= 1 && w >= 1,
         "a non-positive size");
    need(store_w == unit_store::UNIT_STORE_W, "the store's width is not UNIT_STORE_W");
    need(store && rowmap && solid && door_plane && tick_row && params && ended && near_u &&
         far_f && far_u && valid && in_match, "a null device pointer");
    need(doors.n_env() == n_env && door_tiles.n_env() == n_env, "the door tables' env count");
    need(doors.width() == object_tables::OBJ_DOOR_W &&
         door_tiles.width() == object_tables::OBJ_DOOR_TILE_W, "the door tables' widths");
    cudaStream_t s = tstream();
    int32_t* dp = reinterpret_cast<int32_t*>(door_plane);
    cudaMemsetAsync(dp, 0, sizeof(int32_t) * (size_t)n_env * h * w, s);
    const int64_t* trow = reinterpret_cast<const int64_t*>(tick_row);
    const int64_t* prm = reinterpret_cast<const int64_t*>(params);
    const uint8_t* end = reinterpret_cast<const uint8_t*>(ended);
    const int dcap = doors.cap();
    k_door_plane<<<dim3((dcap + 63) / 64, n_env), 64, 0, s>>>(
        doors.records_dev(), doors.count_dev(), dcap, doors.width(), door_tiles.records_dev(),
        door_tiles.cap(), door_tiles.width(), dp, h, w, trow, prm, end);
    const int cells = n_rows * CELLS;
    k_obs_block<<<dim3((cells + 255) / 256, n_env), 256, 0, s>>>(
        reinterpret_cast<const int64_t*>(store), n_units, reinterpret_cast<const int32_t*>(rowmap),
        n_rows, n_slots, reinterpret_cast<const uint8_t*>(solid), dp, h, w, trow, prm, end,
        reinterpret_cast<uint8_t*>(near_u), reinterpret_cast<uint16_t*>(far_f),
        reinterpret_cast<uint8_t*>(far_u), reinterpret_cast<uint8_t*>(valid),
        reinterpret_cast<uint8_t*>(in_match));
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("obs_block: launch failed: ") +
                                 cudaGetErrorString(err));
}

void obs_vector(int n_env, std::uintptr_t store, int n_units, int store_w,
                const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
                int n_teams, std::uintptr_t obs_table, int table_w, std::uintptr_t teams,
                std::uintptr_t tick_row, std::uintptr_t params, std::uintptr_t ended,
                std::uintptr_t vector) {
    need(n_env >= 1 && n_units >= 1 && n_rows >= 1 && n_slots >= 1 && n_teams >= 1,
         "a non-positive size");
    need(n_rows == n_teams * n_slots, "n_rows is not n_teams x n_slots");
    need(store_w == unit_store::UNIT_STORE_W, "the store's width is not UNIT_STORE_W");
    need(table_w == OBS_TAB_W, "the obs table's width is not OBS_TAB_W (regenerate obs_layout.h)");
    need(tables.n_env() == n_env && tables.width() == unit_tables::UNIT_TABLES_W,
         "the device tables' shape");
    need(store && rowmap && obs_table && teams && tick_row && params && ended && vector,
         "a null device pointer");
    cudaStream_t s = tstream();
    k_obs_vector<<<dim3((n_rows + 63) / 64, n_env), 64, 0, s>>>(
        reinterpret_cast<const int64_t*>(store), n_units, tables.values(),
        reinterpret_cast<const int32_t*>(rowmap), n_rows, n_slots, n_teams,
        reinterpret_cast<const int64_t*>(obs_table), reinterpret_cast<const int64_t*>(teams),
        reinterpret_cast<const int64_t*>(tick_row), reinterpret_cast<const int64_t*>(params),
        reinterpret_cast<const uint8_t*>(ended), reinterpret_cast<float*>(vector));
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("obs_vector: launch failed: ") +
                                 cudaGetErrorString(err));
}

namespace {
template <typename T>
T* dev_copy(const T* host, int n) {
    T* d = nullptr;
    if (cudaMalloc(&d, sizeof(T) * (size_t)(n > 0 ? n : 1)) != cudaSuccess)
        throw std::runtime_error("obs probe: cudaMalloc failed");
    if (host) cudaMemcpy(d, host, sizeof(T) * (size_t)n, cudaMemcpyHostToDevice);
    return d;
}
template <typename T>
void dev_back(T* host, T* d, int n) {
    cudaMemcpy(host, d, sizeof(T) * (size_t)n, cudaMemcpyDeviceToHost);
    cudaFree(d);
}
void probe_done(const char* what) {
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}
}  // namespace

void obs_probe_half(const double* x, uint16_t* out, int n, bool via_f32) {
    double* dx = dev_copy(x, n);
    uint16_t* d = dev_copy<uint16_t>(nullptr, n);
    k_probe_half<<<(n + 255) / 256, 256>>>(dx, d, n, via_f32 ? 1 : 0);
    probe_done("obs_probe_half");
    dev_back(out, d, n);
    cudaFree(dx);
}

void obs_probe_f32(const double* x, float* out, int n) {
    double* dx = dev_copy(x, n);
    float* d = dev_copy<float>(nullptr, n);
    k_probe_f32<<<(n + 255) / 256, 256>>>(dx, d, n);
    probe_done("obs_probe_f32");
    dev_back(out, d, n);
    cudaFree(dx);
}

void obs_probe_muladd(const double* a, const double* b, const double* c, double* out, int n,
                      bool fused) {
    double* da = dev_copy(a, n);
    double* db = dev_copy(b, n);
    double* dc = dev_copy(c, n);
    double* d = dev_copy<double>(nullptr, n);
    k_probe_muladd<<<(n + 255) / 256, 256>>>(da, db, dc, d, n, fused ? 1 : 0);
    probe_done("obs_probe_muladd");
    dev_back(out, d, n);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

void obs_probe_squash(const double* x, double* out64, float* out32, int n) {
    double* dx = dev_copy(x, n);
    double* d64 = dev_copy<double>(nullptr, n);
    float* d32 = dev_copy<float>(nullptr, n);
    k_probe_squash<<<(n + 255) / 256, 256>>>(dx, d64, d32, n);
    probe_done("obs_probe_squash");
    dev_back(out64, d64, n);
    dev_back(out32, d32, n);
    cudaFree(dx);
}

const char* obs_layout_fingerprint() { return OBS_LAYOUT_FINGERPRINT; }

}  // namespace breach_cuda
