// ==========================================================================
// #29 P8.5a: THE WINDOWS (see cuda_obs.h), Class T. The twin of
// obs_spec_v1.build_windows' physics, sight and far unit planes (spec 3.2,
// 3.3, 3.5, 3.7), FieldPlanes' raw planes and block reductions, and
// build_vector's air block and felt temperature (spec 1.3), line for line,
// every env on its decision tick with its ended latch clear, on THE tick
// stream. The host builders are the reference.
//
// THE RAW PLANES (FieldPlanes.raw), read from the RESIDENT planes in their own
// dtype and currency, dequantized as the host's *_fixed.dequantize does (a
// float64 divide by the currency's power of two: exact):
//   temperature  gas_fixed.dequantize_q16(temperature)
//   fire         fire_fixed.dequantize(fire)
//   poison / teargas / smoke   gas_fixed.dequantize_trace(gas[g])
//   pressure     atmosphere_fixed.dequantize(atmosphere)
//   wind_x / _y  GameMap.wind_load_plane's W = float64(N_O2 + N_N2) *
//                float64((vx^2 + vy^2) >> 16) (0 on solid), j = floor(floor(W
//                / 65536) * wind_k / 65536) (0 when wind_k <= 0), the wind's
//                direction vx / sqrt(vx*vx + vy*vy) in float64 (0 where the
//                length is 0), raw = j * direction.
// THE FAR REDUCE (FieldPlanes.far_raw): the block grid anchored at tile (0, 0),
// a block's tiles past the map padded 0.0; the first six planes the block MAX
// (order-free: no NaN, no signed zero among dequantized integers); the wind
// pair the raw wind of the block's ARGMAX load (numpy's first maximum, tiles
// in row-major order). One thread per (env, block): k_far_raw.
// THE LAW (physics_value), float64 in the reference's order through the kit
// (never fused), per the observing soldier: temperature / its profile's
// temperature_max; fire as it is; (poison / poison_min_density) x its
// resist_mult[POISON]; teargas / teargas_blind_density; smoke / smoke_dense;
// (pressure - pressure_min) / (pressure_max - pressure_min); the wind / (mass_g
// x cap_q); a zero table value gives 0 (_div's rule, so -0.0 is never made
// there). Then the squash (CUDA's log) and ONE cast: to float16 through the
// kit's hand conversion (the planes), to float32 (the vector's air block).
// Cells off the map (near) / off the block grid (far) read 0.
// THE SIGHT PLANES (spec 3.5): watched = OR over the team's records of
// window_cell FROM the record's centre; sees_target = window_cell FROM the
// cell TO the recorded centre of the entry the decoder aimed at (the
// DeviceObservation.aimed uid found in the list just built at a place past
// the mates' block, as build_list's 1h tail finds it; 0 without one).
// THE FAR UNIT PLANES (spec 3.7): per block of the far window, the other
// living mates at their live centres, the team's records at their recorded
// centres split seen now / not.
// THE FELT TEMPERATURE: the unit library's exchange.felt_temperature_q over
// the resident heat / rad_flux_sweep planes, then t_q / 65536 over the
// profile's temperature_max, squashed.
// A row that is not valid reads 0 in every column and plane this file owns.
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
#include "cuda_unit_damage.cuh"    // unit_lib::felt_temperature_q (used, never added to)
#include "cuda_unit_lib.cuh"
#include "cuda_unit_store.h"       // UnitTablesDevice
#include "obs_layout.h"
#include "unit_draws_layout.h"
#include "unit_store_layout.h"
#include "unit_tables_layout.h"

namespace breach_cuda {

using namespace obs_layout;

static_assert(OBS_N_PHYSICS == 8, "the windows' law follows the eight physics planes by name");
static_assert(OBS_NEAR_F_PLANES == OBS_N_PHYSICS, "near_f is the physics planes");

namespace {

using namespace obs_f64;
using obs_pic::PicLayout;
using obs_pic::pic_layout;
using obs_rows::env_due;
using obs_rows::is_mate;
using obs_rows::is_seen;
using obs_sight::floordiv;

constexpr int CELLS = OBS_WIN * OBS_WIN;
constexpr int NP = OBS_N_PHYSICS;
enum { P_TEMP = 0, P_FIRE, P_POISON, P_TEAR, P_SMOKE, P_PRESS, P_WX, P_WY };

// One env's resident planes.
struct Planes {
    const int32_t* temp;
    const int32_t* fire;
    const int32_t* gas;        // (n_gases, h, w)
    const int32_t* atm;
    const int32_t* wx;
    const int32_t* wy;
    const uint8_t* solid;
    int h, w;
    double k;                  // float(ImpulseTable.wind_k_q)
};

__device__ __forceinline__ Planes planes_of(int e, const int32_t* temp, const int32_t* fire,
                                            const int32_t* gas, int n_gases, const int32_t* atm,
                                            const int32_t* wx, const int32_t* wy,
                                            const uint8_t* solid, int h, int w, const int64_t* ot) {
    const size_t hw = (size_t)h * w;
    return Planes{temp + e * hw, fire + e * hw, gas + e * (size_t)n_gases * hw, atm + e * hw,
                  wx + e * hw,   wy + e * hw,   solid + e * hw,                 h,
                  w,             (double)ot[OBS_TAB_WIND_K_Q]};
}

// FieldPlanes.raw at tile i (in the map), the eight planes in the spec's
// order, and the wind's load W at the tile (the far argmax's key).
__device__ inline double raw8(const Planes& P, size_t i, double* out) {
    const size_t hw = (size_t)P.h * P.w;
    out[P_TEMP] = div((double)P.temp[i], (double)OBS_Q16_ONE);
    out[P_FIRE] = div((double)P.fire[i], (double)OBS_FIRE_ONE);
    out[P_POISON] = div((double)P.gas[OBS_GAS_POISON * hw + i], (double)OBS_TRACE_ONE);
    out[P_TEAR] = div((double)P.gas[OBS_GAS_TEARGAS * hw + i], (double)OBS_TRACE_ONE);
    out[P_SMOKE] = div((double)P.gas[OBS_GAS_SMOKE * hw + i], (double)OBS_TRACE_ONE);
    out[P_PRESS] = div((double)P.atm[i], (double)OBS_ATM_ONE);
    // the wind: wind_load_plane's W (the square exact in int64, the product
    // float64), then unit_impulse.wind_impulse_q's law in float64
    const int64_t vx = P.wx[i], vy = P.wy[i];
    double load = 0.0;
    if (!P.solid[i]) {
        const int64_t v2 = (vx * vx + vy * vy) >> OBS_WIND_SHIFT;
        const int64_t n_raw = (int64_t)P.gas[OBS_GAS_O2 * hw + i] + (int64_t)P.gas[OBS_GAS_N2 * hw + i];
        load = mul((double)n_raw, (double)v2);
    }
    double j = 0.0;
    if (P.k > 0.0)
        j = floor(div(mul(floor(div(load, (double)OBS_FP_ONE)), P.k), (double)OBS_FP_ONE));
    const double fx = (double)vx, fy = (double)vy;
    const double n = __dsqrt_rn(add(mul(fx, fx), mul(fy, fy)));
    const double dx = n > 0.0 ? div(fx, n) : 0.0;
    const double dy = n > 0.0 ? div(fy, n) : 0.0;
    out[P_WX] = mul(j, dx);
    out[P_WY] = mul(j, dy);
    return load;
}

// The observing soldier's table values (physics_value's divisors).
struct Den {
    double tmax, plo, phi;     // its environment profile
    double pd, resist;         // poison_min_density, its resist_mult[POISON]
    double td, sd;             // teargas_blind_density, smoke_dense
    double wd;                 // float(mass_g) * float(cap_q)
};

// u.environment: its [env_profiles.*] row, else its species' own profile
// (unit_schema._d_environment); mitigation_for: the zombie profile on
// is_zombie, else its species'. *bad when a row is past its block.
__device__ inline Den den_of(const int64_t* u, const int64_t* ot, int brk, bool* bad) {
    using namespace unit_store;
    Den d{};
    const int64_t p = u[UNIT_COL_ENV_PROFILE_NAME];
    const int64_t s = u[UNIT_COL_SPECIES_ID];
    int64_t base;
    if (p >= 0) {
        if (p >= OBS_TAB_N_PROFILES) *bad = true;
        base = OBS_TAB_PROFILE + p * OBS_PROFILE_W;
    } else {
        if (s < 0 || s >= OBS_TAB_N_SPECIES) *bad = true;
        base = OBS_TAB_SPECIES_PROFILE + s * OBS_PROFILE_W;
    }
    if (brk == OBS_WIN_BREAK_WIN_PROFILE) base += OBS_PROFILE_W;   // THE BREAK: the next row
    if (*bad || base < 0 || base + OBS_PROFILE_W > OBS_TAB_W) {
        *bad = true;
        return d;
    }
    d.tmax = of_bits(ot[base + OBS_PROFILE_TEMPERATURE_MAX_BITS]);
    d.plo = of_bits(ot[base + OBS_PROFILE_PRESSURE_MIN_BITS]);
    d.phi = of_bits(ot[base + OBS_PROFILE_PRESSURE_MAX_BITS]);
    d.pd = of_bits(ot[OBS_TAB_POISON_MIN_DENSITY_BITS]);
    d.td = of_bits(ot[OBS_TAB_TEARGAS_BLIND_DENSITY_BITS]);
    d.sd = of_bits(ot[OBS_TAB_SMOKE_DENSE_BITS]);
    if (u[UNIT_COL_IS_ZOMBIE]) {
        d.resist = of_bits(ot[OBS_TAB_ZOMBIE_RESIST_POISON_BITS]);
    } else {
        if (s < 0 || s >= OBS_TAB_N_SPECIES) {
            *bad = true;
            return d;
        }
        d.resist = of_bits(ot[OBS_TAB_SPECIES_RESIST_POISON + s]);
    }
    d.wd = mul((double)u[UNIT_COL_MASS_G], (double)ot[OBS_TAB_CAP_Q]);
    return d;
}

// physics_value: the stored ratio before the squash (_div: 0 when the table
// value is 0).
__device__ inline double phys(int k, double raw, const Den& d) {
    switch (k) {
        case P_TEMP:  return d.tmax == 0.0 ? 0.0 : div(raw, d.tmax);
        case P_FIRE:  return raw;
        case P_POISON: return mul(d.pd == 0.0 ? 0.0 : div(raw, d.pd), d.resist);
        case P_TEAR:  return d.td == 0.0 ? 0.0 : div(raw, d.td);
        case P_SMOKE: return d.sd == 0.0 ? 0.0 : div(raw, d.sd);
        case P_PRESS: {
            const double band = sub(d.phi, d.plo);
            return band == 0.0 ? 0.0 : div(sub(raw, d.plo), band);
        }
        default:      return d.wd == 0.0 ? 0.0 : div(raw, d.wd);
    }
}

__device__ __forceinline__ uint16_t half_of(double x, int brk) {
    // THE BREAK win_half: through float32 first (double rounding), never the kit
    return brk == OBS_WIN_BREAK_WIN_HALF ? to_half_bits((double)to_f32(x)) : to_half_bits(x);
}

// The soldier of row r of env e: its store row (nullptr when the row is not
// valid: no unit, not alive, not on team r / S).
__device__ __forceinline__ const int64_t* row_unit(const int64_t* store, int U, const int32_t* rowmap,
                                                   int e, int n_rows, int r, int n_slots) {
    using namespace unit_store;
    const int sr = rowmap[(size_t)e * n_rows + r];
    if (sr < 0 || sr >= U) return nullptr;
    const int64_t* u = store + ((size_t)e * U + sr) * UNIT_STORE_W;
    if (u[UNIT_COL_LIFE_STATE] != OBS_LIFE_ALIVE || u[UNIT_COL_TEAM] != (int64_t)(r / n_slots))
        return nullptr;
    return u;
}

// FieldPlanes.far_raw: one thread per (env, block).
__global__ void k_far_raw(const int32_t* temp, const int32_t* fire, const int32_t* gas, int n_gases,
                          const int32_t* atm, const int32_t* wx, const int32_t* wy,
                          const uint8_t* solid, int h, int w, int hb, int wb, const int64_t* otab,
                          const int64_t* trow, const int64_t* params, const uint8_t* ended,
                          double* far_raw, int brk) {
    const int e = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hb * wb || !env_due(e, trow, params, ended)) return;
    const int bi = idx / wb, bj = idx % wb;
    const Planes P = planes_of(e, temp, fire, gas, n_gases, atm, wx, wy, solid, h, w,
                               otab + (size_t)e * OBS_TAB_W);
    double best[NP];
    double best_load = 0.0;
    for (int a = 0; a < OBS_BLOCK; ++a) {
        for (int b = 0; b < OBS_BLOCK; ++b) {
            const int t = a * OBS_BLOCK + b;               // row-major within the block
            const int64_t ty = (int64_t)bi * OBS_BLOCK + a, tx = (int64_t)bj * OBS_BLOCK + b;
            double raw[NP];
            double load = 0.0;
            if (ty < h && tx < w) {
                load = raw8(P, (size_t)ty * w + tx, raw);
            } else {
                for (int p = 0; p < NP; ++p) raw[p] = 0.0;  // blocks(raw, 0.0): the pad
            }
            for (int p = 0; p < P_WX; ++p)
                best[p] = (t == 0 || raw[p] > best[p]) ? raw[p] : best[p];
            // numpy's argmax: the FIRST largest load (THE BREAK win_argmax: the last)
            const bool take = t == 0 || load > best_load ||
                              (brk == OBS_WIN_BREAK_WIN_ARGMAX && load == best_load);
            if (take) {
                best_load = load;
                best[P_WX] = raw[P_WX];
                best[P_WY] = raw[P_WY];
            }
        }
    }
    const size_t plane = (size_t)hb * wb;
    for (int p = 0; p < NP; ++p) far_raw[((size_t)e * NP + p) * plane + idx] = best[p];
}

// One thread per (env, row, window cell): the eight physics planes of the
// near window at the cell and of the far window at the cell's block.
__global__ void k_planes(const int64_t* store, int U, const int32_t* rowmap, int n_rows,
                         int n_slots, const int64_t* otab, const int32_t* temp,
                         const int32_t* fire, const int32_t* gas, int n_gases, const int32_t* atm,
                         const int32_t* wx, const int32_t* wy, const uint8_t* solid, int h, int w,
                         const double* far_raw, int hb, int wb, const int64_t* trow,
                         const int64_t* params, const uint8_t* ended, uint16_t* near_f,
                         uint16_t* far_f, int32_t* faults, int brk) {
    const int e = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * CELLS || !env_due(e, trow, params, ended)) return;
    const int r = idx / CELLS, cell = idx % CELLS;
    const int i = cell / OBS_WIN, j = cell % OBS_WIN;
    const size_t er = (size_t)e * n_rows + r;
    uint16_t* nf = near_f + er * OBS_NEAR_F_PLANES * CELLS + cell;
    uint16_t* ff = far_f + er * OBS_FAR_F_PLANES * CELLS + OBS_FAR_F_PHYSICS0 * CELLS + cell;
    const int64_t* u = row_unit(store, U, rowmap, e, n_rows, r, n_slots);
    if (!u) {
        for (int p = 0; p < NP; ++p) {
            nf[p * CELLS] = 0;
            ff[p * CELLS] = 0;
        }
        return;
    }
    const int64_t* ot = otab + (size_t)e * OBS_TAB_W;
    bool bad = false;
    const Den d = den_of(u, ot, brk, &bad);
    if (bad) atomicOr(faults + e, OBS_WIN_FAULT_PROFILE);
    const double cap = of_bits(OBS_SQUASH_CAP_BITS);
    const int64_t cx = obs_sight::centre_x(u), cy = obs_sight::centre_y(u);
    // -- near (3.1, 3.2) --
    const int64_t y = cy - OBS_HALF + i, x = cx - OBS_HALF + j;
    if (0 <= y && y < h && 0 <= x && x < w) {
        const Planes P = planes_of(e, temp, fire, gas, n_gases, atm, wx, wy, solid, h, w, ot);
        double raw[NP];
        raw8(P, (size_t)y * w + x, raw);
        for (int p = 0; p < NP; ++p) nf[p * CELLS] = half_of(squash(phys(p, raw[p], d), cap), brk);
    } else {
        for (int p = 0; p < NP; ++p) nf[p * CELLS] = 0;
    }
    // -- far (3.3): the block grid anchored at tile (0, 0) --
    int64_t bi = floordiv(cy, OBS_BLOCK) - OBS_HALF + i;
    const int64_t bj = floordiv(cx, OBS_BLOCK) - OBS_HALF + j;
    if (brk == OBS_WIN_BREAK_WIN_ANCHOR) bi += 1;            // THE BREAK: the anchor one block off
    if (0 <= bi && bi < hb && 0 <= bj && bj < wb) {
        const size_t plane = (size_t)hb * wb;
        const double* fr = far_raw + (size_t)e * NP * plane + (size_t)bi * wb + bj;
        for (int p = 0; p < NP; ++p)
            ff[p * CELLS] = half_of(squash(phys(p, fr[p * plane], d), cap), brk);
    } else {
        for (int p = 0; p < NP; ++p) ff[p * CELLS] = 0;
    }
}

// The recorded centre of the entry the decoder aimed at (build_list's 1h
// tail): the first entry of the unit blocks with the aimed uid; none when it
// is a mate's place, not in the list, or the team holds no record of it.
__device__ inline bool aimed_centre(const int64_t* rows, int U, const int64_t* blk,
                                    const PicLayout& L, int t, int32_t aimed, const int32_t* lu,
                                    const uint8_t* lm, int n_entries, int n_slots, int64_t* ax,
                                    int64_t* ay) {
    using namespace unit_store;
    if (aimed < 0) return false;
    const int caps0 = n_slots - 1;
    int units_end = caps0 + OBS_CAP_SEEN + OBS_CAP_RECORD;
    if (units_end > n_entries) units_end = n_entries;
    int k = -1;
    for (int q = 0; q < units_end; ++q)
        if (lm[q] && lu[q] == aimed) {
            k = q;
            break;
        }
    if (k < caps0) return false;                          // not found, or a mate's place
    for (int q = 0; q < U; ++q) {
        const int64_t* m = rows + (size_t)q * UNIT_STORE_W;
        if (!m[UNIT_COL_VALID] || m[UNIT_COL_ID] != (int64_t)aimed) continue;
        const int64_t* rc = obs_pic::rec_of(blk, L, U, t, q);
        if (!rc[OBS_PIC_REC_PRESENT]) return false;
        *ax = rc[OBS_PIC_REC_CENTRE_X];
        *ay = rc[OBS_PIC_REC_CENTRE_Y];
        return true;
    }
    return false;
}

// One thread per (env, row, window cell): watched, sees_target (near_u) and
// the far unit planes (far_u mates / enemies / records) at the cell.
__global__ void k_sight_units(const int64_t* store, int U, const int32_t* rowmap, int n_rows,
                              int n_slots, int T, const uint8_t* solid, const uint8_t* screen,
                              int h, int w, const uint8_t* visible, const int64_t* state, int W,
                              const int32_t* aimed, const int32_t* list_uid,
                              const uint8_t* list_mask, int n_entries, const int64_t* trow,
                              const int64_t* params, const uint8_t* ended, uint8_t* near_u,
                              uint8_t* far_u, int32_t* faults, int brk) {
    using namespace unit_store;
    const int e = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * CELLS || !env_due(e, trow, params, ended)) return;
    const int r = idx / CELLS, cell = idx % CELLS;
    const int i = cell / OBS_WIN, j = cell % OBS_WIN;
    const size_t er = (size_t)e * n_rows + r;
    uint8_t* nu = near_u + er * OBS_NEAR_U_PLANES * CELLS + cell;
    uint8_t* fu = far_u + er * OBS_FAR_U_PLANES * CELLS + cell;
    const int64_t* u = row_unit(store, U, rowmap, e, n_rows, r, n_slots);
    if (!u) {
        nu[OBS_NEAR_U_WATCHED * CELLS] = 0;
        nu[OBS_NEAR_U_SEES_TARGET * CELLS] = 0;
        fu[OBS_FAR_U_MATES * CELLS] = 0;
        fu[OBS_FAR_U_ENEMIES * CELLS] = 0;
        fu[OBS_FAR_U_RECORDS * CELLS] = 0;
        return;
    }
    const int t = r / n_slots;
    const int64_t* rows = store + (size_t)e * U * UNIT_STORE_W;
    const PicLayout L = pic_layout(T, U);
    const int64_t* blk = state + (size_t)e * W;
    const uint8_t* vis_t = visible + ((size_t)e * (T + 1) + t) * U;
    const uint8_t* so = solid + (size_t)e * h * w;
    const uint8_t* sc = screen + (size_t)e * h * w;
    const int64_t cx = obs_sight::centre_x(u), cy = obs_sight::centre_y(u);
    const int64_t y0 = cy - OBS_HALF, x0 = cx - OBS_HALF;
    bool bad = false;
    // -- watched (3.5): from every record's centre of the team, OR-ed --
    bool watched = false;
    for (int q = 0; q < U && !watched; ++q) {
        const int64_t* rc = obs_pic::rec_of(blk, L, U, t, q);
        if (!rc[OBS_PIC_REC_PRESENT]) continue;
        watched = obs_sight::window_cell(so, sc, h, w, false, false, rc[OBS_PIC_REC_CENTRE_Y],
                                         rc[OBS_PIC_REC_CENTRE_X], y0, x0, OBS_WIN, cell, &bad);
    }
    nu[OBS_NEAR_U_WATCHED * CELLS] = watched ? 1 : 0;
    // -- sees_target (3.5, 1h): from the cell to the aimed entry's centre --
    int64_t ax, ay;
    bool sees = false;
    if (aimed_centre(rows, U, blk, L, t, aimed[er], list_uid + er * n_entries,
                     list_mask + er * n_entries, n_entries, n_slots, &ax, &ay))
        sees = obs_sight::window_cell(so, sc, h, w, false, true, ay, ax, y0, x0, OBS_WIN, cell,
                                      &bad);
    nu[OBS_NEAR_U_SEES_TARGET * CELLS] = sees ? 1 : 0;
    if (bad) atomicOr(faults + e, OBS_WIN_FAULT_SIGHT);
    // -- the far unit planes (3.7): counts per block of the far window --
    const int64_t by0 = floordiv(cy, OBS_BLOCK) - OBS_HALF, bx0 = floordiv(cx, OBS_BLOCK) - OBS_HALF;
    (void)brk;
    int mates = 0, enemies = 0, records = 0;
    const int64_t me = u[UNIT_COL_ID];
    for (int q = 0; q < U; ++q) {
        const int64_t* m = rows + (size_t)q * UNIT_STORE_W;
        if (is_mate(m, t) && m[UNIT_COL_ID] != me &&
            floordiv(obs_sight::centre_y(m), OBS_BLOCK) - by0 == i &&
            floordiv(obs_sight::centre_x(m), OBS_BLOCK) - bx0 == j)
            ++mates;
        const int64_t* rc = obs_pic::rec_of(blk, L, U, t, q);
        if (rc[OBS_PIC_REC_PRESENT] &&
            floordiv(rc[OBS_PIC_REC_CENTRE_Y], OBS_BLOCK) - by0 == i &&
            floordiv(rc[OBS_PIC_REC_CENTRE_X], OBS_BLOCK) - bx0 == j) {
            if (is_seen(vis_t, blk, L, U, t, q))
                ++enemies;
            else
                ++records;
        }
    }
    fu[OBS_FAR_U_MATES * CELLS] = (uint8_t)mates;
    fu[OBS_FAR_U_ENEMIES * CELLS] = (uint8_t)enemies;
    fu[OBS_FAR_U_RECORDS * CELLS] = (uint8_t)records;
}

// One thread per (env, row): the vector's air block (the raw planes at the
// soldier's centre tile, the same law, cast to float32; 0 off the map) and
// felt_temp (the unit library's felt temperature over the profile's edge).
__global__ void k_air(const int64_t* store, int U, const int64_t* tables, const int32_t* rowmap,
                      int n_rows, int n_slots, const int64_t* otab, const int32_t* temp,
                      const int32_t* fire, const int32_t* gas, int n_gases, const int32_t* atm,
                      const int32_t* wx, const int32_t* wy, const uint8_t* solid,
                      const int32_t* heat, const int64_t* rad, int h, int w, const int64_t* trow,
                      const int64_t* params, const uint8_t* ended, float* vector, int32_t* faults,
                      int brk) {
    const int e = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows || !env_due(e, trow, params, ended)) return;
    const size_t er = (size_t)e * n_rows + r;
    float* v = vector + er * OBS_V;
    for (int p = 0; p < NP; ++p) v[OBS_V_AIR_0 + p] = 0.0f;
    v[OBS_V_FELT_TEMP] = 0.0f;
    const int64_t* u = row_unit(store, U, rowmap, e, n_rows, r, n_slots);
    if (!u) return;
    const int64_t* ot = otab + (size_t)e * OBS_TAB_W;
    bool bad = false;
    const Den d = den_of(u, ot, brk, &bad);
    if (bad) atomicOr(faults + e, OBS_WIN_FAULT_PROFILE);
    const double cap = of_bits(OBS_SQUASH_CAP_BITS);
    const int64_t cx = obs_sight::centre_x(u), cy = obs_sight::centre_y(u);
    if (0 <= cy && cy < h && 0 <= cx && cx < w) {
        const Planes P = planes_of(e, temp, fire, gas, n_gases, atm, wx, wy, solid, h, w, ot);
        double raw[NP];
        raw8(P, (size_t)cy * w + cx, raw);
        for (int p = 0; p < NP; ++p) v[OBS_V_AIR_0 + p] = to_f32(squash(phys(p, raw[p], d), cap));
    }
    const int64_t* tab = tables + (size_t)e * unit_tables::UNIT_TABLES_W;
    const size_t hw = (size_t)h * w;
    bool fbad = false;
    const unit_lib::FeltTemp ft =
        unit_lib::felt_temperature_q(u, tab, heat + e * hw, rad + e * hw, h, w, &fbad);
    if (fbad) atomicOr(faults + e, OBS_WIN_FAULT_FELT);
    const double t_felt = div((double)ft.t_felt_q, (double)OBS_FP_ONE);
    v[OBS_V_FELT_TEMP] = to_f32(squash(d.tmax == 0.0 ? 0.0 : div(t_felt, d.tmax), cap));
}

void need(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("obs_planes: ") + what);
}

}  // namespace

void obs_planes(int n_env, std::uintptr_t store, int n_units, int store_w,
                const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
                int n_teams, std::uintptr_t obs_table, int table_w, std::uintptr_t tick_row,
                std::uintptr_t params, std::uintptr_t ended, std::uintptr_t temperature,
                std::uintptr_t fire, std::uintptr_t gas, int n_gases, std::uintptr_t atmosphere,
                std::uintptr_t wind_x, std::uintptr_t wind_y, std::uintptr_t heat,
                std::uintptr_t rad, std::uintptr_t solid, std::uintptr_t screen, int h, int w,
                std::uintptr_t far_raw, int hb, int wb, std::uintptr_t visible,
                std::uintptr_t state, int state_w, std::uintptr_t aimed, std::uintptr_t list_uid,
                std::uintptr_t list_mask, int n_entries, std::uintptr_t near_f,
                std::uintptr_t near_u, std::uintptr_t far_f, std::uintptr_t far_u,
                std::uintptr_t vector, std::uintptr_t faults, int brk) {
    need(n_env >= 1 && n_units >= 1 && n_rows >= 1 && n_slots >= 1 && n_teams >= 1 && h >= 1 &&
         w >= 1 && n_entries >= 1, "a non-positive size");
    need(n_rows == n_teams * n_slots, "n_rows is not n_teams x n_slots");
    need(store_w == unit_store::UNIT_STORE_W, "the store's width is not UNIT_STORE_W");
    need(table_w == OBS_TAB_W, "the obs table's width is not OBS_TAB_W (regenerate obs_layout.h)");
    need(state_w == pic_layout(n_teams, n_units).w, "the picture block's width");
    need(tables.n_env() == n_env && tables.width() == unit_tables::UNIT_TABLES_W,
         "the device tables' shape");
    need(hb == (h + OBS_BLOCK - 1) / OBS_BLOCK && wb == (w + OBS_BLOCK - 1) / OBS_BLOCK,
         "the block grid's shape");
    need(n_gases > OBS_GAS_POISON && n_gases > OBS_GAS_TEARGAS && n_gases > OBS_GAS_SMOKE &&
         n_gases > OBS_GAS_O2 && n_gases > OBS_GAS_N2, "the gas table lacks a row the windows read");
    need(n_entries >= n_slots - 1 + OBS_CAP_SEEN + OBS_CAP_RECORD, "the list's entries");
    need(store && rowmap && obs_table && tick_row && params && ended && temperature && fire &&
         gas && atmosphere && wind_x && wind_y && heat && rad && solid && screen && far_raw &&
         visible && state && aimed && list_uid && list_mask && near_f && near_u && far_f &&
         far_u && vector && faults, "a null device pointer");
    cudaStream_t s = tstream();
    const int64_t* st = reinterpret_cast<const int64_t*>(store);
    const int32_t* rm = reinterpret_cast<const int32_t*>(rowmap);
    const int64_t* ot = reinterpret_cast<const int64_t*>(obs_table);
    const int64_t* trow = reinterpret_cast<const int64_t*>(tick_row);
    const int64_t* prm = reinterpret_cast<const int64_t*>(params);
    const uint8_t* end = reinterpret_cast<const uint8_t*>(ended);
    const int32_t* tp = reinterpret_cast<const int32_t*>(temperature);
    const int32_t* fi = reinterpret_cast<const int32_t*>(fire);
    const int32_t* gs = reinterpret_cast<const int32_t*>(gas);
    const int32_t* at = reinterpret_cast<const int32_t*>(atmosphere);
    const int32_t* vx = reinterpret_cast<const int32_t*>(wind_x);
    const int32_t* vy = reinterpret_cast<const int32_t*>(wind_y);
    const uint8_t* so = reinterpret_cast<const uint8_t*>(solid);
    double* fr = reinterpret_cast<double*>(far_raw);
    int32_t* flt = reinterpret_cast<int32_t*>(faults);
    const int blocks = hb * wb;
    k_far_raw<<<dim3((blocks + 127) / 128, n_env), 128, 0, s>>>(
        tp, fi, gs, n_gases, at, vx, vy, so, h, w, hb, wb, ot, trow, prm, end, fr, brk);
    const int cells = n_rows * CELLS;
    k_planes<<<dim3((cells + 255) / 256, n_env), 256, 0, s>>>(
        st, n_units, rm, n_rows, n_slots, ot, tp, fi, gs, n_gases, at, vx, vy, so, h, w, fr, hb,
        wb, trow, prm, end, reinterpret_cast<uint16_t*>(near_f),
        reinterpret_cast<uint16_t*>(far_f), flt, brk);
    k_sight_units<<<dim3((cells + 255) / 256, n_env), 256, 0, s>>>(
        st, n_units, rm, n_rows, n_slots, n_teams, so, reinterpret_cast<const uint8_t*>(screen),
        h, w, reinterpret_cast<const uint8_t*>(visible), reinterpret_cast<const int64_t*>(state),
        state_w, reinterpret_cast<const int32_t*>(aimed),
        reinterpret_cast<const int32_t*>(list_uid), reinterpret_cast<const uint8_t*>(list_mask),
        n_entries, trow, prm, end, reinterpret_cast<uint8_t*>(near_u),
        reinterpret_cast<uint8_t*>(far_u), flt, brk);
    k_air<<<dim3((n_rows + 63) / 64, n_env), 64, 0, s>>>(
        st, n_units, tables.values(), rm, n_rows, n_slots, ot, tp, fi, gs, n_gases, at, vx, vy,
        so, reinterpret_cast<const int32_t*>(heat), reinterpret_cast<const int64_t*>(rad), h, w,
        trow, prm, end, reinterpret_cast<float*>(vector), flt, brk);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("obs_planes: launch failed: ") +
                                 cudaGetErrorString(err));
}

}  // namespace breach_cuda
