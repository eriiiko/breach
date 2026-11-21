// ==========================================================================
// #29 P8.2: THE OBSERVATION'S SIGHT KERNELS (see cuda_obs.h), Class T.
//
//   obs_vision     who sees whom, every tick, every env in one launch: the
//                  device form of vision.compute's VisionState (seen_by,
//                  visible_by_team) under the match's sight rule, read from
//                  the obs table (never an argument)
//   obs_rays       vision.rays_clear in bulk: (N, P) pairs, one launch
//   obs_windows    vision.window_clear in bulk: (N, Q) windows, one launch
//   obs_lof_pairs  line_of_fire.solid_in_way over every (row, row) pair of
//                  the store (the cover table read), one launch
//
// Every line is cuda_obs_sight.cuh's, over the one device sight function
// (unit_sight::has_los); every float is the float kit's. Nothing here writes
// a synced buffer or syncs; every launch is on THE tick stream.
// ==========================================================================
#include "cuda_obs.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "cuda_batch.cuh"          // tstream(): THE tick stream (blocking)
#include "cuda_obs_float.cuh"
#include "cuda_obs_sight.cuh"
#include "obs_layout.h"
#include "unit_store_layout.h"

namespace breach_cuda {

using namespace obs_layout;

namespace {

void need(bool ok, const char* who, const char* what) {
    if (!ok) throw std::invalid_argument(std::string(who) + ": " + what);
}

void launched(const char* who) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(who) + ": launch failed: " + cudaGetErrorString(err));
}

// The visible-set slot of a team value: a marine team t < n_teams is slot t,
// the zombie team is slot n_teams; anything else is a fault (-1).
__device__ __forceinline__ int team_slot(int64_t team, int n_teams) {
    if (team >= 0 && team < n_teams) return (int)team;
    if (team == OBS_ZOMBIE_TEAM) return n_teams;
    return -1;
}

// One thread per (env, observer row o, target row t): sees[e][o][t] is
// "o is among seen_by[t]" (vision.compute_pairwise / _compute_all_round:
// target and observer living, of different teams, then can_see), and a seen
// pair marks visible[e][slot(team of o)][t] (the union of the team's cones;
// every writer writes 1).
__global__ void k_vision(const int64_t* store, int n_units, const uint8_t* solid,
                         const uint8_t* screen, int h, int w, const int64_t* otab,
                         int n_teams, uint8_t* sees, uint8_t* visible, int32_t* faults) {
    const int e = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_units * n_units) return;
    const int o = idx / n_units, t = idx % n_units;
    const int64_t* rows = store + (size_t)e * n_units * unit_store::UNIT_STORE_W;
    const int64_t* uo = rows + (size_t)o * unit_store::UNIT_STORE_W;
    const int64_t* ut = rows + (size_t)t * unit_store::UNIT_STORE_W;
    using namespace unit_store;
    bool seen = false;
    if (o != t && uo[UNIT_COL_VALID] && ut[UNIT_COL_VALID] &&
        uo[UNIT_COL_LIFE_STATE] == OBS_LIFE_ALIVE && ut[UNIT_COL_LIFE_STATE] == OBS_LIFE_ALIVE &&
        uo[UNIT_COL_TEAM] != ut[UNIT_COL_TEAM]) {
        const int64_t* ot = otab + (size_t)e * OBS_TAB_W;
        bool bad = false;
        seen = obs_sight::can_see(uo, ut, ot[OBS_TAB_SIGHT_RULE],
                                  obs_f64::of_bits(ot[OBS_TAB_CONE_COS_BITS]),
                                  obs_f64::of_bits(ot[OBS_TAB_AWARENESS_R_BITS]),
                                  solid + (size_t)e * h * w, screen + (size_t)e * h * w, h, w,
                                  &bad);
        if (bad) {
            atomicOr(faults + e, 1);
            seen = false;
        }
        if (seen) {
            const int s = team_slot(uo[UNIT_COL_TEAM], n_teams);
            if (s < 0) {
                atomicOr(faults + e, 2);
            } else {
                visible[((size_t)e * (n_teams + 1) + s) * n_units + t] = 1;
            }
        }
    }
    sees[((size_t)e * n_units + o) * n_units + t] = seen ? 1 : 0;
}

__global__ void k_rays(const uint8_t* solid, const uint8_t* screen, int h, int w, int physical,
                       const int32_t* pairs, int n_pairs, uint8_t* out, int32_t* faults) {
    const int e = blockIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_pairs) return;
    const int32_t* p = pairs + ((size_t)e * n_pairs + k) * 4;
    bool bad = false;
    const bool ok = obs_sight::line_clear(solid + (size_t)e * h * w, screen + (size_t)e * h * w,
                                          h, w, physical != 0, p[0], p[1], p[2], p[3], &bad);
    if (bad) atomicOr(faults + e, 1);
    out[(size_t)e * n_pairs + k] = (ok && !bad) ? 1 : 0;
}

__global__ void k_windows(const uint8_t* solid, const uint8_t* screen, int h, int w,
                          int physical, int to_window, const int32_t* queries, int n_q, int wh,
                          int ww, uint8_t* out, int32_t* faults) {
    const int e = blockIdx.y;
    const int64_t cells = (int64_t)wh * ww;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)n_q * cells) return;
    const int64_t q = idx / cells, k = idx % cells;
    const int32_t* p = queries + ((size_t)e * n_q + q) * 4;
    bool bad = false;
    const bool ok = obs_sight::window_cell(solid + (size_t)e * h * w, screen + (size_t)e * h * w,
                                           h, w, physical != 0, to_window != 0, p[0], p[1], p[2],
                                           p[3], ww, k, &bad);
    if (bad) atomicOr(faults + e, 1);
    out[((size_t)e * n_q + q) * cells + k] = (ok && !bad) ? 1 : 0;
}

__global__ void k_lof_pairs(const int64_t* store, int n_units, const uint8_t* solid, int h,
                            int w, const int64_t* covers, int cover_cap,
                            const int32_t* cover_count, uint8_t* out, int32_t* faults) {
    const int e = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_units * n_units) return;
    const int a = idx / n_units, b = idx % n_units;
    const int64_t* rows = store + (size_t)e * n_units * unit_store::UNIT_STORE_W;
    const int64_t* ua = rows + (size_t)a * unit_store::UNIT_STORE_W;
    const int64_t* ub = rows + (size_t)b * unit_store::UNIT_STORE_W;
    bool blocked = false;
    if (ua[unit_store::UNIT_COL_VALID] && ub[unit_store::UNIT_COL_VALID]) {
        bool bad = false;
        blocked = obs_sight::solid_in_way(
            solid + (size_t)e * h * w, h, w, covers + (size_t)e * cover_cap * OBS_COVER_W,
            cover_count[e], obs_sight::centre_x(ua), obs_sight::centre_y(ua),
            obs_sight::centre_x(ub), obs_sight::centre_y(ub), &bad);
        if (bad) atomicOr(faults + e, 1);
    }
    out[((size_t)e * n_units + a) * n_units + b] = blocked ? 1 : 0;
}

}  // namespace

void obs_vision(int n_env, std::uintptr_t store, int n_units, int store_w, std::uintptr_t solid,
                std::uintptr_t screen, int h, int w, std::uintptr_t obs_table, int table_w,
                int n_teams, std::uintptr_t sees, std::uintptr_t visible,
                std::uintptr_t faults) {
    const char* who = "obs_vision";
    need(n_env >= 1 && n_units >= 1 && h >= 1 && w >= 1 && n_teams >= 1, who,
         "a non-positive size");
    need(store_w == unit_store::UNIT_STORE_W, who, "the store's width is not UNIT_STORE_W");
    need(table_w == OBS_TAB_W, who, "the obs table's width is not OBS_TAB_W (regenerate obs_layout.h)");
    need(store && solid && screen && obs_table && sees && visible && faults, who,
         "a null device pointer");
    cudaStream_t s = tstream();
    cudaMemsetAsync(reinterpret_cast<void*>(visible), 0,
                    (size_t)n_env * (n_teams + 1) * n_units, s);
    const int pairs = n_units * n_units;
    k_vision<<<dim3((pairs + 255) / 256, n_env), 256, 0, s>>>(
        reinterpret_cast<const int64_t*>(store), n_units, reinterpret_cast<const uint8_t*>(solid),
        reinterpret_cast<const uint8_t*>(screen), h, w,
        reinterpret_cast<const int64_t*>(obs_table), n_teams, reinterpret_cast<uint8_t*>(sees),
        reinterpret_cast<uint8_t*>(visible), reinterpret_cast<int32_t*>(faults));
    launched(who);
}

void obs_rays(int n_env, std::uintptr_t solid, std::uintptr_t screen, int h, int w,
              bool physical, std::uintptr_t pairs, int n_pairs, std::uintptr_t out,
              std::uintptr_t faults) {
    const char* who = "obs_rays";
    need(n_env >= 1 && h >= 1 && w >= 1 && n_pairs >= 0, who, "a non-positive size");
    need(solid && screen && pairs && out && faults, who, "a null device pointer");
    if (n_pairs == 0) return;
    cudaStream_t s = tstream();
    k_rays<<<dim3((n_pairs + 255) / 256, n_env), 256, 0, s>>>(
        reinterpret_cast<const uint8_t*>(solid), reinterpret_cast<const uint8_t*>(screen), h, w,
        physical ? 1 : 0, reinterpret_cast<const int32_t*>(pairs), n_pairs,
        reinterpret_cast<uint8_t*>(out), reinterpret_cast<int32_t*>(faults));
    launched(who);
}

void obs_windows(int n_env, std::uintptr_t solid, std::uintptr_t screen, int h, int w,
                 bool physical, bool to_window, std::uintptr_t queries, int n_q, int wh, int ww,
                 std::uintptr_t out, std::uintptr_t faults) {
    const char* who = "obs_windows";
    need(n_env >= 1 && h >= 1 && w >= 1 && n_q >= 0 && wh >= 1 && ww >= 1, who,
         "a non-positive size");
    need(solid && screen && queries && out && faults, who, "a null device pointer");
    if (n_q == 0) return;
    cudaStream_t s = tstream();
    const int64_t n = (int64_t)n_q * wh * ww;
    need(n < ((int64_t)1 << 31), who, "too many window cells for one launch");
    k_windows<<<dim3((unsigned)((n + 255) / 256), n_env), 256, 0, s>>>(
        reinterpret_cast<const uint8_t*>(solid), reinterpret_cast<const uint8_t*>(screen), h, w,
        physical ? 1 : 0, to_window ? 1 : 0, reinterpret_cast<const int32_t*>(queries), n_q, wh,
        ww, reinterpret_cast<uint8_t*>(out), reinterpret_cast<int32_t*>(faults));
    launched(who);
}

void obs_lof_pairs(int n_env, std::uintptr_t store, int n_units, int store_w,
                   std::uintptr_t solid, int h, int w, std::uintptr_t covers, int cover_cap,
                   int cover_w, std::uintptr_t cover_count, std::uintptr_t out,
                   std::uintptr_t faults) {
    const char* who = "obs_lof_pairs";
    need(n_env >= 1 && n_units >= 1 && h >= 1 && w >= 1 && cover_cap >= 1, who,
         "a non-positive size");
    need(store_w == unit_store::UNIT_STORE_W, who, "the store's width is not UNIT_STORE_W");
    need(cover_w == OBS_COVER_W, who, "the cover table's width is not OBS_COVER_W");
    need(store && solid && covers && cover_count && out && faults, who, "a null device pointer");
    cudaStream_t s = tstream();
    const int pairs = n_units * n_units;
    k_lof_pairs<<<dim3((pairs + 255) / 256, n_env), 256, 0, s>>>(
        reinterpret_cast<const int64_t*>(store), n_units, reinterpret_cast<const uint8_t*>(solid),
        h, w, reinterpret_cast<const int64_t*>(covers), cover_cap,
        reinterpret_cast<const int32_t*>(cover_count), reinterpret_cast<uint8_t*>(out),
        reinterpret_cast<int32_t*>(faults));
    launched(who);
}

}  // namespace breach_cuda
