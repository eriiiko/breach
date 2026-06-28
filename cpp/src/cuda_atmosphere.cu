// ============================================================================
// CUDA-S7 diffuse_solve implementation — see cuda_atmosphere.h.
// A bit-identical GPU port of AtmosphereSolver::diffuse_solve (atmosphere_solver.cpp
// ~269-601): the implicit Red-Black Gauss-Seidel atmosphere diffusion (residual
// form, per-cell Dinv) + the vacuum BFS/sponge boundary pass + the wind gradient.
//
// Pass chain (separate launches = grid barriers between dependent passes):
//   [μ-gate]  HOST: if (mu_q > MU_EPS_Q) run K0 + GS; else skip them (atm unchanged)
//   K0        Dinv+RHS      rhs=atm (all cells); Dinv=reciprocal_q16(1+mu*wsum),
//                           sentinel 0 on obstacle|wall|vacuum
//   K_GS      RB sweep      for iter in gs_iters: launch RED then BLACK; each non-
//                           solid cell of the colour does the residual-form update
//   K_BFS1    vac dist 1    a non-seed non-solid cell -> 1 if any nbr seed (==0)
//   K_BFS2    vac dist 2    -> 2 if still 255 and any nbr ==1
//   K_SPONGE  sponge relax  per vac_dist tier: scale atm/wave_v, zero per the CPU cases
//   K_WIND    wind grad     wind = -shr_round0(grad(atm+wave_p), 1)
//
// Every per-cell op is a VERBATIM device transcription of the CPU loops — same
// integer ops, same branch structure. The per-face permeability weight is a device
// float bridge (quantize((double)min(perm_i,perm_n)), exactly like the CPU + S5's
// wave_lap; --fmad=false makes the double min/quantize bit-identical).
//
// THE GS CRUX: the Red-Black colour schedule is realised as TWO separate kernel
// launches per iteration (RED then BLACK), so a RED cell reads only the FROZEN
// black neighbours of the previous launch (the grid barrier between launches), and
// vice-versa — order-free by colour, identical on any architecture. The increment
// `inc = round_nearest_q_dev((int64)resi * (int64)dinv[i])` is sign-symmetric
// round-to-nearest (NOT mul_q16's toward-(-inf) truncation): a truncating slip
// biases every cell -1 LSB/sweep = a DC mass sink during the diffusion transient.
//
// THE BFS: double-buffered (read d_vac_in, write d_vac_out) so each pass reads only
// the prior frozen distance level — obviously race-free, order-independent.
//
// last_gs_residual (a non-synced host float diagnostic; nothing reads it) is NOT
// computed on the GPU path — the member is left as the CPU last set.
// ============================================================================
#include "cuda_atmosphere.h"
#include "fixed_point.h"   // q16, quantize, mul_q16, mul_wide, narrow, FP_ONE, FP_SHIFT
#include "cuda_fixedpoint_device.cuh"  // reciprocal_q16_dev, round_nearest_q_dev,
                                       // scale_mag_dev, shr_round0_dev (S7 §2)

#include <cuda_runtime.h>

#include <algorithm>   // std::min (host scalar precompute only)
#include <sstream>
#include <stdexcept>

using namespace fixedpoint;

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in diffuse_solve_gpu/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// ---- device per-face permeability weight (the perm bridge) ------------------
// quantize((double)min(perm_i, perm_n)), 0 if the face is OOB or the neighbour is
// an obstacle/wall (the Neumann/conservation guard — a one-sided flux into a
// zeroed Dirichlet cell would destroy mass; the S1/S3a fix, here integer). EXACTLY
// the CPU `face_q`/`muw` lambdas (atmosphere_solver.cpp ~345-349 / ~388-393). The
// double min + quantize is bit-identical under --fmad=false / /fp:strict.
__device__ __forceinline__ q16 face_perm_q(float perm_i, float perm_n,
                                           bool n_solid) {
    if (n_solid) return 0;                          // obstacle|wall neighbour -> 0
    const float fmin = perm_i < perm_n ? perm_i : perm_n;   // std::min
    return quantize((double)fmin);
}

// ---- K0: Dinv + RHS (atmosphere_solver.cpp ~310-368, cache DROPPED) ----------
// Per cell: rhs[i] = atmosphere[i] (ALL cells, matching the CPU's unconditional
// `for(i) rhs[i]=atmosphere[i]`). Then, for INTERIOR cells (not obstacle|wall|
// vacuum): wsum = Σ4 face_q; denom = FP_ONE + mul_q16(mu_q, wsum); dinv =
// reciprocal_q16_dev(denom). Sentinel dinv = 0 on excluded cells (the GS skips
// them). RECOMPUTED unconditionally — the CPU dinv_key_ cache is dropped (Dinv is
// a pure function of mu_q + masks + perm, so an unconditional recompute is
// identical and GPU-clean).
__global__ void atmos_dinv_rhs(const int32_t* __restrict__ atmosphere,
                               const bool* __restrict__ obstacles,
                               const bool* __restrict__ is_wall,
                               const bool* __restrict__ is_vacuum,
                               const float* __restrict__ perm,
                               int32_t* __restrict__ rhs,
                               int32_t* __restrict__ dinv,
                               int32_t mu_q, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        rhs[i] = atmosphere[i];                     // RHS snapshot (all cells)
        if (obstacles[i] || is_wall[i] || is_vacuum[i]) {
            dinv[i] = 0;                            // excluded sentinel
            continue;
        }
        const int y = i / w;
        const int x = i % w;
        const float perm_i = perm[i];
        q16 wsum = 0;
        if (y > 0)     { const int nb = (y-1)*w + x;
            wsum += face_perm_q(perm_i, perm[nb], obstacles[nb] || is_wall[nb]); }
        if (y < h-1)   { const int nb = (y+1)*w + x;
            wsum += face_perm_q(perm_i, perm[nb], obstacles[nb] || is_wall[nb]); }
        if (x > 0)     { const int nb = y*w + (x-1);
            wsum += face_perm_q(perm_i, perm[nb], obstacles[nb] || is_wall[nb]); }
        if (x < w-1)   { const int nb = y*w + (x+1);
            wsum += face_perm_q(perm_i, perm[nb], obstacles[nb] || is_wall[nb]); }
        const q16 mu_wsum = mul_q16(mu_q, wsum);
        const q16 denom_q = (q16)(FP_ONE + mu_wsum);   // >= FP_ONE
        dinv[i] = reciprocal_q16_dev(denom_q);         // 1/(1+mu*wsum), Q16.16
    }
}

// ---- K_GS: one colour of one Red-Black sweep (atmosphere_solver.cpp ~372-417) -
// THE CRUX. Over cells of the given colour ((x+y)&1 == color), skip obstacle|wall|
// vacuum; gather acc = Σ4 mul_wide(mul_q16(mu_q,face_q), atm[nb]-ai); flux=narrow;
// resi = flux - (ai - rhs[i]); inc = round_nearest((int64)resi*dinv); atm[i]=ai+inc.
// Launched as TWO separate kernels per iter (RED then BLACK) -> a RED cell reads
// only the FROZEN black neighbours of the previous launch (the grid barrier).
// In-place on the atmosphere buffer; reads the live (post-prior-colour) neighbours.
__global__ void atmos_gs_color(int32_t* __restrict__ atmosphere,
                               const int32_t* __restrict__ rhs,
                               const int32_t* __restrict__ dinv,
                               const bool* __restrict__ obstacles,
                               const bool* __restrict__ is_wall,
                               const bool* __restrict__ is_vacuum,
                               const float* __restrict__ perm,
                               int32_t mu_q, int color, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int y = i / w;
        const int x = i % w;
        if (((x + y) & 1) != color) continue;
        if (obstacles[i] || is_wall[i] || is_vacuum[i]) continue;

        const float perm_i = perm[i];
        const q16 ai = atmosphere[i];
        int64_t acc = 0;
        if (y > 0)   { const int nb = (y-1)*w + x;
            const q16 muw = mul_q16(mu_q,
                face_perm_q(perm_i, perm[nb], obstacles[nb] || is_wall[nb]));
            acc += mul_wide(muw, atmosphere[nb] - ai); }
        if (y < h-1) { const int nb = (y+1)*w + x;
            const q16 muw = mul_q16(mu_q,
                face_perm_q(perm_i, perm[nb], obstacles[nb] || is_wall[nb]));
            acc += mul_wide(muw, atmosphere[nb] - ai); }
        if (x > 0)   { const int nb = y*w + (x-1);
            const q16 muw = mul_q16(mu_q,
                face_perm_q(perm_i, perm[nb], obstacles[nb] || is_wall[nb]));
            acc += mul_wide(muw, atmosphere[nb] - ai); }
        if (x < w-1) { const int nb = y*w + (x+1);
            const q16 muw = mul_q16(mu_q,
                face_perm_q(perm_i, perm[nb], obstacles[nb] || is_wall[nb]));
            acc += mul_wide(muw, atmosphere[nb] - ai); }
        const q16 flux = narrow(acc);
        const q16 resi = (q16)(flux - (ai - rhs[i]));
        // increment = resi*Dinv, ROUND-TO-NEAREST sign-symmetric (NOT mul_q16's
        // toward-(-inf) truncation). resi*dinv fits int64 (no 128-bit needed).
        const int64_t inc_wide = (int64_t)resi * (int64_t)dinv[i];
        const q16 inc = round_nearest_q_dev(inc_wide);
        atmosphere[i] = ai + inc;
    }
}

// ---- K_BFS seed (atmosphere_solver.cpp ~476-481) ----------------------------
// vac_dist = 255 default; 0 where is_vacuum && !obstacle && !wall (exposed-vacuum
// breach seeds). Writes BOTH buffers so the first BFS pass can read d_vac_in.
__global__ void vac_seed(const bool* __restrict__ obstacles,
                         const bool* __restrict__ is_wall,
                         const bool* __restrict__ is_vacuum,
                         uint8_t* __restrict__ vac_a,
                         uint8_t* __restrict__ vac_b, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const uint8_t v = (is_vacuum[i] && !obstacles[i] && !is_wall[i])
                          ? (uint8_t)0 : (uint8_t)255;
        vac_a[i] = v;
        vac_b[i] = v;
    }
}

// ---- K_BFS pass (atmosphere_solver.cpp ~482-509, double-buffered) -----------
// For dist `target` (1 or 2): a non-solid cell whose CURRENT dist is still 255 and
// that has any 4-neighbour at `target-1` becomes `target`. Reads the FROZEN `vin`
// (the prior level), writes `vout`. The CPU guards are reproduced:
//   pass1: `if (vac_dist[i]==0 || obstacle|wall) continue;` -> a cell with dist 0
//          (a seed) keeps 0; only 255 cells can become 1.
//   pass2: `if (vac_dist[i]<=1 || obstacle|wall) continue;` -> 0 and 1 keep; only
//          255 cells can become 2.
// Carry forward the unchanged value otherwise (double-buffer => vout must hold the
// full updated grid). Order-free: every read is of the frozen `vin`.
__global__ void vac_bfs_pass(const uint8_t* __restrict__ vin,
                             uint8_t* __restrict__ vout,
                             const bool* __restrict__ obstacles,
                             const bool* __restrict__ is_wall,
                             uint8_t target, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const uint8_t cur = vin[i];
        uint8_t out = cur;                          // default: unchanged
        // CPU guard: skip seeds/closer cells (cur < target) and solids.
        if (cur >= target && !obstacles[i] && !is_wall[i]) {
            const int y = i / w;
            const int x = i % w;
            const uint8_t prev = (uint8_t)(target - 1);
            bool adj = false;
            if (y > 0   && vin[(y-1)*w + x] == prev) adj = true;
            if (y < h-1 && vin[(y+1)*w + x] == prev) adj = true;
            if (x > 0   && vin[y*w + (x-1)] == prev) adj = true;
            if (x < w-1 && vin[y*w + (x+1)] == prev) adj = true;
            if (adj) out = target;
        }
        vout[i] = out;
    }
}

// ---- K_SPONGE: vacuum/sponge BCs (atmosphere_solver.cpp ~527-548) ------------
// Per cell, by its final vac_dist tier (the CPU if/else-if ladder, SAME order):
//   dist==0          : atm=mul_q16(atm,vac_k); wave_p=0; wave_v=0
//   obstacle|wall    : wave_p=0; wave_v=0; atm=0
//   dist==1          : atm=mul_q16(atm,inner_k); wave_v=scale_mag(wave_v,wv_inner); wave_source=0
//   dist==2          : atm=mul_q16(atm,outer_k); wave_v=scale_mag(wave_v,wv_outer); wave_source=mul_q16(wave_source,ws_half)
// Per-cell -> order-free. The if/else order matters: dist==0 wins over the solid
// branch (a seed vacuum cell is also is_vacuum but its dist 0 branch runs first).
__global__ void atmos_sponge(int32_t* __restrict__ atmosphere,
                             int32_t* __restrict__ wave_p,
                             int32_t* __restrict__ wave_v,
                             int32_t* __restrict__ wave_source,
                             const uint8_t* __restrict__ vac_dist,
                             const bool* __restrict__ obstacles,
                             const bool* __restrict__ is_wall,
                             int32_t atm_vac_k_q, int32_t atm_inner_k_q,
                             int32_t atm_outer_k_q, int32_t wv_inner_k_q,
                             int32_t wv_outer_k_q, int32_t ws_half_q, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const uint8_t d = vac_dist[i];
        if (d == 0) {
            atmosphere[i] = mul_q16(atmosphere[i], atm_vac_k_q);
            wave_p[i] = 0;
            wave_v[i] = 0;
        } else if (obstacles[i] || is_wall[i]) {
            wave_p[i] = 0;
            wave_v[i] = 0;
            atmosphere[i] = 0;
        } else if (d == 1) {
            atmosphere[i] = mul_q16(atmosphere[i], atm_inner_k_q);
            wave_v[i] = scale_mag_dev(wave_v[i], wv_inner_k_q);
            wave_source[i] = 0;
        } else if (d == 2) {
            atmosphere[i] = mul_q16(atmosphere[i], atm_outer_k_q);
            wave_v[i] = scale_mag_dev(wave_v[i], wv_outer_k_q);
            wave_source[i] = mul_q16(wave_source[i], ws_half_q);
        }
    }
}

// ---- K_WIND: wind = -grad(atmosphere + wave_p) (atmosphere_solver.cpp ~551-596)
// wind=0 on obstacle|wall|vacuum. Else p_total(idx)=atm[idx]+wave_p[idx] (exact int
// sum); per-face f=quantize(min(perm)) (0 if OOB); p_side=p_here+mul_q16(f,total(nb)
// -p_here); wind_x=-shr_round0(p_right-p_left,1); wind_y=-shr_round0(p_down-p_up,1).
// Reads the POST-SPONGE atmosphere + wave_p (a separate launch = grid barrier).
__global__ void atmos_wind(const int32_t* __restrict__ atmosphere,
                           const int32_t* __restrict__ wave_p,
                           int32_t* __restrict__ wind_x,
                           int32_t* __restrict__ wind_y,
                           const bool* __restrict__ obstacles,
                           const bool* __restrict__ is_wall,
                           const bool* __restrict__ is_vacuum,
                           const float* __restrict__ perm, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (obstacles[i] || is_wall[i] || is_vacuum[i]) {
            wind_x[i] = 0;
            wind_y[i] = 0;
            continue;
        }
        const int y = i / w;
        const int x = i % w;
        const q16 p_here = atmosphere[i] + wave_p[i];   // total(i), exact int sum
        const float perm_i = perm[i];

        const int il = (x > 0)   ? y*w + (x-1) : i;
        const int ir = (x < w-1) ? y*w + (x+1) : i;
        const int iu = (y > 0)   ? (y-1)*w + x : i;
        const int id = (y < h-1) ? (y+1)*w + x : i;
        // face = quantize(min(perm)); 0 at the OOB edge (CPU passes 0 there).
        const q16 f_left  = (x > 0)   ? quantize((double)(perm_i < perm[il] ? perm_i : perm[il])) : 0;
        const q16 f_right = (x < w-1) ? quantize((double)(perm_i < perm[ir] ? perm_i : perm[ir])) : 0;
        const q16 f_up    = (y > 0)   ? quantize((double)(perm_i < perm[iu] ? perm_i : perm[iu])) : 0;
        const q16 f_down  = (y < h-1) ? quantize((double)(perm_i < perm[id] ? perm_i : perm[id])) : 0;

        const q16 t_left  = atmosphere[il] + wave_p[il];
        const q16 t_right = atmosphere[ir] + wave_p[ir];
        const q16 t_up    = atmosphere[iu] + wave_p[iu];
        const q16 t_down  = atmosphere[id] + wave_p[id];

        const q16 p_left  = (q16)(p_here + mul_q16(f_left,  t_left  - p_here));
        const q16 p_right = (q16)(p_here + mul_q16(f_right, t_right - p_here));
        const q16 p_up    = (q16)(p_here + mul_q16(f_up,    t_up    - p_here));
        const q16 p_down  = (q16)(p_here + mul_q16(f_down,  t_down  - p_here));

        wind_x[i] = -shr_round0_dev((q16)(p_right - p_left), 1);
        wind_y[i] = -shr_round0_dev((q16)(p_down  - p_up),   1);
    }
}

}  // namespace

void diffuse_solve_gpu(
    int32_t* atmosphere, int32_t* wave_p, int32_t* wave_v, int32_t* wave_source,
    int32_t* wind_x, int32_t* wind_y,
    const bool* obstacles, const bool* is_wall, const bool* is_vacuum,
    const float* permeability,
    int h, int w, float dt,
    float d_atm, float breach_rate, int gs_iters) {
    const int n = h * w;
    if (n <= 0) return;

    // ---- Host scalar precompute (atmosphere_solver.cpp ~294-296 + ~511-525,
    //      VERBATIM, in double). These are the same constants the CPU folds once. -
    const double dt_d  = (double)dt;
    const double mu_d  = (double)d_atm * dt_d;
    const q16    mu_q  = quantize(mu_d);                 // mu in Q16.16
    const q16 MU_EPS_Q = 655;                            // ~0.01 in Q16.16 (mu>1e-8 gate)

    const double eta_d = (double)std::min(breach_rate * dt, 1.0f);
    const q16 atm_vac_k_q   = quantize(1.0 - eta_d);          // vacuum: (1-eta)
    const q16 atm_inner_k_q = quantize(1.0 - eta_d * 0.5);    // inner: (1-eta*0.5)
    const q16 atm_outer_k_q = quantize(1.0 - eta_d * 0.25);   // outer: (1-eta*0.25)
    const q16 wv_inner_k_q  = quantize(1.0 - (double)std::min(30.0f * dt, 1.0f));
    const q16 wv_outer_k_q  = quantize(1.0 - (double)std::min(15.0f * dt, 1.0f));
    const q16 ws_half_q     = quantize(0.5);

    // ---- Device buffers: the 6 mutated fields + masks + perm + scratch (d_rhs,
    //      d_dinv, the BFS double-buffer d_vac_a/d_vac_b). Per-call H2D/D2H;
    //      residency is S8. -----------------------------------------------------
    const size_t nb    = (size_t)n * sizeof(int32_t);
    const size_t nbool = (size_t)n * sizeof(bool);
    const size_t nbf   = (size_t)n * sizeof(float);
    const size_t nbu8  = (size_t)n * sizeof(uint8_t);

    int32_t *d_atmf = nullptr, *d_wp = nullptr, *d_wv = nullptr, *d_ws = nullptr,
            *d_wx = nullptr, *d_wy = nullptr, *d_rhs = nullptr, *d_dinv = nullptr;
    bool *d_obs = nullptr, *d_wall = nullptr, *d_vac = nullptr;
    float *d_perm = nullptr;
    uint8_t *d_vac_a = nullptr, *d_vac_b = nullptr;

    cuda_check(cudaMalloc(&d_atmf, nb), "malloc atmosphere");
    cuda_check(cudaMalloc(&d_wp, nb), "malloc wave_p");
    cuda_check(cudaMalloc(&d_wv, nb), "malloc wave_v");
    cuda_check(cudaMalloc(&d_ws, nb), "malloc wave_source");
    cuda_check(cudaMalloc(&d_wx, nb), "malloc wind_x");
    cuda_check(cudaMalloc(&d_wy, nb), "malloc wind_y");
    cuda_check(cudaMalloc(&d_rhs, nb), "malloc rhs");
    cuda_check(cudaMalloc(&d_dinv, nb), "malloc dinv");
    cuda_check(cudaMalloc(&d_obs, nbool), "malloc obstacles");
    cuda_check(cudaMalloc(&d_wall, nbool), "malloc is_wall");
    cuda_check(cudaMalloc(&d_vac, nbool), "malloc is_vacuum");
    cuda_check(cudaMalloc(&d_perm, nbf), "malloc permeability");
    cuda_check(cudaMalloc(&d_vac_a, nbu8), "malloc vac_a");
    cuda_check(cudaMalloc(&d_vac_b, nbu8), "malloc vac_b");

    cuda_check(cudaMemcpy(d_atmf, atmosphere, nb, cudaMemcpyHostToDevice), "H2D atmosphere");
    cuda_check(cudaMemcpy(d_wp, wave_p, nb, cudaMemcpyHostToDevice), "H2D wave_p");
    cuda_check(cudaMemcpy(d_wv, wave_v, nb, cudaMemcpyHostToDevice), "H2D wave_v");
    cuda_check(cudaMemcpy(d_ws, wave_source, nb, cudaMemcpyHostToDevice), "H2D wave_source");
    cuda_check(cudaMemcpy(d_obs, obstacles, nbool, cudaMemcpyHostToDevice), "H2D obstacles");
    cuda_check(cudaMemcpy(d_wall, is_wall, nbool, cudaMemcpyHostToDevice), "H2D is_wall");
    cuda_check(cudaMemcpy(d_vac, is_vacuum, nbool, cudaMemcpyHostToDevice), "H2D is_vacuum");
    cuda_check(cudaMemcpy(d_perm, permeability, nbf, cudaMemcpyHostToDevice), "H2D permeability");

    const int block = 256;
    const int grid = (n + block - 1) / block;

    // ---- [μ-gate] (atmosphere_solver.cpp ~309): only run K0 + the GS if mu_q is
    //      above the threshold. Below it the diffusion operator is the identity —
    //      the atmosphere is unchanged by the GS. The sponge + wind ALWAYS run. ---
    if (mu_q > MU_EPS_Q) {
        // K0 Dinv + RHS (recomputed unconditionally; cache dropped).
        atmos_dinv_rhs<<<grid, block>>>(d_atmf, d_obs, d_wall, d_vac, d_perm,
                                        d_rhs, d_dinv, mu_q, h, w);
        cuda_check(cudaGetLastError(), "dinv_rhs launch");
        // K_GS: gs_iters sweeps, each TWO launches (RED color=0 then BLACK color=1).
        // A RED cell reads only the frozen black neighbours of the prior launch.
        for (int iter = 0; iter < gs_iters; ++iter) {
            for (int color = 0; color < 2; ++color) {
                atmos_gs_color<<<grid, block>>>(d_atmf, d_rhs, d_dinv, d_obs, d_wall,
                                                d_vac, d_perm, mu_q, color, h, w);
                cuda_check(cudaGetLastError(), "gs_color launch");
            }
        }
    }

    // ---- vacuum BFS (always): seed, then dist=1 (pass1), then dist=2 (pass2),
    //      double-buffered. After seed both buffers hold the seed level; pass1
    //      reads d_vac_a -> writes d_vac_b; pass2 reads d_vac_b -> writes d_vac_a.
    //      The FINAL distances live in d_vac_a after pass2. -----------------------
    vac_seed<<<grid, block>>>(d_obs, d_wall, d_vac, d_vac_a, d_vac_b, n);
    cuda_check(cudaGetLastError(), "vac_seed launch");
    vac_bfs_pass<<<grid, block>>>(d_vac_a, d_vac_b, d_obs, d_wall, (uint8_t)1, h, w);
    cuda_check(cudaGetLastError(), "bfs1 launch");
    vac_bfs_pass<<<grid, block>>>(d_vac_b, d_vac_a, d_obs, d_wall, (uint8_t)2, h, w);
    cuda_check(cudaGetLastError(), "bfs2 launch");

    // ---- K_SPONGE (always): per-vac_dist tier relax. Reads the FINAL d_vac_a. ---
    atmos_sponge<<<grid, block>>>(d_atmf, d_wp, d_wv, d_ws, d_vac_a, d_obs, d_wall,
                                  atm_vac_k_q, atm_inner_k_q, atm_outer_k_q,
                                  wv_inner_k_q, wv_outer_k_q, ws_half_q, n);
    cuda_check(cudaGetLastError(), "sponge launch");

    // ---- K_WIND (always): wind = -grad(post-sponge atm + wave_p). --------------
    atmos_wind<<<grid, block>>>(d_atmf, d_wp, d_wx, d_wy, d_obs, d_wall, d_vac,
                                d_perm, h, w);
    cuda_check(cudaGetLastError(), "wind launch");

    cuda_check(cudaDeviceSynchronize(), "sync");

    // D2H the 6 mutated fields. masks / perm are read-only — not copied back.
    cuda_check(cudaMemcpy(atmosphere, d_atmf, nb, cudaMemcpyDeviceToHost), "D2H atmosphere");
    cuda_check(cudaMemcpy(wave_p, d_wp, nb, cudaMemcpyDeviceToHost), "D2H wave_p");
    cuda_check(cudaMemcpy(wave_v, d_wv, nb, cudaMemcpyDeviceToHost), "D2H wave_v");
    cuda_check(cudaMemcpy(wave_source, d_ws, nb, cudaMemcpyDeviceToHost), "D2H wave_source");
    cuda_check(cudaMemcpy(wind_x, d_wx, nb, cudaMemcpyDeviceToHost), "D2H wind_x");
    cuda_check(cudaMemcpy(wind_y, d_wy, nb, cudaMemcpyDeviceToHost), "D2H wind_y");

    cudaFree(d_atmf);
    cudaFree(d_wp);
    cudaFree(d_wv);
    cudaFree(d_ws);
    cudaFree(d_wx);
    cudaFree(d_wy);
    cudaFree(d_rhs);
    cudaFree(d_dinv);
    cudaFree(d_obs);
    cudaFree(d_wall);
    cudaFree(d_vac);
    cudaFree(d_perm);
    cudaFree(d_vac_a);
    cudaFree(d_vac_b);
}

namespace {
bool g_atmos_backend_cuda = false;
}
bool atmos_backend_is_cuda() { return g_atmos_backend_cuda; }
void set_atmos_backend_cuda(bool on) { g_atmos_backend_cuda = on; }

}  // namespace breach_cuda
