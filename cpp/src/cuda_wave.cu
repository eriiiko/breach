// ============================================================================
// CUDA-S5 wave solver implementation — see cuda_wave.h.
// A bit-identical GPU port of AtmosphereSolver::wave_substep (atmosphere_solver.cpp
// ~62-261): the explicit damped-wave shockwave substep + the mean_wp reduction +
// the one-sided anomaly transfer.
//
// Eight kernels, one per pass of the CPU wave_substep, launched as a barriered
// chain (separate launches = grid barriers between dependent passes):
//   K1 wave_feed     feed source into wave_p (rate-limited; wave_source drained)
//   K2 wave_lap      Laplacian gather -> lap[] scratch (per-face perm float bridge)
//   K3 wave_vkick    velocity kick (int64 widen: c_sq_dt*lap - damp_dt*wave_v)
//   K4 wave_pupdate  pressure update wave_p += mul_q16(wave_v, dt_q)
//   K5 wave_absorb   per-cell absorption (scale_mag magnitude shrink)
//   K6 wave_bc       zero wave_p/wave_v on obstacle|wall|vacuum
//   K7 wave_reduce   int64 atomicAdd of wave_p over the interior mask -> d_sum
//   K8 wave_transfer atmosphere += round_nearest((wave_p - mean_wp) * xfer_q)
//
// Every per-cell op is a VERBATIM device transcription of the CPU file's loops —
// same integer ops, same branch structure. The per-face permeability weight is a
// device float bridge (quantize((double)min(perm[self],perm[n])), exactly like the
// CPU + S4a's neighbor_q; --fmad=false makes the double min/quantize bit-identical).
//
// THE DETERMINISM CRUX (K7): the mean_wp reduction uses an int64 device
// accumulator with atomicAdd. Integer + is associative + commutative -> the final
// sum is bit-identical regardless of thread/scheduler order (unlike a float
// atomicAdd, which jitters). The mean_round is computed ON THE HOST after reading
// back d_sum, the exact CPU mean_round. The interior predicate is bool topology
// only (no float). No overflow: |wave_p| < 2^31 and the interior count <= h*w
// (<< 2^32 for any real grid), so |sum| < 2^63 with comfortable int64 headroom.
// ============================================================================
#include "cuda_wave.h"
#include "fixed_point.h"   // q16, quantize, mul_q16, mul_wide, narrow, FP_ONE, FP_SHIFT
#include "cuda_fixedpoint_device.cuh"  // scale_mag_dev, round_nearest_q_dev (S5 §3)

#include <cuda_runtime.h>

#include <algorithm>   // std::min (host count is a plain loop; min is CPU-side)
#include <sstream>
#include <stdexcept>

using namespace fixedpoint;

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in wave_substep_gpu/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// ---- K1: feed source (atmosphere_solver.cpp ~116-124) -----------------------
// feed = min(source*feed_rate_dt, source, max_source). wave_p += feed;
// wave_source -= feed. Only when source > source_thresh. Pure integer; integer min
// is exact (std::min replaced by branchless device min on int32).
__global__ void wave_feed(int32_t* __restrict__ wave_p,
                          int32_t* __restrict__ wave_source,
                          int32_t feed_rate_dt_q, int32_t max_source_q,
                          int32_t source_thresh_q, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int32_t src = wave_source[i];
        if (src > source_thresh_q) {
            int32_t feed = mul_q16(src, feed_rate_dt_q);
            if (feed > src) feed = src;                  // std::min(feed, source)
            if (feed > max_source_q) feed = max_source_q; // std::min(feed, cap)
            wave_p[i] += feed;
            wave_source[i] = src - feed;
        }
    }
}

// ---- K2: Laplacian gather (atmosphere_solver.cpp ~137-158) -------------------
// Per face, w = quantize((double)min(perm[self], perm[n])) (the float bridge on
// the weight; the field difference is exact integer); flux = mul_wide(w, wp[n]-wp);
// lap[i] = narrow(Σ4). OOB faces contribute 0 (the CPU `if (y>0)` etc. — Neumann
// via omission, NOT reflection here: an absent face is simply skipped). Reads the
// LIVE wave_p (post-feed). Every thread writes its own lap[i].
__global__ void wave_lap(const int32_t* __restrict__ wave_p,
                         const float* __restrict__ perm,
                         int32_t* __restrict__ lap, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int y = i / w;
        const int x = i % w;
        const int32_t p = wave_p[i];
        const float perm_i = perm[i];

        int64_t acc = 0;
        // up
        if (y > 0) {
            const int nb = (y - 1) * w + x;
            const float pn = perm[nb];
            const float face_f = perm_i < pn ? perm_i : pn;   // std::min
            const int32_t wgt = quantize((double)face_f);
            acc += mul_wide(wgt, wave_p[nb] - p);
        }
        // down
        if (y < h - 1) {
            const int nb = (y + 1) * w + x;
            const float pn = perm[nb];
            const float face_f = perm_i < pn ? perm_i : pn;
            const int32_t wgt = quantize((double)face_f);
            acc += mul_wide(wgt, wave_p[nb] - p);
        }
        // left
        if (x > 0) {
            const int nb = y * w + (x - 1);
            const float pn = perm[nb];
            const float face_f = perm_i < pn ? perm_i : pn;
            const int32_t wgt = quantize((double)face_f);
            acc += mul_wide(wgt, wave_p[nb] - p);
        }
        // right
        if (x < w - 1) {
            const int nb = y * w + (x + 1);
            const float pn = perm[nb];
            const float face_f = perm_i < pn ? perm_i : pn;
            const int32_t wgt = quantize((double)face_f);
            acc += mul_wide(wgt, wave_p[nb] - p);
        }
        lap[i] = narrow(acc);   // one shared truncation -> Q16.16 lap
    }
}

// ---- K3: velocity kick (atmosphere_solver.cpp ~169-173) ---------------------
// wave_v += narrow( mul_wide(c_sq_dt_q, lap) - mul_wide(damp_dt_q, wave_v) ).
// The *dt is folded into c_sq_dt_q / damp_dt_q (host precompute); both terms carry
// in int64 (the OVERFLOW WATCH — c_sq*lap exceeds 32768 BEFORE *dt) and the *dt is
// inside the wide product, narrowed ONCE. In-place on wave_v (each thread its own
// cell; lap frozen from K2).
__global__ void wave_vkick(int32_t* __restrict__ wave_v,
                           const int32_t* __restrict__ lap,
                           int32_t c_sq_dt_q, int32_t damp_dt_q, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int64_t kick_wide = mul_wide(c_sq_dt_q, lap[i])
                                - mul_wide(damp_dt_q, wave_v[i]);
        wave_v[i] += narrow(kick_wide);
    }
}

// ---- K4: pressure update (atmosphere_solver.cpp ~175-178) -------------------
// wave_p += mul_q16(wave_v, dt_q). In-place on wave_p (reads the post-kick wave_v).
__global__ void wave_pupdate(int32_t* __restrict__ wave_p,
                             const int32_t* __restrict__ wave_v,
                             int32_t dt_q, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        wave_p[i] += mul_q16(wave_v[i], dt_q);
    }
}

// ---- K5: absorption (atmosphere_solver.cpp ~180-193) ------------------------
// a = mul_q16(quantize(wave_absorb[i]), absorb_str_dt_q) (the absorb float bridge);
// if a > 0: k = (a < FP_ONE) ? (FP_ONE - a) : 0; wave_v = scale_mag(wave_v, k);
// wave_p = scale_mag(wave_p, k). scale_mag (magnitude shrink) — NOT mul_q16 — so a
// negative value's magnitude can only SHRINK (the S1 absorb idiom). In-place.
__global__ void wave_absorb(int32_t* __restrict__ wave_p,
                            int32_t* __restrict__ wave_v,
                            const float* __restrict__ wave_absorb_f,
                            int32_t absorb_str_dt_q, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int32_t a = mul_q16(quantize((double)wave_absorb_f[i]), absorb_str_dt_q);
        if (a > 0) {
            const int32_t k = (a < FP_ONE) ? (int32_t)(FP_ONE - a) : 0;
            wave_v[i] = scale_mag_dev(wave_v[i], k);
            wave_p[i] = scale_mag_dev(wave_p[i], k);
        }
    }
}

// ---- K6: wave BCs (atmosphere_solver.cpp ~195-201) --------------------------
// Zero wave_p/wave_v on obstacle|wall|vacuum (exact integer 0). In-place.
__global__ void wave_bc(int32_t* __restrict__ wave_p,
                        int32_t* __restrict__ wave_v,
                        const bool* __restrict__ obstacles,
                        const bool* __restrict__ is_wall,
                        const bool* __restrict__ is_vacuum, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (is_wall[i] || is_vacuum[i] || obstacles[i]) {
            wave_p[i] = 0;
            wave_v[i] = 0;
        }
    }
}

// ---- K7: mean_wp reduction (atmosphere_solver.cpp ~204-220) -----------------
// THE DETERMINISM CRUX. Each interior cell atomicAdds (int64)wave_p[i] into a
// device int64 accumulator. The interior predicate is bool topology only
// (!obstacle && !wall && !vacuum). Integer atomicAdd is associative + commutative
// -> the sum is ORDER-FREE -> bit-identical to the CPU mean_sum regardless of
// thread/scheduler order. atomicAdd on 64-bit goes through the unsigned long long
// overload; the two's-complement bit pattern of the int64 sum is preserved exactly
// (modular wrap is irrelevant — the true sum is far inside int64 range). The mean
// (mean_round) is computed ON THE HOST after read-back.
__global__ void wave_reduce(const int32_t* __restrict__ wave_p,
                            const bool* __restrict__ obstacles,
                            const bool* __restrict__ is_wall,
                            const bool* __restrict__ is_vacuum,
                            unsigned long long* __restrict__ d_sum, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const bool interior = (!obstacles[i] && !is_wall[i] && !is_vacuum[i]);
        if (interior) {
            atomicAdd(d_sum, (unsigned long long)(long long)(int64_t)wave_p[i]);
        }
    }
}

// ---- K8: anomaly transfer (atmosphere_solver.cpp ~223-260) ------------------
// Over the SAME interior mask: anom = wave_p - mean_wp (zero-mean, exact int);
// d = round_nearest(anom * xfer_q) (sign-symmetric); atmosphere += d. ONE-SIDED
// forcing — wave_p is NOT drained (Erik's design). mean_wp arrives as a scalar arg
// (computed on the host from the K7 sum). In-place on atmosphere.
__global__ void wave_transfer(const int32_t* __restrict__ wave_p,
                              int32_t* __restrict__ atmosphere,
                              const bool* __restrict__ obstacles,
                              const bool* __restrict__ is_wall,
                              const bool* __restrict__ is_vacuum,
                              int32_t mean_wp, int32_t xfer_q, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const bool interior = (!obstacles[i] && !is_wall[i] && !is_vacuum[i]);
        if (interior) {
            const int32_t anom = wave_p[i] - mean_wp;       // zero-mean (exact int)
            const int64_t prod = (int64_t)anom * (int64_t)xfer_q;
            const int32_t d = round_nearest_q_dev(prod);     // sign-symmetric round
            atmosphere[i] += d;          // one-sided forcing: wave drives the bulk
        }
    }
}

}  // namespace

void wave_substep_gpu(
    int32_t* wave_p, int32_t* wave_v, int32_t* wave_source, int32_t* atmosphere,
    const bool* obstacles, const bool* is_wall, const bool* is_vacuum,
    const float* permeability, const float* wave_absorb_f,
    int h, int w, float dt,
    float c, float damping, float absorb_strength,
    float transfer, float feed_rate, float max_source_per_step) {
    const int n = h * w;
    if (n <= 0) return;

    // ---- Host scalar precompute (atmosphere_solver.cpp ~92-111, VERBATIM, in
    //      double). dt floats per substep; these are per-call. ------------------
    const double dt_d   = (double)dt;
    const double c_d    = (double)c;
    const double c_sq_d = c_d * c_d;                       // c_sq at wave_c
    const int32_t c_sq_dt_q       = quantize(c_sq_d * dt_d);
    const int32_t damp_dt_q       = quantize((double)damping * dt_d);
    const int32_t dt_q            = quantize(dt_d);
    const int32_t xfer_q          = quantize((double)transfer * dt_d);
    const int32_t feed_rate_dt_q  = quantize((double)feed_rate * dt_d);
    const int32_t max_source_q    = quantize((double)max_source_per_step);
    const int32_t source_thresh_q = quantize(0.001);
    const int32_t absorb_str_dt_q = quantize((double)absorb_strength * dt_d);

    // ---- count: #interior cells, computed ON THE HOST (deterministic; the CPU
    //      reduction divides by exactly this). A plain host loop over the masks. --
    int64_t count = 0;
    for (int i = 0; i < n; ++i) {
        if (!obstacles[i] && !is_wall[i] && !is_vacuum[i]) ++count;
    }

    // ---- Device buffers (the 4 mutated fields + masks + perm/absorb + scratch
    //      lap + the int64 reduction accumulator). Per-call H2D/D2H; residency
    //      is S8. ------------------------------------------------------------------
    const size_t nb    = (size_t)n * sizeof(int32_t);
    const size_t nbool = (size_t)n * sizeof(bool);
    const size_t nbf   = (size_t)n * sizeof(float);

    int32_t *d_wp = nullptr, *d_wv = nullptr, *d_ws = nullptr,
            *d_atm = nullptr, *d_lap = nullptr;
    bool *d_obs = nullptr, *d_wall = nullptr, *d_vac = nullptr;
    float *d_perm = nullptr, *d_absorb = nullptr;
    unsigned long long *d_sum = nullptr;

    cuda_check(cudaMalloc(&d_wp, nb), "malloc wave_p");
    cuda_check(cudaMalloc(&d_wv, nb), "malloc wave_v");
    cuda_check(cudaMalloc(&d_ws, nb), "malloc wave_source");
    cuda_check(cudaMalloc(&d_atm, nb), "malloc atmosphere");
    cuda_check(cudaMalloc(&d_lap, nb), "malloc lap");
    cuda_check(cudaMalloc(&d_obs, nbool), "malloc obstacles");
    cuda_check(cudaMalloc(&d_wall, nbool), "malloc is_wall");
    cuda_check(cudaMalloc(&d_vac, nbool), "malloc is_vacuum");
    cuda_check(cudaMalloc(&d_perm, nbf), "malloc permeability");
    cuda_check(cudaMalloc(&d_absorb, nbf), "malloc wave_absorb");
    cuda_check(cudaMalloc(&d_sum, sizeof(unsigned long long)), "malloc d_sum");

    cuda_check(cudaMemcpy(d_wp, wave_p, nb, cudaMemcpyHostToDevice), "H2D wave_p");
    cuda_check(cudaMemcpy(d_wv, wave_v, nb, cudaMemcpyHostToDevice), "H2D wave_v");
    cuda_check(cudaMemcpy(d_ws, wave_source, nb, cudaMemcpyHostToDevice), "H2D wave_source");
    cuda_check(cudaMemcpy(d_atm, atmosphere, nb, cudaMemcpyHostToDevice), "H2D atmosphere");
    cuda_check(cudaMemcpy(d_obs, obstacles, nbool, cudaMemcpyHostToDevice), "H2D obstacles");
    cuda_check(cudaMemcpy(d_wall, is_wall, nbool, cudaMemcpyHostToDevice), "H2D is_wall");
    cuda_check(cudaMemcpy(d_vac, is_vacuum, nbool, cudaMemcpyHostToDevice), "H2D is_vacuum");
    cuda_check(cudaMemcpy(d_perm, permeability, nbf, cudaMemcpyHostToDevice), "H2D permeability");
    cuda_check(cudaMemcpy(d_absorb, wave_absorb_f, nbf, cudaMemcpyHostToDevice), "H2D wave_absorb");
    cuda_check(cudaMemset(d_sum, 0, sizeof(unsigned long long)), "memset d_sum");

    const int block = 256;
    const int grid = (n + block - 1) / block;

    // K1 feed source (in-place on d_wp / d_ws).
    wave_feed<<<grid, block>>>(d_wp, d_ws, feed_rate_dt_q, max_source_q,
                               source_thresh_q, n);
    cuda_check(cudaGetLastError(), "feed launch");
    // K2 Laplacian gather -> d_lap (reads the post-feed wave_p).
    wave_lap<<<grid, block>>>(d_wp, d_perm, d_lap, h, w);
    cuda_check(cudaGetLastError(), "lap launch");
    // K3 velocity kick (in-place on d_wv; reads the frozen d_lap).
    wave_vkick<<<grid, block>>>(d_wv, d_lap, c_sq_dt_q, damp_dt_q, n);
    cuda_check(cudaGetLastError(), "vkick launch");
    // K4 pressure update (in-place on d_wp; reads the post-kick d_wv).
    wave_pupdate<<<grid, block>>>(d_wp, d_wv, dt_q, n);
    cuda_check(cudaGetLastError(), "pupdate launch");
    // K5 absorption (in-place on d_wp / d_wv).
    wave_absorb<<<grid, block>>>(d_wp, d_wv, d_absorb, absorb_str_dt_q, n);
    cuda_check(cudaGetLastError(), "absorb launch");
    // K6 wave BCs (in-place on d_wp / d_wv).
    wave_bc<<<grid, block>>>(d_wp, d_wv, d_obs, d_wall, d_vac, n);
    cuda_check(cudaGetLastError(), "bc launch");
    // K7 mean_wp reduction: int64 atomicAdd over the interior -> d_sum.
    wave_reduce<<<grid, block>>>(d_wp, d_obs, d_wall, d_vac, d_sum, n);
    cuda_check(cudaGetLastError(), "reduce launch");

    // Read back the order-free int64 sum and compute mean_round ON THE HOST — the
    // EXACT CPU mean_round (round-half-away-from-zero, NO pre-shift): sum is Q16.16,
    // count a plain int, so sum/count is the Q16.16 mean directly.
    unsigned long long sum_u = 0;
    cuda_check(cudaMemcpy(&sum_u, d_sum, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost), "D2H d_sum");
    const int64_t sum = (int64_t)sum_u;
    int32_t mean_wp = 0;
    if (count > 0) {
        const int64_t half = count / 2;
        const int64_t m = (sum >= 0) ? (sum + half) / count
                                     : (sum - half) / count;
        mean_wp = (int32_t)m;
    }

    // K8 anomaly transfer (in-place on d_atm; mean_wp is a scalar arg).
    wave_transfer<<<grid, block>>>(d_wp, d_atm, d_obs, d_wall, d_vac,
                                   mean_wp, xfer_q, n);
    cuda_check(cudaGetLastError(), "transfer launch");

    cuda_check(cudaDeviceSynchronize(), "sync");

    // D2H the 4 mutated fields (wave_p, wave_v, wave_source, atmosphere). perm /
    // absorb / masks are read-only — not copied back.
    cuda_check(cudaMemcpy(wave_p, d_wp, nb, cudaMemcpyDeviceToHost), "D2H wave_p");
    cuda_check(cudaMemcpy(wave_v, d_wv, nb, cudaMemcpyDeviceToHost), "D2H wave_v");
    cuda_check(cudaMemcpy(wave_source, d_ws, nb, cudaMemcpyDeviceToHost), "D2H wave_source");
    cuda_check(cudaMemcpy(atmosphere, d_atm, nb, cudaMemcpyDeviceToHost), "D2H atmosphere");

    cudaFree(d_wp);
    cudaFree(d_wv);
    cudaFree(d_ws);
    cudaFree(d_atm);
    cudaFree(d_lap);
    cudaFree(d_obs);
    cudaFree(d_wall);
    cudaFree(d_vac);
    cudaFree(d_perm);
    cudaFree(d_absorb);
    cudaFree(d_sum);
}

namespace {
bool g_wave_backend_cuda = false;
}
bool wave_backend_is_cuda() { return g_wave_backend_cuda; }
void set_wave_backend_cuda(bool on) { g_wave_backend_cuda = on; }

}  // namespace breach_cuda
