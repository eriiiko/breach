// ============================================================================
// CUDA-S4a/S4b smoke solver implementation — see cuda_smoke.h.
// A bit-identical GPU port of SmokeDynamics::step (smoke_dynamics.cpp 187-302)
// and SmokeDynamics::sink_hop (the breach pull, smoke_dynamics.cpp 309-355 — S4b).
//
// Four kernels, one per pass of the CPU step(), launched as a barriered chain on
// ONE gas plane:
//   K1  diffusion Laplacian   (smoke_dynamics.cpp ~224-238) -> lap[] scratch
//   K2  diffusion apply       (~240-256)                    -> smoke in-place
//       (snapshot smoke -> src[] AFTER K2, exactly as the CPU std::vector copy)
//   K3  semi-Lagrangian advection (~276-289 + backtrace_sample_q) -> smoke in-place
//   K4  clamp + zero walls/vacuum (~292-299)                -> smoke in-place
//
// Every per-cell helper (neighbor_q, solid_wall_at, the DDA march, the integer
// bilinear, the renorm) is a VERBATIM device transcription of the CPU file's
// static inline helpers — same integer ops, same branch structure, same negative-
// displacement floor-divide. The renorm reciprocal is reciprocal_q16_dev (the
// shared device Newton reciprocal, bit-identical to fixedpoint::reciprocal_q16).
// ============================================================================
#include "cuda_smoke.h"
#include "cuda_resident.h"  // S8a STEP B: smoke_launch_resident / trace_smoke_resident
#include "fixed_point.h"   // q16, quantize, mul_q16, mul_wide, narrow, FP_ONE, FP_SHIFT
#include "cuda_fixedpoint_device.cuh"  // reciprocal_q16_dev (S4 §0 shared kit)

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

using namespace fixedpoint;

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in smoke_step/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// Smoke is a [0,1] tracer in Q16.16 (mirrors smoke_dynamics.cpp's constants).
constexpr int32_t SMOKE_MAX_Q  = FP_ONE;            // 65536
// Newton-reciprocal floor for the bilinear renorm (FP_ONE>>8 = 256).
constexpr int32_t WSUM_FLOOR_Q = FP_ONE >> 8;       // 256
// The "wsum negligible -> keep self" guard (FP_ONE>>14 = 4).
constexpr int32_t WSUM_EPS_Q   = FP_ONE >> 14;      // 4

// ---- neighbor_q (verbatim device port of smoke_dynamics.cpp:47-56) ----------
// Face-permeability neighbour value with Neumann fallback, Q16.16:
//   f[self] + face*(f[n]-f[self]),  face = min(perm[self], perm[n]) quantized.
// Out-of-bounds reflects (returns f[self]). The perm float bridge: quantize the
// min permeability per face exactly like the CPU (quantize((double)face_f)).
__device__ __forceinline__ int32_t neighbor_q_dev(
        const int32_t* __restrict__ f, const float* __restrict__ perm,
        int y, int x, int dy, int dx, int h, int w) {
    const int self_i = y * w + x;
    const int ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= h || nx < 0 || nx >= w) return f[self_i];
    const int ni = ny * w + nx;
    const float ps = perm[self_i], pn = perm[ni];
    const float face_f = ps < pn ? ps : pn;            // std::min
    const int32_t face_q = quantize((double)face_f);   // perm in [0,1] -> Q16.16
    return f[self_i] + mul_q16(face_q, f[ni] - f[self_i]);
}

// ---- solid_wall_at (verbatim device port of smoke_dynamics.cpp:64-73) -------
// Is the tile that contains the (clamped) sample point a SOLID WALL? A BREACH
// (vacuum that is NOT solid) is deliberately NOT a wall (a vent target).
__device__ __forceinline__ bool solid_wall_at_dev(
        int y, int x,
        const bool* __restrict__ obstacles, const bool* __restrict__ is_wall,
        const bool* __restrict__ is_vacuum, const float* __restrict__ perm,
        int h, int w,
        const bool* __restrict__ is_ambient) {
    if (y < 0 || y >= h || x < 0 || x >= w) return true;   // outside == wall
    const int i = y * w + x;
    // BC: the ambient ring is a trace SINK (venting target), not a wall — the
    // vacuum-breach idiom verbatim (is_ambient nullptr on space -> unchanged).
    const bool amb = is_ambient && is_ambient[i];
    const bool is_breach = (is_vacuum[i] || amb) &&
        !(obstacles[i] || is_wall[i] || perm[i] <= 0.0f);
    if (is_breach) return false;                            // venting target
    return obstacles[i] || is_wall[i] || is_vacuum[i] || amb || perm[i] <= 0.0f;
}

// ---- backtrace_sample_q (verbatim device port of smoke_dynamics.cpp:92-185) -
// The integer semi-Lagrangian back-trace: the sqrt-free DDA wall-clip march, the
// integer bilinear sample, the reciprocal_q16 renorm. EVERY integer op, branch,
// and the NEGATIVE-displacement floor-divide match the CPU bit-for-bit.
__device__ __forceinline__ int32_t backtrace_sample_q_dev(
        const int32_t* __restrict__ src, int x, int y, int32_t bx_q, int32_t by_q,
        const bool* __restrict__ obstacles, const bool* __restrict__ is_wall,
        const bool* __restrict__ is_vacuum, const float* __restrict__ perm,
        int h, int w,
        const bool* __restrict__ is_ambient) {
    // Departure point in Q16.16 (cell index << 16 + displacement).
    int64_t px_q = ((int64_t)x << FP_SHIFT) + bx_q;
    int64_t py_q = ((int64_t)y << FP_SHIFT) + by_q;

    // ---- Wall-clip march (DDA, no sqrt) ----
    const int32_t abx = bx_q >= 0 ? bx_q : -bx_q;
    const int32_t aby = by_q >= 0 ? by_q : -by_q;
    const int32_t amax = abx >= aby ? abx : aby;
    int n_steps = amax >> FP_SHIFT;
    if (amax & (FP_ONE - 1)) n_steps += 1;                 // ceil
    if (n_steps > 0) {
        // floordiv(a, b) for b>0: (a>=0) ? a/b : -((-a + b - 1)/b). The CPU lambda
        // floors toward -inf (matches the Python prototype's `//`); C `/` truncates
        // toward 0. For a NEGATIVE displacement (upwind back-trace, bx_q<0) they
        // differ by 1 count — this floor-divide is the #1 determinism hinge.
        const int b = n_steps;
        const int32_t sx_q = (bx_q >= 0) ? (bx_q / b)
                                         : -(int32_t)(((-(int64_t)bx_q) + b - 1) / b);
        const int32_t sy_q = (by_q >= 0) ? (by_q / b)
                                         : -(int32_t)(((-(int64_t)by_q) + b - 1) / b);
        int64_t cx_q = (int64_t)x << FP_SHIFT;
        int64_t cy_q = (int64_t)y << FP_SHIFT;
        for (int s = 0; s < n_steps; ++s) {
            const int64_t nxp_q = cx_q + sx_q;
            const int64_t nyp_q = cy_q + sy_q;
            const int ti = (int)((nxp_q + (FP_ONE >> 1)) >> FP_SHIFT);
            const int tj = (int)((nyp_q + (FP_ONE >> 1)) >> FP_SHIFT);
            if (solid_wall_at_dev(tj, ti, obstacles, is_wall, is_vacuum, perm, h, w,
                                  is_ambient))
                break;
            cx_q = nxp_q;
            cy_q = nyp_q;
            // BC: the ambient ring is a breach (vent target) — stop the march on it.
            if (tj >= 0 && tj < h && ti >= 0 && ti < w
                    && (is_vacuum[tj * w + ti]
                        || (is_ambient && is_ambient[tj * w + ti])))
                break;                                      // reached the breach
        }
        px_q = cx_q;
        py_q = cy_q;
    }

    // ---- Clamp in-bounds (Q16.16) ----
    const int64_t hi_x = (int64_t)(w - 1) << FP_SHIFT;
    const int64_t hi_y = (int64_t)(h - 1) << FP_SHIFT;
    if (px_q < 0) px_q = 0; else if (px_q > hi_x) px_q = hi_x;
    if (py_q < 0) py_q = 0; else if (py_q > hi_y) py_q = hi_y;

    // ---- Integer bilinear sample ----
    const int x0 = (int)(px_q >> FP_SHIFT);                // floor (px_q >= 0)
    const int y0 = (int)(py_q >> FP_SHIFT);
    const int x1 = (x0 + 1 <= w - 1) ? x0 + 1 : w - 1;
    const int y1 = (y0 + 1 <= h - 1) ? y0 + 1 : h - 1;
    const int32_t fx_q = (int32_t)(px_q - ((int64_t)x0 << FP_SHIFT));
    const int32_t fy_q = (int32_t)(py_q - ((int64_t)y0 << FP_SHIFT));
    const int32_t ifx_q = FP_ONE - fx_q;
    const int32_t ify_q = FP_ONE - fy_q;
    const int32_t w00 = mul_q16(ifx_q, ify_q);
    const int32_t w10 = mul_q16(fx_q,  ify_q);
    const int32_t w01 = mul_q16(ifx_q, fy_q);
    const int32_t w11 = mul_q16(fx_q,  fy_q);
    const int cyx[4][2] = { {y0, x0}, {y0, x1}, {y1, x0}, {y1, x1} };
    const int32_t cw[4] = { w00, w10, w01, w11 };

    int64_t acc = 0;
    int32_t wsum_q = 0;
    for (int k = 0; k < 4; ++k) {
        const int cy_ = cyx[k][0];
        const int cx_ = cyx[k][1];
        const int j = cy_ * w + cx_;
        if (obstacles[j] || is_wall[j] || perm[j] <= 0.0f) continue;  // sealed corner
        // BC: an ambient-ring corner samples 0 (absorbed), the vacuum idiom.
        const int32_t val_q = (is_vacuum[j] || (is_ambient && is_ambient[j])) ? 0 : src[j];
        acc += mul_wide(cw[k], val_q);
        wsum_q += cw[k];
    }
    if (wsum_q <= WSUM_EPS_Q) return src[y * w + x];        // negligible -> keep self

    const int32_t wsum_clamped = (wsum_q < WSUM_FLOOR_Q) ? WSUM_FLOOR_Q : wsum_q;
    const int32_t recip_q = reciprocal_q16_dev(wsum_clamped);   // 1/wsum, Q16.16
    const int32_t acc_q = narrow(acc);                     // Q.32 -> Q16.16
    return mul_q16(acc_q, recip_q);                        // (sum w*d)/wsum
}

// ---- K1: diffusion Laplacian (smoke_dynamics.cpp:224-238) -------------------
// lap[i] = (up+down+left+right) - 4*s, the permeability-weighted 4-neighbour
// gather. Reads the LIVE smoke (all neighbours are pre-diffusion values) + perm.
// Every thread writes its own lap[i] (scratch fully written before K2 reads it).
__global__ void smoke_lap(const int32_t* __restrict__ smoke,
                          const float* __restrict__ perm,
                          int32_t* __restrict__ lap, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int y = i / w;
        const int x = i % w;
        const int32_t s = smoke[i];
        const int32_t s_up    = neighbor_q_dev(smoke, perm, y, x, -1,  0, h, w);
        const int32_t s_down  = neighbor_q_dev(smoke, perm, y, x,  1,  0, h, w);
        const int32_t s_left  = neighbor_q_dev(smoke, perm, y, x,  0, -1, h, w);
        const int32_t s_right = neighbor_q_dev(smoke, perm, y, x,  0,  1, h, w);
        lap[i] = (s_up + s_down + s_left + s_right) - 4 * s;
    }
}

// ---- K2: diffusion apply (smoke_dynamics.cpp:240-256) -----------------------
// d_eff = d_smoke*(1 + wind_diffusion_scale*|wind|^2) in DOUBLE; |wind|^2 via
// mul_wide (Q.32) dequantized ONCE; coeff_q = quantize(d_eff*dt); smoke += coeff*lap.
// --fmad=false keeps the double fold from contracting -> bit-identical to the CPU
// /fp:strict path. In-place on smoke (each thread its own cell, lap frozen).
__global__ void smoke_diffuse(int32_t* __restrict__ smoke,
                              const int32_t* __restrict__ wind_x,
                              const int32_t* __restrict__ wind_y,
                              const int32_t* __restrict__ lap,
                              double d_smoke, double wind_diffusion_scale,
                              double actual_dt, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int64_t wind_sq_q32 = mul_wide(wind_x[i], wind_x[i])
                                  + mul_wide(wind_y[i], wind_y[i]);
        const double wind_sq = (double)wind_sq_q32
                             / ((double)FP_ONE * (double)FP_ONE);   // Q.32 -> real
        const double d_eff = d_smoke * (1.0 + wind_diffusion_scale * wind_sq);
        const int32_t coeff_q = quantize(d_eff * actual_dt);
        smoke[i] += mul_q16(coeff_q, lap[i]);
    }
}

// ---- K3: semi-Lagrangian advection (smoke_dynamics.cpp:276-289) -------------
// Non-(obstacle|wall|vacuum) cells back-trace upwind into `src` (the post-
// diffusion snapshot); other cells keep their post-diffusion value (the CPU
// `continue` — here a no-op write of smoke[i] = smoke[i] is avoided by guarding).
// bx_q = -mul_q16(wind_x, dt_adv_q). Reads src (frozen snapshot) + wind + masks.
__global__ void smoke_advect(int32_t* __restrict__ smoke,
                             const int32_t* __restrict__ src,
                             const int32_t* __restrict__ wind_x,
                             const int32_t* __restrict__ wind_y,
                             const bool* __restrict__ obstacles,
                             const bool* __restrict__ is_wall,
                             const bool* __restrict__ is_vacuum,
                             const float* __restrict__ perm,
                             int32_t dt_adv_q, int h, int w,
                             const bool* __restrict__ is_ambient) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        // BC: skip the ambient ring as a destination (it is a sink), like vacuum.
        if (obstacles[i] || is_wall[i] || is_vacuum[i]
                || (is_ambient && is_ambient[i])) continue;  // keep snapshot val
        const int y = i / w;
        const int x = i % w;
        const int32_t bx_q = -mul_q16(wind_x[i], dt_adv_q);
        const int32_t by_q = -mul_q16(wind_y[i], dt_adv_q);
        smoke[i] = backtrace_sample_q_dev(src, x, y, bx_q, by_q,
                                          obstacles, is_wall, is_vacuum, perm, h, w,
                                          is_ambient);
    }
}

// ---- K4: clamp + zero walls/vacuum (smoke_dynamics.cpp:292-299) -------------
__global__ void smoke_clamp(int32_t* __restrict__ smoke,
                            const bool* __restrict__ is_wall,
                            const bool* __restrict__ is_vacuum, int n,
                            const bool* __restrict__ is_ambient) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        // BC: the ambient ring is a trace sink — zeroed like vacuum every step.
        if (is_wall[i] || is_vacuum[i] || (is_ambient && is_ambient[i])) {
            smoke[i] = 0;
        } else {
            int32_t v = smoke[i];
            if (v < 0) v = 0;
            else if (v > SMOKE_MAX_Q) v = SMOKE_MAX_Q;
            smoke[i] = v;
        }
    }
}

// (S4b smoke_sink_hop DELETED — audit Patch A / A9, 2026-08-04. Its CPU twin
// SmokeDynamics::sink_hop was removed by the EOS refactor P3 (see the tombstone
// at smoke_dynamics.h:34 — venting is native to the compressible solver now), so
// this kernel and its host wrapper were an orphaned GPU port of a pass that no
// longer exists on either the live tick or the CPU side. Nothing called them:
// no C++ caller, and no Python caller of the `cuda_smoke_sink_hop` binding,
// which is deleted with them. ~150 lines that still compiled and shipped.)

// ---- S8a trace-decay kernel: the run_substeps decay loop, on the device.
// VERBATIM of physics_engine.cpp's per-cell decay loop: lost = mul_q16(v,
// frac_q) for v>0, REMOVED from the trace plane. frac_q is the host-
// quantized decay*dt (clamped [0,FP_ONE]) passed in. Order-independent
// (each cell independent) -> bit-identical to the CPU decay.
// P-T0 (energy-books arc, design §2.6 — the trace 0% ruling): the credit
// this kernel used to pay into inert_N2 (an `n2_slice` output param) is
// DELETED — decayed mass simply VANISHES now, same as the CPU twin
// (physics_engine.cpp's run_substeps trace loop). This device twin was
// MISSED by the P-T0 spec's explicit site list (it names only
// physics_engine.cpp:498-525 + the bindings.cpp:2812 note) — found by the
// CPU<->CUDA resident-path lockstep gates going red (test_cuda_thermal_
// mass.py et al.) until this credit was deleted here too.
__global__ void trace_decay(int32_t* __restrict__ gas_slice,
                            int32_t frac_q, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int32_t v = gas_slice[i];
        if (v <= 0) continue;
        const int32_t lost = mul_q16(v, frac_q);
        if (lost <= 0) continue;
        gas_slice[i] = v - lost;
    }
}

}  // namespace

void smoke_step(
    int32_t* smoke,
    const int32_t* wind_x, const int32_t* wind_y,
    const bool* obstacles, const bool* is_wall, const bool* is_vacuum,
    const float* permeability,
    int h, int w, float dt,
    float d_smoke, float wind_diffusion_scale, float advection_rate,
    const bool* is_ambient) {   // BC: ambient ring trace sink (nullptr = space)
    const int n = h * w;
    if (n <= 0) return;

    // ---- Host scalar precompute (smoke_dynamics.cpp:199,267-268, VERBATIM, in
    //      double). actual_dt = dt; dt_adv = advection_rate*actual_dt; quantize. --
    const double actual_dt = (double)dt;
    const double dt_adv = (double)advection_rate * actual_dt;
    const int32_t dt_adv_q = quantize(dt_adv);
    const double d_smoke_d = (double)d_smoke;
    const double wds_d = (double)wind_diffusion_scale;

    // ---- Device buffers (the gas plane + wind + masks + perm + scratch lap/src).
    //      Per-call H2D/D2H of the plane (S1/S3 pattern); residency is S8. -------
    const size_t nb    = (size_t)n * sizeof(int32_t);
    const size_t nbool = (size_t)n * sizeof(bool);
    const size_t nbf   = (size_t)n * sizeof(float);

    int32_t *d_gas = nullptr, *d_wx = nullptr, *d_wy = nullptr,
            *d_lap = nullptr, *d_src = nullptr;
    bool *d_obs = nullptr, *d_wall = nullptr, *d_vac = nullptr;
    float *d_perm = nullptr;

    cuda_check(cudaMalloc(&d_gas, nb), "malloc smoke");
    cuda_check(cudaMalloc(&d_wx, nb), "malloc wind_x");
    cuda_check(cudaMalloc(&d_wy, nb), "malloc wind_y");
    cuda_check(cudaMalloc(&d_lap, nb), "malloc lap");
    cuda_check(cudaMalloc(&d_src, nb), "malloc src");
    cuda_check(cudaMalloc(&d_obs, nbool), "malloc obstacles");
    cuda_check(cudaMalloc(&d_wall, nbool), "malloc is_wall");
    cuda_check(cudaMalloc(&d_vac, nbool), "malloc is_vacuum");
    cuda_check(cudaMalloc(&d_perm, nbf), "malloc permeability");

    cuda_check(cudaMemcpy(d_gas, smoke, nb, cudaMemcpyHostToDevice), "H2D smoke");
    cuda_check(cudaMemcpy(d_wx, wind_x, nb, cudaMemcpyHostToDevice), "H2D wind_x");
    cuda_check(cudaMemcpy(d_wy, wind_y, nb, cudaMemcpyHostToDevice), "H2D wind_y");
    cuda_check(cudaMemcpy(d_obs, obstacles, nbool, cudaMemcpyHostToDevice), "H2D obstacles");
    cuda_check(cudaMemcpy(d_wall, is_wall, nbool, cudaMemcpyHostToDevice), "H2D is_wall");
    cuda_check(cudaMemcpy(d_vac, is_vacuum, nbool, cudaMemcpyHostToDevice), "H2D is_vacuum");
    cuda_check(cudaMemcpy(d_perm, permeability, nbf, cudaMemcpyHostToDevice), "H2D permeability");

    // BC: optional ambient ring mask (nullptr on space maps -> the kernels take
    // the byte-identical space path via the `is_ambient && ...` short-circuit).
    bool* d_amb = nullptr;
    if (is_ambient) {
        cuda_check(cudaMalloc(&d_amb, nbool), "malloc is_ambient");
        cuda_check(cudaMemcpy(d_amb, is_ambient, nbool, cudaMemcpyHostToDevice), "H2D is_ambient");
    }

    const int block = 256;
    const int grid = (n + block - 1) / block;

    // K1 diffusion Laplacian -> d_lap (reads the live smoke).
    smoke_lap<<<grid, block>>>(d_gas, d_perm, d_lap, h, w);
    cuda_check(cudaGetLastError(), "lap launch");
    // K2 diffusion apply (in-place on d_gas).
    smoke_diffuse<<<grid, block>>>(d_gas, d_wx, d_wy, d_lap,
                                   d_smoke_d, wds_d, actual_dt, n);
    cuda_check(cudaGetLastError(), "diffuse launch");
    // Snapshot the POST-DIFFUSION smoke into d_src (matches the CPU std::vector
    // copy taken AFTER the diffusion apply, BEFORE the advection). A device-to-
    // device copy = the exact int32 snapshot the back-trace reads.
    cuda_check(cudaMemcpy(d_src, d_gas, nb, cudaMemcpyDeviceToDevice), "D2D src snapshot");
    // K3 semi-Lagrangian advection (in-place on d_gas; reads the frozen d_src).
    smoke_advect<<<grid, block>>>(d_gas, d_src, d_wx, d_wy,
                                  d_obs, d_wall, d_vac, d_perm, dt_adv_q, h, w,
                                  d_amb);
    cuda_check(cudaGetLastError(), "advect launch");
    // K4 clamp + zero walls/vacuum/ambient (in-place on d_gas).
    smoke_clamp<<<grid, block>>>(d_gas, d_wall, d_vac, n, d_amb);
    cuda_check(cudaGetLastError(), "clamp launch");

    cuda_check(cudaDeviceSynchronize(), "sync");

    cuda_check(cudaMemcpy(smoke, d_gas, nb, cudaMemcpyDeviceToHost), "D2H smoke");

    cudaFree(d_gas);
    cudaFree(d_wx);
    cudaFree(d_wy);
    cudaFree(d_lap);
    cudaFree(d_src);
    cudaFree(d_obs);
    cudaFree(d_wall);
    cudaFree(d_vac);
    cudaFree(d_perm);
    if (d_amb) cudaFree(d_amb);
}

// ---- STEP B launch core (S8a residency): the K1..K4 sequence, LAUNCH ONLY.
// No cudaMalloc / cudaMemcpy (H2D/D2H) / cudaFree / cudaDeviceSynchronize — the
// caller owns allocation, transfer, scratch (d_lap/d_src = n int32), and the
// single sync. The D2D post-diffusion snapshot (d_src <- d_gas) stays here (it is
// intrinsic to the algorithm — a device-to-device copy, NOT a host transfer). The
// host scalar precompute is identical bits to smoke_step's, so smoke_step
// (per-call) and trace_smoke_resident (once per tick) share ONE body.
void smoke_launch_resident(
    int32_t* d_gas,
    const int32_t* d_wind_x, const int32_t* d_wind_y,
    const bool* d_obstacles, const bool* d_is_wall, const bool* d_is_vacuum,
    const float* d_perm, const bool* d_is_ambient,
    int32_t* d_lap, int32_t* d_src,
    int h, int w, float dt,
    float d_smoke_f, float wind_diffusion_scale, float advection_rate) {
    const int n = h * w;
    if (n <= 0) return;

    // Host scalar precompute (smoke_dynamics.cpp:199,267-268, VERBATIM, double).
    const double actual_dt = (double)dt;
    const double dt_adv = (double)advection_rate * actual_dt;
    const int32_t dt_adv_q = quantize(dt_adv);
    const double d_smoke_d = (double)d_smoke_f;
    const double wds_d = (double)wind_diffusion_scale;

    const int block = 256;
    const int grid = (n + block - 1) / block;

    smoke_lap<<<grid, block>>>(d_gas, d_perm, d_lap, h, w);
    cuda_check(cudaGetLastError(), "res lap launch");
    smoke_diffuse<<<grid, block>>>(d_gas, d_wind_x, d_wind_y, d_lap,
                                   d_smoke_d, wds_d, actual_dt, n);
    cuda_check(cudaGetLastError(), "res diffuse launch");
    // Post-diffusion snapshot into d_src (the CPU std::vector copy taken AFTER
    // the diffusion apply, BEFORE the advection). D2D on the null stream, so the
    // stream order K2 < copy < K3 holds without an explicit sync.
    cuda_check(cudaMemcpy(d_src, d_gas, (size_t)n * sizeof(int32_t),
                          cudaMemcpyDeviceToDevice), "res D2D src snapshot");
    smoke_advect<<<grid, block>>>(d_gas, d_src, d_wind_x, d_wind_y,
                                  d_obstacles, d_is_wall, d_is_vacuum, d_perm,
                                  dt_adv_q, h, w, d_is_ambient);
    cuda_check(cudaGetLastError(), "res advect launch");
    smoke_clamp<<<grid, block>>>(d_gas, d_is_wall, d_is_vacuum, n, d_is_ambient);
    cuda_check(cudaGetLastError(), "res clamp launch");
    // NO sync — the caller (trace_smoke_resident) owns the one sync per tick.
}

// ---- STEP B: the whole per-tick trace-plane loop, device-resident (FLOOR item 3).
// Persistent C++-owned lap/src scratch keyed by (h,w). Per non-conservative plane:
// smoke_launch_resident + the decay kernel. ONE sync at the end. NO per-plane
// malloc/H2D/D2H.
namespace {
struct SmokeResidentScratch {
    int h = 0, w = 0;
    int32_t *lap = nullptr, *src = nullptr;
    void free_all() {
        if (lap) cudaFree(lap);
        if (src) cudaFree(src);
        lap = src = nullptr;
    }
    void ensure(int H, int W) {
        if (H == h && W == w && lap) return;
        free_all();
        h = H; w = W;
        const size_t n = (size_t)H * W;
        cuda_check(cudaMalloc(&lap, n * sizeof(int32_t)), "res malloc lap");
        cuda_check(cudaMalloc(&src, n * sizeof(int32_t)), "res malloc src");
    }
};
SmokeResidentScratch g_smoke_res;
}  // namespace

void trace_smoke_resident(
    int32_t* d_gas_base,
    const int32_t* d_wind_x, const int32_t* d_wind_y,
    const bool* d_solid, const bool* d_is_vacuum, const float* d_perm,
    const bool* d_is_ambient,
    int h, int w, int n_gases, int inert_n2_idx,
    const bool* gas_conservative, const float* gas_diffusion,
    const float* gas_decay,
    float dt, float advection_rate, float wind_diffusion_scale) {
    const int n = h * w;
    if (n <= 0) return;
    g_smoke_res.ensure(h, w);
    const int block = 256;
    const int grid = (n + block - 1) / block;
    // P-T0 (design §2.6): the decay->inert_N2 credit is DELETED (this
    // kernel no longer touches inert_N2 at all); `inert_n2_idx` is kept as
    // a parameter for ABI/back-compat with the existing call site (the
    // physics_engine.cpp `(void)inert_n2_idx;` idiom).
    (void)inert_n2_idx;
    for (int gi = 0; gi < n_gases; ++gi) {
        if (gas_conservative[gi]) continue;   // bulk planes ride the EOS solve
        int32_t* gas_slice = d_gas_base + (size_t)gi * n;
        // The all-zero-plane `.any()` skip is DROPPED (arithmetic no-op — see the
        // header note); processing every trace plane is bit-identical.
        // obstacles == is_wall == solid (run_substeps passes solid for both).
        smoke_launch_resident(gas_slice, d_wind_x, d_wind_y,
                              d_solid, d_solid, d_is_vacuum, d_perm, d_is_ambient,
                              g_smoke_res.lap, g_smoke_res.src,
                              h, w, dt, gas_diffusion[gi], wind_diffusion_scale,
                              advection_rate);
        // decay (once per tick, right after this plane's advection). The
        // decayed count simply VANISHES — P-T0 deleted the credit.
        const float decay_gi = gas_decay[gi];
        if (decay_gi > 0.0f) {
            using namespace fixedpoint;
            q16 frac_q = quantize((double)decay_gi * (double)dt);
            if (frac_q < 0) frac_q = 0;
            if (frac_q > FP_ONE) frac_q = FP_ONE;
            if (frac_q > 0) {
                trace_decay<<<grid, block>>>(gas_slice, frac_q, n);
                cuda_check(cudaGetLastError(), "res decay launch");
            }
        }
    }
    cuda_check(cudaDeviceSynchronize(), "trace_smoke_resident sync");
}

namespace {
bool g_smoke_backend_cuda = false;
}
bool smoke_backend_is_cuda() { return g_smoke_backend_cuda; }
void set_smoke_backend_cuda(bool on) { g_smoke_backend_cuda = on; }

}  // namespace breach_cuda
