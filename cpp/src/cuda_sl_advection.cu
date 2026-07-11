// ============================================================================
// EOS P6.2 — fused 3-field SL advection implementation — see cuda_sl_advection.h.
// A bit-identical GPU port of the EOS solver's substep-loop advection chain
// (eos_solver.cpp: the per-tick cmask build + step 1a/1b/1f, i.e. exactly what
// eos_sl_advect_reference replays on the CPU).
//
// Two kernels, launched as a barriered chain:
//   K0  cmask build   (eos_solver.cpp cmask_ loop)      -> cmask[] (once/tick)
//   K1  fused advect  (eos_solver.cpp substep loop a+b+f) -> u/T in-place,
//       reading the frozen (vx,vy,T) src snapshot (D2D memcpy per substep —
//       the exact analogue of the CPU's vx_src_/vy_src_/t_src_ vector copies)
//
// eos_backtrace_sample3_q_dev is a VERBATIM device transcription of the CPU
// file's anon-namespace eos_backtrace_sample3_q — same integer ops, same
// branch structure, same negative-displacement floor-divide, same zero-
// displacement / all-live-corner / negligible-wsum fast paths (documented
// behavior, not rounding drift). The renorm reciprocal is reciprocal_q16_dev
// (the shared device Newton reciprocal, bit-identical to
// fixedpoint::reciprocal_q16). Pure GATHER: each destination cell writes only
// itself — no scatter hazard, no atomics (review §1.4).
// ============================================================================
#include "cuda_sl_advection.h"
#include "fixed_point.h"   // q16, quantize, mul_q16, mul_wide, narrow, FP_ONE, FP_SHIFT
#include "cuda_fixedpoint_device.cuh"  // reciprocal_q16_dev (shared device kit)

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

using namespace fixedpoint;

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in eos_sl_advect/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// The bilinear-renorm guards (eos_backtrace_sample3_q's constants, verbatim).
constexpr int32_t WSUM_EPS_Q   = FP_ONE >> 14;   // 4
constexpr int32_t WSUM_FLOOR_Q = FP_ONE >> 8;    // 256

// ---- digest: the cheap FNV-1a-style running hash over a Q16.16 buffer -------
// HOST-side (review §2.6: per-call P6 kernels return their buffers to host
// anyway, so every gate digest is a D2H + host hash). Byte-for-byte the
// anon-namespace digest_of in eos_solver.cpp (sequential, order-dependent,
// pure integer).
uint64_t digest_of_host(const int32_t* buf, int n, uint64_t seed) {
    uint64_t h = seed ^ 1469598103934665603ULL;
    for (int i = 0; i < n; ++i) {
        h ^= (uint64_t)(uint32_t)buf[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// ---- K0: cmask build (eos_solver.cpp's per-tick cmask_ loop, verbatim) ------
// 0 = sealed (solid || perm <= 0) — wall to the march, dead corner
// 1 = breach (vacuum, open)       — march target, zero-valued corner
// 2 = live   (open air)           — regular corner
// The perm float test is a COMPARISON only (no float arithmetic) — bit-exact.
__global__ void sl_cmask_build(const bool* __restrict__ solid,
                               const bool* __restrict__ is_vacuum,
                               const float* __restrict__ perm,
                               uint8_t* __restrict__ cmask, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (solid[i] || perm[i] <= 0.0f) cmask[i] = 0;
        else if (is_vacuum[i])           cmask[i] = 1;
        else                             cmask[i] = 2;
    }
}

// ---- eos_backtrace_sample3_q (VERBATIM device port, eos_solver.cpp) ---------
// The FUSED 3-field integer semi-Lagrangian back-trace: one DDA wall-clip
// march + one bilinear weight set shared by (vx, vy, T), which ride the SAME
// displacement -u*dt_s from the same cell. Fast paths (zero displacement ->
// source values outright; all-open corners -> skip the Newton renorm;
// negligible wsum -> keep self) are the CPU's own documented behavior.
struct FusedSample { int32_t vx, vy, t; };

__device__ __forceinline__ bool sl_solid_wall_at(
        const uint8_t* __restrict__ cmask, int ty, int tx, int h, int w) {
    // Original predicate: wall == OOB || solid || perm<=0 || (vacuum-sealed);
    // a vacuum && open cell is a BREACH (not a wall). Table form: wall <=>
    // OOB || cmask == 0 (the sealed set); breach <=> cmask == 1.
    if (ty < 0 || ty >= h || tx < 0 || tx >= w) return true;
    const int i = ty * w + tx;
    if (cmask[i] == 1) return false;   // breach: march may enter
    return cmask[i] == 0 || false;     // sealed: wall (CPU form kept verbatim)
}

__device__ __forceinline__ FusedSample eos_backtrace_sample3_q_dev(
        const int32_t* __restrict__ src_vx, const int32_t* __restrict__ src_vy,
        const int32_t* __restrict__ src_t,
        int x, int y, int32_t bx_q, int32_t by_q,
        const uint8_t* __restrict__ cmask, int h, int w) {
    const int i0 = y * w + x;
    if (bx_q == 0 && by_q == 0) {
        return { src_vx[i0], src_vy[i0], src_t[i0] };
    }
    int64_t px_q = ((int64_t)x << FP_SHIFT) + bx_q;
    int64_t py_q = ((int64_t)y << FP_SHIFT) + by_q;

    const int32_t abx = bx_q >= 0 ? bx_q : -bx_q;
    const int32_t aby = by_q >= 0 ? by_q : -by_q;
    const int32_t amax = abx >= aby ? abx : aby;
    int n_steps = amax >> FP_SHIFT;
    if (amax & (FP_ONE - 1)) n_steps += 1;                 // ceil

    if (n_steps > 0) {
        // floordiv toward -inf (the CPU lambda; C `/` truncates toward 0 —
        // for a NEGATIVE displacement they differ by 1 count: the #1
        // determinism hinge, same as cuda_smoke.cu's port).
        const int b = n_steps;
        const int32_t sx_q = (bx_q >= 0) ? (bx_q / b)
                                         : -(int32_t)(((-(int64_t)bx_q) + b - 1) / b);
        const int32_t sy_q = (by_q >= 0) ? (by_q / b)
                                         : -(int32_t)(((-(int64_t)by_q) + b - 1) / b);
        int64_t cx_q = (int64_t)x << FP_SHIFT;
        int64_t cy_q = (int64_t)y << FP_SHIFT;
        for (int st = 0; st < n_steps; ++st) {
            const int64_t nxp_q = cx_q + sx_q;
            const int64_t nyp_q = cy_q + sy_q;
            const int ti = (int)((nxp_q + (FP_ONE >> 1)) >> FP_SHIFT);
            const int tj = (int)((nyp_q + (FP_ONE >> 1)) >> FP_SHIFT);
            if (sl_solid_wall_at(cmask, tj, ti, h, w)) break;
            cx_q = nxp_q;
            cy_q = nyp_q;
            // original: stop after stepping ONTO a vacuum tile. Any vacuum
            // tile the march can be standing on here is a BREACH (sealed
            // vacuum forms are walls and broke above) == cmask 1.
            if (tj >= 0 && tj < h && ti >= 0 && ti < w && cmask[tj * w + ti] == 1) break;
        }
        px_q = cx_q;
        py_q = cy_q;
    }

    const int64_t hi_x = (int64_t)(w - 1) << FP_SHIFT;
    const int64_t hi_y = (int64_t)(h - 1) << FP_SHIFT;
    if (px_q < 0) px_q = 0; else if (px_q > hi_x) px_q = hi_x;
    if (py_q < 0) py_q = 0; else if (py_q > hi_y) py_q = hi_y;

    const int x0 = (int)(px_q >> FP_SHIFT);
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

    int64_t ax = 0, ay = 0, at = 0;
    int32_t wsum_q = 0;
    for (int k = 0; k < 4; ++k) {
        const int j = cyx[k][0] * w + cyx[k][1];
        const uint8_t m = cmask[j];
        if (m == 0) continue;                              // sealed corner
        if (m == 1) { wsum_q += cw[k]; continue; }         // breach: value 0
        ax += mul_wide(cw[k], src_vx[j]);
        ay += mul_wide(cw[k], src_vy[j]);
        at += mul_wide(cw[k], src_t[j]);
        wsum_q += cw[k];
    }
    if (wsum_q <= WSUM_EPS_Q) {
        return { src_vx[i0], src_vy[i0], src_t[i0] };
    }
    if (wsum_q >= FP_ONE - 4) {   // all corners live: renorm ~= identity
        return { narrow(ax), narrow(ay), narrow(at) };
    }
    const int32_t wsum_clamped = (wsum_q < WSUM_FLOOR_Q) ? WSUM_FLOOR_Q : wsum_q;
    const int32_t recip_q = reciprocal_q16_dev(wsum_clamped);
    return { mul_q16(narrow(ax), recip_q),
             mul_q16(narrow(ay), recip_q),
             mul_q16(narrow(at), recip_q) };
}

// ---- K1: fused SL advect (eos_solver.cpp substep loop a+b, + f subsumed) ----
// Per destination cell, reading ONLY the frozen src snapshots (the CPU's
// vx_src_/vy_src_/t_src_) + cmask — a pure gather, one writer per cell:
//   solid       -> u := 0, T untouched (the CPU `continue`);
//   otherwise   -> backtrace -u_src*dt_s, fused sample, u := sample,
//                  T := is_vacuum ? 0 : sample.
// Step 1f's "zero u on solid" re-pass is subsumed: this kernel already zeroes
// solid cells' u and nothing between substeps re-touches u (the interleaved
// bulk flux, P6.1, writes only gas planes) — arithmetically identical.
__global__ void sl_advect3(int32_t* __restrict__ wind_x,
                           int32_t* __restrict__ wind_y,
                           int32_t* __restrict__ temperature,
                           const int32_t* __restrict__ src_vx,
                           const int32_t* __restrict__ src_vy,
                           const int32_t* __restrict__ src_t,
                           const bool* __restrict__ solid,
                           const bool* __restrict__ is_vacuum,
                           const uint8_t* __restrict__ cmask,
                           int32_t dt_s_q, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (solid[i]) { wind_x[i] = 0; wind_y[i] = 0; continue; }
        const int y = i / w;
        const int x = i % w;
        const int32_t bx_q = -mul_q16(src_vx[i], dt_s_q);
        const int32_t by_q = -mul_q16(src_vy[i], dt_s_q);
        const FusedSample fs = eos_backtrace_sample3_q_dev(
            src_vx, src_vy, src_t, x, y, bx_q, by_q, cmask, h, w);
        wind_x[i] = fs.vx;
        wind_y[i] = fs.vy;
        temperature[i] = is_vacuum[i] ? 0 : fs.t;
    }
}

}  // namespace

uint64_t eos_sl_advect(
    int32_t* wind_x, int32_t* wind_y, int32_t* temperature,
    const bool* solid, const bool* is_vacuum,
    const float* dyn_permeability,
    int h, int w, float dt, int n_sub) {
    const int n = h * w;
    if (n <= 0 || dt <= 0.0f || n_sub < 1) return 0;

    // ---- Host scalar precompute (eos_solver.cpp, VERBATIM, in double):
    //      dt_s_d = (double)dt / n_sub; dt_s_q = quantize(dt_s_d). Constant
    //      across substeps (the CPU re-quantizes the same double each substep
    //      — computing it once here is the identical value). /fp:strict host. -
    const double dt_s_d = (double)dt / (double)n_sub;
    const q16 dt_s_q = quantize(dt_s_d);

    // ---- Device buffers (3 fields + 3 src snapshots + masks + perm + cmask).
    //      Per-call H2D/D2H (S1/S3/S4a pattern); residency is S8. -------------
    const size_t nb    = (size_t)n * sizeof(int32_t);
    const size_t nbool = (size_t)n * sizeof(bool);
    const size_t nbf   = (size_t)n * sizeof(float);

    int32_t *d_wx = nullptr, *d_wy = nullptr, *d_t = nullptr,
            *d_svx = nullptr, *d_svy = nullptr, *d_st = nullptr;
    bool *d_sol = nullptr, *d_vac = nullptr;
    float *d_perm = nullptr;
    uint8_t *d_cmask = nullptr;

    cuda_check(cudaMalloc(&d_wx, nb), "malloc wind_x");
    cuda_check(cudaMalloc(&d_wy, nb), "malloc wind_y");
    cuda_check(cudaMalloc(&d_t,  nb), "malloc temperature");
    cuda_check(cudaMalloc(&d_svx, nb), "malloc src_vx");
    cuda_check(cudaMalloc(&d_svy, nb), "malloc src_vy");
    cuda_check(cudaMalloc(&d_st,  nb), "malloc src_t");
    cuda_check(cudaMalloc(&d_sol, nbool), "malloc solid");
    cuda_check(cudaMalloc(&d_vac, nbool), "malloc is_vacuum");
    cuda_check(cudaMalloc(&d_perm, nbf), "malloc permeability");
    cuda_check(cudaMalloc(&d_cmask, (size_t)n), "malloc cmask");

    cuda_check(cudaMemcpy(d_wx, wind_x, nb, cudaMemcpyHostToDevice), "H2D wind_x");
    cuda_check(cudaMemcpy(d_wy, wind_y, nb, cudaMemcpyHostToDevice), "H2D wind_y");
    cuda_check(cudaMemcpy(d_t, temperature, nb, cudaMemcpyHostToDevice), "H2D temperature");
    cuda_check(cudaMemcpy(d_sol, solid, nbool, cudaMemcpyHostToDevice), "H2D solid");
    cuda_check(cudaMemcpy(d_vac, is_vacuum, nbool, cudaMemcpyHostToDevice), "H2D is_vacuum");
    cuda_check(cudaMemcpy(d_perm, dyn_permeability, nbf, cudaMemcpyHostToDevice), "H2D permeability");

    const int block = 256;
    const int grid = (n + block - 1) / block;

    // K0: cmask build, ONCE per tick (solid/vacuum/perm constant within it).
    sl_cmask_build<<<grid, block>>>(d_sol, d_vac, d_perm, d_cmask, n);
    cuda_check(cudaGetLastError(), "cmask launch");

    for (int s = 0; s < n_sub; ++s) {
        // Snapshot the pre-substep (vx, vy, T) into the frozen src buffers —
        // the exact int32 snapshot the CPU's vx_src_/vy_src_/t_src_ copies
        // take at the top of every substep. K1 then writes the live fields
        // while reading ONLY these (the gather's source-buffer contract).
        cuda_check(cudaMemcpy(d_svx, d_wx, nb, cudaMemcpyDeviceToDevice), "D2D src_vx");
        cuda_check(cudaMemcpy(d_svy, d_wy, nb, cudaMemcpyDeviceToDevice), "D2D src_vy");
        cuda_check(cudaMemcpy(d_st,  d_t,  nb, cudaMemcpyDeviceToDevice), "D2D src_t");
        sl_advect3<<<grid, block>>>(d_wx, d_wy, d_t, d_svx, d_svy, d_st,
                                    d_sol, d_vac, d_cmask, dt_s_q, h, w);
        cuda_check(cudaGetLastError(), "advect launch");
    }

    cuda_check(cudaDeviceSynchronize(), "sync");

    cuda_check(cudaMemcpy(wind_x, d_wx, nb, cudaMemcpyDeviceToHost), "D2H wind_x");
    cuda_check(cudaMemcpy(wind_y, d_wy, nb, cudaMemcpyDeviceToHost), "D2H wind_y");
    cuda_check(cudaMemcpy(temperature, d_t, nb, cudaMemcpyDeviceToHost), "D2H temperature");

    cudaFree(d_wx);
    cudaFree(d_wy);
    cudaFree(d_t);
    cudaFree(d_svx);
    cudaFree(d_svy);
    cudaFree(d_st);
    cudaFree(d_sol);
    cudaFree(d_vac);
    cudaFree(d_perm);
    cudaFree(d_cmask);

    // Host-side digest, byte-for-byte the CPU's last-substep digest_advect
    // expression: digest_of(wx, digest_of(wy, digest_of(T, 0))).
    return digest_of_host(wind_x, n,
           digest_of_host(wind_y, n,
           digest_of_host(temperature, n, 0)));
}

// ---- EOS P6.5: device-pointer launchers (header rationale) ------------------
// Same anonymous-namespace kernels as the isolated entry above — ONE
// transcription — launched with the identical block/grid shape on caller-owned
// device buffers, so the P6.5 orchestrator can chain them with the bulk-flux
// kernels while u/T/gas stay device-resident across the whole substep loop.
void sl_cmask_build_device(const bool* d_solid, const bool* d_vacuum,
                           const float* d_perm, uint8_t* d_cmask, int n) {
    const int block = 256;
    const int grid = (n + block - 1) / block;
    sl_cmask_build<<<grid, block>>>(d_solid, d_vacuum, d_perm, d_cmask, n);
    cuda_check(cudaGetLastError(), "cmask launch (P6.5 chained)");
}

void sl_advect3_device(int32_t* d_wind_x, int32_t* d_wind_y,
                       int32_t* d_temperature,
                       const int32_t* d_src_vx, const int32_t* d_src_vy,
                       const int32_t* d_src_t,
                       const bool* d_solid, const bool* d_vacuum,
                       const uint8_t* d_cmask,
                       int32_t dt_s_q, int h, int w) {
    const int n = h * w;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    sl_advect3<<<grid, block>>>(d_wind_x, d_wind_y, d_temperature,
                                d_src_vx, d_src_vy, d_src_t,
                                d_solid, d_vacuum, d_cmask, dt_s_q, h, w);
    cuda_check(cudaGetLastError(), "advect launch (P6.5 chained)");
}

namespace {
bool g_sl_advection_backend_cuda = false;
}
bool sl_advection_backend_is_cuda() { return g_sl_advection_backend_cuda; }
void set_sl_advection_backend_cuda(bool on) { g_sl_advection_backend_cuda = on; }

}  // namespace breach_cuda
