// ============================================================================
// CUDA-S3 water solver implementation — see cuda_water.h.
// A bit-identical GPU port of WaterSolver::step (water_solver.cpp lines 41-327).
// step_ripple (render-only float) is NOT ported.
// ============================================================================
#include "cuda_water.h"
#include "fixed_point.h"   // q16, quantize, mul_q16, mul_wide, recip_mul,
                           // scale_mag, make_recip, tan_poly, FP_ONE, FP_SHIFT

#include <cuda_runtime.h>

#include <algorithm>   // std::max, std::min (host precompute)
#include <cmath>       // (host precompute parity with water_solver.cpp)
#include <sstream>
#include <stdexcept>

using namespace fixedpoint;

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in water_step/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// ---- device 128-bit signed product, arithmetic-shifted right ---------------
// IMPORTANT DETERMINISM NOTE (verified at build): nvcc on Windows with the MSVC
// host compiler does NOT define __int128 (nor __SIZEOF_INT128__) for DEVICE code
// — it is a gcc/clang extension absent from the MSVC-targeting device pass. So
// the spec's "verify nvcc takes the __int128 branch" check FAILS here, and we
// take the sanctioned fallback: a device-local 128-bit helper built from CUDA's
// __mul64hi intrinsic (the signed high 64 bits of a 64x64 product). This is
// EXACTLY the hi:lo combine the host fixed_point.h MSVC paths (_mul128 in
// recip_mul / flux_to_dq) already use — so it is bit-identical to the CPU by
// construction (the same single arithmetic >>S of the full 128-bit product).
//
// mul128_shr_signed(a, b, S) == (int64_t)( (a*b) >> S ), 0 < S < 64, where a*b
// is the full SIGNED 128-bit product. lo = low 64 bits (mod 2^64), hi = signed
// high 64 bits (__mul64hi). The arithmetic 128-bit >>S re-combines them the same
// way the MSVC host code does: ((uint64_t)lo >> S) | ((uint64_t)hi << (64 - S)).
__device__ __forceinline__ int64_t mul128_shr_signed(int64_t a, int64_t b, int S) {
    const long long hi = __mul64hi((long long)a, (long long)b);   // signed hi 64
    const unsigned long long lo = (unsigned long long)((long long)a * (long long)b);
    const long long res = (long long)((lo >> S) |
                                      ((unsigned long long)hi << (64 - S)));
    return (int64_t)res;
}

// flux_to_dq — the CPU lambda (water_solver.cpp:208-230) as a __device__ helper.
// flux_wide (Q32.32) * dt_over_dx_q (Q16.16), >> 32 leaves Q16.16. The 128-bit
// intermediate (via mul128_shr_signed) is the SAME single truncation the CPU
// MSVC _mul128 path produces (proven bit-identical by
// tests/_s1_flux_truncation_check.py).
__device__ __forceinline__ q16 flux_to_dq_dev(int64_t flux_wide, q16 dt_over_dx_q) {
    return (q16)mul128_shr_signed(flux_wide, (int64_t)dt_over_dx_q, 32);
}

// recip_mul — the central-difference gradient's reciprocal multiply, as a
// __device__ helper (RECIP_SHIFT == 32). The header fixedpoint::recip_mul is
// FP_HD, but under MSVC-host nvcc its device instantiation would resolve to the
// _MSC_VER `_mul128` branch (a HOST-ONLY intrinsic) — so we call this local one
// on the device instead, bit-identical to the host _mul128 path.
__device__ __forceinline__ q16 recip_mul_dev(q16 x_q16, int64_t recip) {
    return (q16)mul128_shr_signed((int64_t)x_q16, recip, RECIP_SHIFT);
}

// ---- K1: surface potential (§1, Q16.16 metres) -----------------------------
// surface[i] = floor_at(i) + tilt_col + tilt_row + depth[i] (+ head bridge if on).
// The per-tile tilt products run in DOUBLE on the device (--fmad=false keeps them
// from contracting -> bit-identical to the CPU /fp:strict path). Every thread
// writes its own cell -> no race, no uninitialised scratch.
__global__ void water_surface(const int32_t* __restrict__ depth,
                              const int32_t* __restrict__ floor_height, // nullable
                              int has_floor,
                              const float* __restrict__ atm_f,          // nullable
                              const float* __restrict__ wave_f,         // nullable
                              int head_on, float kp_f,
                              q16 tan_tx, q16 tan_ty,
                              double cx, double cy, double dx_d,
                              int32_t* __restrict__ surface, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int y = i / w;
        const int x = i % w;
        // tilt_row / tilt_col — the CPU hoists tilt_row to the row loop (pure
        // optimization; the per-tile value is identical). Recompute both here.
        const q16 tilt_row = mul_q16(tan_ty, quantize(((double)y - cy) * dx_d));
        const q16 tilt_col = mul_q16(tan_tx, quantize(((double)x - cx) * dx_d));
        const q16 fl = has_floor ? floor_height[i] : 0;
        q16 s = fl + tilt_col + tilt_row + depth[i];
        if (head_on) {
            // FLOAT BRIDGE: atm/wave_p still float. head_f = kp_f*(atm+wp) in
            // FLOAT (--fmad=false prevents the mul(add()) from fusing), then
            // quantize((double)head_f). atm/wave null -> 0 (gated).
            const float atm_v = atm_f ? atm_f[i] : 0.0f;
            const float wp_v  = wave_f ? wave_f[i] : 0.0f;
            const float head_f = kp_f * (atm_v + wp_v);
            s += quantize((double)head_f);
        }
        surface[i] = s;
    }
}

// ---- K2: damped explicit velocity kick (§2) --------------------------------
// Central difference; Neumann mirror of the centre value at solid/out-of-bounds.
// v = 0 on solid; componentwise clamp to +-v_max. Reads the FROZEN surface +
// own vx/vy. The gradient uses recip_mul_dev (the device-local 128-bit helper,
// bit-identical to the host _mul128 recip_mul). Every cell fully written.
__global__ void water_velocity(const int32_t* __restrict__ surface,
                               int32_t* __restrict__ vx_io,
                               int32_t* __restrict__ vy_io,
                               const bool* __restrict__ solid,
                               q16 g_dt_q, q16 damp_dt_q, q16 v_max_q,
                               int64_t recip_two_dx, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (solid[i]) {
            vx_io[i] = 0;
            vy_io[i] = 0;
            continue;
        }
        const int y = i / w;
        const int x = i % w;
        const q16 s_c = surface[i];
        const q16 s_e = (x < w - 1 && !solid[i + 1]) ? surface[i + 1] : s_c;
        const q16 s_w = (x > 0     && !solid[i - 1]) ? surface[i - 1] : s_c;
        const q16 s_s = (y < h - 1 && !solid[i + w]) ? surface[i + w] : s_c;
        const q16 s_n = (y > 0     && !solid[i - w]) ? surface[i - w] : s_c;
        const q16 dsdx = recip_mul_dev((q16)(s_e - s_w), recip_two_dx);
        const q16 dsdy = recip_mul_dev((q16)(s_s - s_n), recip_two_dx);
        q16 vx = (q16)((int64_t)vx_io[i]
                       - mul_q16(g_dt_q, dsdx)
                       - mul_q16(damp_dt_q, vx_io[i]));
        q16 vy = (q16)((int64_t)vy_io[i]
                       - mul_q16(g_dt_q, dsdy)
                       - mul_q16(damp_dt_q, vy_io[i]));
        vx = max(-v_max_q, min(v_max_q, vx));
        vy = max(-v_max_q, min(v_max_q, vy));
        vx_io[i] = vx;
        vy_io[i] = vy;
    }
}

// ---- K3: donor-cell upwind face fluxes (§3, PRE-update depth) --------------
// fx[i]/fy[i] as WIDE int64 (Q32.32). Solid/border faces carry NO flux -> 0.
// Reads the UPDATED vx/vy (K2 done) + the FROZEN depth. Every cell writes both
// of its own faces (0 when the face does not exist) -> no uninitialised scratch.
__global__ void water_flux(const int32_t* __restrict__ vx,
                          const int32_t* __restrict__ vy,
                          const int32_t* __restrict__ depth,
                          const bool* __restrict__ solid,
                          int64_t* __restrict__ fx,
                          int64_t* __restrict__ fy, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int y = i / w;
        const int x = i % w;
        int64_t f_e = 0;
        int64_t f_s = 0;
        if (x < w - 1 && !solid[i] && !solid[i + 1]) {
            const q16 v_face = (q16)(((int64_t)vx[i] + vx[i + 1]) >> 1);
            const q16 donor = (v_face > 0) ? depth[i] : depth[i + 1];
            f_e = mul_wide(v_face, donor);
        }
        if (y < h - 1 && !solid[i] && !solid[i + w]) {
            const q16 v_face = (q16)(((int64_t)vy[i] + vy[i + w]) >> 1);
            const q16 donor = (v_face > 0) ? depth[i] : depth[i + w];
            f_s = mul_wide(v_face, donor);
        }
        fx[i] = f_e;
        fy[i] = f_s;
    }
}

// ---- K4: per-face depth-delta dq (the CONSERVATIVE unit) -------------------
// dq = flux_to_dq(flux). 0 when the flux is 0 (matches the CPU `if (fx[i] != 0)`
// guard exactly — the scratch is fully written, never read uninitialised).
__global__ void water_dq(const int64_t* __restrict__ fx,
                        const int64_t* __restrict__ fy,
                        q16 dt_over_dx_q,
                        int32_t* __restrict__ dq_e,
                        int32_t* __restrict__ dq_s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        dq_e[i] = (fx[i] != 0) ? flux_to_dq_dev(fx[i], dt_over_dx_q) : 0;
        dq_s[i] = (fy[i] != 0) ? flux_to_dq_dev(fy[i], dt_over_dx_q) : 0;
    }
}

// ---- K5: per-cell OUTFLOW LIMITER factor (mass-exactness) -------------------
// out_sum = sum of OUTGOING dq magnitudes; if out_sum > depth, scale = (depth<<16)
// / out_sum (an exact int64 divide). FP_ONE (unlimited) otherwise. Reads the
// FROZEN dq_e/dq_s + depth. Every cell writes its own scale -> no race.
__global__ void water_scale(const int32_t* __restrict__ dq_e,
                           const int32_t* __restrict__ dq_s,
                           const int32_t* __restrict__ depth,
                           int32_t* __restrict__ scale_q, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int y = i / w;
        const int x = i % w;
        int64_t out_sum = 0;
        if (x < w - 1 && dq_e[i] > 0)     out_sum += dq_e[i];       // east, leaving
        if (x > 0     && dq_e[i - 1] < 0) out_sum -= dq_e[i - 1];   // west, leaving
        if (y < h - 1 && dq_s[i] > 0)     out_sum += dq_s[i];       // south, leaving
        if (y > 0     && dq_s[i - w] < 0) out_sum -= dq_s[i - w];   // north, leaving
        if (out_sum > (int64_t)depth[i]) {
            scale_q[i] = (q16)(((int64_t)depth[i] << FP_SHIFT) / out_sum);
        } else {
            scale_q[i] = FP_ONE;   // default (unlimited) IS read by K6
        }
    }
}

// ---- K6: apply the donor-cell scale to each face's dq (scale_mag) ----------
// Reads the FROZEN scale_q + the frozen dq. Writes its OWN dq_e[i]/dq_s[i] (the
// east + south faces it owns), reading the donor's scale (i for outgoing +dq,
// the neighbour for incoming -dq). scale_mag (NOT mul_q16) shrinks on magnitude.
__global__ void water_scale_apply(int32_t* __restrict__ dq_e,
                                 int32_t* __restrict__ dq_s,
                                 const int32_t* __restrict__ scale_q,
                                 int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int y = i / w;
        const int x = i % w;
        if (x < w - 1 && dq_e[i] != 0)
            dq_e[i] = scale_mag(dq_e[i], (dq_e[i] > 0) ? scale_q[i] : scale_q[i + 1]);
        if (y < h - 1 && dq_s[i] != 0)
            dq_s[i] = scale_mag(dq_s[i], (dq_s[i] > 0) ? scale_q[i] : scale_q[i + w]);
    }
}

// ---- K7: apply divergence (gather-then-apply; conservative +/- form) -------
// depth[i] -= (dq_e[i] - dq_e[i-1]) + (dq_s[i] - dq_s[i-w]). Reads the FROZEN
// (scaled) dq_e/dq_s; writes its own depth in place.
__global__ void water_diverge(int32_t* __restrict__ depth,
                             const int32_t* __restrict__ dq_e,
                             const int32_t* __restrict__ dq_s, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int y = i / w;
        const int x = i % w;
        const q16 d_e = (x < w - 1) ? dq_e[i]     : 0;
        const q16 d_w = (x > 0)     ? dq_e[i - 1] : 0;
        const q16 d_s = (y < h - 1) ? dq_s[i]     : 0;
        const q16 d_n = (y > 0)     ? dq_s[i - w] : 0;
        depth[i] = (q16)((int64_t)depth[i]
                         - ((int64_t)(d_e - d_w) + (int64_t)(d_s - d_n)));
    }
}

// ---- K8: clamps (§4) -------------------------------------------------------
// depth = max(depth, 0); 0 on solid; snap to 0 below depth_eps.
__global__ void water_clamp(int32_t* __restrict__ depth,
                           const bool* __restrict__ solid,
                           q16 depth_eps_q, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        q16 d = max(depth[i], (q16)0);
        if (solid[i] || d < depth_eps_q) d = 0;
        depth[i] = d;
    }
}

}  // namespace

void water_step(
    int32_t* water_depth, int32_t* flow_vx, int32_t* flow_vy,
    const int32_t* floor_height, const float* atmosphere, const float* wave_p,
    const bool* solid, int h, int w, float dt, float tilt_x, float tilt_y,
    float g, float damping, float dx, float k_p, float v_max, float depth_eps) {
    const int n = h * w;
    if (n <= 0) return;

    // ---- Host scalar precompute (water_solver.cpp lines 54-79, VERBATIM, in
    //      double). Identical host code -> identical scalar constants; make_recip
    //      stays host-only. ------------------------------------------------------
    const double dt_d = (double)dt;
    const double dx_d = (double)dx;
    const double g_d  = (double)g;
    const double damp_d = (double)damping;

    const q16 g_dt_q    = quantize(g_d * dt_d);
    const q16 damp_dt_q = quantize(damp_d * dt_d);
    const q16 v_max_q   = quantize((double)v_max);
    const int64_t recip_two_dx = make_recip(2.0 * dx_d);
    const q16 dt_over_dx_q = quantize(dt_d / dx_d);
    const q16 depth_eps_q  = quantize((double)depth_eps);

    // tilt clamp + tan poly (water_solver.cpp:92-106).
    const double TILT_MAX = 0.610865;  // 35 deg in radians
    double txd = std::max(-TILT_MAX, std::min(TILT_MAX, (double)tilt_x));
    double tyd = std::max(-TILT_MAX, std::min(TILT_MAX, (double)tilt_y));
    const q16 tan_tx = tan_poly(quantize(txd));
    const q16 tan_ty = tan_poly(quantize(tyd));
    const double cx = 0.5 * (double)w;
    const double cy = 0.5 * (double)h;

    const bool head_on = (k_p != 0.0f);
    const float kp_f = k_p;

    // ---- Device buffers (inputs + shared scratch). -----------------------------
    const size_t nb   = (size_t)n * sizeof(int32_t);
    const size_t nb64 = (size_t)n * sizeof(int64_t);
    const size_t nbool = (size_t)n * sizeof(bool);
    const size_t nbf  = (size_t)n * sizeof(float);

    int32_t *d_depth = nullptr, *d_vx = nullptr, *d_vy = nullptr,
            *d_floor = nullptr, *d_surface = nullptr,
            *d_dq_e = nullptr, *d_dq_s = nullptr, *d_scale = nullptr;
    int64_t *d_fx = nullptr, *d_fy = nullptr;
    bool *d_solid = nullptr;
    float *d_atm = nullptr, *d_wave = nullptr;

    cuda_check(cudaMalloc(&d_depth, nb), "malloc depth");
    cuda_check(cudaMalloc(&d_vx, nb), "malloc vx");
    cuda_check(cudaMalloc(&d_vy, nb), "malloc vy");
    cuda_check(cudaMalloc(&d_surface, nb), "malloc surface");
    cuda_check(cudaMalloc(&d_dq_e, nb), "malloc dq_e");
    cuda_check(cudaMalloc(&d_dq_s, nb), "malloc dq_s");
    cuda_check(cudaMalloc(&d_scale, nb), "malloc scale");
    cuda_check(cudaMalloc(&d_fx, nb64), "malloc fx");
    cuda_check(cudaMalloc(&d_fy, nb64), "malloc fy");
    cuda_check(cudaMalloc(&d_solid, nbool), "malloc solid");
    if (floor_height) cuda_check(cudaMalloc(&d_floor, nb), "malloc floor");
    if (head_on && atmosphere) cuda_check(cudaMalloc(&d_atm, nbf), "malloc atm");
    if (head_on && wave_p) cuda_check(cudaMalloc(&d_wave, nbf), "malloc wave");

    cuda_check(cudaMemcpy(d_depth, water_depth, nb, cudaMemcpyHostToDevice), "H2D depth");
    cuda_check(cudaMemcpy(d_vx, flow_vx, nb, cudaMemcpyHostToDevice), "H2D vx");
    cuda_check(cudaMemcpy(d_vy, flow_vy, nb, cudaMemcpyHostToDevice), "H2D vy");
    cuda_check(cudaMemcpy(d_solid, solid, nbool, cudaMemcpyHostToDevice), "H2D solid");
    if (d_floor) cuda_check(cudaMemcpy(d_floor, floor_height, nb, cudaMemcpyHostToDevice), "H2D floor");
    if (d_atm)   cuda_check(cudaMemcpy(d_atm, atmosphere, nbf, cudaMemcpyHostToDevice), "H2D atm");
    if (d_wave)  cuda_check(cudaMemcpy(d_wave, wave_p, nbf, cudaMemcpyHostToDevice), "H2D wave");

    const int block = 256;
    const int grid = (n + block - 1) / block;

    // K1 surface
    water_surface<<<grid, block>>>(d_depth, d_floor, (floor_height != nullptr),
                                   d_atm, d_wave, head_on ? 1 : 0, kp_f,
                                   tan_tx, tan_ty, cx, cy, dx_d,
                                   d_surface, h, w);
    cuda_check(cudaGetLastError(), "surface launch");
    // K2 velocity (in-place on d_vx/d_vy)
    water_velocity<<<grid, block>>>(d_surface, d_vx, d_vy, d_solid,
                                    g_dt_q, damp_dt_q, v_max_q, recip_two_dx, h, w);
    cuda_check(cudaGetLastError(), "velocity launch");
    // K3 flux
    water_flux<<<grid, block>>>(d_vx, d_vy, d_depth, d_solid, d_fx, d_fy, h, w);
    cuda_check(cudaGetLastError(), "flux launch");
    // K4 dq
    water_dq<<<grid, block>>>(d_fx, d_fy, dt_over_dx_q, d_dq_e, d_dq_s, n);
    cuda_check(cudaGetLastError(), "dq launch");
    // K5 scale
    water_scale<<<grid, block>>>(d_dq_e, d_dq_s, d_depth, d_scale, h, w);
    cuda_check(cudaGetLastError(), "scale launch");
    // K6 scale-apply (in-place on d_dq_e/d_dq_s)
    water_scale_apply<<<grid, block>>>(d_dq_e, d_dq_s, d_scale, h, w);
    cuda_check(cudaGetLastError(), "scale-apply launch");
    // K7 diverge (in-place on d_depth)
    water_diverge<<<grid, block>>>(d_depth, d_dq_e, d_dq_s, h, w);
    cuda_check(cudaGetLastError(), "diverge launch");
    // K8 clamp (in-place on d_depth)
    water_clamp<<<grid, block>>>(d_depth, d_solid, depth_eps_q, n);
    cuda_check(cudaGetLastError(), "clamp launch");

    cuda_check(cudaDeviceSynchronize(), "sync");

    cuda_check(cudaMemcpy(water_depth, d_depth, nb, cudaMemcpyDeviceToHost), "D2H depth");
    cuda_check(cudaMemcpy(flow_vx, d_vx, nb, cudaMemcpyDeviceToHost), "D2H vx");
    cuda_check(cudaMemcpy(flow_vy, d_vy, nb, cudaMemcpyDeviceToHost), "D2H vy");

    cudaFree(d_depth);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_surface);
    cudaFree(d_dq_e);
    cudaFree(d_dq_s);
    cudaFree(d_scale);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_solid);
    if (d_floor) cudaFree(d_floor);
    if (d_atm)   cudaFree(d_atm);
    if (d_wave)  cudaFree(d_wave);
}

namespace {
bool g_water_backend_cuda = false;
}
bool water_backend_is_cuda() { return g_water_backend_cuda; }
void set_water_backend_cuda(bool on) { g_water_backend_cuda = on; }

}  // namespace breach_cuda
