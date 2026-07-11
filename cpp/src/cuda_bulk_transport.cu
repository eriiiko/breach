// ============================================================================
// EOS P6.1 bulk donor-cell flux implementation — see cuda_bulk_transport.h.
// A bit-identical GPU port of bulk_flux_transport_cached (bulk_transport.cpp),
// which is itself the WaterSolver donor-cell block generalized to per-face
// coefficients — so these kernels are cuda_water.cu K3-K8 with the same
// mechanical deltas the CPU applied (per-face coeff arrays, N planes,
// solid+vacuum clamp). Precedent + determinism argument:
// docs/eos_p6_gpu_alignment_review.md §1.3; design §2.2 / §7.
//
// SEQUENTIAL-DEPENDENCE AUDIT (done before porting, per the P6 discipline):
// the CPU pass is five full-grid loops per plane; every loop reads ONLY
// buffers frozen by the previous loop (dq/scale/N) or read-only inputs
// (wind/coeff/masks), and every iteration writes ONLY its own cell / its own
// two faces:
//   stage 1  writes dq_e[i], dq_s[i]      reads N (pre-update), wind, coeff
//   stage 2  writes scale_q[i]            reads dq (frozen), N (pre-update)
//   stage 2b writes dq_e[i], dq_s[i]      reads own dq + scale_q (frozen —
//                                         scale_q is NOT written in this loop)
//   stage 3  writes N[i]                  reads dq (frozen post-scale) only
//   stage 4  writes N[i]                  reads own N + masks
// -> no intra-loop order dependence anywhere; a kernel-launch boundary at
// each loop boundary reproduces the CPU bit-for-bit. The plane loop is
// independent too (disjoint N planes; wind/coeff read-only). The all-zero-
// plane early-exit scan is host-side here exactly as on the CPU (and is
// arithmetically a no-op regardless — an all-zero plane produces all-zero
// fluxes, scale FP_ONE, unchanged N, and a clamp that re-writes zeros).
#include "cuda_bulk_transport.h"
#include "fixed_point.h"   // q16, mul_wide, scale_mag, FP_ONE, FP_SHIFT (FP_HD)
#include "cuda_fixedpoint_device.cuh"  // flux_to_dq_dev (hoisted here in P6.1)

#include <cuda_runtime.h>

#include <algorithm>   // std::min (host coefficient hoist)
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace fixedpoint;

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in bulk_flux_transport/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// ---- B1: donor-cell upwind face fluxes -> per-face dq (stage 1) ------------
// The CPU fuses water's K3 (mul_wide flux) and K4 (flux_to_dq) into one loop
// body with NO intermediate wide buffer — mirrored exactly: the Q32.32
// flux_wide lives in a register between the two truncation-identical steps.
// A solid cell / border / sealed face (coeff 0) carries dq 0 — the same zero
// the CPU's pre-fill + branch-skip left there (flux_to_dq(x, 0) == 0 anyway).
// Every thread writes BOTH its own faces -> scratch fully written, no memset.
__global__ void bulk_flux_dq(const int32_t* __restrict__ N,
                             const int32_t* __restrict__ wind_x,
                             const int32_t* __restrict__ wind_y,
                             const bool* __restrict__ solid,
                             const int32_t* __restrict__ coeffE,
                             const int32_t* __restrict__ coeffS,
                             int32_t* __restrict__ dq_e,
                             int32_t* __restrict__ dq_s, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int y = i / w;
        const int x = i % w;
        q16 d_e = 0;
        q16 d_s = 0;
        if (!solid[i]) {
            if (x < w - 1 && coeffE[i] != 0) {
                const q16 v_face = (q16)(((int64_t)wind_x[i] + wind_x[i + 1]) >> 1);
                const q16 donor = (v_face > 0) ? N[i] : N[i + 1];
                const int64_t flux_wide = mul_wide(v_face, donor);   // Q32.32
                d_e = flux_to_dq_dev(flux_wide, coeffE[i]);
            }
            if (y < h - 1 && coeffS[i] != 0) {
                const q16 v_face = (q16)(((int64_t)wind_y[i] + wind_y[i + w]) >> 1);
                const q16 donor = (v_face > 0) ? N[i] : N[i + w];
                const int64_t flux_wide = mul_wide(v_face, donor);
                d_s = flux_to_dq_dev(flux_wide, coeffS[i]);
            }
        }
        dq_e[i] = d_e;
        dq_s[i] = d_s;
    }
}

// ---- B2: per-cell OUTFLOW LIMITER factor (stage 2, mass-exactness) ----------
// Identical to cuda_water.cu K5 with depth -> N: out_sum of OUTGOING dq
// magnitudes; if out_sum > N, scale = (N << 16) / out_sum (exact int64
// divide); FP_ONE (unlimited) otherwise. Reads FROZEN dq + pre-update N;
// writes only its own scale -> no race, scratch fully written.
__global__ void bulk_scale(const int32_t* __restrict__ dq_e,
                           const int32_t* __restrict__ dq_s,
                           const int32_t* __restrict__ N,
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
        if (out_sum > (int64_t)N[i]) {
            scale_q[i] = (q16)(((int64_t)N[i] << FP_SHIFT) / out_sum);
        } else {
            scale_q[i] = FP_ONE;   // default (unlimited) IS read by B3
        }
    }
}

// ---- B3: apply the donor's scale to each face's dq (stage 2b, scale_mag) ---
// Identical to cuda_water.cu K6: reads the FROZEN scale_q (never written in
// this kernel) + its OWN dq_e[i]/dq_s[i]; updates them in place. scale_mag
// (NOT mul_q16) shrinks on the MAGNITUDE — the over-drain guard.
__global__ void bulk_scale_apply(int32_t* __restrict__ dq_e,
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

// ---- B4: apply divergence (stage 3; gather-then-apply conservative form) ---
// Identical to cuda_water.cu K7: dq_e[i] is the SAME value removed from i and
// added to i+1 — mass conserved to the LSB regardless of rounding. Reads the
// FROZEN (scaled) dq; writes its own N in place.
__global__ void bulk_diverge(int32_t* __restrict__ N,
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
        N[i] = (int32_t)((int64_t)N[i]
                         - ((int64_t)(d_e - d_w) + (int64_t)(d_s - d_n)));
    }
}

// ---- B5: clamps (stage 4) ---------------------------------------------------
// N = 0 on solid (defensive) AND vacuum (the DELIBERATE breach sink — mass
// legitimately leaves the system); else max(N, 0). Exact CPU branch order.
__global__ void bulk_clamp(int32_t* __restrict__ N,
                           const bool* __restrict__ solid,
                           const bool* __restrict__ is_vacuum, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (solid[i] || is_vacuum[i]) {
            N[i] = 0;
        } else if (N[i] < 0) {
            N[i] = 0;
        }
    }
}

}  // namespace

void bulk_flux_transport_cached(
        int32_t* gas, const bool* gas_conservative, int n_gases,
        const int32_t* wind_x, const int32_t* wind_y,
        const bool* solid, const bool* is_vacuum,
        const int32_t* coeffE, const int32_t* coeffS,
        int h, int w) {
    const int n = h * w;
    if (n <= 0 || n_gases <= 0) return;

    const size_t nb    = (size_t)n * sizeof(int32_t);
    const size_t nbool = (size_t)n * sizeof(bool);

    int32_t *d_N = nullptr, *d_wx = nullptr, *d_wy = nullptr,
            *d_coeffE = nullptr, *d_coeffS = nullptr,
            *d_dq_e = nullptr, *d_dq_s = nullptr, *d_scale = nullptr;
    bool *d_solid = nullptr, *d_vac = nullptr;

    cuda_check(cudaMalloc(&d_N, nb), "malloc N");
    cuda_check(cudaMalloc(&d_wx, nb), "malloc wind_x");
    cuda_check(cudaMalloc(&d_wy, nb), "malloc wind_y");
    cuda_check(cudaMalloc(&d_coeffE, nb), "malloc coeffE");
    cuda_check(cudaMalloc(&d_coeffS, nb), "malloc coeffS");
    cuda_check(cudaMalloc(&d_dq_e, nb), "malloc dq_e");
    cuda_check(cudaMalloc(&d_dq_s, nb), "malloc dq_s");
    cuda_check(cudaMalloc(&d_scale, nb), "malloc scale");
    cuda_check(cudaMalloc(&d_solid, nbool), "malloc solid");
    cuda_check(cudaMalloc(&d_vac, nbool), "malloc is_vacuum");

    // Shared (per-call-constant) inputs, uploaded once for all planes.
    cuda_check(cudaMemcpy(d_wx, wind_x, nb, cudaMemcpyHostToDevice), "H2D wind_x");
    cuda_check(cudaMemcpy(d_wy, wind_y, nb, cudaMemcpyHostToDevice), "H2D wind_y");
    cuda_check(cudaMemcpy(d_coeffE, coeffE, nb, cudaMemcpyHostToDevice), "H2D coeffE");
    cuda_check(cudaMemcpy(d_coeffS, coeffS, nb, cudaMemcpyHostToDevice), "H2D coeffS");
    cuda_check(cudaMemcpy(d_solid, solid, nbool, cudaMemcpyHostToDevice), "H2D solid");
    cuda_check(cudaMemcpy(d_vac, is_vacuum, nbool, cudaMemcpyHostToDevice), "H2D is_vacuum");

    const int block = 256;
    const int grid = (n + block - 1) / block;

    for (int gi = 0; gi < n_gases; ++gi) {
        if (!gas_conservative[gi]) continue;
        int32_t* N = gas + (size_t)gi * n;

        // Skip an all-zero plane — the SAME host-side early-exit scan as the
        // CPU (and a no-op arithmetically; see the audit in the file header).
        bool any = false;
        for (int i = 0; i < n; ++i) { if (N[i] != 0) { any = true; break; } }
        if (!any) continue;

        cuda_check(cudaMemcpy(d_N, N, nb, cudaMemcpyHostToDevice), "H2D N");

        // B1 face flux + dq
        bulk_flux_dq<<<grid, block>>>(d_N, d_wx, d_wy, d_solid,
                                      d_coeffE, d_coeffS, d_dq_e, d_dq_s, h, w);
        cuda_check(cudaGetLastError(), "flux_dq launch");
        // B2 limiter factor
        bulk_scale<<<grid, block>>>(d_dq_e, d_dq_s, d_N, d_scale, h, w);
        cuda_check(cudaGetLastError(), "scale launch");
        // B3 scale-apply (in-place on d_dq_e/d_dq_s)
        bulk_scale_apply<<<grid, block>>>(d_dq_e, d_dq_s, d_scale, h, w);
        cuda_check(cudaGetLastError(), "scale-apply launch");
        // B4 diverge (in-place on d_N)
        bulk_diverge<<<grid, block>>>(d_N, d_dq_e, d_dq_s, h, w);
        cuda_check(cudaGetLastError(), "diverge launch");
        // B5 clamp (in-place on d_N)
        bulk_clamp<<<grid, block>>>(d_N, d_solid, d_vac, n);
        cuda_check(cudaGetLastError(), "clamp launch");

        cuda_check(cudaDeviceSynchronize(), "sync");
        cuda_check(cudaMemcpy(N, d_N, nb, cudaMemcpyDeviceToHost), "D2H N");
    }

    cudaFree(d_N);
    cudaFree(d_wx);
    cudaFree(d_wy);
    cudaFree(d_coeffE);
    cudaFree(d_coeffS);
    cudaFree(d_dq_e);
    cudaFree(d_dq_s);
    cudaFree(d_scale);
    cudaFree(d_solid);
    cudaFree(d_vac);
}

void bulk_flux_transport(
        int32_t* gas, const bool* gas_conservative, int n_gases,
        const int32_t* wind_x, const int32_t* wind_y,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability,
        int h, int w, float dt) {
    // Legacy entry (pybind/P1-test path): hoist the per-face coefficient
    // exactly as the CPU legacy entry does (bulk_transport.cpp:47-66,
    // VERBATIM — same min/quantize/mul_q16 chain, host-side, /fp:strict),
    // then forward to the cached GPU entry. Identical host code ->
    // identical coefficients -> the A/B gate compares ONLY the kernels.
    const int n = h * w;
    const q16 dt_q = quantize((double)dt);
    std::vector<q16> coeffE(n, 0), coeffS(n, 0);
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (solid[i]) continue;
            if (x < w - 1 && !solid[i + 1]) {
                const float face_f = std::min(dyn_permeability[i], dyn_permeability[i + 1]);
                if (face_f > 0.0f)
                    coeffE[i] = mul_q16(quantize((double)face_f), dt_q);
            }
            if (y < h - 1 && !solid[i + w]) {
                const float face_f = std::min(dyn_permeability[i], dyn_permeability[i + w]);
                if (face_f > 0.0f)
                    coeffS[i] = mul_q16(quantize((double)face_f), dt_q);
            }
        }
    }
    bulk_flux_transport_cached(gas, gas_conservative, n_gases,
                               wind_x, wind_y, solid, is_vacuum,
                               coeffE.data(), coeffS.data(), h, w);
}

namespace {
bool g_bulk_flux_backend_cuda = false;
}
bool bulk_flux_backend_is_cuda() { return g_bulk_flux_backend_cuda; }
void set_bulk_flux_backend_cuda(bool on) { g_bulk_flux_backend_cuda = on; }

}  // namespace breach_cuda
