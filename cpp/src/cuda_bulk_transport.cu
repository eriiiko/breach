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

// ---- B5: clamps (stage 4) + the AMBIENT ring reset + rail --------------------
// N = 0 on solid (defensive) AND vacuum (the DELIBERATE breach sink — mass
// legitimately leaves the system); else max(N, 0). Exact CPU branch order.
// BC (spec §1): the ambient ring is the vacuum sink's TWIN — a per-substep
// CLAMP to the reservoir value n_amb, with the boundary_flux rail (spec §5)
// recording Σ(N_pre_reset − n_amb) via a signed int64 atomicAdd (two's-
// complement on unsigned long long — the cuda_combustion.cu:157 precedent;
// integer sums are order-free, so the device total == the CPU sequential sum).
// is_ambient nullptr on space maps -> the exact legacy clamp (byte-identical).
__global__ void bulk_clamp(int32_t* __restrict__ N,
                           const bool* __restrict__ solid,
                           const bool* __restrict__ is_vacuum, int n,
                           const bool* __restrict__ is_ambient, int32_t n_amb,
                           unsigned long long* __restrict__ rail) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (solid[i] || is_vacuum[i]) {
            N[i] = 0;
        } else if (is_ambient && is_ambient[i]) {
            if (rail) atomicAdd(rail,
                (unsigned long long)((int64_t)N[i] - (int64_t)n_amb));
            N[i] = n_amb;
        } else if (N[i] < 0) {
            N[i] = 0;
        }
    }
}

// ===========================================================================
// P-E1 kernels — the energy books (design §2.1). Each is the CPU loop of the
// same name in bulk_transport.cpp, transcribed body-for-body.
// ===========================================================================

// ---- stage 2c: bank the APPLIED per-face dq (CPU's accum_dq loop) ----------
__global__ void bulk_dq_accum(int64_t* __restrict__ dqsum_e,
                              int64_t* __restrict__ dqsum_s,
                              const int32_t* __restrict__ dq_e,
                              const int32_t* __restrict__ dq_s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        dqsum_e[i] += (int64_t)dq_e[i];
        dqsum_s[i] += (int64_t)dq_s[i];
    }
}

// The participation predicate (CPU e_participates, verbatim).
__device__ __forceinline__ bool e_part_dev(int i, const bool* solid,
                                           const bool* ts, const bool* is_vacuum,
                                           const bool* is_ambient) {
    return !solid[i] && !ts[i] && !is_vacuum[i]
           && !(is_ambient != nullptr && is_ambient[i]);
}

// ---- n_bulk accumulate (one plane per launch; the CPU's inner gi sum) ------
__global__ void bulk_nb_accum(int64_t* __restrict__ nb,
                              const int32_t* __restrict__ N, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        nb[i] += (int64_t)N[i];
    }
}

// arc #54 §2.7 row 1: floordiv(dq * E, N) — EXACT, in 64 bits only (the
// bulk_transport.cpp `price_face` twin, verbatim).
__device__ __forceinline__ int64_t price_face_dev(int64_t dq, int64_t e_i, int64_t n_i) {
    const int64_t q = floordiv_q(e_i, n_i);
    const int64_t r = e_i - q * n_i;              // 0 <= r < n_i (floor form)
    return dq * q + floordiv_q(dq * r, n_i);
}

// ---- stage 3: e apply, GATHER form (CPU face order E, W, S, N) -------------
// arc #54 (design §2.7 row 1): prices faces off the LIVE `gas_energy`
// snapshot (e_pre/n_pre, frozen PRE this call's mass flux — bulk_transport.
// cpp's `donate`/`receive` closures, ported to plain branches: a
// non-participating donor/receiver has no gas_energy, so an outgoing debit
// is priced at the DONOR's own (e_pre, n_pre) and an incoming credit from a
// non-participating neighbour is MINTED at `t_amb_raw` absolute).
__global__ void bulk_e_apply_v2(int64_t* __restrict__ gas_energy,
                                const int64_t* __restrict__ e_pre,
                                const int64_t* __restrict__ n_pre,
                                const int64_t* __restrict__ dqsum_e,
                                const int64_t* __restrict__ dqsum_s,
                                const bool* __restrict__ solid,
                                const bool* __restrict__ ts,
                                const bool* __restrict__ is_vacuum,
                                const bool* __restrict__ is_ambient,
                                int32_t t_amb_raw,
                                unsigned long long* __restrict__ cnt,
                                int h, int w) {
    const int n = h * w;
    const int64_t t_amb64 = (int64_t)t_amb_raw;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (!e_part_dev(i, solid, ts, is_vacuum, is_ambient)) continue;
        const int y = i / w;
        const int x = i % w;
        const int64_t e_own = e_pre[i], n_own = n_pre[i];
        int64_t de = 0;
        if (x < w - 1) {                       // EAST face of i
            const int64_t q = dqsum_e[i];
            if (q > 0) {
                const int64_t phi = (n_own >= 1) ? price_face_dev(q, e_own, n_own) : 0;
                de -= phi;
                if (ts[i + 1]) atomicAdd(&cnt[0], (unsigned long long)phi);   // e_ts_residual
            } else if (q < 0) {
                const int64_t qq = -q;
                de += !e_part_dev(i + 1, solid, ts, is_vacuum, is_ambient)
                    ? qq * t_amb64
                    : ((n_pre[i + 1] >= 1) ? price_face_dev(qq, e_pre[i + 1], n_pre[i + 1]) : 0);
            }
        }
        if (x > 0) {                           // WEST face of i
            const int64_t q = dqsum_e[i - 1];
            if (q > 0) {
                de += !e_part_dev(i - 1, solid, ts, is_vacuum, is_ambient)
                    ? q * t_amb64
                    : ((n_pre[i - 1] >= 1) ? price_face_dev(q, e_pre[i - 1], n_pre[i - 1]) : 0);
            } else if (q < 0) {
                const int64_t qq = -q;
                const int64_t phi = (n_own >= 1) ? price_face_dev(qq, e_own, n_own) : 0;
                de -= phi;
                if (ts[i - 1]) atomicAdd(&cnt[0], (unsigned long long)phi);
            }
        }
        if (y < h - 1) {                       // SOUTH face of i
            const int64_t q = dqsum_s[i];
            if (q > 0) {
                const int64_t phi = (n_own >= 1) ? price_face_dev(q, e_own, n_own) : 0;
                de -= phi;
                if (ts[i + w]) atomicAdd(&cnt[0], (unsigned long long)phi);
            } else if (q < 0) {
                const int64_t qq = -q;
                de += !e_part_dev(i + w, solid, ts, is_vacuum, is_ambient)
                    ? qq * t_amb64
                    : ((n_pre[i + w] >= 1) ? price_face_dev(qq, e_pre[i + w], n_pre[i + w]) : 0);
            }
        }
        if (y > 0) {                           // NORTH face of i
            const int64_t q = dqsum_s[i - w];
            if (q > 0) {
                de += !e_part_dev(i - w, solid, ts, is_vacuum, is_ambient)
                    ? q * t_amb64
                    : ((n_pre[i - w] >= 1) ? price_face_dev(q, e_pre[i - w], n_pre[i - w]) : 0);
            } else if (q < 0) {
                const int64_t qq = -q;
                const int64_t phi = (n_own >= 1) ? price_face_dev(qq, e_own, n_own) : 0;
                de -= phi;
                if (ts[i - w]) atomicAdd(&cnt[0], (unsigned long long)phi);
            }
        }
        gas_energy[i] += de;
        atomicAdd(&cnt[5], (unsigned long long)de);   // e_transport_net (arc #54 §2.8)
    }
}

// ---- stage 4: the MIRROR REFRESH (arc #54 §2.6/§2.7 row 1) ----------------
// T := floordiv(E, n_bulk_new) - T_AMB_raw, a MIRROR-ONLY read — the
// authoritative T_MIN/T_MAX_PHYS rails (with their gas_energy write-back)
// are the EOS's once-per-tick recovery (design §2.6); this clamp only bounds
// the int32 narrow and books nothing. The N_EPS wipe stays authoritative
// here (a cell with no capacity to divide by cannot carry energy forward).
__global__ void bulk_e_recover_v2(int32_t* __restrict__ temperature,
                                  int64_t* __restrict__ gas_energy,
                                  const int64_t* __restrict__ nb_new,
                                  const int64_t* __restrict__ dqsum_e,
                                  const int64_t* __restrict__ dqsum_s,
                                  const bool* __restrict__ solid,
                                  const bool* __restrict__ ts,
                                  const bool* __restrict__ is_vacuum,
                                  const bool* __restrict__ is_ambient,
                                  int32_t t_min_q, int32_t t_max_phys_q,
                                  int32_t t_amb_raw,
                                  unsigned long long* __restrict__ cnt,
                                  int h, int w) {
    const int n = h * w;
    const int64_t t_amb64 = (int64_t)t_amb_raw;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (!e_part_dev(i, solid, ts, is_vacuum, is_ambient)) {
            // T-WRITE guard (ruling A1) + the vacuum/ring wipe, verbatim.
            if (!solid[i] && !ts[i]) temperature[i] = 0;
            continue;
        }
        const int y = i / w;
        const int x = i % w;
        const int64_t n_new = nb_new[i];
        const bool active =
               (x < w - 1 && dqsum_e[i]     != 0)
            || (x > 0     && dqsum_e[i - 1] != 0)
            || (y < h - 1 && dqsum_s[i]     != 0)
            || (y > 0     && dqsum_s[i - w] != 0);
        if (active) {
            atomicAdd(&cnt[3], (unsigned long long)(int64_t)1);   // n_active_flux
            atomicAdd(&cnt[4], (unsigned long long)n_new);        // n_bulk_active_sum
        }
        if (n_new < (int64_t)1) {              // N_EPS = 1 raw count
            const int64_t e_amb = n_new * t_amb64;
            atomicAdd(&cnt[1], (unsigned long long)(gas_energy[i] - e_amb));   // e_wipe_sum
            gas_energy[i] = e_amb;
            temperature[i] = 0;
            continue;
        }
        int64_t t_new = floordiv_q(gas_energy[i], n_new) - t_amb64;
        // MIRROR-ONLY clamps: no write-back, no booking (see the stage header).
        if (t_new < (int64_t)t_min_q)           t_new = (int64_t)t_min_q;
        else if (t_new > (int64_t)t_max_phys_q) t_new = (int64_t)t_max_phys_q;
        temperature[i] = (int32_t)t_new;
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
        // B5 clamp (in-place on d_N). Isolated entry: no ambient ring.
        bulk_clamp<<<grid, block>>>(d_N, d_solid, d_vac, n, nullptr, 0, nullptr);
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

// ---- EOS P6.5: device-pointer launcher (header rationale) -------------------
// The SAME five anonymous-namespace kernels as the per-call loop above — ONE
// transcription — for one conservative plane on caller-owned device buffers.
// Launch order B1..B5 on the default stream: each kernel boundary is the
// CPU's loop boundary, exactly as in the isolated entry.
void bulk_flux_plane_device(
        int32_t* d_N,
        const int32_t* d_wind_x, const int32_t* d_wind_y,
        const bool* d_solid, const bool* d_is_vacuum,
        const int32_t* d_coeffE, const int32_t* d_coeffS,
        int32_t* d_dq_e, int32_t* d_dq_s, int32_t* d_scale,
        int h, int w,
        const bool* d_is_ambient, int32_t n_amb, unsigned long long* d_rail,
        int64_t* d_dqsum_e, int64_t* d_dqsum_s) {
    const int n = h * w;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    bulk_flux_dq<<<grid, block>>>(d_N, d_wind_x, d_wind_y, d_solid,
                                  d_coeffE, d_coeffS, d_dq_e, d_dq_s, h, w);
    cuda_check(cudaGetLastError(), "flux_dq launch (P6.5 chained)");
    bulk_scale<<<grid, block>>>(d_dq_e, d_dq_s, d_N, d_scale, h, w);
    cuda_check(cudaGetLastError(), "scale launch (P6.5 chained)");
    bulk_scale_apply<<<grid, block>>>(d_dq_e, d_dq_s, d_scale, h, w);
    cuda_check(cudaGetLastError(), "scale-apply launch (P6.5 chained)");
    // P-E1 stage 2c: bank the APPLIED dq — AFTER the limiter/scale_mag and
    // BEFORE the divergence apply, exactly where the CPU banks it.
    if (d_dqsum_e && d_dqsum_s) {
        bulk_dq_accum<<<grid, block>>>(d_dqsum_e, d_dqsum_s, d_dq_e, d_dq_s, n);
        cuda_check(cudaGetLastError(), "dq_accum launch (P-E1)");
    }
    bulk_diverge<<<grid, block>>>(d_N, d_dq_e, d_dq_s, h, w);
    cuda_check(cudaGetLastError(), "diverge launch (P6.5 chained)");
    // BC: the ambient ring reset + rail (nullptr/0/nullptr = space path).
    bulk_clamp<<<grid, block>>>(d_N, d_solid, d_is_vacuum, n,
                                d_is_ambient, n_amb, d_rail);
    cuda_check(cudaGetLastError(), "clamp launch (P6.5 chained)");
}

// ---- P-E1: the energy-books orchestration (header contract) -----------------
// Four pinned stages, each a launch (or a short launch run) whose boundary IS
// the CPU's loop boundary — the same argument that makes B1..B5 bit-identical.
// Stage 2's per-plane chain is the UNCHANGED mass pass, so gas planes stay
// byte-for-byte what they were before this patch.
void bulk_flux_energy_transport_device(
        int32_t* const* d_gas_planes, int n_cons,
        int32_t* d_temperature,
        int64_t* d_gas_energy, int32_t t_amb_raw, int32_t t_max_phys_q,
        const int32_t* d_wind_x, const int32_t* d_wind_y,
        const bool* d_solid, const bool* d_is_vacuum, const bool* d_ts,
        const int32_t* d_coeffE, const int32_t* d_coeffS,
        int32_t t_min_q, int h, int w,
        int64_t* d_e, int64_t* d_nb, int64_t* d_dqsum_e, int64_t* d_dqsum_s,
        int32_t* d_dq_e, int32_t* d_dq_s, int32_t* d_scale,
        unsigned long long* d_ecnt,
        const bool* d_is_ambient, const int32_t* n_amb_cons,
        unsigned long long* const* d_rail) {
    const int n = h * w;
    if (n <= 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    const size_t n8 = (size_t)n * sizeof(int64_t);

    // ---- stage 1: snapshot (E, n_bulk) PRE-flux (arc #54 design §2.7) -----
    // `d_e` is now a PLAIN SNAPSHOT of the live `gas_energy` (D2D copy) —
    // no rebuild from T (the pre-#54 `bulk_e_build`, deleted).
    cuda_check(cudaMemcpy(d_e, d_gas_energy, n8, cudaMemcpyDeviceToDevice),
              "D2D e_pre snapshot");
    cuda_check(cudaMemset(d_nb, 0, n8), "memset nb (pre)");
    for (int k = 0; k < n_cons; ++k) {
        bulk_nb_accum<<<grid, block>>>(d_nb, d_gas_planes[k], n);
        cuda_check(cudaGetLastError(), "nb_accum (pre)");
    }

    // ---- stage 2: the mass flux, banking the APPLIED per-face dq ----------
    cuda_check(cudaMemset(d_dqsum_e, 0, n8), "memset dqsum_e");
    cuda_check(cudaMemset(d_dqsum_s, 0, n8), "memset dqsum_s");
    for (int k = 0; k < n_cons; ++k) {
        bulk_flux_plane_device(
            d_gas_planes[k], d_wind_x, d_wind_y, d_solid, d_is_vacuum,
            d_coeffE, d_coeffS, d_dq_e, d_dq_s, d_scale, h, w,
            d_is_ambient, n_amb_cons ? n_amb_cons[k] : 0,
            d_rail ? d_rail[k] : nullptr,
            d_dqsum_e, d_dqsum_s);
    }

    // ---- stage 3: e apply (gather), priced off the LIVE energy field ------
    bulk_e_apply_v2<<<grid, block>>>(d_gas_energy, d_e, d_nb,
                                     d_dqsum_e, d_dqsum_s,
                                     d_solid, d_ts, d_is_vacuum, d_is_ambient,
                                     t_amb_raw, d_ecnt, h, w);
    cuda_check(cudaGetLastError(), "e_apply");

    // ---- stage 4: the mirror refresh (n_bulk POST, T = floordiv(E, n)) ----
    cuda_check(cudaMemset(d_nb, 0, n8), "memset nb (post)");
    for (int k = 0; k < n_cons; ++k) {
        bulk_nb_accum<<<grid, block>>>(d_nb, d_gas_planes[k], n);
        cuda_check(cudaGetLastError(), "nb_accum (post)");
    }
    bulk_e_recover_v2<<<grid, block>>>(d_temperature, d_gas_energy, d_nb,
                                       d_dqsum_e, d_dqsum_s,
                                       d_solid, d_ts, d_is_vacuum, d_is_ambient,
                                       t_min_q, t_max_phys_q, t_amb_raw,
                                       d_ecnt, h, w);
    cuda_check(cudaGetLastError(), "e_recover");
}

namespace {
bool g_bulk_flux_backend_cuda = false;
}
bool bulk_flux_backend_is_cuda() { return g_bulk_flux_backend_cuda; }
void set_bulk_flux_backend_cuda(bool on) { g_bulk_flux_backend_cuda = on; }

}  // namespace breach_cuda
