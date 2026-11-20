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
#include "cuda_resident.h"   // trace_tail_launch_resident / trace_tail_resident (P2b)
#include "bulk_transport.h"  // trace_diffusion_dd_q (the CPU tail's host fold, shared)
#include "trace_decay.h"     // the shared decay rule + its host fold (P2b)
#include "fixed_point.h"   // q16, mul_wide, scale_mag, FP_ONE, FP_SHIFT (FP_HD)
#include "cuda_fixedpoint_device.cuh"  // flux_to_dq_dev (hoisted here in P6.1)
#include "cuda_batch.cuh"               // #29 P5: the (N, h, w) launch shape

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
                             int32_t* __restrict__ dq_s, int h, int w,
                             size_t gs, const EnvSub* __restrict__ es, int sub) {
    if (sub_off(es, sub)) return;   // #29 P5: a masked env does nothing
    const int n = h * w;
    N = bptr(N, gs); wind_x = bptr(wind_x, n); wind_y = bptr(wind_y, n);
    solid = bptr(solid, n); coeffE = bptr(coeffE, n); coeffS = bptr(coeffS, n);
    dq_e = bptr(dq_e, n); dq_s = bptr(dq_s, n);
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
                           int32_t* __restrict__ scale_q, int h, int w,
                           size_t gs, const EnvSub* __restrict__ es, int sub) {
    if (sub_off(es, sub)) return;
    const int n = h * w;
    dq_e = bptr(dq_e, n); dq_s = bptr(dq_s, n); N = bptr(N, gs);
    scale_q = bptr(scale_q, n);
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
                                 int h, int w,
                                 const EnvSub* __restrict__ es, int sub) {
    if (sub_off(es, sub)) return;
    const int n = h * w;
    dq_e = bptr(dq_e, n); dq_s = bptr(dq_s, n); scale_q = bptr(scale_q, n);
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
                             const int32_t* __restrict__ dq_s, int h, int w,
                             size_t gs, const EnvSub* __restrict__ es, int sub) {
    if (sub_off(es, sub)) return;
    const int n = h * w;
    N = bptr(N, gs); dq_e = bptr(dq_e, n); dq_s = bptr(dq_s, n);
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
                           unsigned long long* __restrict__ rail,
                           size_t gs, size_t rstride,
                           const EnvSub* __restrict__ es, int sub) {
    if (sub_off(es, sub)) return;
    N = bptr(N, gs); solid = bptr(solid, (size_t)n); is_vacuum = bptr(is_vacuum, (size_t)n);
    is_ambient = bptr(is_ambient, (size_t)n); rail = bptr(rail, rstride);
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
                              const int32_t* __restrict__ dq_s, int n,
                              const EnvSub* __restrict__ es, int sub) {
    if (sub_off(es, sub)) return;
    dqsum_e = bptr(dqsum_e, (size_t)n); dqsum_s = bptr(dqsum_s, (size_t)n);
    dq_e = bptr(dq_e, (size_t)n); dq_s = bptr(dq_s, (size_t)n);
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
                              const int32_t* __restrict__ N, int n,
                              size_t gs, const EnvSub* __restrict__ es, int sub) {
    if (sub_off(es, sub)) return;
    nb = bptr(nb, (size_t)n); N = bptr(N, gs);
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
                                int h, int w,
                                size_t cstride, const EnvSub* __restrict__ es, int sub) {
    if (sub_off(es, sub)) return;
    const int n = h * w;
    gas_energy = bptr(gas_energy, n); e_pre = bptr(e_pre, n); n_pre = bptr(n_pre, n);
    dqsum_e = bptr(dqsum_e, n); dqsum_s = bptr(dqsum_s, n);
    solid = bptr(solid, n); ts = bptr(ts, n); is_vacuum = bptr(is_vacuum, n);
    is_ambient = bptr(is_ambient, n); cnt = bptr(cnt, cstride);
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
                                  int h, int w,
                                  size_t cstride, const EnvSub* __restrict__ es, int sub) {
    if (sub_off(es, sub)) return;
    const int n = h * w;
    temperature = bptr(temperature, n); gas_energy = bptr(gas_energy, n);
    nb_new = bptr(nb_new, n); dqsum_e = bptr(dqsum_e, n); dqsum_s = bptr(dqsum_s, n);
    solid = bptr(solid, n); ts = bptr(ts, n); is_vacuum = bptr(is_vacuum, n);
    is_ambient = bptr(is_ambient, n); cnt = bptr(cnt, cstride);
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

// ===========================================================================
// Smoke transport v2 (#12, design §2.1) — the TRACE planes ride the bulk face
// flux. Stage 3b / 3c, each the CPU loop of the same name in
// bulk_transport.cpp (bulk_flux_energy_transport_cached) transcribed
// body-for-body. The CPU is THE oracle (P1).
// ===========================================================================

// The TRACE participation predicate (CPU t_participates, verbatim): wider than
// e_part_dev by the thermal-solid (crate) tiles — air seeps through their pores
// and the trace goes with it (D3).
__device__ __forceinline__ bool t_part_dev(int i, const bool* solid,
                                           const bool* is_vacuum,
                                           const bool* is_ambient) {
    return !solid[i] && !is_vacuum[i]
           && !(is_ambient != nullptr && is_ambient[i]);
}

// ---- stage 3b: TRACE apply, GATHER form (CPU face order E, W, S, N) --------
// phi = price_face(dq, S_pre[donor], N_pre[donor]), the stage-3 energy shape
// with S in E's place. Reads ONLY frozen planes (s_pre = the plane's pre-flux
// snapshot, n_pre = d_nb, the pre-flux bulk N; both dqsum planes) and writes
// ONLY its own S[i] -> no race, no atomics on S. A donation to a
// non-participant (vacuum / ring) is the EXPORT, booked by the donor into the
// plane's vent slot via the unsigned-long-long atomicAdd idiom (integer sums
// are order-free, so the device total == the CPU sequential sum); inflow from
// vacuum / ring carries no trace. n_pre is read only at participating cells
// (the loop gate and the receive guard), so the stale value a solid cell's
// d_nb may hold is never seen — the CPU holds 0 there, unread.
__global__ void trace_apply(int32_t* __restrict__ S,
                            const int32_t* __restrict__ s_pre,
                            const int64_t* __restrict__ n_pre,
                            const int64_t* __restrict__ dqsum_e,
                            const int64_t* __restrict__ dqsum_s,
                            const bool* __restrict__ solid,
                            const bool* __restrict__ is_vacuum,
                            const bool* __restrict__ is_ambient,
                            unsigned long long* __restrict__ vent,
                            unsigned long long* __restrict__ clip,
                            unsigned long long* __restrict__ neg,
                            int h, int w,
                            size_t gs, size_t tstride,
                            const EnvSub* __restrict__ es, int sub) {
    if (sub_off(es, sub)) return;
    const int n = h * w;
    S = bptr(S, gs); s_pre = bptr(s_pre, n); n_pre = bptr(n_pre, n);
    dqsum_e = bptr(dqsum_e, n); dqsum_s = bptr(dqsum_s, n);
    solid = bptr(solid, n); is_vacuum = bptr(is_vacuum, n); is_ambient = bptr(is_ambient, n);
    vent = bptr(vent, tstride); clip = bptr(clip, tstride); neg = bptr(neg, tstride);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (!t_part_dev(i, solid, is_vacuum, is_ambient)) continue;
        const int y = i / w;
        const int x = i % w;
        const int64_t s_own = s_pre[i], n_own = n_pre[i];
        int64_t ds = 0;
        int64_t vented = 0;
        if (x < w - 1) {                       // EAST face of i
            const int64_t q = dqsum_e[i];
            if (q > 0) {
                const int64_t phi = (n_own >= 1) ? price_face_dev(q, s_own, n_own) : 0;
                ds -= phi;
                if (!t_part_dev(i + 1, solid, is_vacuum, is_ambient)) vented += phi;
            } else if (q < 0) {
                ds += !t_part_dev(i + 1, solid, is_vacuum, is_ambient) ? 0
                    : ((n_pre[i + 1] >= 1)
                        ? price_face_dev(-q, (int64_t)s_pre[i + 1], n_pre[i + 1]) : 0);
            }
        }
        if (x > 0) {                           // WEST face of i
            const int64_t q = dqsum_e[i - 1];
            if (q > 0) {
                ds += !t_part_dev(i - 1, solid, is_vacuum, is_ambient) ? 0
                    : ((n_pre[i - 1] >= 1)
                        ? price_face_dev(q, (int64_t)s_pre[i - 1], n_pre[i - 1]) : 0);
            } else if (q < 0) {
                const int64_t phi = (n_own >= 1) ? price_face_dev(-q, s_own, n_own) : 0;
                ds -= phi;
                if (!t_part_dev(i - 1, solid, is_vacuum, is_ambient)) vented += phi;
            }
        }
        if (y < h - 1) {                       // SOUTH face of i
            const int64_t q = dqsum_s[i];
            if (q > 0) {
                const int64_t phi = (n_own >= 1) ? price_face_dev(q, s_own, n_own) : 0;
                ds -= phi;
                if (!t_part_dev(i + w, solid, is_vacuum, is_ambient)) vented += phi;
            } else if (q < 0) {
                ds += !t_part_dev(i + w, solid, is_vacuum, is_ambient) ? 0
                    : ((n_pre[i + w] >= 1)
                        ? price_face_dev(-q, (int64_t)s_pre[i + w], n_pre[i + w]) : 0);
            }
        }
        if (y > 0) {                           // NORTH face of i
            const int64_t q = dqsum_s[i - w];
            if (q > 0) {
                ds += !t_part_dev(i - w, solid, is_vacuum, is_ambient) ? 0
                    : ((n_pre[i - w] >= 1)
                        ? price_face_dev(q, (int64_t)s_pre[i - w], n_pre[i - w]) : 0);
            } else if (q < 0) {
                const int64_t phi = (n_own >= 1) ? price_face_dev(-q, s_own, n_own) : 0;
                ds -= phi;
                if (!t_part_dev(i - w, solid, is_vacuum, is_ambient)) vented += phi;
            }
        }
        // §2.4: S >= 0 by construction (Σ_f phi_f <= S_pre: the mass limiter
        // bounds Σ_f dq_f by the donor's own N_pre, floordiv never rounds up).
        // #88 P2: the int32 narrow is the counted saturation, the CPU twin's
        // call ONCE per cell on the net (A4): the excess a receiver cannot
        // store is booked in `clip`, a negative net counted in `neg` (never
        // clamped; every CUDA gate asserts 0). Both slots are >= 0, so the
        // unsigned-long-long atomicAdd is the plain sum.
        int64_t c = 0, ng = 0;
        S[i] = fixedpoint::trace_add_sat((int32_t)s_own, ds, &c, &ng);
        if (vented != 0) atomicAdd(vent, (unsigned long long)vented);
        if (c != 0)  atomicAdd(clip, (unsigned long long)c);
        if (ng != 0) atomicAdd(neg, (unsigned long long)ng);
    }
}

// ---- stage 3c: TRACE wipe, on the POST-flux bulk N (CPU's stage-3c loop) ----
// A participating cell with less than N_EPS (= 1 raw count) of air has nothing
// left to carry its trace: destroyed, booked in the plane's wipe slot. `nb_post`
// is d_nb re-accumulated by stage 4 — the CPU's n_new, the same Σ over the
// conservative planes. Own-cell write, counter via atomicAdd.
__global__ void trace_wipe(int32_t* __restrict__ S,
                           const int64_t* __restrict__ nb_post,
                           const bool* __restrict__ solid,
                           const bool* __restrict__ is_vacuum,
                           const bool* __restrict__ is_ambient,
                           unsigned long long* __restrict__ wipe, int n,
                           size_t gs, size_t tstride,
                           const EnvSub* __restrict__ es, int sub) {
    if (sub_off(es, sub)) return;
    S = bptr(S, gs); nb_post = bptr(nb_post, (size_t)n);
    solid = bptr(solid, (size_t)n); is_vacuum = bptr(is_vacuum, (size_t)n);
    is_ambient = bptr(is_ambient, (size_t)n); wipe = bptr(wipe, tstride);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (!t_part_dev(i, solid, is_vacuum, is_ambient)) continue;
        if (nb_post[i] >= 1) continue;
        const int32_t s = S[i];
        if (s == 0) continue;
        atomicAdd(wipe, (unsigned long long)(int64_t)s);
        S[i] = 0;
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
                                      d_coeffE, d_coeffS, d_dq_e, d_dq_s, h, w,
                                      (size_t)n, nullptr, 0);
        cuda_check(cudaGetLastError(), "flux_dq launch");
        // B2 limiter factor
        bulk_scale<<<grid, block>>>(d_dq_e, d_dq_s, d_N, d_scale, h, w,
                                    (size_t)n, nullptr, 0);
        cuda_check(cudaGetLastError(), "scale launch");
        // B3 scale-apply (in-place on d_dq_e/d_dq_s)
        bulk_scale_apply<<<grid, block>>>(d_dq_e, d_dq_s, d_scale, h, w, nullptr, 0);
        cuda_check(cudaGetLastError(), "scale-apply launch");
        // B4 diverge (in-place on d_N)
        bulk_diverge<<<grid, block>>>(d_N, d_dq_e, d_dq_s, h, w, (size_t)n, nullptr, 0);
        cuda_check(cudaGetLastError(), "diverge launch");
        // B5 clamp (in-place on d_N). Isolated entry: no ambient ring.
        bulk_clamp<<<grid, block>>>(d_N, d_solid, d_vac, n, nullptr, 0, nullptr,
                                    (size_t)n, 0, nullptr, 0);
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
        int64_t* d_dqsum_e, int64_t* d_dqsum_s, const BulkBatch& bb) {
    // #29 P5: (N, h, w) — every launch covers the batch's envs on the tick
    // stream; the gas plane rebases by bb.gs, a masked env does nothing.
    const int n = h * w;
    const int block = 256;
    const dim3 grid = bgrid(n, bb.n_env, block);
    const size_t gs = bb.gs ? bb.gs : (size_t)n;
    cudaStream_t st = tstream();
    bulk_flux_dq<<<grid, block, 0, st>>>(d_N, d_wind_x, d_wind_y, d_solid,
                                  d_coeffE, d_coeffS, d_dq_e, d_dq_s, h, w,
                                  gs, bb.es, bb.sub);
    cuda_check(cudaGetLastError(), "flux_dq launch (P6.5 chained)");
    bulk_scale<<<grid, block, 0, st>>>(d_dq_e, d_dq_s, d_N, d_scale, h, w, gs, bb.es, bb.sub);
    cuda_check(cudaGetLastError(), "scale launch (P6.5 chained)");
    bulk_scale_apply<<<grid, block, 0, st>>>(d_dq_e, d_dq_s, d_scale, h, w, bb.es, bb.sub);
    cuda_check(cudaGetLastError(), "scale-apply launch (P6.5 chained)");
    // P-E1 stage 2c: bank the APPLIED dq — AFTER the limiter/scale_mag and
    // BEFORE the divergence apply, exactly where the CPU banks it.
    if (d_dqsum_e && d_dqsum_s) {
        bulk_dq_accum<<<grid, block, 0, st>>>(d_dqsum_e, d_dqsum_s, d_dq_e, d_dq_s, n,
                                              bb.es, bb.sub);
        cuda_check(cudaGetLastError(), "dq_accum launch (P-E1)");
    }
    bulk_diverge<<<grid, block, 0, st>>>(d_N, d_dq_e, d_dq_s, h, w, gs, bb.es, bb.sub);
    cuda_check(cudaGetLastError(), "diverge launch (P6.5 chained)");
    // BC: the ambient ring reset + rail (nullptr/0/nullptr = space path).
    bulk_clamp<<<grid, block, 0, st>>>(d_N, d_solid, d_is_vacuum, n,
                                d_is_ambient, n_amb, d_rail, gs, bb.rstride,
                                bb.es, bb.sub);
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
        unsigned long long* const* d_rail,
        int32_t* const* d_trace_planes, int n_trace, int32_t* d_spre,
        unsigned long long* d_tvent, unsigned long long* d_twipe,
        unsigned long long* d_tclip, unsigned long long* d_tneg,
        const BulkBatch& bb) {
    const int n = h * w;
    if (n <= 0) return;
    const int block = 256;
    // #29 P5: (N, h, w). Every scratch plane is (N, h, w); the gas planes are
    // the env's (n_gases, h, w) block (stride bb.gs); the books are per env
    // (strides bb.cstride / bb.tstride); a masked env (bb.es, bb.sub) does
    // nothing in any kernel. The whole-batch copies / memsets of SCRATCH below
    // touch no env's state, masked or not.
    const int N = bb.n_env > 0 ? bb.n_env : 1;
    const dim3 grid = bgrid(n, N, block);
    const size_t gs = bb.gs ? bb.gs : (size_t)n;
    const size_t cst = bb.cstride, tst = bb.tstride;
    cudaStream_t st = tstream();
    const size_t n8 = (size_t)n * sizeof(int64_t) * (size_t)N;
    // Smoke transport v2 (#12): the trace stages run only when the caller hands
    // trace planes (the chained path its live ones, the resident path all).
    const bool do_trace = (d_trace_planes != nullptr && n_trace > 0
                           && d_spre != nullptr && d_tvent != nullptr
                           && d_twipe != nullptr && d_tclip != nullptr
                           && d_tneg != nullptr);

    // ---- stage 1: snapshot (E, n_bulk) PRE-flux (arc #54 design §2.7) -----
    // `d_e` is now a PLAIN SNAPSHOT of the live `gas_energy` (D2D copy) —
    // no rebuild from T (the pre-#54 `bulk_e_build`, deleted).
    cuda_check(cudaMemcpyAsync(d_e, d_gas_energy, n8, cudaMemcpyDeviceToDevice, st),
              "D2D e_pre snapshot");
    cuda_check(cudaMemsetAsync(d_nb, 0, n8, st), "memset nb (pre)");
    for (int k = 0; k < n_cons; ++k) {
        bulk_nb_accum<<<grid, block, 0, st>>>(d_nb, d_gas_planes[k], n, gs, bb.es, bb.sub);
        cuda_check(cudaGetLastError(), "nb_accum (pre)");
    }

    // ---- stage 2: the mass flux, banking the APPLIED per-face dq ----------
    cuda_check(cudaMemsetAsync(d_dqsum_e, 0, n8, st), "memset dqsum_e");
    cuda_check(cudaMemsetAsync(d_dqsum_s, 0, n8, st), "memset dqsum_s");
    for (int k = 0; k < n_cons; ++k) {
        BulkBatch pb = bb;
        bulk_flux_plane_device(
            d_gas_planes[k], d_wind_x, d_wind_y, d_solid, d_is_vacuum,
            d_coeffE, d_coeffS, d_dq_e, d_dq_s, d_scale, h, w,
            d_is_ambient, n_amb_cons ? n_amb_cons[k] : 0,
            d_rail ? d_rail[k] : nullptr,
            d_dqsum_e, d_dqsum_s, pb);
    }

    // ---- stage 3: e apply (gather), priced off the LIVE energy field ------
    bulk_e_apply_v2<<<grid, block, 0, st>>>(d_gas_energy, d_e, d_nb,
                                     d_dqsum_e, d_dqsum_s,
                                     d_solid, d_ts, d_is_vacuum, d_is_ambient,
                                     t_amb_raw, d_ecnt, h, w, cst, bb.es, bb.sub);
    cuda_check(cudaGetLastError(), "e_apply");

    // ---- stage 3b: TRACE apply (smoke transport v2, design §2.1) -----------
    // BETWEEN stage 3 and stage 4's `d_nb` memset: d_nb still holds the PRE-flux
    // bulk N (stage 1's accumulate; stage 2 mutates only the gas planes), which
    // is the CPU's `n_pre`. One snapshot D2D + one gather launch per live plane;
    // the kernel boundary is the CPU's loop boundary. #29 P5: the snapshot of
    // plane k is one 2D copy (N rows of h·w, the env stride gs apart).
    if (do_trace) {
        const size_t n4 = (size_t)n * sizeof(int32_t);
        for (int k = 0; k < n_trace; ++k) {
            cuda_check(cudaMemcpy2DAsync(d_spre, n4, d_trace_planes[k], gs * sizeof(int32_t),
                                         n4, (size_t)N, cudaMemcpyDeviceToDevice, st),
                       "D2D s_pre snapshot");
            trace_apply<<<grid, block, 0, st>>>(d_trace_planes[k], d_spre, d_nb,
                                         d_dqsum_e, d_dqsum_s,
                                         d_solid, d_is_vacuum, d_is_ambient,
                                         &d_tvent[k], &d_tclip[k], &d_tneg[k],
                                         h, w, gs, tst, bb.es, bb.sub);
            cuda_check(cudaGetLastError(), "trace_apply");
        }
    }

    // ---- stage 4: the mirror refresh (n_bulk POST, T = floordiv(E, n)) ----
    cuda_check(cudaMemsetAsync(d_nb, 0, n8, st), "memset nb (post)");
    for (int k = 0; k < n_cons; ++k) {
        bulk_nb_accum<<<grid, block, 0, st>>>(d_nb, d_gas_planes[k], n, gs, bb.es, bb.sub);
        cuda_check(cudaGetLastError(), "nb_accum (post)");
    }
    bulk_e_recover_v2<<<grid, block, 0, st>>>(d_temperature, d_gas_energy, d_nb,
                                       d_dqsum_e, d_dqsum_s,
                                       d_solid, d_ts, d_is_vacuum, d_is_ambient,
                                       t_min_q, t_max_phys_q, t_amb_raw,
                                       d_ecnt, h, w, cst, bb.es, bb.sub);
    cuda_check(cudaGetLastError(), "e_recover");

    // ---- stage 3c: TRACE wipe (smoke transport v2, design §2.1) ------------
    // AFTER stage 4: `d_nb` now holds the POST-flux bulk N (stage 4's memset +
    // re-accumulate) — the CPU's `n_new` — which is exactly why 3c cannot sit
    // before it and 3b cannot sit after the memset.
    if (do_trace) {
        for (int k = 0; k < n_trace; ++k) {
            trace_wipe<<<grid, block, 0, st>>>(d_trace_planes[k], d_nb,
                                        d_solid, d_is_vacuum, d_is_ambient,
                                        &d_twipe[k], n, gs, tst, bb.es, bb.sub);
            cuda_check(cudaGetLastError(), "trace_wipe");
        }
    }
}

// ===========================================================================
// SMOKE TRANSPORT v2 (#12, design §2.2-§2.4, §5 "Resident") — P2b: THE
// ONCE-PER-TICK TRACE TAIL, DEVICE-RESIDENT. The kernel-for-loop transcription
// of trace_tail (bulk_transport.cpp — THE oracle), expression for expression:
//
//   K_sink     stranded zeroing (§2.4) on solid ∨ vacuum ∨ ring, booked SINK,
//              and the Jacobi snapshot S0 of the post-sink plane (the CPU takes
//              its snapshot after step 1; sink writes only its own cell, so the
//              fusion changes nothing);
//   K_diffuse  Jacobi magnitude-truncated face diffusion (§2.2) from S0, each
//              participant gathering its four faces in the pinned E, W, S, N
//              order, outflow to a vacuum / ring cell booked VENT;
//   K_decay    the ceil-rounded decay (§2.3), booked DECAY — its OWN kernel over
//              the shared per-cell rule (trace_decay.h): a decay-rounding
//              amendment is one edit there, never a change of this kernel.
//
// Born (N, h, w)-shaped (RL-batch habits §A, N == 1 today): every plane is
// indexed env·(n_gases·n) + gi·n + cell, every mask env·n + cell; the per-env,
// per-gas folds (is_trace, dd_q, frac_q — dt is a per-env scalar) arrive as a
// device table read by env index; the books are an (N, n_gases, 3) unsigned-
// long-long block (integer atomicAdd is order-free, so each total equals the
// CPU's sequential sum). Every trace plane runs (no host liveness scan): on an
// all-zero plane every stage is an arithmetic no-op with zero counters, which
// is what the CPU's skip-after-scan produces. No plane is clamped; nothing
// here touches the bulk planes, the energy or the air (V5).
// ===========================================================================
namespace {

__device__ __forceinline__ size_t tail_off(int env, int gi, int n_gases, int n) {
    return ((size_t)env * (size_t)n_gases + (size_t)gi) * (size_t)n;
}

// The face permeability the CPU tail quantizes once per tick (its permE/permS
// build), for the face (lo, hi), lo the lower linear index: 0 when either side
// is solid, else min(quantize(min(perm_lo, perm_hi)), ONE) where positive.
// std::min(a, b) is `(b < a) ? b : a` — transcribed as such, operand order and
// all; quantize((double)pf) is the FP_HD kit helper (K_L0_faces' argument: an
// exact float->double, an exact power-of-two multiply, one rounded add).
__device__ __forceinline__ int32_t tail_face_perm(int lo, int hi,
                                                  const bool* __restrict__ solid,
                                                  const float* __restrict__ perm) {
    if (solid[lo] || solid[hi]) return 0;
    const float ff = (perm[hi] < perm[lo]) ? perm[hi] : perm[lo];
    if (!(ff > 0.0f)) return 0;
    const int32_t q = quantize((double)ff);
    return (FP_ONE < q) ? FP_ONE : q;
}

// |F| on a face from the snapshot, positive when the flow is a -> b (a the
// lower index): the scale_mag idiom — truncate the MAGNITUDE, then sign it, so
// F(a, b) == −F(b, a) and no drift. trace_tail's face_flux lambda, verbatim.
__device__ __forceinline__ int64_t tail_face_flux(int32_t s_a, int32_t s_b,
                                                  int32_t dd_q, int32_t perm_q) {
    if (perm_q == 0) return 0;
    const int64_t c = (int64_t)mul_q16(dd_q, perm_q);
    const int64_t d = (int64_t)s_a - (int64_t)s_b;
    if (d == 0 || c == 0) return 0;
    const int64_t mag = (c * (d > 0 ? d : -d)) >> FP_SHIFT;
    return d > 0 ? mag : -mag;
}

__global__ void trace_tail_zero(unsigned long long* __restrict__ cnt, int m) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < m;
         k += gridDim.x * blockDim.x)
        cnt[k] = 0ull;
}

// Step 1 (§2.4) + the S0 snapshot. One thread per (env, cell); own-cell writes.
__global__ void trace_tail_sink(int32_t* __restrict__ gas,
                                int32_t* __restrict__ s0,
                                const bool* __restrict__ solid,
                                const bool* __restrict__ is_vacuum,
                                const bool* __restrict__ is_ambient,
                                const int32_t* __restrict__ scal,
                                unsigned long long* __restrict__ cnt,
                                int n_env, int n_gases, int n) {
    const long long total = (long long)n_env * n;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (long long)gridDim.x * blockDim.x) {
        const int env = (int)(t / n);
        const int i = (int)(t - (long long)env * n);
        const int m = env * n + i;                       // the env's mask index
        const bool part = t_part_dev(m, solid, is_vacuum, is_ambient);
        for (int gi = 0; gi < n_gases; ++gi) {
            const size_t ge = (size_t)env * n_gases + gi;
            if (scal[ge * TRACE_TAIL_SCAL + 0] == 0) continue;   // not a trace plane
            const size_t off = tail_off(env, gi, n_gases, n) + i;
            int32_t s = gas[off];
            if (s != 0 && !part) {
                atomicAdd(&cnt[ge * TRACE_TAIL_CNT_SLOTS + TRACE_TAIL_SINK],
                          (unsigned long long)(int64_t)s);
                s = 0;
                gas[off] = 0;
            }
            s0[off] = s;
        }
    }
}

// Step 2 (§2.2): Jacobi from S0. A non-participant is never written (step 1
// left it 0, and the CPU's loop skips it). A (env, gas) with dd_q == 0 is
// skipped — the CPU skips the whole diffusion step; S already equals S0.
__global__ void trace_tail_diffuse(int32_t* __restrict__ gas,
                                   const int32_t* __restrict__ s0,
                                   const bool* __restrict__ solid,
                                   const bool* __restrict__ is_vacuum,
                                   const bool* __restrict__ is_ambient,
                                   const float* __restrict__ perm,
                                   const int32_t* __restrict__ scal,
                                   unsigned long long* __restrict__ cnt,
                                   int n_env, int n_gases, int h, int w) {
    const int n = h * w;
    const long long total = (long long)n_env * n;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (long long)gridDim.x * blockDim.x) {
        const int env = (int)(t / n);
        const int i = (int)(t - (long long)env * n);
        const int mb = env * n;                          // the env's mask base
        if (!t_part_dev(mb + i, solid, is_vacuum, is_ambient)) continue;
        const int y = i / w;
        const int x = i - y * w;
        const bool* sol = solid + mb;
        const float* pm = perm + mb;
        // The four face permeabilities (plane-independent), and whether the
        // neighbour across each face is a non-participant (its outflow is VENT).
        const int32_t pE = (x < w - 1) ? tail_face_perm(i, i + 1, sol, pm) : 0;
        const int32_t pW = (x > 0)     ? tail_face_perm(i - 1, i, sol, pm) : 0;
        const int32_t pS = (y < h - 1) ? tail_face_perm(i, i + w, sol, pm) : 0;
        const int32_t pN = (y > 0)     ? tail_face_perm(i - w, i, sol, pm) : 0;
        const bool oE = (x < w - 1) && !t_part_dev(mb + i + 1, solid, is_vacuum, is_ambient);
        const bool oW = (x > 0)     && !t_part_dev(mb + i - 1, solid, is_vacuum, is_ambient);
        const bool oS = (y < h - 1) && !t_part_dev(mb + i + w, solid, is_vacuum, is_ambient);
        const bool oN = (y > 0)     && !t_part_dev(mb + i - w, solid, is_vacuum, is_ambient);
        for (int gi = 0; gi < n_gases; ++gi) {
            const size_t ge = (size_t)env * n_gases + gi;
            if (scal[ge * TRACE_TAIL_SCAL + 0] == 0) continue;
            const int32_t dd_q = scal[ge * TRACE_TAIL_SCAL + 1];
            if (dd_q <= 0) continue;
            const int32_t* S0 = s0 + tail_off(env, gi, n_gases, n);
            int64_t ds = 0, vented = 0;
            // pinned face order E, W, S, N (the stage-3b order), CPU verbatim.
            if (x < w - 1) {
                const int64_t f = tail_face_flux(S0[i], S0[i + 1], dd_q, pE);
                ds -= f;
                if (f > 0 && oE) vented += f;
            }
            if (x > 0) {
                const int64_t f = tail_face_flux(S0[i - 1], S0[i], dd_q, pW);
                ds += f;
                if (f < 0 && oW) vented -= f;
            }
            if (y < h - 1) {
                const int64_t f = tail_face_flux(S0[i], S0[i + w], dd_q, pS);
                ds -= f;
                if (f > 0 && oS) vented += f;
            }
            if (y > 0) {
                const int64_t f = tail_face_flux(S0[i - w], S0[i], dd_q, pN);
                ds += f;
                if (f < 0 && oN) vented -= f;
            }
            gas[tail_off(env, gi, n_gases, n) + i] = (int32_t)((int64_t)S0[i] + ds);
            if (vented != 0)
                atomicAdd(&cnt[ge * TRACE_TAIL_CNT_SLOTS + TRACE_TAIL_VENT],
                          (unsigned long long)vented);
        }
    }
}

// Step 3 (§2.3): THE DECAY KERNEL, isolated. Per (env, cell, trace gas) with
// frac_q > 0 and v > 0: lost = trace_decay::lost(v, frac_q) — the ONE shared
// rule (trace_decay.h), also called by the CPU's trace_decay_plane.
__global__ void trace_tail_decay(int32_t* __restrict__ gas,
                                 const int32_t* __restrict__ scal,
                                 unsigned long long* __restrict__ cnt,
                                 int n_env, int n_gases, int n) {
    const long long total = (long long)n_env * n;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (long long)gridDim.x * blockDim.x) {
        const int env = (int)(t / n);
        const int i = (int)(t - (long long)env * n);
        for (int gi = 0; gi < n_gases; ++gi) {
            const size_t ge = (size_t)env * n_gases + gi;
            if (scal[ge * TRACE_TAIL_SCAL + 0] == 0) continue;
            const int32_t frac_q = scal[ge * TRACE_TAIL_SCAL + 2];
            if (frac_q <= 0) continue;
            const size_t off = tail_off(env, gi, n_gases, n) + i;
            const int64_t v = gas[off];
            if (v <= 0) continue;
            const int64_t lost = trace_decay::lost(v, frac_q);
            gas[off] = (int32_t)(v - lost);
            if (lost != 0)
                atomicAdd(&cnt[ge * TRACE_TAIL_CNT_SLOTS + TRACE_TAIL_DECAY],
                          (unsigned long long)lost);
        }
    }
}

}  // namespace

int trace_tail_launch_resident(
        int n_env, int h, int w, int n_gases,
        int32_t* d_gas,
        const bool* d_solid, const bool* d_is_vacuum, const bool* d_is_ambient,
        const float* d_perm,
        const int32_t* d_scal,
        int32_t* d_s0,
        unsigned long long* d_cnt) {
    const int n = h * w;
    if (n_env <= 0 || n <= 0 || n_gases <= 0) return 0;
    const int block = 256;
    const long long total = (long long)n_env * n;
    const int grid = (int)((total + block - 1) / block);
    const int m = n_env * n_gases * TRACE_TAIL_CNT_SLOTS;
    trace_tail_zero<<<(m + block - 1) / block, block, 0, tstream()>>>(d_cnt, m);
    cuda_check(cudaGetLastError(), "trace_tail_zero");
    trace_tail_sink<<<grid, block, 0, tstream()>>>(d_gas, d_s0, d_solid, d_is_vacuum,
                                     d_is_ambient, d_scal, d_cnt,
                                     n_env, n_gases, n);
    cuda_check(cudaGetLastError(), "trace_tail_sink");
    trace_tail_diffuse<<<grid, block, 0, tstream()>>>(d_gas, d_s0, d_solid, d_is_vacuum,
                                        d_is_ambient, d_perm, d_scal, d_cnt,
                                        n_env, n_gases, h, w);
    cuda_check(cudaGetLastError(), "trace_tail_diffuse");
    trace_tail_decay<<<grid, block, 0, tstream()>>>(d_gas, d_scal, d_cnt, n_env, n_gases, n);
    cuda_check(cudaGetLastError(), "trace_tail_decay");
    return 4;
}

// (#29 P5: the N == 1 wrapper trace_tail_resident is DELETED — the batched
// resident tick's driver, cuda_resident_tick.cu resident_trace_tail, folds
// the same host folds into the same launch core for every env, its books in
// the tick's one arena.)

namespace {
bool g_bulk_flux_backend_cuda = false;
}
bool bulk_flux_backend_is_cuda() { return g_bulk_flux_backend_cuda; }
void set_bulk_flux_backend_cuda(bool on) { g_bulk_flux_backend_cuda = on; }

}  // namespace breach_cuda
