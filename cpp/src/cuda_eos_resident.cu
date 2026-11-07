// ============================================================================
// CUDA-S8a Path A Ã¢â‚¬â€ the fully device-resident EOS tick Ã¢â‚¬â€ see cuda_eos_resident.h
// Design (BUILD AGAINST): docs/cuda_s8a_path_a_impl_2026-07-21.md (v2).
//
// STRUCTURE (mirrors eos_step_cuda stage-for-stage; Ã‚Â§-refs are the design's):
//   host pre-stage        Ã¢â‚¬â€ the SHARED eos_host_prestage (Ã‚Â§3.2.2) on the
//                           authoritative mirror (all reductions; Ã‚Â§0).
//   device substep loop   Ã¢â‚¬â€ the SAME sl_advect3/bulk_flux launchers the
//                           per-call path chains (Ã‚Â§3.2.6c), on the resident
//                           fields + persistent scratch.
//   device mid-stage      Ã¢â‚¬â€ K_div_u / K_ntot / K_pstar (Ã‚Â§3.2.6d-f): per-cell
//                           verbatim transcriptions of the host mid-stage.
//   device MG build       Ã¢â‚¬â€ the mg_build_levels port (Ã‚Â§4): per-cell and
//                           per-coarse-cell SINGLE-WRITER gathers + per-cell
//                           integer divides, level-sequential launches. THE
//                           EVERY-CELL-WRITE RULE: the CPU assign(0)s every
//                           array then writes subsets; the persistent
//                           hierarchy carries last tick's bytes, so every
//                           kernel writes EVERY cell of EVERY output array
//                           (else-branches write the assign's 0).
//   device solve          Ã¢â‚¬â€ eos_mg_vcycle_resident (the SHARED schedule).
//   device kick/compress  Ã¢â‚¬â€ kick_compression_launch_resident (SHARED core).
//   device store          Ã¢â‚¬â€ K_store_atm (Ã‚Â§3.2.6j).
//
// DETERMINISM (Ã‚Â§0/Ã‚Â§4): no parallel reductions anywhere (the two atomics Ã¢â‚¬â€
// bulk rail, kick counters Ã¢â‚¬â€ are order-free integer sums, digest-proven
// per-call). Every kernel is a single-writer gather; every cross-kernel
// dependency sits at a same-stream launch boundary == the CPU's pass/level
// boundary. Integer divides truncate toward zero identically on MSVC and
// nvcc. quantize((double)pf) on device is ONE identically-rounded IEEE
// double add after two exact steps (exact float->double, exact power-of-two
// multiply); FMA contraction is harmless because the product is exact.
// K_ntot accumulates int32 as uint32 (defined wrap == the hosts' observed
// int32 wrap for all inputs Ã¢â‚¬â€ design Ã‚Â§3.2.6e; do NOT widen to int64).
//
// TELEMETRY: boundary_flux_ ASSIGNED and the five rail counters ACCUMULATED
// exactly as eos_step_cuda does (smoke transport v2, P2b: boundary_flux_'s
// TRACE slots + trace_wipe_sum_ assigned from the stage-3b/3c book block); d_rail/d_cnt are PERSISTENT and therefore
// cudaMemset EVERY tick (Ã‚Â§3.2.5 Ã¢â‚¬â€ the per-call wrappers' memsets moved
// here). Digests / dbg probes / host levels_ are stale on this path (Ã‚Â§3.3).
// ============================================================================
#include "cuda_eos_resident.h"
#include "cuda_eos_step.h"         // eos_host_prestage (the shared pre-stage)
#include "cuda_resident.h"         // kick_scalar_folds + kick launch core
#include "cuda_mg_solve.h"         // MGLevelDevPtrs + eos_mg_vcycle_resident
#include "cuda_sl_advection.h"     // sl_cmask_build_device / sl_advect3_device
#include "cuda_bulk_transport.h"   // bulk_flux_plane_device
#include "eos_solver.h"            // EOSSolver (config + mg_scalar_folds + telemetry)
#include "fixed_point.h"           // q16, quantize, mul_q16 (FP_HD)
#include "cuda_fixedpoint_device.cuh"  // mul128_shr_signed, reciprocal_q16_dev
#include "cuda_batch.cuh"               // #29 P5: the (N, h, w) launch shape
#include "cuda_resident_tick.h"         // EnvSub

#include <cuda_runtime.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace fixedpoint;

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in eos_step_resident/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

long long g_eos_resident_calls = 0;
long long g_eos_nsub_reads = 0;   // #29 P5: the counted per-tick max(n_sub) reads

constexpr int MG_MAX_LEVELS_RES = 9;   // == eos_solver.cpp's level-count cap
// #88 P2: the resident trace book block is 4 (n_gases)-wide blocks: vent, wipe,
// clip, neg hits (the last two APPENDED; see EOSResidentScratch::tcnt).
constexpr int RESIDENT_TRACE_CNT_BLOCKS = 4;

// ---- solid-mirror neighbor read (eos_solver.cpp mirror_idx, verbatim) -------
__device__ __forceinline__ int mirror_idx_dev_r(
        int self_i, int ny, int nx, int h, int w,
        const bool* __restrict__ solid) {
    if (ny < 0 || ny >= h || nx < 0 || nx >= w) return self_i;
    const int ni = ny * w + nx;
    if (solid[ni]) return self_i;
    return ni;
}

// ---- face-form divergence face read (eos_solver.cpp face2_u, verbatim) -----
// arc #54 P-G1d / D4: 2Ã‚Â·ÃƒÂ»_f, with ÃƒÂ» = 0 at a SOLID (or out-of-bounds) face and
// ÃƒÂ» = (u_i + u_j)/2 on an open one; the Ã‚Â½ rides in `inv_2dx_q`. Derivation and
// the interior bit-identity argument: eos_solver.cpp's copy.
__device__ __forceinline__ int64_t face2_u_dev_r(
        const int32_t* __restrict__ u, int self_i, int ny, int nx,
        int h, int w, const bool* __restrict__ solid) {
    if (ny < 0 || ny >= h || nx < 0 || nx >= w) return 0;
    const int ni = ny * w + nx;
    if (solid[ni]) return 0;
    return (int64_t)u[self_i] + (int64_t)u[ni];
}

// ============================================================================
// MID-STAGE kernels Ã¢â‚¬â€ verbatim device transcriptions of eos_step_cuda's host
// mid-stage (cuda_eos_step.cu "HOST MID-STAGE" block), one writer per cell.
// ============================================================================

// div(u*) Ã¢â‚¬â€ the central-difference divergence on the post-substep wind.
// BC (audit (b)): the ring is a Dirichlet boundary Ã¢â‚¬â€ div(u*) = 0.
__global__ void K_div_u(int32_t* __restrict__ div_u,
                        const int32_t* __restrict__ wind_x,
                        const int32_t* __restrict__ wind_y,
                        const bool* __restrict__ solid,
                        const bool* __restrict__ is_vacuum,
                        const bool* __restrict__ is_ambient,
                        int32_t inv_2dx_q, int h, int w) {
    const int n = h * w;
    div_u = bptr(div_u, n); wind_x = bptr(wind_x, n); wind_y = bptr(wind_y, n);
    solid = bptr(solid, n); is_vacuum = bptr(is_vacuum, n); is_ambient = bptr(is_ambient, n);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (solid[i] || is_vacuum[i] || (is_ambient && is_ambient[i])) {
            div_u[i] = 0; continue;
        }
        const int y = i / w;
        const int x = i - y * w;
        // arc #54 P-G1d (D4): FACE FORM, ÃƒÂ» = 0 at solid faces.
        const int64_t fw = face2_u_dev_r(wind_x, i, y, x - 1, h, w, solid);
        const int64_t fe = face2_u_dev_r(wind_x, i, y, x + 1, h, w, solid);
        const int64_t fn = face2_u_dev_r(wind_y, i, y - 1, x, h, w, solid);
        const int64_t fs = face2_u_dev_r(wind_y, i, y + 1, x, h, w, solid);
        const q16 dux = mul_q16((q16)(fe - fw), inv_2dx_q);
        const q16 duy = mul_q16((q16)(fs - fn), inv_2dx_q);
        div_u[i] = dux + duy;
    }
}

// #31: the blast's divergence source comes off div(u*), EOSSolver::step's own
// lines. Launched only on a tick with a source; K_div_u above stays as it was.
__global__ void K_div_source(int32_t* __restrict__ div_u,
                             const int32_t* __restrict__ div_source,
                             const bool* __restrict__ solid,
                             const bool* __restrict__ is_vacuum,
                             const bool* __restrict__ is_ambient, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (solid[i] || is_vacuum[i] || (is_ambient && is_ambient[i])) continue;
        div_u[i] = fixedpoint::sat_add_q16_i64(div_u[i], -(int64_t)div_source[i]);
    }
}

// Step 2's Dalton sum (post-substep N) Ã¢â‚¬â€ per cell, gi ASCENDING (the CPU's
// per-cell add sequence). uint32 accumulation = defined wrap, byte-identical
// to the hosts' int32 wrap for all inputs (design Ã‚Â§3.2.6e). P-T0 (design
// Ã‚Â§2.6): n_total Ã¢â€°Â¡ n_bulk Ã¢â‚¬â€ non-conservative (trace) planes are skipped
// outright, not weighted by a retired tms_q.
__global__ void K_ntot(int32_t* __restrict__ ntot,
                       const int32_t* __restrict__ gas_base,
                       const bool* __restrict__ cons_flag,
                       int n_gases, int n) {
    ntot = bptr(ntot, (size_t)n); gas_base = bptr(gas_base, (size_t)n_gases * n);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        int32_t acc = 0;
        for (int gi = 0; gi < n_gases; ++gi) {
            if (!cons_flag[gi]) continue;
            const int32_t v = gas_base[(size_t)gi * n + i];
            acc = (int32_t)((uint32_t)acc + (uint32_t)v);
        }
        ntot[i] = acc;
    }
}

// ---- #29 P4a: the pre-stage's two water-dependent pieces, on the DEVICE ----
// The resident tick now runs the water tail on the device BEFORE the EOS, and
// the tail writes `gas` (W5's steam, W3's evacuation) and `dyn_permeability`
// (W3's seal). The two pre-stage products that read them move here, onto the
// device planes, transcribed from eos_host_prestage line for line:
//
// K_max_du Ã¢â‚¬â€ step()'s KÃ‚Â·|Ã¢Ë†â€¡P|Ã‚Â·dt/NÃŒâ€š grad-scan: per open cell the solid-mirror
// central gradient of p_prev (== atmosphere, the step-0 copy), its larger
// component's magnitude, times Kdt over the floored Dalton N (K_ntot's plane
// from the tick-entry gas). The host keeps the MAX; every du is >= 0 (Kdt >=
// 0, |Ã¢Ë†â€¡P| >= 0, the reciprocal > 0), so an unsigned atomicMax from 0 is the
// same order-free integer max. mul128_shr_signed / reciprocal_q16_dev are the
// proven twins of the host's mul128_shr_host / reciprocal_q16.
__global__ void K_max_du(unsigned long long* __restrict__ out,
                         const int32_t* __restrict__ p_prev,
                         const int32_t* __restrict__ ntot,
                         const bool* __restrict__ solid,
                         int32_t inv_2dx_q, int64_t kdt_raw, int32_t n_floor_q,
                         int h, int w, size_t ostride) {
    const int n = h * w;
    out = bptr(out, ostride); p_prev = bptr(p_prev, n); ntot = bptr(ntot, n);
    solid = bptr(solid, n);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (solid[i]) continue;
        const int y = i / w, x = i % w;
        const int il = mirror_idx_dev_r(i, y, x - 1, h, w, solid);
        const int ir = mirror_idx_dev_r(i, y, x + 1, h, w, solid);
        const int iu = mirror_idx_dev_r(i, y - 1, x, h, w, solid);
        const int id = mirror_idx_dev_r(i, y + 1, x, h, w, solid);
        const int64_t gx = mul128_shr_signed((int64_t)(p_prev[ir] - p_prev[il]),
                                             (int64_t)inv_2dx_q, 16);
        const int64_t gy = mul128_shr_signed((int64_t)(p_prev[id] - p_prev[iu]),
                                             (int64_t)inv_2dx_q, 16);
        const int64_t agx = gx < 0 ? -gx : gx;
        const int64_t agy = gy < 0 ? -gy : gy;
        const int64_t gmag = agx > agy ? agx : agy;
        if (gmag == 0) continue;
        q16 nhat = ntot[i];
        if (nhat < n_floor_q) nhat = n_floor_q;
        const q16 inv_n = reciprocal_q16_dev(nhat);
        const int64_t t1 = mul128_shr_signed(kdt_raw, gmag, 16);
        const int64_t du = mul128_shr_signed(t1, (int64_t)inv_n, 16);
        if (du > 0) atomicMax(out, (unsigned long long)du);
    }
}

// K_coeffs Ã¢â‚¬â€ step()'s donor-cell face-coefficient cache (eos_host_prestage's
// coeffE / coeffS), every cell written (0 where the host left the assign's
// 0): an open east / south face between two non-solid cells carries
// mul_q16(quantize((double)min(perm_i, perm_j)), dt_s_q) when that min is
// positive. std::min(a, b) is (b < a) ? b : a, spelled as such.
__global__ void K_coeffs(int32_t* __restrict__ coeffE, int32_t* __restrict__ coeffS,
                         const bool* __restrict__ solid,
                         const float* __restrict__ perm,
                         int32_t dts_q, int h, int w,
                         const EnvSub* __restrict__ es) {
    const int n = h * w;
    if (es) dts_q = es[benv()].dt_s_q;   // #29 P5: the env's own substep dt
    coeffE = bptr(coeffE, n); coeffS = bptr(coeffS, n);
    solid = bptr(solid, n); perm = bptr(perm, n);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        int32_t cE = 0, cS = 0;
        if (!solid[i]) {
            const int y = i / w, x = i % w;
            if (x < w - 1 && !solid[i + 1]) {
                const float a = perm[i], b = perm[i + 1];
                const float ff = (b < a) ? b : a;
                if (ff > 0.0f) cE = mul_q16(quantize((double)ff), dts_q);
            }
            if (y < h - 1 && !solid[i + w]) {
                const float a = perm[i], b = perm[i + w];
                const float ff = (b < a) ? b : a;
                if (ff > 0.0f) cS = mul_q16(quantize((double)ff), dts_q);
            }
        }
        coeffE[i] = cE;
        coeffS[i] = cS;
    }
}

// Step 2's p* = CÃ‚Â·N_totalÃ‚Â·T_abs (EOS floor at 0); the debug_pstar_from_prev
// MEASUREMENT-ONLY branch reads p_prev (the post-copy bytes, like the CPU).
__global__ void K_pstar(int32_t* __restrict__ pstar,
                        const int32_t* __restrict__ ntot,
                        const int32_t* __restrict__ temperature,
                        const int32_t* __restrict__ p_prev,
                        const bool* __restrict__ solid,
                        const bool* __restrict__ is_vacuum,
                        int32_t t_amb_q, int32_t s_eos_q, int32_t c_q,
                        bool debug_from_prev, int n) {
    pstar = bptr(pstar, (size_t)n); ntot = bptr(ntot, (size_t)n);
    temperature = bptr(temperature, (size_t)n); p_prev = bptr(p_prev, (size_t)n);
    solid = bptr(solid, (size_t)n); is_vacuum = bptr(is_vacuum, (size_t)n);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (solid[i] || is_vacuum[i]) { pstar[i] = 0; continue; }
        int32_t ps;
        if (debug_from_prev) {
            ps = p_prev[i];   // MEASUREMENT-ONLY diagnostic (parity)
        } else {
            // s_eos_q joins t_amb_q, CPU twin verbatim (eos_solver.cpp:290,
            // cuda_eos_step.cu:169/522). Frozen identity (65536) => exact
            // truncation, byte-identical this arc (P-K3).
            const int64_t t_abs_wide = (((int64_t)s_eos_q * (int64_t)temperature[i]) >> 16) + (int64_t)t_amb_q;
            const q16 t_abs = (q16)t_abs_wide;
            const q16 cn = mul_q16(c_q, ntot[i]);
            ps = mul_q16(cn, t_abs);
        }
        if (ps < 0) ps = 0;   // EOS floor
        pstar[i] = ps;
    }
}

// Step 5: P := P_new, materialized once. BC: the solve ran in P' = P Ã¢Ë†â€™ P_amb;
// add P_amb back MASKED to !solid (solids stay 0 absolute Ã¢â‚¬â€ p_new is 0 there
// post-zero-excl either way; ring cells and every regular cell get the add).
__global__ void K_store_atm(int32_t* __restrict__ atmosphere,
                            const int32_t* __restrict__ p_new,
                            const bool* __restrict__ solid,
                            int64_t p_amb, bool ambient_mode, int n) {
    atmosphere = bptr(atmosphere, (size_t)n); p_new = bptr(p_new, (size_t)n);
    solid = bptr(solid, (size_t)n);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (ambient_mode) {
            atmosphere[i] = solid[i]
                ? p_new[i]
                : (int32_t)((int64_t)p_new[i] + p_amb);
        } else {
            atmosphere[i] = p_new[i];
        }
    }
}

// ============================================================================
// MG BUILD kernels Ã¢â‚¬â€ the on-device mg_build_levels port (design Ã‚Â§4). Every
// kernel body is a VERBATIM transcription of eos_solver.cpp:727-1012 with
// mul128_shr -> mul128_shr_signed and reciprocal_q16 -> reciprocal_q16_dev.
// EVERY output array cell is written unconditionally (the every-cell rule).
// ============================================================================

constexpr int64_t MG_M_CAP   = ((int64_t)1) << 38;   // level-0 mass cap
constexpr int64_t MG_M_CAP_L = ((int64_t)1) << 44;   // coarse cap (sums grow Ãƒâ€”4/level)

// Level-0 excl: the branch-priority chain (solidÃ¢â€ â€™2 beats vacuumÃ¢â€ â€™1 beats
// ambientÃ¢â€ â€™1; else 0). Every cell written.
__global__ void K_L0_excl(uint8_t* __restrict__ excl,
                          const bool* __restrict__ solid,
                          const bool* __restrict__ is_vacuum,
                          const bool* __restrict__ is_ambient,
                          bool ambient_mode, int n) {
    excl = bptr(excl, (size_t)n); solid = bptr(solid, (size_t)n);
    is_vacuum = bptr(is_vacuum, (size_t)n); is_ambient = bptr(is_ambient, (size_t)n);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        uint8_t e = 0;
        if (solid[i]) e = 2;
        else if (is_vacuum[i]) e = 1;
        else if (ambient_mode && is_ambient[i]) e = 1;   // BC ring Ã¢â€ â€™ Dirichlet
        excl[i] = e;
    }
}

// Level-0 m / b / P-warm-start (eos_solver.cpp:805-834, verbatim Ã¢â‚¬â€ incl. the
// EXACT clamp sequence: aK floored to 1 BEFORE the divide, then m floored to
// 1 and capped at M_CAP; the ambient rhs -= p_amb BEFORE the m-multiply; the
// widen-then-narrow shifted warm start). excl!=0 cells write m=b=P=0 (the
// CPU's assign).
__global__ void K_L0_mbP(int64_t* __restrict__ m,
                         int64_t* __restrict__ b,
                         int32_t* __restrict__ P,
                         const uint8_t* __restrict__ excl,
                         const int32_t* __restrict__ pstar,
                         const int32_t* __restrict__ div_u,
                         const int32_t* __restrict__ p_prev,
                         int32_t gamma_q, int32_t dt_q, int64_t Kdt2dx2_raw,
                         int64_t p_amb, bool ambient_mode, int n) {
    const size_t sn = (size_t)n;
    m = bptr(m, sn); b = bptr(b, sn); P = bptr(P, sn); excl = bptr(excl, sn);
    pstar = bptr(pstar, sn); div_u = bptr(div_u, sn); p_prev = bptr(p_prev, sn);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (excl[i] != 0) { m[i] = 0; b[i] = 0; P[i] = 0; continue; }
        const int64_t gp_raw = mul128_shr_signed((int64_t)gamma_q, (int64_t)pstar[i], 16);
        int64_t aK = mul128_shr_signed(gp_raw, Kdt2dx2_raw, 16);
        if (aK < 1) aK = 1;
        int64_t mm = (((int64_t)1) << 32) / aK;   // 1/aK at Q16.16 raw
        if (mm < 1) mm = 1;
        if (mm > MG_M_CAP) mm = MG_M_CAP;
        m[i] = mm;
        // rhs_i = p* Ã¢Ë†â€™ (ÃŽÂ³Ã‚Â·p*)Ã‚Â·dtÃ‚Â·div(ÃƒÂ»*) (int64 raw); b = mÃ‚Â·rhs @F8.
        const int64_t gp_dt = mul128_shr_signed(gp_raw, (int64_t)dt_q, 16);
        int64_t rhs_raw = (int64_t)pstar[i]
                              - mul128_shr_signed(gp_dt, (int64_t)div_u[i], 16);
        // BC (spec Ã‚Â§1 "the shift trick"): subtract BEFORE the m multiply.
        if (ambient_mode) rhs_raw -= p_amb;
        b[i] = mul128_shr_signed(mm, rhs_raw, 8);
        // Warm start from the previous tick's solved P; BC: re-shift fresh
        // into PÃ¢â‚¬Â²-space (widen-then-narrow Ã¢â‚¬â€ wrap-deterministic).
        P[i] = ambient_mode
            ? (int32_t)((int64_t)p_prev[i] - p_amb)
            : p_prev[i];
    }
}

// Level-0 face conductances g = perm/NÃŒâ€š (eos_solver.cpp:839-866, verbatim).
// Each cell OWNS its E and S face Ã¢â‚¬â€ single writer; every cell written (0 at
// guard-fail / the x=wÃ¢Ë†â€™1 / y=hÃ¢Ë†â€™1 boundary). THE GUARD IS excl != 2 ON BOTH
// ENDPOINTS (NOT ==0): regularÃ¢â€ â€Dirichlet faces carry conductance Ã¢â‚¬â€ the
// Galerkin anchor depends on them. min(perm) uses std::min select semantics
// ((b<a)?b:a); quantize((double)pf) is the FP_HD helper (exactness argument
// in the file header); N_FLOOR applies HERE ONLY (never to m).
__global__ void K_L0_faces(int64_t* __restrict__ gE,
                           int64_t* __restrict__ gS,
                           const uint8_t* __restrict__ excl,
                           const int32_t* __restrict__ ntot,
                           const float* __restrict__ perm,
                           int32_t n_floor_q, int h, int w) {
    const int n = h * w;
    gE = bptr(gE, n); gS = bptr(gS, n); excl = bptr(excl, n);
    ntot = bptr(ntot, n); perm = bptr(perm, n);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int y = i / w;
        const int x = i - y * w;
        int64_t ge = 0, gs = 0;
        if (x < w - 1) {
            const int j = i + 1;
            if (excl[i] != 2 && excl[j] != 2) {
                const float pf = (perm[j] < perm[i]) ? perm[j] : perm[i];
                if (pf > 0.0f) {
                    q16 nhat = (q16)(((int64_t)ntot[i] + ntot[j]) >> 1);
                    if (nhat < n_floor_q) nhat = n_floor_q;
                    ge = (int64_t)mul_q16(quantize((double)pf),
                                          reciprocal_q16_dev(nhat));
                }
            }
        }
        if (y < h - 1) {
            const int j = i + w;
            if (excl[i] != 2 && excl[j] != 2) {
                const float pf = (perm[j] < perm[i]) ? perm[j] : perm[i];
                if (pf > 0.0f) {
                    q16 nhat = (q16)(((int64_t)ntot[i] + ntot[j]) >> 1);
                    if (nhat < n_floor_q) nhat = n_floor_q;
                    gs = (int64_t)mul_q16(quantize((double)pf),
                                          reciprocal_q16_dev(nhat));
                }
            }
        }
        gE[i] = ge;
        gS[i] = gs;
    }
}

// BC (spec Ã‚Â§3 rung 1, B3b): the ÃÆ’-SPONGE fold (eos_solver.cpp:877-886,
// verbatim Ã¢â‚¬â€ incl. the REACHABLE M_CAP re-clamp). In-place add on the
// already-fully-written m; launched only in ambient mode with a ÃÆ’ grid.
// Runs AFTER K_L0_mbP (b used the un-ÃÆ’'d m) and BEFORE the coarse build +
// recip (both fold ÃÆ’) Ã¢â‚¬â€ the CPU's exact placement.
__global__ void K_L0_sigma(int64_t* __restrict__ m,
                           const uint8_t* __restrict__ excl,
                           const int32_t* __restrict__ sigma, int n) {
    m = bptr(m, (size_t)n); excl = bptr(excl, (size_t)n); sigma = bptr(sigma, (size_t)n);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (excl[i] != 0) continue;
        const int32_t s = sigma[i];
        if (s <= 0) continue;
        int64_t ms = m[i] + (int64_t)s;
        if (ms > MG_M_CAP) ms = MG_M_CAP;
        m[i] = ms;
    }
}

// Coarse excl + mass (eos_solver.cpp:911-956, verbatim): the coarse
// Dirichlet rule (all-vacuumÃ¢â€ â€™1, all-non-regularÃ¢â€ â€™2, any-regularÃ¢â€ â€™0), the mass
// SUM over regular children, and the GALERKIN DIRICHLET ANCHOR (every fine
// face from a regular child to a Dirichlet fine cell lands on the coarse
// diagonal Ã¢â‚¬â€ the exact PÃ¡Âµâ‚¬AP term). Child loops in the CPU's dy,dxx order
// with the fy/fx clipping; the anchor's out-of-block gE[fiÃ¢Ë†â€™1]/gS[fiÃ¢Ë†â€™fw]
// reads are read-only fine-level data. Every coarse cell writes excl, m,
// AND b=0 / P=0 (the CPU's assign Ã¢â‚¬â€ belt-and-braces: the vcycle's restrict
// rewrites both before any read).
__global__ void K_C_excl_m(uint8_t* __restrict__ c_excl,
                           int64_t* __restrict__ c_m,
                           int64_t* __restrict__ c_b,
                           int32_t* __restrict__ c_P,
                           const uint8_t* __restrict__ f_excl,
                           const int64_t* __restrict__ f_m,
                           const int64_t* __restrict__ f_gE,
                           const int64_t* __restrict__ f_gS,
                           int fh, int fw, int ch, int cw) {
    const int cn = ch * cw;
    const size_t fn = (size_t)fh * fw;
    c_excl = bptr(c_excl, cn); c_m = bptr(c_m, cn); c_b = bptr(c_b, cn); c_P = bptr(c_P, cn);
    f_excl = bptr(f_excl, fn); f_m = bptr(f_m, fn); f_gE = bptr(f_gE, fn); f_gS = bptr(f_gS, fn);
    for (int A = blockIdx.x * blockDim.x + threadIdx.x; A < cn;
         A += gridDim.x * blockDim.x) {
        const int Y = A / cw;
        const int X = A - Y * cw;
        int n_child = 0, n_vac = 0, n_sol = 0;
        int64_t m_sum = 0;
        for (int dy = 0; dy < 2; ++dy) {
            for (int dxx = 0; dxx < 2; ++dxx) {
                const int fy = 2 * Y + dy, fx = 2 * X + dxx;
                if (fy >= fh || fx >= fw) continue;
                const int fi = fy * fw + fx;
                ++n_child;
                if (f_excl[fi] == 1) ++n_vac;
                else if (f_excl[fi] == 2) ++n_sol;
                else m_sum += f_m[fi];
            }
        }
        uint8_t e;
        int64_t mA = 0;
        if (n_vac == n_child) e = 1;
        else if (n_vac + n_sol == n_child) e = 2;
        else {
            e = 0;
            int64_t anchor = 0;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dxx = 0; dxx < 2; ++dxx) {
                    const int fy = 2 * Y + dy, fx = 2 * X + dxx;
                    if (fy >= fh || fx >= fw) continue;
                    const int fi = fy * fw + fx;
                    if (f_excl[fi] != 0) continue;
                    if (fx + 1 < fw && f_excl[fi + 1] == 1)   anchor += f_gE[fi];
                    if (fx > 0 && f_excl[fi - 1] == 1)        anchor += f_gE[fi - 1];
                    if (fy + 1 < fh && f_excl[fi + fw] == 1)  anchor += f_gS[fi];
                    if (fy > 0 && f_excl[fi - fw] == 1)       anchor += f_gS[fi - fw];
                }
            }
            mA = m_sum + anchor;
            if (mA > MG_M_CAP_L) mA = MG_M_CAP_L;   // std::min, selected
        }
        c_excl[A] = e;
        c_m[A] = mA;
        c_b[A] = 0;
        c_P[A] = 0;
    }
}

// Coarse faces (eos_solver.cpp:957-988, verbatim): coarse face = SUM of the
// (Ã¢â€°Â¤2) crossing fine faces whose BOTH endpoints are regular (== 0 Ã¢â‚¬â€ a
// regularÃ¢â€ â€™Dirichlet crossing face is the anchor term, NOT inter-cell
// coupling; note the guard DIFFERS from level 0's != 2). Every cell written.
__global__ void K_C_faces(int64_t* __restrict__ c_gE,
                          int64_t* __restrict__ c_gS,
                          const uint8_t* __restrict__ f_excl,
                          const int64_t* __restrict__ f_gE,
                          const int64_t* __restrict__ f_gS,
                          int fh, int fw, int ch, int cw) {
    const int cn = ch * cw;
    const size_t fn = (size_t)fh * fw;
    c_gE = bptr(c_gE, cn); c_gS = bptr(c_gS, cn);
    f_excl = bptr(f_excl, fn); f_gE = bptr(f_gE, fn); f_gS = bptr(f_gS, fn);
    for (int A = blockIdx.x * blockDim.x + threadIdx.x; A < cn;
         A += gridDim.x * blockDim.x) {
        const int Y = A / cw;
        const int X = A - Y * cw;
        int64_t ge = 0, gs = 0;
        if (X < cw - 1) {
            const int fx = 2 * X + 1;   // fine face fx -> fx+1
            for (int dy = 0; dy < 2; ++dy) {
                const int fy = 2 * Y + dy;
                if (fy >= fh || fx + 1 >= fw) continue;
                const int fi = fy * fw + fx;
                if (f_excl[fi] == 0 && f_excl[fi + 1] == 0)
                    ge += f_gE[fi];
            }
        }
        if (Y < ch - 1) {
            const int fy = 2 * Y + 1;
            for (int dxx = 0; dxx < 2; ++dxx) {
                const int fx = 2 * X + dxx;
                if (fx >= fw || fy + 1 >= fh) continue;
                const int fi = fy * fw + fx;
                if (f_excl[fi] == 0 && f_excl[fi + fw] == 0)
                    gs += f_gS[fi];
            }
        }
        c_gE[A] = ge;   // SUM (variational), not average
        c_gS[A] = gs;
    }
}

// Per-level diagonal reciprocals (eos_solver.cpp:992-1009, verbatim): the
// diagonal fold in the CPU's E,W,S,N order with the excl != 2 neighbor
// guards, floor 1, then ONE wide integer divide. Every cell written.
__global__ void K_recip(int64_t* __restrict__ recip,
                        const uint8_t* __restrict__ excl,
                        const int64_t* __restrict__ m,
                        const int64_t* __restrict__ gE,
                        const int64_t* __restrict__ gS,
                        int lh, int lw) {
    const int n = lh * lw;
    recip = bptr(recip, n); excl = bptr(excl, n); m = bptr(m, n);
    gE = bptr(gE, n); gS = bptr(gS, n);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (excl[i] != 0) { recip[i] = 0; continue; }
        const int y = i / lw;
        const int x = i - y * lw;
        int64_t d_raw = m[i];
        if (x < lw - 1 && excl[i + 1] != 2)  d_raw += gE[i];
        if (x > 0 && excl[i - 1] != 2)       d_raw += gE[i - 1];
        if (y < lh - 1 && excl[i + lw] != 2) d_raw += gS[i];
        if (y > 0 && excl[i - lw] != 2)      d_raw += gS[i - lw];
        if (d_raw < 1) d_raw = 1;
        recip[i] = (((int64_t)1) << 48) / d_raw;   // Q.32 reciprocal
    }
}

// ============================================================================
// The device MG hierarchy + the build driver (shared by the resident tick
// AND the test-only parity probe Ã¢â‚¬â€ one production code path).
// ============================================================================

struct DevLevel {
    int h = 0, w = 0;
    uint8_t* excl = nullptr;
    int64_t *m = nullptr, *gE = nullptr, *gS = nullptr, *recip = nullptr,
            *b = nullptr, *res = nullptr;
    int32_t* P = nullptr;
};

// Host-side replica of mg_build_levels' level-count loop (pure (h,w,config)
// arithmetic Ã¢â‚¬â€ deterministic, no device data).
int compute_n_levels(const EOSSolver& solver, int h, int w) {
    int n_levels = 1;
    int lh = h, lw = w;
    while ((lh < lw ? lh : lw) > solver.mg_min_dim
           && n_levels < MG_MAX_LEVELS_RES) {
        lh = (lh + 1) >> 1;
        lw = (lw + 1) >> 1;
        ++n_levels;
    }
    if (!solver.use_multigrid) n_levels = 1;
    return n_levels;
}

// Launch the whole device build: level-0 (excl Ã¢â€ â€™ m/b/P Ã¢â€ â€™ faces Ã¢â€ â€™ ÃÆ’) then per
// coarse level (excl+m Ã¢â€ â€™ faces) then recip on every level Ã¢â‚¬â€ the CPU's exact
// stage order (ÃÆ’ after b, before coarse + recip). One stream; each launch
// reads only buffers completed by earlier launches (design Ã‚Â§4, verified
// dependency-complete).
void mg_build_device(const DevLevel* lv, int n_levels,
                     const int32_t* d_pstar, const int32_t* d_div_u,
                     const int32_t* d_ntot, const int32_t* d_p_prev,
                     const bool* d_solid, const bool* d_is_vacuum,
                     const float* d_perm,
                     const bool* d_is_ambient, const int32_t* d_sigma,
                     const EOSSolver::MGScalarFolds& fold,
                     int32_t p_amb, bool ambient_mode, int n_env = 1) {
    // #29 P5: every level is (N, lh, lw); every launch covers the N envs.
    const int n0 = lv[0].h * lv[0].w;
    const int block = 256;
    cudaStream_t st = tstream();
    auto grid_for = [&](int n) { return bgrid(n, n_env, block); };

    K_L0_excl<<<grid_for(n0), block, 0, st>>>(lv[0].excl, d_solid, d_is_vacuum,
                                       d_is_ambient, ambient_mode, n0);
    cuda_check(cudaGetLastError(), "K_L0_excl");
    K_L0_mbP<<<grid_for(n0), block, 0, st>>>(lv[0].m, lv[0].b, lv[0].P, lv[0].excl,
                                      d_pstar, d_div_u, d_p_prev,
                                      fold.gamma_q, fold.dt_q,
                                      fold.Kdt2dx2_raw,
                                      (int64_t)p_amb, ambient_mode, n0);
    cuda_check(cudaGetLastError(), "K_L0_mbP");
    K_L0_faces<<<grid_for(n0), block, 0, st>>>(lv[0].gE, lv[0].gS, lv[0].excl,
                                        d_ntot, d_perm, fold.n_floor_q,
                                        lv[0].h, lv[0].w);
    cuda_check(cudaGetLastError(), "K_L0_faces");
    if (ambient_mode && d_sigma) {
        K_L0_sigma<<<grid_for(n0), block, 0, st>>>(lv[0].m, lv[0].excl, d_sigma, n0);
        cuda_check(cudaGetLastError(), "K_L0_sigma");
    }
    for (int l = 1; l < n_levels; ++l) {
        const DevLevel& F = lv[l - 1];
        const DevLevel& C = lv[l];
        const int cn = C.h * C.w;
        K_C_excl_m<<<grid_for(cn), block, 0, st>>>(C.excl, C.m, C.b, C.P,
                                            F.excl, F.m, F.gE, F.gS,
                                            F.h, F.w, C.h, C.w);
        cuda_check(cudaGetLastError(), "K_C_excl_m");
        K_C_faces<<<grid_for(cn), block, 0, st>>>(C.gE, C.gS, F.excl, F.gE, F.gS,
                                           F.h, F.w, C.h, C.w);
        cuda_check(cudaGetLastError(), "K_C_faces");
    }
    for (int l = 0; l < n_levels; ++l) {
        const int ln = lv[l].h * lv[l].w;
        K_recip<<<grid_for(ln), block, 0, st>>>(lv[l].recip, lv[l].excl, lv[l].m,
                                         lv[l].gE, lv[l].gS, lv[l].h, lv[l].w);
        cuda_check(cudaGetLastError(), "K_recip");
    }
}

// ============================================================================
// #29 P5: THE DEVICE PRE-STAGE (design Ã‚Â§4.2 "n_sub per env", Ã‚Â§5.1's first
// check). The host pre-stage's three reductions Ã¢â‚¬â€ t_max_abs (c_LOCAL's T
// scan), max_rad (max|u|Ã‚Â²) and max_du (the KÃ‚Â·|Ã¢Ë†â€¡P|Ã‚Â·dt/NÃŒâ€š scan, K_max_du above)
// Ã¢â‚¬â€ become PER-ENV device reductions (integer maxima: atomicMax is order-free);
// the per-cell planes the host used to build and upload every tick (cap2, the
// Ã‚Â§2.5 absorb plane; coeffE / coeffS since P4a) are built by device kernels
// from the resident planes; one thread per env folds its reductions into its
// substep count (eos_prestage_finish's transcription) and picks its row of
// the FOLD TABLE (built on the host once per config). The ONE host read is the
// batch's max(n_sub), 8 bytes, which drives the masked substep loop (P9
// replaces it with a WHILE graph node).
// ============================================================================

// The per-cell scan: capÃ‚Â² (eos_host_prestage's loop, verbatim), the env's
// T_abs and |u|Ã‚Â² maxima, and the absorb plane (eos_step_resident's old host
// hoist, verbatim). Every cell of cap2 / absorb_q written.
__global__ void K_pre_scan(int64_t* __restrict__ cap2, int32_t* __restrict__ absorb_q,
                           unsigned long long* __restrict__ pre,
                           const int32_t* __restrict__ temperature,
                           const int32_t* __restrict__ wind_x,
                           const int32_t* __restrict__ wind_y,
                           const float* __restrict__ wave_absorb,
                           const bool* __restrict__ solid,
                           const bool* __restrict__ is_vacuum,
                           const bool* __restrict__ ts,
                           int32_t s_eos_q, int32_t t_amb_q,
                           int64_t c_amb2_q32, int64_t u_max2_q32, int64_t ratio_umax,
                           int32_t absorb_dt_q, int n) {
    const size_t sn = (size_t)n;
    cap2 = bptr(cap2, sn); absorb_q = bptr(absorb_q, sn);
    pre = bptr(pre, (size_t)EOS_PRE_SLOTS);
    temperature = bptr(temperature, sn); wind_x = bptr(wind_x, sn); wind_y = bptr(wind_y, sn);
    wave_absorb = bptr(wave_absorb, sn); solid = bptr(solid, sn);
    is_vacuum = bptr(is_vacuum, sn); ts = bptr(ts, sn);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        absorb_q[i] = mul_q16(quantize((double)wave_absorb[i]), absorb_dt_q);
        const int64_t rad = mul_wide(wind_x[i], wind_x[i]) + mul_wide(wind_y[i], wind_y[i]);
        if (rad > 0) atomicMax(&pre[EOS_PRE_MAX_RAD], (unsigned long long)rad);
        if (solid[i] || is_vacuum[i]) { cap2[i] = u_max2_q32; continue; }
        const int64_t t_abs = (((int64_t)s_eos_q * (int64_t)temperature[i]) >> 16)
                              + (int64_t)t_amb_q;
        // The host's running max starts at t_amb_q (> 0): only a larger t_abs
        // (hence positive) can move it.
        if (t_abs > (int64_t)t_amb_q)
            atomicMax(&pre[EOS_PRE_T_MAX_ABS], (unsigned long long)t_abs);
        int64_t t_abs_cap = t_abs;
        if (ts[i] || t_abs_cap < (int64_t)t_amb_q) t_abs_cap = (int64_t)t_amb_q;
        const int64_t ratio = (t_abs_cap << 16) / (int64_t)t_amb_q;   // int64, NO narrow
        cap2[i] = (ratio >= ratio_umax) ? u_max2_q32
                                        : mul128_shr_signed(c_amb2_q32, ratio, 16);
    }
}

// The per-env fold, one thread per env: c_LOCAL (eos_host_prestage's tail of
// the T scan), max|u|, then eos_prestage_finish verbatim; the env's fold-table
// row (its c_LOCAL written in); the batch max of n_sub; a refused row (its
// k_flux_q out of range at build) raises the env's failure flag.
__global__ void K_pre_fold(EnvSub* __restrict__ es, unsigned long long* __restrict__ pre,
                           unsigned long long* __restrict__ nsub_max,
                           const EnvSub* __restrict__ table, int n_sub_cap,
                           int32_t t_amb_q, int32_t c_amb_q, int32_t u_max_q,
                           int32_t dt_q, int32_t cfl_dx_q, int n_env) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= n_env) return;
    unsigned long long* P = pre + (size_t)env * EOS_PRE_SLOTS;
    int64_t t_max_abs_raw = (int64_t)t_amb_q;
    if ((int64_t)P[EOS_PRE_T_MAX_ABS] > t_max_abs_raw)
        t_max_abs_raw = (int64_t)P[EOS_PRE_T_MAX_ABS];
    const int32_t ratio_q = (int32_t)((t_max_abs_raw << 16) / (int64_t)t_amb_q);
    const q16 sqrt_ratio = sqrt_q16_dev((int64_t)ratio_q << 16);   // Q.32 radicand
    q16 c_local_q = mul_q16(c_amb_q, sqrt_ratio);
    if (c_local_q < c_amb_q) c_local_q = c_amb_q;                   // never below ambient
    const q16 max_u = sqrt_q16_dev((int64_t)P[EOS_PRE_MAX_RAD]);
    // eos_prestage_finish (cuda_eos_step.cu), verbatim:
    int64_t u_est_raw = (int64_t)max_u + (int64_t)P[EOS_PRE_MAX_DU] + 1;
    const int64_t u_est_cap = ((int64_t)c_local_q > (int64_t)u_max_q)
        ? (int64_t)c_local_q : (int64_t)u_max_q;
    if (u_est_raw > u_est_cap) u_est_raw = u_est_cap;
    const int64_t numer_wide = mul128_shr_signed((int64_t)dt_q, u_est_raw, 16);
    const int64_t numer_c = numer_wide < (int64_t)INT32_MAX ? numer_wide : (int64_t)INT32_MAX;
    int n_sub = ceil_div((q16)numer_c, cfl_dx_q);
    if (n_sub < 1) n_sub = 1;
    if (n_sub > n_sub_cap) n_sub = n_sub_cap;
    EnvSub r = table[n_sub];
    r.c_local_q = c_local_q;
    es[env] = r;
    P[EOS_PRE_N_SUB] = (unsigned long long)n_sub;
    P[EOS_PRE_C_LOCAL] = (unsigned long long)(int64_t)c_local_q;
    if (r.k_flux_q == 0) P[EOS_PRE_FAIL] = 1ULL;
    atomicMax(nsub_max, (unsigned long long)n_sub);
}

// ============================================================================
// Persistent scratch Ã¢â‚¬â€ C++-owned, keyed (N, h, w, n_levels, n_cons, n_gases)
// (design Ã‚Â§3.2.5 + #29 P5's (N, h, w) key: n_levels in the key so a
// use_multigrid/mg_min_dim toggle re-keys). Allocated once, reused every tick;
// zero cudaMalloc steady-state. Every plane is (N, h, w); the books live in the
// resident tick's arena (EosBooksDev), not here.
// ============================================================================

struct EOSResidentScratch {
    int n_env = 0, h = 0, w = 0, n_levels = 0, n_cons = 0, n_gases = 0;
    // substep loop
    int32_t *svx = nullptr, *svy = nullptr, *st = nullptr;
    uint8_t* cmask = nullptr;
    int32_t *coeffE = nullptr, *coeffS = nullptr;
    int32_t *dq_e = nullptr, *dq_s = nullptr, *scale = nullptr;
    // P-E1 (design Ã‚Â§2.1.2): the transient energy plane, the n_bulk accumulator
    // and the applied-dq face planes Ã¢â‚¬â€ int64, scratch only, never synced.
    int64_t *e = nullptr, *nbulk = nullptr, *dqsum_e = nullptr, *dqsum_s = nullptr;
    // mid-stage + kick
    int32_t *div_u = nullptr, *ntot = nullptr, *pstar = nullptr,
            *absorb_q = nullptr;
    // #31: the blast's source, uploaded on a tick that has one (per env; an
    // env without one gets zeros, which K_div_source subtracts as a no-op).
    int32_t* div_source = nullptr;
    int32_t* div_source_h = nullptr;   // its pinned staging copy
    // VELOCITY-CLAMP (P-V1, D2v2): the per-cell capÃ‚Â² plane Ã¢â‚¬â€ #29 P5: BUILT ON
    // THE DEVICE every tick by K_pre_scan (no longer an H2D).
    int64_t* cap2 = nullptr;
    bool* cons_flag = nullptr;                 // (n_gases,) device flags
    std::vector<uint8_t> cons_flag_c;          // what cons_flag holds (upload-if-changed)
    // arc #54 P-G2: K3's per-tick scratch (design Ã‚Â§2.4/Ã‚Â§2.5).
    int64_t* e0 = nullptr;
    int32_t *pcur = nullptr, *s_plane = nullptr;
    // Smoke transport v2 (#12, P2b): the per-plane S_pre snapshot.
    int32_t* spre = nullptr;
    // #29 P5: the per-env fold-table rows this tick, the table, the pinned
    // read of max(n_sub).
    EnvSub* es = nullptr;
    EnvSub* table = nullptr;
    int table_rows = 0;
    std::vector<double> table_key;
    EnergyFluxScalarFolds ef_base;             // the folds' n_sub-free scalars
    unsigned long long* nsub_h = nullptr;
    // MG hierarchy
    DevLevel lv[MG_MAX_LEVELS_RES];

    void free_all() {
        auto f = [](void* p) { if (p) cudaFree(p); };
        f(svx); f(svy); f(st); f(cmask); f(coeffE); f(coeffS);
        f(dq_e); f(dq_s); f(scale); f(div_u); f(ntot); f(pstar); f(div_source);
        f(absorb_q); f(cap2); f(cons_flag);
        f(e); f(nbulk); f(dqsum_e); f(dqsum_s);
        f(e0); f(pcur); f(s_plane); f(spre); f(es);
        if (div_source_h) cudaFreeHost(div_source_h);
        div_source_h = nullptr;
        svx = svy = st = coeffE = coeffS = dq_e = dq_s = scale = nullptr;
        div_u = ntot = pstar = absorb_q = div_source = nullptr;
        cap2 = nullptr; cmask = nullptr; cons_flag = nullptr; cons_flag_c.clear();
        e = nbulk = dqsum_e = dqsum_s = nullptr;
        e0 = nullptr; pcur = nullptr; s_plane = nullptr; spre = nullptr; es = nullptr;
        for (auto& L : lv) {
            f(L.excl); f(L.m); f(L.gE); f(L.gS); f(L.recip); f(L.b);
            f(L.res); f(L.P);
            L = DevLevel{};
        }
        n_env = h = w = n_levels = n_cons = n_gases = 0;
    }

    void ensure(int N, int H, int W, int NL, int NC, int NG) {
        if (N == n_env && H == h && W == w && NL == n_levels && NC == n_cons
            && NG == n_gases && svx) return;
        free_all();
        n_env = N; h = H; w = W; n_levels = NL; n_cons = NC; n_gases = NG;
        const size_t n = (size_t)N * H * W;
        auto a32 = [&](int32_t** p, size_t cnt_, const char* what) {
            cuda_check(cudaMalloc(p, cnt_ * 4), what);
        };
        auto a64 = [&](int64_t** p, size_t cnt_, const char* what) {
            cuda_check(cudaMalloc(p, cnt_ * 8), what);
        };
        a32(&svx, n, "res malloc svx");
        a32(&svy, n, "res malloc svy");
        a32(&st, n, "res malloc st");
        cuda_check(cudaMalloc(&cmask, n), "res malloc cmask");
        a32(&coeffE, n, "res malloc coeffE");
        a32(&coeffS, n, "res malloc coeffS");
        a32(&dq_e, n, "res malloc dq_e");
        a32(&dq_s, n, "res malloc dq_s");
        a32(&scale, n, "res malloc scale");
        a32(&div_u, n, "res malloc div_u");
        a32(&div_source, n, "res malloc div_source");
        cuda_check(cudaMallocHost((void**)&div_source_h, n * 4), "res malloc pinned div_source");
        a32(&ntot, n, "res malloc ntot");
        a32(&pstar, n, "res malloc pstar");
        a32(&absorb_q, n, "res malloc absorb_q");
        cuda_check(cudaMalloc(&cons_flag, (size_t)(NG > 0 ? NG : 1)), "res malloc cons_flag");
        a64(&e0, n, "res malloc e0");
        a32(&pcur, n, "res malloc pcur");
        a32(&s_plane, n, "res malloc s_plane");
        a64(&cap2, n, "res malloc cap2");
        a64(&e, n, "res malloc e");
        a64(&nbulk, n, "res malloc nbulk");
        a64(&dqsum_e, n, "res malloc dqsum_e");
        a64(&dqsum_s, n, "res malloc dqsum_s");
        a32(&spre, n, "res malloc spre");
        cuda_check(cudaMalloc(&es, (size_t)N * sizeof(EnvSub)), "res malloc es");
        if (!nsub_h)
            cuda_check(cudaMallocHost((void**)&nsub_h, 8), "res malloc pinned nsub");
        int lh = H, lw = W;
        for (int l = 0; l < NL; ++l) {
            DevLevel& L = lv[l];
            L.h = lh; L.w = lw;
            const size_t ln = (size_t)N * lh * lw;
            cuda_check(cudaMalloc(&L.excl, ln), "res malloc L.excl");
            cuda_check(cudaMalloc(&L.m, ln * 8), "res malloc L.m");
            cuda_check(cudaMalloc(&L.gE, ln * 8), "res malloc L.gE");
            cuda_check(cudaMalloc(&L.gS, ln * 8), "res malloc L.gS");
            cuda_check(cudaMalloc(&L.recip, ln * 8), "res malloc L.recip");
            cuda_check(cudaMalloc(&L.b, ln * 8), "res malloc L.b");
            cuda_check(cudaMalloc(&L.res, ln * 8), "res malloc L.res");
            cuda_check(cudaMalloc(&L.P, ln * 4), "res malloc L.P");
            lh = (lh + 1) >> 1;
            lw = (lw + 1) >> 1;
        }
    }
};

EOSResidentScratch g_eos_res;

// THE FOLD TABLE (design Ã‚Â§4.2): one row per n_sub in [1, N_SUB_MAX] (row 0
// unused) Ã¢â‚¬â€ dt_s_q = quantize(dt / n_sub), the energy flux's k_flux_q and
// flux_pu_cap through energy_flux_scalar_folds (THE transcription). Built on
// the host once per config (keyed by every input), range-checked at build: a
// row whose k_flux_q is refused (the fold's own range guard) is kept with
// k_flux_q = 0, and an env that selects it raises EOS_PRE_FAIL Ã¢â‚¬â€ the host
// throws the fold's own message at the books fold (the sweep's pattern).
void ensure_fold_table(const EOSSolver& solver, float dt, EOSResidentScratch& S) {
    const int rows = solver.N_SUB_MAX + 1;
    const std::vector<double> key = {
        (double)dt, (double)solver.dx, (double)solver.adiabatic_index,
        (double)solver.T_AMB_K, (double)solver.C, (double)solver.T_MIN,
        (double)solver.T_MAX_PHYS, (double)solver.N_SUB_MAX};
    if (S.table && S.table_rows == rows && S.table_key == key) return;
    if (solver.N_SUB_MAX < 1)
        throw std::runtime_error("eos_step_resident: N_SUB_MAX < 1");
    std::vector<EnvSub> t((size_t)rows);
    std::memset(t.data(), 0, t.size() * sizeof(EnvSub));
    const double dt_d = (double)dt;
    for (int k = 1; k < rows; ++k) {
        EnvSub& r = t[(size_t)k];
        r.n_sub = k;
        r.dt_s_q = quantize(dt_d / (double)k);
        try {
            const EnergyFluxScalarFolds ef = energy_flux_scalar_folds(
                dt, solver.dx, solver.adiabatic_index, solver.T_AMB_K, solver.C,
                solver.T_MIN, solver.T_MAX_PHYS, k);
            r.k_flux_q = ef.k_flux_q;
            r.flux_pu_cap = ef.flux_pu_cap;
            S.ef_base = ef;          // t_amb_q / t_min_q / t_max_phys_q / c_q: n_sub-free
        } catch (const std::runtime_error&) {
            r.k_flux_q = 0;          // refused at build: selecting it fails the env
            r.flux_pu_cap = 0;
        }
    }
    if (S.table) cudaFree(S.table);
    S.table = nullptr;
    cuda_check(cudaMalloc(&S.table, (size_t)rows * sizeof(EnvSub)), "malloc fold table");
    cuda_check(cudaMemcpy(S.table, t.data(), (size_t)rows * sizeof(EnvSub),
                          cudaMemcpyHostToDevice), "H2D fold table");
    S.table_rows = rows;
    S.table_key = key;
}

// The device pre-stage on (N, h, w) planes: zeroed books expected (the caller
// memsets the pre slots and nsub_max). p_prev := atmosphere, the scan, the
// tick-entry Dalton N (K_ntot), the per-env max_du, the per-env fold, then
// coeffE / coeffS at each env's own dt_s_q. Launch only, on the tick stream.
// ONE transcription: the resident tick and the parity probe both run it.
void device_prestage(const EOSSolver& solver, const EOSHostPrestage& pre,
                     const KickScalarFolds& kf, EOSResidentScratch& S,
                     int N, int h, int w, int n_gases,
                     const int32_t* d_atmosphere, int32_t* d_wave_p,
                     const int32_t* d_wind_x, const int32_t* d_wind_y,
                     const int32_t* d_temperature, const int32_t* d_gas_base,
                     const bool* d_solid, const bool* d_is_vacuum, const bool* d_ts,
                     const float* d_dyn_permeability, const float* d_dyn_wave_absorb,
                     unsigned long long* d_pre, unsigned long long* d_nsub_max) {
    const int n = h * w;
    const int blk = 256;
    const dim3 g = bgrid(n, N, blk);
    cudaStream_t stv = tstream();
    // step 0 on device: P_prev := P (the host pre-stage's mirror copy).
    cuda_check(cudaMemcpyAsync(d_wave_p, d_atmosphere, (size_t)N * n * 4,
                               cudaMemcpyDeviceToDevice, stv), "D2D p_prev");
    K_pre_scan<<<g, blk, 0, stv>>>(S.cap2, S.absorb_q, d_pre, d_temperature,
                                   d_wind_x, d_wind_y, d_dyn_wave_absorb,
                                   d_solid, d_is_vacuum, d_ts,
                                   pre.s_eos_q, pre.t_amb_q, pre.c_amb2_q32,
                                   pre.u_max2_q32, pre.ratio_umax, kf.absorb_dt_q, n);
    cuda_check(cudaGetLastError(), "K_pre_scan");
    K_ntot<<<g, blk, 0, stv>>>(S.ntot, d_gas_base, S.cons_flag, n_gases, n);
    cuda_check(cudaGetLastError(), "K_ntot (pre-stage)");
    K_max_du<<<g, blk, 0, stv>>>(d_pre + EOS_PRE_MAX_DU, d_wave_p, S.ntot, d_solid,
                                 pre.inv_2dx_q, pre.kdt_raw, pre.n_floor_q, h, w,
                                 (size_t)EOS_PRE_SLOTS);
    cuda_check(cudaGetLastError(), "K_max_du");
    K_pre_fold<<<(N + 63) / 64, 64, 0, stv>>>(S.es, d_pre, d_nsub_max, S.table,
                                              solver.N_SUB_MAX, pre.t_amb_q, pre.c_amb_q,
                                              pre.u_max_q, pre.dt_q, pre.cfl_dx_q, N);
    cuda_check(cudaGetLastError(), "K_pre_fold");
    K_coeffs<<<g, blk, 0, stv>>>(S.coeffE, S.coeffS, d_solid, d_dyn_permeability,
                                 0, h, w, S.es);
    cuda_check(cudaGetLastError(), "K_coeffs");
}

void upload_cons_flags(EOSResidentScratch& S, const bool* gas_conservative, int n_gases) {
    std::vector<uint8_t> c((size_t)n_gases);
    for (int g = 0; g < n_gases; ++g) c[(size_t)g] = gas_conservative[g] ? 1 : 0;
    if (S.cons_flag_c == c) return;
    if (n_gases > 0)
        cuda_check(cudaMemcpy(S.cons_flag, c.data(), (size_t)n_gases,
                              cudaMemcpyHostToDevice), "H2D cons_flag");
    S.cons_flag_c = c;
}

}  // namespace

void div_source_launch_resident(
        int32_t* d_div_u, const int32_t* d_div_source,
        const bool* d_solid, const bool* d_is_vacuum, const bool* d_is_ambient,
        int N, int h, int w) {
    const int n = N * h * w;
    const int block = 256;
    K_div_source<<<(n + block - 1) / block, block, 0, tstream()>>>(
        d_div_u, d_div_source, d_solid, d_is_vacuum, d_is_ambient, n);
    cuda_check(cudaGetLastError(), "K_div_source");
}

long long eos_resident_calls() { return g_eos_resident_calls; }
long long eos_nsub_reads() { return g_eos_nsub_reads; }

// ============================================================================
// The resident EOS stage over the batch Ã¢â‚¬â€ see the header / the file banner for
// the stage map. #29 P5: (N, h, w) on every core, the device pre-stage, both
// substep loops to the batch's max n_sub with per-env masks, the books per env
// in the caller's arena (folded by the tick's one books D2H), one stream, no
// device sync and no allocation in steady state. Returns max(n_sub).
// ============================================================================
int eos_step_resident(
        const EOSSolver& solver,
        int N, int h, int w, float dt,
        const bool* gas_conservative, int n_gases,
        const int32_t* n_amb, int32_t p_amb, bool ambient_mode,
        const int32_t* const* div_source_host,
        int32_t* d_atmosphere, int32_t* d_wave_p,
        int32_t* d_wind_x, int32_t* d_wind_y,
        int32_t* d_temperature, int32_t* d_gas_base,
        const bool* d_solid, const bool* d_is_vacuum,
        const float* d_dyn_permeability, const float* d_dyn_wave_absorb,
        const bool* d_is_ambient,
        const int32_t* d_sponge_sigma, const int32_t* d_sponge_udamp,
        const bool* d_thermal_solid,
        int64_t* d_gas_energy,
        const EosBooksDev& books) {

    const int n = h * w;
    if (n <= 0 || dt <= 0.0f || N <= 0) return 0;   // step()'s degenerate early-out
    if (n_gases > EOS_BOOK_GAS_MAX) {
        throw std::invalid_argument("eos_step_resident: n_gases above EOS_BOOK_GAS_MAX");
    }
    if (ambient_mode && N != 1) {
        throw std::invalid_argument(
            "eos_step_resident: an ambient map runs only unbatched (N = 1): the sky "
            "exchange is a host bracket (ACCEPTED GAP, design Ã‚Â§4.2)");
    }
    ++g_eos_resident_calls;
    cudaStream_t stv = tstream();

    // THERMAL-MASS AXIS / P-E1: d_ts is the PARTICIPATION mask of the energy
    // build / recovery (ruling A1: the EOS never writes a ts tile's T); the P2
    // device fallback is d_solid.
    const bool* d_ts = d_thermal_solid ? d_thermal_solid : d_solid;

    // ---- the per-config scalar folds (ONE transcription each) ------------
    const EOSHostPrestage pre = eos_prestage_scalars(solver, dt, gas_conservative, n_gases);
    const KickScalarFolds kf = kick_scalar_folds(
        dt, solver.c_max, solver.dx, solver.adiabatic_index,
        solver.absorb_strength, solver.N_FLOOR_SOLVER, solver.U_MAX,
        // P-E3 (design Ã‚Â§2.8): interior drag + heat counterparty.
        // drag-law v2 (docs/drag_law_v2_design_2026-08-23.md): k_drag2.
        solver.k_drag, solver.k_drag2,
        // arc #54 Ã‚Â§2.1: T_AMB_K folds the derived k_ke constant.
        solver.T_AMB_K);
    const EOSSolver::MGScalarFolds mf = solver.mg_scalar_folds(dt);

    const int n_levels = compute_n_levels(solver, h, w);
    const int n_cons = (int)pre.cons.size();
    g_eos_res.ensure(N, h, w, n_levels, n_cons, n_gases);
    EOSResidentScratch& S = g_eos_res;
    upload_cons_flags(S, gas_conservative, n_gases);
    ensure_fold_table(solver, dt, S);

    // ---- PER-TICK ZERO RULE (design Ã‚Â§3.2.5): the EOS books, every env, ONE
    //      memset (the caller's arena lays them out contiguously). ---------
    cuda_check(cudaMemsetAsync(books.zero_begin, 0, books.zero_bytes, stv), "memset EOS books");
    const bool use_rail = ambient_mode && n_cons > 0;

    // Smoke transport v2 (#12, design Ã‚Â§2.1/Ã‚Â§5, P2b): EVERY trace plane rides
    // stages 3b/3c Ã¢â‚¬â€ the resident path skips none (no host scan, RL habits Ã‚Â§A;
    // an all-zero plane prices phi = floordiv(dqÃ‚Â·0, N) = 0 and stays zero with
    // zero counters, so this equals the CPU's skip-after-scan exactly).
    std::vector<int32_t*> trace_planes;
    for (int gi = 0; gi < n_gases; ++gi) {
        if (gas_conservative[gi]) continue;
        trace_planes.push_back(d_gas_base + (size_t)gi * n);
    }
    const int n_trace = (int)trace_planes.size();

    // ---- THE DEVICE PRE-STAGE + the ONE host read (max n_sub) ------------
    device_prestage(solver, pre, kf, S, N, h, w, n_gases, d_atmosphere, d_wave_p,
                    d_wind_x, d_wind_y, d_temperature, d_gas_base,
                    d_solid, d_is_vacuum, d_ts, d_dyn_permeability, d_dyn_wave_absorb,
                    books.pre, books.nsub_max);
    cuda_check(cudaMemcpyAsync(S.nsub_h, books.nsub_max, 8, cudaMemcpyDeviceToHost, stv),
               "D2H max n_sub");
    tick_sync_point("max_n_sub");   // the counted per-tick read (P9: a WHILE node)
    const int nsub_max = (int)*S.nsub_h;
    ++g_eos_nsub_reads;
    if (nsub_max < 1 || nsub_max > solver.N_SUB_MAX)
        throw std::logic_error("eos_step_resident: max(n_sub) outside [1, N_SUB_MAX]");

    // ---- K0: cmask ONCE per tick (the proven P6.2 device build). ---------
    sl_cmask_build_device(d_solid, d_is_vacuum, d_dyn_permeability, S.cmask,
                          N * n, d_is_ambient);
    // ---- P-E1 per-plane host side-tables the energy entry reads ----------
    std::vector<int32_t*> gas_planes(n_cons, nullptr);
    std::vector<int32_t> n_amb_cons(n_cons, 0);
    std::vector<unsigned long long*> rail_ptrs(n_cons, nullptr);
    for (int k = 0; k < n_cons; ++k) {
        gas_planes[k] = d_gas_base + (size_t)pre.cons[k] * n;
        n_amb_cons[k] = ambient_mode ? n_amb[pre.cons[k]] : 0;
        rail_ptrs[k] = use_rail ? books.rail + k : nullptr;
    }

    // arc #54 Ã‚Â§2.7 row 1: the mirror-only ceiling the substep transport's
    // recovery clamps against.
    const q16 t_max_phys_q_local = quantize((double)solver.T_MAX_PHYS);
    const size_t nb = (size_t)N * n * 4;

    // ---- DEVICE SUBSTEP LOOP (the P6.5 chain), to the batch's max n_sub; an
    //      env past its own n_sub sits out (every kernel masks it). --------
    for (int s = 0; s < nsub_max; ++s) {
        cuda_check(cudaMemcpyAsync(S.svx, d_wind_x, nb, cudaMemcpyDeviceToDevice, stv),
                   "D2D src_vx");
        cuda_check(cudaMemcpyAsync(S.svy, d_wind_y, nb, cudaMemcpyDeviceToDevice, stv),
                   "D2D src_vy");
        cuda_check(cudaMemcpyAsync(S.st, d_temperature, nb, cudaMemcpyDeviceToDevice, stv),
                   "D2D src_t");
        sl_advect3_device(d_wind_x, d_wind_y,
                          S.svx, S.svy, S.st,
                          d_solid, S.cmask, 0, h, w, N, S.es, s);
        BulkBatch bb;
        bb.n_env = N;
        bb.gs = (size_t)n_gases * n;
        bb.cstride = (size_t)EOS_ECNT_SLOTS;
        bb.rstride = (size_t)EOS_BOOK_GAS_MAX;
        bb.tstride = (size_t)EOS_TCNT_SLOTS;
        bb.es = S.es;
        bb.sub = s;
        // P-E1: the mass flux with the thermal energy riding it (design Ã‚Â§2.1).
        bulk_flux_energy_transport_device(
            gas_planes.data(), n_cons, d_temperature,
            d_gas_energy, pre.t_amb_q, t_max_phys_q_local,   // arc #54 Ã‚Â§2.2/Ã‚Â§2.7
            d_wind_x, d_wind_y,
            d_solid, d_is_vacuum, d_ts, S.coeffE, S.coeffS, pre.t_min_q, h, w,
            S.e, S.nbulk, S.dqsum_e, S.dqsum_s,
            S.dq_e, S.dq_s, S.scale, books.ecnt,
            d_is_ambient, n_amb_cons.data(),
            use_rail ? rail_ptrs.data() : nullptr,
            // smoke transport v2 (#12) stages 3b/3c on the resident planes:
            // vent -> boundary_flux_'s trace slots, wipe -> trace_wipe_sum_.
            n_trace ? trace_planes.data() : nullptr, n_trace, S.spre,
            n_trace ? books.tcnt : nullptr,
            n_trace ? books.tcnt + EOS_BOOK_GAS_MAX : nullptr,
            n_trace ? books.tcnt + 2 * EOS_BOOK_GAS_MAX : nullptr,    // #88 P2 clip
            n_trace ? books.tcnt + 3 * EOS_BOOK_GAS_MAX : nullptr,    // #88 P2 neg hits
            bb);
    }

    // ---- DEVICE MID-STAGE: div(u*), Dalton N, p* (per-cell, verbatim). ---
    const int block = 256;
    const dim3 grid = bgrid(n, N, block);
    K_div_u<<<grid, block, 0, stv>>>(S.div_u, d_wind_x, d_wind_y, d_solid,
                                     d_is_vacuum, d_is_ambient, pre.inv_2dx_q, h, w);
    cuda_check(cudaGetLastError(), "K_div_u");
    // #31: an env's blast source, staged per env (zeros where an env has
    // none: the subtraction of 0 is the identity); launched only when one env
    // has a source.
    bool any_source = false;
    for (int e = 0; e < N; ++e) any_source = any_source || div_source_host[e] != nullptr;
    if (any_source) {
        for (int e = 0; e < N; ++e) {
            int32_t* dst = S.div_source_h + (size_t)e * n;
            if (div_source_host[e]) std::memcpy(dst, div_source_host[e], (size_t)n * 4);
            else std::memset(dst, 0, (size_t)n * 4);
        }
        cuda_check(cudaMemcpyAsync(S.div_source, S.div_source_h, (size_t)N * n * 4,
                                   cudaMemcpyHostToDevice, stv), "H2D div_source");
        div_source_launch_resident(S.div_u, S.div_source, d_solid, d_is_vacuum,
                                   d_is_ambient, N, h, w);
    }
    {
        // trace_mass_scale arg RETIRED (P-T0, design Ã‚Â§2.6).
        K_ntot<<<grid, block, 0, stv>>>(S.ntot, d_gas_base, S.cons_flag, n_gases, n);
        cuda_check(cudaGetLastError(), "K_ntot");
    }
    K_pstar<<<grid, block, 0, stv>>>(S.pstar, S.ntot, d_temperature, d_wave_p,
                                     d_solid, d_is_vacuum, pre.t_amb_q, pre.s_eos_q, pre.c_q,
                                     solver.debug_pstar_from_prev, n);
    cuda_check(cudaGetLastError(), "K_pstar");

    // ---- DEVICE MG BUILD (design Ã‚Â§4) + the SHARED solve schedule, per env
    //      on the (N, h, w) layout (#29 P5). --------------------------------
    mg_build_device(S.lv, n_levels, S.pstar, S.div_u, S.ntot, d_wave_p,
                    d_solid, d_is_vacuum, d_dyn_permeability,
                    d_is_ambient, ambient_mode ? d_sponge_sigma : nullptr,
                    mf, p_amb, ambient_mode, N);
    {
        MGLevelDevPtrs views[MG_MAX_LEVELS_RES];
        for (int l = 0; l < n_levels; ++l) {
            const DevLevel& L = S.lv[l];
            views[l].h = L.h; views[l].w = L.w;
            views[l].excl = L.excl; views[l].m = L.m;
            views[l].gE = L.gE; views[l].gS = L.gS; views[l].recip = L.recip;
            views[l].b = L.b; views[l].P = L.P; views[l].res = L.res;
        }
        eos_mg_vcycle_resident(views, n_levels, solver.use_multigrid,
                               solver.mg_cycles, solver.mg_nu1, solver.mg_nu2,
                               solver.mg_coarsest_sweeps, solver.S, N);
    }

    // ---- KICK (the SHARED launch core, K1 only Ã¢â‚¬â€ arc #54 P-G2: K2 deleted)
    //      on the zeroed L0.P (== the per-call p_new bytes, PRE un-shift)
    //      with the device post-substep Dalton. The KE brackets (design
    //      Ã‚Â§2.3) land straight in the persistent `d_gas_energy`.
    kick_compression_launch_resident(
        d_wind_x, d_wind_y, d_gas_energy, S.lv[0].P, S.ntot, S.absorb_q,
        d_solid, d_is_vacuum, kf, S.cap2,   // D2v2: the per-cell capÃ‚Â² plane
        books.kick, h, w,
        d_is_ambient, d_sponge_udamp,
        d_ts,   // arc #54 Ã‚Â§2.3 F5: the KE brackets export on thermal_solid
        N, (size_t)KICK_CNT_SLOTS);

    // ---- step 5: P := P_new (+P_amb masked, ambient). --------------------
    K_store_atm<<<grid, block, 0, stv>>>(d_atmosphere, S.lv[0].P, d_solid,
                                         (int64_t)p_amb, ambient_mode, n);
    cuda_check(cudaGetLastError(), "K_store_atm");

    // ---- K3 + the once-per-tick RECOVERY (arc #54 design Ã‚Â§2.4/Ã‚Â§2.5/Ã‚Â§2.6) Ã¢â‚¬â€
    //      MUST run after K_store_atm: it consumes the ABSOLUTE d_atmosphere
    //      (the step-5 un-shift), never the shifted L0.P (design Ã‚Â§2.4 pin).
    //      #29 P5: the sub-cycles to max(n_sub), each env's k_flux_q /
    //      flux_pu_cap from its fold-table row; the n_sub-free scalars of the
    //      folds from the table build (ensure_fold_table's ef_base).
    EnergyFluxScalarFolds ef = S.ef_base;   // the n_sub-free scalars (any valid row's)
    ef.n_sub = nsub_max;
    energy_flux_launch_resident(
        d_gas_energy, d_temperature, d_atmosphere, d_wind_x, d_wind_y,
        S.ntot, d_solid, d_is_vacuum, d_ts, d_is_ambient,
        ef, S.e0, S.pcur, S.s_plane, books.flux, h, w,
        N, (size_t)FLUX_CNT_SLOTS, S.es, nsub_max);
    return nsub_max;
}

// ============================================================================
// TEST-ONLY: the build-parity probe (gate PART 1c) Ã¢â‚¬â€ host build vs the SAME
// production device-build path, byte-compared per level per array.
// ============================================================================
long long eos_mg_build_parity(
        const EOSSolver& solver,
        const int32_t* pstar, const int32_t* div_u, const int32_t* n_total,
        const int32_t* p_prev,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability,
        int h, int w, float dt,
        const bool* is_ambient, int32_t p_amb,
        const int32_t* sponge_sigma,
        std::string* report) {
    const int n = h * w;
    if (n <= 0 || dt <= 0.0f) return -1;
    const bool ambient_mode = (is_ambient != nullptr);

    // Host reference: the live mg_build_levels on the identical inputs.
    const int n_levels = solver.mg_build_levels(
        pstar, div_u, n_total, p_prev, solid, is_vacuum, dyn_permeability,
        h, w, dt, is_ambient, p_amb, sponge_sigma);
    if (n_levels <= 0) return -1;
    const auto& L = solver.mg_levels();

    // Device inputs + a TEMPORARY hierarchy (never the resident scratch).
    std::vector<void*> allocs;
    auto dev_alloc = [&](size_t bytes) -> void* {
        void* d = nullptr;
        cuda_check(cudaMalloc(&d, bytes), "parity malloc");
        allocs.push_back(d);
        return d;
    };
    auto dev_upload = [&](const void* src, size_t bytes) -> void* {
        void* d = dev_alloc(bytes);
        cuda_check(cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice),
                   "parity H2D");
        return d;
    };
    auto free_all = [&]() {
        for (void* d : allocs) cudaFree(d);
        allocs.clear();
    };

    long long mismatches = 0;
    std::ostringstream rep;
    try {
        const size_t nb = (size_t)n * 4;
        const int32_t* d_pstar = (const int32_t*)dev_upload(pstar, nb);
        const int32_t* d_div_u = (const int32_t*)dev_upload(div_u, nb);
        const int32_t* d_ntot  = (const int32_t*)dev_upload(n_total, nb);
        const int32_t* d_pprev = (const int32_t*)dev_upload(p_prev, nb);
        const bool* d_solid = (const bool*)dev_upload(solid, (size_t)n);
        const bool* d_vac   = (const bool*)dev_upload(is_vacuum, (size_t)n);
        const float* d_perm = (const float*)dev_upload(dyn_permeability, nb);
        const bool* d_amb = ambient_mode
            ? (const bool*)dev_upload(is_ambient, (size_t)n) : nullptr;
        const int32_t* d_sigma = (ambient_mode && sponge_sigma)
            ? (const int32_t*)dev_upload(sponge_sigma, nb) : nullptr;

        DevLevel lv[MG_MAX_LEVELS_RES];
        {
            int lh = h, lw = w;
            for (int l = 0; l < n_levels; ++l) {
                lv[l].h = lh; lv[l].w = lw;
                const size_t ln = (size_t)lh * lw;
                lv[l].excl  = (uint8_t*)dev_alloc(ln);
                lv[l].m     = (int64_t*)dev_alloc(ln * 8);
                lv[l].gE    = (int64_t*)dev_alloc(ln * 8);
                lv[l].gS    = (int64_t*)dev_alloc(ln * 8);
                lv[l].recip = (int64_t*)dev_alloc(ln * 8);
                lv[l].b     = (int64_t*)dev_alloc(ln * 8);
                lv[l].res   = (int64_t*)dev_alloc(ln * 8);
                lv[l].P     = (int32_t*)dev_alloc(ln * 4);
                lh = (lh + 1) >> 1;
                lw = (lw + 1) >> 1;
            }
        }
        // Poison the hierarchy so a missing every-cell write CANNOT
        // accidentally match the host's zeros (the stale-bytes hazard the
        // rule exists for Ã¢â‚¬â€ this makes the probe sensitive to it).
        for (int l = 0; l < n_levels; ++l) {
            const size_t ln = (size_t)lv[l].h * lv[l].w;
            cuda_check(cudaMemset(lv[l].excl, 0xAB, ln), "poison");
            cuda_check(cudaMemset(lv[l].m, 0xAB, ln * 8), "poison");
            cuda_check(cudaMemset(lv[l].gE, 0xAB, ln * 8), "poison");
            cuda_check(cudaMemset(lv[l].gS, 0xAB, ln * 8), "poison");
            cuda_check(cudaMemset(lv[l].recip, 0xAB, ln * 8), "poison");
            cuda_check(cudaMemset(lv[l].b, 0xAB, ln * 8), "poison");
            cuda_check(cudaMemset(lv[l].P, 0xAB, ln * 4), "poison");
        }

        const EOSSolver::MGScalarFolds mf = solver.mg_scalar_folds(dt);
        mg_build_device(lv, n_levels, d_pstar, d_div_u, d_ntot, d_pprev,
                        d_solid, d_vac, d_perm, d_amb, d_sigma,
                        mf, p_amb, ambient_mode);
        cuda_check(cudaDeviceSynchronize(), "parity sync");

        // Byte-compare every level's excl/m/gE/gS/recip/b/P (res is vcycle
        // scratch Ã¢â‚¬â€ the build never writes it on either side).
        auto compare = [&](int l, const char* name, const void* dev_ptr,
                           const void* host_ptr, size_t bytes,
                           size_t elem_size) {
            std::vector<uint8_t> buf(bytes);
            cuda_check(cudaMemcpy(buf.data(), dev_ptr, bytes,
                                  cudaMemcpyDeviceToHost), "parity D2H");
            if (std::memcmp(buf.data(), host_ptr, bytes) != 0) {
                size_t bad_elems = 0;
                const size_t n_elems = bytes / elem_size;
                for (size_t e = 0; e < n_elems; ++e) {
                    if (std::memcmp(buf.data() + e * elem_size,
                                    (const uint8_t*)host_ptr + e * elem_size,
                                    elem_size) != 0)
                        ++bad_elems;
                }
                mismatches += (long long)bad_elems;
                rep << "  level " << l << " " << name << ": "
                    << bad_elems << "/" << n_elems << " cells differ\n";
            }
        };
        for (int l = 0; l < n_levels; ++l) {
            const size_t ln = (size_t)lv[l].h * lv[l].w;
            compare(l, "excl",  lv[l].excl,  L[l].excl.data(),  ln,     1);
            compare(l, "m",     lv[l].m,     L[l].m.data(),     ln * 8, 8);
            compare(l, "gE",    lv[l].gE,    L[l].gE.data(),    ln * 8, 8);
            compare(l, "gS",    lv[l].gS,    L[l].gS.data(),    ln * 8, 8);
            compare(l, "recip", lv[l].recip, L[l].recip.data(), ln * 8, 8);
            compare(l, "b",     lv[l].b,     L[l].b.data(),     ln * 8, 8);
            compare(l, "P",     lv[l].P,     L[l].P.data(),     ln * 4, 4);
        }
        free_all();
    } catch (...) {
        free_all();
        throw;
    }
    if (report) *report = rep.str();
    return mismatches;
}

// ============================================================================
// TEST-ONLY (#29 P5, design Ã‚Â§5.1's first check): the DEVICE pre-stage against
// the HOST pre-stage at N = 1 on the same tick-entry planes Ã¢â‚¬â€ n_sub, c_LOCAL,
// dt_s_q, the capÃ‚Â² plane, the absorb plane and coeffE / coeffS, compared
// exactly. Runs device_prestage (the tick's own code path) on a private
// scratch; allocates and copies: never on the hot path.
// ============================================================================
long long eos_prestage_parity(
        const EOSSolver& solver,
        const int32_t* atmosphere, const int32_t* wind_x, const int32_t* wind_y,
        const int32_t* temperature, const int32_t* gas,
        const bool* gas_conservative, int n_gases,
        const bool* solid, const bool* is_vacuum, const bool* thermal_solid,
        const float* dyn_permeability, const float* dyn_wave_absorb,
        int h, int w, float dt, std::string* report) {
    const int n = h * w;
    if (n <= 0 || dt <= 0.0f) return -1;
    // ---- the HOST pre-stage (the per-call path's own, host_scans = true) --
    std::vector<int32_t> p_prev((size_t)n, 0);
    EOSHostPrestage hp = eos_host_prestage(
        solver, atmosphere, p_prev.data(), wind_x, wind_y, temperature,
        gas, gas_conservative, n_gases, solid, is_vacuum, dyn_permeability,
        h, w, dt, /*ambient_mode=*/false, thermal_solid, /*host_scans=*/true);
    const KickScalarFolds kf = kick_scalar_folds(
        dt, solver.c_max, solver.dx, solver.adiabatic_index,
        solver.absorb_strength, solver.N_FLOOR_SOLVER, solver.U_MAX,
        solver.k_drag, solver.k_drag2, solver.T_AMB_K);
    std::vector<int32_t> absorb_h((size_t)n);
    for (int i = 0; i < n; ++i)
        absorb_h[(size_t)i] = mul_q16(quantize((double)dyn_wave_absorb[i]), kf.absorb_dt_q);

    // ---- the DEVICE pre-stage on uploaded copies ---------------------------
    EOSResidentScratch S;
    std::vector<void*> allocs;
    auto up = [&](const void* src, size_t bytes) -> void* {
        void* d = nullptr;
        cuda_check(cudaMalloc(&d, bytes > 0 ? bytes : 8), "parity malloc");
        allocs.push_back(d);
        if (src) cuda_check(cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice), "parity H2D");
        return d;
    };
    long long bad = 0;
    std::ostringstream rep;
    try {
        const EOSHostPrestage pre = eos_prestage_scalars(solver, dt, gas_conservative, n_gases);
        S.ensure(1, h, w, compute_n_levels(solver, h, w), (int)pre.cons.size(), n_gases);
        upload_cons_flags(S, gas_conservative, n_gases);
        ensure_fold_table(solver, dt, S);
        const size_t nb = (size_t)n * 4;
        int32_t* d_atm  = (int32_t*)up(atmosphere, nb);
        int32_t* d_wp   = (int32_t*)up(nullptr, nb);
        int32_t* d_wx   = (int32_t*)up(wind_x, nb);
        int32_t* d_wy   = (int32_t*)up(wind_y, nb);
        int32_t* d_t    = (int32_t*)up(temperature, nb);
        int32_t* d_gas  = (int32_t*)up(gas, nb * (size_t)n_gases);
        bool* d_sol = (bool*)up(solid, (size_t)n);
        bool* d_vac = (bool*)up(is_vacuum, (size_t)n);
        bool* d_ts  = (bool*)up(thermal_solid ? thermal_solid : solid, (size_t)n);
        float* d_perm = (float*)up(dyn_permeability, nb);
        float* d_wabs = (float*)up(dyn_wave_absorb, nb);
        unsigned long long* d_pre = (unsigned long long*)up(nullptr, (size_t)EOS_PRE_SLOTS * 8 + 8);
        cuda_check(cudaMemset(d_pre, 0, (size_t)EOS_PRE_SLOTS * 8 + 8), "parity memset");
        // poison the planes the pre-stage must write in full
        cuda_check(cudaMemset(S.cap2, 0xAB, (size_t)n * 8), "poison");
        cuda_check(cudaMemset(S.absorb_q, 0xAB, nb), "poison");
        cuda_check(cudaMemset(S.coeffE, 0xAB, nb), "poison");
        cuda_check(cudaMemset(S.coeffS, 0xAB, nb), "poison");
        device_prestage(solver, pre, kf, S, 1, h, w, n_gases, d_atm, d_wp, d_wx, d_wy, d_t,
                        d_gas, d_sol, d_vac, d_ts, d_perm, d_wabs, d_pre, d_pre + EOS_PRE_SLOTS);
        cuda_check(cudaDeviceSynchronize(), "parity sync");
        unsigned long long hpre[EOS_PRE_SLOTS + 1];
        cuda_check(cudaMemcpy(hpre, d_pre, sizeof(hpre), cudaMemcpyDeviceToHost), "parity D2H");
        EnvSub es;
        cuda_check(cudaMemcpy(&es, S.es, sizeof(EnvSub), cudaMemcpyDeviceToHost), "parity D2H es");
        auto scalar = [&](const char* name, long long dev, long long host) {
            if (dev != host) {
                ++bad;
                rep << "  " << name << ": device " << dev << " != host " << host << "\n";
            }
        };
        scalar("n_sub", (long long)hpre[EOS_PRE_N_SUB], hp.n_sub);
        scalar("n_sub (row)", es.n_sub, hp.n_sub);
        scalar("max n_sub", (long long)hpre[EOS_PRE_SLOTS], hp.n_sub);
        scalar("c_local_q", (long long)(int64_t)hpre[EOS_PRE_C_LOCAL], hp.c_local_q);
        scalar("dt_s_q", es.dt_s_q, hp.dt_s_q);
        auto plane = [&](const char* name, const void* dev, const void* host, size_t elems,
                         size_t es_) {
            std::vector<uint8_t> buf(elems * es_);
            cuda_check(cudaMemcpy(buf.data(), dev, buf.size(), cudaMemcpyDeviceToHost),
                       "parity D2H plane");
            size_t nb_ = 0;
            for (size_t e = 0; e < elems; ++e)
                if (std::memcmp(buf.data() + e * es_, (const uint8_t*)host + e * es_, es_) != 0)
                    ++nb_;
            if (nb_) {
                bad += (long long)nb_;
                rep << "  " << name << ": " << nb_ << "/" << elems << " cells differ\n";
            }
        };
        plane("cap2", S.cap2, hp.cap2.data(), (size_t)n, 8);
        plane("absorb_q", S.absorb_q, absorb_h.data(), (size_t)n, 4);
        plane("coeffE", S.coeffE, hp.coeffE.data(), (size_t)n, 4);
        plane("coeffS", S.coeffS, hp.coeffS.data(), (size_t)n, 4);
        plane("p_prev", d_wp, p_prev.data(), (size_t)n, 4);
        if (report) {
            std::ostringstream hdr;
            hdr << "n_sub " << hp.n_sub << " c_local_q " << hp.c_local_q
                << " dt_s_q " << hp.dt_s_q << "\n";
            *report = hdr.str() + rep.str();
        }
    } catch (...) {
        for (void* d : allocs) cudaFree(d);
        S.free_all();
        if (S.table) cudaFree(S.table);
        if (S.nsub_h) cudaFreeHost(S.nsub_h);
        throw;
    }
    for (void* d : allocs) cudaFree(d);
    S.free_all();
    if (S.table) cudaFree(S.table);
    if (S.nsub_h) cudaFreeHost(S.nsub_h);
    return bad;
}

}  // namespace breach_cuda
