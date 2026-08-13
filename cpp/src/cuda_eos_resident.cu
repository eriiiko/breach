// ============================================================================
// CUDA-S8a Path A — the fully device-resident EOS tick — see cuda_eos_resident.h
// Design (BUILD AGAINST): docs/cuda_s8a_path_a_impl_2026-07-21.md (v2).
//
// STRUCTURE (mirrors eos_step_cuda stage-for-stage; §-refs are the design's):
//   host pre-stage        — the SHARED eos_host_prestage (§3.2.2) on the
//                           authoritative mirror (all reductions; §0).
//   device substep loop   — the SAME sl_advect3/bulk_flux launchers the
//                           per-call path chains (§3.2.6c), on the resident
//                           fields + persistent scratch.
//   device mid-stage      — K_div_u / K_ntot / K_pstar (§3.2.6d-f): per-cell
//                           verbatim transcriptions of the host mid-stage.
//   device MG build       — the mg_build_levels port (§4): per-cell and
//                           per-coarse-cell SINGLE-WRITER gathers + per-cell
//                           integer divides, level-sequential launches. THE
//                           EVERY-CELL-WRITE RULE: the CPU assign(0)s every
//                           array then writes subsets; the persistent
//                           hierarchy carries last tick's bytes, so every
//                           kernel writes EVERY cell of EVERY output array
//                           (else-branches write the assign's 0).
//   device solve          — eos_mg_vcycle_resident (the SHARED schedule).
//   device kick/compress  — kick_compression_launch_resident (SHARED core).
//   device store          — K_store_atm (§3.2.6j).
//
// DETERMINISM (§0/§4): no parallel reductions anywhere (the two atomics —
// bulk rail, kick counters — are order-free integer sums, digest-proven
// per-call). Every kernel is a single-writer gather; every cross-kernel
// dependency sits at a same-stream launch boundary == the CPU's pass/level
// boundary. Integer divides truncate toward zero identically on MSVC and
// nvcc. quantize((double)pf) on device is ONE identically-rounded IEEE
// double add after two exact steps (exact float->double, exact power-of-two
// multiply); FMA contraction is harmless because the product is exact.
// K_ntot accumulates int32 as uint32 (defined wrap == the hosts' observed
// int32 wrap for all inputs — design §3.2.6e; do NOT widen to int64).
//
// TELEMETRY: boundary_flux_ ASSIGNED and the five rail counters ACCUMULATED
// exactly as eos_step_cuda does; d_rail/d_cnt are PERSISTENT and therefore
// cudaMemset EVERY tick (§3.2.5 — the per-call wrappers' memsets moved
// here). Digests / dbg probes / host levels_ are stale on this path (§3.3).
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

constexpr int MG_MAX_LEVELS_RES = 9;   // == eos_solver.cpp's level-count cap

// ---- solid-mirror neighbor read (eos_solver.cpp mirror_idx, verbatim) -------
__device__ __forceinline__ int mirror_idx_dev_r(
        int self_i, int ny, int nx, int h, int w,
        const bool* __restrict__ solid) {
    if (ny < 0 || ny >= h || nx < 0 || nx >= w) return self_i;
    const int ni = ny * w + nx;
    if (solid[ni]) return self_i;
    return ni;
}

// ============================================================================
// MID-STAGE kernels — verbatim device transcriptions of eos_step_cuda's host
// mid-stage (cuda_eos_step.cu "HOST MID-STAGE" block), one writer per cell.
// ============================================================================

// div(u*) — the central-difference divergence on the post-substep wind.
// BC (audit (b)): the ring is a Dirichlet boundary — div(u*) = 0.
__global__ void K_div_u(int32_t* __restrict__ div_u,
                        const int32_t* __restrict__ wind_x,
                        const int32_t* __restrict__ wind_y,
                        const bool* __restrict__ solid,
                        const bool* __restrict__ is_vacuum,
                        const bool* __restrict__ is_ambient,
                        int32_t inv_2dx_q, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (solid[i] || is_vacuum[i] || (is_ambient && is_ambient[i])) {
            div_u[i] = 0; continue;
        }
        const int y = i / w;
        const int x = i - y * w;
        const int il = mirror_idx_dev_r(i, y, x - 1, h, w, solid);
        const int ir = mirror_idx_dev_r(i, y, x + 1, h, w, solid);
        const int iu = mirror_idx_dev_r(i, y - 1, x, h, w, solid);
        const int id = mirror_idx_dev_r(i, y + 1, x, h, w, solid);
        const q16 dux = mul_q16(wind_x[ir] - wind_x[il], inv_2dx_q);
        const q16 duy = mul_q16(wind_y[id] - wind_y[iu], inv_2dx_q);
        div_u[i] = dux + duy;
    }
}

// Step 2's Dalton sum (post-substep N) — per cell, gi ASCENDING (the CPU's
// per-cell add sequence). uint32 accumulation = defined wrap, byte-identical
// to the hosts' int32 wrap for all inputs (design §3.2.6e).
__global__ void K_ntot(int32_t* __restrict__ ntot,
                       const int32_t* __restrict__ gas_base,
                       const bool* __restrict__ cons_flag,
                       int n_gases, int32_t tms_q, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        int32_t acc = 0;
        for (int gi = 0; gi < n_gases; ++gi) {
            const int32_t v = gas_base[(size_t)gi * n + i];
            const int32_t term = cons_flag[gi] ? v : mul_q16(tms_q, v);
            acc = (int32_t)((uint32_t)acc + (uint32_t)term);
        }
        ntot[i] = acc;
    }
}

// Step 2's p* = C·N_total·T_abs (EOS floor at 0); the debug_pstar_from_prev
// MEASUREMENT-ONLY branch reads p_prev (the post-copy bytes, like the CPU).
__global__ void K_pstar(int32_t* __restrict__ pstar,
                        const int32_t* __restrict__ ntot,
                        const int32_t* __restrict__ temperature,
                        const int32_t* __restrict__ p_prev,
                        const bool* __restrict__ solid,
                        const bool* __restrict__ is_vacuum,
                        int32_t t_amb_q, int32_t s_eos_q, int32_t c_q,
                        bool debug_from_prev, int n) {
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

// Step 5: P := P_new, materialized once. BC: the solve ran in P' = P − P_amb;
// add P_amb back MASKED to !solid (solids stay 0 absolute — p_new is 0 there
// post-zero-excl either way; ring cells and every regular cell get the add).
__global__ void K_store_atm(int32_t* __restrict__ atmosphere,
                            const int32_t* __restrict__ p_new,
                            const bool* __restrict__ solid,
                            int64_t p_amb, bool ambient_mode, int n) {
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
// MG BUILD kernels — the on-device mg_build_levels port (design §4). Every
// kernel body is a VERBATIM transcription of eos_solver.cpp:727-1012 with
// mul128_shr -> mul128_shr_signed and reciprocal_q16 -> reciprocal_q16_dev.
// EVERY output array cell is written unconditionally (the every-cell rule).
// ============================================================================

constexpr int64_t MG_M_CAP   = ((int64_t)1) << 38;   // level-0 mass cap
constexpr int64_t MG_M_CAP_L = ((int64_t)1) << 44;   // coarse cap (sums grow ×4/level)

// Level-0 excl: the branch-priority chain (solid→2 beats vacuum→1 beats
// ambient→1; else 0). Every cell written.
__global__ void K_L0_excl(uint8_t* __restrict__ excl,
                          const bool* __restrict__ solid,
                          const bool* __restrict__ is_vacuum,
                          const bool* __restrict__ is_ambient,
                          bool ambient_mode, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        uint8_t e = 0;
        if (solid[i]) e = 2;
        else if (is_vacuum[i]) e = 1;
        else if (ambient_mode && is_ambient[i]) e = 1;   // BC ring → Dirichlet
        excl[i] = e;
    }
}

// Level-0 m / b / P-warm-start (eos_solver.cpp:805-834, verbatim — incl. the
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
        // rhs_i = p* − (γ·p*)·dt·div(û*) (int64 raw); b = m·rhs @F8.
        const int64_t gp_dt = mul128_shr_signed(gp_raw, (int64_t)dt_q, 16);
        int64_t rhs_raw = (int64_t)pstar[i]
                              - mul128_shr_signed(gp_dt, (int64_t)div_u[i], 16);
        // BC (spec §1 "the shift trick"): subtract BEFORE the m multiply.
        if (ambient_mode) rhs_raw -= p_amb;
        b[i] = mul128_shr_signed(mm, rhs_raw, 8);
        // Warm start from the previous tick's solved P; BC: re-shift fresh
        // into P′-space (widen-then-narrow — wrap-deterministic).
        P[i] = ambient_mode
            ? (int32_t)((int64_t)p_prev[i] - p_amb)
            : p_prev[i];
    }
}

// Level-0 face conductances g = perm/N̂ (eos_solver.cpp:839-866, verbatim).
// Each cell OWNS its E and S face — single writer; every cell written (0 at
// guard-fail / the x=w−1 / y=h−1 boundary). THE GUARD IS excl != 2 ON BOTH
// ENDPOINTS (NOT ==0): regular↔Dirichlet faces carry conductance — the
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

// BC (spec §3 rung 1, B3b): the σ-SPONGE fold (eos_solver.cpp:877-886,
// verbatim — incl. the REACHABLE M_CAP re-clamp). In-place add on the
// already-fully-written m; launched only in ambient mode with a σ grid.
// Runs AFTER K_L0_mbP (b used the un-σ'd m) and BEFORE the coarse build +
// recip (both fold σ) — the CPU's exact placement.
__global__ void K_L0_sigma(int64_t* __restrict__ m,
                           const uint8_t* __restrict__ excl,
                           const int32_t* __restrict__ sigma, int n) {
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
// Dirichlet rule (all-vacuum→1, all-non-regular→2, any-regular→0), the mass
// SUM over regular children, and the GALERKIN DIRICHLET ANCHOR (every fine
// face from a regular child to a Dirichlet fine cell lands on the coarse
// diagonal — the exact PᵀAP term). Child loops in the CPU's dy,dxx order
// with the fy/fx clipping; the anchor's out-of-block gE[fi−1]/gS[fi−fw]
// reads are read-only fine-level data. Every coarse cell writes excl, m,
// AND b=0 / P=0 (the CPU's assign — belt-and-braces: the vcycle's restrict
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
// (≤2) crossing fine faces whose BOTH endpoints are regular (== 0 — a
// regular→Dirichlet crossing face is the anchor term, NOT inter-cell
// coupling; note the guard DIFFERS from level 0's != 2). Every cell written.
__global__ void K_C_faces(int64_t* __restrict__ c_gE,
                          int64_t* __restrict__ c_gS,
                          const uint8_t* __restrict__ f_excl,
                          const int64_t* __restrict__ f_gE,
                          const int64_t* __restrict__ f_gS,
                          int fh, int fw, int ch, int cw) {
    const int cn = ch * cw;
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
// AND the test-only parity probe — one production code path).
// ============================================================================

struct DevLevel {
    int h = 0, w = 0;
    uint8_t* excl = nullptr;
    int64_t *m = nullptr, *gE = nullptr, *gS = nullptr, *recip = nullptr,
            *b = nullptr, *res = nullptr;
    int32_t* P = nullptr;
};

// Host-side replica of mg_build_levels' level-count loop (pure (h,w,config)
// arithmetic — deterministic, no device data).
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

// Launch the whole device build: level-0 (excl → m/b/P → faces → σ) then per
// coarse level (excl+m → faces) then recip on every level — the CPU's exact
// stage order (σ after b, before coarse + recip). One stream; each launch
// reads only buffers completed by earlier launches (design §4, verified
// dependency-complete).
void mg_build_device(const DevLevel* lv, int n_levels,
                     const int32_t* d_pstar, const int32_t* d_div_u,
                     const int32_t* d_ntot, const int32_t* d_p_prev,
                     const bool* d_solid, const bool* d_is_vacuum,
                     const float* d_perm,
                     const bool* d_is_ambient, const int32_t* d_sigma,
                     const EOSSolver::MGScalarFolds& fold,
                     int32_t p_amb, bool ambient_mode) {
    const int n0 = lv[0].h * lv[0].w;
    const int block = 256;
    auto grid_for = [&](int n) { return (n + block - 1) / block; };

    K_L0_excl<<<grid_for(n0), block>>>(lv[0].excl, d_solid, d_is_vacuum,
                                       d_is_ambient, ambient_mode, n0);
    cuda_check(cudaGetLastError(), "K_L0_excl");
    K_L0_mbP<<<grid_for(n0), block>>>(lv[0].m, lv[0].b, lv[0].P, lv[0].excl,
                                      d_pstar, d_div_u, d_p_prev,
                                      fold.gamma_q, fold.dt_q,
                                      fold.Kdt2dx2_raw,
                                      (int64_t)p_amb, ambient_mode, n0);
    cuda_check(cudaGetLastError(), "K_L0_mbP");
    K_L0_faces<<<grid_for(n0), block>>>(lv[0].gE, lv[0].gS, lv[0].excl,
                                        d_ntot, d_perm, fold.n_floor_q,
                                        lv[0].h, lv[0].w);
    cuda_check(cudaGetLastError(), "K_L0_faces");
    if (ambient_mode && d_sigma) {
        K_L0_sigma<<<grid_for(n0), block>>>(lv[0].m, lv[0].excl, d_sigma, n0);
        cuda_check(cudaGetLastError(), "K_L0_sigma");
    }
    for (int l = 1; l < n_levels; ++l) {
        const DevLevel& F = lv[l - 1];
        const DevLevel& C = lv[l];
        const int cn = C.h * C.w;
        K_C_excl_m<<<grid_for(cn), block>>>(C.excl, C.m, C.b, C.P,
                                            F.excl, F.m, F.gE, F.gS,
                                            F.h, F.w, C.h, C.w);
        cuda_check(cudaGetLastError(), "K_C_excl_m");
        K_C_faces<<<grid_for(cn), block>>>(C.gE, C.gS, F.excl, F.gE, F.gS,
                                           F.h, F.w, C.h, C.w);
        cuda_check(cudaGetLastError(), "K_C_faces");
    }
    for (int l = 0; l < n_levels; ++l) {
        const int ln = lv[l].h * lv[l].w;
        K_recip<<<grid_for(ln), block>>>(lv[l].recip, lv[l].excl, lv[l].m,
                                         lv[l].gE, lv[l].gS, lv[l].h, lv[l].w);
        cuda_check(cudaGetLastError(), "K_recip");
    }
}

// ============================================================================
// Persistent scratch — C++-owned, keyed (h, w, n_levels, n_cons, n_gases)
// (design §3.2.5: n_levels in the key so a use_multigrid/mg_min_dim toggle
// re-keys). Allocated once, reused every tick; zero cudaMalloc steady-state.
// ============================================================================

struct EOSResidentScratch {
    int h = 0, w = 0, n_levels = 0, n_cons = 0, n_gases = 0;
    // substep loop
    int32_t *svx = nullptr, *svy = nullptr, *st = nullptr;
    uint8_t* cmask = nullptr;
    // THERMAL-MASS AXIS, P-EOS: the T-ONLY occluder mask (cmask with every
    // thermal_solid tile forced sealed). Persistent like every other scratch
    // plane and REBUILT every tick — the mask is NOT static (on_tile_changed
    // patches it when a crate burns out), so a cached copy would go stale.
    uint8_t* tcmask = nullptr;
    int32_t *coeffE = nullptr, *coeffS = nullptr;
    int32_t *dq_e = nullptr, *dq_s = nullptr, *scale = nullptr;
    // mid-stage + kick
    int32_t *div_u = nullptr, *ntot = nullptr, *pstar = nullptr,
            *absorb_q = nullptr;
    bool* cons_flag = nullptr;                 // (n_gases,) device flags
    unsigned long long* rail = nullptr;        // (n_cons,)
    unsigned long long* cnt = nullptr;         // (5,)
    // MG hierarchy
    DevLevel lv[MG_MAX_LEVELS_RES];

    void free_all() {
        auto f = [](void* p) { if (p) cudaFree(p); };
        f(svx); f(svy); f(st); f(cmask); f(tcmask); f(coeffE); f(coeffS);
        f(dq_e); f(dq_s); f(scale); f(div_u); f(ntot); f(pstar);
        f(absorb_q); f(cons_flag); f(rail); f(cnt);
        svx = svy = st = coeffE = coeffS = dq_e = dq_s = scale = nullptr;
        div_u = ntot = pstar = absorb_q = nullptr;
        cmask = nullptr; tcmask = nullptr;
        cons_flag = nullptr; rail = nullptr; cnt = nullptr;
        for (auto& L : lv) {
            f(L.excl); f(L.m); f(L.gE); f(L.gS); f(L.recip); f(L.b);
            f(L.res); f(L.P);
            L = DevLevel{};
        }
        h = w = n_levels = n_cons = n_gases = 0;
    }

    void ensure(int H, int W, int NL, int NC, int NG) {
        if (H == h && W == w && NL == n_levels && NC == n_cons
            && NG == n_gases && svx) return;
        free_all();
        h = H; w = W; n_levels = NL; n_cons = NC; n_gases = NG;
        const size_t n = (size_t)H * W;
        auto a32 = [&](int32_t** p, size_t cnt_, const char* what) {
            cuda_check(cudaMalloc(p, cnt_ * 4), what);
        };
        a32(&svx, n, "res malloc svx");
        a32(&svy, n, "res malloc svy");
        a32(&st, n, "res malloc st");
        cuda_check(cudaMalloc(&cmask, n), "res malloc cmask");
        // THERMAL-MASS AXIS: allocated unconditionally (one byte/cell, the
        // cmask precedent) and simply left unwritten/unread on maps where the
        // thermal and gas media cannot diverge — the scratch key stays
        // (h,w,n_levels,n_cons,n_gases), so no re-keying hazard is introduced.
        cuda_check(cudaMalloc(&tcmask, n), "res malloc tcmask");
        a32(&coeffE, n, "res malloc coeffE");
        a32(&coeffS, n, "res malloc coeffS");
        a32(&dq_e, n, "res malloc dq_e");
        a32(&dq_s, n, "res malloc dq_s");
        a32(&scale, n, "res malloc scale");
        a32(&div_u, n, "res malloc div_u");
        a32(&ntot, n, "res malloc ntot");
        a32(&pstar, n, "res malloc pstar");
        a32(&absorb_q, n, "res malloc absorb_q");
        cuda_check(cudaMalloc(&cons_flag, (size_t)NG), "res malloc cons_flag");
        if (NC > 0)
            cuda_check(cudaMalloc(&rail, (size_t)NC * 8), "res malloc rail");
        cuda_check(cudaMalloc(&cnt, 5 * 8), "res malloc cnt");
        int lh = H, lw = W;
        for (int l = 0; l < NL; ++l) {
            DevLevel& L = lv[l];
            L.h = lh; L.w = lw;
            const size_t ln = (size_t)lh * lw;
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

}  // namespace

long long eos_resident_calls() { return g_eos_resident_calls; }

// ============================================================================
// The resident EOS tick — see the header / the file banner for the stage map.
// ============================================================================
void eos_step_resident(
        const EOSSolver& solver,
        const int32_t* atmosphere,
        int32_t* p_prev,
        const int32_t* wind_x, const int32_t* wind_y,
        const int32_t* temperature,
        const int32_t* gas, const bool* gas_conservative, int n_gases,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability, const float* dyn_wave_absorb,
        int h, int w, float dt,
        const bool* is_ambient, const int32_t* n_amb, int32_t p_amb,
        const bool* thermal_solid,
        int32_t* d_atmosphere, int32_t* d_wave_p,
        int32_t* d_wind_x, int32_t* d_wind_y,
        int32_t* d_temperature, int32_t* d_gas_base,
        const bool* d_solid, const bool* d_is_vacuum,
        const float* d_dyn_permeability,
        const bool* d_is_ambient,
        const int32_t* d_sponge_sigma, const int32_t* d_sponge_udamp,
        const bool* d_thermal_solid) {

    const int n = h * w;
    if (n <= 0 || dt <= 0.0f) return;   // step()'s degenerate early-out
    ++g_eos_resident_calls;

    const bool ambient_mode = (is_ambient != nullptr);
    // THERMAL-MASS AXIS, P-EOS: the medium mask + the T-only occluder predicate.
    // `use_tsol` requires BOTH the mirror (for the shared host predicate — all
    // reductions read the authoritative mirror, design §0) and the device copy
    // (what the kernels read). d_ts is the P2 device fallback: nothing is
    // allocated or copied on the legacy path.
    const bool use_tsol = (thermal_solid != nullptr) && (d_thermal_solid != nullptr);
    const bool* d_ts = use_tsol ? d_thermal_solid : d_solid;
    const bool t_occlude = use_tsol
        && eos_thermal_occludes(thermal_solid, solid, dyn_permeability, n);

    // ---- HOST PRE-STAGE on the authoritative mirror (the SHARED verbatim
    //      transcription; all reductions — design §0). Writes the mirror
    //      p_prev, boundary_flux_, dbg_last_n_sub / dbg_last_c_local_q. ----
    const EOSHostPrestage pre = eos_host_prestage(
        solver, atmosphere, p_prev, wind_x, wind_y, temperature,
        gas, gas_conservative, n_gases, solid, is_vacuum,
        dyn_permeability, h, w, dt, ambient_mode);

    // ---- host fold helpers (ONE transcription each — design §3.2.3) -------
    const KickScalarFolds kf = kick_scalar_folds(
        dt, solver.c_max, solver.dx, solver.adiabatic_index,
        solver.absorb_strength, solver.N_FLOOR_SOLVER, solver.T_MIN,
        solver.T_WORK_CLAMP, solver.T_MAX_PHYS, solver.U_MAX);
    const EOSSolver::MGScalarFolds mf = solver.mg_scalar_folds(dt);

    // ---- §2.5 hoist: the per-cell absorb plane, on the MIRROR (this is
    //      where body-shielding lives — dyn_wave_absorb is a host input). ---
    std::vector<int32_t> absorb_q(n);
    for (int i = 0; i < n; ++i)
        absorb_q[i] = mul_q16(quantize((double)dyn_wave_absorb[i]),
                              kf.absorb_dt_q);

    const int n_levels = compute_n_levels(solver, h, w);
    const int n_cons = (int)pre.cons.size();
    g_eos_res.ensure(h, w, n_levels, n_cons, n_gases);
    EOSResidentScratch& S = g_eos_res;

    const size_t nb = (size_t)n * 4;

    // ---- per-tick hoisted-plane H2D (per-tick INPUTS, not mid-tick traffic)
    cuda_check(cudaMemcpy(S.coeffE, pre.coeffE.data(), nb,
                          cudaMemcpyHostToDevice), "H2D coeffE");
    cuda_check(cudaMemcpy(S.coeffS, pre.coeffS.data(), nb,
                          cudaMemcpyHostToDevice), "H2D coeffS");
    cuda_check(cudaMemcpy(S.absorb_q, absorb_q.data(), nb,
                          cudaMemcpyHostToDevice), "H2D absorb_q");
    cuda_check(cudaMemcpy(S.cons_flag, gas_conservative, (size_t)n_gases,
                          cudaMemcpyHostToDevice), "H2D cons_flag");

    // ---- PER-TICK ZERO RULE (design §3.2.5): the persistent rail + counter
    //      buffers carry last tick's sums — memset EVERY tick (the per-call
    //      wrappers' memsets, moved here). --------------------------------
    cuda_check(cudaMemset(S.cnt, 0, 5 * 8), "memset cnt");
    const bool use_rail = ambient_mode && n_cons > 0;
    if (use_rail)
        cuda_check(cudaMemset(S.rail, 0, (size_t)n_cons * 8), "memset rail");

    // ---- step 0 on device: P_prev := P (D2D — same bytes the host copy
    //      just wrote to the mirror wave_p). ------------------------------
    cuda_check(cudaMemcpy(d_wave_p, d_atmosphere, nb,
                          cudaMemcpyDeviceToDevice), "D2D p_prev");

    // ---- K0: cmask ONCE per tick (the proven P6.2 device build). ---------
    sl_cmask_build_device(d_solid, d_is_vacuum, d_dyn_permeability, S.cmask,
                          n, d_is_ambient);
    // ---- K0b (THERMAL-MASS AXIS, P-EOS): the T-ONLY occluder mask, ONCE per
    //      tick, rebuilt from the CURRENT device thermal_solid (the caller keeps
    //      it fresh via the per-tick from_host upload). `S.cmask` is untouched —
    //      velocity/pressure/gas flow must stay identical (ruling §4 item 4).
    const uint8_t* d_tcmask = nullptr;
    if (t_occlude) {
        sl_tcmask_build_device(S.cmask, d_thermal_solid, S.tcmask, n);
        d_tcmask = S.tcmask;
    }

    // ---- DEVICE SUBSTEP LOOP (the P6.5 chain, resident buffers). ---------
    for (int s = 0; s < pre.n_sub; ++s) {
        cuda_check(cudaMemcpy(S.svx, d_wind_x, nb,
                              cudaMemcpyDeviceToDevice), "D2D src_vx");
        cuda_check(cudaMemcpy(S.svy, d_wind_y, nb,
                              cudaMemcpyDeviceToDevice), "D2D src_vy");
        cuda_check(cudaMemcpy(S.st, d_temperature, nb,
                              cudaMemcpyDeviceToDevice), "D2D src_t");
        sl_advect3_device(d_wind_x, d_wind_y, d_temperature,
                          S.svx, S.svy, S.st,
                          d_solid, d_is_vacuum, S.cmask, pre.dt_s_q, h, w,
                          d_is_ambient, d_ts, d_tcmask);
        for (int k = 0; k < n_cons; ++k) {
            bulk_flux_plane_device(
                d_gas_base + (size_t)pre.cons[k] * n,
                d_wind_x, d_wind_y, d_solid, d_is_vacuum,
                S.coeffE, S.coeffS, S.dq_e, S.dq_s, S.scale, h, w,
                d_is_ambient,
                ambient_mode ? n_amb[pre.cons[k]] : 0,
                use_rail ? &S.rail[k] : nullptr);
        }
    }

    // ---- DEVICE MID-STAGE: div(u*), Dalton N, p* (per-cell, verbatim). ---
    const int block = 256;
    const int grid = (n + block - 1) / block;
    K_div_u<<<grid, block>>>(S.div_u, d_wind_x, d_wind_y, d_solid,
                             d_is_vacuum, d_is_ambient, pre.inv_2dx_q, h, w);
    cuda_check(cudaGetLastError(), "K_div_u");
    {
        const q16 tms_q = quantize((double)solver.trace_mass_scale);
        K_ntot<<<grid, block>>>(S.ntot, d_gas_base, S.cons_flag, n_gases,
                                tms_q, n);
        cuda_check(cudaGetLastError(), "K_ntot");
    }
    K_pstar<<<grid, block>>>(S.pstar, S.ntot, d_temperature, d_wave_p,
                             d_solid, d_is_vacuum, pre.t_amb_q, pre.s_eos_q, pre.c_q,
                             solver.debug_pstar_from_prev, n);
    cuda_check(cudaGetLastError(), "K_pstar");

    // ---- DEVICE MG BUILD (design §4) + the SHARED solve schedule. --------
    mg_build_device(S.lv, n_levels, S.pstar, S.div_u, S.ntot, d_wave_p,
                    d_solid, d_is_vacuum, d_dyn_permeability,
                    d_is_ambient, ambient_mode ? d_sponge_sigma : nullptr,
                    mf, p_amb, ambient_mode);
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
                               solver.mg_coarsest_sweeps, solver.S);
    }

    // ---- KICK + COMPRESSION (the SHARED launch core) on the zeroed L0.P
    //      (== the per-call p_new bytes) with the device post-substep Dalton.
    kick_compression_launch_resident(
        d_wind_x, d_wind_y, d_temperature, S.lv[0].P, S.ntot, S.absorb_q,
        d_solid, d_is_vacuum, kf, pre.c_local_q, S.cnt, h, w,
        d_is_ambient, d_sponge_udamp,
        d_ts);   // THERMAL-MASS AXIS: step 4c skips its T write on thermal_solid

    // ---- step 5: P := P_new (+P_amb masked, ambient). --------------------
    K_store_atm<<<grid, block>>>(d_atmosphere, S.lv[0].P, d_solid,
                                 (int64_t)p_amb, ambient_mode, n);
    cuda_check(cudaGetLastError(), "K_store_atm");

    cuda_check(cudaDeviceSynchronize(), "resident sync");

    // ---- telemetry D2H (scalars only): rail ASSIGNED, counters ACCUMULATED
    //      — exactly the per-call semantics (design §3.2.7). ---------------
    if (use_rail) {
        std::vector<unsigned long long> rail_host(n_cons, 0);
        cuda_check(cudaMemcpy(rail_host.data(), S.rail, (size_t)n_cons * 8,
                              cudaMemcpyDeviceToHost), "D2H rail");
        for (int k = 0; k < n_cons; ++k)
            solver.boundary_flux_[pre.cons[k]] = (int64_t)rail_host[k];
    }
    unsigned long long cnt_host[5] = {0, 0, 0, 0, 0};
    cuda_check(cudaMemcpy(cnt_host, S.cnt, 5 * 8, cudaMemcpyDeviceToHost),
               "D2H counters");
    solver.u_clamp_hits      += (int64_t)cnt_host[0];
    solver.u_max_hits        += (int64_t)cnt_host[1];
    solver.work_clamp_hits   += (int64_t)cnt_host[2];
    solver.energy_floor_hits += (int64_t)cnt_host[3];
    solver.t_max_phys_hits   += (int64_t)cnt_host[4];
}

// ============================================================================
// TEST-ONLY: the build-parity probe (gate PART 1c) — host build vs the SAME
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
        // rule exists for — this makes the probe sensitive to it).
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
        // scratch — the build never writes it on either side).
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

}  // namespace breach_cuda
