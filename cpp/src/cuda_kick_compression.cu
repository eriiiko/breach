// ============================================================================
// EOS P6.4 — momentum kick; P-G2 — the face-flux energy step's GPU twin.
// See cuda_kick_compression.h for the stage map. A bit-identical GPU port of
// eos_solver.cpp's EOSSolver::step steps 4 (the kick, WITH the arc #54 §2.3
// kinetic-energy brackets), 6 (the face-flux energy step, §2.4/§2.5) and 7
// (the once-per-tick recovery, §2.6). K2 (the old step-4c compression-work
// kernel) is DELETED with the arc (design §5) — see cuda_kick_compression.h.
//
// Host-side precompute, verbatim step()'s (/fp:strict host pass):
//   * the scalar folds (K_raw/Kdt_raw, inv_2dx_q, absorb_dt_q, k_ke, the
//     flux constant k_flux_q + its int64-corner cap);
//   * step 2's Dalton N_total loop (the kick's 1/N̂ input);
//   * the §2.5 hoist: a_q[i] = mul_q16(quantize(dyn_wave_absorb[i]),
//     absorb_dt_q) — the identical per-cell expression the CPU evaluates
//     inside its loop, computed once into a per-tick plane so the kernels
//     are float-free (same double math, same rounding, per-tick-constant
//     input — the blessed P3 hoist class).
// ============================================================================
#include "cuda_kick_compression.h"
#include "cuda_resident.h" // S8a Path A: KickScalarFolds/EnergyFluxScalarFolds + launch cores
#include "fixed_point.h"   // q16, quantize, mul_q16, sat_add_q16, FP_ONE, floordiv_q, make_recip
#include "cuda_fixedpoint_device.cuh"  // mul128_shr_signed, reciprocal_q16_dev, sqrt_q16_dev

#include <cuda_runtime.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace fixedpoint;

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in eos_kick_compression/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// ---- digest: the cheap FNV-1a-style running hash over a Q16.16 buffer -------
// HOST-side (review §2.6). Byte-for-byte the anon-namespace digest_of in
// eos_solver.cpp (sequential, order-dependent, pure integer).
uint64_t digest_of_host(const int32_t* buf, int n, uint64_t seed) {
    uint64_t h = seed ^ 1469598103934665603ULL;
    for (int i = 0; i < n; ++i) {
        h ^= (uint64_t)(uint32_t)buf[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// ---- solid-mirror neighbor read (eos_solver.cpp mirror_idx, verbatim) -------
__device__ __forceinline__ int mirror_idx_dev(
        int self_i, int ny, int nx, int h, int w,
        const bool* __restrict__ solid) {
    if (ny < 0 || ny >= h || nx < 0 || nx >= w) return self_i;
    const int ni = ny * w + nx;
    if (solid[ni]) return self_i;
    return ni;
}

// ---- ΔKE -> energy, the ONE transcription (eos_solver.cpp's ke_energy) -----
// t = mul128_shr(k_ke_recip_q32, du2_raw, 48)  // Q32·Q32>>48 = Q16 ΔT
// dE = mul128_shr(n_bulk, t, 0)                // Q16·Q16 = Q32 energy
__device__ __forceinline__ int64_t ke_energy_dev(
        int64_t k_ke_recip_q32, int64_t n_bulk, int64_t du2_raw) {
    const int64_t t = mul128_shr_signed(k_ke_recip_q32, du2_raw, 48);
    return mul128_shr_signed(n_bulk, t, 0);
}

// ============================================================================
// K1 — the momentum kick + the per-stage KINETIC-ENERGY BRACKETS (arc #54
// design §2.3). Pure gather: reads its OWN u, the solved-P plane (never
// written here), N_total, the hoisted absorb plane; writes ITS OWN u AND
// (design §2.3) its OWN gas_energy. Every stage — ∇p kick, dyn_wave_absorb,
// the B3c sponge band, the velocity cap, staged drag L/Q — is ruled
// INDIVIDUALLY per cell, verbatim eos_solver.cpp's kick loop (EOSSolver::
// step, NOT the isolated eos_kick_compression_reference twin: the reference
// omits the KE brackets' counter bookkeeping, this is the LIVE path).
// ============================================================================
__global__ void kick_kernel(int32_t* __restrict__ wind_x,
                            int32_t* __restrict__ wind_y,
                            int64_t* __restrict__ gas_energy,   // arc #54 §2.3
                            const int32_t* __restrict__ p_new,
                            const int32_t* __restrict__ n_total,
                            const int32_t* __restrict__ absorb_q,   // §2.5 hoist
                            const bool* __restrict__ solid,
                            const bool* __restrict__ is_vacuum,
                            const bool* __restrict__ ts,
                            int64_t Kdt_raw, int32_t inv_2dx_q,
                            int32_t n_floor_q,
                            const int64_t* __restrict__ cap2_plane,  // D2v2, >= 0
                            int64_t u_max2_q32,                      // D3
                            int32_t kd_q, int32_t kd2_q,
                            int64_t rad_dead_q32,                    // drag-law v2
                            int64_t k_ke_recip_q32,                  // arc #54 §2.1
                            unsigned long long* __restrict__ cnt,
                            int h, int w,
                            const bool* __restrict__ is_ambient,
                            const int32_t* __restrict__ sponge_udamp) {
    const int n = h * w;
    const int64_t KE_SAFE = (int64_t)1 << 27;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        // BC: ring u ≡ 0 — a still boundary, the vacuum idiom.
        if (solid[i] || is_vacuum[i] || (is_ambient && is_ambient[i])) {
            wind_x[i] = 0; wind_y[i] = 0; continue;
        }
        const int y = i / w;
        const int x = i % w;
        const int il = mirror_idx_dev(i, y, x - 1, h, w, solid);
        const int ir = mirror_idx_dev(i, y, x + 1, h, w, solid);
        const int iu = mirror_idx_dev(i, y - 1, x, h, w, solid);
        const int id = mirror_idx_dev(i, y + 1, x, h, w, solid);
        const int64_t gx = mul128_shr_signed((int64_t)(p_new[ir] - p_new[il]),
                                             (int64_t)inv_2dx_q, 16);
        const int64_t gy = mul128_shr_signed((int64_t)(p_new[id] - p_new[iu]),
                                             (int64_t)inv_2dx_q, 16);
        int64_t ux = (int64_t)wind_x[i];
        int64_t uy = (int64_t)wind_y[i];
        // (a) LOAD-SIDE CLAMP (§2.3 R3-#5b) — counted, before any bracket opens.
        if      (ux >  KE_SAFE) { ux =  KE_SAFE; atomicAdd(&cnt[13], 1ULL); }
        else if (ux < -KE_SAFE) { ux = -KE_SAFE; atomicAdd(&cnt[13], 1ULL); }
        if      (uy >  KE_SAFE) { uy =  KE_SAFE; atomicAdd(&cnt[13], 1ULL); }
        else if (uy < -KE_SAFE) { uy = -KE_SAFE; atomicAdd(&cnt[13], 1ULL); }
        const int64_t n_bulk_ke = (int64_t)n_total[i];
        const bool is_ts = (ts && ts[i]);
        const bool ke_stores = !is_ts;
        int64_t u2_prev = ux * ux + uy * uy;
        if (gx != 0 || gy != 0) {   // micro-opt kept: du == 0 exactly at zero
                                    // gradient — skip the reciprocal chain
                                    // (bit-identical, the CPU's own branch)
            q16 nhat = n_total[i];
            if (nhat < n_floor_q) nhat = n_floor_q;
            const q16 inv_n = reciprocal_q16_dev(nhat);
            ux -= mul128_shr_signed(mul128_shr_signed(Kdt_raw, gx, 16),
                                    (int64_t)inv_n, 16);
            uy -= mul128_shr_signed(mul128_shr_signed(Kdt_raw, gy, 16),
                                    (int64_t)inv_n, 16);
        }
        // (b) THE COMPONENT GUARD, moved here and tightened to 2^27 —
        // UNCONDITIONAL (outside the gx/gy branch above, R3-#5a), counted.
        if      (ux >  KE_SAFE) { ux =  KE_SAFE; atomicAdd(&cnt[13], 1ULL); }
        else if (ux < -KE_SAFE) { ux = -KE_SAFE; atomicAdd(&cnt[13], 1ULL); }
        if      (uy >  KE_SAFE) { uy =  KE_SAFE; atomicAdd(&cnt[13], 1ULL); }
        else if (uy < -KE_SAFE) { uy = -KE_SAFE; atomicAdd(&cnt[13], 1ULL); }
        // BRACKET 1 — the ∇p kick (reversible exchange with the field).
        {
            const int64_t u2 = ux * ux + uy * uy;
            const int64_t dE = ke_energy_dev(k_ke_recip_q32, n_bulk_ke, u2 - u2_prev);
            if (ke_stores) {
                gas_energy[i] -= dE;
                atomicAdd(&cnt[9], (unsigned long long)dE);            // e_kick_ke_sum
            } else {
                atomicAdd(&cnt[14], (unsigned long long)(-dE));        // e_ts_ke_sum
            }
            u2_prev = u2;
        }

        // absorption damping u *= (1 − absorb·dt), magnitude-first.
        const q16 a = absorb_q[i];
        if (a > 0) {
            const q16 kk = (a < FP_ONE) ? (q16)(FP_ONE - a) : 0;
            const int64_t mx = mul128_shr_signed(ux < 0 ? -ux : ux, (int64_t)kk, 16);
            const int64_t my = mul128_shr_signed(uy < 0 ? -uy : uy, (int64_t)kk, 16);
            ux = (ux < 0) ? -mx : mx;
            uy = (uy < 0) ? -my : my;
        }
        // BRACKET 2 — dyn_wave_absorb: EXPORTED (D6), unconditional.
        {
            const int64_t u2 = ux * ux + uy * uy;
            const int64_t dE = ke_energy_dev(k_ke_recip_q32, n_bulk_ke, u2 - u2_prev);
            if (ke_stores) atomicAdd(&cnt[10], (unsigned long long)(-dE));  // e_absorb_export_sum
            else           atomicAdd(&cnt[14], (unsigned long long)(-dE));  // e_ts_ke_sum
            u2_prev = u2;
        }

        // BC (spec §3 rung 2, B3c): the u-DAMPING BAND, magnitude-first.
        if (is_ambient && sponge_udamp) {
            const int32_t kd = sponge_udamp[i];
            if (kd > 0) {
                const q16 kk2 = (kd < FP_ONE) ? (q16)(FP_ONE - kd) : 0;
                const int64_t mx = mul128_shr_signed(ux < 0 ? -ux : ux, (int64_t)kk2, 16);
                const int64_t my = mul128_shr_signed(uy < 0 ? -uy : uy, (int64_t)kk2, 16);
                ux = (ux < 0) ? -mx : mx;
                uy = (uy < 0) ? -my : my;
            }
        }
        // BRACKET 3 — the B3c sponge band: EXPORTED (D6).
        {
            const int64_t u2 = ux * ux + uy * uy;
            const int64_t dE = ke_energy_dev(k_ke_recip_q32, n_bulk_ke, u2 - u2_prev);
            if (ke_stores) atomicAdd(&cnt[11], (unsigned long long)(-dE));  // e_sponge_export_sum
            else           atomicAdd(&cnt[14], (unsigned long long)(-dE));  // e_ts_ke_sum
            u2_prev = u2;
        }

        // |u| <= per-cell cap2_plane[i] (VELOCITY-CLAMP, P-V1, D2v2), counted
        // per CELL; the exact rad > cap² test (no Chebyshev pre-test).
        const int64_t cap2_q32 = cap2_plane[i];   // D5: trusted verbatim
        const bool cap_is_umax = (cap2_q32 >= u_max2_q32);
        const int64_t rad = ux * ux + uy * uy;   // <= 2^55 (KE_SAFE guard)
        if (rad > cap2_q32) {
            atomicAdd(&cnt[0], 1ULL);                    // u_clamp_hits
            if (cap_is_umax) atomicAdd(&cnt[1], 1ULL);   // u_max_hits
            const q16 umag    = sqrt_q16_dev(rad);
            const q16 u_cap_q = sqrt_q16_dev(cap2_q32);
            ux = (ux * (int64_t)u_cap_q) / (int64_t)umag;
            uy = (uy * (int64_t)u_cap_q) / (int64_t)umag;
        }
        // BRACKET 4 — the velocity cap: DESTROYED and counted (D6).
        {
            const int64_t u2 = ux * ux + uy * uy;
            const int64_t dE = ke_energy_dev(k_ke_recip_q32, n_bulk_ke, u2 - u2_prev);
            if (ke_stores) atomicAdd(&cnt[12], (unsigned long long)(-dE));  // e_clamp_destroyed_sum
            else           atomicAdd(&cnt[14], (unsigned long long)(-dE));  // e_ts_ke_sum
            u2_prev = u2;
        }

        // P-E3 — interior drag + heat counterparty (design §2.8), drag-law v2
        // (stage L linear, stage Q implicit quadratic). ts cells skip BOTH
        // the drag and the deposit (ruling A1).
        if ((kd_q > 0 || kd2_q > 0) && !is_ts) {
            const int64_t ux_old = ux, uy_old = uy;
            if (kd_q > 0) {
                const q16 kk_drag = (kd_q < FP_ONE) ? (q16)(FP_ONE - kd_q) : 0;
                const int64_t dmx = mul128_shr_signed(ux_old < 0 ? -ux_old : ux_old, (int64_t)kk_drag, 16);
                const int64_t dmy = mul128_shr_signed(uy_old < 0 ? -uy_old : uy_old, (int64_t)kk_drag, 16);
                ux = (ux_old < 0) ? -dmx : dmx;
                uy = (uy_old < 0) ? -dmy : dmy;
            }
            if (kd2_q > 0) {
                const int64_t rad1 = ux * ux + uy * uy;
                if (rad1 >= rad_dead_q32) {
                    const q16 umag = sqrt_q16_dev(rad1);
                    const int64_t prod  = mul128_shr_signed((int64_t)kd2_q, (int64_t)umag, 16);
                    const int64_t denom = (int64_t)FP_ONE + prod;
                    ux = (ux * (int64_t)FP_ONE) / denom;
                    uy = (uy * (int64_t)FP_ONE) / denom;
                }
            }
            const int64_t du2_raw = (ux_old * ux_old + uy_old * uy_old)
                                   - (ux * ux + uy * uy);   // >= 0 structurally
            const int64_t n_bulk = (int64_t)n_total[i];
            atomicAdd(&cnt[5], (unsigned long long)mul128_shr_signed(n_bulk, du2_raw, 16));  // ke_drag_removed

            // BRACKET 5 — DRAG HEAT (D5): the whole removed KE, at the
            // derived k_ke, straight into gas_energy. No heat fraction, no
            // c_v divide, no per-deposit rail (the §2.6 recovery owns it).
            const int64_t dE_drag = ke_energy_dev(k_ke_recip_q32, n_bulk, du2_raw);  // >= 0
            gas_energy[i] += dE_drag;
            atomicAdd(&cnt[6], (unsigned long long)dE_drag);   // e_drag_heat_sum
        }

        wind_x[i] = (int32_t)ux;   // the ONE narrow at store (safe by
        wind_y[i] = (int32_t)uy;   // construction — caps ≪ int32 range)
    }
}

// ============================================================================
// K3 — the face-flux energy step (arc #54 design §2.4/§2.5). The accountable
// predicate, ONE transcription (eos_solver.cpp's `accountable` lambda /
// bulk_transport.cpp's `e_participates`).
// ============================================================================
__device__ __forceinline__ bool accountable_dev(
        int i, const bool* __restrict__ solid, const bool* __restrict__ ts,
        const bool* __restrict__ is_vacuum, const bool* __restrict__ is_ambient) {
    return !solid[i] && !ts[i] && !is_vacuum[i]
           && !(is_ambient != nullptr && is_ambient[i]);
}

// The per-face price, the ONE transcription both passes call (eos_solver.cpp
// §2.4's `face_flux` lambda, ported to a plain device function — no device
// lambdas: this codebase's nvcc pass has no --extended-lambda). `lo`/`hi` is
// the canonical pair (lo < hi); `east` selects the wind component. Returns
// the SIGNED flux (positive == energy flows lo -> hi); `cls` reports the
// face class (0 no face, 1 interior, 2 outflow lo accountable, 3 outflow hi
// accountable). `book` gates the probe/hit telemetry (pass B only, once per
// face — see the call sites below for which visit books).
__device__ __forceinline__ int64_t face_flux_dev(
        int lo, int hi, bool east,
        const int32_t* __restrict__ pcur,
        const int32_t* __restrict__ wind_x, const int32_t* __restrict__ wind_y,
        const int32_t* __restrict__ n_total,
        const bool* __restrict__ solid, const bool* __restrict__ ts,
        const bool* __restrict__ is_vacuum, const bool* __restrict__ is_ambient,
        int32_t k_flux_q, int64_t flux_pu_cap,
        int& cls, bool book, unsigned long long* __restrict__ cnt) {
    cls = 0;
    const bool a_lo = accountable_dev(lo, solid, ts, is_vacuum, is_ambient);
    const bool a_hi = accountable_dev(hi, solid, ts, is_vacuum, is_ambient);
    if (!a_lo && !a_hi) return 0;
    // WALL: solid or thermal_solid on either side kills the face (design F4).
    if (solid[lo] || solid[hi] || ts[lo] || ts[hi]) {
        if (book) {
            const int acc_i = a_lo ? lo : hi;
            const int64_t pa = pcur[acc_i];
            const int64_t ua = east ? wind_x[acc_i] : wind_y[acc_i];
            const int64_t m = mul128_shr_signed(
                mul128_shr_signed(pa, ua < 0 ? -ua : ua, 16),
                (int64_t)k_flux_q, 0);
            if (solid[lo] || solid[hi])
                atomicAdd(&cnt[6], (unsigned long long)m);   // e_wall_work_probe_sum
            else
                atomicAdd(&cnt[5], (unsigned long long)m);   // e_ts_work_sum
        }
        return 0;
    }
    int64_t p_f, u_f;
    if (a_lo && a_hi) {
        const int64_t n_lo = (int64_t)n_total[lo];
        const int64_t n_hi = (int64_t)n_total[hi];
        const int64_t ns = n_lo + n_hi;
        if (ns <= 0) return 0;
        // eq. 15 LITERALLY (D3): harmonic-flavoured N x arithmetic T_abs.
        p_f = floordiv_q((int64_t)pcur[hi] * n_lo + (int64_t)pcur[lo] * n_hi, ns);
        // ARITHMETIC face mean (F16); the 1/2 is folded into k_flux_q.
        u_f = (int64_t)(east ? wind_x[lo] : wind_y[lo])
            + (int64_t)(east ? wind_x[hi] : wind_y[hi]);
        cls = 1;
    } else {
        const int acc_i = a_lo ? lo : hi;
        p_f = pcur[acc_i];
        // ring/vacuum u == 0, so (u_acc + 0)/2 * 2 == u_acc.
        u_f = east ? wind_x[acc_i] : wind_y[acc_i];
        cls = a_lo ? 2 : 3;
    }
    if (u_f == 0 || p_f <= 0) return 0;
    const int64_t uabs = u_f < 0 ? -u_f : u_f;
    // The PINNED magnitude chain (§2.4, R3-#4): Q16*Q16>>16 = Q16, *Q16>>0 = Q32.
    int64_t pu = mul128_shr_signed(p_f, uabs, 16);
    if (pu > flux_pu_cap) { pu = flux_pu_cap; if (book) atomicAdd(&cnt[2], 1ULL); }  // flux_sat_hits
    const int64_t mag = pu * (int64_t)k_flux_q;   // <= 2^60 by the cap
    return (u_f > 0) ? mag : -mag;   // sign AFTER truncation
}

// The donor-only rail's per-face scale, the ONE transcription (eos_solver.cpp
// §2.4's `apply_scale` lambda, ported to a plain device function).
__device__ __forceinline__ int64_t apply_scale_dev(
        int64_t f, int donor, int self,
        const int32_t* __restrict__ s_plane,
        const bool* __restrict__ solid, const bool* __restrict__ ts,
        const bool* __restrict__ is_vacuum, const bool* __restrict__ is_ambient,
        unsigned long long* __restrict__ cnt) {
    const int64_t s = accountable_dev(donor, solid, ts, is_vacuum, is_ambient)
                     ? (int64_t)s_plane[donor] : (int64_t)FP_ONE;
    if (s >= (int64_t)FP_ONE) return f;
    const int64_t m = mul128_shr_signed(f < 0 ? -f : f, s, 16);
    if (donor == self)
        atomicAdd(&cnt[3], (unsigned long long)((f < 0 ? -f : f) - m));  // e_energy_floor_sum
    return (f < 0) ? -m : m;
}

// ---- the sub-cycle pressure refresh, INCREMENT FORM (design §2.4 R3-#6) ----
__global__ void energy_pressure_refresh_kernel(
        int32_t* __restrict__ pcur,
        const int64_t* __restrict__ gas_energy, const int64_t* __restrict__ e0,
        const int32_t* __restrict__ atmosphere, const int32_t* __restrict__ n_total,
        const bool* __restrict__ solid, const bool* __restrict__ ts,
        const bool* __restrict__ is_vacuum, const bool* __restrict__ is_ambient,
        int32_t c_q, int32_t t_max_phys_q, int32_t t_amb_q,
        unsigned long long* __restrict__ cnt, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (!accountable_dev(i, solid, ts, is_vacuum, is_ambient)) { pcur[i] = 0; continue; }
        int64_t p = (int64_t)atmosphere[i]
                  + mul128_shr_signed((int64_t)c_q, gas_energy[i] - e0[i], 32);
        if (p < 0) { p = 0; atomicAdd(&cnt[0], 1ULL); }   // p_face_floor_hits
        const int64_t e_ceil = (int64_t)n_total[i]
            * ((int64_t)t_max_phys_q + (int64_t)t_amb_q);
        const int64_t p_ceil = mul128_shr_signed((int64_t)c_q, e_ceil, 32);
        if (p > p_ceil) { p = p_ceil; atomicAdd(&cnt[1], 1ULL); }  // p_face_ceil_hits
        if (p > (int64_t)INT32_MAX) p = INT32_MAX;
        pcur[i] = (int32_t)p;
    }
}

// ---- PASS A: OUT_i and the donor-only rail scale s_i (design §2.4 F3) -----
__global__ void energy_flux_pass_a_kernel(
        const int32_t* __restrict__ pcur,
        const int32_t* __restrict__ wind_x, const int32_t* __restrict__ wind_y,
        const int32_t* __restrict__ n_total, const int64_t* __restrict__ gas_energy,
        const bool* __restrict__ solid, const bool* __restrict__ ts,
        const bool* __restrict__ is_vacuum, const bool* __restrict__ is_ambient,
        int32_t k_flux_q, int64_t flux_pu_cap, int64_t t_min_abs_raw,
        int32_t* __restrict__ s_plane, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (!accountable_dev(i, solid, ts, is_vacuum, is_ambient)) { s_plane[i] = FP_ONE; continue; }
        const int y = i / w;
        const int x = i - y * w;
        int64_t out = 0;
        int cls;
        if (x < w - 1) {   // EAST face of i: pair (i, i+1), i is lo.
            const int64_t f = face_flux_dev(i, i + 1, true, pcur, wind_x, wind_y,
                n_total, solid, ts, is_vacuum, is_ambient, k_flux_q, flux_pu_cap,
                cls, false, nullptr);
            if (cls != 0 && f > 0) out += f;
        }
        if (x > 0) {       // WEST face of i: pair (i-1, i), i is hi.
            const int64_t f = face_flux_dev(i - 1, i, true, pcur, wind_x, wind_y,
                n_total, solid, ts, is_vacuum, is_ambient, k_flux_q, flux_pu_cap,
                cls, false, nullptr);
            if (cls != 0 && f < 0) out += -f;
        }
        if (y < h - 1) {   // SOUTH face of i: pair (i, i+w), i is lo.
            const int64_t f = face_flux_dev(i, i + w, false, pcur, wind_x, wind_y,
                n_total, solid, ts, is_vacuum, is_ambient, k_flux_q, flux_pu_cap,
                cls, false, nullptr);
            if (cls != 0 && f > 0) out += f;
        }
        if (y > 0) {       // NORTH face of i: pair (i-w, i), i is hi.
            const int64_t f = face_flux_dev(i - w, i, false, pcur, wind_x, wind_y,
                n_total, solid, ts, is_vacuum, is_ambient, k_flux_q, flux_pu_cap,
                cls, false, nullptr);
            if (cls != 0 && f < 0) out += -f;
        }
        int64_t head = gas_energy[i] - (int64_t)n_total[i] * t_min_abs_raw;
        if (head < 0) head = 0;
        if (head >= out) {
            s_plane[i] = FP_ONE;
        } else {
            s_plane[i] = (int32_t)floordiv_q(head, (out >> 16) + 1);
        }
    }
}

// ---- PASS B: apply, each face scaled by its DONOR's s (design §2.4/§2.5) --
__global__ void energy_flux_pass_b_kernel(
        const int32_t* __restrict__ pcur,
        const int32_t* __restrict__ wind_x, const int32_t* __restrict__ wind_y,
        const int32_t* __restrict__ n_total, int64_t* __restrict__ gas_energy,
        const bool* __restrict__ solid, const bool* __restrict__ ts,
        const bool* __restrict__ is_vacuum, const bool* __restrict__ is_ambient,
        int32_t k_flux_q, int64_t flux_pu_cap,
        const int32_t* __restrict__ s_plane,
        unsigned long long* __restrict__ cnt, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (!accountable_dev(i, solid, ts, is_vacuum, is_ambient)) continue;
        const int y = i / w;
        const int x = i - y * w;
        int64_t de = 0, exp_out = 0;
        int cls;
        if (x < w - 1) {                 // EAST: i is lo
            const int64_t f = face_flux_dev(i, i + 1, true, pcur, wind_x, wind_y,
                n_total, solid, ts, is_vacuum, is_ambient, k_flux_q, flux_pu_cap,
                cls, true, cnt);
            if (cls != 0) {
                const int64_t a = apply_scale_dev(f, f > 0 ? i : i + 1, i,
                    s_plane, solid, ts, is_vacuum, is_ambient, cnt);
                de -= a;
                if (cls != 1) exp_out += a;
            }
        }
        if (x > 0) {                     // WEST: i is hi
            const int64_t f = face_flux_dev(i - 1, i, true, pcur, wind_x, wind_y,
                n_total, solid, ts, is_vacuum, is_ambient, k_flux_q, flux_pu_cap,
                cls, true, cnt);
            if (cls != 0) {
                const int64_t a = apply_scale_dev(f, f > 0 ? i - 1 : i, i,
                    s_plane, solid, ts, is_vacuum, is_ambient, cnt);
                de += a;
                if (cls != 1) exp_out -= a;
            }
        }
        if (y < h - 1) {                 // SOUTH: i is lo
            const int64_t f = face_flux_dev(i, i + w, false, pcur, wind_x, wind_y,
                n_total, solid, ts, is_vacuum, is_ambient, k_flux_q, flux_pu_cap,
                cls, true, cnt);
            if (cls != 0) {
                const int64_t a = apply_scale_dev(f, f > 0 ? i : i + w, i,
                    s_plane, solid, ts, is_vacuum, is_ambient, cnt);
                de -= a;
                if (cls != 1) exp_out += a;
            }
        }
        if (y > 0) {                     // NORTH: i is hi
            const int64_t f = face_flux_dev(i - w, i, false, pcur, wind_x, wind_y,
                n_total, solid, ts, is_vacuum, is_ambient, k_flux_q, flux_pu_cap,
                cls, true, cnt);
            if (cls != 0) {
                const int64_t a = apply_scale_dev(f, f > 0 ? i - w : i, i,
                    s_plane, solid, ts, is_vacuum, is_ambient, cnt);
                de += a;
                if (cls != 1) exp_out -= a;
            }
        }
        gas_energy[i] += de;
        if (exp_out != 0)
            atomicAdd(&cnt[4], (unsigned long long)exp_out);   // e_work_export_sum
    }
}

// ---- the once-per-tick RECOVERY (design §2.6) ------------------------------
__global__ void energy_recovery_kernel(
        int64_t* __restrict__ gas_energy, int32_t* __restrict__ temperature,
        const int32_t* __restrict__ n_total,
        const bool* __restrict__ solid, const bool* __restrict__ ts,
        const bool* __restrict__ is_vacuum, const bool* __restrict__ is_ambient,
        int32_t t_min_q, int32_t t_max_phys_q, int32_t t_amb_q,
        unsigned long long* __restrict__ cnt, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (!accountable_dev(i, solid, ts, is_vacuum, is_ambient)) continue;
        const int64_t nb = (int64_t)n_total[i];
        if (nb < 1) {   // N_EPS_RAW
            const int64_t e_amb = nb * (int64_t)t_amb_q;
            atomicAdd(&cnt[10], (unsigned long long)(gas_energy[i] - e_amb));  // e_wipe_sum
            gas_energy[i] = e_amb;
            temperature[i] = 0;
            continue;
        }
        int64_t t_rel = floordiv_q(gas_energy[i], nb) - (int64_t)t_amb_q;
        if (t_rel < (int64_t)t_min_q) {
            t_rel = (int64_t)t_min_q;
            atomicAdd(&cnt[7], 1ULL);   // energy_floor_hits
            const int64_t e_new = nb * (t_rel + (int64_t)t_amb_q);
            atomicAdd(&cnt[9], (unsigned long long)(e_new - gas_energy[i]));  // e_rail_sum
            gas_energy[i] = e_new;
        } else if (t_rel > (int64_t)t_max_phys_q) {
            t_rel = (int64_t)t_max_phys_q;
            atomicAdd(&cnt[8], 1ULL);   // t_max_phys_hits
            const int64_t e_new = nb * (t_rel + (int64_t)t_amb_q);
            atomicAdd(&cnt[9], (unsigned long long)(e_new - gas_energy[i]));  // e_rail_sum
            gas_energy[i] = e_new;
        }
        temperature[i] = (int32_t)t_rel;
    }
}

}  // namespace

// ---- S8a Path A: the per-tick scalar folds, K1's ONE transcription --------
KickScalarFolds kick_scalar_folds(
        float dt, float c_max, float dx, float adiabatic_index,
        float absorb_strength, float n_floor_solver, float u_max,
        float k_drag, float k_drag2,
        float t_amb_k) {
    KickScalarFolds f;
    f.n_floor_q    = quantize((double)n_floor_solver);
    const q16 u_max_q = quantize((double)u_max);
    f.u_max2_q32   = (int64_t)u_max_q * (int64_t)u_max_q;
    const double gamma_d = (double)adiabatic_index;
    const double dt_d    = (double)dt;
    const q16 dt_q       = quantize(dt_d);
    const double dx_d    = std::max((double)dx, 1e-6);
    f.inv_2dx_q    = quantize(1.0 / (2.0 * dx_d));
    const double K_d = (double)c_max * (double)c_max / gamma_d;
    const int64_t K_raw = (int64_t)(K_d * 65536.0 + 0.5);
    f.Kdt_raw      = mul128_shr(K_raw, (int64_t)dt_q, 16);
    f.absorb_dt_q  = quantize((double)absorb_strength * dt_d);
    f.kd_q         = quantize((double)k_drag * dt_d);
    f.kd2_q        = quantize((double)k_drag2 * dt_d);
    f.rad_dead_q32 = 0;
    if (f.kd2_q > 0) {
        const int64_t U0 = ((int64_t)FP_ONE + (int64_t)f.kd2_q - 1) / (int64_t)f.kd2_q;
        f.rad_dead_q32 = U0 * U0;
    }
    // arc #54 §2.1: k_ke = γ(γ−1)·T_AMB_K/(2·c_max²), folded as make_recip
    // (1/k_ke) — the IDENTICAL double expression eos_solver.cpp's step() and
    // the isolated reference both use.
    const double k_ke_d = gamma_d * (gamma_d - 1.0) * (double)t_amb_k
                        / (2.0 * (double)c_max * (double)c_max);
    f.k_ke_recip_q32 = make_recip(1.0 / std::max(k_ke_d, 1e-12));
    return f;
}

// ---- S8a Path A: the LAUNCH CORE — K1 only, on device pointers ------------
void kick_compression_launch_resident(
        int32_t* d_wind_x, int32_t* d_wind_y,
        int64_t* d_gas_energy,
        const int32_t* d_p_new, const int32_t* d_ntot,
        const int32_t* d_absorb_q,
        const bool* d_solid, const bool* d_is_vacuum,
        const KickScalarFolds& folds,
        const int64_t* d_cap2_plane,
        unsigned long long* d_cnt, int h, int w,
        const bool* d_is_ambient, const int32_t* d_sponge_udamp,
        const bool* d_ts) {
    const int n = h * w;
    if (n <= 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    kick_kernel<<<grid, block>>>(d_wind_x, d_wind_y, d_gas_energy, d_p_new,
                                 d_ntot, d_absorb_q, d_solid, d_is_vacuum, d_ts,
                                 folds.Kdt_raw, folds.inv_2dx_q,
                                 folds.n_floor_q, d_cap2_plane,
                                 folds.u_max2_q32,
                                 folds.kd_q, folds.kd2_q, folds.rad_dead_q32,
                                 folds.k_ke_recip_q32,
                                 d_cnt, h, w,
                                 d_is_ambient, d_sponge_udamp);
    cuda_check(cudaGetLastError(), "kick launch");
}

// ---- the flux constant's fold (arc #54 §2.1/§2.4), the ONE transcription --
EnergyFluxScalarFolds energy_flux_scalar_folds(
        float dt, float dx, float adiabatic_index, float t_amb_k, float c_value,
        float t_min, float t_max_phys, int n_sub) {
    EnergyFluxScalarFolds f;
    f.t_amb_q      = std::max<q16>(1, quantize((double)t_amb_k));
    f.t_min_q      = quantize((double)t_min);
    f.t_max_phys_q = quantize((double)t_max_phys);
    f.c_q          = quantize((double)c_value);
    f.n_sub        = std::max(1, n_sub);
    const double dt_d   = (double)dt;
    const double dx_d   = std::max((double)dx, 1e-6);
    const double dt_s_d = dt_d / (double)f.n_sub;
    const double k_work_d = ((double)adiabatic_index - 1.0) * (double)t_amb_k;
    f.k_flux_q = quantize(k_work_d * dt_s_d / (2.0 * dx_d));
    if (f.k_flux_q <= 0 || f.k_flux_q > (1 << 24)) {
        throw std::runtime_error(
            "gas-energy flux constant k_flux_q out of range — check dt / dx / "
            "adiabatic_index / T_AMB_K (design §2.4 range guard)");
    }
    const int64_t FLUX_MAG_CAP = (int64_t)1 << 60;
    f.flux_pu_cap = FLUX_MAG_CAP / (int64_t)f.k_flux_q;
    return f;
}

// ---- the LAUNCH CORE — K3 (two passes/sub-cycle) + the recovery -----------
void energy_flux_launch_resident(
        int64_t* d_gas_energy, int32_t* d_temperature,
        const int32_t* d_atmosphere,
        const int32_t* d_wind_x, const int32_t* d_wind_y,
        const int32_t* d_n_total,
        const bool* d_solid, const bool* d_is_vacuum, const bool* d_ts,
        const bool* d_is_ambient,
        const EnergyFluxScalarFolds& folds,
        int64_t* d_e0, int32_t* d_pcur, int32_t* d_s_plane,
        unsigned long long* d_cnt, int h, int w) {
    const int n = h * w;
    if (n <= 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    const int64_t t_min_abs_raw = (int64_t)folds.t_min_q + (int64_t)folds.t_amb_q;

    cuda_check(cudaMemcpy(d_e0, d_gas_energy, (size_t)n * sizeof(int64_t),
                          cudaMemcpyDeviceToDevice), "D2D e0 snapshot");

    for (int k = 0; k < folds.n_sub; ++k) {
        energy_pressure_refresh_kernel<<<grid, block>>>(
            d_pcur, d_gas_energy, d_e0, d_atmosphere, d_n_total,
            d_solid, d_ts, d_is_vacuum, d_is_ambient,
            folds.c_q, folds.t_max_phys_q, folds.t_amb_q, d_cnt, n);
        cuda_check(cudaGetLastError(), "energy pressure refresh");

        energy_flux_pass_a_kernel<<<grid, block>>>(
            d_pcur, d_wind_x, d_wind_y, d_n_total, d_gas_energy,
            d_solid, d_ts, d_is_vacuum, d_is_ambient,
            folds.k_flux_q, folds.flux_pu_cap, t_min_abs_raw,
            d_s_plane, h, w);
        cuda_check(cudaGetLastError(), "energy flux pass A");

        energy_flux_pass_b_kernel<<<grid, block>>>(
            d_pcur, d_wind_x, d_wind_y, d_n_total, d_gas_energy,
            d_solid, d_ts, d_is_vacuum, d_is_ambient,
            folds.k_flux_q, folds.flux_pu_cap,
            d_s_plane, d_cnt, h, w);
        cuda_check(cudaGetLastError(), "energy flux pass B");
    }

    energy_recovery_kernel<<<grid, block>>>(
        d_gas_energy, d_temperature, d_n_total,
        d_solid, d_ts, d_is_vacuum, d_is_ambient,
        folds.t_min_q, folds.t_max_phys_q, folds.t_amb_q, d_cnt, n);
    cuda_check(cudaGetLastError(), "energy recovery");
}

// ---- per-call entry: K1 only (the isolated gate; digest_velocity + KICK_CNT_SLOTS)
void eos_kick_compression(
    int32_t* wind_x, int32_t* wind_y, int64_t* gas_energy,
    const int32_t* p_new,
    const int32_t* gas, const bool* gas_conservative, int n_gases,
    const bool* solid, const bool* is_vacuum,
    const float* dyn_wave_absorb,
    int h, int w, float dt, const int64_t* cap2_plane,
    float c_max, float dx, float adiabatic_index, float absorb_strength,
    float n_floor_solver, float u_max,
    float k_drag, float k_drag2,
    float t_amb_k,
    uint64_t* digest_velocity_out,
    int64_t* counters_out /* [KICK_CNT_SLOTS] */,
    const bool* is_ambient, const int32_t* sponge_udamp,
    const bool* thermal_solid) {
    const int n = h * w;
    for (int c = 0; c < KICK_CNT_SLOTS; ++c) counters_out[c] = 0;
    *digest_velocity_out = 0;
    if (n <= 0 || dt <= 0.0f) return;

    const KickScalarFolds folds = kick_scalar_folds(
        dt, c_max, dx, adiabatic_index, absorb_strength, n_floor_solver,
        u_max, k_drag, k_drag2, t_amb_k);
    const q16 absorb_dt_q = folds.absorb_dt_q;

    std::vector<int32_t> n_total(n, 0);
    for (int gi = 0; gi < n_gases; ++gi) {
        if (!gas_conservative[gi]) continue;
        const int32_t* plane = gas + (size_t)gi * n;
        for (int i = 0; i < n; ++i) n_total[i] += plane[i];
    }

    std::vector<int32_t> absorb_q(n);
    for (int i = 0; i < n; ++i)
        absorb_q[i] = mul_q16(quantize((double)dyn_wave_absorb[i]), absorb_dt_q);

    const size_t nb    = (size_t)n * sizeof(int32_t);
    const size_t nb8   = (size_t)n * sizeof(int64_t);
    const size_t nbool = (size_t)n * sizeof(bool);

    int32_t *d_wx = nullptr, *d_wy = nullptr,
            *d_pn = nullptr, *d_ntot = nullptr, *d_aq = nullptr;
    int64_t* d_ge = nullptr;
    bool *d_sol = nullptr, *d_vac = nullptr;
    int64_t* d_cap2 = nullptr;
    unsigned long long* d_cnt = nullptr;

    cuda_check(cudaMalloc(&d_wx, nb), "malloc wind_x");
    cuda_check(cudaMalloc(&d_wy, nb), "malloc wind_y");
    cuda_check(cudaMalloc(&d_ge, nb8), "malloc gas_energy");
    cuda_check(cudaMalloc(&d_pn, nb), "malloc p_new");
    cuda_check(cudaMalloc(&d_ntot, nb), "malloc n_total");
    cuda_check(cudaMalloc(&d_aq, nb), "malloc absorb_q");
    cuda_check(cudaMalloc(&d_sol, nbool), "malloc solid");
    cuda_check(cudaMalloc(&d_vac, nbool), "malloc is_vacuum");
    cuda_check(cudaMalloc(&d_cap2, nb8), "malloc cap2_plane");
    cuda_check(cudaMalloc(&d_cnt, KICK_CNT_SLOTS * sizeof(unsigned long long)), "malloc counters");

    cuda_check(cudaMemcpy(d_wx, wind_x, nb, cudaMemcpyHostToDevice), "H2D wind_x");
    cuda_check(cudaMemcpy(d_wy, wind_y, nb, cudaMemcpyHostToDevice), "H2D wind_y");
    cuda_check(cudaMemcpy(d_ge, gas_energy, nb8, cudaMemcpyHostToDevice), "H2D gas_energy");
    cuda_check(cudaMemcpy(d_pn, p_new, nb, cudaMemcpyHostToDevice), "H2D p_new");
    cuda_check(cudaMemcpy(d_ntot, n_total.data(), nb, cudaMemcpyHostToDevice), "H2D n_total");
    cuda_check(cudaMemcpy(d_aq, absorb_q.data(), nb, cudaMemcpyHostToDevice), "H2D absorb_q");
    cuda_check(cudaMemcpy(d_sol, solid, nbool, cudaMemcpyHostToDevice), "H2D solid");
    cuda_check(cudaMemcpy(d_vac, is_vacuum, nbool, cudaMemcpyHostToDevice), "H2D is_vacuum");
    cuda_check(cudaMemcpy(d_cap2, cap2_plane, nb8, cudaMemcpyHostToDevice), "H2D cap2_plane");
    cuda_check(cudaMemset(d_cnt, 0, KICK_CNT_SLOTS * sizeof(unsigned long long)), "memset counters");

    bool* d_amb = nullptr;
    int32_t* d_udamp = nullptr;
    if (is_ambient) {
        cuda_check(cudaMalloc(&d_amb, nbool), "malloc is_ambient");
        cuda_check(cudaMemcpy(d_amb, is_ambient, nbool, cudaMemcpyHostToDevice), "H2D is_ambient");
    }
    if (sponge_udamp) {
        cuda_check(cudaMalloc(&d_udamp, nb), "malloc sponge_udamp");
        cuda_check(cudaMemcpy(d_udamp, sponge_udamp, nb, cudaMemcpyHostToDevice), "H2D sponge_udamp");
    }

    bool* d_tsol = nullptr;
    if (thermal_solid) {
        cuda_check(cudaMalloc(&d_tsol, nbool), "malloc thermal_solid");
        cuda_check(cudaMemcpy(d_tsol, thermal_solid, nbool,
                              cudaMemcpyHostToDevice), "H2D thermal_solid");
    }
    const bool* d_ts = thermal_solid ? d_tsol : d_sol;

    kick_compression_launch_resident(d_wx, d_wy, d_ge, d_pn, d_ntot, d_aq,
                                     d_sol, d_vac, folds, d_cap2,
                                     d_cnt, h, w, d_amb, d_udamp, d_ts);

    cuda_check(cudaDeviceSynchronize(), "sync");

    cuda_check(cudaMemcpy(wind_x, d_wx, nb, cudaMemcpyDeviceToHost), "D2H wind_x");
    cuda_check(cudaMemcpy(wind_y, d_wy, nb, cudaMemcpyDeviceToHost), "D2H wind_y");
    cuda_check(cudaMemcpy(gas_energy, d_ge, nb8, cudaMemcpyDeviceToHost), "D2H gas_energy");
    std::vector<unsigned long long> cnt_host(KICK_CNT_SLOTS, 0);
    cuda_check(cudaMemcpy(cnt_host.data(), d_cnt, KICK_CNT_SLOTS * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost), "D2H counters");

    cudaFree(d_wx);
    cudaFree(d_wy);
    cudaFree(d_ge);
    cudaFree(d_pn);
    cudaFree(d_ntot);
    cudaFree(d_aq);
    cudaFree(d_sol);
    cudaFree(d_vac);
    cudaFree(d_cap2);
    cudaFree(d_cnt);
    if (d_amb)   cudaFree(d_amb);
    if (d_udamp) cudaFree(d_udamp);
    if (d_tsol)  cudaFree(d_tsol);

    for (int c = 0; c < KICK_CNT_SLOTS; ++c) counters_out[c] = (int64_t)cnt_host[c];

    *digest_velocity_out = digest_of_host(wind_x, n, digest_of_host(wind_y, n, 0));
}

// ---- per-call entry: K3 + recovery (the isolated gate) --------------------
void eos_energy_flux(
    int64_t* gas_energy, int32_t* temperature,
    const int32_t* atmosphere,
    const int32_t* wind_x, const int32_t* wind_y,
    const int32_t* n_total,
    const bool* solid, const bool* is_vacuum,
    int h, int w, float dt, int n_sub,
    float dx, float adiabatic_index, float t_amb_k, float c_value,
    float t_min, float t_max_phys,
    int64_t* counters_out /* [FLUX_CNT_SLOTS] */,
    const bool* is_ambient,
    const bool* thermal_solid) {
    const int n = h * w;
    for (int c = 0; c < FLUX_CNT_SLOTS; ++c) counters_out[c] = 0;
    if (n <= 0 || dt <= 0.0f) return;

    const EnergyFluxScalarFolds folds = energy_flux_scalar_folds(
        dt, dx, adiabatic_index, t_amb_k, c_value, t_min, t_max_phys, n_sub);

    const size_t nb    = (size_t)n * sizeof(int32_t);
    const size_t nb8   = (size_t)n * sizeof(int64_t);
    const size_t nbool = (size_t)n * sizeof(bool);

    int32_t *d_atm = nullptr, *d_wx = nullptr, *d_wy = nullptr,
            *d_ntot = nullptr, *d_temp = nullptr;
    int64_t *d_ge = nullptr, *d_e0 = nullptr;
    int32_t *d_pcur = nullptr, *d_splane = nullptr;
    bool *d_sol = nullptr, *d_vac = nullptr;
    unsigned long long* d_cnt = nullptr;

    cuda_check(cudaMalloc(&d_atm, nb), "malloc atmosphere");
    cuda_check(cudaMalloc(&d_wx, nb), "malloc wind_x");
    cuda_check(cudaMalloc(&d_wy, nb), "malloc wind_y");
    cuda_check(cudaMalloc(&d_ntot, nb), "malloc n_total");
    cuda_check(cudaMalloc(&d_temp, nb), "malloc temperature");
    cuda_check(cudaMalloc(&d_ge, nb8), "malloc gas_energy");
    cuda_check(cudaMalloc(&d_e0, nb8), "malloc e0 scratch");
    cuda_check(cudaMalloc(&d_pcur, nb), "malloc pcur scratch");
    cuda_check(cudaMalloc(&d_splane, nb), "malloc s_plane scratch");
    cuda_check(cudaMalloc(&d_sol, nbool), "malloc solid");
    cuda_check(cudaMalloc(&d_vac, nbool), "malloc is_vacuum");
    cuda_check(cudaMalloc(&d_cnt, FLUX_CNT_SLOTS * sizeof(unsigned long long)), "malloc counters");

    cuda_check(cudaMemcpy(d_atm, atmosphere, nb, cudaMemcpyHostToDevice), "H2D atmosphere");
    cuda_check(cudaMemcpy(d_wx, wind_x, nb, cudaMemcpyHostToDevice), "H2D wind_x");
    cuda_check(cudaMemcpy(d_wy, wind_y, nb, cudaMemcpyHostToDevice), "H2D wind_y");
    cuda_check(cudaMemcpy(d_ntot, n_total, nb, cudaMemcpyHostToDevice), "H2D n_total");
    cuda_check(cudaMemcpy(d_temp, temperature, nb, cudaMemcpyHostToDevice), "H2D temperature");
    cuda_check(cudaMemcpy(d_ge, gas_energy, nb8, cudaMemcpyHostToDevice), "H2D gas_energy");
    cuda_check(cudaMemcpy(d_sol, solid, nbool, cudaMemcpyHostToDevice), "H2D solid");
    cuda_check(cudaMemcpy(d_vac, is_vacuum, nbool, cudaMemcpyHostToDevice), "H2D is_vacuum");
    cuda_check(cudaMemset(d_cnt, 0, FLUX_CNT_SLOTS * sizeof(unsigned long long)), "memset counters");

    bool* d_amb = nullptr;
    if (is_ambient) {
        cuda_check(cudaMalloc(&d_amb, nbool), "malloc is_ambient");
        cuda_check(cudaMemcpy(d_amb, is_ambient, nbool, cudaMemcpyHostToDevice), "H2D is_ambient");
    }
    bool* d_tsol = nullptr;
    if (thermal_solid) {
        cuda_check(cudaMalloc(&d_tsol, nbool), "malloc thermal_solid");
        cuda_check(cudaMemcpy(d_tsol, thermal_solid, nbool,
                              cudaMemcpyHostToDevice), "H2D thermal_solid");
    }
    const bool* d_ts = thermal_solid ? d_tsol : d_sol;

    energy_flux_launch_resident(d_ge, d_temp, d_atm, d_wx, d_wy, d_ntot,
                                d_sol, d_vac, d_ts, d_amb,
                                folds, d_e0, d_pcur, d_splane, d_cnt, h, w);

    cuda_check(cudaDeviceSynchronize(), "sync");

    cuda_check(cudaMemcpy(gas_energy, d_ge, nb8, cudaMemcpyDeviceToHost), "D2H gas_energy");
    cuda_check(cudaMemcpy(temperature, d_temp, nb, cudaMemcpyDeviceToHost), "D2H temperature");
    std::vector<unsigned long long> cnt_host(FLUX_CNT_SLOTS, 0);
    cuda_check(cudaMemcpy(cnt_host.data(), d_cnt, FLUX_CNT_SLOTS * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost), "D2H counters");

    cudaFree(d_atm); cudaFree(d_wx); cudaFree(d_wy); cudaFree(d_ntot);
    cudaFree(d_temp); cudaFree(d_ge); cudaFree(d_e0); cudaFree(d_pcur);
    cudaFree(d_splane); cudaFree(d_sol); cudaFree(d_vac); cudaFree(d_cnt);
    if (d_amb)  cudaFree(d_amb);
    if (d_tsol) cudaFree(d_tsol);

    for (int c = 0; c < FLUX_CNT_SLOTS; ++c) counters_out[c] = (int64_t)cnt_host[c];
}

namespace {
bool g_kick_compression_backend_cuda = false;
}
bool kick_compression_backend_is_cuda() { return g_kick_compression_backend_cuda; }
void set_kick_compression_backend_cuda(bool on) { g_kick_compression_backend_cuda = on; }

}  // namespace breach_cuda
