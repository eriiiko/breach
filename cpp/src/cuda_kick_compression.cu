// ============================================================================
// EOS P6.4 — momentum kick + compression work — see cuda_kick_compression.h.
// A bit-identical GPU port of the EOS solver's post-solve tail
// (eos_solver.cpp: step 4 + step 4c, i.e. exactly what
// eos_kick_compression_reference replays on the CPU).
//
// Two kernels, launched as a barriered chain (same stream — the launch
// boundary IS the CPU's pass boundary; K2's div(u_new) reads NEIGHBOR u, so
// K1 must have completed grid-wide first):
//   K1  kick        (eos_solver.cpp step 4)  -> wind_x/wind_y in-place
//   K2  compression (eos_solver.cpp step 4c) -> temperature in-place
//
// Every arithmetic step is a VERBATIM device transcription of the CPU loop:
// the (int64_t)(Pn[ir]-Pn[il]) int32-subtract-then-widen, the staged
// mul128_shr chains (mul128_shr_signed — the same hi:lo combine as the MSVC
// host path), reciprocal_q16_dev / sqrt_q16_dev (verbatim device mirrors),
// the magnitude-first absorption shrink, the ±2^30 RAD_SAFE pre-clamp, the
// counted per-cell cap2_plane[i] scale-to-cap (VELOCITY-CLAMP, P-V1, D2v2:
// exact rad > cap² test, D6 exact int64-divide rescale — replaces the old
// global min(c_LOCAL, U_MAX) + Chebyshev pre-test), the FP_HD mul_q16 /
// sat_add_q16 bodies, and the exclusive 4c rail chain. Rail counters are
// device atomicAdds — pure +1 per engaging CELL (the CPU increments ONCE per
// cell: the |u| clamp inside the single magnitude test, the 4c rails in an
// if/else-if chain), so the totals are order-free and deterministic.
//
// Host-side precompute, verbatim step()'s (/fp:strict host pass):
//   * the scalar folds (K_raw/Kdt_raw, inv_2dx_q, absorb_dt_q, rails);
//   * step 2's Dalton N_total loop (the kick's 1/N̂ input);
//   * the §2.5 hoist: a_q[i] = mul_q16(quantize(dyn_wave_absorb[i]),
//     absorb_dt_q) — the identical per-cell expression the CPU evaluates
//     inside its loop, computed once into a per-tick plane so the kernels
//     are float-free (same double math, same rounding, per-tick-constant
//     input — the blessed P3 hoist class).
// ============================================================================
#include "cuda_kick_compression.h"
#include "cuda_resident.h" // S8a Path A: KickScalarFolds + the launch core
#include "fixed_point.h"   // q16, quantize, mul_q16, sat_add_q16, FP_ONE
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

// ---- host mul128_shr (step()'s file-local helper, for the Kdt_raw fold) -----
// MSVC-host path mirror; only used on the HOST here (the Kdt_raw scalar fold).
#if defined(__SIZEOF_INT128__)
inline int64_t mul128_shr_host(int64_t a, int64_t b, int shift) {
    return (int64_t)(((__int128)a * (__int128)b) >> shift);
}
#else
inline int64_t mul128_shr_host(int64_t a, int64_t b, int shift) {
    long long hi;
    long long lo = _mul128((long long)a, (long long)b, &hi);
    unsigned long long ulo = (unsigned long long)lo;
    return (int64_t)((ulo >> shift) | ((unsigned long long)hi << (64 - shift)));
}
#endif

// ---- solid-mirror neighbor read (eos_solver.cpp mirror_idx, verbatim) -------
__device__ __forceinline__ int mirror_idx_dev(
        int self_i, int ny, int nx, int h, int w,
        const bool* __restrict__ solid) {
    if (ny < 0 || ny >= h || nx < 0 || nx >= w) return self_i;
    const int ni = ny * w + nx;
    if (solid[ni]) return self_i;
    return ni;
}

// Counter slots (the reference's counters_out[9] order).
// 0 u_clamp_hits, 1 u_max_hits, 2 work_clamp_hits, 3 energy_floor_hits,
// 4 t_max_phys_hits, 5 ke_drag_removed, 6 e_drag_deposit, 7 e_drag_drop_sum,
// 8 e_drag_rail_clipped (P-E3, design §2.8 — slots 5-8 are int64 ENERGY
// SUMS via atomicAdd on two's-complement, not hit counts, all non-negative
// by construction).

// ---- K1: the momentum kick (eos_solver.cpp step 4, verbatim) ----------------
// Pure gather: reads its OWN u, the solved-P plane (never written here),
// N_total, the hoisted absorb plane; writes ITS OWN u AND (P-E3) its OWN T —
// own-cell T write is race-free (K2 reads only NEIGHBOUR u, never T here;
// the launch boundary between K1 and K2 is the CPU's pass boundary). Counter
// increments are per-CELL (one magnitude event) or int64 energy sums, both
// via order-free atomics.
__global__ void kick_kernel(int32_t* __restrict__ wind_x,
                            int32_t* __restrict__ wind_y,
                            int32_t* __restrict__ temperature,   // P-E3: own-cell T
                            const int32_t* __restrict__ p_new,
                            const int32_t* __restrict__ n_total,
                            const int32_t* __restrict__ absorb_q,   // §2.5 hoist
                            const bool* __restrict__ solid,
                            const bool* __restrict__ is_vacuum,
                            const bool* __restrict__ ts,             // P-E3
                            int64_t Kdt_raw, int32_t inv_2dx_q,
                            int32_t n_floor_q,
                            const int64_t* __restrict__ cap2_plane,  // D2v2, >= 0
                            int64_t u_max2_q32,                      // D3
                            int32_t kd_q, int32_t heat_frac_q,       // P-E3
                            int64_t recip_cv, int32_t t_max_phys_q,  // P-E3
                            unsigned long long* __restrict__ cnt,
                            int h, int w,
                            const bool* __restrict__ is_ambient,
                            const int32_t* __restrict__ sponge_udamp) {
    const int n = h * w;
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
        // |∇P| staged 128-bit exactly as the CPU: int32 subtract, widen, shift.
        const int64_t gx = mul128_shr_signed((int64_t)(p_new[ir] - p_new[il]),
                                             (int64_t)inv_2dx_q, 16);
        const int64_t gy = mul128_shr_signed((int64_t)(p_new[id] - p_new[iu]),
                                             (int64_t)inv_2dx_q, 16);
        int64_t ux = (int64_t)wind_x[i];
        int64_t uy = (int64_t)wind_y[i];
        if (gx != 0 || gy != 0) {   // micro-opt kept: du == 0 exactly at zero
                                    // gradient — skip the reciprocal chain
                                    // (bit-identical, the CPU's own branch)
            q16 nhat = n_total[i];
            if (nhat < n_floor_q) nhat = n_floor_q;
            const q16 inv_n = reciprocal_q16_dev(nhat);
            // du = (K·dt)·∇P·(1/N̂) — staged 128-bit, the documented order.
            ux -= mul128_shr_signed(mul128_shr_signed(Kdt_raw, gx, 16),
                                    (int64_t)inv_n, 16);
            uy -= mul128_shr_signed(mul128_shr_signed(Kdt_raw, gy, 16),
                                    (int64_t)inv_n, 16);
        }

        // absorption damping u *= (1 − absorb·dt) on the wide value,
        // magnitude-first (the CPU's sign-symmetric shrink). `a` is the
        // host-hoisted mul_q16(quantize(absorb[i]), absorb_dt_q) plane.
        const q16 a = absorb_q[i];
        if (a > 0) {
            const q16 kk = (a < FP_ONE) ? (q16)(FP_ONE - a) : 0;
            const int64_t mx = mul128_shr_signed(ux < 0 ? -ux : ux, (int64_t)kk, 16);
            const int64_t my = mul128_shr_signed(uy < 0 ? -uy : uy, (int64_t)kk, 16);
            ux = (ux < 0) ? -mx : mx;
            uy = (uy < 0) ? -my : my;
        }

        // BC (spec §3 rung 2): the u-DAMPING BAND — |u| *= (1 − k(d)),
        // magnitude-first, immediately after the absorb chain (the CPU's exact
        // placement/ordering; mul128_shr_signed == the host mul128_shr). Gated
        // on ambient mode, no-op at k==0 -> space maps byte-identical.
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

        // |u| <= per-cell cap2_plane[i] (VELOCITY-CLAMP, P-V1, D2v2/D5),
        // counted per CELL; RAD_SAFE component pre-clamp (±2^30) so
        // rad = ux²+uy² is int64-safe and the final narrow cannot wrap —
        // the CPU chain verbatim. EXACT rad > cap² test (D3/audit defect 2:
        // no component Chebyshev pre-test — that let diagonal flow up to
        // √2×cap through; the exact test closes it, only a CLAMPED cell
        // pays the sqrt below).
        const int64_t cap2_q32 = cap2_plane[i];   // D5: trusted verbatim
        const bool cap_is_umax = (cap2_q32 >= u_max2_q32);
        const int64_t RAD_SAFE = (int64_t)1 << 30;
        if      (ux >  RAD_SAFE) ux =  RAD_SAFE;
        else if (ux < -RAD_SAFE) ux = -RAD_SAFE;
        if      (uy >  RAD_SAFE) uy =  RAD_SAFE;
        else if (uy < -RAD_SAFE) uy = -RAD_SAFE;
        const int64_t rad = ux * ux + uy * uy;   // int64-safe (guard above)
        if (rad > cap2_q32) {
            atomicAdd(&cnt[0], 1ULL);                    // u_clamp_hits
            if (cap_is_umax) atomicAdd(&cnt[1], 1ULL);   // u_max_hits
            const q16 umag    = sqrt_q16_dev(rad);
            const q16 u_cap_q = sqrt_q16_dev(cap2_q32);
            // D6 exact rescale: trunc-toward-0 integer divide, C++/CUDA
            // bit-identical (both truncate toward zero); int64-safe
            // unconditionally (sqrt_q16_dev self-clamps at INT32_MAX, so
            // |ux*u_cap_q| < 2^30*2^31 = 2^61).
            ux = (ux * (int64_t)u_cap_q) / (int64_t)umag;
            uy = (uy * (int64_t)u_cap_q) / (int64_t)umag;
        }

        // P-E3 — interior drag + heat counterparty (design §2.8), VERBATIM
        // device transcription of eos_solver.cpp's kick-loop insertion: PER
        // TICK, after the |u| cap, before the store; ts cells skip both the
        // drag and the deposit (ruling A1). Dormancy BY BRANCH on kd_q.
        if (kd_q > 0 && !(ts && ts[i])) {
            const int64_t ux_old = ux, uy_old = uy;
            const q16 kk_drag = (kd_q < FP_ONE) ? (q16)(FP_ONE - kd_q) : 0;
            const int64_t dmx = mul128_shr_signed(ux_old < 0 ? -ux_old : ux_old, (int64_t)kk_drag, 16);
            const int64_t dmy = mul128_shr_signed(uy_old < 0 ? -uy_old : uy_old, (int64_t)kk_drag, 16);
            ux = (ux_old < 0) ? -dmx : dmx;
            uy = (uy_old < 0) ? -dmy : dmy;

            const int64_t du2_raw = (ux_old * ux_old + uy_old * uy_old)
                                   - (ux * ux + uy * uy);   // >= 0 structurally
            const int64_t n_bulk = (int64_t)n_total[i];
            atomicAdd(&cnt[5], (unsigned long long)mul128_shr_signed(n_bulk, du2_raw, 16));

            const int64_t dE_cell_q16 = (du2_raw >> 16) >> 1;
            const int64_t dT_intended_wide =
                drag_dT_wide_q16_dev(dE_cell_q16, heat_frac_q, recip_cv);
            const int32_t drop_frac_q = (int32_t)(FP_ONE - heat_frac_q);
            const int64_t dT_drop_wide =
                drag_dT_wide_q16_dev(dE_cell_q16, drop_frac_q, recip_cv);
            atomicAdd(&cnt[7], (unsigned long long)mul128_shr_signed(n_bulk, dT_drop_wide, 0));

            const int32_t dT_intended_narrow =
                (dT_intended_wide > (int64_t)INT32_MAX)
                    ? INT32_MAX : (int32_t)dT_intended_wide;
            const int32_t t_old = temperature[i];
            int32_t t_candidate = sat_add_q16(t_old, dT_intended_narrow);
            if (t_candidate > t_max_phys_q) t_candidate = t_max_phys_q;
            const int64_t dT_applied = (int64_t)t_candidate - (int64_t)t_old;
            const int64_t dT_clipped = dT_intended_wide - dT_applied;
            atomicAdd(&cnt[6], (unsigned long long)mul128_shr_signed(n_bulk, dT_applied, 0));
            atomicAdd(&cnt[8], (unsigned long long)mul128_shr_signed(n_bulk, dT_clipped, 0));

            if (n_bulk >= 1) temperature[i] = t_candidate;
        }

        wind_x[i] = (int32_t)ux;   // the ONE narrow at store (CPU comment: safe
        wind_y[i] = (int32_t)uy;   // by construction — caps ≪ int32 range)
    }
}

// ---- K2: compression work (eos_solver.cpp step 4c, verbatim) ----------------
// Pure gather: reads NEIGHBOR u (frozen — K1 completed at the kernel
// boundary) + its OWN T; writes ONLY its own T. The 4c rails are the CPU's
// exclusive if/else-if chain, counted per CELL via order-free atomics.
__global__ void compression_kernel(const int32_t* __restrict__ wind_x,
                                   const int32_t* __restrict__ wind_y,
                                   int32_t* __restrict__ temperature,
                                   const int32_t* __restrict__ n_total,   // P-E4
                                   const bool* __restrict__ solid,
                                   const bool* __restrict__ is_vacuum,
                                   int32_t inv_2dx_q, int32_t gamma_m1_q,
                                   int32_t dt_q, int32_t work_clamp_q,
                                   int64_t recip_n_work_ref,   // P-E4
                                   int32_t t_min_q, int32_t t_max_phys_q,
                                   unsigned long long* __restrict__ cnt,
                                   int h, int w,
                                   const bool* __restrict__ is_ambient,
                                   const bool* __restrict__ ts) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        // BC: the ring is skipped like vacuum — no compression work.
        // THERMAL-MASS AXIS, P-EOS, T-WRITE SITE 2/2 (ruling A1) — the CPU guard
        // verbatim: compression work is work done ON GAS BY COMPRESSION, and an
        // OBJECT does not compress, so the EOS may not touch a thermal_solid
        // tile's temperature. `ts` is d_solid on the legacy path (the P2 device
        // fallback idiom), where the added term is redundant.
        if (solid[i] || (ts && ts[i]) || is_vacuum[i]
                || (is_ambient && is_ambient[i])) continue;
        const int y = i / w;
        const int x = i % w;
        const int il = mirror_idx_dev(i, y, x - 1, h, w, solid);
        const int ir = mirror_idx_dev(i, y, x + 1, h, w, solid);
        const int iu = mirror_idx_dev(i, y - 1, x, h, w, solid);
        const int id = mirror_idx_dev(i, y + 1, x, h, w, solid);
        const q16 dux = mul_q16(wind_x[ir] - wind_x[il], inv_2dx_q);
        const q16 duy = mul_q16(wind_y[id] - wind_y[iu], inv_2dx_q);
        const q16 div_new = dux + duy;
        q16 k = mul_q16(gamma_m1_q, div_new);
        k = mul_q16(k, dt_q);
        // P-E4 TRUST GATE (design §2.4), VERBATIM device transcription of
        // the CPU's block: fade k toward 0 for thin/untrustworthy N,
        // magnitude-first (scale_mag) so a negative k fades TOWARD zero,
        // never past it. recip_mul_dev is the device 128-bit reciprocal
        // multiply (recip_mul's device twin); work_fade_clamp01_q is the
        // shared FP_HD clamp01 tail (no 128-bit ops, identical both sides).
        {
            const q16 ratio = recip_mul_dev(n_total[i], recip_n_work_ref);
            const q16 fade = work_fade_clamp01_q(ratio);
            k = scale_mag(k, fade);
        }
        // P-E4 REVERSIBLE WORK (design §2.7), VERBATIM device transcription:
        // magnitude-first clamp, single-compare form (pinned) — identical
        // hit semantics to the old signed if/else-if pair.
        const bool k_neg = (k < 0);
        q16 w_mag = k_neg ? (q16)(-(int64_t)k) : k;
        if (w_mag > work_clamp_q) { w_mag = work_clamp_q; atomicAdd(&cnt[2], 1ULL); }
        q16 t_new;
        if (k_neg) {
            // COMPRESSION — KEPT VERBATIM (design §2.7): bit-identical to HEAD.
            const q16 k_signed = (q16)(-(int64_t)w_mag);
            const q16 dT = mul_q16(k_signed, temperature[i]);
            t_new = sat_add_q16(temperature[i], (q16)(-(int64_t)dT));
        } else {
            // EXPANSION (k >= 0, including the pinned k==0 identity): the
            // reversible inverse via the shared floordiv_q helper (P-E1's
            // recovery divide) — plain `/` would MINT on sub-ambient T.
            t_new = (q16)floordiv_q((int64_t)temperature[i] << 16,
                                    (int64_t)FP_ONE + (int64_t)w_mag);
        }
        if (t_new < t_min_q) { t_new = t_min_q; atomicAdd(&cnt[3], 1ULL); }
        else if (t_new > t_max_phys_q) { t_new = t_max_phys_q; atomicAdd(&cnt[4], 1ULL); }
        temperature[i] = t_new;
    }
}

}  // namespace

// ---- S8a Path A: the per-tick scalar folds, factored to ONE transcription
// (cuda_resident.h — design §3.2.3). VERBATIM the fold block that lived at
// the top of eos_kick_compression (pure code motion; /fp:strict host pass).
KickScalarFolds kick_scalar_folds(
        float dt, float c_max, float dx, float adiabatic_index,
        float absorb_strength, float n_floor_solver, float t_min,
        float t_work_clamp, float t_max_phys, float u_max,
        float k_drag, float k_drag_heat_frac, float c_v, float n_work_ref) {
    KickScalarFolds f;
    f.n_floor_q    = quantize((double)n_floor_solver);
    f.t_min_q      = quantize((double)t_min);
    f.t_max_phys_q = quantize((double)t_max_phys);
    f.u_max_q      = quantize((double)u_max);
    // VELOCITY-CLAMP (P-V1, D3): u_max² (Q32.32), the SAME fold every kick
    // site derives from u_max_q.
    f.u_max2_q32   = (int64_t)f.u_max_q * (int64_t)f.u_max_q;
    const double gamma_d = (double)adiabatic_index;
    f.gamma_m1_q   = quantize(gamma_d - 1.0);
    const double dt_d    = (double)dt;
    f.dt_q         = quantize(dt_d);
    const double dx_d    = std::max((double)dx, 1e-6);
    f.inv_2dx_q    = quantize(1.0 / (2.0 * dx_d));
    const double K_d = (double)c_max * (double)c_max / gamma_d;
    const int64_t K_raw = (int64_t)(K_d * 65536.0 + 0.5);
    f.Kdt_raw      = mul128_shr_host(K_raw, (int64_t)f.dt_q, 16);
    f.absorb_dt_q  = quantize((double)absorb_strength * dt_d);
    f.work_clamp_q = quantize((double)t_work_clamp);
    // P-E3 (design §2.8): the drag folds, the SAME per-tick-not-per-cell
    // idiom absorb_dt_q already uses; make_recip is the host double-divide
    // the CPU reference's own recip_cv fold uses.
    f.kd_q         = quantize((double)k_drag * dt_d);
    f.heat_frac_q  = quantize((double)k_drag_heat_frac);
    f.recip_cv     = make_recip(std::max((double)c_v, 1e-6));
    // P-E4 (design §2.4): the trust-gate fold, verbatim step()'s.
    f.recip_n_work_ref = make_recip(std::max((double)n_work_ref, 1e-6));
    return f;
}

// ---- S8a Path A: the LAUNCH CORE — K1 then K2 on device pointers, nothing
// else (contract in cuda_resident.h). The per-call entry below wraps it.
void kick_compression_launch_resident(
        int32_t* d_wind_x, int32_t* d_wind_y, int32_t* d_temperature,
        const int32_t* d_p_new, const int32_t* d_ntot,
        const int32_t* d_absorb_q,
        const bool* d_solid, const bool* d_is_vacuum,
        const KickScalarFolds& folds,
        const int64_t* d_cap2_plane,   // VELOCITY-CLAMP (P-V1, D2v2), >= 0
        unsigned long long* d_cnt, int h, int w,
        const bool* d_is_ambient, const int32_t* d_sponge_udamp,
        const bool* d_ts) {
    const int n = h * w;
    if (n <= 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    // K1 kick, then K2 compression — same stream, so K2 sees K1's completed
    // grid-wide u (the CPU pass boundary). P-E3: K1 also takes temperature +
    // ts now (own-cell T write from K1 is race-free — K2 reads only
    // neighbour u, never T, at this point in the tick).
    kick_kernel<<<grid, block>>>(d_wind_x, d_wind_y, d_temperature, d_p_new,
                                 d_ntot, d_absorb_q, d_solid, d_is_vacuum, d_ts,
                                 folds.Kdt_raw, folds.inv_2dx_q,
                                 folds.n_floor_q, d_cap2_plane,
                                 folds.u_max2_q32,
                                 folds.kd_q, folds.heat_frac_q,
                                 folds.recip_cv, folds.t_max_phys_q,
                                 d_cnt, h, w,
                                 d_is_ambient, d_sponge_udamp);
    cuda_check(cudaGetLastError(), "kick launch");
    compression_kernel<<<grid, block>>>(d_wind_x, d_wind_y, d_temperature,
                                        d_ntot,   // P-E4: the trust-gate input
                                        d_solid, d_is_vacuum,
                                        folds.inv_2dx_q, folds.gamma_m1_q,
                                        folds.dt_q, folds.work_clamp_q,
                                        folds.recip_n_work_ref,   // P-E4
                                        folds.t_min_q, folds.t_max_phys_q,
                                        d_cnt, h, w, d_is_ambient, d_ts);
    cuda_check(cudaGetLastError(), "compression launch");
}

void eos_kick_compression(
    int32_t* wind_x, int32_t* wind_y, int32_t* temperature,
    const int32_t* p_new,
    const int32_t* gas, const bool* gas_conservative, int n_gases,
    const bool* solid, const bool* is_vacuum,
    const float* dyn_wave_absorb,
    int h, int w, float dt, const int64_t* cap2_plane,   // D2v2 (h,w), >= 0
    float c_max, float dx, float adiabatic_index, float absorb_strength,
    float n_floor_solver, float t_min, float t_work_clamp,
    float t_max_phys, float u_max,   // trace_mass_scale param RETIRED (P-T0)
    float k_drag, float k_drag_heat_frac, float c_v,   // P-E3 (design §2.8)
    float n_work_ref,   // P-E4 (design §2.4): the compression-work trust gate
    uint64_t* digest_velocity_out, uint64_t* digest_compression_out,
    int64_t* counters_out /* [9] */,
    const bool* is_ambient, const int32_t* sponge_udamp,     // BC
    const bool* thermal_solid) {   // THERMAL-MASS AXIS, P-EOS
    const int n = h * w;
    for (int c = 0; c < 9; ++c) counters_out[c] = 0;
    *digest_velocity_out = 0;
    *digest_compression_out = 0;
    if (n <= 0 || dt <= 0.0f) return;

    // ---- Host scalar precompute — the shared ONE-transcription fold helper
    //      (S8a Path A code motion; identical expressions, identical bits). --
    const KickScalarFolds folds = kick_scalar_folds(
        dt, c_max, dx, adiabatic_index, absorb_strength, n_floor_solver,
        t_min, t_work_clamp, t_max_phys, u_max,
        k_drag, k_drag_heat_frac, c_v, n_work_ref);
    const q16 absorb_dt_q = folds.absorb_dt_q;

    // ---- step 2's Dalton sum (verbatim host loop — the kick's N̂ input). ----
    // P-T0 (design §2.6): n_total ≡ n_bulk; trace planes skipped outright.
    std::vector<int32_t> n_total(n, 0);
    {
        for (int gi = 0; gi < n_gases; ++gi) {
            if (!gas_conservative[gi]) continue;
            const int32_t* plane = gas + (size_t)gi * n;
            for (int i = 0; i < n; ++i) n_total[i] += plane[i];
        }
    }

    // ---- §2.5 hoist: the per-cell absorption factor, folded once per tick.
    //      a_q[i] == the CPU's in-loop mul_q16(quantize(absorb[i]),
    //      absorb_dt_q) — identical expression, identical inputs. ------------
    std::vector<int32_t> absorb_q(n);
    for (int i = 0; i < n; ++i)
        absorb_q[i] = mul_q16(quantize((double)dyn_wave_absorb[i]), absorb_dt_q);

    // ---- Device buffers (per-call H2D/D2H — the P6.1/P6.2 pattern). --------
    const size_t nb    = (size_t)n * sizeof(int32_t);
    const size_t nb8   = (size_t)n * sizeof(int64_t);
    const size_t nbool = (size_t)n * sizeof(bool);

    int32_t *d_wx = nullptr, *d_wy = nullptr, *d_t = nullptr,
            *d_pn = nullptr, *d_ntot = nullptr, *d_aq = nullptr;
    bool *d_sol = nullptr, *d_vac = nullptr;
    int64_t* d_cap2 = nullptr;   // VELOCITY-CLAMP (P-V1, D2v2)
    unsigned long long* d_cnt = nullptr;

    cuda_check(cudaMalloc(&d_wx, nb), "malloc wind_x");
    cuda_check(cudaMalloc(&d_wy, nb), "malloc wind_y");
    cuda_check(cudaMalloc(&d_t,  nb), "malloc temperature");
    cuda_check(cudaMalloc(&d_pn, nb), "malloc p_new");
    cuda_check(cudaMalloc(&d_ntot, nb), "malloc n_total");
    cuda_check(cudaMalloc(&d_aq, nb), "malloc absorb_q");
    cuda_check(cudaMalloc(&d_sol, nbool), "malloc solid");
    cuda_check(cudaMalloc(&d_vac, nbool), "malloc is_vacuum");
    cuda_check(cudaMalloc(&d_cap2, nb8), "malloc cap2_plane");
    cuda_check(cudaMalloc(&d_cnt, 9 * sizeof(unsigned long long)), "malloc counters");

    cuda_check(cudaMemcpy(d_wx, wind_x, nb, cudaMemcpyHostToDevice), "H2D wind_x");
    cuda_check(cudaMemcpy(d_wy, wind_y, nb, cudaMemcpyHostToDevice), "H2D wind_y");
    cuda_check(cudaMemcpy(d_t, temperature, nb, cudaMemcpyHostToDevice), "H2D temperature");
    cuda_check(cudaMemcpy(d_pn, p_new, nb, cudaMemcpyHostToDevice), "H2D p_new");
    cuda_check(cudaMemcpy(d_ntot, n_total.data(), nb, cudaMemcpyHostToDevice), "H2D n_total");
    cuda_check(cudaMemcpy(d_aq, absorb_q.data(), nb, cudaMemcpyHostToDevice), "H2D absorb_q");
    cuda_check(cudaMemcpy(d_sol, solid, nbool, cudaMemcpyHostToDevice), "H2D solid");
    cuda_check(cudaMemcpy(d_vac, is_vacuum, nbool, cudaMemcpyHostToDevice), "H2D is_vacuum");
    cuda_check(cudaMemcpy(d_cap2, cap2_plane, nb8, cudaMemcpyHostToDevice), "H2D cap2_plane");
    cuda_check(cudaMemset(d_cnt, 0, 9 * sizeof(unsigned long long)), "memset counters");

    // BC (spec §1/§3): optional ambient ring mask + u-damping band grid. Only
    // allocated/uploaded on an ambient map; nullptr on space -> the kernels take
    // the byte-identical space path (the `is_ambient && ...` short-circuits).
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

    // THERMAL-MASS AXIS, P-EOS: the medium mask K2 skips its T write on. The P2
    // device-fallback idiom — d_ts = thermal_solid ? d_tsol : d_sol — so the
    // legacy (nullptr) path allocates and copies NOTHING and is not a second
    // code path.
    bool* d_tsol = nullptr;
    if (thermal_solid) {
        cuda_check(cudaMalloc(&d_tsol, nbool), "malloc thermal_solid");
        cuda_check(cudaMemcpy(d_tsol, thermal_solid, nbool,
                              cudaMemcpyHostToDevice), "H2D thermal_solid");
    }
    const bool* d_ts = thermal_solid ? d_tsol : d_sol;

    // The SHARED launch core (S8a Path A) — the identical K1/K2 launch pair
    // this entry always ran; only the call shape moved.
    kick_compression_launch_resident(d_wx, d_wy, d_t, d_pn, d_ntot, d_aq,
                                     d_sol, d_vac, folds, d_cap2,
                                     d_cnt, h, w, d_amb, d_udamp, d_ts);

    cuda_check(cudaDeviceSynchronize(), "sync");

    cuda_check(cudaMemcpy(wind_x, d_wx, nb, cudaMemcpyDeviceToHost), "D2H wind_x");
    cuda_check(cudaMemcpy(wind_y, d_wy, nb, cudaMemcpyDeviceToHost), "D2H wind_y");
    cuda_check(cudaMemcpy(temperature, d_t, nb, cudaMemcpyDeviceToHost), "D2H temperature");
    unsigned long long cnt_host[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    cuda_check(cudaMemcpy(cnt_host, d_cnt, 9 * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost), "D2H counters");

    cudaFree(d_wx);
    cudaFree(d_wy);
    cudaFree(d_t);
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

    for (int c = 0; c < 9; ++c) counters_out[c] = (int64_t)cnt_host[c];

    // Host-side digests, byte-for-byte step()'s expressions:
    //   digest_velocity    = digest_of(wx, digest_of(wy, 0))
    //   digest_compression = digest_of(T, 0)
    *digest_velocity_out = digest_of_host(wind_x, n, digest_of_host(wind_y, n, 0));
    *digest_compression_out = digest_of_host(temperature, n, 0);
}

namespace {
bool g_kick_compression_backend_cuda = false;
}
bool kick_compression_backend_is_cuda() { return g_kick_compression_backend_cuda; }
void set_kick_compression_backend_cuda(bool on) { g_kick_compression_backend_cuda = on; }

}  // namespace breach_cuda
