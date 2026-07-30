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
// counted min(c_LOCAL, U_MAX) scale-to-cap, the FP_HD mul_q16 / sat_add_q16
// bodies, and the exclusive 4c rail chain. Rail counters are device
// atomicAdds — pure +1 per engaging CELL (the CPU increments ONCE per cell:
// the |u| clamp inside the single magnitude test, the 4c rails in an
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

// Counter slots (the reference's counters_out[5] order).
// 0 u_clamp_hits, 1 u_max_hits, 2 work_clamp_hits, 3 energy_floor_hits,
// 4 t_max_phys_hits.

// ---- K1: the momentum kick (eos_solver.cpp step 4, verbatim) ----------------
// Pure gather: reads its OWN u, the solved-P plane (never written here),
// N_total, the hoisted absorb plane; writes ONLY its own u. Counter
// increments are per-CELL (one magnitude event), via order-free atomics.
__global__ void kick_kernel(int32_t* __restrict__ wind_x,
                            int32_t* __restrict__ wind_y,
                            const int32_t* __restrict__ p_new,
                            const int32_t* __restrict__ n_total,
                            const int32_t* __restrict__ absorb_q,   // §2.5 hoist
                            const bool* __restrict__ solid,
                            const bool* __restrict__ is_vacuum,
                            int64_t Kdt_raw, int32_t inv_2dx_q,
                            int32_t n_floor_q, int32_t c_local_q,
                            int32_t u_max_q,
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

        // |u| ≤ u_cap = min(c_LOCAL, U_MAX), counted per CELL; RAD_SAFE
        // component pre-clamp (±2^30) so rad = ux²+uy² is int64-safe and the
        // final narrow cannot wrap — the CPU chain verbatim.
        const q16 u_cap_q = (c_local_q < u_max_q) ? c_local_q : u_max_q;
        const bool cap_is_umax = (u_max_q < c_local_q);
        const int64_t RAD_SAFE = (int64_t)1 << 30;
        if      (ux >  RAD_SAFE) ux =  RAD_SAFE;
        else if (ux < -RAD_SAFE) ux = -RAD_SAFE;
        if      (uy >  RAD_SAFE) uy =  RAD_SAFE;
        else if (uy < -RAD_SAFE) uy = -RAD_SAFE;
        const int64_t ax = ux < 0 ? -ux : ux;
        const int64_t ay = uy < 0 ? -uy : uy;
        if ((ax > (int64_t)u_cap_q) || (ay > (int64_t)u_cap_q)) {
            const int64_t rad = ux * ux + uy * uy;   // int64-safe (guard above)
            const q16 umag = sqrt_q16_dev(rad);
            if (umag > u_cap_q) {
                atomicAdd(&cnt[0], 1ULL);                    // u_clamp_hits
                if (cap_is_umax) atomicAdd(&cnt[1], 1ULL);   // u_max_hits
                const q16 scale = reciprocal_q16_dev(umag);
                ux = mul128_shr_signed(mul128_shr_signed(ux, (int64_t)scale, 16),
                                       (int64_t)u_cap_q, 16);
                uy = mul128_shr_signed(mul128_shr_signed(uy, (int64_t)scale, 16),
                                       (int64_t)u_cap_q, 16);
            }
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
                                   const bool* __restrict__ solid,
                                   const bool* __restrict__ is_vacuum,
                                   int32_t inv_2dx_q, int32_t gamma_m1_q,
                                   int32_t dt_q, int32_t work_clamp_q,
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
        if (k > work_clamp_q)       { k = work_clamp_q;  atomicAdd(&cnt[2], 1ULL); }
        else if (k < -work_clamp_q) { k = -work_clamp_q; atomicAdd(&cnt[2], 1ULL); }
        const q16 dT = mul_q16(k, temperature[i]);
        q16 t_new = sat_add_q16(temperature[i], (q16)(-(int64_t)dT));
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
        float t_work_clamp, float t_max_phys, float u_max) {
    KickScalarFolds f;
    f.n_floor_q    = quantize((double)n_floor_solver);
    f.t_min_q      = quantize((double)t_min);
    f.t_max_phys_q = quantize((double)t_max_phys);
    f.u_max_q      = quantize((double)u_max);
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
    return f;
}

// ---- S8a Path A: the LAUNCH CORE — K1 then K2 on device pointers, nothing
// else (contract in cuda_resident.h). The per-call entry below wraps it.
void kick_compression_launch_resident(
        int32_t* d_wind_x, int32_t* d_wind_y, int32_t* d_temperature,
        const int32_t* d_p_new, const int32_t* d_ntot,
        const int32_t* d_absorb_q,
        const bool* d_solid, const bool* d_is_vacuum,
        const KickScalarFolds& folds, int32_t c_local_q,
        unsigned long long* d_cnt, int h, int w,
        const bool* d_is_ambient, const int32_t* d_sponge_udamp,
        const bool* d_ts) {
    const int n = h * w;
    if (n <= 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    // K1 kick, then K2 compression — same stream, so K2 sees K1's completed
    // grid-wide u (the CPU pass boundary).
    kick_kernel<<<grid, block>>>(d_wind_x, d_wind_y, d_p_new, d_ntot,
                                 d_absorb_q, d_solid, d_is_vacuum,
                                 folds.Kdt_raw, folds.inv_2dx_q,
                                 folds.n_floor_q, c_local_q,
                                 folds.u_max_q, d_cnt, h, w,
                                 d_is_ambient, d_sponge_udamp);
    cuda_check(cudaGetLastError(), "kick launch");
    compression_kernel<<<grid, block>>>(d_wind_x, d_wind_y, d_temperature,
                                        d_solid, d_is_vacuum,
                                        folds.inv_2dx_q, folds.gamma_m1_q,
                                        folds.dt_q, folds.work_clamp_q,
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
    int h, int w, float dt, int32_t c_local_q,
    float c_max, float dx, float adiabatic_index, float absorb_strength,
    float n_floor_solver, float t_min, float t_work_clamp,
    float t_max_phys, float u_max, float trace_mass_scale,
    uint64_t* digest_velocity_out, uint64_t* digest_compression_out,
    int64_t* counters_out /* [5] */,
    const bool* is_ambient, const int32_t* sponge_udamp,     // BC
    const bool* thermal_solid) {   // THERMAL-MASS AXIS, P-EOS
    const int n = h * w;
    for (int c = 0; c < 5; ++c) counters_out[c] = 0;
    *digest_velocity_out = 0;
    *digest_compression_out = 0;
    if (n <= 0 || dt <= 0.0f) return;

    // ---- Host scalar precompute — the shared ONE-transcription fold helper
    //      (S8a Path A code motion; identical expressions, identical bits). --
    const KickScalarFolds folds = kick_scalar_folds(
        dt, c_max, dx, adiabatic_index, absorb_strength, n_floor_solver,
        t_min, t_work_clamp, t_max_phys, u_max);
    const q16 absorb_dt_q = folds.absorb_dt_q;

    // ---- step 2's Dalton sum (verbatim host loop — the kick's N̂ input). ----
    std::vector<int32_t> n_total(n, 0);
    {
        const q16 tms_q = quantize((double)trace_mass_scale);
        for (int gi = 0; gi < n_gases; ++gi) {
            const int32_t* plane = gas + (size_t)gi * n;
            if (gas_conservative[gi]) {
                for (int i = 0; i < n; ++i) n_total[i] += plane[i];
            } else {
                for (int i = 0; i < n; ++i) n_total[i] += mul_q16(tms_q, plane[i]);
            }
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
    const size_t nbool = (size_t)n * sizeof(bool);

    int32_t *d_wx = nullptr, *d_wy = nullptr, *d_t = nullptr,
            *d_pn = nullptr, *d_ntot = nullptr, *d_aq = nullptr;
    bool *d_sol = nullptr, *d_vac = nullptr;
    unsigned long long* d_cnt = nullptr;

    cuda_check(cudaMalloc(&d_wx, nb), "malloc wind_x");
    cuda_check(cudaMalloc(&d_wy, nb), "malloc wind_y");
    cuda_check(cudaMalloc(&d_t,  nb), "malloc temperature");
    cuda_check(cudaMalloc(&d_pn, nb), "malloc p_new");
    cuda_check(cudaMalloc(&d_ntot, nb), "malloc n_total");
    cuda_check(cudaMalloc(&d_aq, nb), "malloc absorb_q");
    cuda_check(cudaMalloc(&d_sol, nbool), "malloc solid");
    cuda_check(cudaMalloc(&d_vac, nbool), "malloc is_vacuum");
    cuda_check(cudaMalloc(&d_cnt, 5 * sizeof(unsigned long long)), "malloc counters");

    cuda_check(cudaMemcpy(d_wx, wind_x, nb, cudaMemcpyHostToDevice), "H2D wind_x");
    cuda_check(cudaMemcpy(d_wy, wind_y, nb, cudaMemcpyHostToDevice), "H2D wind_y");
    cuda_check(cudaMemcpy(d_t, temperature, nb, cudaMemcpyHostToDevice), "H2D temperature");
    cuda_check(cudaMemcpy(d_pn, p_new, nb, cudaMemcpyHostToDevice), "H2D p_new");
    cuda_check(cudaMemcpy(d_ntot, n_total.data(), nb, cudaMemcpyHostToDevice), "H2D n_total");
    cuda_check(cudaMemcpy(d_aq, absorb_q.data(), nb, cudaMemcpyHostToDevice), "H2D absorb_q");
    cuda_check(cudaMemcpy(d_sol, solid, nbool, cudaMemcpyHostToDevice), "H2D solid");
    cuda_check(cudaMemcpy(d_vac, is_vacuum, nbool, cudaMemcpyHostToDevice), "H2D is_vacuum");
    cuda_check(cudaMemset(d_cnt, 0, 5 * sizeof(unsigned long long)), "memset counters");

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
                                     d_sol, d_vac, folds, c_local_q,
                                     d_cnt, h, w, d_amb, d_udamp, d_ts);

    cuda_check(cudaDeviceSynchronize(), "sync");

    cuda_check(cudaMemcpy(wind_x, d_wx, nb, cudaMemcpyDeviceToHost), "D2H wind_x");
    cuda_check(cudaMemcpy(wind_y, d_wy, nb, cudaMemcpyDeviceToHost), "D2H wind_y");
    cuda_check(cudaMemcpy(temperature, d_t, nb, cudaMemcpyDeviceToHost), "D2H temperature");
    unsigned long long cnt_host[5] = {0, 0, 0, 0, 0};
    cuda_check(cudaMemcpy(cnt_host, d_cnt, 5 * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost), "D2H counters");

    cudaFree(d_wx);
    cudaFree(d_wy);
    cudaFree(d_t);
    cudaFree(d_pn);
    cudaFree(d_ntot);
    cudaFree(d_aq);
    cudaFree(d_sol);
    cudaFree(d_vac);
    cudaFree(d_cnt);
    if (d_amb)   cudaFree(d_amb);
    if (d_udamp) cudaFree(d_udamp);
    if (d_tsol)  cudaFree(d_tsol);

    for (int c = 0; c < 5; ++c) counters_out[c] = (int64_t)cnt_host[c];

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
