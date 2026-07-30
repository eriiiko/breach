// ============================================================================
// CUDA temperature solver implementation — see cuda_temperature.h.
// A bit-identical GPU port of TemperatureSolver::step (temperature_solver.cpp).
//
// EOS P6.6 (docs/eos_p6_gpu_alignment_review.md §4): extended from the S1
// solid-only convert/conduct/cool to the FULL unified-temperature step — the
// Pass 0 gas-T zero-at-vacuum + semi-Lagrangian advection, the Pass 1 open-air
// v2.4 absorption-∝-density radiant deposit (n_bulk divisor, T_MAX_PHYS rail),
// on top of the already-mirrored Pass 2 conduction + Pass 3 cooling. Every pass
// is a per-cell / gather single-writer kernel over frozen inputs (each cell reads
// neighbours/snapshots, writes only its own T), so the GPU result is byte-for-byte
// identical to the CPU on every architecture. The only read-after-write ordering
// the CPU relies on is at the PASS BOUNDARIES (zero-vacuum -> snapshot -> advect
// -> convert -> conduct -> cool), each reproduced by a separate kernel launch
// (a global barrier); no cell ever reads another cell's same-pass write.
//
// THERMAL-MASS AXIS, P2 (2026-07-30 — docs/thermal_mass_axis_design_2026-07-25.md
// + docs/thermal_mass_axis_build_addendum_2026-07-30.md §3): the GPU mirror of
// P1. Every per-medium branch keys on the THERMAL mask `thermal_solid`
// (`thermal_mass > 0`, GameMap.thermal_solid), NOT on the FLOW mask `solid`
// (`permeability <= 0`) — because furniture (permeability 0.5, the deliberate
// "shield but not seal" soft body) is permeable AND a thermal solid, and keying
// the medium on flow put a burning crate's object temperature into the GAS
// regime where the fire's own plume advected it away. EXACTLY the same SIX sites
// the CPU marks "MEDIUM-TEST SITE n/6" are swapped here, marked identically; the
// mapping is one-to-one so the two files stay readable side by side:
//   1/6 temp_zero_vacuum's `!ts[i]` guard      (CPU temperature_solver.cpp Pass 0a)
//   2/6 temp_advect's open-air skip            (CPU Pass 0b)
//   3/6 gas_wall_at (ray-walk occluder)        (CPU gas_wall_at)
//   4/6 the bilinear gather's sealed corner    (CPU gas_backtrace_sample_q)
//   5/6 temp_convert_unified's medium branch   (CPU Pass 1)
//   6/6 temp_cool's COOL_SHIFT decay guard     (CPU Pass 3)
// `solid` is NOT otherwise read by this TU any more: it survives only as the
// documented nullptr fallback for `thermal_solid`. Conduction (temp_conduct) is
// κ-keyed via face_shift and is deliberately NOT one of the six (design §2.2),
// so furniture (conductivity 0) has COOL_SHIFT as its ONE loss channel. On any
// furniture-free map `thermal_solid == solid` elementwise, so this is
// byte-identical there (addendum D4) — the patch's gate (a).
//
// COOL-SHIFT AXIS (2026-07-30): the LOSS-side twin of thermal_mass. MEDIUM-TEST
// SITE 6/6 (temp_cool) additionally takes a per-tile decay shift
// (`cool_shift_grid`, GameMap.cool_shift) instead of the single global
// COOL_SHIFT, because the thermal-mass arc made furniture a thermal solid whose
// ONLY loss channel is that decay — and 2^5/24 == 1.3 s is right for thin hull
// plate and absurd for a wooden crate. The vacuum-exposed rate stays ONE global
// rule applied as an OFFSET (cool_shift - cool_shift_vacuum, floored at
// cool_shift_floor == SHIFT_MIN), so each material keeps exactly one dial. With
// every material seeded at the old global this is bit-identical to the pre-axis
// kernel; the CPU twin is temperature_solver.cpp Pass 3, line for line.
// ============================================================================
#include "cuda_temperature.h"
#include "fixed_point.h"              // quantize/make_recip/mul_q16/mul_wide/narrow
#include "cuda_fixedpoint_device.cuh" // heat_saturating_add_dev, reciprocal_q16_dev,
                                       // recip_mul_dev

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace breach_cuda {

namespace {

using namespace fixedpoint;

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in temperature_step/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// Direction order MUST match temperature_solver.cpp: 0=N,1=S,2=E,3=W.
__device__ __forceinline__ int dy_of(int d) {
    // {-1, +1, 0, 0}
    return (d == 0) ? -1 : (d == 1) ? 1 : 0;
}
__device__ __forceinline__ int dx_of(int d) {
    // {0, 0, +1, -1}
    return (d == 2) ? 1 : (d == 3) ? -1 : 0;
}

// ---- Pass 0a: gas-T zero at OPEN (non-thermal-solid) vacuum cells (§4) ------
// The structural invariant, UNCONDITIONAL (runs whether or not advection does):
// a true breach (is_vacuum && !thermal_solid) holds no gas, so no gas-T — energy
// leaves with the venting gas. The `!thermal_solid` guard is load-bearing: a
// space-exposed hull tile (vacuum AND solid) keeps its real solid-thermal state.
// MEDIUM-TEST SITE 1/6: that guard is now the THERMAL medium, so a space-exposed
// CRATE keeps its object temperature for exactly the same reason a hull tile
// does; the hull case is unchanged (hull is both solid and thermal_solid).
// Per-cell, no race.
__global__ void temp_zero_vacuum(int32_t* __restrict__ temperature,
                                 const bool* __restrict__ thermal_solid,
                                 const bool* __restrict__ is_vacuum, int n,
                                 const bool* __restrict__ is_ambient) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        // BC (audit (b)): the ambient ring radiates to the T_amb sky — wiped to
        // ΔT=0 exactly like a vacuum breach (is_ambient nullptr on space maps).
        if ((is_vacuum[i] || (is_ambient && is_ambient[i])) && !thermal_solid[i])
            temperature[i] = 0;
    }
}

// ---- Pass 0b advection sampler — VERBATIM port of gas_backtrace_sample_q -----
// (temperature_solver.cpp): integer DDA wall-clip march + integer bilinear
// sample + Newton-reciprocal renorm, specialized to thermal_solid/is_vacuum.
// reciprocal_q16 -> reciprocal_q16_dev (bit-identical); mul_q16/mul_wide/narrow
// are FP_HD device-clean. Every arithmetic step matches the CPU by construction.
// MEDIUM-TEST SITE 3/6 (the ray-walk occluder): gas-T no longer advects ACROSS a
// crate tile, because a crate holds an OBJECT temperature (design §2.3).
__device__ __forceinline__ bool gas_wall_at(int y, int x,
                                             const bool* thermal_solid,
                                             int h, int w) {
    if (y < 0 || y >= h || x < 0 || x >= w) return true;   // outside == wall
    return thermal_solid[y * w + x];
}

__device__ int32_t gas_backtrace_sample_q_dev(
        const int32_t* src, int x, int y, int32_t bx_q, int32_t by_q,
        const bool* thermal_solid, const bool* is_vacuum, int h, int w) {
    const int32_t GAS_WSUM_FLOOR_Q = FP_ONE >> 8;
    const int32_t GAS_WSUM_EPS_Q   = FP_ONE >> 14;

    int64_t px_q = ((int64_t)x << FP_SHIFT) + bx_q;
    int64_t py_q = ((int64_t)y << FP_SHIFT) + by_q;

    // ---- Wall-clip march (DDA, no sqrt) ----
    const int32_t abx = bx_q >= 0 ? bx_q : -bx_q;
    const int32_t aby = by_q >= 0 ? by_q : -by_q;
    const int32_t amax = abx >= aby ? abx : aby;
    int n_steps = amax >> FP_SHIFT;
    if (amax & (FP_ONE - 1)) n_steps += 1;                   // ceil
    if (n_steps > 0) {
        // floordiv(a, b) with a int32, b > 0 (matches the CPU lambda exactly).
        const int32_t sx_q = (bx_q >= 0) ? (bx_q / n_steps)
                             : -(int32_t)(((-(int64_t)bx_q) + n_steps - 1) / n_steps);
        const int32_t sy_q = (by_q >= 0) ? (by_q / n_steps)
                             : -(int32_t)(((-(int64_t)by_q) + n_steps - 1) / n_steps);
        int64_t cx_q = (int64_t)x << FP_SHIFT;
        int64_t cy_q = (int64_t)y << FP_SHIFT;
        for (int s = 0; s < n_steps; ++s) {
            const int64_t nxp_q = cx_q + sx_q;
            const int64_t nyp_q = cy_q + sy_q;
            const int ti = (int)((nxp_q + (FP_ONE >> 1)) >> FP_SHIFT);
            const int tj = (int)((nyp_q + (FP_ONE >> 1)) >> FP_SHIFT);
            if (gas_wall_at(tj, ti, thermal_solid, h, w)) break;
            cx_q = nxp_q;
            cy_q = nyp_q;
            if (tj >= 0 && tj < h && ti >= 0 && ti < w && is_vacuum[tj * w + ti])
                break;                                        // reached the breach
        }
        px_q = cx_q;
        py_q = cy_q;
    }

    // ---- Clamp in-bounds (Q16.16) ----
    const int64_t hi_x = (int64_t)(w - 1) << FP_SHIFT;
    const int64_t hi_y = (int64_t)(h - 1) << FP_SHIFT;
    if (px_q < 0) px_q = 0; else if (px_q > hi_x) px_q = hi_x;
    if (py_q < 0) py_q = 0; else if (py_q > hi_y) py_q = hi_y;

    // ---- Integer bilinear sample ----
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

    int64_t acc = 0;
    int32_t wsum_q = 0;
    for (int k = 0; k < 4; ++k) {
        const int cy_ = cyx[k][0];
        const int cx_ = cyx[k][1];
        const int j = cy_ * w + cx_;
        // MEDIUM-TEST SITE 4/6 (sealed corner in the bilinear gather).
        if (thermal_solid[j]) continue;                      // sealed corner
        const int32_t val_q = is_vacuum[j] ? 0 : src[j];      // breach corner == 0
        acc += mul_wide(cw[k], val_q);
        wsum_q += cw[k];
    }
    if (wsum_q <= GAS_WSUM_EPS_Q) return src[y * w + x];      // negligible -> keep self

    const int32_t wsum_clamped = (wsum_q < GAS_WSUM_FLOOR_Q) ? GAS_WSUM_FLOOR_Q : wsum_q;
    const int32_t recip_q = reciprocal_q16_dev(wsum_clamped);
    const int32_t acc_q = narrow(acc);
    return mul_q16(acc_q, recip_q);
}

// ---- Pass 0b: semi-Lagrangian advection on the open-air mask ----------------
// One thread per cell reads the FROZEN snapshot `src` and writes only its own
// temperature[i] (open-air cells only; thermal-solid/vacuum keep their Pass-0a
// value).
__global__ void temp_advect(int32_t* __restrict__ temperature,
                            const int32_t* __restrict__ src,
                            const int32_t* __restrict__ wind_x,
                            const int32_t* __restrict__ wind_y,
                            const bool* __restrict__ thermal_solid,
                            const bool* __restrict__ is_vacuum,
                            int32_t dt_adv_q, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        // MEDIUM-TEST SITE 2/6: the gas-advection mask is the complement of the
        // THERMAL medium (a crate is not open air thermally, even though gas
        // flows through it).
        if (thermal_solid[i] || is_vacuum[i]) continue;      // open-air mask only
        const int y = i / w;
        const int x = i % w;
        const int32_t bx_q = -mul_q16(wind_x[i], dt_adv_q);
        const int32_t by_q = -mul_q16(wind_y[i], dt_adv_q);
        temperature[i] = gas_backtrace_sample_q_dev(
            src, x, y, bx_q, by_q, thermal_solid, is_vacuum, h, w);
    }
}

// ---- Pass 1: heat -> temperature deposit (§1.2 solids; §4.3 open-air) -------
// Solid: the UNCHANGED bit-shift. Open-air (non-vacuum): the v2.4 absorption-∝-
// density radiant deposit ΔT = E_abs/(N·c_v) — N from `n_src` (n_bulk, or the
// atmosphere density proxy when n_bulk is null; the host points n_src at whichever
// the CPU would read). Both branches clamp at the counted T_MAX_PHYS rail; each
// engagement atomicAdds the (order-free) hit counter. Single writer per cell.
__global__ void temp_convert_unified(int32_t* __restrict__ temperature,
                                     const int32_t* __restrict__ heat,
                                     const int32_t* __restrict__ heat_inv_shift,
                                     const bool* __restrict__ thermal_solid,
                                     const bool* __restrict__ is_vacuum,
                                     const int32_t* __restrict__ n_src,
                                     int64_t recip_cv, int32_t n_floor_q,
                                     int32_t t_max_phys_q,
                                     unsigned long long* __restrict__ hits, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int32_t deposit = heat[i];
        if (deposit <= 0) continue;                          // nothing to convert
        int32_t t = temperature[i];
        // MEDIUM-TEST SITE 5/6: the heat->T convert branch. A THERMAL solid takes
        // the free per-tile bit-shift (heat >> log2(thermal_mass)); gas takes the
        // N-divided radiative deposit below.
        if (thermal_solid[i]) {
            const int shift = heat_inv_shift[i];             // log2(thermal_mass)
            const int32_t gain = deposit >> shift;           // Q16.16 / 2^shift
            heat_saturating_add_dev(&t, gain);
            if (t > t_max_phys_q) { t = t_max_phys_q; atomicAdd(hits, 1ULL); }
            temperature[i] = t;
        } else if (!is_vacuum[i]) {
            // v2.4 absorption-proportional radiant deposit (optically-thin form):
            //   E_abs = deposit · min(N, N_AMB)/N_AMB   (N_AMB == FP_ONE)
            //   ΔT    = E_abs / (max(N, N_FLOOR_HEAT) · c_v)
            int32_t N_raw = n_src[i];
            if (N_raw < 0) N_raw = 0;                        // no negative density
            const int32_t e_abs = (N_raw >= FP_ONE)
                ? deposit                                    // ambient+: exact old path
                : mul_q16(deposit, (q16)N_raw);              // thin gas: ∝ density
            int32_t N_q = N_raw;
            if (N_q < n_floor_q) N_q = n_floor_q;            // N_FLOOR_HEAT
            const int32_t recip_N_q = reciprocal_q16_dev(N_q);
            const int32_t e_over_n  = mul_q16(e_abs, recip_N_q);
            const int32_t dT = recip_mul_dev(e_over_n, recip_cv);
            heat_saturating_add_dev(&t, dT);
            if (t > t_max_phys_q) { t = t_max_phys_q; atomicAdd(hits, 1ULL); }
            temperature[i] = t;
        }
    }
}

// ---- Pass 2: conduction relaxation (§2.2, gather, double-buffered) ---------
// Reads the FROZEN temperature, writes temp_new[i]. The DIFFERENCE is shifted,
// not the neighbour (equal neighbours -> exactly 0). int64 accumulator, identical
// to the CPU. Every cell is fully written (air -> all NO_FACE -> acc=0 -> temp_new
// == ti), so temp_new has no uninitialised read (scratch hygiene).
__global__ void temp_conduct(const int32_t* __restrict__ temperature,
                             int32_t* __restrict__ temp_new,
                             const int32_t* __restrict__ face_shift,
                             int no_face, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int y = i / w;
        const int x = i % w;
        const int32_t* fs = &face_shift[i * 4];
        const int32_t ti = temperature[i];
        long long acc = 0;
        for (int d = 0; d < 4; ++d) {
            const int s = fs[d];
            if (s == no_face) continue;
            const int ny = y + dy_of(d);
            const int nx = x + dx_of(d);
            if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
            const int32_t tn = temperature[ny * w + nx];
            acc += (long long)(tn - ti) >> s;                // arithmetic shift
        }
        temp_new[i] = (int32_t)((long long)ti + acc);
    }
}

// ---- Pass 3: ambient cooling (§3, thermal solids only, vacuum-exposed 4x) ---
// In-place on temperature[i]; reads own cell + neighbours' is_vacuum/atmosphere
// (frozen -> safe). Symmetric round-toward-0 shift; the dead-band is preserved.
// COOL-SHIFT AXIS (2026-07-30) — the exact device twin of the CPU Pass 3. The
// base decay shift is now PER TILE (`cool_shift_grid`, null -> the `cool_shift`
// scalar) and the vacuum-exposed shift is that base minus the ONE global
// offset (cool_shift - cool_shift_vacuum, computed on the host and passed in as
// `vac_offset`), clamped at `cool_shift_floor`. Rationale for the offset form
// (one dial per material; the 4x space discount is a property of the boundary,
// not of the material) lives at the CPU site — temperature_solver.cpp Pass 3.
__global__ void temp_cool(int32_t* __restrict__ temperature,
                          const bool* __restrict__ thermal_solid,
                          const bool* __restrict__ is_vacuum,
                          const int32_t* __restrict__ atmosphere,
                          const int32_t* __restrict__ cool_shift_grid,
                          int cool_shift, int vac_offset, int cool_shift_floor,
                          int32_t thresh_q, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        // MEDIUM-TEST SITE 6/6: COOL_SHIFT ambient decay is the SOLID thermal
        // regime's loss channel. furniture's conductivity is 0 (NO_FACE both
        // ways -> no conduction in or out), so with the crate now inside this
        // pass COOL_SHIFT is its ONE loss channel — one clean dial (§2.2).
        if (!thermal_solid[i]) continue;
        const int32_t t = temperature[i];
        if (t == 0) continue;
        const int y = i / w;
        const int x = i % w;
        bool exposed = false;
        for (int d = 0; d < 4; ++d) {
            const int ny = y + dy_of(d);
            const int nx = x + dx_of(d);
            if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
            const int ni = ny * w + nx;
            if (is_vacuum[ni] || atmosphere[ni] < thresh_q) {
                exposed = true;
                break;
            }
        }
        const int base_shift =
            (cool_shift_grid != nullptr) ? (int)cool_shift_grid[i] : cool_shift;
        int shift = base_shift;
        if (exposed) {
            shift = base_shift - vac_offset;
            if (shift < cool_shift_floor) shift = cool_shift_floor;
        }
        const int32_t loss = (t < 0) ? -((-t) >> shift) : (t >> shift);
        temperature[i] = t - loss;
    }
}

}  // namespace

int64_t temperature_step(
    int32_t* temperature, const int32_t* heat, const int32_t* heat_inv_shift,
    const int32_t* face_shift, const bool* solid, const bool* is_vacuum,
    const int32_t* atmosphere, const int32_t* n_bulk,
    const int32_t* wind_x, const int32_t* wind_y,
    int no_face, int cool_shift, int cool_shift_vacuum, float o2_vacuum_thresh,
    float c_v, float n_floor_heat, float gas_advection_rate, float t_max_phys,
    int h, int w, float dt,
    const bool* is_ambient,     // BC: ring wiped to ΔT=0 in Pass 0 (nullptr=space)
    const bool* thermal_solid,  // thermal-mass axis: medium mask (nullptr -> solid)
    const int32_t* cool_shift_grid,  // cool-shift axis: per-tile decay shift
                                      // (nullptr -> the cool_shift scalar)
    int cool_shift_floor) {     // low clamp on the vacuum offset (== SHIFT_MIN)
    const int n = h * w;
    if (n <= 0) return 0;

    // The SAME once-per-step host boundary casts the CPU does (round-to-nearest
    // quantize; make_recip for 1/c_v). Every scalar the kernels need is derived
    // here so the device code is float-free.
    const int32_t thresh_q = quantize((double)o2_vacuum_thresh);
    const double c_v_safe = (c_v > 0.0f) ? (double)c_v : 1.0;
    const int64_t recip_cv = make_recip(c_v_safe);
    const int32_t n_floor_q = quantize((double)n_floor_heat);
    const int32_t t_max_phys_q = quantize((double)t_max_phys);
    const bool do_advect = (dt > 0.0f && wind_x != nullptr && wind_y != nullptr);
    // COOL-SHIFT AXIS: the vacuum discount as a DIFFERENCE, computed ONCE on
    // the host exactly as the CPU solver's Pass 3 does (`const int vac_offset =
    // cool_shift - cool_shift_vacuum;`). Pure integer, no boundary cast.
    const int vac_offset = cool_shift - cool_shift_vacuum;

    const size_t nb = (size_t)n * sizeof(int32_t);
    const size_t nbool = (size_t)n * sizeof(bool);
    int32_t *d_temp = nullptr, *d_temp_new = nullptr, *d_heat = nullptr,
            *d_his = nullptr, *d_fs = nullptr, *d_atm = nullptr,
            *d_nbulk = nullptr, *d_src = nullptr, *d_wx = nullptr, *d_wy = nullptr,
            *d_csg = nullptr;
    bool *d_solid = nullptr, *d_vac = nullptr, *d_tsol = nullptr;
    unsigned long long* d_hits = nullptr;

    cuda_check(cudaMalloc(&d_temp, nb), "malloc temp");
    cuda_check(cudaMalloc(&d_temp_new, nb), "malloc temp_new");
    cuda_check(cudaMalloc(&d_heat, nb), "malloc heat");
    cuda_check(cudaMalloc(&d_his, nb), "malloc heat_inv_shift");
    cuda_check(cudaMalloc(&d_fs, nb * 4), "malloc face_shift");
    cuda_check(cudaMalloc(&d_atm, nb), "malloc atmosphere");
    cuda_check(cudaMalloc(&d_solid, nbool), "malloc solid");
    cuda_check(cudaMalloc(&d_vac, nbool), "malloc is_vacuum");
    cuda_check(cudaMalloc(&d_hits, sizeof(unsigned long long)), "malloc hits");
    if (n_bulk) cuda_check(cudaMalloc(&d_nbulk, nb), "malloc n_bulk");
    // THERMAL-MASS AXIS: the medium mask rides as its OWN plane only when the
    // caller supplies one; with nullptr the kernels are pointed straight at
    // d_solid, mirroring the CPU's `ts = thermal_solid ? thermal_solid : solid`
    // — so the fallback allocates and copies nothing (and is not a second code
    // path). `solid` itself keeps its unconditional upload: it IS that fallback.
    if (thermal_solid) cuda_check(cudaMalloc(&d_tsol, nbool), "malloc thermal_solid");
    // COOL-SHIFT AXIS: same nullable-plane idiom — with nullptr the kernel is
    // handed a null pointer and falls back to the `cool_shift` scalar per cell,
    // the exact CPU twin, so the fallback allocates and copies nothing.
    if (cool_shift_grid) cuda_check(cudaMalloc(&d_csg, nb), "malloc cool_shift_grid");
    if (do_advect) {
        cuda_check(cudaMalloc(&d_src, nb), "malloc src");
        cuda_check(cudaMalloc(&d_wx, nb), "malloc wind_x");
        cuda_check(cudaMalloc(&d_wy, nb), "malloc wind_y");
    }

    cuda_check(cudaMemcpy(d_temp, temperature, nb, cudaMemcpyHostToDevice), "H2D temp");
    cuda_check(cudaMemcpy(d_heat, heat, nb, cudaMemcpyHostToDevice), "H2D heat");
    cuda_check(cudaMemcpy(d_his, heat_inv_shift, nb, cudaMemcpyHostToDevice), "H2D his");
    cuda_check(cudaMemcpy(d_fs, face_shift, nb * 4, cudaMemcpyHostToDevice), "H2D fs");
    cuda_check(cudaMemcpy(d_atm, atmosphere, nb, cudaMemcpyHostToDevice), "H2D atm");
    cuda_check(cudaMemcpy(d_solid, solid, nbool, cudaMemcpyHostToDevice), "H2D solid");
    cuda_check(cudaMemcpy(d_vac, is_vacuum, nbool, cudaMemcpyHostToDevice), "H2D vac");
    if (thermal_solid)
        cuda_check(cudaMemcpy(d_tsol, thermal_solid, nbool, cudaMemcpyHostToDevice),
                   "H2D thermal_solid");
    if (cool_shift_grid)
        cuda_check(cudaMemcpy(d_csg, cool_shift_grid, nb, cudaMemcpyHostToDevice),
                   "H2D cool_shift_grid");
    // BC: optional ambient ring mask for the Pass-0 wipe (nullptr on space maps).
    bool* d_amb = nullptr;
    if (is_ambient) {
        cuda_check(cudaMalloc(&d_amb, nbool), "malloc is_ambient");
        cuda_check(cudaMemcpy(d_amb, is_ambient, nbool, cudaMemcpyHostToDevice), "H2D is_ambient");
    }
    if (n_bulk) cuda_check(cudaMemcpy(d_nbulk, n_bulk, nb, cudaMemcpyHostToDevice), "H2D nbulk");
    if (do_advect) {
        cuda_check(cudaMemcpy(d_wx, wind_x, nb, cudaMemcpyHostToDevice), "H2D wx");
        cuda_check(cudaMemcpy(d_wy, wind_y, nb, cudaMemcpyHostToDevice), "H2D wy");
    }
    cuda_check(cudaMemset(d_hits, 0, sizeof(unsigned long long)), "memset hits");

    // The N divisor source Pass 1 reads: n_bulk when supplied, else the atmosphere
    // density proxy — EXACTLY the CPU's `n_bulk ? n_bulk[i] : atmosphere[i]`.
    const int32_t* d_nsrc = n_bulk ? d_nbulk : d_atm;

    // THERMAL-MASS AXIS: the mask the SIX medium tests read — the exact device
    // twin of the CPU solver's `const bool* ts = thermal_solid ? thermal_solid
    // : solid`. Every kernel below takes `d_ts`, never `d_solid`.
    const bool* d_ts = thermal_solid ? (const bool*)d_tsol : (const bool*)d_solid;

    const int block = 256;
    const int grid = (n + block - 1) / block;

    // Pass 0a: zero gas-T at open vacuum cells (unconditional, in-place on d_temp).
    temp_zero_vacuum<<<grid, block>>>(d_temp, d_ts, d_vac, n, d_amb);
    cuda_check(cudaGetLastError(), "zero_vacuum launch");

    // Pass 0b: semi-Lagrangian advection (only when wind + dt>0, matching the CPU
    // guard). The snapshot is taken AFTER the zero-vacuum write (the CPU order).
    if (do_advect) {
        const double dt_adv = (double)gas_advection_rate * (double)dt;
        const int32_t dt_adv_q = quantize(dt_adv);
        cuda_check(cudaMemcpy(d_src, d_temp, nb, cudaMemcpyDeviceToDevice), "D2D src");
        temp_advect<<<grid, block>>>(d_temp, d_src, d_wx, d_wy, d_ts, d_vac,
                                     dt_adv_q, h, w);
        cuda_check(cudaGetLastError(), "advect launch");
    }

    // Pass 1: unified convert (in-place on d_temp; rail counter -> d_hits).
    temp_convert_unified<<<grid, block>>>(d_temp, d_heat, d_his, d_ts, d_vac,
                                          d_nsrc, recip_cv, n_floor_q,
                                          t_max_phys_q, d_hits, n);
    cuda_check(cudaGetLastError(), "convert launch");

    // Pass 2: conduct (d_temp -> d_temp_new), then copy back (the CPU swap).
    temp_conduct<<<grid, block>>>(d_temp, d_temp_new, d_fs, no_face, h, w);
    cuda_check(cudaGetLastError(), "conduct launch");
    cuda_check(cudaMemcpy(d_temp, d_temp_new, nb, cudaMemcpyDeviceToDevice), "D2D swap");

    // Pass 3: cool (in-place on d_temp).
    temp_cool<<<grid, block>>>(d_temp, d_ts, d_vac, d_atm, d_csg,
                               cool_shift, vac_offset, cool_shift_floor,
                               thresh_q, h, w);
    cuda_check(cudaGetLastError(), "cool launch");
    cuda_check(cudaDeviceSynchronize(), "sync");

    unsigned long long hits = 0;
    cuda_check(cudaMemcpy(&hits, d_hits, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost), "D2H hits");
    cuda_check(cudaMemcpy(temperature, d_temp, nb, cudaMemcpyDeviceToHost), "D2H temp");

    cudaFree(d_temp);
    cudaFree(d_temp_new);
    cudaFree(d_heat);
    cudaFree(d_his);
    cudaFree(d_fs);
    cudaFree(d_atm);
    cudaFree(d_solid);
    cudaFree(d_vac);
    cudaFree(d_hits);
    if (d_nbulk) cudaFree(d_nbulk);
    if (d_src) cudaFree(d_src);
    if (d_wx) cudaFree(d_wx);
    if (d_wy) cudaFree(d_wy);
    if (d_amb) cudaFree(d_amb);
    if (d_tsol) cudaFree(d_tsol);
    if (d_csg) cudaFree(d_csg);

    return (int64_t)hits;
}

namespace {
bool g_temp_backend_cuda = false;
}
bool temperature_backend_is_cuda() { return g_temp_backend_cuda; }
void set_temperature_backend_cuda(bool on) { g_temp_backend_cuda = on; }

}  // namespace breach_cuda
