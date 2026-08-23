// ============================================================================
// EOS P6.5 — the full eos.step per-call GPU orchestration — see cuda_eos_step.h
// (the pass-boundary map lives there; per-stage rationale inline below).
//
// Every HOST stage in this file is a VERBATIM transcription of the matching
// EOSSolver::step stage (eos_solver.cpp — stage line references inline), in
// the same fold order, on the same /fp:strict host floor (CMake gives every
// .cu -Xcompiler=/fp:strict), using the same header-inline fixedpoint helpers
// — identical bits by determinism of quantize/IEEE double and pure-integer
// code. Every DEVICE stage launches kernels that exist in exactly ONE
// transcription each (cuda_sl_advection.cu / cuda_bulk_transport.cu via the
// P6.5 device-pointer launchers; cuda_mg_solve.cu / cuda_kick_compression.cu
// via their proven isolated entries).
//
// ORDERING NOTE (the interleave the isolated gates did not cover): the CPU
// substep loop is  [advect(u,T) -> bulk flux(gas) -> zero-u-on-solid] x n_sub.
// Advection reads/writes only u/T; bulk flux reads u (post-advection, this
// substep) and writes only gas planes — the ONLY cross-kernel dependency is
// "flux s reads advect s's u", which one in-order stream preserves exactly
// (kernel order == CPU pass order). zero-u-on-solid is subsumed by the
// advection kernel (the proven P6.2 argument). The all-zero-plane skip is
// dropped (review §1.3: arithmetically a no-op — kept as a host-side scan on
// the CPU only for perf).
// ============================================================================
#include "cuda_eos_step.h"
#include "cuda_sl_advection.h"     // sl_cmask_build_device / sl_advect3_device
#include "cuda_bulk_transport.h"   // bulk_flux_plane_device
#include "cuda_mg_solve.h"         // eos_mg_vcycle (the proven P6.3 entry)
#include "cuda_kick_compression.h" // eos_kick_compression (the proven P6.4 entry)
#include "eos_solver.h"            // EOSSolver (config + mg_build_levels + telemetry)
#include "fixed_point.h"           // q16, quantize, mul_q16, mul_wide, sqrt_q16,
                                   // reciprocal_q16, ceil_div, FP_ONE

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#if !defined(__SIZEOF_INT128__) && defined(_MSC_VER)
#include <intrin.h>
#endif

using namespace fixedpoint;

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in eos_step_cuda/" << what << ": "
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

// ---- host mul128_shr (eos_solver.cpp's file-local helper, verbatim) ---------
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
inline int mirror_idx_host(int self_i, int ny, int nx, int h, int w,
                           const bool* solid) {
    if (ny < 0 || ny >= h || nx < 0 || nx >= w) return self_i;
    const int ni = ny * w + nx;
    if (solid[ni]) return self_i;
    return ni;
}

long long g_eos_step_cuda_calls = 0;

}  // namespace

bool eos_step_backend_is_cuda() {
    return sl_advection_backend_is_cuda()
        && bulk_flux_backend_is_cuda()
        && mg_solve_backend_is_cuda()
        && kick_compression_backend_is_cuda();
}

// ============================================================================
// S8a Path A: the shared HOST pre-stage — the VERBATIM step() transcription
// block below eos_step_cuda has always run, factored (PURE CODE MOTION) so
// the device-resident entry consumes the identical bits. See the header.
// ============================================================================
EOSHostPrestage eos_host_prestage(
        const EOSSolver& solver,
        const int32_t* atmosphere,
        int32_t* p_prev,
        const int32_t* wind_x, const int32_t* wind_y,
        const int32_t* temperature,
        const int32_t* gas, const bool* gas_conservative, int n_gases,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability,
        int h, int w, float dt,
        bool ambient_mode,
        const bool* thermal_solid) {
    EOSHostPrestage pre;
    const int n = h * w;
    // THERMAL-MASS AXIS: the same nullptr -> `solid` fallback step()'s `ts`
    // fold uses. VELOCITY-CLAMP (P-V1, D4): the cap2 fold below reads it.
    const bool* ts = (thermal_solid != nullptr) ? thermal_solid : solid;

    // The boundary_flux rail (spec §5): zero it each tick in ambient mode; the
    // per-substep bulk reset accumulates into it on device, copied back later.
    if (ambient_mode) {
        if ((int)solver.boundary_flux_.size() != n_gases)
            solver.boundary_flux_.assign(n_gases, 0);
        else
            std::fill(solver.boundary_flux_.begin(), solver.boundary_flux_.end(), (int64_t)0);
    } else if (!solver.boundary_flux_.empty()) {
        solver.boundary_flux_.clear();
    }

    // ---- step 0: P_prev := P (pure copy) ---------------------------------
    for (int i = 0; i < n; ++i) p_prev[i] = atmosphere[i];

    // DEBUG probe parity: T at step-1 entry.
    if (solver.dbg_probe_idx >= 0 && solver.dbg_probe_idx < n)
        solver.dbg_T_pre_advect = temperature[solver.dbg_probe_idx];

    // ---- per-tick scalar constants (step()'s folds, verbatim) ------------
    const q16 n_floor_q = quantize((double)solver.N_FLOOR_SOLVER);
    // FLOORED AT 1 COUNT — the CPU twin's A7 fix, verbatim (eos_solver.cpp:278).
    // Divisor of the c_LOCAL ratio at :165 and, via pre.t_amb_q (:258), the
    // resident path's too. Must stay bit-identical to the CPU fold.
    const q16 t_amb_q   = std::max<q16>(1, quantize((double)solver.T_AMB_K));
    // P-E1: the recovery T_MIN clamp fold — the SAME `quantize((double)T_MIN)`
    // expression step() performs, hoisted here so BOTH GPU dispatch paths read
    // one transcription (design §2.1.5).
    pre.t_min_q = quantize((double)solver.T_MIN);
    // s_eos_q: fold of S_EOS, verbatim CPU twin (eos_solver.cpp:290). At the
    // frozen identity (65536) the t_abs product below has zero low bits, so
    // the SAR is exact truncation — see eos_solver.cpp's comment for the
    // off-identity T<0 flooring convention.
    const q16 s_eos_q   = quantize((double)solver.S_EOS);
    // D-3 RELEASE-LIVE GUARD (design §4, docs/tabs_compression_work_design_
    // 2026-08-20.md): the byte-identical companion to EOSSolver::step's own
    // guard (eos_solver.cpp, beside its s_eos_q fold) — step 4c's t_abs =
    // T + t_amb_q form is honest only while S_EOS == 1 AND
    // T_MIN > -T_AMB_K; assert() is dead in Release, so this is a plain,
    // always-compiled, once-per-tick check on this dispatch path too.
    if (s_eos_q != FP_ONE || pre.t_min_q <= -(int64_t)t_amb_q) {
        throw std::runtime_error(
            "T_abs compression work requires S_EOS==1 and T_MIN > -T_AMB_K; "
            "see docs/tabs_compression_work_design_2026-08-20.md D-3");
    }
    const q16 c_q       = quantize((double)solver.C);
    // VELOCITY-CLAMP (P-V1, D2v2/v2.4): u_max_q fold — the CPU twin's rail
    // (eos_solver.cpp:384), missing here pre-P-V1 (the kick never needed it
    // on this backend until the per-cell cap plane).
    const q16 u_max_q   = quantize((double)solver.U_MAX);
    const double gamma_d = (double)solver.adiabatic_index;
    const double dt_d    = (double)dt;
    const q16 dt_q       = quantize(dt_d);
    const double dx_d    = std::max((double)solver.dx, 1e-6);
    const q16 inv_2dx_q  = quantize(1.0 / (2.0 * dx_d));
    const double K_d = (double)solver.c_max * (double)solver.c_max / gamma_d;
    const int64_t K_raw = (int64_t)(K_d * 65536.0 + 0.5);
    const int64_t Kdt_raw = mul128_shr_host(K_raw, (int64_t)dt_q, 16);

    // ---- c_LOCAL = c_amb·sqrt(T_max_abs/T_AMB) (step()'s scan, verbatim) --
    const q16 c_amb_q = quantize((double)solver.c_max);
    // VELOCITY-CLAMP (P-V1, D2v2): the per-cell cap² fold, SAME scan, SAME
    // tick-entry T basis as c_LOCAL — the eos_solver.cpp scan's TWIN
    // transcription (design's "TWO transcription sites, no third").
    const int64_t c_amb2_q32 = (int64_t)c_amb_q * (int64_t)c_amb_q;   // Q32.32
    const int64_t u_max2_q32 = (int64_t)u_max_q * (int64_t)u_max_q;   // Q32.32
    const double ru = (double)u_max_q / (double)c_amb_q;
    const int64_t ratio_umax = (int64_t)(ru * ru * 65536.0) + 1;

    pre.cap2.assign(n, 0);
    int64_t t_max_abs_raw = (int64_t)t_amb_q;
    for (int i = 0; i < n; ++i) {
        if (solid[i] || is_vacuum[i]) { pre.cap2[i] = u_max2_q32; continue; }
        const int64_t t_abs = (((int64_t)s_eos_q * (int64_t)temperature[i]) >> 16) + (int64_t)t_amb_q;
        if (t_abs > t_max_abs_raw) t_max_abs_raw = t_abs;

        // D4 + D1: ts-gas cells get the AMBIENT cap; floor at ambient. A
        // LOCAL copy: t_max_abs_raw above must stay on the unfloored,
        // un-ts'd t_abs.
        int64_t t_abs_cap = t_abs;
        if (ts[i] || t_abs_cap < (int64_t)t_amb_q) t_abs_cap = (int64_t)t_amb_q;
        const int64_t ratio = (t_abs_cap << 16) / (int64_t)t_amb_q;   // int64, NO narrow
        pre.cap2[i] = (ratio >= ratio_umax)
            ? u_max2_q32
            : mul128_shr_host(c_amb2_q32, ratio, 16);
    }
    const int32_t ratio_q = (int32_t)((t_max_abs_raw << 16) / (int64_t)t_amb_q);
    const q16 sqrt_ratio = sqrt_q16((int64_t)ratio_q << 16);   // Q.32 radicand
    q16 c_local_q = mul_q16(c_amb_q, sqrt_ratio);
    if (c_local_q < c_amb_q) c_local_q = c_amb_q;   // never below ambient
    solver.dbg_last_c_local_q = c_local_q;

    // ---- substep count: max|u|, Dalton N, K·|∇P|·dt/N̂ scan, ceil_div ------
    // (step()'s reductions, verbatim — host-side per review §1.6.)
    int64_t max_rad = 0;
    for (int i = 0; i < n; ++i) {
        const int64_t rad = mul_wide(wind_x[i], wind_x[i])
                          + mul_wide(wind_y[i], wind_y[i]);
        if (rad > max_rad) max_rad = rad;
    }
    const q16 max_u = sqrt_q16(max_rad);
    // P-T0 (design §2.6): n_total ≡ n_bulk; trace planes skipped outright.
    std::vector<int32_t> n_total(n, 0);
    {
        for (int gi = 0; gi < n_gases; ++gi) {
            if (!gas_conservative[gi]) continue;
            const int32_t* plane = gas + (size_t)gi * n;
            for (int i = 0; i < n; ++i) n_total[i] += plane[i];
        }
    }
    int64_t max_du_raw = 0;
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (solid[i]) continue;
            const int il = mirror_idx_host(i, y, x - 1, h, w, solid);
            const int ir = mirror_idx_host(i, y, x + 1, h, w, solid);
            const int iu = mirror_idx_host(i, y - 1, x, h, w, solid);
            const int id = mirror_idx_host(i, y + 1, x, h, w, solid);
            const int64_t gx = mul128_shr_host((int64_t)(p_prev[ir] - p_prev[il]),
                                               (int64_t)inv_2dx_q, 16);
            const int64_t gy = mul128_shr_host((int64_t)(p_prev[id] - p_prev[iu]),
                                               (int64_t)inv_2dx_q, 16);
            const int64_t agx = gx < 0 ? -gx : gx;
            const int64_t agy = gy < 0 ? -gy : gy;
            const int64_t gmag = agx > agy ? agx : agy;
            if (gmag == 0) continue;
            q16 nhat = n_total[i];
            if (nhat < n_floor_q) nhat = n_floor_q;
            const q16 inv_n = reciprocal_q16(nhat);
            const int64_t t1 = mul128_shr_host(Kdt_raw, gmag, 16);
            const int64_t du = mul128_shr_host(t1, (int64_t)inv_n, 16);
            if (du > max_du_raw) max_du_raw = du;
        }
    }
    int64_t u_est_raw = (int64_t)max_u + max_du_raw + 1;   // +1 count eps
    // D7 (VELOCITY-CLAMP, P-V1): the clip widens to max(c_LOCAL, U_MAX) —
    // verbatim the CPU twin's fold (eos_solver.cpp) — stored |u| may now
    // reach U_MAX on hot cells even though c_LOCAL is folded from entry-T.
    const int64_t u_est_cap = ((int64_t)c_local_q > (int64_t)u_max_q)
        ? (int64_t)c_local_q : (int64_t)u_max_q;
    if (u_est_raw > u_est_cap) u_est_raw = u_est_cap;
    const q16 cfl_dx_q = quantize((double)solver.CFL_ADV * dx_d);
    const int64_t numer_wide = mul128_shr_host((int64_t)dt_q, u_est_raw, 16);
    int n_sub = std::max(1, ceil_div(
        (q16)std::min<int64_t>(numer_wide, (int64_t)INT32_MAX), cfl_dx_q));
    if (n_sub > solver.N_SUB_MAX) n_sub = solver.N_SUB_MAX;
    solver.dbg_last_n_sub = n_sub;
    const double dt_s_d = dt_d / (double)n_sub;
    // dt_s_q: the CPU re-quantizes the SAME double each substep — one fold
    // here is the identical value (the proven P6.2 host fold).
    const q16 dt_s_q = quantize(dt_s_d);

    // ---- donor-cell face-coefficient cache (step()'s hoist, verbatim) ----
    pre.coeffE.assign(n, 0);
    pre.coeffS.assign(n, 0);
    {
        const q16 dts_q_c = quantize(dt_s_d);
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            for (int x = 0; x < w; ++x) {
                const int i = row + x;
                if (solid[i]) continue;
                if (x < w - 1 && !solid[i + 1]) {
                    const float ff = std::min(dyn_permeability[i], dyn_permeability[i + 1]);
                    if (ff > 0.0f) pre.coeffE[i] = mul_q16(quantize((double)ff), dts_q_c);
                }
                if (y < h - 1 && !solid[i + w]) {
                    const float ff = std::min(dyn_permeability[i], dyn_permeability[i + w]);
                    if (ff > 0.0f) pre.coeffS[i] = mul_q16(quantize((double)ff), dts_q_c);
                }
            }
        }
    }

    // Conservative-plane index list (the CPU's gi order preserved; the
    // planes are independent — disjoint N, read-only wind/coeffs).
    for (int gi = 0; gi < n_gases; ++gi)
        if (gas_conservative[gi]) pre.cons.push_back(gi);

    pre.t_amb_q   = t_amb_q;
    pre.s_eos_q   = s_eos_q;
    pre.c_q       = c_q;
    pre.inv_2dx_q = inv_2dx_q;
    pre.c_local_q = c_local_q;
    pre.n_sub     = n_sub;
    pre.dt_s_q    = dt_s_q;
    return pre;
}

long long eos_step_cuda_calls() { return g_eos_step_cuda_calls; }

void eos_step_cuda(
        const EOSSolver& solver,
        int32_t* atmosphere,
        int32_t* p_prev,
        int32_t* wind_x, int32_t* wind_y,
        int32_t* temperature,
        int32_t* gas, const bool* gas_conservative, int n_gases,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability, const float* dyn_wave_absorb,
        int h, int w, float dt,
        const bool* is_ambient, const int32_t* n_amb, int32_t p_amb,
        const int32_t* sponge_sigma, const int32_t* sponge_udamp,
        const bool* thermal_solid) {

    const int n = h * w;
    if (n <= 0 || dt <= 0.0f) return;   // step()'s degenerate early-out
    ++g_eos_step_cuda_calls;

    // BC: dormancy BY BRANCH — every ambient edit gated on this (space maps take
    // the byte-identical path). Mirrors EOSSolver::step's ambient_mode.
    const bool ambient_mode = (is_ambient != nullptr);
    // THERMAL-MASS AXIS, P-EOS: does the thermal medium diverge from the gas
    // medium anywhere? The SHARED predicate (eos_solver.h) — one transcription,
    // so this host and EOSSolver::step can never disagree about whether the
    // T-only occluder mask is live.
    // P-E1 (design §2.1.1): the A2 T-only occluder mask retires with the SL
    // T sample on this backend too — `eos_thermal_occludes` has no consumer
    // here any more. `d_ts` survives: it is the PARTICIPATION mask of the new
    // energy build/recovery (ruling A1 — the EOS never writes a ts tile's T).

    // ======================================================================
    // HOST PRE-STAGE — the shared verbatim step() transcription (S8a Path A
    // pure code motion into eos_host_prestage above; boundary_flux_ reset,
    // step-0 p_prev copy, scalar folds, c_LOCAL/n_sub scans, coeffE/S, cons).
    // ======================================================================
    const EOSHostPrestage pre = eos_host_prestage(
        solver, atmosphere, p_prev, wind_x, wind_y, temperature,
        gas, gas_conservative, n_gases, solid, is_vacuum,
        dyn_permeability, h, w, dt, ambient_mode, thermal_solid);
    const q16 t_amb_q   = pre.t_amb_q;
    const q16 s_eos_q   = pre.s_eos_q;
    const q16 c_q       = pre.c_q;
    const q16 inv_2dx_q = pre.inv_2dx_q;
    // VELOCITY-CLAMP (P-V1, D2v2): the local c_local_q unpack RETIRES here —
    // the kick now reads pre.cap2 (below), not this scalar; pre.c_local_q
    // itself still feeds dbg_last_c_local_q via eos_host_prestage.
    const int n_sub     = pre.n_sub;
    const q16 dt_s_q    = pre.dt_s_q;
    const std::vector<int32_t>& coeffE = pre.coeffE;
    const std::vector<int32_t>& coeffS = pre.coeffS;
    const std::vector<int>& cons = pre.cons;
    // Mid-stage Dalton scratch (re-zeroed + refilled below, exactly like the
    // CPU's member-cache reuse — a fresh vector holds the same bytes).
    std::vector<int32_t> n_total(n, 0);

    // ======================================================================
    // DEVICE SUBSTEP LOOP — u/T/bulk-gas device-resident across all n_sub
    // substeps (the P6.5 chaining); kernel order == the CPU pass order.
    // ======================================================================
    const size_t nb    = (size_t)n * sizeof(int32_t);
    const size_t nbool = (size_t)n * sizeof(bool);
    const size_t nbf   = (size_t)n * sizeof(float);

    std::vector<void*> allocs;
    auto dev_alloc = [&](size_t bytes) -> void* {
        void* d = nullptr;
        cuda_check(cudaMalloc(&d, bytes), "malloc");
        allocs.push_back(d);
        return d;
    };
    auto free_all = [&]() {
        for (void* d : allocs) cudaFree(d);
        allocs.clear();
    };

    try {
        int32_t* d_wx  = (int32_t*)dev_alloc(nb);
        int32_t* d_wy  = (int32_t*)dev_alloc(nb);
        int32_t* d_t   = (int32_t*)dev_alloc(nb);
        int32_t* d_svx = (int32_t*)dev_alloc(nb);
        int32_t* d_svy = (int32_t*)dev_alloc(nb);
        int32_t* d_st  = (int32_t*)dev_alloc(nb);
        bool*    d_sol = (bool*)dev_alloc(nbool);
        bool*    d_vac = (bool*)dev_alloc(nbool);
        float*   d_perm = (float*)dev_alloc(nbf);
        uint8_t* d_cmask = (uint8_t*)dev_alloc((size_t)n);
        // THERMAL-MASS AXIS, P-EOS: ONE static-shaped mask upload per tick (the
        // sponge-grid precedent) + the T-only occluder mask, allocated only where
        // it can differ from cmask. d_ts falls back to d_sol (the P2 idiom), so
        // the nullptr path allocates and copies NOTHING.
        bool* d_tsol = nullptr;
        if (thermal_solid) d_tsol = (bool*)dev_alloc(nbool);
        int32_t* d_coeffE = (int32_t*)dev_alloc(nb);
        int32_t* d_coeffS = (int32_t*)dev_alloc(nb);
        int32_t* d_dq_e  = (int32_t*)dev_alloc(nb);
        int32_t* d_dq_s  = (int32_t*)dev_alloc(nb);
        int32_t* d_scale = (int32_t*)dev_alloc(nb);
        // P-E1: the transient energy plane + the applied-dq face planes + the
        // n_bulk accumulator (all int64), and the 5-slot counter block.
        const size_t nb8 = (size_t)n * sizeof(int64_t);
        int64_t* d_e       = (int64_t*)dev_alloc(nb8);
        int64_t* d_nbulk   = (int64_t*)dev_alloc(nb8);
        int64_t* d_dqsum_e = (int64_t*)dev_alloc(nb8);
        int64_t* d_dqsum_s = (int64_t*)dev_alloc(nb8);
        unsigned long long* d_ecnt =
            (unsigned long long*)dev_alloc(5 * sizeof(unsigned long long));
        cuda_check(cudaMemset(d_ecnt, 0, 5 * sizeof(unsigned long long)),
                   "memset e-counters");
        std::vector<int32_t*> d_gas(cons.size());
        for (size_t k = 0; k < cons.size(); ++k)
            d_gas[k] = (int32_t*)dev_alloc(nb);
        // P-E1 per-plane host side-tables the energy entry reads.
        std::vector<int32_t> n_amb_cons(cons.size(), 0);
        std::vector<unsigned long long*> rail_ptrs(cons.size(), nullptr);

        cuda_check(cudaMemcpy(d_wx, wind_x, nb, cudaMemcpyHostToDevice), "H2D wind_x");
        cuda_check(cudaMemcpy(d_wy, wind_y, nb, cudaMemcpyHostToDevice), "H2D wind_y");
        cuda_check(cudaMemcpy(d_t, temperature, nb, cudaMemcpyHostToDevice), "H2D temperature");
        cuda_check(cudaMemcpy(d_sol, solid, nbool, cudaMemcpyHostToDevice), "H2D solid");
        cuda_check(cudaMemcpy(d_vac, is_vacuum, nbool, cudaMemcpyHostToDevice), "H2D is_vacuum");
        cuda_check(cudaMemcpy(d_perm, dyn_permeability, nbf, cudaMemcpyHostToDevice), "H2D permeability");
        if (d_tsol)
            cuda_check(cudaMemcpy(d_tsol, thermal_solid, nbool,
                                  cudaMemcpyHostToDevice), "H2D thermal_solid");
        const bool* d_ts = thermal_solid ? d_tsol : d_sol;
        cuda_check(cudaMemcpy(d_coeffE, coeffE.data(), nb, cudaMemcpyHostToDevice), "H2D coeffE");
        cuda_check(cudaMemcpy(d_coeffS, coeffS.data(), nb, cudaMemcpyHostToDevice), "H2D coeffS");
        for (size_t k = 0; k < cons.size(); ++k)
            cuda_check(cudaMemcpy(d_gas[k], gas + (size_t)cons[k] * n, nb,
                                  cudaMemcpyHostToDevice), "H2D gas plane");

        // BC: upload the ambient ring mask (device) + allocate the per-plane
        // int64 boundary_flux rail (zeroed once; the bulk clamp atomicAdds into
        // it each substep). nullptr/empty on space maps -> byte-identical path.
        bool* d_amb = nullptr;
        unsigned long long* d_rail = nullptr;
        if (ambient_mode) {
            d_amb = (bool*)dev_alloc(nbool);
            cuda_check(cudaMemcpy(d_amb, is_ambient, nbool, cudaMemcpyHostToDevice), "H2D is_ambient");
            if (!cons.empty()) {
                d_rail = (unsigned long long*)dev_alloc(cons.size() * sizeof(unsigned long long));
                cuda_check(cudaMemset(d_rail, 0, cons.size() * sizeof(unsigned long long)), "memset rail");
            }
        }

        // K0: cmask ONCE per tick (solid/vacuum/perm constant within it) —
        // the proven P6.2 device build. BC: d_amb folds the ring into breach.
        sl_cmask_build_device(d_sol, d_vac, d_perm, d_cmask, n, d_amb);

        for (int s = 0; s < n_sub; ++s) {
            // -- a+b. FUSED SL advection (P6.2 K1) on the frozen snapshot --
            cuda_check(cudaMemcpy(d_svx, d_wx, nb, cudaMemcpyDeviceToDevice), "D2D src_vx");
            cuda_check(cudaMemcpy(d_svy, d_wy, nb, cudaMemcpyDeviceToDevice), "D2D src_vy");
            cuda_check(cudaMemcpy(d_st,  d_t,  nb, cudaMemcpyDeviceToDevice), "D2D src_t");
            sl_advect3_device(d_wx, d_wy, d_svx, d_svy, d_st,
                              d_sol, d_cmask, dt_s_q, h, w);
            // -- d. bulk O2/N2 donor-cell flux on THIS substep's u, WITH the
            //    thermal energy riding it (P-E1, design §2.1). The mass chain
            //    inside is the UNCHANGED P6.1 B1..B5 per plane. BC: the ring
            //    reset clamps N to n_amb[plane] + accumulates the rail.
            for (size_t k = 0; k < cons.size(); ++k)
                n_amb_cons[k] = ambient_mode ? n_amb[cons[k]] : 0;
            for (size_t k = 0; k < cons.size(); ++k)
                rail_ptrs[k] = d_rail ? &d_rail[k] : nullptr;
            bulk_flux_energy_transport_device(
                d_gas.data(), (int)cons.size(), d_t, d_wx, d_wy,
                d_sol, d_vac, d_ts, d_coeffE, d_coeffS, pre.t_min_q, h, w,
                d_e, d_nbulk, d_dqsum_e, d_dqsum_s,
                d_dq_e, d_dq_s, d_scale, d_ecnt,
                d_amb, n_amb_cons.data(),
                d_rail ? rail_ptrs.data() : nullptr);
            // -- f. zero u on solid: subsumed by the advection kernel (the
            //    proven P6.2 argument; nothing above re-touches u). --------
        }

        cuda_check(cudaDeviceSynchronize(), "substep-loop sync");

        // ---- D2H at the substep/solve boundary (forced host work: the
        //      host digests + the solve inputs mg_build_levels consumes). ---
        cuda_check(cudaMemcpy(wind_x, d_wx, nb, cudaMemcpyDeviceToHost), "D2H wind_x");
        cuda_check(cudaMemcpy(wind_y, d_wy, nb, cudaMemcpyDeviceToHost), "D2H wind_y");
        cuda_check(cudaMemcpy(temperature, d_t, nb, cudaMemcpyDeviceToHost), "D2H temperature");
        for (size_t k = 0; k < cons.size(); ++k)
            cuda_check(cudaMemcpy(gas + (size_t)cons[k] * n, d_gas[k], nb,
                                  cudaMemcpyDeviceToHost), "D2H gas plane");
        // BC: copy the accumulated per-plane rail back into the solver's rail
        // (byte-identical to the CPU: integer sums are order-free, so the
        // device atomicAdd total == the CPU sequential sum).
        // P-E1: the energy-transport counters back to the solver (int64
        // atomicAdd on two's complement is order-free, so the device totals
        // equal the CPU's sequential sums exactly).
        {
            unsigned long long ec[5] = {0, 0, 0, 0, 0};
            cuda_check(cudaMemcpy(ec, d_ecnt, sizeof(ec),
                                  cudaMemcpyDeviceToHost), "D2H e-counters");
            solver.e_ts_residual     = (int64_t)ec[0];
            solver.e_wipe_sum        = (int64_t)ec[1];
            solver.e_floor_sum       = (int64_t)ec[2];
            solver.n_active_flux     = (int64_t)ec[3];
            solver.n_bulk_active_sum = (int64_t)ec[4];
        }
        if (ambient_mode && d_rail) {
            std::vector<unsigned long long> rail_host(cons.size(), 0);
            cuda_check(cudaMemcpy(rail_host.data(), d_rail,
                                  cons.size() * sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost), "D2H rail");
            for (size_t k = 0; k < cons.size(); ++k)
                solver.boundary_flux_[cons[k]] = (int64_t)rail_host[k];
        }
        free_all();
    } catch (...) {
        free_all();
        throw;
    }

    // step()'s last-substep digests, byte-for-byte (bulk flux never touches
    // u/T and zero-u-on-solid is idempotent post-advection, so the post-loop
    // fields ARE the last-substep checkpoint fields — the P6.2/P6.1-gated
    // property).
    solver.digest_advect = digest_of_host(wind_x, n,
                           digest_of_host(wind_y, n,
                           digest_of_host(temperature, n, 0)));
    {
        uint64_t bfd = 0;
        for (int gi = 0; gi < n_gases; ++gi)
            bfd = digest_of_host(gas + (size_t)gi * n, n, bfd);
        solver.digest_bulk_flux = bfd;
    }

    // DEBUG probe parity: T after the SL substep loop.
    if (solver.dbg_probe_idx >= 0 && solver.dbg_probe_idx < n)
        solver.dbg_T_post_advect = temperature[solver.dbg_probe_idx];

    // ======================================================================
    // HOST MID-STAGE — step()'s div(u*), step-2 Dalton sum and p*, VERBATIM
    // (their outputs must be host buffers: mg_build_levels' contract).
    // ======================================================================
    std::vector<int32_t> div_u(n, 0);
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            // BC (audit (b)): the ring is a Dirichlet boundary — div(u*)=0.
            if (solid[i] || is_vacuum[i] || (ambient_mode && is_ambient[i])) { div_u[i] = 0; continue; }
            const int il = mirror_idx_host(i, y, x - 1, h, w, solid);
            const int ir = mirror_idx_host(i, y, x + 1, h, w, solid);
            const int iu = mirror_idx_host(i, y - 1, x, h, w, solid);
            const int id = mirror_idx_host(i, y + 1, x, h, w, solid);
            const q16 dux = mul_q16(wind_x[ir] - wind_x[il], inv_2dx_q);
            const q16 duy = mul_q16(wind_y[id] - wind_y[iu], inv_2dx_q);
            div_u[i] = dux + duy;
        }
    }

    // step 2's Dalton sum (post-substep N — the same n_total scratch reused,
    // exactly like the CPU's member cache). P-T0 (design §2.6): n_total ≡
    // n_bulk; trace planes skipped outright.
    {
        for (int i = 0; i < n; ++i) n_total[i] = 0;
        for (int gi = 0; gi < n_gases; ++gi) {
            if (!gas_conservative[gi]) continue;
            const int32_t* plane = gas + (size_t)gi * n;
            for (int i = 0; i < n; ++i) n_total[i] += plane[i];
        }
    }
    std::vector<int32_t> pstar(n, 0);
    for (int i = 0; i < n; ++i) {
        if (solid[i] || is_vacuum[i]) { pstar[i] = 0; continue; }
        if (solver.debug_pstar_from_prev) {
            pstar[i] = p_prev[i];   // MEASUREMENT-ONLY diagnostic (parity)
        } else {
            const int64_t t_abs_wide = (((int64_t)s_eos_q * (int64_t)temperature[i]) >> 16) + (int64_t)t_amb_q;
            const q16 t_abs = (q16)t_abs_wide;
            const q16 cn = mul_q16(c_q, n_total[i]);
            pstar[i] = mul_q16(cn, t_abs);
        }
        if (pstar[i] < 0) pstar[i] = 0;   // EOS floor
    }
    solver.digest_pstar = digest_of_host(pstar.data(), n, 0);

    // ======================================================================
    // 3. PRESSURE SOLVE — host-built hierarchy (the SAME mg_build_levels the
    //    CPU calls, review §2.7) + the ENTIRE iteration on device (P6.3).
    // ======================================================================
    // BC: the SHIFT (rhs − P_amb, warm-start re-shift), the ring→Dirichlet excl,
    // and the σ-diagonal all live in the SHARED host-side mg_build_levels —
    // forwarding the ambient args here makes the device solve inherit them.
    const int n_levels = solver.mg_build_levels(
        pstar.data(), div_u.data(), n_total.data(), p_prev,
        solid, is_vacuum, dyn_permeability, h, w, dt,
        ambient_mode ? is_ambient : nullptr, p_amb,
        ambient_mode ? sponge_sigma : nullptr);
    std::vector<int32_t> p_new(n, 0);
    {
        const auto& L = solver.mg_levels();
        std::vector<MGLevelHostView> views(n_levels);
        for (int lv = 0; lv < n_levels; ++lv) {
            views[lv].h = L[lv].h;
            views[lv].w = L[lv].w;
            views[lv].excl  = L[lv].excl.data();
            views[lv].m     = L[lv].m.data();
            views[lv].gE    = L[lv].gE.data();
            views[lv].gS    = L[lv].gS.data();
            views[lv].recip = L[lv].recip.data();
            views[lv].b     = L[lv].b.data();
            views[lv].P     = L[lv].P.data();
        }
        solver.digest_helmholtz = eos_mg_vcycle(
            views.data(), n_levels, solver.use_multigrid,
            solver.mg_cycles, solver.mg_nu1, solver.mg_nu2,
            solver.mg_coarsest_sweeps, solver.S,
            p_new.data(), nullptr, nullptr);
    }

    // ======================================================================
    // 4 + 4c. KICK + COMPRESSION WORK — the proven P6.4 entry on the solved
    //    P (p_new == the zeroed level-0 P == step 5's atmosphere bytes);
    //    per-call rail counters accumulated into the solver's cumulative
    //    members exactly as step() increments them.
    // ======================================================================
    {
        uint64_t dig_vel = 0, dig_comp = 0;
        int64_t cnts[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        eos_kick_compression(
            wind_x, wind_y, temperature, p_new.data(),
            gas, gas_conservative, n_gases, solid, is_vacuum,
            dyn_wave_absorb, h, w, dt, pre.cap2.data(),   // D2v2
            solver.c_max, solver.dx, solver.adiabatic_index,
            solver.absorb_strength, solver.N_FLOOR_SOLVER, solver.T_MIN,
            solver.T_WORK_CLAMP, solver.T_MAX_PHYS, solver.U_MAX,
            // P-E3 (design §2.8): interior drag + heat counterparty.
            // drag-law v2 (docs/drag_law_v2_design_2026-08-23.md): k_drag2.
            solver.k_drag, solver.k_drag2, solver.k_drag_heat_frac, solver.c_v,
            // P-E4 (design §2.4): the compression-work trust gate.
            solver.n_work_ref,
            // T_ABS COMPRESSION WORK (P-W1a, design §5): ambient K.
            solver.T_AMB_K,
            &dig_vel, &dig_comp, cnts,   // trace_mass_scale arg RETIRED (P-T0)
            // BC: ring velocity zero + compression skip + the u-damping band.
            ambient_mode ? is_ambient : nullptr,
            ambient_mode ? sponge_udamp : nullptr,
            // THERMAL-MASS AXIS: step 4c skips its T write on thermal_solid.
            thermal_solid);
        solver.digest_velocity    = dig_vel;
        solver.digest_compression = dig_comp;
        solver.u_clamp_hits      += cnts[0];
        solver.u_max_hits        += cnts[1];
        solver.work_clamp_hits   += cnts[2];
        solver.energy_floor_hits += cnts[3];
        solver.t_max_phys_hits   += cnts[4];
        // P-E3: PER-TICK semantics (assigned, not accumulated — the P-E1
        // reset-at-step()-entry idiom; this dispatch runs once per tick).
        solver.ke_drag_removed     = cnts[5];
        solver.e_drag_deposit      = cnts[6];
        solver.e_drag_drop_sum     = cnts[7];
        solver.e_drag_rail_clipped = cnts[8];
    }

    // DEBUG probe parity: T after step 4c (compression work).
    if (solver.dbg_probe_idx >= 0 && solver.dbg_probe_idx < n)
        solver.dbg_T_post_compression = temperature[solver.dbg_probe_idx];

    // ======================================================================
    // 5. P := P_new — materialized ONCE (the `atmosphere` alias).
    // ======================================================================
    // BC (spec §1 "the shift trick"): the solve ran in P′ = P − P_amb, so add
    // P_amb back — MASKED to !solid (solids stay 0 absolute; ring cells (excl==1,
    // P′=0) and every regular cell get the add). Mirrors EOSSolver::step's
    // branch-gated store; space maps take the untouched byte-identical path.
    if (ambient_mode) {
        const int64_t pa = (int64_t)p_amb;
        for (int i = 0; i < n; ++i)
            atmosphere[i] = solid[i] ? p_new[i] : (int32_t)((int64_t)p_new[i] + pa);
    } else {
        for (int i = 0; i < n; ++i) atmosphere[i] = p_new[i];
    }
}

}  // namespace breach_cuda
