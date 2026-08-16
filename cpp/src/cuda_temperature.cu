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
#include "temperature_solver.h"       // P-E2a: the SHARED conduction energy kit
                                       // (conduction::cell_capacity_q /
                                       // face_energy_q / opposite_dir) — one
                                       // transcription, both backends.
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

// ---- P-E2a: the energy-counter slot map (design §2.3) -----------------------
// One 6-slot int64 block, atomicAdd'd (order-free on two's complement, which is
// what makes a VALUE sum legal here at all). The host folds it into the CPU
// solver's own fields so telemetry is identical whichever backend ran. Slot
// order is pinned and mirrored by cuda_temperature.h / bindings.cpp.
enum : int {
    C_COND_TRUNC = 0,   // e_cond_trunc_sum   (endpoint floordiv residual, ≤ 0)
    C_COND_CAP   = 1,   // e_cond_cap_sum     (capacity floor/ceiling, signed)
    C_LIMIT_HITS = 2,   // cond_limit_hits    (constraint-4 engagements)
    C_COOL       = 3,   // e_cool_sum         (Pass 3 / sky, SIGNED)
    C_VAC_WIPE   = 4,   // e_vac_wipe_sum     (Pass 0a breach wipe, SIGNED)
    C_RING_PIN   = 5,   // e_ring_pin_sum     (Pass 0a ring pin, SIGNED)
    C_SLOTS      = 6
};

__device__ __forceinline__ void cadd(unsigned long long* c, int slot, int64_t v) {
    if (v != 0) atomicAdd(&c[slot], (unsigned long long)v);
}

// ---- P-E2a: the capacity planes (the CPU's pre-pass build, verbatim) --------
// Depends only on FROZEN inputs (medium mask, N, the two dials) — never on T —
// so one kernel ahead of every pass is the exact device twin of the CPU's
// once-per-step loop. `cap_used` is the divisor of record; `cap_real` prices
// the counters.
__global__ void temp_cap_build(int64_t* __restrict__ cap_used,
                               int64_t* __restrict__ cap_real,
                               const bool* __restrict__ thermal_solid,
                               const int32_t* __restrict__ heat_inv_shift,
                               const int32_t* __restrict__ n_src,
                               int32_t n_floor_q, int32_t c_v_q, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        int64_t cu = 0, cr = 0;
        conduction::cell_capacity_q(thermal_solid[i], heat_inv_shift[i],
                                    n_src[i], n_floor_q, c_v_q, &cu, &cr);
        cap_used[i] = cu;
        cap_real[i] = cr;
    }
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
                                 const bool* __restrict__ is_ambient,
                                 const int64_t* __restrict__ cap_real,
                                 unsigned long long* __restrict__ cnt) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        // BC (audit (b)): the ambient ring radiates to the T_amb sky — wiped to
        // ΔT=0 exactly like a vacuum breach (is_ambient nullptr on space maps).
        if ((is_vacuum[i] || (is_ambient && is_ambient[i])) && !thermal_solid[i]) {
            // P-E2a (L3-6): law unchanged, both wipes named as SIGNED channels
            // (they CREATE whenever they pin a sub-ambient cell up to 0). The
            // vacuum-before-ring attribution order is the CPU block's, pinned.
            const int32_t t_old = temperature[i];
            if (t_old != 0)
                cadd(cnt, is_vacuum[i] ? C_VAC_WIPE : C_RING_PIN,
                     -(int64_t)t_old * cap_real[i]);
            temperature[i] = 0;
        }
    }
}
// ---- Pass 0b gas-T SL advection (`gas_wall_at`, `gas_backtrace_sample_q_dev`,
// `temp_advect`) — DELETED at P-E1 (energy-books arc, design §2.1.1; round-1
// finding L3-5), IDENTICALLY to the CPU twin in temperature_solver.cpp. This was
// the engine's second semi-Lagrangian T-COPIER — a temperature copy onto mass it
// never paid for — live on this backend only because the caller happens to pass
// null winds. Gas temperature is now transported once and conservatively by the
// EOS energy books. MEDIUM-TEST SITES 2/6, 3/6 and 4/6 lived here and retire
// with it; 1/6, 5/6 and 6/6 are untouched and still marked below.

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
                                     const int32_t* __restrict__ rad_net,
                                     int64_t recip_cv, int32_t n_floor_q,
                                     int32_t t_max_phys_q,
                                     unsigned long long* __restrict__ hits,
                                     unsigned long long* __restrict__ low_hits,
                                     int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        // ---- P-R4 SIGNED radiation fold — the CPU block verbatim ----------
        // FIRST, and NOT gated by the `deposit <= 0` skip below (that skip
        // would swallow every radiative loss). shr_round0 = symmetric
        // round-toward-0; sat_add_q16 = the SIGNED saturating add (a positive-
        // only add would drop the losses). Order pinned: radiation, then heat.
        if (rad_net != nullptr && thermal_solid[i]) {
            const int32_t rn = rad_net[i];
            if (rn != 0) {
                int32_t tr = temperature[i];
                const int32_t dTr = shr_round0(rn, heat_inv_shift[i]);
                tr = sat_add_q16(tr, dTr);
                if (tr > t_max_phys_q) { tr = t_max_phys_q; atomicAdd(hits, 1ULL); }
                // P-F1a (v7.2): the LOW rail — the CPU block verbatim. The
                // radiation fold is the only SIGNED path into `temperature`;
                // 0 is the ambient floor. Counted, and required INERT in every
                // gate scenario (the budget argument, see the CPU comment).
                if (tr < 0) { tr = 0; atomicAdd(low_hits, 1ULL); }
                temperature[i] = tr;
            }
        }
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

// ---- Pass 2: conduction — ENERGY FORM (P-E2a, design §2.3) -----------------
// The CPU loop (temperature_solver.cpp Pass 2) transcribed body-for-body, over
// the SHARED `conduction::` kit so neither backend can carry its own copy of
// the law. Reads the FROZEN temperature + the frozen capacity planes, writes
// temp_new[i] and nothing else — still a single-writer gather, still no
// atomics for the physics itself (only the VALUE-SUM counters atomicAdd, which
// is order-free on two's complement). Every cell is fully written (a cell with
// no live face takes the ΔE == 0 early-out and copies ti through), so temp_new
// has no uninitialised read.
__global__ void temp_conduct(const int32_t* __restrict__ temperature,
                             int32_t* __restrict__ temp_new,
                             const int32_t* __restrict__ face_shift,
                             const int64_t* __restrict__ cap_used,
                             const int64_t* __restrict__ cap_real,
                             unsigned long long* __restrict__ cnt,
                             int no_face, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int y = i / w;
        const int x = i % w;
        const int32_t* fs = &face_shift[i * 4];
        const int64_t ti = (int64_t)temperature[i];
        const int64_t cap_i = cap_used[i];
        int64_t de = 0;
        int64_t lim_hits = 0;          // thread-local; folded once, below
        for (int d = 0; d < 4; ++d) {
            const int s_i = fs[d];
            if (s_i == no_face) continue;
            const int ny = y + dy_of(d);
            const int nx = x + dx_of(d);
            if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
            const int j = ny * w + nx;
            // The neighbour's facing entry — the face is skipped and rated
            // identically from both ends BY CONSTRUCTION (CPU comment).
            const int s_j = face_shift[j * 4 + conduction::opposite_dir(d)];
            if (s_j == no_face) continue;
            const int s = (s_i > s_j) ? s_i : s_j;
            de += conduction::face_energy_q(ti, (int64_t)temperature[j],
                                            cap_i, cap_used[j], s, &lim_hits);
        }
        cadd(cnt, C_LIMIT_HITS, lim_hits);
        if (de == 0) {                                  // exact rest
            temp_new[i] = (int32_t)ti;
            continue;
        }
        const int64_t dT = fixedpoint::floordiv_q(de, cap_i);
        cadd(cnt, C_COND_TRUNC, dT * cap_i - de);
        cadd(cnt, C_COND_CAP,   dT * (cap_real[i] - cap_i));
        temp_new[i] = (int32_t)(ti + dT);
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
                          const int64_t* __restrict__ cap_real,
                          unsigned long long* __restrict__ cnt,
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
        // P-E2a (L3-6): law unchanged; Pass 3 is a SIGNED channel — it relaxes
        // toward 0 from BOTH sides, so on a sub-ambient tile it CREATES.
        cadd(cnt, C_COOL, -(int64_t)loss * cap_real[i]);
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
    int cool_shift_floor,       // low clamp on the vacuum offset (== SHIFT_MIN)
    int64_t* low_rail_hits_out, // P-F1a: Pass-1 LOW rail count (nullable)
    const int32_t* rad_net,     // P-R4: SIGNED radiation accumulator (nullable)
    int64_t* energy_counters_out) {  // P-E2a: 6 slots (C_* enum), nullable;
                                      // accumulated (+=) into the caller's
                                      // TemperatureSolver fields
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
    // P-E2a: c_v as a Q16.16 MULTIPLIER for the conduction capacity (Pass 1
    // needs its reciprocal; the capacity needs the value). Same dial, same
    // once-per-step boundary cast the CPU does.
    const int32_t c_v_q = quantize(c_v_safe);
    // P-E1: Pass 0b (gas-T SL advection) is RETIRED — `wind_x`/`wind_y`/`dt`
    // and `gas_advection_rate` survive only as inert back-compat surface, and
    // nothing on this backend reads them any more (CPU twin identical).
    (void)wind_x; (void)wind_y; (void)gas_advection_rate;
    // COOL-SHIFT AXIS: the vacuum discount as a DIFFERENCE, computed ONCE on
    // the host exactly as the CPU solver's Pass 3 does (`const int vac_offset =
    // cool_shift - cool_shift_vacuum;`). Pure integer, no boundary cast.
    const int vac_offset = cool_shift - cool_shift_vacuum;

    const size_t nb = (size_t)n * sizeof(int32_t);
    const size_t nbool = (size_t)n * sizeof(bool);
    int32_t *d_temp = nullptr, *d_temp_new = nullptr, *d_heat = nullptr,
            *d_his = nullptr, *d_fs = nullptr, *d_atm = nullptr,
            *d_nbulk = nullptr,
            *d_csg = nullptr;
    bool *d_solid = nullptr, *d_vac = nullptr, *d_tsol = nullptr;
    unsigned long long* d_hits = nullptr;
    unsigned long long* d_low_hits = nullptr;   // P-F1a: LOW rail count
    // P-E2a: the two capacity planes + the 6-slot energy counter block.
    int64_t *d_cap_used = nullptr, *d_cap_real = nullptr;
    unsigned long long* d_cnt = nullptr;

    cuda_check(cudaMalloc(&d_temp, nb), "malloc temp");
    cuda_check(cudaMalloc(&d_temp_new, nb), "malloc temp_new");
    cuda_check(cudaMalloc(&d_heat, nb), "malloc heat");
    cuda_check(cudaMalloc(&d_his, nb), "malloc heat_inv_shift");
    cuda_check(cudaMalloc(&d_fs, nb * 4), "malloc face_shift");
    cuda_check(cudaMalloc(&d_atm, nb), "malloc atmosphere");
    cuda_check(cudaMalloc(&d_solid, nbool), "malloc solid");
    cuda_check(cudaMalloc(&d_vac, nbool), "malloc is_vacuum");
    cuda_check(cudaMalloc(&d_hits, sizeof(unsigned long long)), "malloc hits");
    cuda_check(cudaMalloc(&d_low_hits, sizeof(unsigned long long)), "malloc low_hits");
    cuda_check(cudaMalloc(&d_cap_used, (size_t)n * sizeof(int64_t)), "malloc cap_used");
    cuda_check(cudaMalloc(&d_cap_real, (size_t)n * sizeof(int64_t)), "malloc cap_real");
    cuda_check(cudaMalloc(&d_cnt, C_SLOTS * sizeof(unsigned long long)), "malloc cnt");
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
    // P-R4: same nullable-plane idiom — with nullptr the kernel is handed a
    // null pointer and skips the fold, the exact CPU twin.
    int32_t* d_radnet = nullptr;
    if (rad_net) cuda_check(cudaMalloc(&d_radnet, nb), "malloc rad_net");

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
    if (rad_net)
        cuda_check(cudaMemcpy(d_radnet, rad_net, nb, cudaMemcpyHostToDevice),
                   "H2D rad_net");
    // BC: optional ambient ring mask for the Pass-0 wipe (nullptr on space maps).
    bool* d_amb = nullptr;
    if (is_ambient) {
        cuda_check(cudaMalloc(&d_amb, nbool), "malloc is_ambient");
        cuda_check(cudaMemcpy(d_amb, is_ambient, nbool, cudaMemcpyHostToDevice), "H2D is_ambient");
    }
    if (n_bulk) cuda_check(cudaMemcpy(d_nbulk, n_bulk, nb, cudaMemcpyHostToDevice), "H2D nbulk");
    cuda_check(cudaMemset(d_hits, 0, sizeof(unsigned long long)), "memset hits");
    cuda_check(cudaMemset(d_low_hits, 0, sizeof(unsigned long long)), "memset low_hits");
    cuda_check(cudaMemset(d_cnt, 0, C_SLOTS * sizeof(unsigned long long)), "memset cnt");

    // The N divisor source Pass 1 reads: n_bulk when supplied, else the atmosphere
    // density proxy — EXACTLY the CPU's `n_bulk ? n_bulk[i] : atmosphere[i]`.
    const int32_t* d_nsrc = n_bulk ? d_nbulk : d_atm;

    // THERMAL-MASS AXIS: the mask the SIX medium tests read — the exact device
    // twin of the CPU solver's `const bool* ts = thermal_solid ? thermal_solid
    // : solid`. Every kernel below takes `d_ts`, never `d_solid`.
    const bool* d_ts = thermal_solid ? (const bool*)d_tsol : (const bool*)d_solid;

    const int block = 256;
    const int grid = (n + block - 1) / block;

    // P-E2a: the capacity planes, built ONCE ahead of every pass (frozen
    // inputs only) — the exact device twin of the CPU's pre-pass loop.
    temp_cap_build<<<grid, block>>>(d_cap_used, d_cap_real, d_ts, d_his,
                                    d_nsrc, n_floor_q, c_v_q, n);
    cuda_check(cudaGetLastError(), "cap_build launch");

    // Pass 0a: zero gas-T at open vacuum cells (unconditional, in-place on d_temp).
    temp_zero_vacuum<<<grid, block>>>(d_temp, d_ts, d_vac, n, d_amb,
                                      d_cap_real, d_cnt);
    cuda_check(cudaGetLastError(), "zero_vacuum launch");

    // (Pass 0b — gas-T SL advection — RETIRED at P-E1; see the file header.)

    // Pass 1: unified convert (in-place on d_temp; rail counter -> d_hits).
    temp_convert_unified<<<grid, block>>>(d_temp, d_heat, d_his, d_ts, d_vac,
                                          d_nsrc, d_radnet, recip_cv, n_floor_q,
                                          t_max_phys_q, d_hits, d_low_hits, n);
    cuda_check(cudaGetLastError(), "convert launch");

    // Pass 2: conduct (d_temp -> d_temp_new), then copy back (the CPU swap).
    temp_conduct<<<grid, block>>>(d_temp, d_temp_new, d_fs, d_cap_used,
                                  d_cap_real, d_cnt, no_face, h, w);
    cuda_check(cudaGetLastError(), "conduct launch");
    cuda_check(cudaMemcpy(d_temp, d_temp_new, nb, cudaMemcpyDeviceToDevice), "D2D swap");

    // Pass 3: cool (in-place on d_temp).
    temp_cool<<<grid, block>>>(d_temp, d_ts, d_vac, d_atm, d_csg,
                               d_cap_real, d_cnt,
                               cool_shift, vac_offset, cool_shift_floor,
                               thresh_q, h, w);
    cuda_check(cudaGetLastError(), "cool launch");
    cuda_check(cudaDeviceSynchronize(), "sync");

    unsigned long long hits = 0, low_hits = 0;
    cuda_check(cudaMemcpy(&hits, d_hits, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost), "D2H hits");
    cuda_check(cudaMemcpy(&low_hits, d_low_hits, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost), "D2H low_hits");
    if (low_rail_hits_out) *low_rail_hits_out += (int64_t)low_hits;
    // P-E2a: fold the 6-slot energy block into the caller's accumulators.
    // Two's-complement round-trip through unsigned long long is exact.
    {
        unsigned long long cnt_h[C_SLOTS] = {0, 0, 0, 0, 0, 0};
        cuda_check(cudaMemcpy(cnt_h, d_cnt, C_SLOTS * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost), "D2H cnt");
        if (energy_counters_out) {
            for (int k = 0; k < C_SLOTS; ++k)
                energy_counters_out[k] += (int64_t)cnt_h[k];
        }
    }
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
    cudaFree(d_low_hits);
    cudaFree(d_cap_used);
    cudaFree(d_cap_real);
    cudaFree(d_cnt);
    if (d_nbulk) cudaFree(d_nbulk);
    if (d_amb) cudaFree(d_amb);
    if (d_tsol) cudaFree(d_tsol);
    if (d_csg) cudaFree(d_csg);
    if (d_radnet) cudaFree(d_radnet);

    return (int64_t)hits;
}

namespace {
bool g_temp_backend_cuda = false;
}
bool temperature_backend_is_cuda() { return g_temp_backend_cuda; }
void set_temperature_backend_cuda(bool on) { g_temp_backend_cuda = on; }

}  // namespace breach_cuda
