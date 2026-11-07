// ============================================================================
// CUDA-S6 fire solver implementation — see cuda_fire.h.
// A bit-identical GPU port of FireSimulation::step (fire_simulation.cpp ~44-301),
// RE-DERIVED for the EOS refactor (P6.8): the O2 gate reads the REAL bulk O2
// density plane `n_o2` (not the old atmosphere/P proxy). The own-tile plume->T
// shim (formerly P3 below) was DELETED at P-R2 (docs/radiation_raycaster_
// extinction_ruling_2026-07-31.md A2) — it was the one `temperature[]` writer
// bypassing heat_inv_shift; TemperatureSolver is now the field's ONLY writer,
// and P-R4's radiation pass will be the next writer, through its own rad_net[]
// plane. Pipeline: per-tile signed-logistic intensity feedback → wall
// burn-through → final clamp. (Smoke emission into neighbours was DELETED at
// P-S1, 2026-08-15 — Erik's single-source ruling, docs/smoke_single_source_
// design_2026-07-24.md: it was an ex-nihilo scatter the storm audit traced
// into a real mass/pressure pump, docs/storm_audit_2026-08-14.md §4.2.
// Combustion soot is now the ONE fire-smoke source, cpp/src/combustion.cpp.)
//
// THREE device passes (P2, P5-P6 — P3's slot retired at P-R2, P4's at P-S1,
// NEITHER renumbered, so the numbering stays anchored to the CPU step's
// history), one per CPU pass, launched as a barriered chain (separate
// launches = grid barriers between dependent passes). P1 (the host max
// early-exit) is done on the HOST before any launch:
//   P1  early-exit         HOST *max_element(fire) < thresh -> return {} (untouched)
//   P2  logistic feedback  fire += dt*(grow-die); snap-extinguish (own-tile; O2 gate
//                          reads n_o2 neighbour mean — read-only field)
//   P5  wall burn-through  wall_hp[i] -= dmg; collect destroyed; fire[i]=0
//   P6  final clamp        fire -> [0, FP_ONE]                      (own-tile;
//                          `smoke` is neither read nor written here: the
//                          smoke clamp was DELETED with smoke transport v2,
//                          #12 design §2.4 — see fire_clamp below)
//
// Pass order P2 → P5 → P6 matters: P5 reads the P2-updated `fire` (frozen
// between launches) and zeroes it on destroyed cells; P6 clamps last. P2
// reads temperature (the `hot` gate) only — no pass writes it anymore this
// patch (temperature is copied through unchanged, same as the CPU side). NO
// cross-cell within-pass dependence exists (every pass is an own-index write
// with read-only neighbour reads or an order-free counter scatter), so this
// parallel schedule reproduces the CPU sequential result bit-for-bit — there
// is NO combustion-style Gauss-Seidel coupling here.
// Host scalar precompute mirrors the CPU load-time block exactly — all config
// constants quantized once in double, many via load-time make_recip/recip_mul
// (NOT per-cell reciprocal_q16).
//
// Every per-cell op is a VERBATIM device transcription of the CPU loops — same
// integer ops, same PINNED left-fold mul_q16 tree, same branch structure. The
// new transcendental is sqrt_q16_dev (the device floor-isqrt, bit-identical to
// fixedpoint::sqrt_q16). Continuous-O2 law (docs/continuous_o2_law_design_2026-
// 07-24.md): the O2 gate reads a per-cell MOLE FRACTION (Σn_o2/Σn_total over
// open neighbours) via reciprocal_q16_dev, NOT fixedpoint::mean_round on
// absolute density (RETIRED from this gate). The make_recip reciprocals use
// recip_mul_dev (the device 128-bit path).
//
// THE FORMER DETERMINISM CRUX (P4, RETIRED at P-S1): this used to be the one
// scatter in the fire kernel — 4 smoke emissions per source thread deposited
// with integer atomicAdd, proven order-free because the deposit depended only
// on fire[src], never on a neighbour's current smoke. With P4 deleted, P5's
// destroyed-index counter below is the ONLY remaining scatter in this kernel.
//
// P5 destroyed-list collection: a device int counter (atomicAdd for a slot) + a
// device array of packed indices (sized n, worst case every cell destroyed); copy
// counter + array back to the host, SORT THE PACKED INDICES, and build the
// std::vector<pair> in row-major order — the CPU's order.
//
// NOT "any order" (corrected by audit Patch A / A5, 2026-08-04). The atomicAdd
// slot assignment is arbitrary and varies run-to-run, and the consumer
// GameMap.destroy_wall is order-dependent, so the unsorted list was a real
// CPU!=GPU and GPU!=GPU divergence. Measured on this branch before the fix, in
// the gate's OWN scenario: 4 walls destroyed in tick 1, CPU
// [(6,9),(9,6),(10,13),(13,10)] vs GPU [(13,10),(6,9),(9,6),(10,13)]. The gate
// checked SET equality and so was structurally blind to it; it now compares
// LISTS (tests/cuda_fire_check.py) — order, drops and dupes all caught.
// ============================================================================
#include "cuda_fire.h"
#include "cuda_resident.h"   // #29 P4a: fire_launch_resident, FireScalars
#include "fixed_point.h"   // q16, quantize, mul_q16, mul_wide, narrow_round, FP_ONE, FP_SHIFT, make_recip, mean_round
#include "cuda_fixedpoint_device.cuh"  // sqrt_q16_dev, recip_mul_dev (S6 §2 shared kit)
#include "o2_pressure_factor.h"        // issue #7: the SAME FP_HD g(p) the CPU step calls

#include <cuda_runtime.h>

#include <algorithm>   // std::max_element (the host P1 early-exit reduction)
#include <sstream>
#include <stdexcept>

using namespace fixedpoint;

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in fire_step/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// 4-connected neighbour offsets (the open-neighbour pressure mean + smoke spread).
__device__ __constant__ int D4_dy[4] = {-1, 1, 0, 0};
__device__ __constant__ int D4_dx[4] = { 0, 0, -1, 1};

// Integer clamp to [0, FP_ONE] (the [0,1] tracer saturation, exact). Verbatim of
// the CPU clamp01_q (fire_simulation.cpp:17-21).
__device__ __forceinline__ q16 clamp01_q_dev(q16 v) {
    if (v < 0) return 0;
    if (v > FP_ONE) return FP_ONE;
    return v;
}

// Integer clamp to [0, cap] (R1 renormalized o2f — VERBATIM device port of the
// CPU clamp0cap_q, fire_simulation.cpp).
__device__ __forceinline__ q16 clamp0cap_q_dev(q16 v, q16 cap) {
    if (v < 0) return 0;
    if (v > cap) return cap;
    return v;
}

// Hermite smoothstep on [edge0, edge1] -> [0, FP_ONE], clamped outside, in Q16.16.
// VERBATIM device port of the CPU smoothstep_q (fire_simulation.cpp:35-42): the
// PINNED multiply tree t = clamp01((x-edge0)*recip_span); t2=t*t;
// three_minus = 3 - 2t; return t2 * three_minus. recip_span is a load-time
// make_recip reciprocal -> recip_mul_dev on the device. edge1<=edge0 -> a step.
// RETAINED past the continuous-O2 law (which replaced this gate's use of it,
// same as the CPU tombstone) — kept for the tombstone + any future consumer;
// no longer called by fire_logistic below.
[[maybe_unused]] __device__ __forceinline__ q16 smoothstep_q_dev(
        q16 edge0, q16 edge1, q16 x, int64_t recip_span, bool degenerate) {
    if (degenerate) return (x < edge0) ? 0 : FP_ONE;
    const q16 t = clamp01_q_dev(recip_mul_dev(x - edge0, recip_span));
    const q16 t2 = mul_q16(t, t);                              // t*t
    const q16 three_minus = (q16)(3 * FP_ONE) - (q16)(t << 1); // 3 - 2t
    return mul_q16(t2, three_minus);                           // t^2 * (3 - 2t)
}

// ---- P2: per-tile signed-logistic feedback (fire_simulation.cpp ~111-192) ----
// Each flammable lit tile reads its own intensity I, temperature T, fuel F (from
// wall_hp), neighbour pressure P (the O2 proxy, 4-nbr open mean), wind magnitude
// W (sqrt of wind²), then steps I by dt*(grow - die) with the PINNED left-fold
// mul_q16 tree, clamps, and snap-extinguishes below I_min. Own-cell write to fire.
// All scalar dials arrive as host-precomputed Q16.16 / make_recip args.
__global__ void fire_logistic(int32_t* __restrict__ fire,
                              const int32_t* __restrict__ atmosphere,  // #7 (nullable)
                              // #29 P4a: the O2 plane is plane `o2_idx` of the
                              // env's (n_gases, h, w) gas block
                              const int32_t* __restrict__ gas, int n_gases, int o2_idx,
                              const int32_t* __restrict__ n_total,
                              const int32_t* __restrict__ wall_hp,
                              const int32_t* __restrict__ temperature,
                              const int32_t* __restrict__ wind_x,
                              const int32_t* __restrict__ wind_y,
                              const bool* __restrict__ is_wall,
                              const bool* __restrict__ is_vacuum,
                              const bool* __restrict__ flammable,
                              const int64_t* __restrict__ fuel_recip,
                              const int32_t* __restrict__ fire_T_ext_plane,
                              int h, int w, int n_env,
                              const int32_t* __restrict__ d_dt_q,     // (N,) per env
                              const int32_t* __restrict__ d_gate,     // (N,) nullable
                              int32_t gate_thresh_q,
                              int32_t k_grow_q, int32_t k_die_q,
                              int32_t k_wind_fan_q, int32_t k_wind_strip_q,
                              int32_t fire_T_ext_q, int32_t x_ext_q,
                              int32_t X_N_FLOOR, int32_t I_min_q, int32_t INV_C,
                              bool temp_is_identity, int64_t recip_temp_scale,
                              int64_t recip_fuel_ref, int64_t recip_T_span,
                              int64_t recip_x_span, bool x_degenerate,
                              int32_t o2f_cap_q,                      // R1
                              o2_pressure::Factor pf) {               // #7
    const int n = h * w;
    const int total = n_env * n;
    const bool p_on = (atmosphere != nullptr);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        // #29 P4a: (N, h, w) -- the env, its base cell and (y, x) inside it.
        const int env = i / n;
        // The host early-exit (P1), as a per-env device gate: an env whose
        // max(fire) is below the threshold steps nothing (fields untouched).
        if (d_gate != nullptr && d_gate[env] < gate_thresh_q) continue;
        if (!flammable[i]) continue;        // only fuel burns
        const q16 I = fire[i];
        if (I <= 0) continue;               // unlit -> nothing to step

        const int base = env * n;
        const int li = i - base;
        const int y = li / w, x = li % w;
        const int32_t* n_o2 = gas + ((size_t)env * n_gases + o2_idx) * n - base;
        const q16 dt_q = d_dt_q[env];

        // T: temperature in game-units. Identity fast-path when temp_scale==FP_ONE.
        const q16 T = temp_is_identity
            ? temperature[i]
            : recip_mul_dev(temperature[i], recip_temp_scale);

        // F: fuel from remaining wall HP, normalised by THIS TILE'S OWN full
        // health (fuel-fraction axis, 2026-07-30), clamped to [0,1]. The
        // per-tile reciprocal comes from the `fuel_recip` plane when supplied,
        // else the scalar recip_fuel_ref fallback — VERBATIM the CPU branch.
        const int64_t recip_fuel = fuel_recip ? fuel_recip[i] : recip_fuel_ref;
        const q16 F = clamp01_q_dev(recip_mul_dev(wall_hp[i], recip_fuel));

        // X: local O2 MOLE FRACTION over OPEN (non-solid, non-vacuum)
        // 4-neighbours — continuous-O2 law (design §2.1). Both sums int64,
        // exact, order-free. No open nbr -> both sums 0 -> den floors -> X = 0.
        // Issue #7 (VERBATIM the CPU gather): the same loop and predicate also
        // sum the materialized pressure and count the open neighbours.
        int64_t sum_o2 = 0;
        int64_t sum_tot = 0;
        int64_t sum_p = 0;
        int64_t n_open = 0;
        for (int d = 0; d < 4; ++d) {
            const int ny = y + D4_dy[d], nx = x + D4_dx[d];
            if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                const int ni = base + ny * w + nx;
                if (!is_wall[ni] && !is_vacuum[ni]) {
                    sum_o2  += (int64_t)n_o2[ni];      // exact, order-free
                    sum_tot += (int64_t)n_total[ni];   // exact, order-free
                    if (p_on) sum_p += (int64_t)atmosphere[ni];   // exact, order-free
                    ++n_open;
                }
            }
        }
        // X = Σn_o2 / max(Σn_total, floor), ONE per-cell reciprocal_q16_dev
        // divide (the SAME primitive combustion's heat deposit uses).
        const q16 den = (sum_tot < (int64_t)X_N_FLOOR) ? X_N_FLOOR : (q16)sum_tot;
        const q16 X = mul_q16((q16)sum_o2, reciprocal_q16_dev(den));

        // W: wind magnitude. Q.32 radicand (wx²+wy²) -> floor-isqrt -> Q16.16.
        const int64_t rad = mul_wide(wind_x[i], wind_x[i])
                          + mul_wide(wind_y[i], wind_y[i]);
        const q16 W = sqrt_q16_dev(rad);

        // Gates. `hot` reads THIS TILE'S OWN extinction temperature (ruling A3
        // ride-along) from the nullable `fire_T_ext_plane`, else the scalar
        // fallback — VERBATIM the CPU branch (fire_simulation.cpp).
        const q16 T_ext_i = fire_T_ext_plane ? fire_T_ext_plane[i] : fire_T_ext_q;
        const q16 hot = clamp01_q_dev(recip_mul_dev(T - T_ext_i, recip_T_span));
        // o2f is LINEAR in X (the continuous-O2 law), R1 RENORMALIZED (VERBATIM
        // the CPU branch): X <= X_ext -> 0 (extinction), X == X_amb -> 1
        // (ambient, by construction), clamped at o2f_cap_q (not FP_ONE). The
        // degenerate span falls back to a step at X_ext (0 or the cap).
        const q16 o2f = x_degenerate
            ? ((X < x_ext_q) ? (q16)0 : o2f_cap_q)
            : clamp0cap_q_dev(recip_mul_dev(X - x_ext_q, recip_x_span), o2f_cap_q);
        // THE PRESSURE FACTOR (issue #7) — VERBATIM the CPU step: g of the
        // open-neighbour mean pressure, through the SAME FP_HD
        // o2_pressure::factor, folded into o2f by ONE mul_q16.
        const q16 p_nbr = p_on ? mean_round(sum_p, n_open) : (q16)0;
        const q16 g = p_on ? o2_pressure::factor(pf, p_nbr) : (q16)FP_ONE;
        const q16 o2f_p = mul_q16(o2f, g);
        const q16 avail = mul_q16(F, o2f_p);

        // THE CAPACITY LAW (P-R3, ruling A3) — VERBATIM the CPU sequence
        // (fire_simulation.cpp): the capacity factor is the SIGNED
        // resource-proportional gap, not the hardwired (1 - I).
        //   gap  = avail*hot - mul_q16(I, INV_C)                  (SIGNED sub)
        //   grow = k_grow; * I; * gap; * wind_fan
        // PINNED left-fold mul_q16, each narrowing once, in this EXACT order —
        // any deviation is a CPU<->CUDA bit divergence, which is gate (d).
        const q16 avail_hot = mul_q16(avail, hot);          // avail*hot (hoisted)
        const q16 gap = avail_hot - mul_q16(I, INV_C);      // - I/c (SIGNED)
        const q16 one_minus_I = (q16)FP_ONE - I;                       // (1 - I) — wind strip only
        const q16 wind_fan = (q16)FP_ONE + mul_q16(k_wind_fan_q, W);   // (1 + k_wind_fan*W)
        q16 grow = k_grow_q;
        grow = mul_q16(grow, I);
        grow = mul_q16(grow, gap);
        grow = mul_q16(grow, wind_fan);

        // die = k_die*max(0, 1-avail*hot)*I + k_wind_strip*W*(1-I)*I. R1 DIE-TERM
        // SIGN FIX (VERBATIM the CPU branch): avail*hot can now exceed 1 (O2
        // enrichment above ambient), which would flip (1-avail*hot) NEGATIVE —
        // an ANTI-DEATH term. max(0,.) floors it: enrichment only helps via
        // grow/I_cap, never subtracts from die.
        const q16 one_minus_ah_raw = (q16)FP_ONE - avail_hot;      // (1 - avail*hot), SIGNED
        const q16 one_minus_ah = max((q16)0, one_minus_ah_raw);    // R1: floored at 0
        q16 die_a = k_die_q;
        die_a = mul_q16(die_a, one_minus_ah);
        die_a = mul_q16(die_a, I);
        q16 die_b = k_wind_strip_q;
        die_b = mul_q16(die_b, W);
        die_b = mul_q16(die_b, one_minus_I);
        die_b = mul_q16(die_b, I);
        const q16 die = die_a + die_b;                      // signed Q16.16 sum

        // I_next = clamp01(I + dt*(grow - die)); snap-extinguish below I_min.
        const q16 delta = mul_q16(dt_q, grow - die);
        q16 I_next = clamp01_q_dev(I + delta);
        if (I_next < I_min_q) I_next = 0;
        fire[i] = I_next;
    }
}

// ---- P4: smoke emission SCATTER — DELETED (P-S1, 2026-08-15) -----------------
// The ex-nihilo `smoke[nbr] += round(emission*dt*fire[src])` atomicAdd scatter
// that lived here is gone (Erik's single-source ruling, docs/smoke_single_
// source_design_2026-07-24.md; killed the smoke->N2 pressure pump measured by
// docs/storm_audit_2026-08-14.md §4.2). Combustion soot (cpp/src/combustion.cpp)
// is now the ONE fire-smoke source, on both CPU and GPU. P4's slot is retired,
// not renumbered — see the file header.

// ---- P5: wall burn-through + destroyed collection (fire_simulation.cpp ~245-256)
// wall_hp[i] -= round_nearest(wall_damage*dt*fire[i]*hotf[i]); if (wall_hp<=0 &&
// flammable && is_wall) collect (via atomicAdd counter -> packed index array) and
// fire[i]=0. Reads the P2-updated fire (frozen since P2). Own-cell wall_hp/fire
// writes; the destroyed slot is the only scatter (a counter atomicAdd, order
// arbitrary -> the gate checks SET equality... actually LIST equality, see the
// host-side sort below).
//
// R3 hot-burns-faster (VERBATIM device port of the CPU destruction-loop patch,
// fire_simulation.cpp): hotf is recomputed HERE (not threaded from P2) for
// locality — the kernel reads the same planes anyway. Reuses P2's
// temperature/fire_T_ext_plane device buffers and host-precomputed scalars —
// no new allocation.
// R4 (Erik's ruling 2026-09-06, VERBATIM twin of the CPU gate): depletion is
// flammable-gated — fire may not destroy (or drain) non-flammable tiles.
__global__ void fire_burn(int32_t* __restrict__ fire,
                          int32_t* __restrict__ wall_hp,
                          const int32_t* __restrict__ temperature,
                          const int32_t* __restrict__ fire_T_ext_plane,
                          const bool* __restrict__ is_wall,
                          const bool* __restrict__ flammable,
                          int* __restrict__ d_counter,          // (N,)
                          int* __restrict__ d_destroyed_idx,    // (N, n)
                          int32_t wall_damage_q,
                          const int32_t* __restrict__ d_dt_q,   // (N,)
                          const int32_t* __restrict__ d_gate,   // (N,) nullable
                          int32_t gate_thresh_q,
                          int32_t fire_T_ext_q, int64_t recip_T_span,
                          int32_t hotf_cap_q,
                          bool temp_is_identity, int64_t recip_temp_scale,
                          int n, int n_env) {
    const int total = n_env * n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        const int env = i / n;
        if (d_gate != nullptr && d_gate[env] < gate_thresh_q) continue;
        const q16 dt_q = d_dt_q[env];
        if (fire[i] > 0 && flammable[i]) {
            const q16 T_ext_i = fire_T_ext_plane ? fire_T_ext_plane[i] : fire_T_ext_q;
            const q16 T_i = temp_is_identity
                ? temperature[i]
                : recip_mul_dev(temperature[i], recip_temp_scale);
            const q16 hotf = clamp0cap_q_dev(
                recip_mul_dev(T_i - T_ext_i, recip_T_span), hotf_cap_q);
            const q16 wd = mul_q16(mul_q16(wall_damage_q, dt_q), hotf);
                                                             // wall_damage*dt*hotf
            const int64_t wide = mul_wide(wd, fire[i]);     // * I (wide for round)
            const q16 dmg = narrow_round(wide);             // >= 0 (positive depletion)
            wall_hp[i] -= dmg;
            if (wall_hp[i] <= 0 && is_wall[i]) {   // flammable holds (loop gate)
                const int slot = atomicAdd(&d_counter[env], 1);   // a unique slot
                // the packed linear index INSIDE the env (y*w + x)
                d_destroyed_idx[(size_t)env * n + slot] = i - env * n;
                fire[i] = 0;
            }
        }
    }
}

// ---- P6: final clamp (fire_simulation.cpp, the final clamp) -------------------
// fire clamps to [0, FP_ONE]. Own-cell, in-place. The SMOKE clamp that stood
// here ([0, FP_ONE] on every tile, every tick) is DELETED, mirroring the CPU
// (smoke transport v2, #12, design §2.4): it cut every compressed tile back to
// 1 each tick, silently undoing the trace's conservation. No trace plane is
// clamped anywhere outside a reader. `smoke` therefore never reaches the
// device from this entry any more (it stays in the signature for back-compat,
// the CPU's `(void)smoke` idiom).
__global__ void fire_clamp(int32_t* __restrict__ fire, int n, int n_env,
                           const int32_t* __restrict__ d_gate, int32_t gate_thresh_q) {
    const int total = n_env * n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        if (d_gate != nullptr && d_gate[i / n] < gate_thresh_q) continue;
        fire[i] = clamp01_q_dev(fire[i]);
    }
}

// ---- #29 P4a: the core's prep (destroyed counters + the gate, zeroed) and
// the gate's reduction. The CPU / per-call P1 early exit is
//     max_fire = 0; for each cell max_fire = max(max_fire, fire[i]);
//     if (max_fire < quantize(0.001)) return {}   (nothing touched)
// -- an order-free integer max starting at 0, which is exactly an atomicMax
// over a slot zeroed first. Every later kernel skips an env whose gate is
// below the threshold, so a gated-off env's fields are untouched, the CPU's
// early return as a per-env mask (RL-batch habits: no host gating).
__global__ void fire_prep(int* __restrict__ d_counter, int32_t* __restrict__ d_gate,
                          int n_env) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n_env;
         e += gridDim.x * blockDim.x) {
        d_counter[e] = 0;
        if (d_gate != nullptr) d_gate[e] = 0;
    }
}

__global__ void fire_gate_max(const int32_t* __restrict__ fire,
                              int32_t* __restrict__ d_gate, int n, int n_env) {
    const int total = n_env * n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        const int32_t f = fire[i];
        if (f > 0) atomicMax(&d_gate[i / n], f);
    }
}

}  // namespace

// ============================================================================
// #29 P4a — THE LAUNCH CORE (cuda_resident.h's contract): launch only, no
// malloc / transfer / sync, (N, h, w)-shaped. fire_bake_scalars is the host
// precompute moved VERBATIM out of fire_step (fire_simulation.cpp's load-time
// block, in double); the per-env dt_q rides a device array.
// ============================================================================
FireScalars fire_bake_scalars(
    float k_grow, float k_die, float fire_T_ext, float fire_T_span,
    float fuel_ref, float o2_frac_ext, float I_min,
    float k_wind_fan, float k_wind_strip, float wall_damage,
    float temp_scale, float I_cap_per_avail,
    float o2_frac_amb, float o2f_cap, float hotf_cap,
    int32_t p_ext_q, int32_t p_full_q) {
    FireScalars sc;
    // P1's threshold (the early exit, host-side or as the device gate).
    sc.max_fire_thresh_q = quantize(0.001);   // ~66 counts
    sc.temp_is_identity = (temp_scale == (float)FP_ONE) ? 1 : 0;
    sc.recip_temp_scale = sc.temp_is_identity ? 0 : make_recip((double)temp_scale);
    sc.k_grow_q       = quantize((double)k_grow);
    sc.k_die_q        = quantize((double)k_die);
    sc.k_wind_fan_q   = quantize((double)k_wind_fan);
    sc.k_wind_strip_q = quantize((double)k_wind_strip);
    sc.fire_T_ext_q   = quantize((double)fire_T_ext);
    sc.I_min_q        = quantize((double)I_min);
    sc.wall_damage_q  = quantize((double)wall_damage);
    // CAPACITY LAW (P-R3, ruling A3): INV_C = 1/I_cap_per_avail, VERBATIM the
    // CPU load-time block (double, then ONE quantize; <= 0 -> 0 == ceiling OFF).
    sc.INV_C = (I_cap_per_avail > 0.0f)
        ? quantize(1.0 / (double)I_cap_per_avail) : (q16)0;
    sc.recip_fuel_ref = make_recip((double)fuel_ref);
    sc.recip_T_span   = make_recip((double)fire_T_span);
    // Continuous-O2 law span, R1 RENORMALIZATION (VERBATIM of fire_simulation.
    // cpp's x_ext_q/x_span/x_degenerate/recip_x_span/o2f_cap_q block): the
    // upper end is the AMBIENT reference o2_frac_amb; the clamp's upper edge
    // the o2f_cap dial.
    sc.x_ext_q = quantize((double)o2_frac_ext);
    const double x_span = (double)o2_frac_amb - (double)o2_frac_ext;
    sc.x_degenerate = (x_span <= 0.0) ? 1 : 0;
    sc.recip_x_span = sc.x_degenerate ? 0 : make_recip(x_span);
    sc.o2f_cap_q    = quantize((double)o2f_cap);
    // R3: the hotf ramp's ceiling, VERBATIM the CPU's hotf_cap_q bake.
    sc.hotf_cap_q   = quantize((double)hotf_cap);
    sc.X_N_FLOOR    = quantize(0.01);   // 655 counts, SAME as CPU
    // THE PRESSURE FACTOR (issue #7): the span reciprocal baked on the HOST by
    // the SAME o2_pressure::bake the CPU step calls (integer reciprocal_q16).
    const o2_pressure::Factor pf = o2_pressure::bake(p_ext_q, p_full_q);
    sc.p_ext_q = pf.p_ext_q;
    sc.p_full_q = pf.p_full_q;
    sc.p_inv_span_q = pf.inv_span_q;
    return sc;
}

int32_t fire_dt_q(float dt) {
    // The step constant dt_q, VERBATIM fire_step's (one quantize of the tick).
    return quantize((double)dt);
}

int fire_launch_resident(
    int n_env, int h, int w,
    int32_t* d_fire, const int32_t* d_atmosphere,
    const int32_t* d_gas, int n_gases, int o2_idx,
    const int32_t* d_n_total, int32_t* d_wall_hp, const int32_t* d_temperature,
    const int32_t* d_wind_x, const int32_t* d_wind_y,
    const bool* d_is_wall, const bool* d_is_vacuum, const bool* d_flammable,
    const int64_t* d_fuel_recip, const int32_t* d_fire_T_ext_plane,
    const int32_t* d_dt_q, const FireScalars& sc,
    int32_t* d_gate, int* d_counter, int* d_destroyed_idx) {
    const int n = h * w;
    if (n_env <= 0 || n <= 0) return 0;
    if (o2_idx < 0 || o2_idx >= n_gases) {
        throw std::invalid_argument("fire_launch_resident: o2_idx outside the gas block");
    }
    const int total = n_env * n;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    o2_pressure::Factor pf;
    pf.p_ext_q = sc.p_ext_q;
    pf.p_full_q = sc.p_full_q;
    pf.inv_span_q = sc.p_inv_span_q;
    int launches = 0;

    // K0: the destroyed counters (and the gate, when one rides), zeroed.
    fire_prep<<<(n_env + block - 1) / block, block>>>(d_counter, d_gate, n_env);
    cuda_check(cudaGetLastError(), "prep launch");
    ++launches;
    // P1 as a device gate: the per-env max(fire), atomicMax from 0.
    if (d_gate != nullptr) {
        fire_gate_max<<<grid, block>>>(d_fire, d_gate, n, n_env);
        cuda_check(cudaGetLastError(), "gate launch");
        ++launches;
    }

    // P2 logistic feedback (in-place on fire; O2 gate reads the O2 plane's
    // neighbour mean; reads wall_hp/temp/wind/masks).
    fire_logistic<<<grid, block>>>(
        d_fire, d_atmosphere, d_gas, n_gases, o2_idx, d_n_total, d_wall_hp,
        d_temperature, d_wind_x, d_wind_y, d_is_wall, d_is_vacuum,
        d_flammable, d_fuel_recip, d_fire_T_ext_plane, h, w, n_env,
        d_dt_q, d_gate, sc.max_fire_thresh_q,
        sc.k_grow_q, sc.k_die_q, sc.k_wind_fan_q, sc.k_wind_strip_q,
        sc.fire_T_ext_q, sc.x_ext_q, sc.X_N_FLOOR, sc.I_min_q, sc.INV_C,
        sc.temp_is_identity != 0, sc.recip_temp_scale,
        sc.recip_fuel_ref, sc.recip_T_span, sc.recip_x_span,
        sc.x_degenerate != 0, sc.o2f_cap_q, pf);
    cuda_check(cudaGetLastError(), "logistic launch");
    ++launches;

    // P4 smoke emission scatter DELETED (P-S1) — see the file header.

    // P5 wall burn-through (in-place on wall_hp; zeroes fire on destroyed
    // AFTER P2 read it; collects the destroyed indices per env).
    fire_burn<<<grid, block>>>(d_fire, d_wall_hp, d_temperature, d_fire_T_ext_plane,
                               d_is_wall, d_flammable, d_counter, d_destroyed_idx,
                               sc.wall_damage_q, d_dt_q, d_gate, sc.max_fire_thresh_q,
                               sc.fire_T_ext_q, sc.recip_T_span, sc.hotf_cap_q,
                               sc.temp_is_identity != 0, sc.recip_temp_scale,
                               n, n_env);
    cuda_check(cudaGetLastError(), "burn launch");
    ++launches;

    // P6 final clamp (in-place on fire).
    fire_clamp<<<grid, block>>>(d_fire, n, n_env, d_gate, sc.max_fire_thresh_q);
    cuda_check(cudaGetLastError(), "clamp launch");
    ++launches;
    return launches;
}

std::vector<std::pair<int, int>> fire_step(
    int32_t* fire, const int32_t* atmosphere, const int32_t* n_o2,
    const int32_t* n_total,
    int32_t* smoke, int32_t* wall_hp, int32_t* temperature,
    const int32_t* wind_x, const int32_t* wind_y,
    const bool* is_wall, const bool* is_vacuum, const bool* flammable,
    int h, int w, float dt,
    float k_grow, float k_die, float fire_T_ext, float fire_T_span,
    float fuel_ref, float o2_frac_ext, float o2_frac_full, float I_min,
    float k_wind_fan, float k_wind_strip,
    float wall_damage,
    float temp_scale, float I_cap_per_avail,
    float o2_frac_amb, float o2f_cap,    // R1 (see header): sustain span upper ref
                                          // + the new enrichment-ceiling dial
    float hotf_cap,                      // R3 (see header): the hotf ramp's ceiling
    const int64_t* fuel_recip,           // FUEL-FRACTION AXIS (nullable, see header)
    const int32_t* fire_T_ext_plane,     // PER-MATERIAL T_ext (nullable, see header)
    int32_t p_ext_q, int32_t p_full_q) { // #7 PRESSURE FACTOR edges (see header)
    // `atmosphere` is READ again since issue #7: the pressure factor's input
    // (it was vestigial from EOS P4 until then). Uploaded below when non-null.
    (void)smoke;          // P-S1: no pass reads or writes it (file header)
    (void)o2_frac_full;   // R1: the sustain law reads o2_frac_amb, not this
    const int n = h * w;
    if (n <= 0) return {};

    // ---- Host scalar precompute — ONE transcription, the core's. ----------
    const FireScalars sc = fire_bake_scalars(
        k_grow, k_die, fire_T_ext, fire_T_span, fuel_ref, o2_frac_ext, I_min,
        k_wind_fan, k_wind_strip, wall_damage, temp_scale, I_cap_per_avail,
        o2_frac_amb, o2f_cap, hotf_cap, p_ext_q, p_full_q);

    // ---- P1: HOST max early-exit (fire_simulation.cpp ~61-66). An ORDER-FREE
    //      integer max reduction; the threshold is a pinned Q16.16 compare. If no
    //      tile is above thresh, the CPU returns {} and leaves ALL fields
    //      UNCHANGED — so we skip every kernel + H2D/D2H and return empty.
    //      (The resident tick runs the same exit as the core's device gate.) --
    q16 max_fire = 0;
    for (int i = 0; i < n; ++i) max_fire = std::max(max_fire, fire[i]);
    if (max_fire < sc.max_fire_thresh_q) return {};
    const int32_t dt_q = fire_dt_q(dt);

    // ---- Device buffers (the mutated fields + read-only fields/masks + the
    //      destroyed counter/index array). Per-call H2D/D2H. -------------------
    const size_t nb    = (size_t)n * sizeof(int32_t);
    const size_t nbool = (size_t)n * sizeof(bool);

    int32_t *d_fire = nullptr, *d_n_o2 = nullptr, *d_n_total = nullptr,
            *d_whp = nullptr, *d_temp = nullptr, *d_wx = nullptr, *d_wy = nullptr,
            *d_dt = nullptr;
    bool *d_wall = nullptr, *d_vac = nullptr, *d_flam = nullptr;
    int *d_counter = nullptr, *d_destroyed_idx = nullptr;
    // FUEL-FRACTION AXIS / PER-MATERIAL T_ext / #7: the nullable-plane idiom —
    // nullptr host plane -> nullptr device plane, nothing allocated or copied.
    int64_t *d_fuel_recip = nullptr;
    int32_t *d_T_ext_plane = nullptr;
    int32_t *d_atm = nullptr;

    cuda_check(cudaMalloc(&d_fire, nb), "malloc fire");
    cuda_check(cudaMalloc(&d_n_o2, nb), "malloc n_o2");
    cuda_check(cudaMalloc(&d_n_total, nb), "malloc n_total");
    cuda_check(cudaMalloc(&d_whp, nb), "malloc wall_hp");
    cuda_check(cudaMalloc(&d_temp, nb), "malloc temperature");
    cuda_check(cudaMalloc(&d_wx, nb), "malloc wind_x");
    cuda_check(cudaMalloc(&d_wy, nb), "malloc wind_y");
    cuda_check(cudaMalloc(&d_wall, nbool), "malloc is_wall");
    cuda_check(cudaMalloc(&d_vac, nbool), "malloc is_vacuum");
    cuda_check(cudaMalloc(&d_flam, nbool), "malloc flammable");
    cuda_check(cudaMalloc(&d_counter, sizeof(int)), "malloc counter");
    cuda_check(cudaMalloc(&d_destroyed_idx, (size_t)n * sizeof(int)),
               "malloc destroyed_idx");
    cuda_check(cudaMalloc(&d_dt, sizeof(int32_t)), "malloc dt_q");

    cuda_check(cudaMemcpy(d_fire, fire, nb, cudaMemcpyHostToDevice), "H2D fire");
    cuda_check(cudaMemcpy(d_n_o2, n_o2, nb, cudaMemcpyHostToDevice), "H2D n_o2");
    cuda_check(cudaMemcpy(d_n_total, n_total, nb, cudaMemcpyHostToDevice), "H2D n_total");
    cuda_check(cudaMemcpy(d_whp, wall_hp, nb, cudaMemcpyHostToDevice), "H2D wall_hp");
    cuda_check(cudaMemcpy(d_temp, temperature, nb, cudaMemcpyHostToDevice), "H2D temperature");
    cuda_check(cudaMemcpy(d_wx, wind_x, nb, cudaMemcpyHostToDevice), "H2D wind_x");
    cuda_check(cudaMemcpy(d_wy, wind_y, nb, cudaMemcpyHostToDevice), "H2D wind_y");
    cuda_check(cudaMemcpy(d_wall, is_wall, nbool, cudaMemcpyHostToDevice), "H2D is_wall");
    cuda_check(cudaMemcpy(d_vac, is_vacuum, nbool, cudaMemcpyHostToDevice), "H2D is_vacuum");
    cuda_check(cudaMemcpy(d_flam, flammable, nbool, cudaMemcpyHostToDevice), "H2D flammable");
    cuda_check(cudaMemcpy(d_dt, &dt_q, sizeof(int32_t), cudaMemcpyHostToDevice), "H2D dt_q");
    if (fuel_recip) {
        cuda_check(cudaMalloc(&d_fuel_recip, (size_t)n * sizeof(int64_t)),
                   "malloc fuel_recip");
        cuda_check(cudaMemcpy(d_fuel_recip, fuel_recip,
                              (size_t)n * sizeof(int64_t),
                              cudaMemcpyHostToDevice), "H2D fuel_recip");
    }
    if (fire_T_ext_plane) {
        cuda_check(cudaMalloc(&d_T_ext_plane, nb), "malloc fire_T_ext_plane");
        cuda_check(cudaMemcpy(d_T_ext_plane, fire_T_ext_plane, nb,
                              cudaMemcpyHostToDevice), "H2D fire_T_ext_plane");
    }
    if (atmosphere) {
        cuda_check(cudaMalloc(&d_atm, nb), "malloc atmosphere");
        cuda_check(cudaMemcpy(d_atm, atmosphere, nb,
                              cudaMemcpyHostToDevice), "H2D atmosphere");
    }

    // The core (N == 1; the n_o2 plane is a one-plane gas block). No device
    // gate: the host exit above already decided.
    fire_launch_resident(1, h, w, d_fire, d_atm, d_n_o2, 1, 0, d_n_total, d_whp,
                         d_temp, d_wx, d_wy, d_wall, d_vac, d_flam,
                         d_fuel_recip, d_T_ext_plane, d_dt, sc,
                         /*d_gate=*/nullptr, d_counter, d_destroyed_idx);

    cuda_check(cudaDeviceSynchronize(), "sync");

    // D2H the 3 mutated fields (fire, wall_hp, temperature). n_o2/wind/masks
    // are read-only — not copied back. (atmosphere is read-only — uploaded for
    // the #7 pressure factor, never returned.)
    cuda_check(cudaMemcpy(fire, d_fire, nb, cudaMemcpyDeviceToHost), "D2H fire");
    cuda_check(cudaMemcpy(wall_hp, d_whp, nb, cudaMemcpyDeviceToHost), "D2H wall_hp");
    cuda_check(cudaMemcpy(temperature, d_temp, nb, cudaMemcpyDeviceToHost), "D2H temperature");

    // Read the destroyed counter + the packed-index array, then build the
    // std::vector<pair> on the host — IN ROW-MAJOR ORDER, matching the CPU.
    //
    // ORDER IS LOAD-BEARING (audit Patch A / A5, 2026-08-04): the kernel
    // assigns slots with atomicAdd (arrival order, arbitrary); the CPU twin
    // walks i = 0..n row-major, and GameMap.destroy_wall is ORDER-DEPENDENT.
    // Sorting the PACKED LINEAR indices ascending is exactly row-major.
    int counter = 0;
    cuda_check(cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost),
               "D2H counter");
    std::vector<int> idx;
    if (counter > 0) {
        if (counter > n) counter = n;   // defensive (cannot exceed n slots)
        idx.resize((size_t)counter);
        cuda_check(cudaMemcpy(idx.data(), d_destroyed_idx,
                              (size_t)counter * sizeof(int), cudaMemcpyDeviceToHost),
                   "D2H destroyed_idx");
    }
    std::vector<std::pair<int, int>> destroyed = fire_destroyed_row_major(idx, w);

    cudaFree(d_fire);
    cudaFree(d_n_o2);
    cudaFree(d_n_total);
    cudaFree(d_whp);
    cudaFree(d_temp);
    cudaFree(d_wx);
    cudaFree(d_wy);
    cudaFree(d_wall);
    cudaFree(d_vac);
    cudaFree(d_flam);
    cudaFree(d_counter);
    cudaFree(d_destroyed_idx);
    cudaFree(d_dt);
    cudaFree(d_fuel_recip);   // nullptr-safe (no plane supplied -> never allocated)
    cudaFree(d_T_ext_plane);  // nullptr-safe (same nullable-plane idiom)
    cudaFree(d_atm);          // nullptr-safe (#7, same idiom)

    return destroyed;
}

std::vector<std::pair<int, int>> fire_destroyed_row_major(std::vector<int> idx, int w) {
    // ascending packed linear index == the CPU's i = 0..n walk (A5 above)
    std::sort(idx.begin(), idx.end());
    std::vector<std::pair<int, int>> destroyed;
    destroyed.reserve(idx.size());
    for (const int li : idx) destroyed.push_back({li / w, li % w});   // (y, x)
    return destroyed;
}

namespace {
bool g_fire_backend_cuda = false;
}
bool fire_backend_is_cuda() { return g_fire_backend_cuda; }
void set_fire_backend_cuda(bool on) { g_fire_backend_cuda = on; }

}  // namespace breach_cuda
