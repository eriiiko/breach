// ============================================================================
// CUDA-S6 fire solver implementation — see cuda_fire.h.
// A bit-identical GPU port of FireSimulation::step (fire_simulation.cpp ~44-314),
// RE-DERIVED for the EOS refactor (P6.8): the O2 gate now reads the REAL bulk O2
// density plane `n_o2` (not the old atmosphere/P proxy), and the own-tile plume
// deposit is the plume->T shim (temperature deposit, T_FLAME_MAX self-limiter —
// eos-p3fix-thermal-ceiling), NOT the retired atmosphere-overpressure write.
// Pipeline: per-tile signed-logistic intensity feedback → own-tile plume->T →
// smoke emission into neighbours → wall burn-through → final clamp.
//
// FIVE device passes (P2-P6), one per CPU pass, launched as a barriered chain
// (separate launches = grid barriers between dependent passes). P1 (the host max
// early-exit) is done on the HOST before any launch:
//   P1  early-exit         HOST *max_element(fire) < thresh -> return {} (untouched)
//   P2  logistic feedback  fire += dt*(grow-die); snap-extinguish (own-tile; O2 gate
//                          reads n_o2 neighbour mean — read-only field)
//   P3  plume->T shim      temperature[i] += clamped dT deposit      (own-tile write)
//   P4  smoke emission     smoke[nbr] += round(emit(fire[src]))     (SCATTER atomicAdd)
//   P5  wall burn-through  wall_hp[i] -= dmg; collect destroyed; fire[i]=0
//   P6  final clamp        fire, smoke -> [0, FP_ONE]               (own-tile)
//
// Pass order P2 → P3 → P4 → P5 → P6 matters: P3/P4/P5 all read the P2-updated
// `fire` (frozen between launches); P5 zeroes `fire` on destroyed cells AFTER
// P3/P4 have read it; P6 clamps last. P2 reads temperature (the `hot` gate) and
// P3 WRITES temperature — separate launches barrier between them, so P2 reads the
// tick-entry T (matching the CPU's fully-sequential logistic-then-plume order).
// NO cross-cell within-pass dependence exists (every pass is an own-index write
// with read-only neighbour reads or an order-free atomic/counter scatter), so this
// parallel schedule reproduces the CPU sequential result bit-for-bit — there is NO
// combustion-style Gauss-Seidel coupling here. Host scalar precompute mirrors the
// CPU load-time block exactly — all config constants quantized once in double,
// many via load-time make_recip/recip_mul (NOT per-cell reciprocal_q16).
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
// THE DETERMINISM CRUX (P4): the 4 smoke emissions per source thread are deposited
// with integer atomicAdd. The deposit depends ONLY on fire[src] (NOT on the
// neighbour's current smoke — VERIFIED in the CPU code, lines ~223-234: delta_q is
// computed from `I = fire[y*w+x]` before the neighbour loop, which just adds the
// SAME delta_q to each non-wall neighbour). Integer + is associative + commutative
// -> the per-neighbour sum of overlapping deposits is ORDER-FREE -> bit-identical
// to the CPU's sequential row-major adds. The S2 raycaster's saturating-int atomic
// is precedent; here it is a plain non-saturating add.
//
// P5 destroyed-list collection: a device int counter (atomicAdd for a slot) + a
// device array of packed indices (sized n, worst case every cell destroyed); copy
// counter + array back to the host and build the std::vector<pair> (any order). The
// gate checks set equality + length -> no drops/dupes.
// ============================================================================
#include "cuda_fire.h"
#include "fixed_point.h"   // q16, quantize, mul_q16, mul_wide, narrow_round, FP_ONE, FP_SHIFT, make_recip, mean_round
#include "cuda_fixedpoint_device.cuh"  // sqrt_q16_dev, recip_mul_dev (S6 §2 shared kit)

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
                              const int32_t* __restrict__ n_o2,
                              const int32_t* __restrict__ n_total,
                              const int32_t* __restrict__ wall_hp,
                              const int32_t* __restrict__ temperature,
                              const int32_t* __restrict__ wind_x,
                              const int32_t* __restrict__ wind_y,
                              const bool* __restrict__ is_wall,
                              const bool* __restrict__ is_vacuum,
                              const bool* __restrict__ flammable,
                              const int64_t* __restrict__ fuel_recip,
                              int h, int w,
                              int32_t dt_q, int32_t k_grow_q, int32_t k_die_q,
                              int32_t k_wind_fan_q, int32_t k_wind_strip_q,
                              int32_t fire_T_ext_q, int32_t x_ext_q,
                              int32_t X_N_FLOOR, int32_t I_min_q,
                              bool temp_is_identity, int64_t recip_temp_scale,
                              int64_t recip_fuel_ref, int64_t recip_T_span,
                              int64_t recip_x_span, bool x_degenerate) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (!flammable[i]) continue;        // only fuel burns
        const q16 I = fire[i];
        if (I <= 0) continue;               // unlit -> nothing to step

        const int y = i / w, x = i % w;

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
        int64_t sum_o2 = 0;
        int64_t sum_tot = 0;
        for (int d = 0; d < 4; ++d) {
            const int ny = y + D4_dy[d], nx = x + D4_dx[d];
            if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                const int ni = ny * w + nx;
                if (!is_wall[ni] && !is_vacuum[ni]) {
                    sum_o2  += (int64_t)n_o2[ni];      // exact, order-free
                    sum_tot += (int64_t)n_total[ni];   // exact, order-free
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

        // Gates. o2f is LINEAR in X (the continuous-O2 law), clamped [0,1]:
        // X <= X_ext -> 0 (extinction), X >= X_full -> 1 (pure O2; ambient air
        // lands at 0.092). The degenerate span falls back to a step at X_ext.
        const q16 hot = clamp01_q_dev(recip_mul_dev(T - fire_T_ext_q, recip_T_span));
        const q16 o2f = x_degenerate
            ? ((X < x_ext_q) ? (q16)0 : (q16)FP_ONE)
            : clamp01_q_dev(recip_mul_dev(X - x_ext_q, recip_x_span));
        const q16 avail = mul_q16(F, o2f);

        // grow = k_grow * avail * hot * I * (1 - I) * (1 + k_wind_fan*W). PINNED
        // left-fold mul_q16, each narrowing once, in this EXACT order.
        const q16 one_minus_I = (q16)FP_ONE - I;                       // (1 - I)
        const q16 wind_fan = (q16)FP_ONE + mul_q16(k_wind_fan_q, W);   // (1 + k_wind_fan*W)
        q16 grow = k_grow_q;
        grow = mul_q16(grow, avail);
        grow = mul_q16(grow, hot);
        grow = mul_q16(grow, I);
        grow = mul_q16(grow, one_minus_I);
        grow = mul_q16(grow, wind_fan);

        // die = k_die*(1 - avail*hot)*I + k_wind_strip*W*(1 - I)*I.
        const q16 avail_hot = mul_q16(avail, hot);          // avail*hot
        const q16 one_minus_ah = (q16)FP_ONE - avail_hot;   // (1 - avail*hot)
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

// ---- P3: own-tile plume ENERGY DEPOSIT — the plume->T shim (EOS refactor P3;
// self-limiter T-gated, eos-p3fix-thermal-ceiling / decisions.md #16;
// fire_simulation.cpp ~228-262) --------------------------------------------
// REPLACES the retired pressure write (P is solver-owned now — a direct
// atmosphere write would be clobbered next tick). The gain scalar becomes a
// small dT energy deposit, self-limited against a PHYSICAL flame ceiling
// (T_FLAME_MAX) measured on the SAME quantity being deposited (T, not P):
//   sat  = clamp01(1 - temperature[i]/T_FLAME_MAX)
//   gain = fire_pressure_gain * I * sat * dt        (round-to-nearest deposit)
//   dT   = gain * temp_gain_scale                   (round-to-nearest deposit)
//   dT   = min(dT, T_FLAME_MAX - temperature[i])    (belt-and-suspenders cap)
//   temperature[i] = sat_add_q16(temperature[i], dT)   (own-index -> no race)
// VERBATIM device transcription of the CPU plume loop: same clamp01 on sat,
// same PINNED left-fold, same sign-symmetric narrow_round, same headroom
// hard-cap, same saturating add. Own-index write only -> order-free.
__global__ void fire_plume(const int32_t* __restrict__ fire,
                           int32_t* __restrict__ temperature,
                           int32_t gain_q, int32_t dt_q,
                           int32_t temp_gain_scale_q, int32_t t_flame_max_q,
                           int64_t recip_T_flame_max, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const q16 I = fire[i];
        if (I <= 0) continue;
        // sat = clamp01(1 - temperature[i]/T_FLAME_MAX). temperature[i] may be
        // negative (below ambient) or above the ceiling -> clamp to [0,1].
        const q16 sat = clamp01_q_dev((q16)FP_ONE -
                                      recip_mul_dev(temperature[i], recip_T_flame_max));
        // gain = gain_q * I * sat * dt. PINNED left-fold; round-to-nearest the
        // final narrow (deposit). sat >= 0 so gain >= 0.
        q16 g = mul_q16(gain_q, I);            // gain_q * I
        g = mul_q16(g, sat);                   // * (1 - T/T_FLAME_MAX)
        const q16 gain = narrow_round_signed(mul_wide(g, dt_q));
        if (gain > 0) {
            q16 dT = narrow_round_signed(mul_wide(gain, temp_gain_scale_q));
            // Belt-and-suspenders hard cap: never deposit PAST T_FLAME_MAX in
            // one tick even if the smooth taper under-clamps at extreme dt/gain.
            const q16 headroom = (temperature[i] < t_flame_max_q)
                ? (q16)(t_flame_max_q - temperature[i]) : 0;
            if (dT > headroom) dT = headroom;
            if (dT > 0) temperature[i] = sat_add_q16(temperature[i], dT);
        }
    }
}

// ---- P4: smoke emission SCATTER (fire_simulation.cpp ~221-237) ----------------
// delta_q = round_nearest(emission*dt*fire[src]) — depends ONLY on fire[src]. Each
// source thread atomicAdds delta_q into each non-wall 4-neighbour's smoke. Integer
// atomicAdd is associative + commutative -> order-free -> the per-neighbour sum is
// bit-identical to the CPU's sequential row-major adds (the determinism crux).
__global__ void fire_smoke_emit(const int32_t* __restrict__ fire,
                                int32_t* __restrict__ smoke,
                                const bool* __restrict__ is_wall,
                                int32_t emission_q, int32_t dt_q, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const q16 I = fire[i];
        if (I <= 0) continue;
        const int y = i / w, x = i % w;
        // delta = smoke_emission * dt * I (positive). PINNED order; round-to-nearest.
        const q16 ed = mul_q16(emission_q, dt_q);       // emission*dt
        const int64_t wide = mul_wide(ed, I);           // * I (wide for round)
        const q16 delta_q = narrow_round(wide);         // >= 0 (positive deposit)
        for (int d = 0; d < 4; ++d) {
            const int ny = y + D4_dy[d], nx = x + D4_dx[d];
            if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                const int ni = ny * w + nx;
                if (!is_wall[ni]) {
                    atomicAdd(&smoke[ni], (int)delta_q);   // order-free integer add
                }
            }
        }
    }
}

// ---- P5: wall burn-through + destroyed collection (fire_simulation.cpp ~245-256)
// wall_hp[i] -= round_nearest(wall_damage*dt*fire[i]); if (wall_hp<=0 && flammable
// && is_wall) collect (via atomicAdd counter -> packed index array) and fire[i]=0.
// Reads the P2-updated fire (frozen since P2; P3/P4 already read it before this
// kernel zeroes it). Own-cell wall_hp/fire writes; the destroyed slot is the only
// scatter (a counter atomicAdd, order arbitrary -> the gate checks SET equality).
__global__ void fire_burn(int32_t* __restrict__ fire,
                          int32_t* __restrict__ wall_hp,
                          const bool* __restrict__ is_wall,
                          const bool* __restrict__ flammable,
                          int* __restrict__ d_counter,
                          int* __restrict__ d_destroyed_idx,
                          int32_t wall_damage_q, int32_t dt_q, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (fire[i] > 0) {
            const q16 wd = mul_q16(wall_damage_q, dt_q);    // wall_damage*dt
            const int64_t wide = mul_wide(wd, fire[i]);     // * I (wide for round)
            const q16 dmg = narrow_round(wide);             // >= 0 (positive depletion)
            wall_hp[i] -= dmg;
            if (wall_hp[i] <= 0 && flammable[i] && is_wall[i]) {
                const int slot = atomicAdd(d_counter, 1);   // grab a unique slot
                d_destroyed_idx[slot] = i;                  // packed linear index
                fire[i] = 0;
            }
        }
    }
}

// ---- P6: final clamp (fire_simulation.cpp ~260-264) --------------------------
// fire clamps to [0, FP_ONE]; smoke clamps the same. Own-cell, in-place.
__global__ void fire_clamp(int32_t* __restrict__ fire,
                           int32_t* __restrict__ smoke, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        fire[i] = clamp01_q_dev(fire[i]);
        int32_t s = smoke[i];
        if (s < 0) s = 0;
        else if (s > FP_ONE) s = FP_ONE;
        smoke[i] = s;
    }
}

}  // namespace

std::vector<std::pair<int, int>> fire_step(
    int32_t* fire, const int32_t* atmosphere, const int32_t* n_o2,
    const int32_t* n_total,
    int32_t* smoke, int32_t* wall_hp, int32_t* temperature,
    const int32_t* wind_x, const int32_t* wind_y,
    const bool* is_wall, const bool* is_vacuum, const bool* flammable,
    int h, int w, float dt,
    float k_grow, float k_die, float fire_T_ext, float fire_T_span,
    float fuel_ref, float o2_frac_ext, float o2_frac_full, float I_min,
    float k_wind_fan, float k_wind_strip, float fire_pressure_gain,
    float smoke_emission, float wall_damage,
    float temp_scale, float temp_gain_scale, float T_FLAME_MAX,
    const int64_t* fuel_recip) {   // FUEL-FRACTION AXIS (nullable, see header)
    (void)atmosphere;   // EOS P4: vestigial — the CPU step keeps it in its
                        // signature (ABI parity) but no longer reads it (the O2
                        // gate moved to n_o2; the plume self-limiter to T).
    const int n = h * w;
    if (n <= 0) return {};

    // ---- P1: HOST max early-exit (fire_simulation.cpp ~61-66). An ORDER-FREE
    //      integer max reduction; the threshold is a pinned Q16.16 compare. If no
    //      tile is above thresh, the CPU returns {} and leaves ALL fields
    //      UNCHANGED — so we skip every kernel + H2D/D2H and return empty. -------
    const q16 max_fire_thresh_q = quantize(0.001);   // ~66 counts
    q16 max_fire = 0;
    for (int i = 0; i < n; ++i) max_fire = std::max(max_fire, fire[i]);
    if (max_fire < max_fire_thresh_q) return {};

    // ---- Host scalar precompute (fire_simulation.cpp ~81-104, VERBATIM, in
    //      double). All config constants quantized once; the divides by config
    //      constants are load-time make_recip reciprocals (NOT per-cell). --------
    const bool    temp_is_identity = (temp_scale == (float)FP_ONE);
    const int64_t recip_temp_scale = temp_is_identity
        ? 0 : make_recip((double)temp_scale);

    const q16 dt_q          = quantize((double)dt);
    const q16 k_grow_q      = quantize((double)k_grow);
    const q16 k_die_q       = quantize((double)k_die);
    const q16 k_wind_fan_q  = quantize((double)k_wind_fan);
    const q16 k_wind_strip_q = quantize((double)k_wind_strip);
    const q16 fire_T_ext_q  = quantize((double)fire_T_ext);
    const q16 I_min_q       = quantize((double)I_min);
    const q16 gain_q        = quantize((double)fire_pressure_gain);
    const q16 emission_q    = quantize((double)smoke_emission);
    const q16 wall_damage_q = quantize((double)wall_damage);
    // Plume->T shim constants (EOS P3 / eos-p3fix-thermal-ceiling).
    const q16 temp_gain_scale_q = quantize((double)temp_gain_scale);
    const q16 t_flame_max_q     = quantize((double)T_FLAME_MAX);

    const int64_t recip_fuel_ref  = make_recip((double)fuel_ref);
    const int64_t recip_T_span    = make_recip((double)fire_T_span);
    // Continuous-O2 law span (VERBATIM of fire_simulation.cpp's x_ext_q/x_span/
    // x_degenerate/recip_x_span/X_N_FLOOR block): o2f = clamp01((X - X_ext) /
    // (X_full - X_ext)). X_ext = 0 gives span == X_full (pure proportional);
    // X_full <= X_ext (misconfig) -> a step at X_ext. FULL-RESPONSE REFERENCE
    // SPLIT (2026-07-30): the upper end is the PURE-O2 reference o2_frac_full,
    // NOT o2_frac_amb (which made ambient the ceiling).
    const q16 x_ext_q              = quantize((double)o2_frac_ext);
    const double  x_span           = (double)o2_frac_full - (double)o2_frac_ext;
    const bool    x_degenerate     = (x_span <= 0.0);
    const int64_t recip_x_span     = x_degenerate ? 0 : make_recip(x_span);
    const q16 X_N_FLOOR             = quantize(0.01);   // 655 counts, SAME as CPU
    const int64_t recip_T_flame_max = make_recip((double)T_FLAME_MAX);

    // ---- Device buffers (the 4 mutated fields + read-only fields/masks + the
    //      destroyed counter/index array). Per-call H2D/D2H; residency is S8. ----
    const size_t nb    = (size_t)n * sizeof(int32_t);
    const size_t nbool = (size_t)n * sizeof(bool);

    int32_t *d_fire = nullptr, *d_n_o2 = nullptr, *d_n_total = nullptr,
            *d_smoke = nullptr,
            *d_whp = nullptr, *d_temp = nullptr, *d_wx = nullptr, *d_wy = nullptr;
    bool *d_wall = nullptr, *d_vac = nullptr, *d_flam = nullptr;
    int *d_counter = nullptr, *d_destroyed_idx = nullptr;
    // FUEL-FRACTION AXIS: the OPTIONAL per-tile 1/hp plane. nullptr host plane
    // -> nullptr device plane, nothing allocated and nothing copied, and the
    // kernel takes the scalar fallback — the documented nullable-plane idiom
    // the cool-shift axis uses on the temperature kernel.
    int64_t *d_fuel_recip = nullptr;

    cuda_check(cudaMalloc(&d_fire, nb), "malloc fire");
    cuda_check(cudaMalloc(&d_n_o2, nb), "malloc n_o2");
    cuda_check(cudaMalloc(&d_n_total, nb), "malloc n_total");
    cuda_check(cudaMalloc(&d_smoke, nb), "malloc smoke");
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

    cuda_check(cudaMemcpy(d_fire, fire, nb, cudaMemcpyHostToDevice), "H2D fire");
    cuda_check(cudaMemcpy(d_n_o2, n_o2, nb, cudaMemcpyHostToDevice), "H2D n_o2");
    cuda_check(cudaMemcpy(d_n_total, n_total, nb, cudaMemcpyHostToDevice), "H2D n_total");
    cuda_check(cudaMemcpy(d_smoke, smoke, nb, cudaMemcpyHostToDevice), "H2D smoke");
    cuda_check(cudaMemcpy(d_whp, wall_hp, nb, cudaMemcpyHostToDevice), "H2D wall_hp");
    cuda_check(cudaMemcpy(d_temp, temperature, nb, cudaMemcpyHostToDevice), "H2D temperature");
    cuda_check(cudaMemcpy(d_wx, wind_x, nb, cudaMemcpyHostToDevice), "H2D wind_x");
    cuda_check(cudaMemcpy(d_wy, wind_y, nb, cudaMemcpyHostToDevice), "H2D wind_y");
    cuda_check(cudaMemcpy(d_wall, is_wall, nbool, cudaMemcpyHostToDevice), "H2D is_wall");
    cuda_check(cudaMemcpy(d_vac, is_vacuum, nbool, cudaMemcpyHostToDevice), "H2D is_vacuum");
    cuda_check(cudaMemcpy(d_flam, flammable, nbool, cudaMemcpyHostToDevice), "H2D flammable");
    cuda_check(cudaMemset(d_counter, 0, sizeof(int)), "memset counter");
    if (fuel_recip) {
        cuda_check(cudaMalloc(&d_fuel_recip, (size_t)n * sizeof(int64_t)),
                   "malloc fuel_recip");
        cuda_check(cudaMemcpy(d_fuel_recip, fuel_recip,
                              (size_t)n * sizeof(int64_t),
                              cudaMemcpyHostToDevice), "H2D fuel_recip");
    }

    const int block = 256;
    const int grid = (n + block - 1) / block;

    // P2 logistic feedback (in-place on d_fire; O2 gate reads d_n_o2 neighbour
    // mean; reads wall_hp/temp/wind/masks).
    fire_logistic<<<grid, block>>>(
        d_fire, d_n_o2, d_n_total, d_whp, d_temp, d_wx, d_wy, d_wall, d_vac,
        d_flam, d_fuel_recip, h, w,
        dt_q, k_grow_q, k_die_q, k_wind_fan_q, k_wind_strip_q, fire_T_ext_q,
        x_ext_q, X_N_FLOOR, I_min_q, temp_is_identity, recip_temp_scale,
        recip_fuel_ref, recip_T_span, recip_x_span, x_degenerate);
    cuda_check(cudaGetLastError(), "logistic launch");

    // P3 plume->T shim (in-place on d_temp; reads the P2-updated d_fire). Barriers
    // after P2, so d_temp reads here are the tick-entry T (matching the CPU order).
    fire_plume<<<grid, block>>>(d_fire, d_temp, gain_q, dt_q,
                                temp_gain_scale_q, t_flame_max_q,
                                recip_T_flame_max, n);
    cuda_check(cudaGetLastError(), "plume launch");

    // P4 smoke emission scatter (atomicAdd into d_smoke; reads the P2-updated d_fire).
    fire_smoke_emit<<<grid, block>>>(d_fire, d_smoke, d_wall, emission_q, dt_q, h, w);
    cuda_check(cudaGetLastError(), "smoke_emit launch");

    // P5 wall burn-through (in-place on d_whp; zeroes d_fire on destroyed AFTER
    // P3/P4 read it; collects the destroyed indices via the device counter).
    fire_burn<<<grid, block>>>(d_fire, d_whp, d_wall, d_flam, d_counter,
                               d_destroyed_idx, wall_damage_q, dt_q, n);
    cuda_check(cudaGetLastError(), "burn launch");

    // P6 final clamp (in-place on d_fire / d_smoke).
    fire_clamp<<<grid, block>>>(d_fire, d_smoke, n);
    cuda_check(cudaGetLastError(), "clamp launch");

    cuda_check(cudaDeviceSynchronize(), "sync");

    // D2H the 4 mutated fields (fire, smoke, wall_hp, temperature). n_o2/wind/masks
    // are read-only — not copied back. (atmosphere is now read-only + vestigial —
    // never uploaded, never returned.)
    cuda_check(cudaMemcpy(fire, d_fire, nb, cudaMemcpyDeviceToHost), "D2H fire");
    cuda_check(cudaMemcpy(smoke, d_smoke, nb, cudaMemcpyDeviceToHost), "D2H smoke");
    cuda_check(cudaMemcpy(wall_hp, d_whp, nb, cudaMemcpyDeviceToHost), "D2H wall_hp");
    cuda_check(cudaMemcpy(temperature, d_temp, nb, cudaMemcpyDeviceToHost), "D2H temperature");

    // Read the destroyed counter + the packed-index array, then build the
    // std::vector<pair> on the host (any order — the gate checks SET equality).
    int counter = 0;
    cuda_check(cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost),
               "D2H counter");
    std::vector<std::pair<int, int>> destroyed;
    if (counter > 0) {
        if (counter > n) counter = n;   // defensive (cannot exceed n slots)
        std::vector<int> idx((size_t)counter);
        cuda_check(cudaMemcpy(idx.data(), d_destroyed_idx,
                              (size_t)counter * sizeof(int), cudaMemcpyDeviceToHost),
                   "D2H destroyed_idx");
        destroyed.reserve((size_t)counter);
        for (int k = 0; k < counter; ++k) {
            const int li = idx[(size_t)k];
            destroyed.push_back({li / w, li % w});   // (y, x), matching the CPU
        }
    }

    cudaFree(d_fire);
    cudaFree(d_n_o2);
    cudaFree(d_n_total);
    cudaFree(d_smoke);
    cudaFree(d_whp);
    cudaFree(d_temp);
    cudaFree(d_wx);
    cudaFree(d_wy);
    cudaFree(d_wall);
    cudaFree(d_vac);
    cudaFree(d_flam);
    cudaFree(d_counter);
    cudaFree(d_destroyed_idx);
    cudaFree(d_fuel_recip);   // nullptr-safe (no plane supplied -> never allocated)

    return destroyed;
}

namespace {
bool g_fire_backend_cuda = false;
}
bool fire_backend_is_cuda() { return g_fire_backend_cuda; }
void set_fire_backend_cuda(bool on) { g_fire_backend_cuda = on; }

}  // namespace breach_cuda
