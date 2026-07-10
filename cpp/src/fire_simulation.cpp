#include "fire_simulation.h"
#include "fixed_point.h"
#include <algorithm>
#include <cstdint>

// Neighbor offsets: 4-connected (the open-neighbour pressure mean + smoke spread).
static constexpr int D4[][2] = {{-1,0},{1,0},{0,-1},{0,1}};

static inline bool in_bounds(int y, int x, int h, int w) {
    return y >= 0 && y < h && x >= 0 && x < w;
}

namespace fp = fixedpoint;
// q16 is the global Q16.16 type alias (int32_t), declared in fixed_point.h.

// Integer clamp to [0, FP_ONE] (the [0,1] tracer saturation, exact).
static inline q16 clamp01_q(q16 v) {
    if (v < 0) return 0;
    if (v > fp::FP_ONE) return fp::FP_ONE;
    return v;
}

// The round-to-nearest deposit narrows (the UNBIASED-DEPOSIT idiom, S2a/S2c lesson:
// the plume + smoke + wall-burn are fire SOURCES/sinks, not cancelling flux pairs,
// so they want round-half, NOT the mul_q16 truncation toward -inf) live in
// fixed_point.h as the shared narrow_round / narrow_round_signed — ONE definition of
// the deterministic deposit rounding for fire + atmosphere, so a future revision of
// the rounding convention can't desync one TU from another.

// Hermite smoothstep on [edge0, edge1] -> [0, FP_ONE], clamped outside, in Q16.16.
// PINNED multiply tree (plan §2.5, master plan §2.4 fire row — deterministic by
// construction): t = clamp01((x - edge0) * recip_span); then t2 = t*t,
// three_minus = (3<<16) - 2t, return t2 * three_minus. recip_span is the load-time
// reciprocal of (edge1 - edge0) (a config constant). edge1 <= edge0 -> a step.
static inline q16 smoothstep_q(q16 edge0, q16 edge1, q16 x,
                               int64_t recip_span, bool degenerate) {
    if (degenerate) return (x < edge0) ? 0 : fp::FP_ONE;
    const q16 t = clamp01_q(fp::recip_mul(x - edge0, recip_span));
    const q16 t2 = fp::mul_q16(t, t);                         // t*t
    const q16 three_minus = (q16)(3 * fp::FP_ONE) - (q16)(t << 1);   // 3 - 2t
    return fp::mul_q16(t2, three_minus);                     // t^2 * (3 - 2t)
}

std::vector<std::pair<int, int>> FireSimulation::step(
    q16* fire,                    // S3b: Q16.16 int32 (was float)
    const q16* atmosphere,        // S2c: Q16.16 int32 == P (EOS P3: read-only, plume only)
    const q16* n_o2,               // EOS P4: Q16.16 int32 real O2 density (the O2 gate)
    int32_t* smoke,               // S2b: Q16.16
    q16* wall_hp,                 // S3b: Q16.16 int32 (was float)
    q16* temperature,             // EOS P3: mutable (plume->T shim)
    const q16* wind_x,            // S2c/S3b: Q16.16 int32 (read-only)
    const q16* wind_y,            // S2c/S3b: Q16.16 int32 (read-only)
    const bool* is_wall,
    const bool* is_vacuum,
    const bool* flammable,
    int h, int w,
    float dt
) const {
    const int n = h * w;
    const auto& p = params;

    // Early exit if no fire (mirrors the old behaviour). max_fire is an ORDER-FREE
    // integer max reduction; the threshold is a pinned Q16.16 compare.
    const q16 max_fire_thresh_q = fp::quantize(0.001);   // ~66 counts (0.001*65536)
    q16 max_fire = 0;
    for (int i = 0; i < n; ++i) max_fire = std::max(max_fire, fire[i]);
    if (max_fire < max_fire_thresh_q) return {};

    // --- Load-time constants (computed ONCE per step, in double, then quantized —
    //     the LOCKED S1 idiom: IEEE double is bit-identical cross-machine for these
    //     scalar/constant computations; NO per-cell float). All the fire divides
    //     are by CONFIG CONSTANTS -> load-time reciprocals (plan §2.3), not the
    //     per-cell reciprocal_q16. ---
    // temp_scale == FP_ONE (the shipped default): T in game-units IS the raw
    // temperature[i] in Q16.16 (the dequantize-then-requantize-for-compare cancels
    // at the shared scale), so the inv_temp divide is the IDENTITY. temp_scale is a
    // RUNTIME config value (physics_runner binds it from [thermal] TEMP_SCALE), so we
    // do NOT rely on a Release-stripped assert: when it equals FP_ONE we take the
    // identity fast path (no per-cell divide), and when it does NOT we fall back to a
    // load-time reciprocal multiply (recip_mul by 1/temp_scale) — correct AND
    // deterministic for ANY temp_scale, not just the shipped one.
    const bool    temp_is_identity = (p.temp_scale == (float)fp::FP_ONE);
    const int64_t recip_temp_scale = temp_is_identity
        ? 0 : fp::make_recip((double)p.temp_scale);

    const q16 dt_q          = fp::quantize((double)dt);
    const q16 k_grow_q      = fp::quantize((double)p.k_grow);
    const q16 k_die_q       = fp::quantize((double)p.k_die);
    const q16 k_wind_fan_q  = fp::quantize((double)p.k_wind_fan);
    const q16 k_wind_strip_q = fp::quantize((double)p.k_wind_strip);
    const q16 fire_T_ext_q  = fp::quantize((double)p.fire_T_ext);
    const q16 P_min_q       = fp::quantize((double)p.P_min);
    const q16 P_full_q      = fp::quantize((double)p.P_full);
    const q16 I_min_q       = fp::quantize((double)p.I_min);
    const q16 gain_q        = fp::quantize((double)p.fire_pressure_gain);
    const q16 emission_q    = fp::quantize((double)p.smoke_emission);
    const q16 wall_damage_q = fp::quantize((double)p.wall_damage);

    // Load-time reciprocals for the config-constant divides (make_recip/recip_mul).
    const int64_t recip_fuel_ref  = fp::make_recip((double)p.fuel_ref);       // F = wall_hp/fuel_ref
    const int64_t recip_T_span    = fp::make_recip((double)p.fire_T_span);    // hot ramp
    const double  P_span          = (double)p.P_full - (double)p.P_min;       // smoothstep span
    const bool    P_degenerate    = (P_span <= 0.0);
    const int64_t recip_P_span    = P_degenerate ? 0 : fp::make_recip(P_span);
    // eos-p3fix-thermal-ceiling: the plume's self-limiter now gates on T
    // (see FireParams::T_FLAME_MAX doc) instead of the structurally-dead
    // atmosphere/p_expand_ref gate (retired, see FireParams::p_expand_ref).
    const int64_t recip_T_flame_max = fp::make_recip((double)p.T_FLAME_MAX);

    // --- Per-tile signed-logistic FEEDBACK (fire_design_proposal §2 + §5) ---
    // Spread is gone (radiation -> heat -> temperature -> ignition handles it);
    // this is purely the life/death of an already-lit tile. Each tile reads its own
    // intensity I, temperature T, fuel F (from wall_hp), neighbour pressure P (the
    // O2 proxy) and wind magnitude W, then steps I by dt*(grow - die). ALL Q16.16.
    for (int i = 0; i < n; ++i) {
        if (!flammable[i]) continue;        // only fuel burns
        const q16 I = fire[i];
        if (I <= 0) continue;               // unlit -> nothing to step (ignition is elsewhere)

        const int y = i / w, x = i % w;

        // T: temperature field in game-units (Q16.16). With temp_scale == FP_ONE the
        // raw temperature[i] IS T (identity); otherwise divide by temp_scale via the
        // load-time reciprocal. T may be negative (it stores dT); `hot`'s clamp01
        // handles it.
        const q16 T = temp_is_identity
            ? temperature[i]
            : fp::recip_mul(temperature[i], recip_temp_scale);

        // F: fuel from remaining wall HP, normalised. recip_mul by the load-time
        // reciprocal of fuel_ref, then clamp01.
        const q16 F = clamp01_q(fp::recip_mul(wall_hp[i], recip_fuel_ref));

        // O2: mean n_o2 over OPEN (non-solid, non-vacuum) 4-neighbours — the
        // fire reads INCOMING fresh air (its own tile holds no gas — it is
        // solid). int64 sum + mean_round (round-half-away-from-zero) — the
        // EXACT predicate the Python ignition twin shares (closes the S3a
        // exact-tie gap, review item #1). EOS refactor P4 (design §6): reads
        // the REAL bulk O2 density plane, NOT the atmosphere/P proxy — the
        // decompression-extinguishes-fire mechanism is now genuine oxygen
        // depletion, not a pressure stand-in. No open neighbour -> count 0 ->
        // O2 = 0 (mean_round guard).
        int64_t sum_o2 = 0;
        int64_t count = 0;
        for (const auto& d : D4) {
            int ny = y + d[0], nx = x + d[1];
            int ni = ny * w + nx;
            if (in_bounds(ny, nx, h, w) && !is_wall[ni] && !is_vacuum[ni]) {
                sum_o2 += (int64_t)n_o2[ni];   // exact, order-free
                count += 1;
            }
        }
        const q16 P = fp::mean_round(sum_o2, count);

        // W: wind magnitude from the SHARED wind field (= -grad p incl. waves, so a
        // grenade shockwave is a transient spike -> firestorm / blow-out). The int64
        // Q.32 radicand (wx^2 + wy^2), then the new floor-isqrt sqrt_q16 -> Q16.16.
        const int64_t rad = fp::mul_wide(wind_x[i], wind_x[i])
                          + fp::mul_wide(wind_y[i], wind_y[i]);
        const q16 W = fp::sqrt_q16(rad);

        // Gates.
        const q16 hot = clamp01_q(fp::recip_mul(T - fire_T_ext_q, recip_T_span));
        const q16 o2  = smoothstep_q(P_min_q, P_full_q, P, recip_P_span, P_degenerate);
        const q16 avail = fp::mul_q16(F, o2);

        // Signed logistic update, fanned + stripped by wind. PINNED MULTIPLY ORDER
        // (master plan §2.4 / plan §5.2 — the chained-truncation association hazard):
        // a LEFT-FOLD of mul_q16, each narrowing once, in this EXACT sequence on
        // every peer -> deterministic by construction.
        //
        //   grow = k_grow * avail * hot * I * (1 - I) * (1 + k_wind_fan*W)
        const q16 one_minus_I = (q16)fp::FP_ONE - I;             // (1 - I)
        const q16 wind_fan = (q16)fp::FP_ONE + fp::mul_q16(k_wind_fan_q, W);  // (1 + k_wind_fan*W)
        q16 grow = k_grow_q;                       // k_grow
        grow = fp::mul_q16(grow, avail);           // * avail
        grow = fp::mul_q16(grow, hot);             // * hot
        grow = fp::mul_q16(grow, I);               // * I
        grow = fp::mul_q16(grow, one_minus_I);     // * (1 - I)
        grow = fp::mul_q16(grow, wind_fan);        // * (1 + k_wind_fan*W)

        //   die = k_die * (1 - avail*hot) * I  +  k_wind_strip * W * (1 - I) * I
        const q16 avail_hot = fp::mul_q16(avail, hot);          // avail*hot
        const q16 one_minus_ah = (q16)fp::FP_ONE - avail_hot;   // (1 - avail*hot)
        q16 die_a = k_die_q;                       // k_die
        die_a = fp::mul_q16(die_a, one_minus_ah);  // * (1 - avail*hot)
        die_a = fp::mul_q16(die_a, I);             // * I
        q16 die_b = k_wind_strip_q;                // k_wind_strip
        die_b = fp::mul_q16(die_b, W);             // * W
        die_b = fp::mul_q16(die_b, one_minus_I);   // * (1 - I)
        die_b = fp::mul_q16(die_b, I);             // * I
        const q16 die = die_a + die_b;             // signed Q16.16 sum

        // I_next = clamp01(I + dt*(grow - die)). grow - die is a signed Q16.16
        // subtract; dt*(.) is one mul_q16; the add is exact.
        const q16 delta = fp::mul_q16(dt_q, grow - die);
        q16 I_next = clamp01_q(I + delta);
        if (I_next < I_min_q) I_next = 0;          // snap-extinguish (discrete output)
        fire[i] = I_next;
    }

    // --- Own-tile plume ENERGY DEPOSIT (EOS refactor P3 — the minimal
    //     plume->T shim, design §8 patch P3 writer row; self-limiter fixed
    //     eos-p3fix-thermal-ceiling, decisions.md #16) --------------------
    // REPLACES the old own-tile `atmosphere += gain` overpressure write:
    // P is solver-owned now (materialized once/tick by eos_solver), so a
    // direct write here would be silently clobbered next tick — "the pop
    // never goes inert" means the plume must feed the EOS instead (T -> p*
    // -> the Helmholtz solve -> outward u, natively).
    //
    // Self-limiter: T-based, not P-based (see FireParams::T_FLAME_MAX for
    // the full root-cause writeup — the P-based gate read `atmosphere[i]`
    // at the plume's OWN tile, which the EOS solver force-zeroes for every
    // SOLID cell, so the old gate never actually engaged). Same SMOOTH
    // taper shape as the retired gate (`sat = clamp01(1 - x/ref)`), now
    // measured against the thing actually being deposited:
    //   sat  = clamp01(1 - temperature[i]/T_FLAME_MAX)
    //   gain = fire_pressure_gain * I * sat * dt
    //   dT   = gain * temp_gain_scale
    // `sat` -> 0 as T[i] -> T_FLAME_MAX, so the deposit tapers to nothing at
    // the physical ceiling instead of riding fire intensity unbounded.
    // ROUND-TO-NEAREST (S2a/S2c unbiased-deposit lesson) so a long firestorm
    // does not accumulate a truncation DC bias. The final write is a
    // SATURATING add (fixed_point.h `sat_add_q16`) — independent of the
    // limiter, wrapping past the int32 ceiling is a correctness bug on its
    // own (a stacked-source pathological case could still clear T_FLAME_MAX
    // in one deposit before the NEXT tick's sat gate catches it).
    const q16 temp_gain_scale_q = fp::quantize((double)p.temp_gain_scale);
    const q16 t_flame_max_q = fp::quantize((double)p.T_FLAME_MAX);
    if (dbg_probe_idx >= 0 && dbg_probe_idx < n) dbg_plume_dT = 0;   // DEBUG probe (temporary)
    for (int i = 0; i < n; ++i) {
        const q16 I = fire[i];
        if (I <= 0) continue;
        // (1 - T[i]/T_FLAME_MAX), clamped to [0,1] (clamp01_q, top of file):
        // T[i] can be negative (below ambient) or already above T_FLAME_MAX
        // (a stacked heat source, e.g. combustion/explosion) -- either way
        // the gate must not amplify gain past 1x or go negative (the old
        // gate's "guarded, gain may be negative" footgun this replaces).
        const q16 sat = clamp01_q((q16)fp::FP_ONE -
                                   fp::recip_mul(temperature[i], recip_T_flame_max));
        // gain = gain_q * I * sat * dt. PINNED left-fold; round-to-nearest the final
        // narrow (the deposit). Carry one wide product at the end for the round.
        q16 g = fp::mul_q16(gain_q, I);            // gain_q * I
        g = fp::mul_q16(g, sat);                   // * (1 - T/T_FLAME_MAX)
        // final * dt with ROUND-TO-NEAREST (deposit); g >= 0 now (sat clamped
        // >= 0), so this is always a non-negative deposit.
        const q16 gain = fp::narrow_round_signed(fp::mul_wide(g, dt_q));
        if (gain > 0) {
            q16 dT = fp::narrow_round_signed(fp::mul_wide(gain, temp_gain_scale_q));
            // Belt-and-suspenders hard cap: never deposit PAST T_FLAME_MAX
            // in one tick even if the smooth taper above under-clamps at
            // extreme dt/gain products (the "min(deposit, headroom)" form
            // the investigation named as the alternative shape).
            const q16 headroom = (temperature[i] < t_flame_max_q)
                ? (q16)(t_flame_max_q - temperature[i]) : 0;
            if (dT > headroom) dT = headroom;
            if (dT > 0) {
                temperature[i] = fp::sat_add_q16(temperature[i], dT);
                if (i == dbg_probe_idx) dbg_plume_dT = dT;   // DEBUG probe (temporary)
            }
        }
    }

    // --- Fire produces smoke in neighbouring air tiles (KEPT) ---
    // smoke is Q16.16. The emission delta (smoke_emission*dt*I) is a small positive
    // Q16.16 product, ROUND-TO-NEAREST (unbiased deposit), integer-added — order-free
    // + deterministic.
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const q16 I = fire[y * w + x];
            if (I <= 0) continue;
            // delta = smoke_emission * dt * I (all positive Q16.16). PINNED order;
            // round-to-nearest the final narrow (deposit).
            const q16 ed = fp::mul_q16(emission_q, dt_q);       // emission*dt
            const int64_t wide = fp::mul_wide(ed, I);           // * I (wide for round)
            const q16 delta_q = fp::narrow_round(wide);         // >= 0 (positive deposit)
            for (const auto& d : D4) {
                int ny = y + d[0], nx = x + d[1];
                if (in_bounds(ny, nx, h, w) && !is_wall[ny * w + nx]) {
                    smoke[ny * w + nx] += delta_q;     // Q16.16 integer add
                }
            }
        }
    }

    // --- Fire damages walls, collect destroyed tiles (KEPT — the fuel-consumption
    //     brake: as wall_hp -> 0, F -> 0 in the feedback, the fire starves) ---
    // wall_hp -= wall_damage * dt * fire[i] (all positive Q16.16). The depletion is
    // fractional (< 1 HP/tick) so wall_hp NEEDS the Q16.16 fraction. ROUND-TO-NEAREST
    // the depletion (an unbiased sink off the fuel store). wall_hp <= 0 -> destroyed.
    std::vector<std::pair<int, int>> destroyed;
    for (int i = 0; i < n; ++i) {
        if (fire[i] > 0) {
            const q16 wd = fp::mul_q16(wall_damage_q, dt_q);    // wall_damage*dt
            const int64_t wide = fp::mul_wide(wd, fire[i]);     // * I (wide for round)
            const q16 dmg = fp::narrow_round(wide);             // >= 0 (positive depletion)
            wall_hp[i] -= dmg;
            if (wall_hp[i] <= 0 && flammable[i] && is_wall[i]) {
                destroyed.push_back({i / w, i % w});
                fire[i] = 0;
            }
        }
    }

    // --- Final clamp (fire and smoke only — atmosphere is unclamped) ---
    // fire clamps to [0, FP_ONE] (Q16.16 [0,1]); smoke clamps the same.
    for (int i = 0; i < n; ++i) {
        fire[i] = clamp01_q(fire[i]);
        if (smoke[i] < 0) smoke[i] = 0;
        else if (smoke[i] > fp::FP_ONE) smoke[i] = fp::FP_ONE;
    }

    return destroyed;
}
