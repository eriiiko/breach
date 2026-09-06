#include "fire_simulation.h"
#include "fixed_point.h"
#include <algorithm>
#include <cstdint>

// Continuous O2->combustion law (docs/continuous_o2_law_design_2026-07-24.md):
// the O2 sustain factor is LINEAR in the local O2 mole fraction with an
// extinction limit. Technique credit:
//   Peatross, M.J. & Beyler, C.L., "Ventilation effects on compartment fire
//   behavior", Fire Safety Science 5:403-414, 1997 — compartment burning rate
//   declines ~linearly with O2 volume fraction below ambient (the linear law).
//   (Extinction-limit ~13-16 vol-% O2: Beyler, SFPE Handbook, flammability
//   limits.) Both archived under docs/papers/.

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

// Integer clamp to [0, cap] (R1 renormalized o2f — the enrichment ceiling is a
// CONFIG dial, not FP_ONE, so this is a small generalization of clamp01_q
// above, not a replacement for it: everything else in this file still wants
// the plain [0,1] clamp).
static inline q16 clamp0cap_q(q16 v, q16 cap) {
    if (v < 0) return 0;
    if (v > cap) return cap;
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
// RETAINED past the continuous-O2 law (which replaced the fire O2 gate's use of
// it): kept for the tombstone + any future smoothstep consumer (design §2.2).
[[maybe_unused]] static inline q16 smoothstep_q(q16 edge0, q16 edge1, q16 x,
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
    const q16* n_o2,               // EOS P4: Q16.16 int32 real O2 density (fraction numerator)
    const q16* n_total,            // continuous-O2 law: Q16.16 int32 real N_total (fraction denom)
    int32_t* smoke,               // S2b: Q16.16
    q16* wall_hp,                 // S3b: Q16.16 int32 (was float)
    q16* temperature,             // mutable (signature unchanged); READ only as
                                  // of P-R2 — the plume->T shim write is deleted
                                  // (docs/radiation_raycaster_extinction_ruling_
                                  // 2026-07-31.md A2); P-R4's radiation pass is
                                  // the next writer, through its own rad_net[]
                                  // plane, not this parameter.
    const q16* wind_x,            // S2c/S3b: Q16.16 int32 (read-only)
    const q16* wind_y,            // S2c/S3b: Q16.16 int32 (read-only)
    const bool* is_wall,
    const bool* is_vacuum,
    const bool* flammable,
    int h, int w,
    float dt,
    const int64_t* fuel_recip,     // FUEL-FRACTION AXIS: per-tile 1/hp (nullable)
    const q16* fire_T_ext_plane    // PER-MATERIAL T_ext: per-tile Q16.16 (nullable)
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
    const q16 I_min_q       = fp::quantize((double)p.I_min);
    // CAPACITY LAW (P-R3, ruling A3): INV_C = 1/I_cap_per_avail, computed in
    // DOUBLE at load and quantized once — the LOCKED S1 boundary idiom (a
    // config-constant divide never happens per cell). It is a plain q16 (not a
    // make_recip) because it multiplies a Q16.16 intensity through the same
    // mul_q16 the rest of the tree uses, and 1/2.53 = 0.395 sits comfortably in
    // range. `I_cap_per_avail <= 0` means "capacity ceiling OFF": INV_C = 0
    // makes `gap == avail*hot` and growth unbounded — the deliberate,
    // documented answer to a divide-by-zero misconfig (same spirit as
    // fuel_recip_from_hp(0) == 0 meaning "no fuel"), and the probe idiom the
    // o2f-readout tests use to collapse the multiply chain to a single factor.
    const q16 INV_C = (p.I_cap_per_avail > 0.0f)
        ? fp::quantize(1.0 / (double)p.I_cap_per_avail) : (q16)0;
    const q16 wall_damage_q = fp::quantize((double)p.wall_damage);
    // R3 (fire session #12, docs/fire_3c_design_2026-09-01.md "Ruling R3"):
    // the ceiling on the destruction/demand-side hotf ramp (see the wall-burn
    // site below and combustion.cpp's demand site). Same load-time idiom as
    // o2f_cap_q above.
    const q16 hotf_cap_q    = fp::quantize((double)p.hotf_cap);

    // Load-time reciprocals for the config-constant divides (make_recip/recip_mul).
    // FUEL-FRACTION AXIS (2026-07-30): the FALLBACK fuel normaliser, used only
    // when the caller supplies no per-tile `fuel_recip` plane. With a plane, F
    // divides by THIS tile's material's own full-health hp (baked with the very
    // same make_recip, in GameMap.fuel_recip) — see the header's `fuel_ref`
    // tombstone for why one global could not serve wood (hp 60) and a crate
    // (hp 30) at once. Uniform plane == this scalar -> byte-identical.
    const int64_t recip_fuel_ref  = fp::make_recip((double)p.fuel_ref);       // fallback 1/hp
    const int64_t recip_T_span    = fp::make_recip((double)p.fire_T_span);    // hot ramp
    // Continuous-O2 law span, R1 RENORMALIZATION (fire session #12, docs/
    // fire_3c_design_2026-09-01.md "Ruling R1"):
    //   o2f = clamp((X - X_ext) / (X_amb - X_ext), 0, o2f_cap)
    // recip_x_span is the load-time reciprocal of the span (like recip_T_span);
    // X_ext = 0 gives span == X_amb (pure-proportional X/X_amb, NOT degenerate).
    // X_span <= 0 (X_amb <= X_ext, a misconfig) -> a step at X_ext. The upper
    // end is now the AMBIENT reference o2_frac_amb, NOT the pure-O2 reference
    // o2_frac_full (R1 supersedes the 2026-07-30 FULL-RESPONSE REFERENCE SPLIT
    // for the SUSTAIN law only — o2_frac_full stays live on the DEMAND side,
    // combustion.cpp's o2f_j, unchanged): ambient air now reads o2f == 1.0 by
    // construction, and the clamp's upper edge is the NEW o2f_cap dial (not
    // FP_ONE) so enrichment above ambient can still register, bounded.
    const q16 x_ext_q             = fp::quantize((double)p.o2_frac_ext);
    const double  x_span          = (double)p.o2_frac_amb - (double)p.o2_frac_ext;
    const bool    x_degenerate    = (x_span <= 0.0);
    const int64_t recip_x_span    = x_degenerate ? 0 : fp::make_recip(x_span);
    const q16 o2f_cap_q           = fp::quantize((double)p.o2f_cap);
    // Mole-fraction divide floor: den = max(Σn_total, X_N_FLOOR). Guards the
    // per-cell reciprocal_q16 (undefined at denom <= 0, spurious at {1,2}) AND
    // makes a near-vacuum cell (open-neighbour total gas < 1% of one ambient
    // cell) read X ~= 0 rather than a spurious high mole fraction on trace gas.
    // A legit thermally-expanded flame-edge cell holds N ~= 0.07-0.09 (config
    // note) -> Σ >> this floor, so it never engages for a real burn. Same value
    // host + device (the fraction division must be bit-identical CPU<->CUDA).
    const q16 X_N_FLOOR           = fp::quantize(0.01);   // 655 counts

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

        // F: fuel from remaining wall HP, normalised by THIS TILE'S OWN full
        // health (fuel-fraction axis, 2026-07-30). recip_mul by the load-time
        // reciprocal — per tile from the `fuel_recip` plane when supplied, else
        // the scalar fuel_ref fallback — then clamp01. Still ONE multiply and a
        // shift per cell: `fuel_recip` is baked at LOAD, exactly as the scalar
        // is, so the sim path keeps its no-divide / no-libm contract.
        const int64_t recip_fuel = fuel_recip ? fuel_recip[i] : recip_fuel_ref;
        const q16 F = clamp01_q(fp::recip_mul(wall_hp[i], recip_fuel));

        // X: local O2 MOLE FRACTION over OPEN (non-solid, non-vacuum)
        // 4-neighbours — the fire reads INCOMING fresh air (its own tile holds
        // no gas — it is solid). BOTH sums are int64, exact, order-free (the
        // same open-neighbour predicate the Python ignition twin shares). The
        // continuous-O2 law reads the FRACTION Σn_o2/Σn_total, NOT the absolute
        // n_o2 mean: invariant under thermal expansion, so hot thin gas at
        // ambient composition burns (closes the density trap; design §2.1).
        // No open neighbour -> both sums 0 -> den floors -> X = 0.
        int64_t sum_o2 = 0;
        int64_t sum_tot = 0;
        for (const auto& d : D4) {
            int ny = y + d[0], nx = x + d[1];
            int ni = ny * w + nx;
            if (in_bounds(ny, nx, h, w) && !is_wall[ni] && !is_vacuum[ni]) {
                sum_o2  += (int64_t)n_o2[ni];      // exact, order-free
                sum_tot += (int64_t)n_total[ni];   // exact, order-free
            }
        }
        // X = Σn_o2 / max(Σn_total, floor), ONE per-cell reciprocal_q16 divide
        // (the SAME primitive combustion's heat deposit uses — bit-identical
        // CPU<->CUDA). Σn_total <= ~4*N_max fits q16 comfortably; sum_o2 <=
        // sum_tot so X <= 1 and the product never overflows (design §2.1).
        const q16 den = (sum_tot < (int64_t)X_N_FLOOR) ? X_N_FLOOR : (q16)sum_tot;
        const q16 X = fp::mul_q16((q16)sum_o2, fp::reciprocal_q16(den));

        // W: wind magnitude from the SHARED wind field (= -grad p incl. waves, so a
        // grenade shockwave is a transient spike -> firestorm / blow-out). The int64
        // Q.32 radicand (wx^2 + wy^2), then the new floor-isqrt sqrt_q16 -> Q16.16.
        const int64_t rad = fp::mul_wide(wind_x[i], wind_x[i])
                          + fp::mul_wide(wind_y[i], wind_y[i]);
        const q16 W = fp::sqrt_q16(rad);

        // Gates. `hot` reads THIS TILE'S OWN extinction temperature (ruling A3
        // ride-along, 2026-07-31): per tile from the `fire_T_ext_plane` when
        // supplied — that material's `ignition_temp - ignition_to_ext_delta`,
        // quantized at LOAD in GameMap.fire_T_ext_plane — else the scalar
        // `fire_T_ext` fallback, which is the pre-derivation law bit-for-bit.
        // `fire_T_span` stays GLOBAL, so this is still ONE subtract + ONE
        // recip_mul + a clamp: the sim path keeps its no-divide contract.
        const q16 T_ext_i = fire_T_ext_plane ? fire_T_ext_plane[i] : fire_T_ext_q;
        const q16 hot = clamp01_q(fp::recip_mul(T - T_ext_i, recip_T_span));
        // o2f is LINEAR in X (the continuous-O2 law), R1 RENORMALIZED: X <= X_ext
        // -> 0 (extinction), X == X_amb -> 1 (ambient, by construction — NOT the
        // old pure-O2-normalized 0.092), clamped at the NEW o2f_cap ceiling (not
        // FP_ONE) so locally enriched O2 above ambient can still register,
        // bounded. The degenerate span (X_amb <= X_ext misconfig) falls back to a
        // step at X_ext (0 or the cap, since "full response" now means the cap).
        const q16 o2f = x_degenerate
            ? ((X < x_ext_q) ? (q16)0 : o2f_cap_q)
            : clamp0cap_q(fp::recip_mul(X - x_ext_q, recip_x_span), o2f_cap_q);
        const q16 avail = fp::mul_q16(F, o2f);

        // Signed logistic update, fanned + stripped by wind. PINNED MULTIPLY ORDER
        // (master plan §2.4 / plan §5.2 — the chained-truncation association hazard):
        // a LEFT-FOLD of mul_q16, each narrowing once, in this EXACT sequence on
        // every peer -> deterministic by construction.
        //
        // THE CAPACITY LAW (P-R3, ruling A3): the growth term's capacity factor
        // is no longer the hardwired `(1 - I)` — it is the RESOURCE-PROPORTIONAL
        // gap `avail*hot - I/c`, i.e. the logistic `k_grow*a*I*(1 - I/(c*a))`
        // with `a = avail*hot` cancelled out of the bracket (which is why no
        // division survives). THE NEW PINNED SEQUENCE, in this exact order:
        //
        //   gap  = avail*hot - mul_q16(I, INV_C)                  (SIGNED sub)
        //   grow = k_grow;  grow = mul(grow, I);  grow = mul(grow, gap);
        //                   grow = mul(grow, wind_fan)
        //
        // `gap` may be NEGATIVE — a fire sitting above its (just-shrunken)
        // capacity, e.g. the tick after local O2 drops. That makes `grow`
        // negative and the fire decays toward the new capacity through the
        // existing signed-delta path; nothing extra is needed, and mul_q16's
        // truncation-toward-minus-infinity is the same convention the whole
        // tree already uses. `avail_hot` is hoisted ABOVE the growth block (it
        // was computed for `die` below) because both terms now read it — one
        // multiply, one value, no chance of the two disagreeing.
        const q16 avail_hot = fp::mul_q16(avail, hot);          // avail*hot
        const q16 gap = avail_hot - fp::mul_q16(I, INV_C);      // - I/c (SIGNED)
        const q16 one_minus_I = (q16)fp::FP_ONE - I;             // (1 - I) — wind strip only
        const q16 wind_fan = (q16)fp::FP_ONE + fp::mul_q16(k_wind_fan_q, W);  // (1 + k_wind_fan*W)
        q16 grow = k_grow_q;                       // k_grow
        grow = fp::mul_q16(grow, I);               // * I
        grow = fp::mul_q16(grow, gap);             // * gap   (signed)
        grow = fp::mul_q16(grow, wind_fan);        // * (1 + k_wind_fan*W)

        //   die = k_die * max(0, 1 - avail*hot) * I  +  k_wind_strip * W * (1 - I) * I
        //   R1 DIE-TERM SIGN FIX: avail*hot can now exceed 1 (O2 enrichment above
        //   ambient, since o2f is no longer clamped to [0,1] — see o2f above), which
        //   would flip (1 - avail*hot) NEGATIVE — an ANTI-DEATH term (die going
        //   negative would ADD to I). max(0, .) floors it at zero: enrichment can
        //   only help a fire through grow/I_cap, never subtract from die.
        const q16 one_minus_ah_raw = (q16)fp::FP_ONE - avail_hot;    // (1 - avail*hot), SIGNED
        const q16 one_minus_ah = std::max((q16)0, one_minus_ah_raw); // R1: floored at 0
        q16 die_a = k_die_q;                       // k_die
        die_a = fp::mul_q16(die_a, one_minus_ah);  // * max(0, 1 - avail*hot)
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

    // --- Fire-step smoke scatter DELETED (P-S1, 2026-08-15) -----------------
    // Erik's single-source ruling (docs/smoke_single_source_design_2026-07-24.md,
    // 2026-07-24): fire smoke has ONE source — the physically-bookkept
    // combustion soot channel (`soot_yield`, cpp/src/combustion.cpp) — not two.
    // This block used to add `smoke_emission*dt*I` to each lit tile's open
    // 4-neighbours EX NIHILO (no debit anywhere), which the storm audit
    // (docs/storm_audit_2026-08-14.md §4.2) traced into a real mass/pressure
    // pump: the P4 decay->inert_N2 credit (physics_engine.cpp) then converted
    // that unbacked trace mass into full-pressure-weight bulk N2 — two
    // individually-defensible rules composing into a sealed room gaining
    // +42% of its bulk gas inventory in 200s. See
    // docs/smoke_single_source_asbuilt_2026-08-15.md for the as-built record
    // and measured before/after numbers. The CUDA mirror (cuda_fire.cu's
    // fire_smoke_emit kernel) is deleted the same way. `smoke_emission` is
    // retired from FireParams/config/bindings entirely (a stale key now
    // loud-errors at load, src/simulation/physics_runner.py).

    // --- Fire damages walls, collect destroyed tiles (KEPT — the fuel-consumption
    //     brake: as wall_hp -> 0, F -> 0 in the feedback, the fire starves) ---
    // wall_hp -= wall_damage * dt * fire[i] (all positive Q16.16). The depletion is
    // fractional (< 1 HP/tick) so wall_hp NEEDS the Q16.16 fraction. ROUND-TO-NEAREST
    // the depletion (an unbiased sink off the fuel store). wall_hp <= 0 -> destroyed.
    std::vector<std::pair<int, int>> destroyed;
    for (int i = 0; i < n; ++i) {
        if (fire[i] > 0) {
            // R3 hot-burns-faster (fire session #12, docs/fire_3c_design_2026-
            // 09-01.md "Ruling R3"): destruction now rides `hotf` — the SAME
            // (T - T_ext)/T_span ramp as `hot` above, but capped at hotf_cap
            // instead of 1, so a saturated fire destroys its own fuel bed
            // FASTER, not just draws O2 faster (combustion.cpp's demand-side
            // twin reads the identical law). Recomputed HERE rather than
            // threaded from the sustain loop above, because THIS loop is not
            // flammable-gated — it runs on every fire[i] > 0 tile, including a
            // non-flammable "ghost" fire (tests/_xarch_perfield_digest.py's
            // P-R4 lineage note), which already pays wall_damage*dt*I
            // unconditionally today; hotf simply rides along on the same
            // unconditional term. `hot`'s clamp01_q gate above is UNCHANGED —
            // this is a separate, uncapped-at-1 read of the identical ramp.
            const q16 T_ext_i = fire_T_ext_plane ? fire_T_ext_plane[i] : fire_T_ext_q;
            const q16 T_i = temp_is_identity
                ? temperature[i]
                : fp::recip_mul(temperature[i], recip_temp_scale);
            const q16 hotf = clamp0cap_q(fp::recip_mul(T_i - T_ext_i, recip_T_span),
                                         hotf_cap_q);
            const q16 wd = fp::mul_q16(fp::mul_q16(wall_damage_q, dt_q), hotf);
                                                                 // wall_damage*dt*hotf
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
