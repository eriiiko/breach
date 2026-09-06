#pragma once
// Fire simulation — signed-logistic intensity FEEDBACK, smoke emission, plume
// pressure deposit, wall burn-through (engine/06 §5 + fire_design_proposal §2/§3).
//
// Spread is NO LONGER cellular: it comes entirely from radiation -> heat ->
// temperature -> ignition (apply_temperature_ignition, wired in the sim). This
// step is purely the per-tile life/death of an ALREADY-lit fire:
//
//   T     = temperature[i]                         (Q16.16; temp_scale == FP_ONE)
//   F     = clamp01(wall_hp[i] / hp_full[i])       (fuel fraction of THIS tile's
//                                                   OWN material — the per-tile
//                                                   `fuel_recip` plane below;
//                                                   falls back to the global
//                                                   fuel_ref when absent)
//   X     = Σn_o2 / Σn_total  over OPEN (non-solid, non-vacuum) 4-neighbours
//                                                  (local O2 MOLE FRACTION)
//   W     = sqrt(wind_x^2 + wind_y^2)              (the SHARED wind field)
//   hot   = clamp01((T - T_ext[i]) / T_span)       (T_ext is PER MATERIAL — the
//                                                   `fire_T_ext_plane` below,
//                                                   baked ignition_temp[mat] -
//                                                   ignition_to_ext_delta; falls
//                                                   back to the global scalar
//                                                   `fire_T_ext` when absent.
//                                                   T_span STAYS global)
//   o2f   = clamp((X - o2_frac_ext)/(o2_frac_amb - o2_frac_ext), 0, o2f_cap)
//                                                   (LINEAR; the continuous-O2
//                                                   law — Peatross & Beyler
//                                                   1997; REPLACES the old
//                                                   smoothstep(P_min,P_full) on
//                                                   ABSOLUTE n_o2 density. R1
//                                                   RENORMALIZATION (fire
//                                                   session #12, docs/fire_3c_
//                                                   design_2026-09-01.md): the
//                                                   denominator's upper end is
//                                                   the AMBIENT reference
//                                                   o2_frac_amb (0.21), NOT
//                                                   pure O2 — so ambient air
//                                                   reads o2f == 1.0, not
//                                                   0.092. The clamp's UPPER
//                                                   edge is no longer 1 —
//                                                   `o2f_cap` (5.0) is the
//                                                   enrichment flare ceiling,
//                                                   since a raw pure-O2 tile
//                                                   would otherwise reach
//                                                   10.875)
//   hotf  = clamp((T - T_ext[i])/T_span, 0, hotf_cap)  (R3 "hot-burns-faster",
//                                                   fire session #12, docs/
//                                                   fire_3c_design_2026-09-01.md
//                                                   "Ruling R3": the SAME ramp
//                                                   as `hot` above, SAME T_ext/
//                                                   T_span, but the ceiling is
//                                                   `hotf_cap` (10.0) instead of
//                                                   1 — `hot` itself STAYS
//                                                   clamped at 1 and keeps
//                                                   gating the I-ODE sustain
//                                                   term below UNCHANGED. hotf
//                                                   is read ONLY by the two
//                                                   RATE sites it was built
//                                                   for: combustion.cpp's
//                                                   demand and the wall-burn
//                                                   term below — a saturated
//                                                   fire draws O2 and destroys
//                                                   its fuel bed faster, not
//                                                   just "burns" in the I-ODE
//                                                   sense)
//   avail = F * o2f                                 (can now exceed 1 — O2
//                                                   enrichment above ambient)
//   gap   = avail*hot - I / I_cap_per_avail          (SIGNED — negative when the
//                                                     fire sits ABOVE its own,
//                                                     resource-sized capacity)
//   grow  = k_grow * I * gap * (1 + k_wind_fan * W)
//   die   = k_die * max(0, 1 - avail*hot) * I  +  k_wind_strip * W * (1-I) * I
//                                                   (R1 DIE-TERM SIGN FIX: avail*hot
//                                                   can now exceed 1 under enrichment,
//                                                   which would flip (1-avail*hot)
//                                                   negative -- ANTI-DEATH. The max(0,.)
//                                                   floors it: enrichment can only help
//                                                   through grow/I_cap, never subtract
//                                                   from die.)
//   I    += dt * (grow - die);  clamp01;  snap to 0 below I_min
//
// THE CAPACITY LAW (P-R3, 2026-07-31 — docs/radiation_raycaster_extinction_
// ruling_2026-07-31.md A3, on Erik's ruling R-b). The growth term's carrying
// capacity used to be the hardwired constant 1 (the `(1-I)` factor). It is now
// RESOURCE-PROPORTIONAL: `I_cap = I_cap_per_avail * avail * hot`, i.e.
//     grow = k_grow * avail*hot * I * (1 - I/I_cap)
// with `avail*hot` cancelling out of the bracket, which is why the implemented
// form carries no division. The fixed point becomes
//     I_eq = c * (a - r*(1-a)),   a = avail*hot,  r = k_die/k_grow,  c = I_cap_per_avail
// and the sustain threshold keeps its old shape `a > r/(1+r)`.
//
// WHY (the defect it closes): under the old law `r` set BOTH the equilibrium
// intensity AND the extinction wall — `I_eq = 1 - r(1-a)/a` — so asking for a
// small fire (I_eq 0.21) forced `r` up against the operating point and left the
// fire only 1.242x of headroom on the product `F*o2f*hot`. Measured consequences
// (ruling §5): a crate could never lose more than 19.5% of its hp before dying
// (fuel-governed death was unreachable), and the literature-anchored O2
// extinction limit `o2_frac_ext` = 0.13 was DEAD CODE because the logistic wall
// bit first, at X = 0.1944. Moving size into `c` gives each dial exactly one job:
// `c` = size, `k_grow` = tempo, `k_die` = where the death wall sits. Same
// tombstone shape as `fuel_ref` / `cool_shift` / `o2_frac_amb` before it: one
// parameter that had been doing two jobs, split.
//
// Pressure (replaces the old O2-consumption subtraction, which sucked smoke IN):
//   atmosphere[i] += max(fire_pressure_gain * I * (1 - atmosphere[i]/p_expand_ref) * dt, 0)
// An OWN-tile overpressure -> wind = -grad p points OUTWARD -> smoke pushed away.
// The sustain read P is the NEIGHBOUR mean, so the fire reads incoming fresh air,
// not its own bump.
//
// Determinism (S3b): the whole logistic is INTEGER Q16.16 (fire/wall_hp int32,
// atmosphere/wind/temperature int32). Cross-machine bit-identical: integer
// +/-/*/>> are exact + associative, and the per-cell sqrt is a fixed-iteration
// floor-isqrt (fixed_point.h::sqrt_q16, the arc's first per-cell transcendental).
// The multiply tree order is PINNED (left-fold mul_q16); the wall-burn deposit
// ROUND-TO-NEARESTs (an unbiased source — the plume deposit was deleted at P-R2,
// the smoke-emission scatter at P-S1; wall-burn is the sole survivor of that
// family). The discrete outputs (the I_min extinguish flip, the wall_hp<=0
// burn-through list) are integer compares -> bit-deterministic.

#include <vector>
#include <utility>
#include <cstdint>

struct FireParams {
    // --- signed-logistic feedback (fire_design_proposal §2) ---
    float k_grow         = 4.0f;   // logistic growth gain (1/s) — TEMPO only now
    float k_die          = 2.0f;   // decay rate when starved/cold (1/s)
    // CAPACITY LAW (P-R3, ruling A3): the growth term's carrying capacity per
    // unit availability, `I_cap = I_cap_per_avail * avail * hot`. THE SIZE DIAL
    // — the ONLY thing that sets how big a fire gets at a given resource level
    // (`I_eq ~= c*a`), leaving `k_grow` free to mean tempo and `k_die` free to
    // put the death wall at the physical limits. Its reciprocal is baked ONCE
    // at load (`INV_C = quantize(1/c)`, the S1 double-then-quantize boundary
    // idiom) so the sim path stays divide-free. `<= 0` is legal and means
    // "capacity ceiling OFF" (INV_C = 0 -> unbounded growth): a deliberate
    // guard against a divide-by-zero misconfig, and the probe idiom the
    // o2f-readout tests use to collapse the multiply chain.
    float I_cap_per_avail = 2.53f; // c — capacity per unit availability
    // fire_T_ext: the FALLBACK extinction temperature, used only when the
    // caller supplies no per-tile `fire_T_ext_plane` (ruling A3's ride-along,
    // 2026-07-31). It was ONE GLOBAL standing in for a PER-MATERIAL quantity —
    // `fire_T_ext` sits on the same axis as the per-material `ignition_temp`,
    // and the shipped 350 exceeds BOTH shipped ignition temps (wood 300,
    // furniture 280), so a tile could ignite below its own sustain floor and
    // snap straight back out. It is now DERIVED per material,
    // `fire_T_ext[mat] = ignition_temp[mat] - ignition_to_ext_delta`, which
    // makes the invariant `fire_T_ext < ignition_temp` STRUCTURAL instead of a
    // thing a config author has to remember. Same FALLBACK-only tombstone shape
    // as `fuel_ref` below. `fire_T_span` deliberately stays GLOBAL: it is the
    // width of the ramp, not its foot.
    float fire_T_ext     = 350.0f; // FALLBACK extinction temperature (no plane)
    float fire_T_span    = 150.0f; // width of the `hot` ramp above T_ext (GLOBAL)
    // fuel_ref: SUPERSEDED as the fuel normaliser by the per-tile `fuel_recip`
    // plane (fuel-fraction axis, 2026-07-30 — see FireSimulation::step). It was
    // ONE GLOBAL standing in for a PER-MATERIAL quantity: F is meant to be "the
    // fraction of THIS tile's fuel remaining", and 60.0 is WOOD's hp, so a
    // full-health furniture crate (hp 30) permanently read F = 0.5 — half burnt
    // out before it was ever lit, and below the sustain ceiling at ambient O2 at
    // any intensity. Lowering it is NOT the fix (at 30, wood would clamp at
    // F = 1 until it had lost half its mass). KEPT, and still live, in exactly
    // one role: the FALLBACK divisor when a caller passes no `fuel_recip` plane
    // (nullptr), which is the pre-axis behaviour bit-for-bit. Same tombstone
    // shape as `o2_frac_amb` above.
    float fuel_ref       = 60.0f;  // fallback wall_hp normaliser (no plane)
    // --- continuous O2 law (docs/continuous_o2_law_design_2026-07-24.md) -------
    // The O2 factor is LINEAR in the local O2 MOLE FRACTION X = Σn_o2/Σn_total
    // over open neighbours (Peatross & Beyler 1997: compartment burning rate
    // declines ~linearly with O2 volume fraction), carrying an extinction limit:
    //   o2f = clamp((X - o2_frac_ext) / (o2_frac_amb - o2_frac_ext), 0, o2f_cap)
    // This REPLACES the old smoothstep(P_min, P_full) on ABSOLUTE n_o2 density —
    // the fraction is invariant under thermal expansion, so hot thin gas at
    // ambient composition burns (closes the v2.4 "density trap" / hot-zone rescale
    // saga; only true vitiation starves a fire). P_min/P_full below are RETIRED
    // from the sustain law (tombstoned — kept only so old configs/bindings that
    // still set them do not hard-error; no longer read by step()).
    float o2_frac_ext    = 0.13f;  // X_ext: flame-extinction O2 mole fraction
                                   //  (~13% physical limit; 0 = pure proportional)
    // R1 O2f-RENORMALIZATION (fire session #12, 2026-09-01 — docs/fire_3c_
    // design_2026-09-01.md "Ruling R1"). The SUSTAIN span's upper reference is now
    // o2_frac_amb (ambient, 0.21) instead of the FULL-RESPONSE pure-O2 reference
    // o2_frac_full below: ambient air always produced o2f == 0.092 under the old
    // pure-O2 normalization (every OTHER dial — I_cap_per_avail, k_die — was
    // secretly compensating for that near-zero baseline), so a mild local O2 dip
    // (0.21 -> 0.165) halved an already-tiny o2f and collapsed heat deposit long
    // before the physical o2_frac_ext=0.13 extinction limit engaged (measured
    // X_death 0.176, rising — docs/fire_3c_prebench_2026-09-01.md). Renormalizing
    // by ambient makes o2f == 1.0 AT ambient by construction, so a fire's heat
    // deposit holds up near the true O2 floor instead of dying of cold well above
    // it. o2f is NO LONGER clamped to [0,1] — see `o2f_cap` below, the new upper
    // bound (enrichment above ambient can now register, e.g. O2 reservoirs/leaks).
    // o2_frac_full is RETIRED from THIS law (tombstone — see below); the DEMAND
    // side (combustion.cpp's o2f_j, "how fast it drinks") is UNCHANGED and still
    // reads o2_frac_full: two roles, two shapes, by ruling.
    // o2f_cap: the enrichment ceiling on the renormalized ratio. Ambient air now
    // lands at (0.21-0.13)/(0.21-0.13) = 1.0 exactly; a raw pure-O2 tile would
    // reach (1.0-0.13)/(0.21-0.13) = 10.875 -- capped at 5.0 (Erik's choice, the
    // "enrichment flare" ceiling) so a locally O2-flooded tile can burn hotter
    // than ambient without an unbounded runaway.
    float o2f_cap         = 5.0f;  // NEW (R1): upper clamp on the renormalized o2f
    // hotf_cap (R3, fire session #12, docs/fire_3c_design_2026-09-01.md "Ruling
    // R3" — Erik's "hot-burns-faster" catch: `hot` clamps at 1, so above ~T_ext
    // + T_span extra heat bought ZERO extra rate). `hotf` is the SAME (T-T_ext)/
    // T_span ramp as `hot`, but its ceiling is THIS dial instead of 1, and it is
    // read only at the two RATE sites (combustion.cpp's demand, and this file's
    // wall-burn term below) — `hot` itself is UNTOUCHED and keeps gating the
    // I-ODE sustain term at its usual [0,1]. WHY LINEAR, NOT ARRHENIUS (the
    // load-bearing argument, R2's finding): losses are dominated by T^4
    // radiation (~99.6% at the measured plateau), so a LINEAR hotf can never
    // outrun them — an equilibrium always exists by construction, and Erik's O2
    // cap (o2f_cap above) is a second, independent ceiling on an already-stable
    // system. An exponential (Arrhenius) hotf eventually beats T^4, making
    // stability depend on O2 running out first — fragile exactly where oxygen
    // is generous (breach airflow, enriched rooms). 10.0 pairs with o2f_cap's
    // 5.0 as the OTHER enrichment/overheat ceiling in the system.
    float hotf_cap        = 10.0f; // NEW (R3): ceiling on the uncapped-at-1 hotf ramp
    // o2_frac_full: RETIRED from the SUSTAIN law by R1 (was the FULL-RESPONSE
    // REFERENCE SPLIT's pure-O2 upper reference, 2026-07-30). Kept, like
    // o2_frac_amb was before it, ONLY because combustion.cpp's DEMAND-side o2f_j
    // (unchanged by R1) still reads it, and old configs/bindings that still set
    // it should not hard-error. Editing it moves the DEMAND (drink-rate) law, not
    // the sustain law above (which now reads o2_frac_amb).
    float o2_frac_full   = 1.00f;  // X_full: DEMAND-side (combustion.cpp) pure-O2
                                   //  reference only — NOT read by step() below.
    float o2_frac_amb    = 0.21f;  // X_amb: what the ambient atmosphere IS (reads
                                   //  the level's authored [ambient] o2_frac; 0.21
                                   //  fallback — one source of truth with the BC).
                                   //  R1: NOW LIVE in step() below — the sustain
                                   //  availability law's span upper reference.
    float P_min          = 0.60f;  // RETIRED (see o2_frac_ext/amb above) — was the
                                   //  smoothstep low edge on absolute n_o2
    float P_full         = 1.00f;  // RETIRED — was the smoothstep full edge
    float I_min          = 0.02f;  // snap-to-zero extinguish floor

    // --- wind coupling (fire_design_proposal §5; Erik's addition) ---
    // k_wind_fan / k_wind_strip are scaled against the shared wind field's
    // magnitude; both NEED TUNING vs the live wind scale (a shockwave is a large
    // transient spike). Defaults are deliberately gentle.
    float k_wind_fan     = 0.5f;   // (1 + k_wind_fan*W) fans growth (firestorm)
    float k_wind_strip   = 0.5f;   // W*(1-I)*I blows out small/marginal fires

    // --- plume pressure deposit (fire_design_proposal §3) ---
    // p_expand_ref: RETIRED as the plume's saturation gate (eos-p3fix-
    // thermal-ceiling; the plume deposit itself — the `fire_pressure_gain` /
    // `temp_gain_scale` / `T_FLAME_MAX` trio that used to live here — was
    // DELETED at P-R2, docs/radiation_raycaster_extinction_ruling_2026-07-
    // 31.md A2: it was the one `temperature[]` writer bypassing
    // `heat_inv_shift`); kept only so old configs/bindings that still set it
    // do not hard-error. No longer read by step().
    float p_expand_ref       = 1.30f;

    // --- kept behaviours ---
    // smoke_emission TOMBSTONE (P-S1, 2026-08-15): dead key — the ex-nihilo
    // smoke scatter it fed was DELETED (Erik's single-source ruling, docs/
    // smoke_single_source_design_2026-07-24.md; killed the smoke->N2 pressure
    // pump, docs/storm_audit_2026-08-14.md §4.2). Combustion soot
    // (`soot_yield`, cpp/src/combustion.cpp) is now the ONE fire-smoke
    // source. Unlike the other TOMBSTONEs on this struct (p_expand_ref,
    // P_min/P_full, ...), this key is NOT left wired for back-compat: a
    // stale config still carrying it loud-errors at load
    // (src/simulation/physics_runner.py) rather than silently doing
    // nothing, so nobody tunes a dial that no longer has a mechanism behind
    // it. See docs/smoke_single_source_asbuilt_2026-08-15.md.
    // R3 (fire session #12, docs/fire_3c_design_2026-09-01.md "Ruling R3"):
    // the destruction term is now `wall_damage*dt*I*hotf` (step() below), not
    // `wall_damage*dt*I` — a saturated fire (hotf > 1) destroys its own fuel
    // bed FASTER, not just draws O2 faster (combustion.cpp's demand-side
    // twin). NEUTRAL-LANDING RE-SIZE (same ruling, both dials by the SAME
    // factor f_ref = 2.0717, derived from the post-R1 open-control plateau
    // T=452.9 game and furniture's fire_T_ext=80/fire_T_span=180 ->
    // hotf(452.9) = (452.9-80)/180 = 2.0717): 0.03 -> 0.03/2.0717 = 0.01448,
    // so destruction at the REFERENCE plateau is unchanged and only moves
    // above/below it (config.toml carries the shipped value + full
    // derivation).
    float wall_damage    = 0.4f;   // wall HP lost per second per unit intensity
                                   //  (burn-through IS the fuel-consumption brake)
                                   //  TIMES hotf since R3 (see above)

    // Q16.16 scale of the `temperature` field (== HEAT_SCALE / TEMP_SCALE). Fixed
    // at construction; exposed so Python/config and C++ never disagree.
    float temp_scale     = 65536.0f;
};

class FireSimulation {
public:
    FireParams params;

    // Returns vector of (y, x) coordinates where walls burned through.
    // Python must call destroy_wall() for each of these.
    //
    //   fire        : int32 (h, w) Q16.16 intensity in [0,1], mutated in place (S3b).
    //   atmosphere  : int32 (h, w) Q16.16 (S2c) == P (EOS P3), READ-ONLY (the
    //                 plume's own-tile saturation gate ONLY, since EOS P4 —
    //                 see `n_o2` below for the O2 gate). The plume no longer
    //                 writes it — see `temperature` below.
    //   n_o2        : int32 (h, w) Q16.16 (EOS refactor P4, design §6): the
    //                 REAL bulk O2 density plane (gmap.gas[O2]), READ-ONLY —
    //                 the neighbour-mean O2 gate's input, REPLACING the old
    //                 atmosphere/P proxy. Solid cells hold 0 (no gas), matching
    //                 `atmosphere`'s pre-P4 convention there.
    //   smoke       : int32 (h, w) Q16.16 (S2b), mutable but READ-mostly as of
    //                 P-S1 (2026-08-15) — the fire step no longer ADDS to it
    //                 (the ex-nihilo emission scatter is deleted; combustion
    //                 soot is the ONE fire-smoke source, cpp/src/combustion.cpp).
    //                 Still passed through the final [0, FP_ONE] clamp below,
    //                 alongside `fire`.
    //   wall_hp     : int32 (h, w) Q16.16 (S3b), burn-through depletes it (the fuel
    //                 brake); fractional depletion needs the Q16.16 fraction.
    //   temperature : int32 (h, w) Q16.16, READ (the T gate) ONLY as of P-R2 —
    //                 the plume->T shim deposit that used to write it here is
    //                 DELETED (docs/radiation_raycaster_extinction_ruling_
    //                 2026-07-31.md A2); P-R4's radiation pass will be the
    //                 next writer, through its own rad_net[] plane, not this
    //                 parameter.
    //   wind_x/wind_y : int32 (h, w) Q16.16 (S2c), the SHARED wind field (= -grad p
    //                 incl. waves), READ-ONLY (the W = |wind| term, via sqrt_q16).
    //   is_wall     : bool (h, w) solid mask (a fire tile is itself solid).
    //   is_vacuum   : bool (h, w) vacuum mask (excluded from the O2 neighbour mean).
    //   flammable   : bool (h, w) fuel mask (fire only lives on fuel).
    //   fuel_recip  : int64 (h, w) OPTIONAL (nullptr -> the scalar `fuel_ref`
    //                 fallback, i.e. the pre-axis law bit-for-bit). The
    //                 FUEL-FRACTION AXIS (2026-07-30): per tile, the
    //                 `fixedpoint::make_recip` reciprocal of that tile's
    //                 MATERIAL's full-health hp, baked once at load in
    //                 GameMap.fuel_recip. F = clamp01(recip_mul(wall_hp[i], r))
    //                 with r taken per tile, so a crate reads its own fuel
    //                 fraction instead of wood's. int64 because a RECIP_SHIFT=32
    //                 reciprocal exceeds int32 for small divisors; the runtime
    //                 op is still ONE multiply (recip_mul), NO divide.
    //   fire_T_ext_plane : int32 (h, w) OPTIONAL (nullptr -> the scalar
    //                 `fire_T_ext` fallback, i.e. the pre-derivation law
    //                 bit-for-bit). PER-MATERIAL EXTINCTION TEMPERATURE
    //                 (ruling A3 ride-along, 2026-07-31): per tile, that
    //                 material's `ignition_temp - ignition_to_ext_delta`,
    //                 QUANTIZED to Q16.16 once at load in GameMap.fire_T_ext_
    //                 plane, so `hot = clamp01((T - plane[i]) * recip_T_span)`
    //                 is the same one multiply + clamp it always was. Same
    //                 nullable-plane idiom as `fuel_recip` above: a UNIFORM
    //                 plane holding quantize(fire_T_ext) is byte-identical to
    //                 passing no plane at all.
    std::vector<std::pair<int, int>> step(
        int32_t* fire,             // S3b: Q16.16 (was float)
        const int32_t* atmosphere, // S2c: Q16.16 == P (EOS P3: READ-ONLY, plume only)
        const int32_t* n_o2,       // EOS P4: Q16.16 real O2 density (mole-fraction numerator)
        const int32_t* n_total,    // continuous-O2 law: Q16.16 real N_total (Σ conservative
                                   //  bulk planes = O2+N2), READ-ONLY — the mole-fraction
                                   //  DENOMINATOR (X = Σn_o2/Σn_total over open neighbours)
        int32_t* smoke,            // S2b: Q16.16 (fire emission round + added)
        int32_t* wall_hp,          // S3b: Q16.16 (was float)
        int32_t* temperature,      // mutable (signature unchanged); READ only
                                   //  as of P-R2 — the plume->T shim write is
                                   //  deleted
        const int32_t* wind_x,     // S2c/S3b: Q16.16
        const int32_t* wind_y,     // S2c/S3b: Q16.16
        const bool* is_wall,
        const bool* is_vacuum,
        const bool* flammable,
        int h, int w,
        float dt,
        const int64_t* fuel_recip = nullptr,  // FUEL-FRACTION AXIS (see above)
        const int32_t* fire_T_ext_plane = nullptr  // PER-MATERIAL T_ext (see above)
    ) const;

    // --- DEBUG probe (temporary instrumentation). dbg_probe_idx = -1 disables
    // (one branch/tile, no other cost). Its former partner `dbg_plume_dT` (the
    // plume->T shim's traced-cell deposit, Q16.16 counts) was removed with the
    // shim (P-R2 — docs/radiation_raycaster_extinction_ruling_2026-07-31.md
    // A2); nothing currently reads dbg_probe_idx, left wired for a future probe.
    int dbg_probe_idx = -1;
};
