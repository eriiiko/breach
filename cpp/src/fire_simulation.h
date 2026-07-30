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
//   hot   = clamp01((T - T_ext) / T_span)
//   o2f   = clamp01((X - o2_frac_ext)/(o2_frac_full - o2_frac_ext)) (LINEAR; the
//                                                   continuous-O2 law — Peatross &
//                                                   Beyler 1997; REPLACES the old
//                                                   smoothstep(P_min,P_full) on
//                                                   ABSOLUTE n_o2 density. The
//                                                   denominator's upper end is the
//                                                   FULL-RESPONSE reference
//                                                   o2_frac_full (pure O2), NOT
//                                                   ambient — see below)
//   avail = F * o2f
//   grow  = k_grow * avail * hot * I * (1-I) * (1 + k_wind_fan * W)
//   die   = k_die * (1 - avail*hot) * I  +  k_wind_strip * W * (1-I) * I
//   I    += dt * (grow - die);  clamp01;  snap to 0 below I_min
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
// The multiply tree order is PINNED (left-fold mul_q16); the plume + smoke-emission
// + wall-burn deposits ROUND-TO-NEAREST (unbiased sources). The discrete outputs
// (the I_min extinguish flip, the wall_hp<=0 burn-through list) are integer compares
// -> bit-deterministic. Plume deposit is an own-index write -> order-independent.

#include <vector>
#include <utility>
#include <cstdint>

struct FireParams {
    // --- signed-logistic feedback (fire_design_proposal §2) ---
    float k_grow         = 4.0f;   // logistic growth gain (1/s)
    float k_die          = 2.0f;   // decay rate when starved/cold (1/s)
    float fire_T_ext     = 350.0f; // extinction temperature (~ignition_temp + 50)
    float fire_T_span    = 150.0f; // width of the `hot` ramp above T_ext
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
    // The O2 factor is now LINEAR in the local O2 MOLE FRACTION X = Σn_o2/Σn_total
    // over open neighbours (Peatross & Beyler 1997: compartment burning rate
    // declines ~linearly with O2 volume fraction), carrying an extinction limit:
    //   o2f = clamp01((X - o2_frac_ext) / (o2_frac_full - o2_frac_ext))
    // This REPLACES the old smoothstep(P_min, P_full) on ABSOLUTE n_o2 density —
    // the fraction is invariant under thermal expansion, so hot thin gas at
    // ambient composition burns (closes the v2.4 "density trap" / hot-zone rescale
    // saga; only true vitiation starves a fire). P_min/P_full below are RETIRED
    // from the sustain law (tombstoned — kept only so old configs/bindings that
    // still set them do not hard-error; no longer read by step()).
    float o2_frac_ext    = 0.13f;  // X_ext: flame-extinction O2 mole fraction
                                   //  (~13% physical limit; 0 = pure proportional)
    // FULL-RESPONSE REFERENCE SPLIT (2026-07-30). The span's upper end used to be
    // o2_frac_amb — so ambient air always produced o2f == 1 and the clamp01 made
    // AMBIENT the ceiling. Locally elevated O2 (reservoirs, leaks, wind delivery)
    // was therefore invisible BY CONSTRUCTION: at X = 0.30 the raw ratio 2.125 was
    // clamped straight back to 1.0. Splitting the two roles makes o2f a true
    // physical fraction — "O2 above extinction, normalized to PURE oxygen" — the
    // clamp effectively never binds, and headroom always exists. Ambient air now
    // lands at (0.21 - 0.13)/(1 - 0.13) = 0.092.
    float o2_frac_full   = 1.00f;  // X_full: the O2 mole fraction at which o2f
                                   //  reaches 1 (pure O2). NOT the ambient
                                   //  atmosphere and NOT map-overridden — a fixed
                                   //  physical reference. Setting it to
                                   //  o2_frac_amb reproduces the pre-split law.
    float o2_frac_amb    = 0.21f;  // X_amb: what the ambient atmosphere IS (reads
                                   //  the level's authored [ambient] o2_frac; 0.21
                                   //  fallback — one source of truth with the BC).
                                   //  NO LONGER read by step(): the availability
                                   //  law normalizes by o2_frac_full above. Kept
                                   //  because it is the per-map ambient record and
                                   //  configs/levels/bindings still set it.
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
    float fire_pressure_gain = 0.15f; // own-tile overpressure gain (1/s)
    // p_expand_ref: RETIRED as the plume's saturation gate (eos-p3fix-
    // thermal-ceiling — see T_FLAME_MAX below); kept only so old configs/
    // bindings that still set it do not hard-error. No longer read by step().
    float p_expand_ref       = 1.30f;
    // EOS refactor P3 (design §8 patch P3 writer row: "fire plume -> a
    // minimal plume->T shim"): the plume no longer writes `atmosphere`
    // directly (P is solver-owned now, materialized once/tick — a writer
    // fighting that would be overwritten next tick anyway). Instead the
    // gain scalar (fire_pressure_gain*I*sat*dt) becomes a small ΔT energy
    // deposit: temperature += gain * temp_gain_scale. This is the MINIMAL
    // shim named in the design (not a real ΔE/(N*c_v) energy budget) —
    // "the pop never goes inert" during the P3->P4 window; a TUNING DIAL,
    // feel-gated at P5 like the old fire_pressure_gain was.
    float temp_gain_scale    = 50.0f;
    // T_FLAME_MAX (eos-p3fix-thermal-ceiling, decisions.md #16): the plume's
    // self-limiting ceiling, in the SAME ΔT-above-ambient Q16.16 convention
    // as `temperature` (T_AMB_K lives on EOSSolver — see eos_solver.h). The
    // pre-refactor plume was self-limiting via
    // `sat = 1 - atmosphere[i]/p_expand_ref` — atmosphere WAS the field the
    // plume drove directly, on its OWN (solid) tile, so that gate tracked
    // its own cumulative deposit correctly. Post-P3, `atmosphere` (== P) is
    // materialized by the EOS solver, which forces P = 0 at every SOLID
    // cell (eos_solver.cpp: "vacuum Dirichlet + solid zero") — a fire tile
    // IS solid (fire_simulation.cpp's own O2-neighbour-mean comment: "its
    // own tile holds no gas"), so `atmosphere[i]` at the plume's own tile is
    // permanently 0 and `sat` is permanently ~1.0: the self-limiter never
    // engaged (a structural unit/placement mismatch, not a tuning miss —
    // measured: the deposit rode fire intensity unclamped every tick).
    // Fix: gate on the ACTUAL quantity being deposited (T, not P), against a
    // physical ceiling — real wood/typical flames run ~2000-2300 K absolute;
    // T_FLAME_MAX defaults to 2000 (K above T_AMB_K). Same smooth taper
    // shape as the old gate (`sat = clamp01(1 - x/ref)`) for continuity of
    // feel, just measured against T instead of P.
    float T_FLAME_MAX        = 2000.0f;

    // --- kept behaviours ---
    float smoke_emission = 0.8f;   // smoke produced per second per unit intensity
    float wall_damage    = 0.4f;   // wall HP lost per second per unit intensity
                                   //  (burn-through IS the fuel-consumption brake)

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
    //   smoke       : int32 (h, w) Q16.16 (S2b), fire ADDS to it (kept). The
    //                 emission delta smoke_emission*dt*I is round-to-nearest and
    //                 integer-added — order-free, deterministic.
    //   wall_hp     : int32 (h, w) Q16.16 (S3b), burn-through depletes it (the fuel
    //                 brake); fractional depletion needs the Q16.16 fraction.
    //   temperature : int32 (h, w) Q16.16, READ (the T gate) + WRITE (EOS P3:
    //                 the plume->T shim deposit, `temp_gain_scale` above,
    //                 REPLACES the old atmosphere-plume own-tile write).
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
    std::vector<std::pair<int, int>> step(
        int32_t* fire,             // S3b: Q16.16 (was float)
        const int32_t* atmosphere, // S2c: Q16.16 == P (EOS P3: READ-ONLY, plume only)
        const int32_t* n_o2,       // EOS P4: Q16.16 real O2 density (mole-fraction numerator)
        const int32_t* n_total,    // continuous-O2 law: Q16.16 real N_total (Σ conservative
                                   //  bulk planes = O2+N2), READ-ONLY — the mole-fraction
                                   //  DENOMINATOR (X = Σn_o2/Σn_total over open neighbours)
        int32_t* smoke,            // S2b: Q16.16 (fire emission round + added)
        int32_t* wall_hp,          // S3b: Q16.16 (was float)
        int32_t* temperature,      // EOS P3: mutable (plume->T shim write)
        const int32_t* wind_x,     // S2c/S3b: Q16.16
        const int32_t* wind_y,     // S2c/S3b: Q16.16
        const bool* is_wall,
        const bool* is_vacuum,
        const bool* flammable,
        int h, int w,
        float dt,
        const int64_t* fuel_recip = nullptr   // FUEL-FRACTION AXIS (see above)
    ) const;

    // --- DEBUG probe (temporary instrumentation, eos-p3fix-thermal-ceiling
    // investigation, decisions.md #16): the plume->T shim's deposit at ONE
    // traced cell this call. dbg_probe_idx = -1 disables (one branch/tile,
    // no other cost). Raw Q16.16 counts.
    int dbg_probe_idx = -1;
    mutable int32_t dbg_plume_dT = 0;
};
