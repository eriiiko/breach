#pragma once
// ===========================================================================
// THE O2 LAW'S PRESSURE FACTOR — issue #7 ("vacuum kills fire"), Erik's ruling
// of 2026-09-23 (docs/fire_vacuum_pressure_factor_brief_2026-09-23.md §2).
//
//   g(p) = clamp01( (p - p_ext) / (p_full - p_ext) )
//
// multiplies the O2 MOLE-FRACTION factor at BOTH reads of the O2 law:
//   * the intensity ODE's `o2f` (fire_simulation.cpp + cuda_fire.cu), where p is
//     the mean pressure of the SAME open 4-neighbours the fraction X pools;
//   * combustion's claim gate `o2f_j` (combustion.cpp + cuda_combustion.cu),
//     where p is the pressure of the SAME air cell j that X_j reads.
// So each read of the law sees one parcel of air through both of its factors.
//
// WHY A SECOND FACTOR. Venting removes O2 and N2 together, so the mole
// fraction stays ~0.21 down to zero molecules: the fraction law alone held a
// fire at full intensity in a room vented to 0.013 atm (docs/ray_engine_v2_
// scheme_study_2026-09-13/report_m3.md finding 4).
//
// WHY PRESSURE, NOT DENSITY (the ruling). The July density trap (docs/
// continuous_o2_law_design_2026-07-24.md §2.1) was a gate on ABSOLUTE density:
// thermally expanded air is THIN (low N) and read as "no oxygen". Hot air at
// ambient pressure is thin but NOT low-pressure — the EOS holds
// P = C·N·T_abs at ambient while N falls — so a pressure factor leaves a hot
// room's fire exactly as the fraction law has it, and only a room that has
// actually lost its gas drops below p_ext.
//
// THE PRESSURE READ is `atmosphere`: the engine's ONE materialized pressure P
// (engine/04: "atmosphere is a zero-copy ALIAS of the derived P", materialized
// once per tick in EOSSolver step 5, before any consumer — combustion and the
// fire step both run after it). Its unit is the atm, Q16.16: P = C·N_total·T_abs
// with C = 1/T_AMB_K and ambient N_total = quantize(p_amb), so 1 atm == FP_ONE
// (the ambient-map pin sits at 65632, 1.0015 atm — the quantization lattice,
// ambient.py). Nothing here derives a second pressure.
//
// THE EDGES arrive already in that unit (door 2): PhysicsRunner quantizes the
// [physics.fire] keys `p_ext_atm` / `p_full_atm` ONCE at load through the
// pressure field's own boundary module (atmosphere_fixed). The span reciprocal
// is baked host-side ONCE per step with the kit's integer reciprocal_q16 (door
// 1 — no floating-point anywhere on this path), like the fraction law's own
// span reciprocal. `factor()` is FP_HD: the CPU TUs and the CUDA kernels call
// THIS function, so the two backends cannot disagree about the law.
//
// EXACT EDGES (what the tests pin):
//   * p >= p_full  ->  g == FP_ONE exactly, and mul_q16(o2f, FP_ONE) == o2f, so
//     at or above p_full the pre-#7 law is reproduced bit for bit. This is an
//     explicit branch because the bare ramp's value at p == p_full depends on
//     which way the span's integer reciprocal rounds: at the ruled 0.1 / 0.5
//     atm it rounds up and the ramp happens to land on FP_ONE, but at e.g.
//     0.1 / 0.4 atm it rounds down and the ramp stops at 65535. The branch
//     makes the edge exact for ANY configured span.
//   * p <= p_ext   ->  g == 0.
//   * between      ->  mul_q16(p - p_ext, 1/span), truncating like the rest of
//     the kit, never above FP_ONE.
//
// LITERATURE for the two edges (full notes, and what was / was not verified:
// docs/papers/continuous_o2_law_citations.md, entries 3-5):
//   Harper, S.A., Juarez, A., Perez, H., Hirsch, D.B. & Beeson, H.D., "Oxygen
//   Partial Pressure and Oxygen Concentration Flammability: Can They Be
//   Correlated?", J. ASTM International (2016), NASA NTRS 20160001047 — self-
//   extinguishment thresholds show "little relation to total pressures above
//   41 kPa"; pressure is "highly influential" below it (the p_full knee).
//   Hirsch, D., Williams, J. & Beeson, H., "Pressure Effects on Oxygen
//   Concentration Flammability Thresholds of Materials for Aerospace
//   Applications", J. Testing and Evaluation (2006), NASA NTRS 20070005041 —
//   near-flat thresholds over 48.2-101.3 kPa. Both archived in docs/papers/.
//   He, X., Wang, J. & Fang, J., "Flammability limits and near-limit chemistry
//   controlled flame spread over thermally thin paper under sub-atmospheric
//   pressure", Fire Safety Journal 120:103042 (2021) — thin cellulose's
//   limiting O2 concentration mapped across 4-45 kPa (the p_ext band).
//
// DORMANT BY DEFAULT: p_full <= 0 means "no pressure limit" and g == FP_ONE for
// every p. That is the FireParams / CombustionSolver default, so every direct-
// binding caller (the law unit tests pass a zero `atmosphere`, which was an
// unread argument before this factor existed) keeps the pre-#7 law bit for bit;
// the live engine binds the ruled edges. A span with p_full <= p_ext (and
// p_full > 0) is a step at p_full; PhysicsRunner refuses such a config at load.
// ===========================================================================

#include <cstdint>

#include "fixed_point.h"   // FP_HD, q16, mul_q16, reciprocal_q16 — the ONE kit

namespace o2_pressure {

struct Factor {
    q16 p_ext_q;     // extinction pressure, Q16.16 atm — g == 0 at or below it
    q16 p_full_q;    // full-effect pressure, Q16.16 atm — g == FP_ONE at or above it
    q16 inv_span_q;  // reciprocal_q16(p_full_q - p_ext_q); 0 when the span is empty
};

// Host-side, ONCE per step: the span reciprocal, hoisted out of the per-cell
// loop exactly as the fraction law hoists its own.
inline Factor bake(q16 p_ext_q, q16 p_full_q) {
    Factor f{p_ext_q, p_full_q, 0};
    const int64_t span = (int64_t)p_full_q - (int64_t)p_ext_q;
    if (span > 0 && span <= (int64_t)INT32_MAX) {
        f.inv_span_q = fixedpoint::reciprocal_q16((q16)span);
    }
    return f;
}

// Per evaluation site: g(p) in Q16.16, in [0, FP_ONE].
FP_HD inline q16 factor(const Factor& f, q16 p) {
    if (f.p_full_q <= 0) return fixedpoint::FP_ONE;   // dormant: no pressure limit
    if (p >= f.p_full_q) return fixedpoint::FP_ONE;   // full effect: exact identity
    if (p <= f.p_ext_q) return 0;                      // below extinction: no fire
    const q16 g = fixedpoint::mul_q16(p - f.p_ext_q, f.inv_span_q);
    return (g > fixedpoint::FP_ONE) ? fixedpoint::FP_ONE : g;
}

}  // namespace o2_pressure
