#pragma once
// Temperature solver — turns the per-tick `heat` deposit into a persistent
// `temperature` field on solids (engine/06 §1; temperature_design_proposal §1).
//
// STEP A scope: the heat -> temperature CONVERSION pass only. Conduction
// relaxation (§2) and ambient cooling (§3) are LATER steps and will be added to
// step() as further passes. Nothing else consumes temperature yet.
//
// Determinism (engine/06 §3, proposal §1.2): both `heat` and `temperature` are
// Q16.16 int32 sharing one scale (TEMP_SCALE == HEAT_SCALE). The conversion is
//
//     temperature[i] = sat_add( temperature[i], heat[i] >> heat_inv_shift[i] )
//
// run on SOLID tiles only. `heat` is a saturating accumulator of NON-NEGATIVE
// deposits, so the arithmetic right shift is on a non-negative value -> portable
// and bit-identical across machines/compilers (no float, no division). The
// saturating add (reused from raycaster.h `heat_saturating_add`) pins at
// INT32_MAX under a firestorm instead of wrapping cold. Air tiles (not solid)
// are skipped, so an air tile that starts at 0 stays bit-exactly 0.

#include <cstdint>

class TemperatureSolver {
public:
    // One tick of thermal work. STEP A: heat -> temperature conversion only.
    //
    //   temperature : Q16.16 int32, (h, w). Persistent field (ΔT above ambient;
    //                 T_ambient == 0, proposal §3.1). Mutated in place.
    //   heat        : Q16.16 int32, (h, w). Per-tick deposit from the ray pass.
    //                 Read NON-DESTRUCTIVELY (the caller clears it at end of tick,
    //                 after this and every other heat consumer).
    //   heat_inv_shift : int32, (h, w). Precomputed per-tile log2(thermal_mass)
    //                 cache (0..30). `heat >> heat_inv_shift` == heat /
    //                 thermal_mass, still Q16.16.
    //   solid       : bool, (h, w). The physics solid mask. Conversion runs on
    //                 solids only; air is skipped (stays 0).
    void step(
        int32_t* temperature,
        const int32_t* heat,
        const int32_t* heat_inv_shift,
        const bool* solid,
        int h, int w
    ) const;
};
