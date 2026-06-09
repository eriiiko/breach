// Temperature solver implementation — engine/06 §1 (heat -> temperature).
// See temperature_solver.h for the determinism contract.

#include "temperature_solver.h"
#include "raycaster.h"   // HEAT_SCALE, heat_saturating_add (shared Q16.16 domain)

void TemperatureSolver::step(
    int32_t* temperature,
    const int32_t* heat,
    const int32_t* heat_inv_shift,
    const bool* solid,
    int h, int w
) const {
    const int n = h * w;

    // ---- Pass 1: heat -> temperature conversion (proposal §1.2) ----
    // Solid tiles only. `heat` is a saturating accumulator of non-negative
    // deposits, so `heat >> shift` is an arithmetic right shift on a
    // non-negative int32 -> divides the Q16.16 quantity by 2^shift while
    // staying Q16.16, bit-identical on every machine. The saturating add pins
    // at INT32_MAX rather than wrapping. Air tiles are skipped (stay 0).
    for (int i = 0; i < n; ++i) {
        if (!solid[i]) continue;          // air / non-solid: no conversion
        int32_t deposit = heat[i];
        if (deposit <= 0) continue;       // nothing to convert this tick
        int shift = heat_inv_shift[i];    // log2(thermal_mass), >= 0
        int32_t gain = deposit >> shift;  // Q16.16 / 2^shift, still Q16.16
        heat_saturating_add(&temperature[i], gain);
    }

    // STEP B (conduction relaxation, §2) and STEP C (ambient cooling, §3) will
    // add further passes here, reading the just-converted field.
}
