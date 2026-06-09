// Temperature solver implementation — engine/06 §1 (heat -> temperature) +
// §2 (conduction relaxation). See temperature_solver.h for the determinism
// contract.

#include "temperature_solver.h"
#include "raycaster.h"   // HEAT_SCALE, heat_saturating_add (shared Q16.16 domain)

// Direction order for the per-tile face_shift cache (MUST match the Python
// bake in GameMap: index 0=N, 1=S, 2=E, 3=W).
namespace {
    constexpr int DIR_N = 0;
    constexpr int DIR_S = 1;
    constexpr int DIR_E = 2;
    constexpr int DIR_W = 3;
    // Row/col offset per direction, same order.
    constexpr int DY[4] = { -1, +1,  0,  0 };
    constexpr int DX[4] = {  0,  0, +1, -1 };
}

void TemperatureSolver::step(
    int32_t* temperature,
    const int32_t* heat,
    const int32_t* heat_inv_shift,
    const int32_t* face_shift,
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

    // ---- Pass 2: conduction relaxation (proposal §2.2) ----
    // Gather stencil, double-buffered so the whole pass reads the FROZEN
    // pre-conduction field and writes a fresh one (order-independent, no
    // scatter, no atomics). For tile i with 4 neighbours n ∈ {N,S,E,W}:
    //
    //     acc = Σ  (temp[n] - temp[i]) >> face_shift[i][dir]
    //     temp_new[i] = temp[i] + acc
    //
    // The DIFFERENCE is shifted, not the neighbour, so equal neighbours give
    // EXACTLY 0 (no drift) and the flux is conservative-shaped. A NO_FACE face
    // (grid edge or κ==0 either side) is skipped, so air (all NO_FACE) is a
    // structural no-op: Σr == 0 -> temp_new == temp, an air tile at 0 stays
    // bit-exactly 0. 64-bit accumulator avoids any intermediate overflow; the
    // final write fits int32 because the result is a convex combination of the
    // (already-int32) field values (§2.6 discrete maximum principle).
    scratch_.resize(n);
    int32_t* temp_new = scratch_.data();
    const int NO_FACE = no_face;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int i = y * w + x;
            const int32_t* fs = &face_shift[i * 4];  // [N,S,E,W] for this tile
            const int32_t ti = temperature[i];
            int64_t acc = 0;
            for (int d = 0; d < 4; ++d) {
                const int s = fs[d];
                if (s == NO_FACE) continue;          // grid edge or κ==0 -> no face
                const int ny = y + DY[d];
                const int nx = x + DX[d];
                // NO_FACE already marks grid edges, so neighbours are in-bounds;
                // guard anyway for robustness against a mis-baked cache.
                if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                const int32_t tn = temperature[ny * w + nx];
                // Signed Q16.16 difference; arithmetic right shift == ÷2^s
                // (rounds toward -inf, deterministic & identical cross-machine).
                acc += (int64_t)(tn - ti) >> s;
            }
            temp_new[i] = (int32_t)((int64_t)ti + acc);
        }
    }

    // Swap temp_new -> temperature (write the new field back in place; the
    // caller's buffer is the persistent one, scratch_ is reused next tick).
    for (int i = 0; i < n; ++i) temperature[i] = temp_new[i];

    // STEP C (ambient cooling, §3) and STEP D (unit damage, §4) will add
    // further passes here, reading the just-conducted field.
}
