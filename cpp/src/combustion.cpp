#include "combustion.h"
#include "fixed_point.h"
#include "raycaster.h"   // heat_saturating_add (shared Q16.16 domain)
#include <algorithm>
#include <cstdint>

using namespace fixedpoint;

namespace {

// 4-connected open-neighbour faces (N, S, W, E) — the SAME idiom
// FireSimulation's own O2/smoke passes use.
static constexpr int D4[][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

static inline bool in_bounds(int y, int x, int h, int w) {
    return y >= 0 && y < h && x >= 0 && x < w;
}

}  // namespace

void CombustionSolver::step(
        int32_t* gas, int n_gases,
        int o2_idx, int inert_n2_idx, int black_smoke_idx,
        int32_t* temperature,
        const int32_t* wall_hp,
        const int32_t* fire,
        const bool* flammable,
        const bool* solid,
        const bool* is_vacuum,
        const int32_t* ignition_temp_q16,
        int h, int w, float dt,
        float c_v, float n_floor_heat) const {

    if (h <= 0 || w <= 0 || dt <= 0.0f) return;
    if (o2_idx < 0 || o2_idx >= n_gases) return;
    if (inert_n2_idx < 0 || inert_n2_idx >= n_gases) return;
    if (black_smoke_idx < 0 || black_smoke_idx >= n_gases) return;

    const int n = h * w;
    int32_t* O2   = gas + (size_t)o2_idx * n;
    int32_t* N2   = gas + (size_t)inert_n2_idx * n;
    int32_t* SOOT = gas + (size_t)black_smoke_idx * n;

    // Load-time constants (double-fold once, then quantize — the LOCKED
    // per-step-scalar idiom shared by eos_solver.cpp / fire_simulation.cpp).
    const q16 burn_cap_q   = quantize((double)burn_rate * (double)dt);
    const q16 o2_thresh_q  = quantize((double)o2_thresh_burn);
    const q16 soot_yield_q = quantize((double)soot_yield);
    const q16 H_fuel_q     = quantize((double)H_fuel);
    const double c_v_safe  = (c_v > 0.0f) ? (double)c_v : 1.0;
    const int64_t recip_cv = make_recip(c_v_safe);              // 1/c_v, once per step
    const q16 n_floor_q    = quantize((double)n_floor_heat);

    if (burn_cap_q <= 0) return;   // nothing burns this tick (dt~0 or burn_rate 0)

    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (!flammable[i] || wall_hp[i] <= 0) continue;   // no fuel

            const q16 ign_q = ignition_temp_q16[i];
            // Cheap row-major PREFILTER (design task: "today's FireSimulation
            // candidates: fire-intensity > 0 or ignition-eligible flammables")
            // — a pure widening; the REAL gate (ign_q>0 && T>=ign_q) below is
            // checked unconditionally, so this never changes the outcome, it
            // only documents that already-burning tiles are always considered.
            const bool prefilter = (fire[i] > 0) ||
                                   (ign_q > 0 && temperature[i] >= ign_q);
            if (!prefilter) continue;
            if (ign_q <= 0 || temperature[i] < ign_q) continue;   // design §5 gate

            for (const auto& d : D4) {
                const int ny = y + d[0], nx = x + d[1];
                if (!in_bounds(ny, nx, h, w)) continue;
                const int j = ny * w + nx;
                if (solid[j] || is_vacuum[j]) continue;   // open-air burn site only

                const q16 o2_j = O2[j];
                if (o2_j <= o2_thresh_q) continue;         // starved — no burn here

                const q16 burn = std::min(burn_cap_q, o2_j);   // saturating (integer)
                if (burn <= 0) continue;

                O2[j] = (int32_t)(o2_j - burn);
                // Exact split: soot + (burn-soot) == burn, so N_total (Dalton)
                // is conserved to the LSB regardless of the soot_yield rounding
                // (decisions.md #12 — "the non-soot fraction ... credited to
                // inert_N2").
                const q16 soot = narrow_round(mul_wide(burn, soot_yield_q));
                const q16 n2_gain = (q16)(burn - soot);
                SOOT[j] += soot;
                N2[j]   += n2_gain;

                // §4.3 heat-deposit reciprocal — SAME idiom/dials as
                // TemperatureSolver's Pass-1 radiative deposit. N_total here
                // is the POST-burn bulk-pair sum at the SAME neighbour cell
                // (the engine's established N proxy for this deposit class).
                q16 n_total_j = (q16)((int64_t)O2[j] + (int64_t)N2[j]);
                if (n_total_j < n_floor_q) { n_total_j = n_floor_q; ++heat_floor_hits; }
                const q16 recip_n  = reciprocal_q16(n_total_j);
                const q16 deposit  = mul_q16(burn, H_fuel_q);       // burn*H_fuel
                const q16 e_over_n = mul_q16(deposit, recip_n);     // .../N
                const q16 dT       = recip_mul(e_over_n, recip_cv); // .../c_v
                heat_saturating_add(&temperature[j], dT);
            }
        }
    }
}
