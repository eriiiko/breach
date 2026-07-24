#include "sky_exchange.h"
#include "fixed_point.h"
#include <cstddef>

using namespace fixedpoint;

// One host pass per tick (see sky_exchange.h). Pure Q16.16 integer, per-tile
// independent → order-free and bit-identical across machines and vs the GPU-
// resident tick (which runs this SAME host call on the mirror after combustion).
void sky_exchange_step(
        int32_t* gas, int n_gases, int o2_idx, int inert_idx,
        const bool* sky_mask, int h, int w,
        int32_t o2_frac_q, int32_t lambda_q,
        int64_t* sky_flux) {
    const int n = h * w;
    // Dormancy: λ == 0 (space maps, unblessed levels, sky_tau_s == 0) → no-op,
    // byte-identical. The runner already gates on this; the guard makes the pass
    // self-contained (and a degenerate index guard keeps it total).
    if (lambda_q == 0 || o2_idx < 0 || inert_idx < 0
            || o2_idx >= n_gases || inert_idx >= n_gases)
        return;

    int32_t* N_o2 = gas + (std::size_t)o2_idx * n;
    int32_t* N_in = gas + (std::size_t)inert_idx * n;

    int64_t flux_o2 = 0;                       // Σ ACTUAL applied ΔN_O2 (post-clamp)
    for (int i = 0; i < n; ++i) {
        if (!sky_mask[i]) continue;
        const int32_t o2_old = N_o2[i];
        const int32_t in_old = N_in[i];
        const int64_t n_tot = (int64_t)o2_old + (int64_t)in_old;   // conserved per tile

        // target = mul_q16(o2_frac_q, N_total) — ambient composition at the LOCAL
        // mass. Wide form (n_tot is int64) but bit-identical to mul_q16 for the
        // non-negative operands here: arithmetic >>16 == floor, exactly mul_q16.
        const int64_t target = ((int64_t)o2_frac_q * n_tot) >> FP_SHIFT;

        // dN = round_signed(λ · (target − N_O2)). Sign-symmetric round (round-
        // half-away-from-zero) so + and − relax identically — no DC bias when a
        // tile sits above ambient (rare: enrichment) vs the usual below-ambient.
        const int64_t diff = target - (int64_t)o2_old;
        const int64_t dN   = narrow_round_signed((int64_t)lambda_q * diff);

        // Apply + defensive clamp N_O2 ∈ [0, N_total]; restate N_inert EXACTLY so
        // the pair sums to N_total to the LSB (no leak, no pressure footprint).
        int64_t o2_new = (int64_t)o2_old + dN;
        if (o2_new < 0)      o2_new = 0;
        if (o2_new > n_tot)  o2_new = n_tot;
        const int64_t in_new = n_tot - o2_new;

        N_o2[i] = (int32_t)o2_new;
        N_in[i] = (int32_t)in_new;
        flux_o2 += (o2_new - o2_old);          // the actual, clamp-aware delta
    }

    if (sky_flux) {
        sky_flux[o2_idx]    += flux_o2;
        sky_flux[inert_idx] -= flux_o2;        // the swap is a transfer: O2 += = inert −=
    }
}
