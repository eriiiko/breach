#pragma once
// SkyExchange — planetside volumetric O2 replenishment ("sky exchange").
//
// docs/sky_exchange_design_2026-07-24.md (Fable, fire-tuning §7 Q2 → Option A,
// composition-swap variant). NOT a published technique — this is Erik's
// local-source model for the vertical mixing a top-down 2-D slice cannot
// resolve: every SKY-CONNECTED interior air tile slowly relaxes its gas
// COMPOSITION toward ambient at FIXED local N_total, as a per-tick source term.
//
// WHERE IT RUNS: once per TICK (not per substep — τ ≫ tick, so substep
// placement buys nothing), on the HOST mirror, IMMEDIATELY AFTER the combustion
// pass (combustion vitiates, sky replenishes, the fire's next-tick read sees the
// net). Combustion is itself a host bracket on the mirror in the GPU-resident
// tick (S8c), so this pass rides that same bracket in BOTH the normal and
// resident ticks — one host implementation, so CPU↔CUDA lockstep is bit-
// identical by construction (design finding, Erik-approved 2026-07-24: no device
// kernel; a device kernel would force an extra D2H/H2D per tick while combustion
// stays host — escalation trigger #2).
//
// THE SWAP (per sky tile i, all Q16.16 integer — deterministic cross-machine):
//     target = mul_q16(o2_frac_q, N_total[i])        // ambient composition at
//                                                     //   the LOCAL mass
//     dN     = round_signed( λ · (target − N_O2[i]) ) // sign-symmetric round
//     N_O2   += dN;  N_inert = N_total − N_O2          // N_total invariant, then
//                                                     //   restated exactly (no
//                                                     //   LSB leak in the pair)
// with defensive clamps N_O2 ∈ [0, N_total]. `o2_frac_q` is the SAME quantized
// ambient mole fraction the ring N-split derives from (simulation/ambient.py —
// one source of truth with the ring clamp). λ (`lambda_q`) = quantize(dt_tick /
// sky_tau_s), hoisted host-side once per tick (like recip_P_span); λ == 0
// disables the pass (dormancy — space maps, unblessed levels).
//
// SCOPE: the conservative O2/inert pair ONLY. Smoke's upward-removal λ is
// DEFERRED (a B2-adjacent look decision); temperature is untouched (COOL_SHIFT
// is already the vertical heat channel); pressure/wind are untouched BY
// CONSTRUCTION (N_total per tile is invariant, so the next-tick EOS
// p* = C·N_total·T sees no change — the design's load-bearing property).
//
// CONSERVATION RAIL (design §1.3): per-plane totals now change volumetrically
// (O2 up, inert down, N_total conserved). `sky_flux` (int64, per gas, per tick)
// accumulates the ACTUAL applied Δ so the open-system conservation gate closes:
//   Δtotal(O2 plane)    = boundary_flux[O2]    + sky_flux[O2]
//   Δtotal(inert plane) = boundary_flux[inert] + sky_flux[inert]
// with sky_flux[O2] == −sky_flux[inert] exactly (the swap is a transfer).

#include <cstdint>

// gas        : (n_gases, h, w) Q16.16 density planes, row-major; O2 + inert
//              mutated in place. o2_idx / inert_idx : gas ids (simulation/gases.py).
// sky_mask   : (h, w) bool — the sky-connected INTERIOR air tiles (GameMap.
//              sky_mask; excludes the ambient ring itself and all solid/vacuum).
// o2_frac_q  : ambient O2 mole fraction, raw Q16 (13763 at 0.21).
// lambda_q   : per-tick relaxation rate dt_tick/sky_tau_s, raw Q16. 0 == no-op.
// sky_flux   : (n_gases) int64 rail, ACCUMULATED into (never cleared here);
//              nullptr to skip accounting. Only the O2 + inert entries move.
void sky_exchange_step(
    int32_t* gas, int n_gases, int o2_idx, int inert_idx,
    const bool* sky_mask, int h, int w,
    int32_t o2_frac_q, int32_t lambda_q,
    int64_t* sky_flux);
