#pragma once
// ============================================================================
// EOS P6.9b — CombustionSolver::step on the GPU (bit-identical, closes P6).
//
// A faithful, bit-identical GPU port of the P6.9a CPU combustion reformulation
// (combustion.cpp / docs/eos_p6_9_combustion_design.md §3–§4). The CPU pass was
// reformulated FROM a row-major scatter INTO two order-free GATHER passes SO
// THAT it is direction-free AND ports to CUDA as a plain bit-identity mirror
// (design §4: "The digest gate is then a formality the design guarantees").
//
// STRUCTURE — the design §4 barrier chain `snapshot(T) → Pass A → face buffers
// → Pass B`, realized as a launch-barriered kernel chain (structurally identical
// to cuda_fire.cu's barriered pass chain):
//
//   K0  snapshot          d_tsnap <- d_temp   (a device-to-device copy; freezes
//                         the ignition gate — a source cannot heat AND ignite a
//                         neighbour the same tick, delta alpha)
//   K1  Pass A (air j)    one thread per OPEN-air cell: gather <=4 flammable
//                         claimant sources, EXACT-INTEGER proportional split of
//                         O2[j] (int64 `/`,`%` + <=4-in-thread remainder
//                         distribution, largest-key/lowest-index tiebreak),
//                         full-drain, single-writer O2/SOOT/N2 + ONE aggregate
//                         heat deposit (post-burn N_total, T_MAX_PHYS rail),
//                         write the 4 direction-keyed face buffers.
//   K2  Pass B (source i) one thread per flammable source: sum its <=4 incoming
//                         face allocations, pay wall_hp -= round(fuel_per_o2 *
//                         burn_i), floored ONCE at FUEL_FLOOR.
//
// WHY bit-identical (design §4): both passes are per-cell functions of frozen
// inputs (d_tsnap, pass-entry O2 read ONLY at the own cell, pre-payment wall_hp,
// masks) and the gas/temperature planes each cell alone writes. No Pass-A gate
// reads live temperature (they read d_tsnap); no cell reads another cell's
// within-pass gas write (O2 is read only at the OWN cell j). The only order is
// the fixed in-cell remainder tiebreak (bounded <=4, in-thread). All Q16.16
// integer; the proportional divide is plain int64 `/`,`%` — a single portable
// answer, bit-identical on CPU and CUDA (CUDA integer divide == CPU integer
// divide). Every non-trivial device op is a shared mirror already proven
// bit-identical to its host form (mul_wide/narrow_round/mul_q16 are FP_HD;
// reciprocal_q16_dev / recip_mul_dev / heat_saturating_add_dev are the CUDA-S4
// device kit). The launch barriers give the Pass-A-before-Pass-B ordering the
// CPU's loop order gives for free.
//
// The rail counters (heat_floor_hits / t_max_phys_hits) are now PER-CELL (design
// §3) and are accumulated with device atomicAdd — a plain integer count is
// order-free, so the total is bit-identical to the CPU regardless of thread
// order (no test may assert their ABSOLUTE value, only CPU==GPU equality).

#include <cstdint>

namespace breach_cuda {

// ONE CombustionSolver::step on the GPU — IN-PLACE on the three mutated gas
// planes (O2 = gas[o2_idx], inert_N2 = gas[inert_n2_idx], black_smoke =
// gas[black_smoke_idx]), `temperature`, and `wall_hp`. `fire` is READ again
// (continuous-O2 law, docs/continuous_o2_law_design_2026-07-24.md §2.3): it is
// the per-claimant intensity factor I_k in demand_k = burn_cap*I_k*o2f_j (the
// P6.9 reformulation had dropped it as an outcome-neutral prefilter; the
// proportional-draw law reinstates it as the demand magnitude). The scalar
// config dials arrive explicitly (the free function has no CombustionSolver),
// now including o2_frac_ext/o2_frac_full — the SAME mole-fraction span dial the
// fire logistic uses (one law, shared constants; the span's upper end is the
// FULL-RESPONSE reference o2_frac_full, pure O2, NOT the ambient dial
// o2_frac_amb, which the law no longer reads). The two per-call rail counts
// are written to *heat_floor_hits / *t_max_phys_hits (added to whatever they
// hold, matching the CPU member `+=` accumulation).
void combustion_step(
    int32_t* gas, int n_gases,
    int o2_idx, int inert_n2_idx, int black_smoke_idx,
    int32_t* temperature, int32_t* wall_hp,
    const int32_t* fire,
    const bool* flammable, const bool* solid, const bool* is_vacuum,
    const int32_t* ignition_temp_q16,
    int h, int w, float dt, float c_v, float n_floor_heat,
    float burn_rate, float o2_thresh_burn, float H_fuel, float soot_yield,
    float fuel_per_o2, float o2_frac_ext, float o2_frac_full, float T_MAX_PHYS,
    int64_t* heat_floor_hits, int64_t* t_max_phys_hits,
    // P-E2b (energy-books arc, design §2.2/§2.5): the energy-sum twin of
    // heat_floor_hits — the combustion floor's destroyed ΔE (see
    // combustion.h's e_deposit_drop_sum doc). Added +=, nullable.
    int64_t* e_deposit_drop_sum = nullptr,
    // THERMAL-MASS AXIS, P-EOS (docs/thermal_mass_eos_ruling_2026-07-30.md §2
    // site 3; rationale in combustion.h's header): on a thermal_solid burn site
    // — a FURNITURE tile, which is open/gas-holding but thermally an OBJECT —
    // the aggregate deposit converts via the tile's own heat_inv_shift instead
    // of the thin pore gas's N divisor. Both nullable; either null -> every site
    // takes the GAS path == the pre-patch behaviour (and the two are identical
    // anyway on any furniture-free map).
    const bool* thermal_solid = nullptr,
    const int32_t* heat_inv_shift = nullptr,
    // P-R4 (ruling A1): the FUEL-BED deposit. `heat` is the Q16.16 heat plane,
    // MUTATED — each claimant receives H_bed * (the O2 it actually got) as a
    // positive SATURATING atomic add at its own cell, exactly the CPU's
    // heat_saturating_add (order-free for non-negative deltas under a monotone
    // clamp, so several air cells feeding one source in any thread order give
    // the identical total). H_BED_M/H_BED_SHIFT are the split constant
    // (combustion.h documents why). heat == nullptr -> no H_bed, byte-identical.
    int32_t* heat = nullptr,
    float H_BED_M = 0.0f,
    int H_BED_SHIFT = 0,
    // D1 (amendment 5): the (max_claimants, h, w) error-feedback DEMAND
    // ACCUMULATOR — SYNCED state, IN/OUT. Single writer per air cell (each
    // thread owns all of its own claimant slots), so no atomics and no order
    // dependence. Full rationale, scale algebra and reset rule: combustion.h.
    int32_t* dem_acc = nullptr,
    // P-O2b — THE EXTENDED OXYGEN DRAW (design v5.2 "F-O2b"). The GPU twin
    // mirrors the CPU law bit for bit: the same baked offset tables (uploaded
    // to __constant__ from combustion.h's single definition, so the backends
    // cannot drift), the same levelled relaxation, the same lexicographic slot
    // reduction, the same re-sited deposit. draw_r == 1 is byte-identical to
    // the shipped 4-face draw on both backends. See combustion.h.
    int draw_r = 1,
    const float* dyn_permeability = nullptr,
    int max_claimants = 4);

// Backend flag: when ON, PhysicsRunner's combustion pass dispatches to
// combustion_step on the GPU instead of the CPU CombustionSolver::step.
// Defaults OFF — flag-off is the EXACT prior CPU call (strictly additive).
bool combustion_backend_is_cuda();
void set_combustion_backend_cuda(bool on);

}  // namespace breach_cuda
