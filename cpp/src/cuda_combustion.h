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
// gas[black_smoke_idx]), `temperature`, and `wall_hp`. `fire` is intentionally
// absent (the reformulation dropped it — combustion.cpp header). The scalar
// config dials arrive explicitly (the free function has no CombustionSolver).
// The two per-call rail counts are written to *heat_floor_hits / *t_max_phys_hits
// (added to whatever they hold, matching the CPU member `+=` accumulation).
void combustion_step(
    int32_t* gas, int n_gases,
    int o2_idx, int inert_n2_idx, int black_smoke_idx,
    int32_t* temperature, int32_t* wall_hp,
    const bool* flammable, const bool* solid, const bool* is_vacuum,
    const int32_t* ignition_temp_q16,
    int h, int w, float dt, float c_v, float n_floor_heat,
    float burn_rate, float o2_thresh_burn, float H_fuel, float soot_yield,
    float fuel_per_o2, float T_MAX_PHYS,
    int64_t* heat_floor_hits, int64_t* t_max_phys_hits);

// Backend flag: when ON, PhysicsRunner's combustion pass dispatches to
// combustion_step on the GPU instead of the CPU CombustionSolver::step.
// Defaults OFF — flag-off is the EXACT prior CPU call (strictly additive).
bool combustion_backend_is_cuda();
void set_combustion_backend_cuda(bool on);

}  // namespace breach_cuda
