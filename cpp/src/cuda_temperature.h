#pragma once
// ============================================================================
// CUDA-S1 — the temperature solver on the GPU (the first real physics kernel).
// ============================================================================
//
// A faithful, bit-identical port of TemperatureSolver::step (temperature_solver.
// cpp): the three-pass heat->temperature CONVERSION (§1), CONDUCTION relaxation
// (§2, double-buffered gather), and ambient COOLING (§3, vacuum-exposed 4x). All
// three passes are pure integer Q16.16 (two's-complement +,-,>>), so the GPU
// result is byte-for-byte identical to the CPU on every architecture — the whole
// point of S1.
//
// Plain C++ declaration header (no CUDA types) so the .cpp TUs (bindings.cpp,
// physics_engine.cpp, compiled by cl.exe even in the CUDA build) can include it;
// cuda_temperature.cu provides the definitions. Compiled only when BREACH_CUDA.
#include <cstdint>

namespace breach_cuda {

// One tick of thermal work on the GPU — IN-PLACE on `temperature`. Mirrors
// TemperatureSolver::step exactly (same args; the scalar dials are passed
// explicitly since this is a free function, and every boundary cast — o2_vacuum_
// thresh, n_floor_heat, T_MAX_PHYS, the c_v reciprocal, gas_advection_rate·dt —
// is done ONCE on the host with the identical fixedpoint::quantize / make_recip
// the CPU uses). face_shift is (h,w,4) int32, dir order N,S,E,W.
//
// EOS P6.6 (docs/eos_p6_gpu_alignment_review.md §4): extended from the S1
// solid-only convert/conduct/cool to the FULL unified-temperature step —
//   Pass 0  gas-T zero-at-open-vacuum (unconditional) + optional semi-Lagrangian
//           advection on wind_x/wind_y (skipped, a clean no-op, when dt<=0 or
//           wind is null — the engine's own path, which advects T in eos.step);
//   Pass 1  heat -> temperature deposit: solids via the bit-shift, open-air via
//           the v2.4 absorption-∝-density ΔT = E_abs/(N·c_v) reciprocal (N from
//           n_bulk, or `atmosphere` as the density proxy when n_bulk is null),
//           BOTH branches clamped at the counted T_MAX_PHYS rail;
//   Pass 2  conduction relaxation (double-buffered gather);
//   Pass 3  ambient cooling (solids only, vacuum-exposed 4x).
// Returns the number of T_MAX_PHYS rail engagements THIS call (folds into the
// solver's own t_max_phys_hits counter — backend-agnostic telemetry).
//
// THERMAL-MASS AXIS AMENDMENT (P2, 2026-07-30 —
// docs/thermal_mass_axis_design_2026-07-25.md + its build addendum §3): read
// every "solid" above as the THERMAL medium `thermal_solid` (`thermal_mass >
// 0`), NOT the FLOW mask `solid` (`permeability <= 0`). This mirrors P1 on the
// CPU (temperature_solver.{h,cpp}) at EXACTLY the same six medium tests, so the
// two backends stay bit-identical on maps that carry furniture — the material
// that is permeable (gas seeps past a crate) AND thermally solid (a crate holds
// an object temperature). `solid` survives here only as the documented nullptr
// fallback for `thermal_solid`; nothing else in the .cu reads it.
int64_t temperature_step(
    int32_t* temperature,           // Q16.16 (h,w) — in/out
    const int32_t* heat,            // Q16.16 (h,w) — per-tick deposit (read)
    const int32_t* heat_inv_shift,  // (h,w) per-tile log2(thermal_mass)
    const int32_t* face_shift,      // (h,w,4) per-tile face shifts (N,S,E,W)
    const bool* solid,              // (h,w) physics FLOW mask (permeability<=0);
                                    // since the thermal-mass axis it is read
                                    // ONLY as the `thermal_solid` fallback
    const bool* is_vacuum,          // (h,w) physics vacuum mask
    const int32_t* atmosphere,      // Q16.16 (h,w) — exposure test + N proxy
    const int32_t* n_bulk,          // Q16.16 (h,w) real N_total; null -> atm proxy
    const int32_t* wind_x,          // Q16.16 (h,w) wind; null -> Pass 0 advect skip
    const int32_t* wind_y,          // Q16.16 (h,w) wind; null -> Pass 0 advect skip
    int no_face,                    // sentinel: face_shift==no_face -> skip
    int cool_shift,                 // interior cooling shift — since the
                                    // cool-shift axis this is (a) the fallback
                                    // when cool_shift_grid is null and (b) the
                                    // reference for the vacuum OFFSET below
    int cool_shift_vacuum,          // space-exposed cooling shift (faster); with
                                    // cool_shift it defines the offset
                                    // (cool_shift - cool_shift_vacuum) applied
                                    // to the per-tile shift on exposed tiles
    float o2_vacuum_thresh,         // config dial (quantized on host)
    float c_v,                      // gas heat capacity (deposit divide)
    float n_floor_heat,             // per-tile N divisor floor (deposit)
    float gas_advection_rate,       // Pass 0 wind->displacement scale
    float t_max_phys,               // v2.4 physical-max T rail (clamp + count)
    int h, int w,
    float dt,                       // tick seconds; <=0 skips Pass 0 advection
    const bool* is_ambient = nullptr,   // BC: ring wiped to ΔT=0 in Pass 0 (null=space)
    // THERMAL-MASS AXIS (P2): the per-medium THERMAL mask (`thermal_mass > 0`,
    // GameMap.thermal_solid) the six medium tests key on instead of `solid`.
    // Default nullptr -> fall back to `solid`, i.e. the pre-patch behaviour and
    // the same back-compat idiom the CPU solver's signature uses. Equal to
    // `solid` elementwise on any furniture-free map (addendum D4), so the
    // fallback is not a second code path in practice.
    const bool* thermal_solid = nullptr,
    // COOL-SHIFT AXIS (2026-07-30): int32 (h,w) — the per-tile AMBIENT-DECAY
    // shift (`GameMap.cool_shift`, the per-material `cool_shift` column
    // projected by the material grid), the LOSS-side twin of `heat_inv_shift`.
    // Pass 3 does `T -= T >> cool_shift_grid[i]`; a vacuum-exposed tile takes
    // `max(cool_shift_floor, cool_shift_grid[i] - (cool_shift -
    // cool_shift_vacuum))` — ONE dial per material, the space discount stays a
    // single global rule. Default nullptr -> the `cool_shift` scalar for every
    // tile (the pre-axis behaviour; same back-compat idiom as `thermal_solid`).
    const int32_t* cool_shift_grid = nullptr,
    // Low clamp on that subtraction, == config [physics.thermal] SHIFT_MIN.
    // Load-bearing: a material legally sitting AT the floor would otherwise
    // derive an exposed shift of 0 == `T -= T` (an instant total wipe).
    int cool_shift_floor = 2,
    // P-F1a (v7.2): out-param for the Pass-1 LOW rail's engagement count (the
    // return value stays the T_MAX_PHYS count, so no existing caller moves).
    // The radiation fold is the only SIGNED path into `temperature`; the rail
    // is a counted diagnostic that must be INERT in every gate scenario.
    int64_t* low_rail_hits_out = nullptr,
    // P-R4 (ruling A1.7): the SIGNED radiation accumulator the raycaster's
    // net-T⁴ exchange fills. Folded in Pass 1 BEFORE the heat deposit, through
    // `shr_round0(rad_net[i], heat_inv_shift[i])` and a SYMMETRIC saturating
    // add — the exact CPU twin (temperature_solver.cpp Pass 1). nullptr -> no
    // fold, byte-identical to pre-P-R4.
    const int32_t* rad_net = nullptr,
    // P-E2a/P-E2b (design §2.3/§2.2): out-param for the SEVEN energy counters,
    // accumulated (+=) into the caller's TemperatureSolver fields so telemetry
    // is identical whichever backend ran. Slot order is PINNED and mirrored by
    // the C_* enum in cuda_temperature.cu and by the CPU field order:
    //   0 e_cond_trunc_sum  1 e_cond_cap_sum  2 cond_limit_hits
    //   3 e_cool_sum        4 e_vac_wipe_sum  5 e_ring_pin_sum
    //   6 e_deposit_drop_sum (P-E2b, Pass-1 attenuation drop, L3-7)
    // nullptr -> the counters are still computed on-device (they cost one
    // atomicAdd per engaged cell) but discarded, exactly like the rail counts.
    int64_t* energy_counters_out = nullptr);

// The number of slots `energy_counters_out` must have room for.
constexpr int TEMPERATURE_ENERGY_SLOTS = 7;

// Backend selection (S1 gate + integration). When true, PhysicsEngine::step_tail
// runs temperature on the GPU instead of the CPU solver. Defaults false so the
// game + suite run on the CPU path unchanged until explicitly switched.
bool temperature_backend_is_cuda();
void set_temperature_backend_cuda(bool on);

}  // namespace breach_cuda
