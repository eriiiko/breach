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
    // T5b step 7 / R1: `cool_shift` and `cool_shift_vacuum` stood here.
    // Pass 3 is deleted on both backends -- the sweep computes the real
    // radiative loss, so a hand-rolled Newtonian relaxation beside it
    // counted the same physics twice.
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
    // T5b step 7: `cool_shift_grid` and `cool_shift_floor` stood here, and
    // are deleted with Pass 3.
    // P-F1a (v7.2): out-param for the Pass-1 LOW rail's engagement count (the
    // return value stays the T_MAX_PHYS count, so no existing caller moves).
    // The radiation fold is the only SIGNED path into `temperature`; the rail
    // is a counted diagnostic that must be INERT in every gate scenario.
    int64_t* low_rail_hits_out = nullptr,
    // P-R4 (ruling A1.7): the SIGNED radiation accumulator the raycaster's
    // net-T⁴ exchange fills. **int64** since ray-engine-v2 P3a-1 (design v3
    // §3, rows 26/35). Folded in Pass 1 BEFORE the heat deposit, through
    // `shr_round0_i64(rad_net[i], heat_inv_shift[i])` and `sat_add_q16_i64`
    // — the exact CPU twin (temperature_solver.cpp Pass 1), and the int64
    // twins agree with the narrow forms on every int32-range value, which is
    // what makes the widening byte-identical. nullptr -> no fold,
    // byte-identical to pre-P-R4.
    const int64_t* rad_net = nullptr,
    // P-E2a/P-E2b/arc #54/P-G5 (design §2.3/§2.2/§2.7 row 3/thermostat ledger):
    // out-param for the energy counters, accumulated (+=) into the caller's
    // TemperatureSolver fields so telemetry is identical whichever backend
    // ran. Slot order is PINNED and mirrored by the C_* enum in
    // cuda_temperature.cu and by the CPU field order:
    //   0 e_cond_trunc_sum  1 e_cond_cap_sum  2 cond_limit_hits
    //   3 e_vac_wipe_sum    4 e_ring_pin_sum
    //   5 e_deposit_drop_sum (P-E2b, Pass-1 attenuation drop, L3-7)
    //   6 e_gas_deposit_sum (arc #54, Pass 1 heat->E on gas, net)
    //   7 e_gas_cond_sum    (arc #54, Pass 2 conduction into gas E, net)
    //   8 e_gas_rail_sum    (arc #54, Pass 1's T_MAX_PHYS rail, signed)
    //   9 e_solid_deposit_sum (P-G5, Pass 1 landing on thermal solids, signed)
    //  10 e_solid_cond_sum    (P-G5, Pass 2 landing on thermal solids, signed)
    //  11 rad_clamp_hits      (T5b, the Pass-1 clamp's engagement COUNT)
    //  12 e_rad_clamp_drop_sum (P5c, the clamp's withheld energy, >= 0 —
    //                          APPENDED, so no pinned index moved)
    //  13 e_rad_boundary_export_sum (P5c follow-up: rad_net on a gas cell
    //                          outside the accountable set, exported; signed)
    //  14 e_rad_floor_drop_sum (P5c follow-up: the floored chain's unlanded
    //                          remainder below n_floor_heat; signed) — both
    //                          APPENDED, no pinned index moved
    // T5b step 7: slots 3 (e_cool_sum) and 12 (e_thermostat_sum) are
    // DELETED with Pass 3, and every survivor below them RENUMBERED. The
    // indices are pinned positional and physics_engine.cpp folds them by
    // index, so the two move together, in this commit.
    // nullptr -> the counters are still computed on-device (they cost one
    // atomicAdd per engaged cell) but discarded, exactly like the rail counts.
    int64_t* energy_counters_out = nullptr,
    // arc #54 §2.7 row 3 (gas-energy conservation, design): the CONSERVED gas
    // energy field. With it supplied (non-null), an ACCOUNTABLE gas cell's
    // Pass-1 deposit and Pass-2 conduction sum land in `gas_energy` through
    // the seam (gas_energy.h — device-compatible, FP_HD) instead of the
    // T-form law; nullptr -> the pre-#54 T-form law, bit for bit. `t_amb_q`
    // is T_AMB_K raw, only read when gas_energy is supplied. P5c: with it,
    // the radiation fold's GAS branch runs too (an accountable gas cell's
    // rad_net through the staged chain, clamped on the deposit), the CPU
    // twin's block verbatim.
    int64_t* gas_energy = nullptr,
    int32_t t_amb_q = 0,
    // P-G5: out-param for `solid_energy_books_sum` — a SNAPSHOT (ASSIGNED,
    // not accumulated) of Σ thermal_mass_raw·T_raw over thermal_solid cells,
    // as of the end of THIS call. Computed on the HOST after the final D2H
    // temperature copy (cheap — one more pass over already-resident host
    // arrays), from the same `conduction::cell_capacity_q` kit the device
    // capacity build uses, so the two backends cannot drift. nullptr ->
    // skipped.
    int64_t* solid_books_out = nullptr,
    // ---- ray-engine-v2, THE FLIP (T5b step 6; design v3 P3) --------------
    // The MAXIMUM-PRINCIPLE CLAMP's two planes, the GPU twin of the CPU
    // solver's `rad_fluence` / `e_table` pair:
    //     T_new = min(T_after, max(T_before, E^-1(Phi)))
    // `rad_fluence` is the sweep's own Phi at the cell (int64 (h,w), H2D'd
    // here); `e_table` is the E° table (int64, EMISSIVE_TABLE_N entries) whose
    // inverse `e_inv_q` is FP_HD and therefore the SAME function the CPU calls.
    // BOTH null -> no clamp, byte-identical to the pre-flip kernel, which is
    // what every direct-binding caller and every pre-flip test still gets.
    // The engagement count comes back in slot 11 (see below).
    const int64_t* rad_fluence = nullptr,
    const int64_t* e_table = nullptr,
    int e_table_n = 0);

// The number of slots `energy_counters_out` must have room for.
// T5b step 6: 13 -> 14 (slot 13 = `rad_clamp_hits`, the clamp's engagement
// COUNT, appended at the END). T5b step 7: 14 -> 12, because C_COOL and
// C_THERMOSTAT are DELETED with Pass 3 -- a REMOVAL, which renumbers the
// survivors (design v3 / L2 names this exact hazard), so the enum in the
// .cu and the by-index fold in physics_engine.cpp are edited with it.
// P5c: 12 -> 13 -- slot 12 = `e_rad_clamp_drop_sum`, APPENDED at the end, so
// every pinned index keeps its meaning (the rule: append, never renumber).
// P5c follow-up: 13 -> 15 -- slots 13 (`e_rad_boundary_export_sum`) and 14
// (`e_rad_floor_drop_sum`), APPENDED the same way.
constexpr int TEMPERATURE_ENERGY_SLOTS = 15;

// Backend selection (S1 gate + integration). When true, PhysicsEngine::step_tail
// runs temperature on the GPU instead of the CPU solver. Defaults false so the
// game + suite run on the CPU path unchanged until explicitly switched.
bool temperature_backend_is_cuda();
void set_temperature_backend_cuda(bool on);

}  // namespace breach_cuda
