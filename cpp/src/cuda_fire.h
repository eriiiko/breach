#pragma once
// ============================================================================
// CUDA-S6 — FireSimulation::step on the GPU (bit-identical).
// ============================================================================
//
// A faithful, bit-identical GPU port of FireSimulation::step
// (fire_simulation.cpp ~44-301), RE-DERIVED for the EOS refactor (P6.8): the O2
// gate reads the REAL bulk O2 density plane `n_o2` (not the old atmosphere/P
// proxy). The own-tile plume->T shim (formerly P3 below) was DELETED at P-R2
// (docs/radiation_raycaster_extinction_ruling_2026-07-31.md A2) — it was the
// one `temperature[]` writer bypassing heat_inv_shift; TemperatureSolver is now
// the field's ONLY writer, and P-R4's radiation pass will be the next writer,
// through its own rad_net[] plane. The per-tile signed-logistic intensity
// FEEDBACK (ignition/spread is elsewhere) → wall burn-through → final clamp.
// (Smoke emission into neighbours was DELETED at P-S1, 2026-08-15 — Erik's
// single-source ruling, docs/smoke_single_source_design_2026-07-24.md; killed
// the smoke->N2 pressure pump measured by docs/storm_audit_2026-08-14.md
// §4.2. Combustion soot is now the ONE fire-smoke source, cpp/src/
// combustion.cpp.) Three passes (P3's slot retired at P-R2, P4's at P-S1,
// NEITHER renumbered), launched as a barriered chain (separate launches =
// grid barriers between dependent passes):
//
//   P1 early-exit         HOST: if max(fire) < thresh -> return {} (fields untouched)
//   P2 logistic feedback  fire += dt*(grow-die); snap-extinguish below I_min
//                         (O2 gate = n_o2 neighbour mean)
//   P5 wall burn-through  wall_hp[i] -= dmg; collect destroyed; fire[i]=0
//   P6 final clamp        fire, smoke -> [0, FP_ONE] (smoke is READ-mostly
//                         since P-S1 — no pass writes it, only clamps)
//
// The synced fields fire, smoke, wall_hp (int32 Q16.16) come out byte-for-byte
// identical to the CPU step on every architecture (tol 0); temperature is now
// read-only through this pass (untouched, so it trivially stays identical too).
// The returned destroyed-walls list matches as a SET (order is irrelevant — the
// caller processes each cell independently; only field state is synced).
//
// P5's destroyed list is collected via a device atomicAdd counter + an index
// array (the ONLY scatter left in this kernel since P4's atomicAdd smoke
// deposit was deleted); the gate checks SET equality (no order, no drops/dupes).
//
// Plain C++ declaration header (no CUDA types) so the .cpp TUs (bindings.cpp,
// physics_engine.cpp, compiled by cl.exe even in the CUDA build) can include it;
// cuda_fire.cu provides the definitions. Compiled only when BREACH_CUDA.
#include <cstdint>
#include <utility>
#include <vector>

namespace breach_cuda {

// ONE FireSimulation::step on the GPU — IN-PLACE on fire / atmosphere / smoke /
// wall_hp (h,w). RETURNS the destroyed-walls list (unlike S1-S5, which only
// mutate fields) — the (y,x) coordinates where walls burned through, in arbitrary
// order. Mirrors FireSimulation::step exactly (the 5 passes above incl. the host
// max early-exit + the atomicAdd smoke scatter + the device-collected destroyed
// list). Because this is a FREE function (not a method on the solver), the
// solver's scalar FireParams dials are passed explicitly:
//   k_grow / k_die / fire_T_ext / fire_T_span / fuel_ref / o2_frac_ext /
//   o2_frac_full / I_min / k_wind_fan / k_wind_strip / wall_damage /
//   temp_scale. (smoke_emission RETIRED at P-S1 — see the file header.)
// All quantized step constants + the load-time reciprocals (make_recip) are
// precomputed ON THE HOST in double, VERBATIM from the CPU load-time block, and
// passed as scalar kernel args.
//
// n_o2 / n_total / wind_x / wind_y / is_wall / is_vacuum / flammable are
// READ-ONLY; atmosphere is read-only + vestigial (unread). temperature is now
// READ-ONLY through this pass too — the plume->T shim that used to write it
// (P3) is deleted (P-R2 — docs/radiation_raycaster_extinction_ruling_2026-07-
// 31.md A2); P-R4's radiation pass is the next writer, through rad_net[].
//
// PERF NOTE (residency is S8): per-call H2D of all fields + masks and a D2H of
// the mutated fields + the destroyed list, once per tick — deliberately
// deferred.
std::vector<std::pair<int, int>> fire_step(
    int32_t* fire,             // Q16.16 (h,w) — in/out (intensity, [0,1])
    const int32_t* atmosphere, // Q16.16 (h,w) — read-only + VESTIGIAL (EOS P4:
                               //   the CPU step keeps it for ABI parity but no
                               //   longer reads it — never uploaded/returned here)
    const int32_t* n_o2,       // Q16.16 (h,w) — read-only (continuous-O2 law: the
                               //   mole-fraction NUMERATOR, real bulk-O2 density)
    const int32_t* n_total,    // Q16.16 (h,w) — read-only (continuous-O2 law: the
                               //   mole-fraction DENOMINATOR, real N_total = Σ
                               //   conservative bulk planes O2+N2)
    int32_t* smoke,            // Q16.16 (h,w) — in/out (emission scatter into nbrs)
    int32_t* wall_hp,          // Q16.16 (h,w) — in/out (burn-through depletion)
    int32_t* temperature,      // Q16.16 (h,w) — READ ONLY (the `hot` gate). The
                               //   plume->T shim WRITE that used to live here
                               //   (EOS P3) is deleted (P-R2); P-R4's radiation
                               //   pass is the next writer, via rad_net[], not
                               //   this parameter. Pointer stays non-const this
                               //   patch (mirrors the CPU signature, unchanged).
    const int32_t* wind_x,     // Q16.16 (h,w) — read-only (|wind| via sqrt_q16_dev)
    const int32_t* wind_y,     // Q16.16 (h,w) — read-only
    const bool* is_wall,
    const bool* is_vacuum,
    const bool* flammable,
    int h, int w, float dt,
    // FireParams dials (verbatim the CPU `params`; p_expand_ref RETIRED — it
    // was the plume self-limiter's saturation gate, and the plume itself is
    // deleted (P-R2); continuous-O2 law: P_min/P_full RETIRED from this gate,
    // REPLACED by the o2_frac_ext/o2_frac_full mole-fraction span — NOTE the
    // span's upper end is the FULL-RESPONSE reference o2_frac_full (pure O2),
    // NOT the ambient dial o2_frac_amb, which the law no longer reads; see
    // FireParams::o2_frac_full):
    float k_grow, float k_die, float fire_T_ext, float fire_T_span,
    float fuel_ref, float o2_frac_ext, float o2_frac_full, float I_min,
    float k_wind_fan, float k_wind_strip,
    float wall_damage,
    float temp_scale,
    // CAPACITY LAW (P-R3, ruling A3) — `c`, the growth term's carrying capacity
    // per unit availability. The host precompute bakes INV_C = quantize(1/c)
    // VERBATIM as the CPU load-time block does; `<= 0` means the ceiling is OFF
    // (INV_C = 0). See FireParams::I_cap_per_avail.
    float I_cap_per_avail,
    // R1 O2f-RENORMALIZATION (fire session #12, docs/fire_3c_design_2026-09-01
    // .md "Ruling R1"): the SUSTAIN span's upper reference is now o2_frac_amb
    // (ambient), NOT o2_frac_full (pure O2, above — which stays live on the
    // DEMAND side, cuda_combustion.cu, unchanged). o2f_cap is the NEW clamp
    // ceiling (o2f is no longer bounded to [0,1] — enrichment above ambient can
    // register, up to this cap). See FireParams::o2_frac_amb / o2f_cap.
    float o2_frac_amb, float o2f_cap,
    // R3 hot-burns-faster (fire session #12, docs/fire_3c_design_2026-09-01
    // .md "Ruling R3"): hotf_cap is the ceiling on the UNCAPPED-AT-1 hotf ramp
    // (the SAME (T-T_ext)/T_span ramp as `hot`, but read only by the wall-burn
    // term below — `hot` itself is unchanged and stays capped at 1 for the
    // sustain gate). See FireParams::hotf_cap / fire_simulation.cpp.
    float hotf_cap,
    // FUEL-FRACTION AXIS (2026-07-30) — OPTIONAL read-only int64 (h,w) plane:
    // per tile, the make_recip reciprocal of that tile's MATERIAL's full-health
    // hp (GameMap.fuel_recip). The fuel term becomes
    // F = clamp01(recip_mul(wall_hp[i], fuel_recip[i])) — this tile's OWN fuel
    // fraction, not wood's. nullptr -> the `fuel_ref` scalar above, i.e. the
    // pre-axis law bit-for-bit (nothing is allocated or copied in that case).
    // The CPU twin takes the identical nullable plane; tol 0 between them.
    const int64_t* fuel_recip = nullptr,
    // PER-MATERIAL EXTINCTION TEMPERATURE (P-R3, ruling A3 ride-along) —
    // OPTIONAL read-only int32 (h,w) Q16.16 plane: per tile, that material's
    // `ignition_temp - ignition_to_ext_delta`, quantized once at load
    // (GameMap.fire_T_ext_plane). `hot = clamp01((T - plane[i]) * recip_T_span)`.
    // nullptr -> the `fire_T_ext` scalar above, i.e. the pre-derivation law
    // bit-for-bit (nothing allocated or copied in that case). The CPU twin
    // takes the identical nullable plane; tol 0 between them.
    const int32_t* fire_T_ext_plane = nullptr);

// Backend selection (S6 gate + integration). When true, PhysicsEngine::step_tail
// runs the fire step on the GPU instead of the CPU FireSimulation::step. Defaults
// false so the game + suite run on the CPU path unchanged until explicitly
// switched.
bool fire_backend_is_cuda();
void set_fire_backend_cuda(bool on);

}  // namespace breach_cuda
