#pragma once
// ============================================================================
// CUDA-S6 — FireSimulation::step on the GPU (bit-identical).
// ============================================================================
//
// A faithful, bit-identical GPU port of FireSimulation::step
// (fire_simulation.cpp ~44-314), RE-DERIVED for the EOS refactor (P6.8): the O2
// gate now reads the REAL bulk O2 density plane `n_o2` (not the old atmosphere/P
// proxy), and the own-tile plume deposit is the plume->T shim (a temperature
// deposit, T_FLAME_MAX self-limiter — eos-p3fix-thermal-ceiling), NOT the retired
// atmosphere-overpressure write. The per-tile signed-logistic intensity FEEDBACK
// (ignition/spread is elsewhere) → own-tile plume->T → smoke emission into
// neighbours → wall burn-through → final clamp. Five passes, launched as a
// barriered chain (separate launches = grid barriers between dependent passes):
//
//   P1 early-exit         HOST: if max(fire) < thresh -> return {} (fields untouched)
//   P2 logistic feedback  fire += dt*(grow-die); snap-extinguish below I_min
//                         (O2 gate = n_o2 neighbour mean)
//   P3 plume->T shim      temperature[i] += clamped dT deposit      (own-tile)
//   P4 smoke emission     smoke[nbr] += round(emit(fire[src]))     (SCATTER, atomicAdd)
//   P5 wall burn-through  wall_hp[i] -= dmg; collect destroyed; fire[i]=0
//   P6 final clamp        fire, smoke -> [0, FP_ONE]
//
// The synced fields fire, smoke, wall_hp, temperature (int32 Q16.16) come out
// byte-for-byte identical to the CPU step on every architecture (tol 0); the
// returned destroyed-walls list matches as a SET (order is irrelevant — the
// caller processes each cell independently; only field state is synced).
//
// DETERMINISM crux (P4): the 4 smoke emissions per source thread are deposited
// with integer atomicAdd. The deposit `delta_q = round(emission*dt*fire[src])`
// depends ONLY on fire[src] (NOT on the neighbour's current smoke), so the
// per-neighbour sum of overlapping deposits is associative + commutative ->
// order-free -> bit-identical to the CPU's sequential row-major adds. P5's
// destroyed list is collected via a device atomicAdd counter + an index array;
// the gate checks SET equality (no order, no drops/dupes).
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
// order. Mirrors FireSimulation::step exactly (the 6 passes above incl. the host
// max early-exit + the atomicAdd smoke scatter + the device-collected destroyed
// list). Because this is a FREE function (not a method on the solver), the
// solver's scalar FireParams dials are passed explicitly:
//   k_grow / k_die / fire_T_ext / fire_T_span / fuel_ref / P_min / P_full /
//   I_min / k_wind_fan / k_wind_strip / fire_pressure_gain / smoke_emission /
//   wall_damage / temp_scale / temp_gain_scale / T_FLAME_MAX.
// All quantized step constants + the load-time reciprocals (make_recip) are
// precomputed ON THE HOST in double, VERBATIM from the CPU load-time block, and
// passed as scalar kernel args.
//
// n_o2 / wind_x / wind_y / is_wall / is_vacuum / flammable are READ-ONLY;
// atmosphere is read-only + vestigial (unread). temperature is IN/OUT (plume->T).
//
// PERF NOTE (residency is S8): per-call H2D of all fields + masks and a D2H of
// the 4 mutated fields + the destroyed list, once per tick — deliberately
// deferred.
std::vector<std::pair<int, int>> fire_step(
    int32_t* fire,             // Q16.16 (h,w) — in/out (intensity, [0,1])
    const int32_t* atmosphere, // Q16.16 (h,w) — read-only + VESTIGIAL (EOS P4:
                               //   the CPU step keeps it for ABI parity but no
                               //   longer reads it — never uploaded/returned here)
    const int32_t* n_o2,       // Q16.16 (h,w) — read-only (EOS P4: the O2 gate's
                               //   real bulk-O2 neighbour-mean input)
    int32_t* smoke,            // Q16.16 (h,w) — in/out (emission scatter into nbrs)
    int32_t* wall_hp,          // Q16.16 (h,w) — in/out (burn-through depletion)
    int32_t* temperature,      // Q16.16 (h,w) — in/out (READ the `hot` gate + WRITE
                               //   the plume->T shim deposit, EOS P3)
    const int32_t* wind_x,     // Q16.16 (h,w) — read-only (|wind| via sqrt_q16_dev)
    const int32_t* wind_y,     // Q16.16 (h,w) — read-only
    const bool* is_wall,
    const bool* is_vacuum,
    const bool* flammable,
    int h, int w, float dt,
    // FireParams dials (verbatim the CPU `params`; p_expand_ref RETIRED — the
    // plume self-limiter now gates on T_FLAME_MAX, not p_expand_ref):
    float k_grow, float k_die, float fire_T_ext, float fire_T_span,
    float fuel_ref, float P_min, float P_full, float I_min,
    float k_wind_fan, float k_wind_strip, float fire_pressure_gain,
    float smoke_emission, float wall_damage,
    float temp_scale, float temp_gain_scale, float T_FLAME_MAX);

// Backend selection (S6 gate + integration). When true, PhysicsEngine::step_tail
// runs the fire step on the GPU instead of the CPU FireSimulation::step. Defaults
// false so the game + suite run on the CPU path unchanged until explicitly
// switched.
bool fire_backend_is_cuda();
void set_fire_backend_cuda(bool on);

}  // namespace breach_cuda
