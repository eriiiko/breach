// PhysicsEngine — per-tick orchestration moved out of Python (Patch 1 S4).
//
// This translation unit is compiled /fp:precise (the global build is /fp:fast;
// see cpp/CMakeLists.txt set_source_files_properties for this file). S4a moves
// only the per-tick TAIL — three pure solver calls — so /fp:precise is
// 0-ULP-trivial here (no new arithmetic lives in this file yet). The strict
// rounding matters for the LATER glue ports (the IMEX substep loop and the
// W3/W5 water accounting), which carry real float math that must match numpy's
// strict-IEEE rounding bit-for-bit; setting the TU up now means those ports
// inherit it without a second CMake change.

#include "physics_engine.h"

#include <algorithm>   // std::max
#include <cmath>       // std::ceil

// Patch 1 S4a — the per-tick TAIL, the three trailing pure-solver-call steps of
// PhysicsRunner.step (everything AFTER the IMEX atmosphere/smoke substep loop):
//
//   1. W6a ripple (visual-only surface wave) — guarded by the dormancy test.
//   2. Fire feedback step — returns the burn-through wall list.
//   3. Temperature: heat->temperature conversion + conduction + ambient cooling.
//
// Same three calls, same argument order, on THIS engine's own solver instances
// (this->water / this->fire / this->temperature). No new arithmetic -> bit-
// identical (gated by the per-cell A/B harness).
std::vector<std::pair<int, int>> PhysicsEngine::step_tail(
        // ripple group
        float* ripple, float* ripple_v,
        const float* water_depth, const float* wave_p,
        const bool* solid,
        // fire group
        float* fire_field, float* atmosphere, float* smoke_field, float* wall_hp,
        const int32_t* temperature, const float* wind_x, const float* wind_y,
        const bool* is_vacuum, const bool* flammable,
        // temperature group
        int32_t* temperature_mut, const int32_t* heat,
        const int32_t* heat_inv_shift, const int32_t* face_shift,
        int h, int w, float sim_time) const {

    // --- 1. W6a ripple (PhysicsRunner._step_ripple) ----------------------
    // Reproduce the Python dormancy guard EXACTLY: skip step_ripple unless
    // there is water now OR leftover ripple anywhere. `np.ndarray.any()` is
    // "any element != 0" — for the float ripple state the equivalent is a
    // scan for any non-zero element (NaN also counts as non-zero, matching
    // numpy's any() truthiness). Ripple is zero wherever depth is zero by
    // construction, so on a dry ship both scans are cheap and the whole tail
    // stays bit-identical to before the water system existed.
    bool water_any = false;
    bool ripple_any = false;
    const int n = h * w;
    for (int i = 0; i < n; ++i) {
        if (water_depth[i] != 0.0f) { water_any = true; break; }
    }
    if (!water_any) {
        for (int i = 0; i < n; ++i) {
            if (ripple[i] != 0.0f) { ripple_any = true; break; }
        }
    }
    if (water_any || ripple_any) {
        // step_ripple writes only ripple / ripple_v; water_depth / wave_p /
        // solid are read-only (the locked canon rule). Same args, same order
        // as the Python call: (ripple, ripple_v, water_depth, wave_p, solid, dt).
        this->water.step_ripple(ripple, ripple_v, water_depth, wave_p, solid,
                                h, w, sim_time);
    }

    // --- 2. Fire feedback step (PhysicsRunner: self.fire.step) ------------
    // Arg order cross-checked against bindings.cpp FireSimulation.step and the
    // Python call site:
    //   fire, atmosphere, smoke, wall_hp, temperature(const), wind_x, wind_y,
    //   is_wall(=solid), is_vacuum, flammable, dt.
    // `is_wall` IS `gmap.solid` (the Python passes gmap.solid as the is_wall
    // arg); `temperature` here is the const view (the previous-tick conduction
    // field the fire reads).
    std::vector<std::pair<int, int>> destroyed = this->fire.step(
        fire_field, atmosphere, smoke_field, wall_hp,
        temperature, wind_x, wind_y,
        solid, is_vacuum, flammable,
        h, w, sim_time);

    // --- 3. Temperature pass (PhysicsRunner: self.temperature.step) ------
    // Arg order cross-checked against bindings.cpp TemperatureSolver.step and
    // the Python call site:
    //   temperature(mut), heat, heat_inv_shift, face_shift, solid, is_vacuum,
    //   atmosphere.
    // `temperature_mut` is the SAME array as the fire's const `temperature`
    // (gmap.temperature in Python) — the binding extracts both a const and a
    // mutable pointer from the one numpy array. The fire read it above; the
    // temperature solver now updates it in place for next tick.
    this->temperature.step(
        temperature_mut, heat, heat_inv_shift, face_shift,
        solid, is_vacuum, atmosphere,
        h, w);

    return destroyed;
}

// Patch 1 S4b — the IMEX atmosphere/smoke substep loop, lifted verbatim from the
// middle of PhysicsRunner.step (the block between _step_water and step_tail).
// BIT-IDENTICAL to the Python: the arithmetic below reproduces numpy's strict-
// IEEE rounding and the EXACT double->float32 cast points at the pybind boundary.
// See the header for the precision contract; the inline comments mark each spot
// where the precision (not just the value) had to be matched.
void PhysicsEngine::run_substeps(
        float* wave_p, float* wave_v, float* wave_source, float* atmosphere,
        float* wind_x, float* wind_y,
        const bool* obstacles, const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability, const float* dyn_wave_absorb,
        float* gas, const float* gas_diffusion, int n_gases,
        const float* sink_x, const float* sink_y,
        int h, int w, float sim_time) {

    // --- The integer cliff: n = max(1, int(ceil(sim_time / dt))) ----------
    // Python: dt = self.atmos.max_dt() is a float32 PROMOTED to a Python double;
    // sim_time / dt is a DOUBLE division; ceil + int truncates. Match it byte for
    // byte: promote max_dt() to double, do the division in double, ceil in double,
    // truncate to int. A 1-ULP slip here flips n and desyncs the whole tick.
    const double dt = (double)this->atmos.max_dt();
    const int n = std::max(1, (int)std::ceil((double)sim_time / dt));
    // dt_actual stays DOUBLE — Python keeps `dt_actual = sim_time / n` as a double
    // and pybind narrows it to float32 only at the .step(...) call boundary.
    const double dt_actual = (double)sim_time / n;
    // dt_smoke = dt_actual * dt_scale, BOTH doubles, narrowed to float32 only at
    // the smoke.step boundary. dt_scale is the float member promoted to double;
    // the order (double-multiply THEN cast) must match pybind. The dt_scale is
    // deliberately applied AGAIN inside smoke.step (the known double-application);
    // we reproduce Python's value exactly and DO NOT "fix" it (not this patch).
    const double dt_smoke = dt_actual * (double)this->smoke.dt_scale;

    const int plane = h * w;  // elements per gas plane (gas is (N,h,w) contiguous)

    for (int s = 0; s < n; ++s) {
        // Atmosphere substep — arg order matches the AtmosphereSolver.step
        // binding (wave_p, wave_v, wave_source, atmosphere, wind_x, wind_y,
        // obstacles, is_wall(=solid), is_vacuum, permeability, wave_absorb, dt).
        // (float)dt_actual reproduces pybind's double->float32 cast.
        this->atmos.step(
            wave_p, wave_v, wave_source, atmosphere,
            wind_x, wind_y,
            obstacles, solid, is_vacuum,
            dyn_permeability,
            dyn_wave_absorb,
            h, w,
            (float)dt_actual);

        // Per-gas smoke transport (engine/05 §6.2, M1). Loop the N gas planes;
        // skip an all-zero plane (reproduces numpy's `.any()` -> "any element
        // != 0"; a 0.0/-0.0/NaN scan, matching numpy truthiness). Set d_smoke
        // BEFORE each step (member-set, EXACTLY as Python — not a parameter).
        for (int gi = 0; gi < n_gases; ++gi) {
            float* gas_slice = gas + (size_t)gi * plane;
            bool any = false;
            for (int i = 0; i < plane; ++i) {
                if (gas_slice[i] != 0.0f) { any = true; break; }
            }
            if (!any) {
                continue;  // empty slice — nothing to transport (matches `.any()`)
            }
            // gas_diffusion[gi] is a float32; (float)... is the exact float32->
            // double->float32 round-trip Python's float(gas_diffusion[gi]) +
            // the d_smoke (float) member store performs — bit-identical.
            this->smoke.d_smoke = (float)gas_diffusion[gi];
            // (float)dt_smoke reproduces pybind's double->float32 cast.
            this->smoke.step(
                gas_slice, wind_x, wind_y,
                sink_x, sink_y,
                obstacles, solid, is_vacuum,
                dyn_permeability,
                h, w,
                (float)dt_smoke);
        }
    }
}
