#pragma once
// PhysicsEngine — owns the (stateless, const-step) solver instances.
//
// engine/02: "the physics engine CONTAINS the grid owner" — this is the C++
// home the CUDA port plugs into. Patch 1 S3 is the SCAFFOLD: the engine owns the
// solvers' lifetime + their tunable params; the per-tick orchestration (the
// substep loop, the W3/W5 glue) moves INTO this class in S4. The solvers are
// genuinely stateless (every step() is const; scratch is the reused mutable
// members from S2), so the engine just holds them and exposes references — no
// arithmetic lives here yet, so S3 is bit-identical by construction.
//
// NOTE (engine/02 + the unification plan v2 §3a): the engine does NOT cache
// field pointers. The solvers re-fetch each numpy array's raw pointer per step()
// (the pybind get_2d pattern), so the engine is robust to GameMap reallocation
// on reset() and to in-place field rewrites. The engine owns SOLVERS, not grids
// (yet); grid residency is a later (GPU) patch.

#include <cstdint>
#include <utility>
#include <vector>

#include "atmosphere_solver.h"
#include "smoke_dynamics.h"
#include "fire_simulation.h"
#include "temperature_solver.h"
#include "raycaster.h"
#include "water_solver.h"

class PhysicsEngine {
public:
    AtmosphereSolver  atmos;
    SmokeDynamics     smoke;
    FireSimulation    fire;
    TemperatureSolver temperature;
    Raycaster         raycaster;
    WaterSolver       water;

    // --- Patch 1 S4a: the per-tick orchestration TAIL --------------------
    // Moves the three trailing PURE-SOLVER-CALL steps of PhysicsRunner.step
    // (everything AFTER the IMEX substep loop) into C++: the W6a ripple, the
    // fire feedback step, and the temperature heat->conduction->cooling pass —
    // in that exact order, calling THIS engine's own solver instances. No new
    // arithmetic: it is the same three calls Python made, so it is bit-identical
    // (gated by the per-cell A/B harness). Lives in physics_engine.cpp, compiled
    // /fp:precise so the LATER glue ports (substep loop, W3/W5 water accounting)
    // inherit strict-IEEE rounding to match numpy.
    //
    // Reproduces the ripple DORMANCY GUARD from PhysicsRunner._step_ripple:
    // step_ripple is skipped unless water_depth.any() || ripple.any().
    //
    // Returns the (y, x) burn-through list from FireSimulation::step (the caller
    // runs gmap.destroy_wall on each), exactly as the Python tail did.
    //
    //   ripple, ripple_v               : float (h, w) — W6a ripple state (mutated)
    //   water_depth                    : float (h, w) — read by ripple + guard
    //   wave_p                         : float (h, w) — ripple splash source (read)
    //   solid / is_wall                : bool  (h, w) — the solid mask
    //   fire, atmosphere, smoke, wall_hp : float (h, w) — fire step (mutated)
    //   temperature                    : int32 (h, w) Q16.16 — fire reads, temp writes
    //   wind_x, wind_y                 : float (h, w) — shared wind (read)
    //   is_vacuum, flammable           : bool  (h, w)
    //   heat                           : int32 (h, w) Q16.16 — heat deposit (read)
    //   heat_inv_shift                 : int32 (h, w) — per-tile inverse mass shift
    //   face_shift                     : int32 (h, w, 4) — conduction face shifts
    std::vector<std::pair<int, int>> step_tail(
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
        int h, int w, float sim_time) const;

    // --- Patch 1 S4b: the IMEX atmosphere/smoke substep loop -------------
    // Moves the per-tick IMEX substep block out of PhysicsRunner.step (Python)
    // into C++ — the loop that runs BETWEEN the water/fire-heat steps (still
    // Python, before) and step_tail (already C++, after). It advances the
    // atmosphere wave+diffusion and the per-gas smoke transport `n` times, where
    // `n` is derived from the atmosphere solver's CFL bound and `sim_time`.
    //
    // BIT-IDENTITY is the whole point — this reproduces Python's arithmetic
    // EXACTLY (the /fp:precise TU makes the FP strict-IEEE; we must match the
    // PRECISION + ORDER numpy's pybind boundary produced):
    //   * `n` is an INTEGER CLIFF: n = max(1, (int)ceil((double)sim_time / dt))
    //     where dt = (double)atmos.max_dt(). DOUBLE division — a 1-ULP slip flips
    //     n and desyncs the whole tick.
    //   * `dt_actual` and `dt_smoke` stay DOUBLE until the solver-call boundary:
    //     dt_actual = (double)sim_time / n; dt_smoke = dt_actual * (double)dt_scale.
    //     They are cast to float ONLY when passed to the solvers — matching
    //     pybind's double->float32 cast at the .step() call site (do NOT pre-
    //     narrow; the order double-multiply-then-cast must match).
    //   * Per-gas loop: gi over the N planes of `gas` ((N,h,w) contiguous, plane
    //     gi at gas + gi*h*w); SKIP an all-zero plane (reproduces numpy .any());
    //     set this->smoke.d_smoke = (float)gas_diffusion[gi] BEFORE each
    //     smoke.step (member-set EXACTLY as Python; NOT a parameter — that is a
    //     later GPU-prep cleanup, not this bit-identical step).
    //
    // sink_fields() stays PYTHON — the runner fetches sink_x/sink_y and passes
    // them in (it is a lazy BFS Python method, not called from C++).
    //
    //   wave_p, wave_v, wave_source : float (h, w) — atmosphere wave state
    //   atmosphere                  : float (h, w) — bulk pressure
    //   wind_x, wind_y              : float (h, w) — written by atmos, read by smoke
    //   obstacles, solid, is_vacuum : bool  (h, w) — masks (solid == is_wall)
    //   dyn_permeability            : float (h, w) — per-tick face permeability
    //   dyn_wave_absorb             : float (h, w) — per-cell wave absorption
    //   gas                         : float (N, h, w) — the per-gas density planes
    //   gas_diffusion               : float (N,)     — per-gas base diffusion
    //   sink_x, sink_y              : float (h, w) — smoke sink direction (Python-fetched)
    void run_substeps(
        float* wave_p, float* wave_v, float* wave_source, float* atmosphere,
        float* wind_x, float* wind_y,
        const bool* obstacles, const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability, const float* dyn_wave_absorb,
        float* gas, const float* gas_diffusion, int n_gases,
        const float* sink_x, const float* sink_y,
        int h, int w, float sim_time);
};
