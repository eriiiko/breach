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

    // --- Patch 1 S4c: the water-layer ARRAY ARITHMETIC -------------------
    // Moves the array-op core of PhysicsRunner._step_water into C++ — the part
    // AFTER the (still-Python) lazy-init + dormancy early-out + sparse
    // source-holds. The runner does those stateful/sparse steps, then calls
    // step_water ONLY when not dormant. What moves here, in order:
    //   1. substep-count derivation + the WaterSolver.step substep loop;
    //   2. the W5 flash-boil vacuum sink (boil-off -> steam puff);
    //   3. the W3 volume displacement (isothermal P*V onto atmosphere) + the
    //      flooded dyn_permeability seal;
    //   4. the final copyto(before, water_depth) — closes the accounting loop.
    //
    // BIT-IDENTITY is the whole point: the arrays are float32, the scalar params
    // are Python doubles, and numpy elementwise `f32_array OP double_scalar`
    // casts the scalar to float32 for the op. We reproduce that EXACTLY (every
    // scalar cast to float at numpy's cast point; /fp:precise makes the f32 ops
    // strict-IEEE, matching numpy). The precision pitfalls, each verified vs
    // numpy 1.26.4 and matched here:
    //   * n = max(1, (int)ceil((double)sim_time / (double)water.max_dt())) — the
    //     integer cliff in DOUBLE; wdt = (float)((double)sim_time / n) at the
    //     water.step boundary (pybind's double->float32 cast).
    //   * W5: boiling = (atmosphere < (f32)boil_p_thresh) & (water_depth > 0.0f);
    //     boil amount = min(water_depth, (f32)((double)boil_rate*(double)sim_time))
    //     — the product is DOUBLE, cast to f32 ONCE, then min in f32. steam puff
    //     = (f32)steam_yield * boiled — the multiply is in FLOAT32 (numpy keeps
    //     the f32 array's dtype; a double-multiply-then-cast is 1 ULP off). The
    //     whole block is guarded by boiling.any() (matches numpy's `.any()`).
    //   * W3: free_before/after = max((f32)ceiling_h - x, (f32)flood_eps) in f32;
    //     ratio = clip(free_before/free_after, lo, hi) = min(max(., lo), hi) with
    //     lo = (f32)(1.0/ratio_cap) (the reciprocal in DOUBLE then cast) and hi =
    //     (f32)ratio_cap; atmosphere *= ratio; flooded = free_after <= (f32)
    //     flood_eps -> dyn_permeability = 0; then before[i] = water_depth[i].
    //
    // KEPT IN PYTHON (the runner does these, then calls step_water): the lazy
    // init (_water_depth_before seed, water.dx bind, _steam_idx resolve), the
    // dormancy early-out, and the sparse source-holds loop. The water pipe params
    // (g/damping/k_p/v_max/depth_eps/h_ref/dx) are already members on this->water
    // (set in _bind_water_params), and this->water.dx is already bound — so
    // step_water calls this->water.step(...) without re-passing them.
    //
    //   water_depth, flow_vx, flow_vy : float (h, w) — pipe-model state (mutated)
    //   floor_height                  : float (h, w) — solver floor (read)
    //   atmosphere                    : float (h, w) — bulk pressure; W3 scales it
    //   wave_p                        : float (h, w) — solver head term (read)
    //   solid                         : bool  (h, w) — static walls
    //   gas                           : float (N, h, w) — steam puff lands in slice
    //   before                        : float (h, w) — the _water_depth_before
    //                                   snapshot; READ by W3, MUTATED by the copyto
    //   dyn_permeability              : float (h, w) — W3 flooded seal (mutated)
    //   steam_idx                     : which gas plane the steam puff adds to
    //   tilt_x, tilt_y                : solver tilt (read)
    //   sim_time                      : tick length (seconds)
    //   ceiling_h..steam_yield        : the W3/W5 scalar params (Python doubles)
    void step_water(
        float* water_depth, float* flow_vx, float* flow_vy,
        const float* floor_height, float* atmosphere, const float* wave_p,
        const bool* solid,
        float* gas,
        float* before, float* dyn_permeability,
        int steam_idx, float tilt_x, float tilt_y,
        int h, int w, float sim_time,
        double ceiling_h, double flood_eps, double ratio_cap,
        double boil_rate, double boil_p_thresh, double steam_yield) const;

    // --- stamp_units: the per-tick dynamic-field rebuild --------------------
    // Moves GameMap.stamp_units' FIELD REBUILD (gamemap.py:485-589) into C++ —
    // a PURE-STRUCTURE move, behavior-identical, 0-ULP by construction (only
    // copies + a boolean compare + per-cell min/max; NO float arithmetic).
    //
    // Per tick, two phases (the exact contract, gamemap.py §a/§b):
    //   a. Reset every dynamic field to its static baseline, IN-PLACE:
    //        obstacles[i]        = (permeability[i] <= 0.0f)   // walls only
    //        dyn_permeability[i] = permeability[i]             // copy
    //        dyn_wave_absorb[i]  = wave_absorb[i]              // copy
    //        dyn_light_atten[i]  = light_atten[i]              // copy (×3 chan)
    //   b. Stamp each living unit's footprint over the flat stamp rows. Python
    //      builds one row per (living-unit, in-bounds footprint-tile) — the unit
    //      iteration + occupied_tiles() + the `u.alive` filter + the bounds
    //      check all stay Python (CPU actors); C++ just applies the combine ops:
    //        dyn_permeability[idx] = min(perm[r], permeability[idx])   // MIN
    //        dyn_wave_absorb[idx]  = max(dyn_wave_absorb[idx], wabs[r]) // MAX
    //        dyn_light_atten[idx]  = max(., atten_{r,g,b}[r]) per-channel // MAX
    //      where idx = ys[r]*w + xs[r]. The defaults (unit_permeability 0.5,
    //      unit_wave_absorb 0.5, light_atten {1,1,1}) are applied PYTHON-side per
    //      unit before flattening (matches the getattr-or-default contract).
    //
    // The atmosphere-refill bit (gamemap.py:586-588) STAYS in Python (Q1, locked):
    // it is not unit-driven and must NOT change. ALL writes here are IN-PLACE so
    // the engine's re-fetched field pointers stay valid.
    //
    //   permeability, wave_absorb : float (h, w) — static material baselines (read)
    //   light_atten               : float (h, w, 3) — static attenuation (read)
    //   dyn_permeability, dyn_wave_absorb : float (h, w) — dynamic targets (write)
    //   dyn_light_atten           : float (h, w, 3) — dynamic target (write)
    //   obstacles                 : bool (h, w) — solid mask (write, walls only)
    //   ys, xs                    : int32 (n_stamp,) — footprint tile (row, col)
    //   perm, wabsorb             : float (n_stamp,) — per-row unit values
    //   atten_r, atten_g, atten_b : float (n_stamp,) — per-row unit opacity (RGB)
    void stamp_units(
        const float* permeability, const float* wave_absorb,
        const float* light_atten,
        float* dyn_permeability, float* dyn_wave_absorb, float* dyn_light_atten,
        bool* obstacles,
        const int32_t* ys, const int32_t* xs,
        const float* perm, const float* wabsorb,
        const float* atten_r, const float* atten_g, const float* atten_b,
        int n_stamp, int h, int w) const;
};
