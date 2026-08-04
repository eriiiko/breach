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
#include "bulk_transport.h"   // EOS refactor P1: bulk O2/N2 donor-cell flux
#include "eos_solver.h"       // EOS refactor P3: the compressible Kwatra solver
#include "combustion.h"       // EOS refactor P4: combustion on real O2

class PhysicsEngine {
public:
    // AtmosphereSolver is RETAINED on the engine (its wave_substep/
    // diffuse_solve are no longer called from run_substeps — `eos` replaces
    // them, EOS refactor P3) so any still-bound Python params / the isolated
    // GPU test bindings keep resolving; the CPU/GPU wave+diffuse dispatch
    // paths it fronted are asserted unreachable in run_substeps below (D7 +
    // the P3 GPU-guard task).
    AtmosphereSolver  atmos;
    SmokeDynamics     smoke;
    FireSimulation    fire;
    TemperatureSolver temperature;
    Raycaster         raycaster;
    WaterSolver       water;
    EOSSolver         eos;   // EOS refactor P3
    CombustionSolver  combustion;   // EOS refactor P4

    // (wave_p_f_ / atm_f_ DELETED — audit Patch A / A9, 2026-08-04. Both float
    // scratch buffers were DECLARED HERE AND NEVER USED: repo-wide grep found
    // the declarations and nothing else. The paragraphs that stood here
    // described step_tail/step_water dequantizing into them, which those
    // functions have not done for some time — the members outlived the bridges
    // they were written for, and the comments outlived the members. The float
    // head/ripple bridges themselves still exist inside step_water; they simply
    // do not stage through an engine-owned buffer.)

    // EOS P3: reused scratch for the bulk-N sum (O2+N2) step_tail hands to
    // TemperatureSolver::step as the real Pass-1 heat-deposit divisor.
    mutable std::vector<int32_t> n_bulk_;

    // --- Patch 1 S4a: the per-tick orchestration TAIL --------------------
    // Moves the three trailing PURE-SOLVER-CALL steps of PhysicsRunner.step
    // (everything AFTER the IMEX substep loop) into C++: the W6a ripple, the
    // fire feedback step, and the temperature heat->conduction->cooling pass —
    // in that exact order, calling THIS engine's own solver instances. No new
    // arithmetic: it is the same three calls Python made, so it is bit-identical
    // (gated by the per-cell A/B harness). Lives in physics_engine.cpp, compiled
    // /fp:strict so the LATER glue ports (substep loop, W3/W5 water accounting)
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
    //   fire                           : int32 (h, w) Q16.16 — S3b: read+written by
    //                                    the INTEGER logistic directly (no bridge)
    //   atmosphere, smoke, wall_hp     : fire step inputs (mutated); all int32 Q16.16
    //   temperature                    : int32 (h, w) Q16.16 — fire reads, temp writes
    //   wind_x, wind_y                 : int32 (h, w) Q16.16 — shared wind (read)
    //   is_vacuum, flammable           : bool  (h, w)
    //   heat                           : int32 (h, w) Q16.16 — heat deposit (read)
    //   heat_inv_shift                 : int32 (h, w) — per-tile inverse mass shift
    //   face_shift                     : int32 (h, w, 4) — conduction face shifts
    // EOS refactor P3: `wave_p` is the repurposed P_prev (ripple splash reads
    // |P - P_prev|); `gas`/`gas_conservative`/`n_gases` are NEW — step_tail
    // sums the conservative bulk planes (O2+N2) into a reused scratch and
    // hands it to TemperatureSolver::step as the REAL N divisor for the
    // Pass-1 heat deposit (closing the P2 `// P3:` density-proxy TODO).
    // EOS refactor P4 (design §6): `o2_idx` is NEW — step_tail slices the O2
    // plane out of `gas` and hands it to FireSimulation::step as the real
    // O2-gate input (n_o2), replacing the atmosphere/P proxy.
    std::vector<std::pair<int, int>> step_tail(
        // ripple group
        float* ripple, float* ripple_v,
        const int32_t* water_depth, const int32_t* p_prev,   // S1: water_depth Q16.16
                                                             // EOS P3: p_prev (was wave_p)
        const bool* solid,
        // fire group — S3b: fire + wall_hp are Q16.16 int32; S2c: atmosphere +
        // wind are Q16.16 int32. EOS P3: atmosphere (== P) is READ-ONLY to the
        // fire now (the plume writes temperature instead).
        int32_t* fire_field, int32_t* atmosphere, int32_t* smoke_field, int32_t* wall_hp,  // S3b: fire+wall_hp Q16.16; S2b: smoke Q16.16; S2c: atm Q16.16
        const int32_t* temperature, const int32_t* wind_x, const int32_t* wind_y,      // S2c: wind Q16.16
        const bool* is_vacuum, const bool* flammable,
        // temperature group
        int32_t* temperature_mut, const int32_t* heat,
        const int32_t* heat_inv_shift, const int32_t* face_shift,
        // THERMAL-MASS AXIS (docs/thermal_mass_axis_design_2026-07-25.md): the
        // per-medium THERMAL mask (thermal_mass > 0) the temperature solver's
        // six medium tests key on, instead of the FLOW mask `solid` above.
        // GameMap.thermal_solid; equals `solid` on any furniture-free map.
        const bool* thermal_solid,
        // COOL-SHIFT AXIS (2026-07-30): the per-tile ambient-decay shift
        // (GameMap.cool_shift) the temperature pass's Pass 3 reads instead of
        // the single global COOL_SHIFT. REQUIRED here (not defaulted) for the
        // same reason `thermal_solid` is: the live engine must never silently
        // fall back to the global. Uniform == the old global on the shipped
        // config, so this is byte-identical on arrival.
        const int32_t* cool_shift_grid,
        // FUEL-FRACTION AXIS (2026-07-30): the per-tile `make_recip` reciprocal
        // of each tile's material's full-health hp (GameMap.fuel_recip), which
        // the fire logistic's fuel term F = clamp01(wall_hp/hp_full) reads
        // instead of the single global [physics.fire] fuel_ref (== WOOD's hp).
        // REQUIRED here (not defaulted) for the same reason `thermal_solid` and
        // `cool_shift_grid` are: the live engine must never silently fall back
        // to the global. A uniform plane == the old global, so this is
        // byte-identical on arrival for any map whose fuel is wood.
        const int64_t* fuel_recip,
        // PER-MATERIAL EXTINCTION TEMPERATURE (P-R3, 2026-07-31 — docs/
        // radiation_raycaster_extinction_ruling_2026-07-31.md A3 ride-along):
        // per tile, that material's `ignition_temp - ignition_to_ext_delta`
        // quantized to Q16.16 (GameMap.fire_T_ext_plane), which the fire
        // logistic's `hot` gate reads instead of the single global
        // [physics.fire] fire_T_ext. REQUIRED here (not defaulted) for the same
        // reason `fuel_recip` is: the live engine must never silently fall back
        // to a global that, at its shipped 350, sits ABOVE both shipped
        // ignition temps — a tile could ignite below its own sustain floor. A
        // uniform plane == the old global, so this is byte-identical on arrival.
        const int32_t* fire_T_ext_plane,
        // EOS P3: bulk-N source for the Pass-1 heat-deposit divisor.
        // EOS P4: o2_idx slices the real O2 gate input out of `gas`.
        const int32_t* gas, const bool* gas_conservative, int n_gases, int o2_idx,
        int h, int w, float sim_time,
        // BC: ambient ring mask forwarded to TemperatureSolver::step's Pass-0
        // wipe (nullptr on space maps = byte-identical).
        const bool* is_ambient = nullptr,
        // P-R4 (docs/radiation_raycaster_extinction_ruling_2026-07-31.md A1):
        // the SIGNED radiation accumulator the fire-plane cast filled at the top
        // of the tick. step_tail hands it straight through to the temperature
        // pass, which folds it (BEFORE the heat deposit) through each tile's own
        // heat_inv_shift. nullptr -> no fold, byte-identical to pre-P-R4.
        const int32_t* rad_net = nullptr) const;

    // --- Patch 1 S4b: the IMEX atmosphere/smoke substep loop -------------
    // Moves the per-tick IMEX substep block out of PhysicsRunner.step (Python)
    // into C++ — the loop that runs BETWEEN the water/fire-heat steps (still
    // Python, before) and step_tail (already C++, after). It advances the
    // atmosphere wave+diffusion and the per-gas smoke transport `n` times, where
    // `n` is derived from the atmosphere solver's CFL bound and `sim_time`.
    //
    // BIT-IDENTITY is the whole point — this reproduces Python's arithmetic
    // EXACTLY (the /fp:strict TU makes the FP strict-IEEE; we must match the
    // PRECISION + ORDER numpy's pybind boundary produced):
    //   * `n` (= n_wave) is an INTEGER CLIFF (Bedrock cliff-patch): n = max(1,
    //     ceil_div(quantize(sim_time), atmos.max_dt_q())) — a pure INTEGER ceil-
    //     divide against a Q16.16 CFL constant. Was n = max(1,(int)ceil(sim_time/
    //     (double)atmos.max_dt())); the double ceil was already correctly-rounded
    //     (cross-platform deterministic) but the integer form removes the last
    //     double from the substep-count path so a CUDA kernel matches the CPU
    //     exactly. n_smoke is likewise integer via fixedpoint::smoke_cliff_count.
    //   * `dt_actual` and `dt_smoke` stay DOUBLE until the solver-call boundary:
    //     dt_actual = (double)sim_time / n; dt_smoke = (double)sim_time / n_smoke.
    //     They are cast to float ONLY when passed to the solvers — matching
    //     pybind's double->float32 cast at the .step() call site (do NOT pre-
    //     narrow; the order double-divide-then-cast must match). The COUNTS are
    //     integer-derived; the per-substep REAL dt that drives the physics is the
    //     same sim_time/count length as before.
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
    //
    // --- EOS refactor P1 (docs/eos_refactor_design.md §2.2) --------------
    // `gas_conservative` (N,) flags the BULK species (O2 / inert_N2,
    // simulation/gases.py) — the two planes that move by donor-cell
    // conservative flux (bulk_transport.cpp) instead of the semi-Lagrangian
    // per-gas loop below. run_substeps calls bulk_flux_transport ONCE per
    // tick, immediately after diffuse_solve computes the fresh wind (step 2)
    // and BEFORE the smoke SL loop (step 3) — riding the SAME once-computed
    // wind, purely additive (no solver change). The existing per-gas SL loop
    // (smoke.step / sink_hop, steps 3-4) SKIPS any plane flagged conservative,
    // so the two transport schemes never both touch the same plane; every
    // legacy (non-bulk) plane's SL transport is untouched (conservative[gi]
    // is false there), so this is 0-ULP for the 5 legacy species.
    // --- EOS refactor P3 (docs/eos_refactor_design.md §3, §8 patch P3) ---
    // The Kwatra solver (`this->eos`) REPLACES AtmosphereSolver::wave_substep
    // + ::diffuse_solve, and its own advection substep loop REPLACES the old
    // n_smoke-substepped semi-Lagrangian loop for the two CONSERVATIVE gas
    // planes (bulk O2/N2 now move ONCE PER EOS SUBSTEP, inside eos.step, via
    // bulk_flux_transport — not once per tick as P1 shipped it). The 5 TRACE
    // planes still ride the per-gas SmokeDynamics::step, but now ONCE per
    // tick (design §3.2 step 4b: "traces advect ONCE per tick on the final
    // velocity") on the solver's post-correction `wind_x`/`wind_y` — the
    // n_smoke CFL-floor substep loop AND the decoupled sink_hop BFS loop are
    // BOTH DELETED (sink_hop + its BFS machinery, decisions.md #3; native
    // venting replaces it). `wave_p` is REPURPOSED as `P_prev` (the design's
    // own "keep the old name, change the meaning" pattern, already applied
    // to `atmosphere`->P — see eos_solver.h); `wave_v`/`wave_source` are
    // RETIRED (no longer read/written here — see gamemap.py for the arrays'
    // fate). `temperature` is a NEW required arg (T, ambient-relative Kelvin).
    //
    //   p_prev              : Q16.16 (h,w) — the repurposed `wave_p` buffer.
    //   atmosphere           : Q16.16 (h,w) — P (read prior tick's value
    //                          implicitly via p_prev; WRITTEN once, step 5).
    //   wind_x/wind_y        : Q16.16 (h,w) — u (self-advected + corrected).
    //   temperature           : Q16.16 (h,w) — T (advected + compression-worked).
    //   gas                   : Q16.16 (n_gases,h,w) — the two conservative
    //                          planes are donor-cell transported EVERY
    //                          eos substep; traces advect once, below.
    //
    // EOS refactor P4 (design §2.2/§5 v2.1, decisions log #12): `gas_decay`
    // (n_gases,) is NEW — the per-gas trace `decay` column (simulation/
    // gases.py, "loaded but never applied" until now), applied ONCE per tick
    // right after each trace plane's own once-per-tick advection below, with
    // the decayed mass credited to `inert_n2_idx`'s plane IN THE SAME CELL
    // ("decay is settling/oxidation into inert bulk, not deletion" — closes
    // the v2.1 residual of decision #12: N_total conserved through the FULL
    // burn-then-decay cycle, not just the burn). `inert_n2_idx` names which
    // gas plane receives the credited mass (0 for the two conservative bulk
    // planes themselves — they carry decay=0 by config contract, gases.py).
    void run_substeps(
        int32_t* p_prev,                                          // was wave_p
        int32_t* atmosphere,                                     // S2c: Q16.16
        int32_t* wind_x, int32_t* wind_y,                        // S2c: Q16.16
        int32_t* temperature,                                    // EOS P3
        const bool* obstacles, const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability, const float* dyn_wave_absorb,
        int32_t* gas, const float* gas_diffusion, int n_gases,   // S2b: gas Q16.16
        const bool* gas_conservative,                             // EOS P1
        const float* gas_decay, int inert_n2_idx,                 // EOS P4
        int h, int w, float sim_time,
        // BC (boundary_conditions_spec_2026-07-19): planetside AMBIENT ring —
        // forwarded to eos.step (nullptr/0 on space maps = byte-identical).
        const bool* is_ambient = nullptr,
        const int32_t* n_amb = nullptr,
        int32_t p_amb = 0,
        const int32_t* sponge_sigma = nullptr,
        const int32_t* sponge_udamp = nullptr,
        // S8a Path B: when false, the EOS step runs but the once-per-tick TRACE
        // smoke loop (+ decay) is SKIPPED — the resident path runs those traces
        // itself on device (trace_smoke_resident) so the 5 per-plane per-call
        // transfers are gone. Default true == the exact prior behaviour.
        bool do_traces = true,
        // THERMAL-MASS AXIS, P-EOS (docs/thermal_mass_eos_ruling_2026-07-30.md
        // §4 item 1): the per-medium THERMAL mask (GameMap.thermal_solid),
        // forwarded verbatim to eos.step / eos_step_cuda. It governs ONLY the
        // solver's two `temperature[]` writes and its T backtrace; `cmask`,
        // hence pressure/velocity/gas flow, is untouched. nullptr on the legacy
        // path -> byte-identical to before this patch.
        const bool* thermal_solid = nullptr);

    // --- S8a Path A: the fully device-resident EOS stage -----------------
    // (docs/cuda_s8a_path_a_impl_2026-07-21.md §3.1.) The resident sibling of
    // run_substeps' EOS dispatch: host mirrors feed the shared pre-stage (all
    // reductions — tick-entry state) + telemetry; the device pointers are the
    // persistent CuPy resident fields (uintptr_t so this header stays
    // CUDA-free; 0 == nullptr for the ambient statics). NO trace loop (the
    // runner drives trace_smoke_resident, as in Path B). Declared on every
    // build; the body THROWS on a non-CUDA build, and on a CUDA build throws
    // unless eos_step_backend_is_cuda() (no CPU fallback for device
    // pointers). Bit-identity gate: tests/cuda_s8a_check.py PART 1a/1b/1c.
    void run_substeps_resident(
        int32_t* p_prev,
        const int32_t* atmosphere,
        const int32_t* wind_x, const int32_t* wind_y,
        const int32_t* temperature,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability, const float* dyn_wave_absorb,
        const int32_t* gas, int n_gases, const bool* gas_conservative,
        int h, int w, float sim_time,
        const bool* is_ambient, const int32_t* n_amb, int32_t p_amb,
        std::uintptr_t d_atmosphere, std::uintptr_t d_wave_p,
        std::uintptr_t d_wind_x, std::uintptr_t d_wind_y,
        std::uintptr_t d_temperature, std::uintptr_t d_gas_base,
        std::uintptr_t d_solid, std::uintptr_t d_is_vacuum,
        std::uintptr_t d_dyn_permeability,
        std::uintptr_t d_is_ambient,
        std::uintptr_t d_sponge_sigma, std::uintptr_t d_sponge_udamp,
        // THERMAL-MASS AXIS, P-EOS: the mask on the MIRROR (for the shared host
        // occlusion predicate — all pre-stage reductions read the mirror) plus
        // its DEVICE copy (what the SL/compression kernels read). 0/nullptr ->
        // the legacy path. The device copy MUST ride the per-tick from_host
        // upload: unlike the sponge grids this mask is not static.
        const bool* thermal_solid = nullptr,
        std::uintptr_t d_thermal_solid = 0);

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
    // scalar cast to float at numpy's cast point; /fp:strict makes the f32 ops
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
    // S1: water_depth/flow_vx/flow_vy/floor_height/before are now int32 Q16.16
    // (metres / m/s). atmosphere/gas/dyn_permeability stay FLOAT (the S2 group) —
    // the W5 boil + W3 displacement are FLOAT BRIDGES that dequantize water_depth
    // at the boundary (marked in the .cpp). The substep-count cliff is integer.
    // EOS refactor P3: `wave_p` param retired (the water head reads the
    // integer `atmosphere` == P directly, no float bridge); `n_gases` added
    // (the W3 occupancy-transition evacuation loop touches every gas plane,
    // not just the W5 steam slice).
    void step_water(
        int32_t* water_depth, int32_t* flow_vx, int32_t* flow_vy,
        const int32_t* floor_height, int32_t* atmosphere,   // S2c: atm Q16.16 == P
        const bool* solid,
        int32_t* gas, int n_gases,   // S2b: gas Q16.16 (W5 steam puff + W3 evacuation)
        int32_t* before, float* dyn_permeability,
        int steam_idx, float tilt_x, float tilt_y,
        int h, int w, float sim_time,
        double ceiling_h, double flood_eps, double ratio_cap,
        double boil_rate, double boil_p_thresh, double steam_yield) const;

    // --- S8a Path B: the water HOST TAIL, split out of step_water -----------
    // The W5 flash-boil vacuum sink + the W3 volume-displacement evacuation +
    // the final copyto(before, water_depth) — EVERYTHING in step_water AFTER the
    // substep loop. Factored so the resident path can run the substep loop on
    // device (water_substeps_resident) and then this host tail on the mirror,
    // byte-for-byte identical to the monolithic step_water (which now calls this
    // helper). No substep loop, no solver call — pure host float/integer arithmetic
    // (/fp:strict), so it is bit-identical whether reached from step_water or the
    // resident path.
    void step_water_tail(
        int32_t* water_depth, int32_t* atmosphere, const bool* solid,
        int32_t* gas, int n_gases, int32_t* before, float* dyn_permeability,
        int steam_idx, int h, int w, float sim_time,
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
