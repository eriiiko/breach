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
#include "emissive_table.h"   // ray-engine-v2: THE E° table, one owner
#include "radiation_sweep.h"  // ray-engine-v2: the radiation sweep (LIVE since T5b)

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
    // Ray-engine-v2 (design v3 §2.6, §11): THE E° table's one owner (the old
    // `raycaster` above keeps a vestigial copy of the SAME bake since T6 —
    // never two bakes), and the radiation sweep that runs as step 2b of
    // step_tail, LIVE since T5b.
    EmissiveTable     emissive;
    RadiationSweep    radiation;
    // Ray-engine-v2 P6a: THE LIGHT EMISSION TABLE L° (checked in, emissive_
    // table.h -- one table, one currency, no dial) and THE REQUEST. Light is
    // REQUESTED, never always on (design v3 §7.1: "headless training simply
    // does not request the light channels"): step_tail runs the sweep's light
    // group only while `light_requested`, and only then touches the light
    // planes. Default OFF, so the live game and every golden are unmoved until
    // P6b's renderer asks for it.
    LightEmissionTable light_emission;
    bool               light_requested = false;

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

    // ---- ray-engine-v2 P5b (design v3 §2.8 / §6.3): THE GAS CURRENCY ------
    // The three integers the temperature fold's Pass 1 divides a gas deposit
    // by, derived from the fold's OWN two dials (`temperature.c_v`,
    // `temperature.n_floor_heat`) by the SAME kit calls TemperatureSolver::
    // step() makes at its top and in Pass 1:
    //     n_floor_q = quantize(n_floor_heat)                 the N floor
    //     c_v_q     = quantize(c_v > 0 ? c_v : 1)           c_v's ONE integer form
    //     recip_cv  = make_recip(c_v_q / 65536)             its exact inverse, Q.32
    // step_tail hands n_floor_q / recip_cv to the radiation sweep (both
    // backends), whose Fleck pre-pass prices every absorbing gas cell's L in
    // them — so the gas arm and the fold cannot disagree about what one heat
    // count is worth in a gas cell. Here rather than in temperature_solver.cpp
    // because the fold is not this patch's to touch (P5c owns its gas branch);
    // tests/test_radiation_sweep_gas_fleck.py holds these three to the fold's
    // own gas deposit, bit for bit, at the shipped dials and at another pair.
    // Out of line in this /fp:strict TU: quantize and make_recip are real
    // arithmetic, and a header-inline copy would compile under bindings.cpp's
    // /fp:fast.
    struct GasCapacityQ {
        int32_t n_floor_q;
        int32_t c_v_q;
        int64_t recip_cv;
    };
    GasCapacityQ gas_capacity_q() const;

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
        // T5b step 7 / thermal model v2 R1: `const int32_t* cool_shift_grid`
        // was a REQUIRED parameter here. Pass 3 is deleted on both backends,
        // so the plane has no reader and the parameter is gone. Removing a
        // required positional argument is a hard compile error at every call
        // site, which is the point: nothing can silently keep passing it.
        // FUEL-FRACTION AXIS (2026-07-30): the per-tile `make_recip` reciprocal
        // of each tile's material's full-health hp (GameMap.fuel_recip), which
        // the fire logistic's fuel term F = clamp01(wall_hp/hp_full) reads
        // instead of the single global [physics.fire] fuel_ref (== WOOD's hp).
        // REQUIRED here (not defaulted) for the same reason `thermal_solid`
        // is: the live engine must never silently fall back
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
        const int64_t* rad_net = nullptr,   // P3a-1: int64 (design v3 row 26/35)
        // arc #54 P-G1b (design §2.7 row 3): the CONSERVED gas energy field,
        // handed straight through to TemperatureSolver::step so its Pass-1
        // deposit and Pass-2 conduction land in the books instead of in the
        // mirror. nullptr keeps the whole tail bit-identical to pre-#54.
        // `t_amb_q` is T_AMB_K in raw Q16.16 counts — the SAME fold EOSSolver
        // does, passed in rather than re-derived so the two cannot drift.
        int64_t* gas_energy = nullptr,
        int32_t t_amb_q = 0,
        // ---- ray-engine-v2 (design v3 §11, orchestrator override row 35):
        // THE SWEEP, LIVE since T5b. Step "2b" — between the fire step and
        // the temperature pass — runs RadiationSweep::run (shear, S16) on the
        // live temperature / extinction / capacity planes into the four
        // int64 planes below, which the temperature fold, the Pass-1 clamp
        // and unit heat damage now read (the old cast that used to fill a
        // separate set of planes is deleted, T6). With the radiation backend
        // on (P4, set_radiation_backend) the same step runs the sweep's CUDA
        // twin instead, bit-identical (cuda_radiation_sweep.h).
        //   heat_atten_q / dyn_heat_atten_q : int32 Q16 (h, w), the extinction
        //                                     planes (GameMap, optics_fixed.py)
        //   rad_net_sweep / rad_flux_sweep / rad_amb_sweep / rad_fluence :
        //                                     int64 (h, w), accumulated (the
        //                                     sweep's own run() zeroes them
        //                                     each tick before its first
        //                                     ordinate — not the conductor)
        //   k_leak_q                        : [physics.radiation] k_leak, Q16
        //   rad_amb_vacuum_q                : the EMISSIVE LEVEL a VACUUM cell
        //        radiates against (thermal model v2 R3), in the E° table's own
        //        units. Every other cell — interior air, solids, and the
        //        ambient ring, which IS room-temperature air — takes E°[0]; the
        //        select happens in RadiationSweep::derive_ambient from the
        //        `is_vacuum` plane already in scope, so nothing is authored and
        //        no material column or level-data field is added.
        //        NEGATIVE means E°[0], i.e. space at room temperature — thermal
        //        v2 R4, the shipped v1 answer, which makes the plane UNIFORM
        //        and the sweep bit-identical to the scalar era. Bound by
        //        PhysicsRunner from [physics.radiation] vacuum_ambient_K.
        // ALL SIX PLANES must be non-null for the sweep to run (dormancy by
        // branch, the tree's idiom); the pybind binding makes them REQUIRED
        // and noconvert, so the live runner cannot omit them. The temperature
        // pass always receives rad_fluence / e_table too, so the maximum-
        // principle clamp is LIVE on the same call (design v3 P3) — only a
        // direct caller that omits them gets the pre-flip no-clamp path.
        const int32_t* heat_atten_q = nullptr,
        const int32_t* dyn_heat_atten_q = nullptr,
        int64_t* rad_net_sweep = nullptr,
        int64_t* rad_flux_sweep = nullptr,
        int64_t* rad_amb_sweep = nullptr,
        int64_t* rad_fluence = nullptr,
        int32_t k_leak_q = 0,
        int64_t rad_amb_vacuum_q = -1,
        // ray-engine-v2 P5a (design v3 §6.3): SMOKE ABSORBS HEAT. The per-gas
        // [gases.*] heat_absorb column in Q16 (GasTable.heat_absorb_q16,
        // n_gases entries). With it the sweep (both backends) reads the smoke
        // term on every gas cell from `gas` above and the bulk sum `n_bulk_`
        // this function already builds — the SAME N the fold divides by and
        // the N_EPS floor keys on. The pybind binding makes it REQUIRED
        // (noconvert); null here is the pre-P5a sweep. Every shipped value is
        // 0.0, so no gas plane is read and the live game does not move.
        const int32_t* gas_heat_absorb_q16 = nullptr,
        // ---- ray-engine-v2 P6a: THE LIGHT CHANNELS (radiation_sweep.h's
        // LightChannels), read only while `light_requested`. Then all five
        // planes are REQUIRED (std::invalid_argument otherwise -- a request
        // that silently computes nothing is the failure this makes loud):
        //   light_atten_q / dyn_light_atten_q : int32 Q16 (h, w, 3), the
        //        material / stamped light extinction (GameMap, optics_fixed)
        //   light_q / light_flux_q / light_glow : int64 (h, w, 3) / (h, w, 2) /
        //        (h, w, 3), OVERWRITTEN by the sweep (it zeroes them itself)
        //   gas_light_absorb_q16 / gas_light_glow_q16 : int32 (n_gases, 3), the
        //        smoke term's columns (GasTable), both or neither -- they read
        //        the gas group above (gas_heat_absorb_q16 given)
        // With the request off every one of them is ignored and the light
        // planes keep whatever they held.
        const int32_t* light_atten_q = nullptr,
        const int32_t* dyn_light_atten_q = nullptr,
        int64_t* light_q = nullptr,
        int64_t* light_flux_q = nullptr,
        int64_t* light_glow = nullptr,
        const int32_t* gas_light_absorb_q16 = nullptr,
        const int32_t* gas_light_glow_q16 = nullptr) const;

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
        int64_t* gas_energy,                                     // arc #54 §2.2
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
        std::uintptr_t d_thermal_solid = 0,
        // arc #54 §2.2 (P-G2): the conserved gas energy field's PERSISTENT
        // device buffer (GameMap's resident `gas_energy`, (h,w) int64). No
        // host-mirror parameter is needed here — nothing in the resident
        // pre-stage reduces over it (unlike thermal_solid's cap2 fold); the
        // caller's from_host/to_host round-trips the mirror around this call.
        std::uintptr_t d_gas_energy = 0);

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
        double boil_rate, double boil_p_thresh, double steam_yield,
        // arc #54 §2.7: forwarded VERBATIM to step_water_tail (the W3
        // evacuation row). All default nullptr -> the pre-#54 path.
        int64_t* gas_energy = nullptr,
        const bool* gas_conservative = nullptr,
        const bool* thermal_solid = nullptr,
        const bool* is_vacuum = nullptr,
        const bool* is_ambient = nullptr,
        int32_t t_amb_raw = 0) const;

    // --- S8a Path B: the water HOST TAIL, split out of step_water -----------
    // The W5 flash-boil vacuum sink + the W3 volume-displacement evacuation +
    // the final copyto(before, water_depth) — EVERYTHING in step_water AFTER the
    // substep loop. Factored so the resident path can run the substep loop on
    // device (water_substeps_resident) and then this host tail on the mirror,
    // byte-for-byte identical to the monolithic step_water (which now calls this
    // helper). No substep loop, no solver call — pure host float/integer arithmetic
    // (/fp:strict), so it is bit-identical whether reached from step_water or the
    // resident path.
    //
    // arc #54 (gas-energy conservation, design §2.7 "water-displacement gas
    // evacuation" — the row v1/v2 MISSED): the W3 evacuation moves N out of a
    // flooding cell into neighbours selected by `!solid && perm > 0` ONLY,
    // i.e. also into VACUUM, the ambient RING and THERMAL_SOLID tiles — and it
    // did so with no temperature/energy argument at all, so under a stored
    // energy field it would have been a silent mass-without-energy mint at
    // every flood. It now takes the field and the four masks:
    //   * only the BULK (gas_conservative) planes carry energy — a trace share
    //     is mass the books never counted;
    //   * an ACCOUNTABLE receiver is credited at the DONOR's T_abs (moved
    //     mass carries its source's temperature — the seam's first rule);
    //   * a share into a NON-accountable cell leaves the books and is booked
    //     to `e_water_evac_export_sum` (R3-#10).
    // Host-side on BOTH backends (confirmed: there is no .cu twin), and it
    // runs BEFORE the EOS — so at P-G1a the entry re-sync would absorb it
    // anyway; wiring it here is what makes P-G1b's D1 flip a deletion rather
    // than a hunt.
    void step_water_tail(
        int32_t* water_depth, int32_t* atmosphere, const bool* solid,
        int32_t* gas, int n_gases, int32_t* before, float* dyn_permeability,
        int steam_idx, int h, int w, float sim_time,
        double ceiling_h, double flood_eps, double ratio_cap,
        double boil_rate, double boil_p_thresh, double steam_yield,
        // arc #54 §2.7 — ALL default nullptr so a caller without the field
        // takes the pre-#54 path byte-for-byte (dormancy BY BRANCH).
        int64_t* gas_energy = nullptr,
        const bool* gas_conservative = nullptr,
        const bool* thermal_solid = nullptr,
        const bool* is_vacuum = nullptr,
        const bool* is_ambient = nullptr,
        int32_t t_amb_raw = 0) const;

    // arc #54 §2.7: energy that left the accountable set through the W3
    // water-displacement evacuation (shares into vacuum / ring / thermal
    // solids). PER-CALL reset, the EOSSolver counter idiom; int64, in the
    // gas_energy Q32 currency.
    mutable int64_t e_water_evac_export_sum = 0;

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
    // Ray-engine-v2 P1 (design v3 §3, §6.2): the FOURTH dynamic output, the
    // integer heat-extinction plane the radiation sweep reads —
    //        dyn_heat_atten_q[i]   = heat_atten_q[i]                  // copy (Q16)
    //        dyn_heat_atten_q[idx] = max(., heat_q[r])                // MAX
    // every dynamic stamp is a MAX, never a sum (a <= d <= ONE, §2.3):
    //   heat_atten_q              : int32 Q16 (h, w) — static material extinction (read)
    //   dyn_heat_atten_q          : int32 Q16 (h, w) — dynamic target (write)
    //   heat_q                    : int32 Q16 (n_stamp,) — per-row unit extinction
    //                               (unit.heat_atten, default 1.0 = an opaque body,
    //                               quantized Python-side by optics_fixed)
    // Ray-engine-v2 P6a (design v3 §4.1, §5): the FIFTH dynamic output, the
    // INTEGER light-extinction twin the sweep's light channels read, per
    // channel exactly as the heat plane:
    //        dyn_light_atten_q[i]   = light_atten_q[i]                (copy, x3)
    //        dyn_light_atten_q[idx] = max(., light_q_rows[r][c])      (MAX, x3)
    //   light_atten_q             : int32 Q16 (h, w, 3) — static material extinction (read)
    //   dyn_light_atten_q         : int32 Q16 (h, w, 3) — dynamic target (write)
    //   light_q_rows              : int32 Q16 (n_stamp, 3) — per-row unit light
    //                               extinction (the SAME unit.light_atten triple the
    //                               float stamp reads, default {1,1,1}, quantized
    //                               Python-side by optics_fixed)
    // The float dyn_light_atten above is the old render march's, deleted at P6c.
    void stamp_units(
        const float* permeability, const float* wave_absorb,
        const float* light_atten,
        float* dyn_permeability, float* dyn_wave_absorb, float* dyn_light_atten,
        bool* obstacles,
        const int32_t* ys, const int32_t* xs,
        const float* perm, const float* wabsorb,
        const float* atten_r, const float* atten_g, const float* atten_b,
        const int32_t* heat_atten_q, int32_t* dyn_heat_atten_q,
        const int32_t* heat_q,
        int n_stamp, int h, int w,
        const int32_t* light_atten_q = nullptr, int32_t* dyn_light_atten_q = nullptr,
        const int32_t* light_q_rows = nullptr) const;
};
