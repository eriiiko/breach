// PhysicsEngine — per-tick orchestration moved out of Python (Patch 1 S4).
//
// This translation unit is compiled /fp:strict (the global build is /fp:fast;
// see cpp/CMakeLists.txt set_source_files_properties for this file). S4a moves
// only the per-tick TAIL — three pure solver calls — so /fp:strict is
// 0-ULP-trivial here (no new arithmetic lives in this file yet). The strict
// rounding matters for the LATER glue ports (the IMEX substep loop and the
// W3/W5 water accounting), which carry real float math that must match numpy's
// strict-IEEE rounding bit-for-bit; setting the TU up now means those ports
// inherit it without a second CMake change.

#include "physics_engine.h"
#include "fixed_point.h"   // S1: Q16.16 toolkit (quantize, ceil_div, max_dt_q)
#ifdef BREACH_HAS_CUDA
#include "cuda_temperature.h"   // CUDA-S1: GPU temperature solver + backend flag
#include "cuda_water.h"         // CUDA-S3: GPU water solver + backend flag
#include "cuda_smoke.h"         // CUDA-S4a: GPU smoke solver + backend flag
#include "cuda_fire.h"          // CUDA-S6: GPU fire solver + backend flag
#include "cuda_eos_step.h"      // EOS P6.5: chained eos.step GPU dispatch
#include "cuda_eos_resident.h"  // S8a Path A: fully device-resident EOS tick
// CUDA-S5 cuda_wave.h / CUDA-S7 cuda_atmosphere.h RETIRED in EOS P6.0 (their
// CPU solvers were replaced by the EOS solve in P3; nothing here called them).
#endif

#include <algorithm>   // std::max, std::min
#include <cstddef>     // std::size_t
#include <cassert>     // EOS P3 GPU-backend-retirement guards
#include <stdexcept>   // S8a Path A: run_substeps_resident's loud-fail throws

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
        const int32_t* water_depth, const int32_t* p_prev,   // S1: water_depth Q16.16
                                                             // EOS P3: p_prev (was wave_p)
        const bool* solid,
        // fire group — S3b: fire + wall_hp are Q16.16 int32 too; S2c: atmosphere +
        // wind are Q16.16 int32. Fire now reads ALL of these as INTEGER directly
        // (the fire-field + atm/wind float bridges are GONE — S3b makes the logistic
        // integer end-to-end). S3c: the temperature pass reads the int32 atmosphere
        // directly too (its threshold is a Q16.16 compare) — the LAST float bridge
        // in the fire/temperature path is GONE. No atm_f_ scratch in step_tail.
        int32_t* fire_field, int32_t* atmosphere, int32_t* smoke_field, int32_t* wall_hp,  // S3b: fire+wall_hp Q16.16; S2b: smoke Q16.16; S2c: atm Q16.16
        const int32_t* temperature, const int32_t* wind_x, const int32_t* wind_y,      // S2c: wind Q16.16
        const bool* is_vacuum, const bool* flammable,
        // temperature group
        int32_t* temperature_mut, const int32_t* heat,
        const int32_t* heat_inv_shift, const int32_t* face_shift,
        // THERMAL-MASS AXIS: the per-medium THERMAL mask for the temperature
        // pass (see physics_engine.h). NOT used by the ripple or fire calls —
        // those keep `solid` (flow/LoS/obstacle) unchanged.
        const bool* thermal_solid,
        // COOL-SHIFT AXIS: per-tile ambient-decay shift for the temperature
        // pass's Pass 3 (see physics_engine.h). Temperature-only.
        const int32_t* cool_shift_grid,
        // FUEL-FRACTION AXIS: per-tile 1/hp for the FIRE pass's fuel term
        // (see physics_engine.h). Fire-only — the temperature pass never
        // reads it.
        const int64_t* fuel_recip,
        // PER-MATERIAL T_ext: per-tile Q16.16 extinction temperature for the
        // FIRE pass's `hot` gate (see physics_engine.h). Fire-only.
        const int32_t* fire_T_ext_plane,
        // EOS P3: bulk-N source (real Pass-1 heat-deposit divisor)
        // EOS P4: o2_idx slices the real O2 gate input out of `gas`
        const int32_t* gas, const bool* gas_conservative, int n_gases, int o2_idx,
        int h, int w, float sim_time,
        const bool* is_ambient,           // BC: ambient ring for the T pre-pass
        const int32_t* rad_net) const {   // P-R4: SIGNED radiation accumulator

    using namespace fixedpoint;

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
        if (water_depth[i] != 0) { water_any = true; break; }   // S1: Q16.16 int
    }
    if (!water_any) {
        for (int i = 0; i < n; ++i) {
            if (ripple[i] != 0.0f) { ripple_any = true; break; }
        }
    }
    if (water_any || ripple_any) {
        // step_ripple writes only ripple / ripple_v; water_depth / atmosphere /
        // p_prev / solid are read-only (the locked canon rule). EOS refactor
        // P3 (design §6 "ripple splash"): the splash source is the per-tick
        // pressure TRANSIENT |P - P_prev|, not the retired wave_p anomaly —
        // pass the integer atmosphere (P) and p_prev (P_prev) directly, no
        // float bridge (water_solver.cpp dequantizes |.| internally).
        this->water.step_ripple(ripple, ripple_v, water_depth, atmosphere, p_prev,
                                solid, h, w, sim_time);
    }

    // --- 2. Fire feedback step (PhysicsRunner: self.fire.step) ------------
    // Arg order cross-checked against bindings.cpp FireSimulation.step and the
    // Python call site:
    //   fire, atmosphere, smoke, wall_hp, temperature(const), wind_x, wind_y,
    //   is_wall(=solid), is_vacuum, flammable, dt.
    // `is_wall` IS `gmap.solid` (the Python passes gmap.solid as the is_wall
    // arg); `temperature` here is the const view (the previous-tick conduction
    // field the fire reads).
    //
    // S3b: the fire logistic is now INTEGER end-to-end. Fire reads fire/atmosphere/
    // wall_hp/wind directly as int32 Q16.16 and writes the int32 atmosphere plume in
    // place — the S3a fire-field bridge and the S2c atm/wind float bridges that fed
    // the fire are GONE. S3c: the LAST float bridge in step_tail (the temperature
    // pass's float-atmosphere read) is now GONE too — temperature reads the int32
    // atmosphere directly. Faithful ORDER preserved: the fire runs FIRST (its plume
    // mutates the int32 atmosphere in place), then the temperature pass reads that
    // SAME post-plume atmosphere — now integer-sourced with NO dequantize scratch.
    // CUDA-S6: dispatch to the GPU fire solver when the backend is switched on
    // (bit-identical to the CPU path — same integer ops; tol 0 on fire/atmosphere/
    // smoke/wall_hp, set-equal destroyed). The GPU entry RETURNS the destroyed
    // vector (unlike S1-S5, which only mutate fields), so this is an assigning
    // dispatch. The CPU solver stays the live fallback; with the flag off (default)
    // this is the exact prior call. On a CPU-only build (no BREACH_HAS_CUDA) only
    // the CPU path compiles. The GPU free function takes the FireParams dials
    // explicitly (it is not a method on the solver).
    // EOS refactor P4 (design §6): slice the real O2 plane out of `gas` —
    // FireSimulation's O2 gate now reads local N_O2, not the atmosphere/P
    // proxy (item 3, decisions log). `atmosphere` still feeds the plume's
    // own-tile saturation gate unchanged.
    const int32_t* n_o2 = gas + (size_t)o2_idx * (size_t)(h * w);

    // Continuous-O2 law (docs/continuous_o2_law_design_2026-07-24.md): the fire
    // logistic + combustion read the local O2 MOLE FRACTION X = Σn_o2/Σn_total.
    // n_bulk_ = Σ conservative bulk planes (O2+N2) is the fraction DENOMINATOR —
    // the SAME real N_total the temperature Pass-1 deposit uses below, so it is
    // built ONCE here (before the fire step) and shared by BOTH consumers (one
    // source of truth). The fire step does not mutate `gas`, so the value the
    // temperature pass reads afterward is identical. Both the normal and the
    // GPU-resident tick funnel through this step_tail, so the resident seam gets
    // the identical N_total with no separate plumbing.
    if (n_bulk_.size() != (size_t)n) n_bulk_.assign(n, 0);
    std::fill(n_bulk_.begin(), n_bulk_.end(), 0);
    for (int gi = 0; gi < n_gases; ++gi) {
        if (!gas_conservative[gi]) continue;
        const int32_t* plane = gas + (size_t)gi * n;
        for (int i = 0; i < n; ++i) n_bulk_[i] += plane[i];
    }

    std::vector<std::pair<int, int>> destroyed;
#ifdef BREACH_HAS_CUDA
    if (breach_cuda::fire_backend_is_cuda()) {
        // EOS P6.8: the re-derived GPU fire kernel (cuda_fire.cu) — O2 gate on
        // the real `n_o2` plane, bit-identical to the CPU FireSimulation::step
        // (tol 0 on fire/smoke/wall_hp/temperature; set-equal destroyed).
        // `atmosphere` is passed for signature parity but is vestigial
        // (unread); `temperature_mut` is passed through READ ONLY as of P-R2
        // (the plume->T shim that used to write it here is deleted — docs/
        // radiation_raycaster_extinction_ruling_2026-07-31.md A2); the
        // FireParams dials are passed explicitly since fire_step is a free
        // function. With the flag off (default) the CPU branch below is the
        // exact prior call.
        destroyed = breach_cuda::fire_step(
            fire_field, atmosphere, n_o2, n_bulk_.data(), smoke_field, wall_hp,
            temperature_mut, wind_x, wind_y,
            solid, is_vacuum, flammable,
            h, w, sim_time,
            this->fire.params.k_grow, this->fire.params.k_die,
            this->fire.params.fire_T_ext, this->fire.params.fire_T_span,
            this->fire.params.fuel_ref,
            this->fire.params.o2_frac_ext, this->fire.params.o2_frac_full,
            this->fire.params.I_min,
            this->fire.params.k_wind_fan, this->fire.params.k_wind_strip,
            this->fire.params.smoke_emission, this->fire.params.wall_damage,
            this->fire.params.temp_scale,
            // CAPACITY LAW (P-R3): `c`, the size dial. The host precompute
            // bakes INV_C = quantize(1/c) exactly as the CPU load-time block.
            this->fire.params.I_cap_per_avail,
            // FUEL-FRACTION AXIS: the per-tile 1/hp plane. The GPU kernel takes
            // it as an extra read-only plane and falls back to the fuel_ref
            // scalar above on nullptr, exactly like the CPU branch — the two
            // must stay bit-identical (tol 0).
            fuel_recip,
            // PER-MATERIAL T_ext (P-R3 ride-along): the same nullable-plane
            // idiom, one plane over.
            fire_T_ext_plane);
    } else
#endif
    {
        destroyed = this->fire.step(
            fire_field, atmosphere, n_o2, n_bulk_.data(), smoke_field, wall_hp,
            temperature_mut, wind_x, wind_y,
            solid, is_vacuum, flammable,
            h, w, sim_time,
            fuel_recip,          // FUEL-FRACTION AXIS: per-tile 1/hp (see header)
            fire_T_ext_plane);   // PER-MATERIAL T_ext (see header)
    }

    // --- 3. Temperature pass (PhysicsRunner: self.temperature.step) ------
    // Arg order cross-checked against bindings.cpp TemperatureSolver.step and
    // the Python call site:
    //   temperature(mut), heat, heat_inv_shift, face_shift, solid, is_vacuum,
    //   atmosphere, wind_x, wind_y, h, w, dt.
    // `temperature_mut` is the SAME array as the fire's const `temperature`
    // (gmap.temperature in Python) — the binding extracts both a const and a
    // mutable pointer from the one numpy array. The fire read it above; the
    // temperature solver now updates it in place for next tick.
    // S3c: temperature reads the POST-fire-plume int32 `atmosphere` DIRECTLY (its
    // vacuum-exposure threshold is now a Q16.16 integer compare inside the TU). The
    // atm_f_ dequantize bridge that stood here is GONE — step_tail has NO float
    // bridge left in the FIRE/TEMPERATURE path (the centrepiece-arc end-state).
    // EOS refactor P2 (docs/eos_refactor_design.md §4, §8 patch P2): the solver
    // now ALSO takes wind_x/wind_y + sim_time (already in scope here — the SAME
    // wind/dt the fire call above used) for its gas-T semi-Lagrangian pre-pass.
    // CUDA-S1: dispatch to the GPU temperature solver when the backend is
    // switched on (bit-identical to the CPU path — same integer ops). The CPU
    // solver remains the live fallback; with the flag off (default) this is the
    // exact prior call. On a CPU-only build (no BREACH_HAS_CUDA) only the CPU
    // path compiles. P2 NOTE: the GPU kernel (cuda_temperature.cu) is UNTOUCHED
    // by this patch (no CUDA in scope — non-goal) and still only implements the
    // solid convert/conduct/cool passes; the gas-T rules are CPU-only until a
    // later P6 GPU port. `temperature_backend_is_cuda()` defaults false, so this
    // is dormant on every build that doesn't explicitly opt in.
    // EOS refactor P3: TemperatureSolver's Pass 0 (gas-T semi-Lagrangian
    // advection on wind_x/wind_y, P2's additive gas-T rule) is now REDUNDANT
    // — eos_solver already advected T inside run_substeps' substep loop
    // (design §3.2 step 1b), on the solver's OWN evolving u, not the once-
    // per-tick wind this pass used. Passing null wind_x/wind_y here uses
    // TemperatureSolver's documented back-compat no-op path (temperature_
    // solver.h: "Skipped entirely when ... wind_x/wind_y are null") to
    // cleanly disable Pass 0 without touching that TU. Passes 1-3 (heat
    // convert, conduction, ambient cooling) are UNCHANGED — Pass 1's
    // ΔT=ΔE/(N*c_v) deposit still reads `atmosphere` as its density-proxy
    // divisor (a documented `// P3:` TODO in temperature_solver.h asks P3 to
    // swap this for the real N_total; NOT done here — flagged as an open
    // item, see the patch's return report).
    // ***  THERMAL-MASS AXIS: the P1 FINDING, and how P-EOS resolved it  ***
    // P1's finding, kept because it explains the shape of the current code:
    //   * step_tail passes wind_x = wind_y = nullptr above, so the temperature
    //     solver's OWN gas-T advection (Pass 0) NEVER RUNS in the engine —
    //     three of the design's six medium sites are dead code here (they stay
    //     live only for the direct Python binding / unit tests).
    //   * the live semi-Lagrangian advection of `temperature` is EOSSolver::
    //     step's step-1b, and its compression-work term is step-4c. Both keyed
    //     on `solid` / `dyn_permeability` (eos_solver.cpp cmask: sealed iff
    //     `solid || dyn_permeability <= 0`), so a furniture tile was a LIVE GAS
    //     CELL there: its T was overwritten by an upwind backtrace sample every
    //     EOS substep and worked on by −P∇·u. MEASURED on the single-crate
    //     bench (warm seed T=280): run_substeps removed 21–35 game/tick from the
    //     crate's T — 2–4× the COOL_SHIFT loss (T>>5 == 8.75 at T=280).
    // RESOLVED by P-EOS (docs/thermal_mass_eos_ruling_2026-07-30.md, Fable,
    // answering docs/thermal_mass_eos_escalation_2026-07-30.md). The ruling's
    // one rule: on a thermal_solid tile `temperature[]` is OWNED by the
    // TemperatureSolver; every other system is a READER. So EOSSolver::step now
    // takes the same nullable `thermal_solid` this call passes below, SKIPS both
    // of its T writes there, and treats the tile as an occluder in its T
    // backtrace (a SECOND, T-only mask — the shared `cmask` is untouched, so
    // pressure/velocity/gas flow and `permeability`/shield-not-seal are
    // unchanged). COOL_SHIFT is therefore genuinely the crate's one loss channel
    // (§2.2's promise), and the crate's T still drives p* = C·N·T — the ruling's
    // A3 "hot pore gas" decision. See run_substeps' dispatch below.
    //
    // EOS P3: the REAL N divisor for Pass 1's ΔT = ΔE/(N·c_v) deposit (closes
    // the P2 `// P3:` density-proxy TODO; floored inside the solver by its own
    // N_FLOOR_HEAT) is `n_bulk_` — now built ONCE at the top of step_tail (above
    // the fire step) and shared with the continuous-O2 law. The fire step does
    // not mutate `gas`, so it is still the correct pre-temperature divisor here.
#ifdef BREACH_HAS_CUDA
    if (breach_cuda::temperature_backend_is_cuda()) {
        // CUDA-P6.6: dispatch the unified temperature pass to the GPU (bit-
        // identical to the CPU solver — same integer ops; gated by
        // tests/cuda_conduction_check.py). The CPU solver stays the live
        // fallback; with the flag off (default) this is the exact prior call.
        // wind is NULL here — like the CPU path, Pass 0 advection is disabled
        // (eos.step already advected T on its own evolving u; §3.2 step 1b).
        // The dials come straight off the solver so config drives both backends;
        // the per-call rail-hit count folds into the solver's own counter so
        // t_max_phys_hits telemetry is identical whichever backend ran.
        // THERMAL-MASS AXIS, P2 (2026-07-30): the GPU kernels now key their six
        // medium tests on `thermal_solid` too (cuda_temperature.cu, sites marked
        // "MEDIUM-TEST SITE n/6" — the same six, one-to-one with the CPU). So
        // the backends agree byte-for-byte on maps that CARRY FURNITURE, not
        // only on furniture-free ones; gated at tol 0 by
        // tests/cuda_thermal_mass_check.py on a furniture-burn scenario, step
        // path AND resident path. `solid` is still handed to the kernel — it is
        // the documented nullptr fallback for the mask — but no longer selects
        // the medium there, exactly as on the CPU side below.
        this->temperature.t_max_phys_hits += breach_cuda::temperature_step(
            temperature_mut, heat, heat_inv_shift, face_shift,
            solid, is_vacuum, atmosphere, n_bulk_.data(),
            nullptr, nullptr,
            this->temperature.no_face, this->temperature.cool_shift,
            this->temperature.cool_shift_vacuum,
            this->temperature.o2_vacuum_thresh,
            this->temperature.c_v, this->temperature.n_floor_heat,
            this->temperature.gas_advection_rate, this->temperature.T_MAX_PHYS,
            h, w, sim_time,
            is_ambient,       // BC: ring wiped to ΔT=0 in Pass 0 (nullptr = space)
            thermal_solid,    // thermal-mass axis: the per-medium THERMAL mask
            // COOL-SHIFT AXIS: the per-tile decay shift + the floor on the
            // vacuum offset, both straight off the solver/GameMap so the two
            // backends read the SAME dials (the cool_shift/cool_shift_vacuum
            // pair above still supplies the offset itself).
            cool_shift_grid,
            this->temperature.cool_shift_floor,
            // P-F1a (v7.2): the Pass-1 LOW rail counter, accumulated into the
            // same solver-side field the CPU path increments — one counter for
            // the diagnostic regardless of backend.
            &this->temperature.t_low_rail_hits,
            // P-R4: the SIGNED radiation fold, on the GPU twin too.
            rad_net);
    } else
#endif
    {
        // THERMAL-MASS AXIS (docs/thermal_mass_axis_design_2026-07-25.md): the
        // solver's six per-medium tests key on `thermal_solid` (thermal_mass >
        // 0), not on the flow mask `solid` — which is still passed (it is the
        // documented nullptr fallback) but no longer selects the medium. On a
        // furniture-free map the two masks are elementwise equal, so this is
        // byte-identical there.
        this->temperature.step(
            temperature_mut, heat, heat_inv_shift, face_shift,
            solid, is_vacuum, atmosphere,
            n_bulk_.data(),
            nullptr, nullptr,
            h, w, sim_time,
            is_ambient,      // BC: ring wiped to ΔT=0 (Pass-0), vacuum idiom
            thermal_solid,
            // COOL-SHIFT AXIS: Pass 3's per-tile decay shift. The solver's own
            // `cool_shift`/`cool_shift_vacuum`/`cool_shift_floor` members still
            // supply the vacuum OFFSET and its clamp.
            cool_shift_grid,
            // P-R4: the SIGNED radiation accumulator (ruling A1.7). Folded in
            // Pass 1 BEFORE the heat deposit, through each tile's own
            // heat_inv_shift, with shr_round0 + a symmetric saturating add.
            rad_net);
    }

    return destroyed;
}

// EOS refactor P3 (docs/eos_refactor_design.md §3, §8 patch P3) — the
// compressible-solver + smoke-tail substep entry. REPLACES the Patch-2a IMEX
// split (wave loop + single diffuse_solve + n_smoke-substepped per-gas SL
// loop + the decoupled sink_hop BFS loop): `this->eos` runs its OWN internal
// advection substep loop (§3.2, N_SUB_MAX-capped) which ALSO transports the
// two conservative bulk planes (O2/N2) every eos substep — P1's once-per-
// tick bulk_flux_transport call site is GONE (subsumed). After eos.step
// returns, the 5 TRACE gas planes advect ONCE on the solver's final
// wind_x/wind_y (§3.2 step 4b — traces do NOT substep); sink_hop + its BFS
// machinery are DELETED (decisions.md #3 — native venting replaces it).
void PhysicsEngine::run_substeps(
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
        const bool* is_ambient, const int32_t* n_amb, int32_t p_amb,
        const int32_t* sponge_sigma, const int32_t* sponge_udamp,
        bool do_traces,                 // BC + S8a
        const bool* thermal_solid) {    // THERMAL-MASS AXIS, P-EOS
    (void)obstacles;   // EOS P3: the solver's own `solid` mask IS the obstacle
                       // set (gamemap.py: obstacles == solid == permeability<=0);
                       // kept as a parameter for ABI/back-compat with the
                       // Python call site's existing positional argument.

    // EOS P6.5 ("the big flip", docs/eos_p6_gpu_alignment_review.md §4):
    // dispatch the WHOLE eos.step tick to the chained GPU orchestration
    // (cuda_eos_step.cu — P6.2 advection + P6.1 bulk flux device-resident
    // through the substep loop, P6.3 solve, P6.4 kick+compression) when
    // EVERY one of the four EOS kernel-surface flags is on (the review is
    // silent on a master flag, so the per-kernel flags are ANDed — a partial
    // set keeps the CPU path). Bit-identical to eos.step (same digests, same
    // rail counters — gated by tests/cuda_eos_step_check.py); with any flag
    // off (the default) this is the exact prior call. The existing
    // water/smoke/fire dispatch idiom, applied to the EOS orchestration.
#ifdef BREACH_HAS_CUDA
    if (breach_cuda::eos_step_backend_is_cuda()) {
        breach_cuda::eos_step_cuda(
            this->eos,
            atmosphere, p_prev, wind_x, wind_y, temperature,
            gas, gas_conservative, n_gases,
            solid, is_vacuum,
            dyn_permeability, dyn_wave_absorb,
            h, w, sim_time,
            is_ambient, n_amb, p_amb, sponge_sigma, sponge_udamp,    // BC (B4)
            // THERMAL-MASS AXIS, P-EOS: the CUDA twin keys its T write / T
            // backtrace on the SAME mask the CPU does, so the two backends agree
            // bit-for-bit on maps that CARRY FURNITURE — gated at tol 0 by
            // tests/cuda_thermal_mass_eos_check.py, step path AND resident path.
            thermal_solid);
    } else
#endif
    {
        // THERMAL-MASS AXIS, P-EOS (docs/thermal_mass_eos_ruling_2026-07-30.md
        // — the ruling that closes P1's escalation, recorded above §3 of
        // step_tail): the EOS is the pass that ACTUALLY moves T in the live
        // engine (step_tail passes wind == nullptr, so the temperature solver's
        // own Pass-0 advection never runs). Handing it `thermal_solid` is what
        // makes the thermal-mass axis reach the engine: on a crate tile the EOS
        // is now a READER of T (it still derives p* = C·N·T from it — ruling A3,
        // hot pore gas), never a writer. `solid` / `dyn_permeability` / the cmask
        // are untouched, so gas still seeps through the crate exactly as before.
        this->eos.step(
            atmosphere, p_prev, wind_x, wind_y, temperature,
            gas, gas_conservative, n_gases,
            solid, is_vacuum,
            dyn_permeability, dyn_wave_absorb,
            h, w, sim_time,
            is_ambient, n_amb, p_amb, sponge_sigma, sponge_udamp,    // BC
            thermal_solid);
    }

    // S8a Path B: the resident path skips this loop (do_traces=false) and runs
    // the trace planes on device itself (trace_smoke_resident) so the 5 per-plane
    // per-call transfers are gone. Default (do_traces=true) is the exact prior
    // behaviour — the CPU + per-call GPU paths are untouched.
    if (!do_traces) return;

    // Traces advect ONCE per tick, on the solver's final (post-correction)
    // wind_x/wind_y — §3.2 step 4b. Skip the two conservative bulk planes
    // (already transported every eos substep) and any all-zero plane
    // (matches numpy `.any()`).
    const int plane = h * w;
    for (int gi = 0; gi < n_gases; ++gi) {
        if (gas_conservative[gi]) continue;
        int32_t* gas_slice = gas + (size_t)gi * plane;
        bool any = false;
        for (int i = 0; i < plane; ++i) {
            if (gas_slice[i] != 0) { any = true; break; }
        }
        if (!any) continue;
        this->smoke.d_smoke = (float)gas_diffusion[gi];
        // EOS P3 UNIT CONVERSION (engine-owned, FLAGGED): the solver's u is
        // real m/s; SmokeDynamics' SL displacement is wind*(advection_rate*
        // dt) in TILES — the physical rate is exactly 1/dx (u*dt/dx tiles).
        // The config advection_rate (900, calibrated against the OLD
        // -grad(P)-in-q16 wind scale) is DEAD at P3 — left un-read here;
        // feel re-tuning is P5's pass. wind_diffusion_scale is likewise
        // old-wind-unit-calibrated (50 * |8 m/s|^2 would explode the
        // forward-Euler diffusion now that the CFL substep floor is gone) —
        // disabled pending P5 recalibration.
        this->smoke.advection_rate = 1.0f / std::max(this->eos.dx, 1e-3f);
        this->smoke.wind_diffusion_scale = 0.0f;
#ifdef BREACH_HAS_CUDA
        // EOS P6.7 (docs/eos_p6_gpu_alignment_review.md §4, P6.7 row): RESOLVE
        // the P3 once-per-tick cadence assert by wiring the real GPU dispatch.
        // The trace CADENCE changed in the EOS refactor (traces advect ONCE per
        // tick on the solver's final corrected wind, not n_smoke-substepped on
        // the old wave loop's wind), but SmokeDynamics::step's per-pass
        // arithmetic is UNCHANGED — so cuda_smoke.cu's smoke_step (the verbatim
        // S4a device mirror: diffusion Laplacian -> post-diffusion src snapshot
        // -> SL back-trace -> clamp/zero) is bit-identical at the new cadence;
        // only the DISPATCH SITE moved. This is the existing water/smoke/fire/
        // eos dispatch idiom: with the flag OFF (default) it is the EXACT prior
        // CPU call (the live CPU path stays byte-identical); with it ON,
        // smoke_step runs this same single once-per-tick step on the GPU. The
        // subsequent P4 decay->inert_N2 credit below stays on the CPU in BOTH
        // paths (it is not part of the advection pass — strictly additive).
        // Gated by tests/cuda_trace_smoke_check.py (key "trace_smoke").
        if (breach_cuda::smoke_backend_is_cuda()) {
            breach_cuda::smoke_step(
                gas_slice, wind_x, wind_y,
                solid, solid, is_vacuum,
                dyn_permeability,
                h, w, sim_time,
                this->smoke.d_smoke,
                this->smoke.wind_diffusion_scale,
                this->smoke.advection_rate,
                is_ambient);   // BC: ambient ring is a trace sink (null=space)
        } else
#endif
        {
            this->smoke.step(
                gas_slice, wind_x, wind_y,
                solid, solid, is_vacuum,
                dyn_permeability,
                h, w,
                sim_time,
                is_ambient);   // BC: ambient ring is a trace sink (null=space)
        }

        // EOS refactor P4 (design §2.2/§5 v2.1, decisions log #12): apply
        // this trace plane's `decay` column ONCE per tick, right after its
        // own once-per-tick advection above — decay is settling/oxidation
        // into inert bulk, NOT deletion, so the lost mass is credited to
        // inert_N2 IN THE SAME CELL. This closes the v2.1 residual of
        // decision #12: N_total is now conserved through the FULL
        // burn-then-decay cycle, not just the combustion burn. The two
        // conservative bulk planes carry decay=0 by config contract
        // (gases.py), so `gas_conservative[gi]` guards this loop out for
        // them structurally (unreachable here already); `inert_n2_idx`
        // itself is skipped defensively so a self-credit can never happen.
        const float decay_gi = gas_decay[gi];
        if (decay_gi > 0.0f && gi != inert_n2_idx) {
            using namespace fixedpoint;
            q16 frac_q = quantize((double)decay_gi * (double)sim_time);
            if (frac_q < 0) frac_q = 0;
            if (frac_q > FP_ONE) frac_q = FP_ONE;   // a decay*dt >= 1.0 removes it all
            if (frac_q > 0) {
                int32_t* n2_slice = gas + (size_t)inert_n2_idx * plane;
                for (int i = 0; i < plane; ++i) {
                    const int32_t v = gas_slice[i];
                    if (v <= 0) continue;
                    const int32_t lost = mul_q16(v, frac_q);
                    if (lost <= 0) continue;
                    gas_slice[i] = v - lost;
                    n2_slice[i] += lost;
                }
            }
        }
    }
}

// ---- S8a Path A: the fully device-resident EOS stage ----------------------
// Contract in the header; the heavy lifting is breach_cuda::eos_step_resident
// (cuda_eos_resident.cu). Compiled on every build; throws where the device
// path is unavailable so a mis-wired caller fails loudly instead of silently
// running a different arithmetic path.
void PhysicsEngine::run_substeps_resident(
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
        const bool* thermal_solid, std::uintptr_t d_thermal_solid) {
#ifdef BREACH_HAS_CUDA
    if (!breach_cuda::eos_step_backend_is_cuda()) {
        throw std::runtime_error(
            "run_substeps_resident: all four EOS kernel backends must be ON "
            "(sl_advection, bulk_flux, mg_solve, kick_compression) — there is "
            "no CPU fallback for a device-pointer call.");
    }
    breach_cuda::eos_step_resident(
        this->eos,
        atmosphere, p_prev, wind_x, wind_y, temperature,
        gas, gas_conservative, n_gases,
        solid, is_vacuum, dyn_permeability, dyn_wave_absorb,
        h, w, sim_time,
        is_ambient, n_amb, p_amb,
        thermal_solid,   // THERMAL-MASS AXIS: the mirror (host predicate)
        reinterpret_cast<int32_t*>(d_atmosphere),
        reinterpret_cast<int32_t*>(d_wave_p),
        reinterpret_cast<int32_t*>(d_wind_x),
        reinterpret_cast<int32_t*>(d_wind_y),
        reinterpret_cast<int32_t*>(d_temperature),
        reinterpret_cast<int32_t*>(d_gas_base),
        reinterpret_cast<const bool*>(d_solid),
        reinterpret_cast<const bool*>(d_is_vacuum),
        reinterpret_cast<const float*>(d_dyn_permeability),
        reinterpret_cast<const bool*>(d_is_ambient),
        reinterpret_cast<const int32_t*>(d_sponge_sigma),
        reinterpret_cast<const int32_t*>(d_sponge_udamp),
        // THERMAL-MASS AXIS: the DEVICE mask the SL/compression kernels read.
        reinterpret_cast<const bool*>(d_thermal_solid));
#else
    (void)p_prev; (void)atmosphere; (void)wind_x; (void)wind_y;
    (void)temperature; (void)solid; (void)is_vacuum; (void)dyn_permeability;
    (void)dyn_wave_absorb; (void)gas; (void)n_gases; (void)gas_conservative;
    (void)h; (void)w; (void)sim_time; (void)is_ambient; (void)n_amb;
    (void)p_amb; (void)d_atmosphere; (void)d_wave_p; (void)d_wind_x;
    (void)d_wind_y; (void)d_temperature; (void)d_gas_base; (void)d_solid;
    (void)d_is_vacuum; (void)d_dyn_permeability; (void)d_is_ambient;
    (void)d_sponge_sigma; (void)d_sponge_udamp;
    (void)thermal_solid; (void)d_thermal_solid;
    throw std::runtime_error(
        "run_substeps_resident requires the CUDA build (BREACH_CUDA=ON).");
#endif
}

// (The "dead code retained for reference during the P3 review window" banner
// that stood here was DELETED — audit Patch A / A9, 2026-08-04. It described
// the pre-P3 IMEX substep loop, a function that no longer exists; P3 merged
// long ago and took it. Left in place it was actively misleading, because it
// had come to sit directly above the LIVE step_water below and read as if it
// described that.)

// Patch 1 S4c — the water-layer ARRAY ARITHMETIC, lifted verbatim from the body
// of PhysicsRunner._step_water (everything AFTER the Python lazy-init + dormancy
// early-out + sparse source-holds): the substep loop, the W5 flash-boil, the W3
// displacement + flooded seal, and the final copyto(before, water_depth).
//
// BIT-IDENTICAL to the Python: the arrays are float32, the scalar params are
// Python doubles, and numpy casts each double scalar to float32 at the op. We
// reproduce that EXACTLY — every scalar cast to `float` at numpy's cast point,
// every array op in float32 (the /fp:strict TU makes them strict-IEEE, matching
// numpy; /fp:strict does NOT reassociate, so per-cell fusion of the W5/W3 array
// ops is bit-identical to numpy's array-at-a-time order — each cell's arithmetic
// is independent, the only cross-cell op is `.any()`, handled by a separate scan).
// See the header for the per-op precision contract; the inline comments mark each
// spot where the PRECISION (not just the value) had to be matched.
void PhysicsEngine::step_water(
        int32_t* water_depth, int32_t* flow_vx, int32_t* flow_vy,
        const int32_t* floor_height, int32_t* atmosphere,   // S2c: atm Q16.16 == P (EOS P3)
        const bool* solid,
        int32_t* gas, int n_gases,   // S2b: gas Q16.16; n_gases EOS P3 (the W3 evacuation loop)
        int32_t* before, float* dyn_permeability,
        int steam_idx, float tilt_x, float tilt_y,
        int h, int w, float sim_time,
        double ceiling_h, double flood_eps, double ratio_cap,
        double boil_rate, double boil_p_thresh, double steam_yield) const {

    using namespace fixedpoint;

    // EOS refactor P3 (design §6 "water head" row): the water head's FLOAT
    // BRIDGE is RETIRED — `atmosphere` is now the derived integer P directly;
    // water.step reads it as Q16.16 (mul_q16 head term inside the TU). No
    // wave_p read (retired; P already carries the acoustic transient).
    const int32_t* atm_bridge = atmosphere;

    // --- Substep count (the INTEGER CLIFF, S1 §5) + the substep loop -------
    // S1: max_dt is a Q16.16 CONSTANT (water.max_dt_q()); the substep count is a
    // deterministic INTEGER ceil-divide: n = ceil(sim_time / max_dt). Bit-
    // identical across peers (no float64 cliff). sim_time is quantized to Q16.16;
    // n = max(1, ceil_div(sim_time_q, max_dt_q)). wdt = sim_time / n stays the
    // REAL substep length passed (as float) to water.step — the solver folds it
    // into its own Q16.16 step constants internally.
    const q16 max_dt_q = this->water.max_dt_q();
    const q16 sim_time_q = quantize((double)sim_time);
    const int n = std::max(1, fixedpoint::ceil_div(sim_time_q, max_dt_q));
    const float wdt = (float)((double)sim_time / n);
    for (int s = 0; s < n; ++s) {
        // Arg order matches WaterSolver::step: (water_depth, flow_vx, flow_vy,
        // floor_height, atmosphere, solid, h, w, dt, tilt_x, tilt_y).
        // water/velocity/floor are Q16.16; atmosphere is the derived integer P
        // (EOS P3 — the head term is a pure-integer mul_q16(k_p, P) inside step;
        // the old float wave_p bridge is retired). this->water.dx and the pipe
        // params are already members on this->water (not re-passed to the CPU
        // method). CUDA-S3: the GPU water_step is a FREE function, so the
        // solver's scalar dials (g/damping/dx/k_p/v_max/depth_eps) are passed
        // explicitly. Bit-identical to the CPU path (same integer ops); the CPU
        // solver stays the live fallback (flag off by default). CPU-only builds
        // (no BREACH_HAS_CUDA) compile only the CPU call.
#ifdef BREACH_HAS_CUDA
        if (breach_cuda::water_backend_is_cuda()) {
            // EOS P6 (water det-fix): the GPU head term now reconciled with the
            // P3 integer P (docs/water_cuda_head_determinism_fix.md) — atm_bridge
            // is the same int32 P plane the CPU reads, so the GPU path is
            // bit-identical (proven by the S3 head-on gate).
            breach_cuda::water_step(water_depth, flow_vx, flow_vy,
                                    floor_height, atm_bridge, solid,
                                    h, w, wdt, tilt_x, tilt_y,
                                    this->water.g, this->water.damping,
                                    this->water.dx, this->water.k_p,
                                    this->water.v_max, this->water.depth_eps);
        } else
#endif
        {
            this->water.step(water_depth, flow_vx, flow_vy,
                             floor_height, atm_bridge,   // integer head (EOS P3)
                             solid, h, w, wdt, tilt_x, tilt_y);
        }
    }

    // S8a Path B: the W5 flash-boil + W3 displacement + copyto are factored into
    // step_water_tail so the resident path can reuse them (on the mirror) after a
    // device-resident substep loop. Byte-for-byte identical — same host code, same
    // call site; step_water simply delegates the tail now.
    step_water_tail(water_depth, atmosphere, solid, gas, n_gases, before,
                    dyn_permeability, steam_idx, h, w, sim_time,
                    ceiling_h, flood_eps, ratio_cap,
                    boil_rate, boil_p_thresh, steam_yield);
}

// --- S8a Path B: the water HOST TAIL (W5 flash-boil + W3 displacement + copyto),
// split verbatim out of step_water. See physics_engine.h for the contract. Pure
// host arithmetic (/fp:strict), bit-identical whether reached from step_water or
// the resident path.
void PhysicsEngine::step_water_tail(
        int32_t* water_depth, int32_t* atmosphere, const bool* solid,
        int32_t* gas, int n_gases, int32_t* before, float* dyn_permeability,
        int steam_idx, int h, int w, float sim_time,
        double ceiling_h, double flood_eps, double ratio_cap,
        double boil_rate, double boil_p_thresh, double steam_yield) const {

    using namespace fixedpoint;
    const int n_cells = h * w;
    const double Q = (double)FP_ONE;   // 65536 — dequantize divisor

    // --- W5 flash-boil vacuum sink (plan W5) — S2c: int<->int -------------
    // atmosphere + gas (steam) are now Q16.16 int32 (S2 group migrated). The boil
    // is a pressure-keyed water->steam SINK: the threshold compare is integer
    // (atmosphere < quantize(boil_p_thresh)), the depth removed and the steam
    // credited are the SAME water->steam quantity, conserved across the boundary.
    // Only depth->steam moves mass here; atmosphere is read-only (the gate).
    //   boiling = (atmosphere < boil_p_thresh) & (water_depth > 0)
    //   boiled  = where(boiling, min(depth_m, boil_rate*sim_time), 0)
    //   depth_m -= boiled;  gas[steam] += steam_yield*boiled
    const q16 boil_p_thresh_q = quantize((double)boil_p_thresh);
    bool boiling_any = false;
    for (int i = 0; i < n_cells; ++i) {
        if (atmosphere[i] < boil_p_thresh_q && water_depth[i] > 0) {
            boiling_any = true;
            break;
        }
    }
    if (boiling_any) {
        const float boil_amount_f = (float)((double)boil_rate * (double)sim_time);
        const float steam_yield_f = (float)steam_yield;
        // The full-rate boil-off as a Q16.16 count (the cap on what a cell sheds
        // this tick); quantize ONCE (round-to-nearest).
        const q16 boil_amount_q = quantize((double)boil_amount_f);
        int32_t* gas_slice = gas + (std::size_t)steam_idx * n_cells;   // S2b: Q16.16
        for (int i = 0; i < n_cells; ++i) {
            // FLOAT BRIDGE until the fire/steam systems migrate: the boil is a real
            // sink. We work in INTEGER depth so the depth removed and the steam
            // credited are the SAME quantity (water->steam conserved across the
            // bridge): the actual boiled counts are min(depth, full-rate cap),
            // removed from the integer depth; the steam puff is
            // steam_yield * dequantize(that), QUANTIZED back to Q16.16 and integer-
            // added into the (now int32) steam gas plane.
            if (atmosphere[i] < boil_p_thresh_q && water_depth[i] > 0) {
                const q16 boiled_q = std::min(water_depth[i], boil_amount_q);
                water_depth[i] -= boiled_q;                    // exact (>=0 by min)
                const float boiled_m = (float)((double)boiled_q / Q);  // DEQUANTIZE
                const q16 puff_q = quantize((double)(steam_yield_f * boiled_m));
                gas_slice[i] += puff_q;                        // steam puff (Q16.16)
            }
        }
    }

    // --- W3 volume displacement -> occupancy-transition EVACUATION (§2.2) --
    // EOS refactor P3 (design §2.2 "occupancy-transition mass rule", §3.1
    // "water rise (W3)"): REPLACES the old `atmosphere *= ratio` pressure
    // multiply (physics_engine.cpp:599 in the pre-P3 tree) — a flooding cell
    // no longer scales a derived pressure field directly (that field is now
    // solver-owned, materialized once per tick); instead it EVACUATES a
    // `(1 - 1/ratio)` fraction of every gas species' N conservatively into
    // its open (non-solid) neighbors, permeability-weighted, with the LAST
    // neighbor absorbing the integer remainder (exact conservation — the sum
    // removed from the flooding cell exactly equals the sum added to its
    // neighbors). `p* = C*N*T` then rises there next tick — no field
    // multiply, no gain constant, matching §3.1's native-physics description.
    // A receding cell (ratio <= 1) is left untouched — the design specifies
    // only the flooding direction; the freed volume's pressure fall-off
    // falls out of the solver on the next tick from N/T alone. A flooding
    // cell with NO open neighbor (a fully enclosed pocket) has nowhere to
    // evacuate to, so its gas is left to compress in place — the physically
    // correct limit for a sealed nook water is filling.
    //   free_before = max(ceiling_h - before_m, flood_eps)
    //   free_after  = max(ceiling_h - depth_m,  flood_eps)
    //   ratio = clip(free_before/free_after, 1/ratio_cap, ratio_cap)
    const float ceiling_h_f = (float)ceiling_h;
    const float flood_eps_f = (float)flood_eps;
    const float clip_lo_f   = (float)(1.0 / ratio_cap);
    const float clip_hi_f   = (float)ratio_cap;
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            const float before_m = (float)((double)before[i] / Q);       // DEQUANTIZE
            const float depth_m  = (float)((double)water_depth[i] / Q);   // DEQUANTIZE
            const float free_before = std::max(ceiling_h_f - before_m, flood_eps_f);
            const float free_after  = std::max(ceiling_h_f - depth_m,  flood_eps_f);
            float ratio = free_before / free_after;
            ratio = std::min(std::max(ratio, clip_lo_f), clip_hi_f);
            if (free_after <= flood_eps_f) {
                dyn_permeability[i] = 0.0f;                 // flooded -> seal (float)
            }
            if (ratio > 1.0f && !solid[i]) {
                // Open (non-solid) 4-neighbors, permeability-weighted.
                int nbs[4]; float wts[4]; int cnt = 0; float wsum = 0.0f;
                const int cand[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
                for (auto& d : cand) {
                    const int ny = y + d[0], nx = x + d[1];
                    if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                    const int ni = ny * w + nx;
                    if (solid[ni]) continue;
                    const float wgt = std::min(dyn_permeability[i], dyn_permeability[ni]);
                    if (wgt <= 0.0f) continue;
                    nbs[cnt] = ni; wts[cnt] = wgt; wsum += wgt; ++cnt;
                }
                if (cnt > 0) {
                    const q16 frac_q = quantize(1.0 - 1.0 / (double)ratio);
                    for (int gi = 0; gi < n_gases; ++gi) {
                        int32_t* plane = gas + (size_t)gi * n_cells;
                        const q16 evac = mul_q16(plane[i], frac_q);
                        if (evac <= 0) continue;
                        q16 distributed = 0;
                        for (int k = 0; k < cnt; ++k) {
                            const q16 share = (k == cnt - 1)
                                ? (q16)(evac - distributed)   // remainder -> exact conservation
                                : mul_q16(evac, quantize((double)(wts[k] / wsum)));
                            plane[nbs[k]] += share;
                            distributed += share;
                        }
                        plane[i] -= distributed;
                    }
                }
            }
            before[i] = water_depth[i];                        // the copyto (integer)
        }
    }
}

// stamp_units — the per-tick dynamic-field rebuild, lifted from the FIELD-REBUILD
// half of GameMap.stamp_units (gamemap.py:485-589). PURE-STRUCTURE move: the ops
// are exact (copies + a boolean compare + per-cell min/max), so there is NO float
// arithmetic and it is 0-ULP by construction — /fp:strict is irrelevant here.
//
// The unit iteration / occupied_tiles() / `u.alive` filter / per-tile bounds
// check / the getattr-or-default for each unit's perm/wabsorb/atten all stay in
// Python (CPU actors own that), flattened into the per-row arrays passed in. This
// reproduces the contract directions EXACTLY: permeability is MIN (never unseal a
// door), wave_absorb is MAX (a body only adds damping), light_atten is per-channel
// MAX (opacity only rises). The atmosphere-refill bit (gamemap.py:586-588) is NOT
// here — it stays Python (Q1, locked). All writes are IN-PLACE (the engine re-
// fetches field pointers each step; reassignment would dangle them).
void PhysicsEngine::stamp_units(
        const float* permeability, const float* wave_absorb,
        const float* light_atten,
        float* dyn_permeability, float* dyn_wave_absorb, float* dyn_light_atten,
        bool* obstacles,
        const int32_t* ys, const int32_t* xs,
        const float* perm, const float* wabsorb,
        const float* atten_r, const float* atten_g, const float* atten_b,
        int n_stamp, int h, int w) const {

    const int n = h * w;

    // --- a. Reset to static baseline, IN-PLACE ----------------------------
    // obstacles = (permeability <= 0.0): WALLS ONLY (units are NOT stamped into
    // obstacles — they are soft bodies, gamemap.py:532). dyn_* are in-place
    // copies of the static material baselines (gamemap.py:536/540/544). Done in
    // one pass over the (h,w) fields; light_atten is interleaved (h,w,3).
    for (int i = 0; i < n; ++i) {
        obstacles[i]        = (permeability[i] <= 0.0f);   // walls only
        dyn_permeability[i] = permeability[i];             // copy
        dyn_wave_absorb[i]  = wave_absorb[i];              // copy
    }
    for (int i = 0; i < n * 3; ++i) {
        dyn_light_atten[i]  = light_atten[i];              // copy (RGB)
    }

    // --- b. Stamp each living unit's footprint over the flat rows ---------
    // One row per (living-unit, in-bounds footprint-tile) — Python already did
    // the `u.alive` filter and the 0<=ty<h && 0<=tx<w bounds check, so every row
    // here is a valid stamp. idx = ys*w + xs (row-major, matching numpy [ty,tx]).
    for (int r = 0; r < n_stamp; ++r) {
        const int idx = ys[r] * w + xs[r];
        // MIN vs the STATIC permeability (gamemap.py:571-572): a body makes an
        // open tile porous but must never RAISE a sealed tile's permeability (the
        // door-stamp leak). Compare against permeability[idx], NOT the running
        // dyn value — exactly as Python (`sp = self.permeability[ty, tx]`).
        const float sp = permeability[idx];
        const float up = perm[r];
        dyn_permeability[idx] = (up < sp) ? up : sp;
        // MAX so a unit only ADDS damping (gamemap.py:575-576).
        const float cur = dyn_wave_absorb[idx];
        const float uw  = wabsorb[r];
        dyn_wave_absorb[idx] = (cur >= uw) ? cur : uw;
        // Per-channel MAX: opacity can only increase (gamemap.py:578-581).
        float* cell = dyn_light_atten + (size_t)idx * 3;
        const float ar = atten_r[r];
        const float ag = atten_g[r];
        const float ab = atten_b[r];
        cell[0] = (cell[0] >= ar) ? cell[0] : ar;
        cell[1] = (cell[1] >= ag) ? cell[1] : ag;
        cell[2] = (cell[2] >= ab) ? cell[2] : ab;
    }
}
