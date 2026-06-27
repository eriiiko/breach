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
#include "fixed_point.h"   // S1: Q16.16 toolkit (quantize, ceil_div, max_dt_q)

#include <algorithm>   // std::max, std::min
#include <cmath>       // std::ceil
#include <cstddef>     // std::size_t

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
        const int32_t* water_depth, const int32_t* wave_p,   // S1: water_depth Q16.16
                                                             // S2a: wave_p Q16.16
        const bool* solid,
        // fire group — S3a: fire is Q16.16 int32 too; S2c: atmosphere + wind are
        // Q16.16 int32 (the fire field + atm/wind bridges are below)
        int32_t* fire_field, int32_t* atmosphere, int32_t* smoke_field, float* wall_hp,  // S3a: fire Q16.16; S2b: smoke Q16.16; S2c: atm Q16.16
        const int32_t* temperature, const int32_t* wind_x, const int32_t* wind_y,      // S2c: wind Q16.16
        const bool* is_vacuum, const bool* flammable,
        // temperature group
        int32_t* temperature_mut, const int32_t* heat,
        const int32_t* heat_inv_shift, const int32_t* face_shift,
        int h, int w, float sim_time) const {

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
        // step_ripple writes only ripple / ripple_v; water_depth / wave_p /
        // solid are read-only (the locked canon rule). Same args, same order
        // as the Python call: (ripple, ripple_v, water_depth, wave_p, solid, dt).
        // S2a FLOAT BRIDGE: wave_p is Q16.16 int32 now; the ripple splash source
        // (k_splash*wave_p, a render-only feel dial) reads it as float — so
        // DEQUANTIZE wave_p into the reused float scratch and pass THAT (the
        // water TU stays float; the bridge collapses when ripple goes integer).
        using namespace fixedpoint;
        if (wave_p_f_.size() != (size_t)n) wave_p_f_.assign(n, 0.0f);
        for (int i = 0; i < n; ++i) wave_p_f_[i] = dequantize_f(wave_p[i]);  // FLOAT BRIDGE
        this->water.step_ripple(ripple, ripple_v, water_depth, wave_p_f_.data(),
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
    // S2c FIRE BRIDGE (the ONE float bridge S2 leaves open — downstream to S3):
    // atmosphere + wind are Q16.16 int32, but the fire reads them as float AND
    // mutates atmosphere (the own-tile plume). DEQUANTIZE atmosphere/wind into the
    // reused float scratch, run the float fire (it reads + writes the float
    // atmosphere), then RE-QUANTIZE the fire-mutated atmosphere back into the int32
    // field (round-to-nearest). The temperature pass reads the SAME float
    // atmosphere scratch (read-only). Collapses to integer<-integer when fire
    // migrates (S3). The plume is a fire SOURCE (non-conserved by design), so the
    // round-trip quantize is bit-safe (deterministic; no synced-field aliasing).
    if (atm_f_.size()    != (size_t)n) atm_f_.assign(n, 0.0f);
    if (wind_x_f_.size() != (size_t)n) wind_x_f_.assign(n, 0.0f);
    if (wind_y_f_.size() != (size_t)n) wind_y_f_.assign(n, 0.0f);
    if (fire_f_.size()   != (size_t)n) fire_f_.assign(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        atm_f_[i]    = dequantize_f(atmosphere[i]);   // FIRE BRIDGE (int32 -> float)
        wind_x_f_[i] = dequantize_f(wind_x[i]);
        wind_y_f_[i] = dequantize_f(wind_y[i]);
        fire_f_[i]   = dequantize_f(fire_field[i]);   // S3a FIRE FIELD BRIDGE (int32 -> float)
    }
    std::vector<std::pair<int, int>> destroyed = this->fire.step(
        fire_f_.data(), atm_f_.data(), smoke_field, wall_hp,
        temperature, wind_x_f_.data(), wind_y_f_.data(),
        solid, is_vacuum, flammable,
        h, w, sim_time);
    // Re-quantize the fire-mutated atmosphere back into the int32 field. Fire only
    // ADDS a small positive plume to its own burning tiles, so most cells are
    // unchanged; round-to-nearest matches the dequantize on the unchanged cells
    // (exact round-trip) and lands the plume increment unbiased.
    for (int i = 0; i < n; ++i) atmosphere[i] = quantize((double)atm_f_[i]);
    // S3a FIRE FIELD BRIDGE close: re-quantize the (float, mutated) fire back into
    // the int32 Q16.16 field, round-to-nearest. The float fire clamps to [0,1], so
    // every cell is in range; round-to-nearest matches the dequantize on unchanged
    // cells (exact round-trip) and lands the logistic step unbiased. The C++
    // logistic itself stays FLOAT this commit — S3b makes it integer and deletes
    // this bridge (S3c then deletes the atm/wind bridges too).
    for (int i = 0; i < n; ++i) fire_field[i] = quantize((double)fire_f_[i]);

    // --- 3. Temperature pass (PhysicsRunner: self.temperature.step) ------
    // Arg order cross-checked against bindings.cpp TemperatureSolver.step and
    // the Python call site:
    //   temperature(mut), heat, heat_inv_shift, face_shift, solid, is_vacuum,
    //   atmosphere.
    // `temperature_mut` is the SAME array as the fire's const `temperature`
    // (gmap.temperature in Python) — the binding extracts both a const and a
    // mutable pointer from the one numpy array. The fire read it above; the
    // temperature solver now updates it in place for next tick.
    // S2c: temperature reads atmosphere as float (read-only, the space-facing
    // threshold) — pass the SAME dequantized float scratch the fire used (it was
    // re-quantized into the int32 field above, but the float view is still valid
    // and read-only here). FIRE BRIDGE (collapses when temperature migrates).
    this->temperature.step(
        temperature_mut, heat, heat_inv_shift, face_shift,
        solid, is_vacuum, atm_f_.data(),
        h, w);

    return destroyed;
}

// Patch 1 S4b — the IMEX atmosphere/smoke substep loop, lifted from the middle
// of PhysicsRunner.step (between _step_water and step_tail).
//
// Patch 2a RESHAPE (BEHAVIOR CHANGE, feel-gated — the 0-ULP A/B harness no
// longer applies): the fused atmos.step that ran `n` times is split into a wave
// loop (n_wave substeps at the wave CFL) + a SINGLE implicit diffusion solve at
// the full sim_time + the per-gas smoke loop (n_wave×, unchanged) relocated to
// run AFTER the single diffuse on the once-computed wind. The integer-cliff `n`
// derivation and the dt_actual/dt_smoke double-until-the-boundary contract are
// preserved exactly; only the loop STRUCTURE changed. See the body comment.
void PhysicsEngine::run_substeps(
        int32_t* wave_p, int32_t* wave_v, int32_t* wave_source,  // S2a: Q16.16
        int32_t* atmosphere,                                     // S2c: Q16.16
        int32_t* wind_x, int32_t* wind_y,                        // S2c: Q16.16
        const bool* obstacles, const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability, const float* dyn_wave_absorb,
        int32_t* gas, const float* gas_diffusion, int n_gases,   // S2b: gas Q16.16
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
    // Patch 2b: dt_scale is GONE. Smoke advects/diffuses on the REAL tick length
    // (sim_time, sub-stepped n_smoke× below for the smoke-CFL floor). The visible
    // wind-ride is preserved by the ×dt_scale²-bumped advection_rate default; the
    // smoke DIFFUSION is now dt_scale² (≈9×) weaker than the shipped build —
    // Erik re-tunes d_smoke / the per-gas [gases.*] diffusion table.

    const int plane = h * w;  // elements per gas plane (gas is (N,h,w) contiguous)

    // === Patch 2: the dt-policy decouple — wave / diffusion / smoke / sink ===
    // BEHAVIOR CHANGE (feel-gated, not 0-ULP). Each system runs on its OWN count:
    //   1. the WAVE substeps `n_wave` times at its CFL dt_actual (Patch 2a);
    //   2. the implicit DIFFUSION solves ONCE per tick at the FULL sim_time, and
    //      computes the wind ONCE (Patch 2a);
    //   3. (Patch 2b) the SMOKE runs `n_smoke`× on the once-computed wind — a
    //      smoke-CFL FLOOR from the spatial-max d_eff (auto-tightens under a
    //      shockwave, ≈1 at rest); WIND-ONLY advection now (no fused sink);
    //   4. (Patch 2b) the breach SINK runs as its OWN loop, K = smoke.vent_hops
    //      one-cell BFS hops per tick — a real "vent cells/tick" dial decoupled
    //      from n_wave (was implicitly n_wave×/tick when fused in the back-trace).
    const int n_wave = n;

    // 1. Wave substeps at the wave CFL (dt_actual), n_wave times.
    for (int s = 0; s < n_wave; ++s) {
        this->atmos.wave_substep(
            wave_p, wave_v, wave_source, atmosphere,
            obstacles, solid, is_vacuum,
            dyn_permeability,
            dyn_wave_absorb,
            h, w,
            (float)dt_actual);
    }

    // 2. Implicit diffusion + BCs + wind, ONCE, at the FULL sim_time. The wind
    // is written here for the smoke below. sim_time is the float passed from
    // Python; the diffuse_solve dt is that full tick length (NOT dt_actual).
    this->atmos.diffuse_solve(
        atmosphere, wave_p, wave_v, wave_source,
        wind_x, wind_y,
        obstacles, solid, is_vacuum,
        dyn_permeability,
        h, w,
        sim_time);

    // 3. Smoke-CFL floor (Patch 2b). Smoke's explicit diffusion is forward-Euler,
    // so it is CFL-bound; the effective diffusion spikes under wind:
    //   d_eff = d_smoke·(1 + wind_diffusion_scale·|wind|²).
    // Use the SPATIAL-MAX |wind|² over the grid (the wind is known now — diffuse
    // wrote it) and the MAX per-gas d_smoke (the worst-case plane), then the
    // forward-Euler stability bound dt < dx²/(4·d_eff_max) with dx=1 (tile units)
    // gives n_smoke = max(1, ceil(sim_time / (dx²/(4·d_eff_max)))). At rest this
    // is ≈1; it tightens only under a shockwave (the safety net against a
    // checkerboard). With dt_scale gone the smoke dt is already ~9× smaller, so
    // this is usually 1 — it only bites in extreme wind.
    // S2c: wind is Q16.16 int32 now. The per-cell |wind|² is an INTEGER max
    // reduction (plan §2.2 #2 — max is order-free for integers, unlike a sum):
    // square each component as mul_wide (Q16.16·Q16.16 -> int64 Q.32), sum, keep
    // the running int64 max. Then dequantize the Q.32 max ONCE to a real
    // max_wind_sq (the n_smoke cliff arithmetic stays in double — it is a
    // config-derived CFL bound, not synced field state; the cliff COUNT is what
    // must be deterministic, and a single dequantize of the order-free integer
    // max feeds it identically on every peer).
    using namespace fixedpoint;
    int64_t max_wind_sq_q32 = 0;     // Q.32 (Q16.16²)
    for (int i = 0; i < plane; ++i) {
        const int64_t ws = mul_wide(wind_x[i], wind_x[i])
                         + mul_wide(wind_y[i], wind_y[i]);
        if (ws > max_wind_sq_q32) max_wind_sq_q32 = ws;
    }
    const double max_wind_sq =
        (double)max_wind_sq_q32 / ((double)FP_ONE * (double)FP_ONE);  // Q.32 -> real
    float d_smoke_max = 0.0f;
    for (int gi = 0; gi < n_gases; ++gi) {
        if (gas_diffusion[gi] > d_smoke_max) d_smoke_max = gas_diffusion[gi];
    }
    const double d_eff_max =
        (double)d_smoke_max *
        (1.0 + (double)this->smoke.wind_diffusion_scale * (double)max_wind_sq);
    int n_smoke = 1;
    if (d_eff_max > 0.0) {
        const double dt_stable = 1.0 / (4.0 * d_eff_max);   // dx²/(4·d_eff), dx=1
        n_smoke = std::max(1, (int)std::ceil((double)sim_time / dt_stable));
    }
    // The smoke dt: the full tick split n_smoke ways, so total advection over the
    // tick is sim_time·wind regardless of n_smoke (the wind is the once-computed
    // quasi-static field). cast to float at the .step() boundary.
    const float dt_smoke = (float)((double)sim_time / n_smoke);

    // Per-gas smoke transport (engine/05 §6.2, M1) — WIND-ONLY now (the breach
    // sink is the separate K-hop loop below). n_smoke× the per-gas loop; skip an
    // all-zero plane (numpy `.any()`); set d_smoke BEFORE each step (member-set).
    for (int s = 0; s < n_smoke; ++s) {
        for (int gi = 0; gi < n_gases; ++gi) {
            int32_t* gas_slice = gas + (size_t)gi * plane;     // S2b: Q16.16
            bool any = false;
            for (int i = 0; i < plane; ++i) {
                if (gas_slice[i] != 0) { any = true; break; }   // integer .any()
            }
            if (!any) {
                continue;  // empty slice — nothing to transport (matches `.any()`)
            }
            this->smoke.d_smoke = (float)gas_diffusion[gi];
            this->smoke.step(
                gas_slice, wind_x, wind_y,
                obstacles, solid, is_vacuum,
                dyn_permeability,
                h, w,
                dt_smoke);
        }
    }

    // 4. The decoupled breach SINK (Patch 2b): K = smoke.vent_hops one-cell BFS
    // hops per tick, its OWN loop AFTER the smoke loop. K is a real "vent
    // cells/tick" dial independent of n_wave (was implicitly n_wave×/tick when
    // the sink was fused into the back-trace). With no breach sink_x/sink_y are
    // all-zero, so each hop is the identity — sealed rooms are untouched. The
    // per-gas `.any()` skip avoids hopping an empty plane.
    const int K = this->smoke.vent_hops;
    for (int k = 0; k < K; ++k) {
        for (int gi = 0; gi < n_gases; ++gi) {
            int32_t* gas_slice = gas + (size_t)gi * plane;     // S2b: Q16.16
            bool any = false;
            for (int i = 0; i < plane; ++i) {
                if (gas_slice[i] != 0) { any = true; break; }   // integer .any()
            }
            if (!any) {
                continue;
            }
            this->smoke.sink_hop(
                gas_slice, sink_x, sink_y,
                obstacles, solid, is_vacuum,
                dyn_permeability,
                h, w);
        }
    }
}

// Patch 1 S4c — the water-layer ARRAY ARITHMETIC, lifted verbatim from the body
// of PhysicsRunner._step_water (everything AFTER the Python lazy-init + dormancy
// early-out + sparse source-holds): the substep loop, the W5 flash-boil, the W3
// displacement + flooded seal, and the final copyto(before, water_depth).
//
// BIT-IDENTICAL to the Python: the arrays are float32, the scalar params are
// Python doubles, and numpy casts each double scalar to float32 at the op. We
// reproduce that EXACTLY — every scalar cast to `float` at numpy's cast point,
// every array op in float32 (the /fp:precise TU makes them strict-IEEE, matching
// numpy; /fp:precise does NOT reassociate, so per-cell fusion of the W5/W3 array
// ops is bit-identical to numpy's array-at-a-time order — each cell's arithmetic
// is independent, the only cross-cell op is `.any()`, handled by a separate scan).
// See the header for the per-op precision contract; the inline comments mark each
// spot where the PRECISION (not just the value) had to be matched.
void PhysicsEngine::step_water(
        int32_t* water_depth, int32_t* flow_vx, int32_t* flow_vy,
        const int32_t* floor_height, int32_t* atmosphere, const int32_t* wave_p,  // S2a: wave_p Q16.16; S2c: atm Q16.16
        const bool* solid,
        int32_t* gas,   // S2b: gas Q16.16
        int32_t* before, float* dyn_permeability,
        int steam_idx, float tilt_x, float tilt_y,
        int h, int w, float sim_time,
        double ceiling_h, double flood_eps, double ratio_cap,
        double boil_rate, double boil_p_thresh, double steam_yield) const {

    using namespace fixedpoint;
    const int n_cells = h * w;
    const double Q = (double)FP_ONE;   // 65536 — dequantize divisor

    // WATER HEAD BRIDGE: the water solver's head term k_p·(atm+wave_p) reads
    // atmosphere + wave_p as FLOAT (the gated head-term bridge inside water.step).
    //   * wave_p (S2a): Q16.16 int32 — dequantize into the reused float scratch.
    //   * atmosphere (S2c): NOW Q16.16 int32 — dequantize into the reused atm_f_
    //     scratch so the water head reads the SAME real pressure as before. The
    //     synced water-head read is thus integer-sourced (one dequantize at the
    //     boundary); ripple's wave_p read stays a documented render-local
    //     dequantize. With k_p != 0 (shipped 0.5) the head IS read every substep,
    //     so both bridges are live. Collapses to integer<-integer when the head
    //     term goes fully integer (a later water/atmosphere unification).
    if (wave_p_f_.size() != (size_t)n_cells) wave_p_f_.assign(n_cells, 0.0f);
    if (atm_f_.size()    != (size_t)n_cells) atm_f_.assign(n_cells, 0.0f);
    for (int i = 0; i < n_cells; ++i) {
        wave_p_f_[i] = dequantize_f(wave_p[i]);       // S2a head bridge
        atm_f_[i]    = dequantize_f(atmosphere[i]);   // S2c head bridge
    }
    const float* wave_p_bridge = wave_p_f_.data();
    const float* atm_bridge    = atm_f_.data();

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
        // floor_height, atmosphere, wave_p, solid, h, w, dt, tilt_x, tilt_y).
        // water/velocity/floor are Q16.16; atmosphere/wave_p stay float (the
        // gated head-term FLOAT BRIDGE lives inside step). this->water.dx and the
        // pipe params are already members on this->water (not re-passed).
        this->water.step(water_depth, flow_vx, flow_vy,
                         floor_height, atm_bridge, wave_p_bridge,   // head bridges
                         solid, h, w, wdt, tilt_x, tilt_y);
    }

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

    // --- W3 volume displacement + flooded seal (plan W3, §5.1) — S2c bridge --
    // before + water_depth are Q16.16; dyn_permeability is float (a structural
    // cache). The isothermal P*V ratio is a real-valued computation (dequantize
    // before/water_depth, form free_before/free_after, the clipped ratio), but the
    // ATMOSPHERE SCALE is now integer: atmosphere = mul_q16(atmosphere, quantize
    // (ratio)). This is a P*V compression of the conserved bulk (a per-cell
    // multiply, NOT conserved-by-design — like the sponge sink). The ratio
    // computation stays float (it is a real ratio of free air columns, not a
    // synced field); only the application to the int32 atmosphere is integer.
    //   free_before = max(ceiling_h - before_m, flood_eps)
    //   free_after  = max(ceiling_h - depth_m,  flood_eps)
    //   ratio = clip(free_before/free_after, 1/ratio_cap, ratio_cap)
    //   atmosphere = mul_q16(atmosphere, quantize(ratio)); flooded -> dyn_perm = 0
    //   before = water_depth   (integer copy)
    const float ceiling_h_f = (float)ceiling_h;
    const float flood_eps_f = (float)flood_eps;
    const float clip_lo_f   = (float)(1.0 / ratio_cap);
    const float clip_hi_f   = (float)ratio_cap;
    for (int i = 0; i < n_cells; ++i) {
        const float before_m = (float)((double)before[i] / Q);       // DEQUANTIZE
        const float depth_m  = (float)((double)water_depth[i] / Q);   // DEQUANTIZE
        const float free_before = std::max(ceiling_h_f - before_m, flood_eps_f);
        const float free_after  = std::max(ceiling_h_f - depth_m,  flood_eps_f);
        float ratio = free_before / free_after;
        ratio = std::min(std::max(ratio, clip_lo_f), clip_hi_f);
        atmosphere[i] = mul_q16(atmosphere[i], quantize((double)ratio));  // P*V (int)
        if (free_after <= flood_eps_f) {
            dyn_permeability[i] = 0.0f;                     // flooded -> seal (float)
        }
        before[i] = water_depth[i];                        // the copyto (integer)
    }
}

// stamp_units — the per-tick dynamic-field rebuild, lifted from the FIELD-REBUILD
// half of GameMap.stamp_units (gamemap.py:485-589). PURE-STRUCTURE move: the ops
// are exact (copies + a boolean compare + per-cell min/max), so there is NO float
// arithmetic and it is 0-ULP by construction — /fp:precise is irrelevant here.
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
