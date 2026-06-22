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
    float max_wind_sq = 0.0f;
    for (int i = 0; i < plane; ++i) {
        const float ws = wind_x[i] * wind_x[i] + wind_y[i] * wind_y[i];
        if (ws > max_wind_sq) max_wind_sq = ws;
    }
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
            float* gas_slice = gas + (size_t)gi * plane;
            bool any = false;
            for (int i = 0; i < plane; ++i) {
                if (gas_slice[i] != 0.0f) { any = true; break; }
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
            float* gas_slice = gas + (size_t)gi * plane;
            bool any = false;
            for (int i = 0; i < plane; ++i) {
                if (gas_slice[i] != 0.0f) { any = true; break; }
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
        float* water_depth, float* flow_vx, float* flow_vy,
        const float* floor_height, float* atmosphere, const float* wave_p,
        const bool* solid,
        float* gas,
        float* before, float* dyn_permeability,
        int steam_idx, float tilt_x, float tilt_y,
        int h, int w, float sim_time,
        double ceiling_h, double flood_eps, double ratio_cap,
        double boil_rate, double boil_p_thresh, double steam_yield) const {

    const int n_cells = h * w;

    // --- Substep count + the WaterSolver.step substep loop ----------------
    // Python: wdt_max = self.water.max_dt() (a float32 PROMOTED to a Python
    // double); n = max(1, int(ceil(sim_time / wdt_max))) — DOUBLE division +
    // ceil + int truncation (the S4b integer-cliff pattern; a 1-ULP slip flips n).
    // wdt = sim_time / n stays a Python double, narrowed to float32 only at the
    // water.step boundary (pybind's double->float32 cast).
    const double wdt_max = (double)this->water.max_dt();
    const int n = std::max(1, (int)std::ceil((double)sim_time / wdt_max));
    const float wdt = (float)((double)sim_time / n);
    for (int s = 0; s < n; ++s) {
        // Arg order matches the WaterSolver.step binding: (water_depth, flow_vx,
        // flow_vy, floor_height, atmosphere, wave_p, solid, dt, tilt_x, tilt_y).
        // floor_height/atmosphere/wave_p are nullable in the binding but here are
        // always passed (the Python call site passes all three). this->water.dx is
        // already bound (set Python-side on the lazy init), and the pipe params are
        // members on this->water — so they are NOT re-passed.
        this->water.step(water_depth, flow_vx, flow_vy,
                         floor_height, atmosphere, wave_p,
                         solid, h, w, wdt, tilt_x, tilt_y);
    }

    // --- W5 flash-boil vacuum sink (plan W5) ------------------------------
    // Python:
    //   boiling = (atmosphere < boil_p_thresh) & (water_depth > 0)
    //   if boiling.any():
    //       boiled = np.where(boiling,
    //                         np.minimum(water_depth, boil_rate*sim_time), 0.0)
    //       water_depth -= boiled
    //       gas[steam_idx] += (steam_yield * boiled).astype(np.float32)
    // numpy semantics matched exactly:
    //   * `atmosphere < boil_p_thresh`: the double scalar is cast to float32 for
    //     the comparison -> atmosphere[i] < (float)boil_p_thresh.
    //   * `water_depth > 0`: 0 (int) vs float32 -> water_depth[i] > 0.0f.
    //   * `boil_rate*sim_time`: DOUBLE * DOUBLE = double, computed ONCE; the
    //     np.minimum then casts that double to float32 ONCE -> min(water_depth[i],
    //     (float)boil_amount_d). (sim_time is the float passed from Python,
    //     promoted to double for the product — matching Python's double*double.)
    //   * `steam_yield * boiled`: numpy keeps the float32 array's dtype, so the
    //     multiply is in FLOAT32 ((float)steam_yield * boiled[i]); a double-
    //     multiply-then-cast is 1 ULP off (verified vs numpy 1.26.4). .astype is
    //     then a no-op.
    //   * `.any()` guard: scan boiling first; skip the whole block if nothing
    //     boils (matches the Python `if boiling.any():`).
    const float boil_p_thresh_f = (float)boil_p_thresh;
    bool boiling_any = false;
    for (int i = 0; i < n_cells; ++i) {
        if (atmosphere[i] < boil_p_thresh_f && water_depth[i] > 0.0f) {
            boiling_any = true;
            break;
        }
    }
    if (boiling_any) {
        // boil_rate*sim_time in DOUBLE, cast to float32 ONCE (the np.minimum cast).
        const float boil_amount_f = (float)((double)boil_rate * (double)sim_time);
        const float steam_yield_f = (float)steam_yield;   // the (f32)steam_yield cast
        float* gas_slice = gas + (std::size_t)steam_idx * n_cells;
        for (int i = 0; i < n_cells; ++i) {
            // boiled = where(boiling, minimum(water_depth, boil_amount), 0.0) — f32
            float boiled;
            if (atmosphere[i] < boil_p_thresh_f && water_depth[i] > 0.0f) {
                boiled = std::min(water_depth[i], boil_amount_f);
            } else {
                boiled = 0.0f;
            }
            water_depth[i] -= boiled;
            // (steam_yield * boiled) in FLOAT32 (numpy keeps the f32 dtype).
            gas_slice[i] += steam_yield_f * boiled;
        }
    }

    // --- W3 volume displacement + flooded seal (plan W3, canon §5.1) ------
    // Python:
    //   free_before = np.maximum(ceiling_h - before, flood_eps)
    //   free_after  = np.maximum(ceiling_h - water_depth, flood_eps)
    //   ratio = np.clip(free_before / free_after, 1.0 / ratio_cap, ratio_cap)
    //   np.multiply(atmosphere, ratio, out=atmosphere)
    //   flooded = free_after <= flood_eps
    //   dyn_permeability[flooded] = 0.0
    //   np.copyto(before, water_depth)
    // numpy semantics matched exactly:
    //   * `ceiling_h - x`: double scalar cast to float32, subtraction in float32
    //     -> (float)ceiling_h - x[i].
    //   * `np.maximum(., flood_eps)`: flood_eps cast to float32 -> max(., (float)
    //     flood_eps).
    //   * `np.clip(x, lo, hi)` == min(max(x, lo), hi) in float32 with the bounds
    //     cast to f32; the LOW bound 1.0/ratio_cap is computed in DOUBLE then cast
    //     to f32 (the reciprocal precision point), HIGH bound is (float)ratio_cap.
    //   * `atmosphere *= ratio` in float32.
    //   * `flooded = free_after <= flood_eps` -> free_after[i] <= (float)flood_eps
    //     (same f32 flood_eps as the maxima above).
    //   * the copyto closes the accounting loop: before[i] = water_depth[i].
    // Per-cell fusion is safe (each cell independent, no reassociation under
    // /fp:precise) and bit-identical to numpy's array-at-a-time order.
    const float ceiling_h_f = (float)ceiling_h;
    const float flood_eps_f = (float)flood_eps;
    const float clip_lo_f   = (float)(1.0 / ratio_cap);   // reciprocal in DOUBLE, then f32
    const float clip_hi_f   = (float)ratio_cap;
    for (int i = 0; i < n_cells; ++i) {
        const float free_before = std::max(ceiling_h_f - before[i], flood_eps_f);
        const float free_after  = std::max(ceiling_h_f - water_depth[i], flood_eps_f);
        // np.clip == min(max(x, lo), hi) in float32.
        float ratio = free_before / free_after;
        ratio = std::min(std::max(ratio, clip_lo_f), clip_hi_f);
        atmosphere[i] *= ratio;                            // P*V const
        if (free_after <= flood_eps_f) {
            dyn_permeability[i] = 0.0f;                     // flooded -> seal the cell
        }
        before[i] = water_depth[i];                        // the copyto
    }
}
