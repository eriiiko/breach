#pragma once
// WaterSolver — the pipe model: damped velocity + donor-cell upwind mass flux.
//
// Canon design: docs/architecture/engine/07_fluid_and_water.md §2.
// Build plan:   docs/water_implementation_plan.md Step W1.
//
// Per step (metres / m/s / seconds throughout — no tile-unit constants):
//   1. surface = floor_height + tilt_offset + water_depth
//                (+ k_p*(atmosphere + wave_p) — GATED: k_p == 0 never reads them)
//   2. v += dt*(-g*grad(surface) - damping*v)  — central difference; Neumann
//      MIRROR of the centre value at solid neighbours; out-of-bounds is solid
//      (grid border = wall). v = 0 on solid; componentwise clamp to +-v_max.
//   3. Donor-cell upwind face fluxes from PRE-update depth (gather), zeroed at
//      solid faces; per-cell OUTFLOW LIMITER (a cell can be donor on up to 4
//      faces — scale ITS outgoing fluxes by depth*dx/(dt*out_sum) when
//      out_sum*dt/dx > depth, so the non-negative clamp cannot create mass);
//      then apply the divergence in one pass.
//   4. depth = max(depth, 0); 0 on solid; snapped to 0 below depth_eps.
//
// Deterministic: no RNG, gather-then-apply everywhere, fixed iteration order.
// Stateless: params are public members; step takes raw pointers (house
// pattern, AtmosphereSolver). Python owns the substep loop via max_dt().

struct WaterSolver {
    // tunables (bound from config [physics.water])
    float g         = 9.81f;   // m/s^2 (prototype-validated)
    float damping   = 1.0f;    // 1/s pipe friction — lifted from fluid_test.py:39 (the
                               // side-by-side run that justified the model); tune in [0.5, 1.0]
    float dx        = 0.333f;  // m — set from the level's tile_size_m, never assumed
    float k_p       = 0.0f;    // pressure head, m per pressure-unit; 0 == head OFF (W4 turns on)
    float v_max     = 8.0f;    // m/s safety clamp (safe WITH the outflow limiter)
    float depth_eps = 1e-5f;   // m snap-to-zero (kills denormal creep)
    float h_ref     = 2.5f;    // m reference column for the CFL bound (= ceiling_h)

    // Plain constants for the CFL head margin (NOT tunables):
    static constexpr float P_REF    = 1.0f;  // atm — reference pressure in the bound
    static constexpr float HEAD_REF = 0.2f;  // m — W4 documented worst-case free column

    // House pattern (AtmosphereSolver::max_dt): Python owns the substep loop,
    // the solver owns the bound. Plain wave CFL at the reference depth, with a
    // margin for the head-term stiffening (W4):
    //   max_dt = 0.5 * dx / sqrt(g * h_ref * (1 + k_p * P_REF / HEAD_REF))
    // k_p = 0 -> 0.5*dx/sqrt(g*h_ref) = 33.6 ms at dx=1/3, h_ref=2.5.
    // This is a REAL CFL: linearised, the pipe model is a damped wave with
    // c = sqrt(g*depth); damping removes the wet/dry blow-up, NOT the wave CFL.
    float max_dt() const;

    void step(float* water_depth, float* flow_vx, float* flow_vy,
              const float* floor_height,          // nullable -> flat zero
              const float* atmosphere,            // nullable -> no head term
              const float* wave_p,                // nullable -> no head term
              const bool*  solid,                 // STATIC walls (gmap.solid) — units do NOT block water
              int h, int w, float dt,
              float tilt_x, float tilt_y) const;  // radians about grid centre; sane range |tilt| < ~30 deg
};
