#pragma once
#include <vector>
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

    // W6a ripple tunables (the VISUAL-ONLY surface wave, canon §6 — it NEVER
    // feeds back into transport; step() above never reads ripple state):
    float gamma_r   = 2.0f;    // 1/s ripple damping (gamma in v += dt*(c2*lap - gamma*v))
    float h_cap     = 0.25f;   // m deep-water cap: c2 = g*min(depth, h_cap) (lambda/2pi splice)
    float k_amp     = 0.5f;    // amplitude clamp |ripple| <= k_amp*depth (waves no taller than the water)
    float k_splash  = 2.0f;    // wave_p -> ripple_v splash gain — a PURE FEEL dial (Erik's eyeball, W6b)

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

    // --- W6a: the ripple field (canon §6, plan W6a) ----------------------
    // STATIC CFL bound for step_ripple at the deep-water cap: the ripple's
    // wave speed is c = sqrt(g*min(depth, h_cap)) <= sqrt(g*h_cap), so
    //   ripple_max_dt = 0.5*dx/sqrt(g*h_cap)   (~106 ms at dx = 1/3)
    // — derived once, far above any tick we use: ONE step_ripple call per
    // tick, no substep loop (unlike max_dt() above, which substeps divide).
    float ripple_max_dt() const;

    // Advance the visual-only ripple displacement field by one damped
    // kick-drift wave step (canon §6's stencil; same scheme family as the
    // atmosphere wave, but in SI units — c2 is m^2/s^2, the laplacian
    // carries the REQUIRED 1/dx^2). Per call:
    //   1. splash source: ripple_v += k_splash*wave_p where depth > 0
    //      (wave_p nullable -> no splash, never read — W1's discipline)
    //   2. kick:  ripple_v += dt*(c2*lap(ripple) - gamma_r*ripple_v),
    //      c2 = g*min(depth, h_cap); laplacian from the PRE-update ripple
    //      (gather-then-apply); Neumann mirror at solid/out-of-bounds —
    //      DRY neighbours read as-is (ripple 0 there: the absorbing shore)
    //   3. drift: ripple += dt*ripple_v; THEN clamp |ripple| <= k_amp*depth
    //      (gamma_r eats clamp-injected energy)
    //   4. ripple = ripple_v = 0 where depth == 0 or solid
    // Reads water_depth/wave_p/solid as CONST — the locked canon rule:
    // the ripple NEVER feeds back into transport. Deterministic: no RNG,
    // fixed iteration order.
    void step_ripple(float* ripple, float* ripple_v,
                     const float* water_depth,
                     const float* wave_p,           // nullable -> no splash source
                     const bool*  solid,
                     int h, int w, float dt) const;

private:
    // Reused per-step scratch for step() (GPU-prep: no per-step alloc; on CUDA a
    // per-step std::vector becomes a per-step cudaMalloc). `mutable` so const
    // step() can use them. step_ripple has no per-step vector — nothing there.
    //
    // ALL are accessed in step() through the SWAP idiom (each is read by a
    // BRANCHY float loop, and fx_/fy_/scale_ are also self-written across
    // iterations; under /fp:fast a member pointer/ref shifts the float rounding
    // and an __restrict promise would miscompile the self-aliasing). The swap
    // keeps them genuine LOCAL std::vectors inside step() (bit-identical
    // codegen) while RETAINING their allocation across steps.
    //   zeros_scratch_ — lazy all-zeros stand-in for nullable fields; stays zero
    //                    across steps (re-assigned zeros only on size change).
    //   surface_       — surface potential; FULLY overwritten each step.
    //   fx_, fy_       — face fluxes; init 0 but only non-solid/non-border faces
    //                    are written and the remaining 0s ARE read later -> the
    //                    swap re-assigns 0 each step.
    //   scale_         — outflow limiter; default 1.0 IS read in the scale pass
    //                    -> the swap re-assigns 1.0 each step.
    mutable std::vector<float> zeros_scratch_;
    mutable std::vector<float> surface_;
    mutable std::vector<float> fx_;
    mutable std::vector<float> fy_;
    mutable std::vector<float> scale_;
};
