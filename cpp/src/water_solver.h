#pragma once
#include <vector>
#include <cstdint>
#include "fixed_point.h"   // q16, mul_q16, recip_mul, tan_poly, ceil_div
// WaterSolver — the pipe model: damped velocity + donor-cell upwind mass flux.
//
// Canon design: docs/architecture/engine/07_fluid_and_water.md §2.
// Build plan:   docs/water_implementation_plan.md Step W1.
// S1 (docs/s1_water_fixed_point_plan.md): the SYNCED state — water_depth,
// flow_vx, flow_vy — is now int32 Q16.16 (metres / m/s, scale 2^16). The
// transport core is pure integer (bit-identical cross-machine); the ripple stays
// float (render-only) and reads the depth dequantized.
//
// Per step (metres / m/s / seconds; the depth/velocity FIELDS are Q16.16 ints,
// the scalar params g/damping/dt/dx/k_p remain real and are pre-combined into
// Q16.16 step constants once per step() call):
//   1. surface = floor_height + tilt_offset + water_depth        (Q16.16 metres)
//                (+ k_p*(atmosphere + wave_p) — GATED + a FLOAT BRIDGE: atm/wave_p
//                 are still float; the head term is computed in float and
//                 quantized back into the Q16.16 surface — marked in the .cpp).
//   2. v += dt*(-g*grad(surface) - damping*v)  — central difference; Neumann
//      MIRROR of the centre value at solid neighbours; out-of-bounds is solid
//      (grid border = wall). v = 0 on solid; componentwise clamp to +-v_max.
//      (grad uses a precomputed reciprocal of 2*dx; the g*dt / damping*dt
//      products are Q16.16 step constants.)
//   3. Donor-cell upwind face fluxes from PRE-update depth (gather), zeroed at
//      solid faces; per-cell OUTFLOW LIMITER (a cell can be donor on up to 4
//      faces — scale ITS outgoing fluxes by depth*dx/(dt*out_sum) when
//      out_sum*dt/dx > depth, so the non-negative clamp cannot create mass); the
//      per-FACE depth-delta is gathered ONCE (a single Q16.16 value) and applied
//      +dq to one cell, -dq to the neighbour, so the >>16 narrow conserves mass
//      to the LSB (S1 P2).
//   4. depth = max(depth, 0); 0 on solid; snapped to 0 below depth_eps.
//
// Deterministic: integer transport (no float in the core), no RNG, gather-then-
// apply everywhere, fixed iteration order. Stateless: params are public members;
// step takes raw pointers (house pattern, AtmosphereSolver). Python owns the
// substep loop via max_dt() (now a Q16.16 constant — see max_dt_q()).

struct WaterSolver {
    // tunables (bound from config [physics.water]) — REAL params (config doubles
    // narrowed to float at the pybind boundary). They are combined into Q16.16
    // step constants once per step() call; the FIELDS are integer, these are not.
    float g         = 9.81f;   // m/s^2 (prototype-validated)
    float damping   = 1.0f;    // 1/s pipe friction — lifted from fluid_test.py:39 (the
                               // side-by-side run that justified the model); tune in [0.5, 1.0]
    float dx        = 0.333f;  // m — set from the level's tile_size_m, never assumed
    float k_p       = 0.0f;    // pressure head, m per pressure-unit; 0 == head OFF (W4 turns on)
    float v_max     = 8.0f;    // m/s safety clamp (safe WITH the outflow limiter)
    float depth_eps = 1e-5f;   // m snap-to-zero (kills denormal creep)
    float h_ref     = 2.5f;    // m reference column for the CFL bound (= ceiling_h)

    // W6a ripple tunables (the VISUAL-ONLY surface wave, canon §6 — it NEVER
    // feeds back into transport; step() above never reads ripple state). Ripple
    // stays FLOAT (render-only, never synced): step_ripple reads water_depth
    // DEQUANTIZED (int32 -> float, /65536) at its c2 = g*min(depth,h_cap) read.
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
    //
    // S1: max_dt is a LOAD-TIME CONSTANT (config only). The sqrt is computed ONCE
    // on the CPU in double (IEEE sqrt is correctly-rounded -> bit-identical across
    // machines) and quantized to a Q16.16 constant via max_dt_q(). The float
    // max_dt() stays (it is what Python's CFL display / older callers read) but
    // the SUBSTEP-COUNT CLIFF uses max_dt_q() + fixedpoint::ceil_div (no float).
    float max_dt() const;
    // The same CFL bound as a Q16.16 constant (seconds). Used by the integer
    // substep-count derivation (n = ceil_div(sim_time_q, max_dt_q)).
    q16   max_dt_q() const;

    // The integer transport step. The synced fields are Q16.16 int32:
    //   water_depth : metres, CONSERVED.
    //   flow_vx/vy  : m/s, persistent velocity.
    // floor_height is also Q16.16 metres (quantized at load). atmosphere/wave_p
    // stay FLOAT (the S2 group) — read only through the gated head-term FLOAT
    // BRIDGE. tilt_x/tilt_y are real radians (the tan poly runs internally).
    void step(q16* water_depth, q16* flow_vx, q16* flow_vy,
              const q16* floor_height,            // nullable -> flat zero (Q16.16 m)
              const float* atmosphere,            // nullable -> no head term (FLOAT BRIDGE)
              const float* wave_p,                // nullable -> no head term (FLOAT BRIDGE)
              const bool*  solid,                 // STATIC walls (gmap.solid) — units do NOT block water
              int h, int w, float dt,
              float tilt_x, float tilt_y) const;  // radians about grid centre; clamped internally

    // --- W6a: the ripple field (canon §6, plan W6a) — STAYS FLOAT ----------
    // STATIC CFL bound for step_ripple at the deep-water cap: the ripple's
    // wave speed is c = sqrt(g*min(depth, h_cap)) <= sqrt(g*h_cap), so
    //   ripple_max_dt = 0.5*dx/sqrt(g*h_cap)   (~106 ms at dx = 1/3)
    // — derived once, far above any tick we use: ONE step_ripple call per
    // tick, no substep loop. RENDER-ONLY -> stays float (no determinism need).
    float ripple_max_dt() const;

    // Advance the visual-only ripple displacement field by one damped
    // kick-drift wave step (canon §6's stencil). RENDER-ONLY -> FLOAT. The
    // depth coupling (c2 = g*min(depth, h_cap)) reads water_depth DEQUANTIZED
    // (Q16.16 int32 -> float /65536) at that read — the only place the float
    // ripple touches the integer depth. Per call:
    //   1. splash source: ripple_v += k_splash*wave_p where depth > 0
    //      (wave_p nullable -> no splash, never read — W1's discipline)
    //   2. kick:  ripple_v += dt*(c2*lap(ripple) - gamma_r*ripple_v),
    //      c2 = g*min(dequantize(depth), h_cap); laplacian from PRE-update
    //      ripple (gather-then-apply); Neumann mirror at solid/out-of-bounds
    //   3. drift: ripple += dt*ripple_v; THEN clamp |ripple| <= k_amp*depth
    //   4. ripple = ripple_v = 0 where depth == 0 or solid
    // Reads water_depth (Q16.16) / wave_p / solid as CONST. Deterministic but
    // NOT synced (float, render-only).
    void step_ripple(float* ripple, float* ripple_v,
                     const q16*  water_depth,       // Q16.16 m (dequantized on read)
                     const float* wave_p,           // nullable -> no splash source
                     const bool*  solid,
                     int h, int w, float dt) const;

private:
    // Reused per-step scratch for step() (GPU-prep: no per-step alloc). `mutable`
    // so const step() can use them. Now INTEGER (Q16.16) buffers for the
    // transport core; surface_ is Q16.16 metres, dq_e_/dq_s_ are the per-face
    // Q16.16 depth-deltas (gathered once, applied +/- -> conservation), scale_q_
    // is the per-cell Q16.16 outflow-limiter factor (FP_ONE == 1.0 unlimited).
    //   surface_   — surface potential (Q16.16 m); FULLY overwritten each step.
    //   fx_, fy_   — face fluxes as WIDE int64 (Q32.32, = mul_wide(v_face,depth));
    //                init 0, only non-solid/non-border faces written, the 0s read.
    //   dq_e_, dq_s_ — per-face Q16.16 depth-delta (dt/dx * scaled flux), the
    //                CONSERVATIVE unit applied +/- in the divergence pass.
    //   scale_q_   — outflow limiter factor in Q16.16 (FP_ONE default IS read).
    mutable std::vector<q16>      surface_;
    mutable std::vector<int64_t>  fx_;
    mutable std::vector<int64_t>  fy_;
    mutable std::vector<q16>      dq_e_;
    mutable std::vector<q16>      dq_s_;
    mutable std::vector<q16>      scale_q_;
};
