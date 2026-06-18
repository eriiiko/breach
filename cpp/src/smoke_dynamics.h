#pragma once
// Smoke dynamics — diffusion + advection by precomputed wind field.
// Wind is computed by the AtmosphereSolver (gradient of atmosphere + wave_p).

#include <vector>

class SmokeDynamics {
public:
    float d_smoke             = 0.4f;   // base smoke diffusion coefficient
    float advection_rate      = 25.0f;  // advection strength by wind field
    float dt_scale            = 1.0f;   // time multiplier (>1 = smoke reacts faster)
    float wind_diffusion_scale = 0.0f;  // wind-dependent diffusion: D = d_smoke * (1 + scale * |wind|)
    float sink_strength       = 0.0f;   // smoke-side sink-pull toward nearest breach (0 = off)

    // Single step of size dt. Uses precomputed wind field from atmosphere solver.
    void step(
        float* smoke,
        const float* wind_x,       // precomputed: -d/dx(atmosphere + wave_p)
        const float* wind_y,       // precomputed: -d/dy(atmosphere + wave_p)
        const float* sink_x,       // smoke-side sink direction toward nearest breach (unit-ish, 0 if none)
        const float* sink_y,
        const bool* obstacles,
        const bool* is_wall,
        const bool* is_vacuum,
        const float* permeability,
        int h, int w,
        float dt
    ) const;

private:
    // Reused per-step scratch (GPU-prep: no per-step alloc). `mutable` so the
    // const step() can use them (temperature_solver idiom).
    //   lap_ — diffusion Laplacian; FULLY overwritten each step before read,
    //          read in a pure-FP loop -> accessed via `float* __restrict` (it
    //          aliases no field pointer; restores fresh-local /fp:fast codegen).
    //   src_ — pre-advection snapshot (was a COPY of smoke). It is read by the
    //          BRANCHY advection loop that writes smoke; a member pointer there
    //          shifts the float codegen, so it uses the SWAP idiom (stays a
    //          genuine local, storage retained) and is re-copied from smoke.
    mutable std::vector<float> lap_;
    mutable std::vector<float> src_;
};
