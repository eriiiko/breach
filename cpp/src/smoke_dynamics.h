#pragma once
// Smoke dynamics — diffusion + advection by precomputed wind field.
// Wind is computed by the AtmosphereSolver (gradient of atmosphere + wave_p).

class SmokeDynamics {
public:
    float d_smoke             = 0.4f;   // base smoke diffusion coefficient
    float advection_rate      = 25.0f;  // advection strength by wind field
    float dt_scale            = 1.0f;   // time multiplier (>1 = smoke reacts faster)
    float wind_diffusion_scale = 0.0f;  // wind-dependent diffusion: D = d_smoke * (1 + scale * |wind|)

    // Single step of size dt. Uses precomputed wind field from atmosphere solver.
    void step(
        float* smoke,
        const float* wind_x,       // precomputed: -d/dx(atmosphere + wave_p)
        const float* wind_y,       // precomputed: -d/dy(atmosphere + wave_p)
        const bool* obstacles,
        const bool* is_wall,
        const bool* is_vacuum,
        const float* permeability,
        int h, int w,
        float dt
    ) const;
};
