#pragma once
// Smoke dynamics — diffusion + advection by precomputed wind field.
// Wind is computed by the AtmosphereSolver (gradient of atmosphere + wave_p).

class SmokeDynamics {
public:
    float d_smoke           = 0.4f;   // smoke diffusion coefficient
    float advection_rate    = 25.0f;  // advection strength by wind field

    // Single step of size dt. Uses precomputed wind field from atmosphere solver.
    void step(
        float* smoke,
        const float* wind_x,       // precomputed: d/dx(atmosphere + wave_p)
        const float* wind_y,       // precomputed: d/dy(atmosphere + wave_p)
        const bool* obstacles,
        const bool* is_wall,
        const bool* is_vacuum,
        int h, int w,
        float dt
    ) const;
};
