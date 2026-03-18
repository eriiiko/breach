#pragma once
// Smoke dynamics — diffusion + advection by precomputed wind field.

class SmokeDynamics {
public:
    float d_smoke           = 0.4f;   // smoke diffusion coefficient
    float advection_rate    = 25.0f;  // advection strength by wind field

    void step(
        float* smoke,
        const float* wind_x,      // precomputed wind field (from wave solver)
        const float* wind_y,
        const bool* obstacles,
        const bool* is_wall,
        const bool* is_vacuum,
        int h, int w,
        float dt
    ) const;
};
