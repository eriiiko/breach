#pragma once
// Smoke dynamics — diffusion + advection by atmosphere and wave gradients.

class SmokeDynamics {
public:
    float d_smoke           = 0.4f;   // smoke diffusion coefficient
    float advection_rate    = 25.0f;  // advection by atmosphere gradient
    float wave_advection    = 80.0f;  // advection by wave pressure gradient

    void step(
        float* smoke,
        const float* atmosphere,
        const float* wave_p,
        const bool* obstacles,
        const bool* is_wall,
        const bool* is_vacuum,
        int h, int w,
        float dt
    ) const;
};
