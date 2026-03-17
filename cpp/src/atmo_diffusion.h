#pragma once
// Atmosphere diffusion solver — pressure equalization on a 2D grid.
// Simple explicit Euler with Neumann BCs at obstacles.

class AtmoDiffusion {
public:
    // Physical parameters (set from Python config, hot-reloadable)
    float d_atm = 50.0f;  // diffusion coefficient

    // Advance diffusion by sim_time seconds.
    void step(
        float* atmosphere,
        const bool* obstacles,
        const bool* is_wall,
        const bool* is_vacuum,
        int h, int w,
        float sim_time
    ) const;
};
