#pragma once
// Wave equation solver — shockwave propagation on a 2D grid.
// Operates directly on the atmosphere (pressure) field.
// Also computes wind field (gradient of pressure) for use by smoke/fire.

class WaveSolver {
public:
    // Physical parameters (set from Python config, hot-reloadable)
    float c        = 300.0f;   // wave speed (tiles/s)
    float damping  = 3.0f;     // velocity damping (1/s)
    float feed_rate = 200.0f;  // source -> pressure feed rate (1/s)

    // Advance wave equation by sim_time seconds.
    // The wave equation operates directly on the atmosphere field.
    // Wind field (gradient) is computed and stored in wind_x/wind_y.
    // All arrays are h x w, row-major, owned by numpy.
    void step(
        float* atmosphere,    // pressure field (unified — wave + bulk)
        float* wave_v,        // velocity field (dp/dt)
        float* wave_source,   // pending pressure deposits
        float* wind_x,        // output: x-component of wind (dp/dx)
        float* wind_y,        // output: y-component of wind (dp/dy)
        const bool* obstacles,// walls + units (Neumann BC)
        const bool* is_wall,  // static walls (zero pressure)
        const bool* is_vacuum,// vacuum tiles (Dirichlet BC: p=0)
        int h, int w,
        float sim_time
    ) const;
};
