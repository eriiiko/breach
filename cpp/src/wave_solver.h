#pragma once
// Wave equation solver — shockwave propagation on a 2D grid.
// Operates directly on numpy arrays via raw pointers (zero-copy).

class WaveSolver {
public:
    // Physical parameters (set from Python config, hot-reloadable)
    float c        = 300.0f;   // wave speed (tiles/s)
    float damping  = 3.0f;     // velocity damping (1/s)
    float transfer = 0.5f;     // wave -> atmosphere coupling (1/s)
    float feed_rate = 200.0f;  // source -> wave_p feed rate (1/s)

    // Advance wave equation by sim_time seconds.
    // All arrays are h x w, row-major, owned by numpy.
    void step(
        float* wave_p,        // pressure field
        float* wave_v,        // velocity field
        float* wave_source,   // pending pressure deposits
        float* atmosphere,    // atmosphere (receives wave energy)
        const bool* obstacles,// walls + units (Neumann BC)
        const bool* is_wall,  // static walls (zero pressure)
        const bool* is_vacuum,// vacuum tiles (zero pressure)
        int h, int w,
        float sim_time
    ) const;
};
