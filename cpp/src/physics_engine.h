#pragma once
// PhysicsEngine — orchestrates all physics subsystems in a single C++ tick.
// Interleaves wave and smoke substeps so smoke rides the shockwave.

#include "wave_solver.h"
#include "smoke_dynamics.h"
#include "atmo_diffusion.h"

class PhysicsEngine {
public:
    WaveSolver wave;
    SmokeDynamics smoke;
    AtmoDiffusion diffusion;

    // Run a full physics tick: interleaved wave + smoke substeps, then diffusion.
    // All arrays are h x w, row-major, owned by numpy.
    void tick(
        float* atmosphere,
        float* wave_v,
        float* wave_source,
        float* wind_x,
        float* wind_y,
        float* smoke_field,
        const bool* obstacles,
        const bool* is_wall,
        const bool* is_vacuum,
        int h, int w,
        float sim_time
    ) const;
};
