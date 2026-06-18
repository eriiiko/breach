#pragma once
// AtmosphereSolver — IMEX scheme: explicit wave + implicit diffusion.
//
// Two-field architecture:
//   wave_p  = acoustic pressure (zero-mean shockwave component)
//   atmosphere = bulk air pressure (slow diffusion/decompression)
//
// Per substep:
//   1. Feed wave_source → wave_p
//   2. Explicit wave kick: v += dt*(c²*Δ(wave_p) - γ*v), wave_p += dt*v
//   3. Transfer wave anomaly into atmosphere
//   4. Compute u* = atmosphere (pre-diffusion target)
//   5. Implicit diffusion: solve (I - D*dt*Δ) atm_new = u*  [Gauss-Seidel]
//   6. Boundary conditions (relaxation at vacuum, sponge layer)
//   7. Wind = gradient of (atmosphere + wave_p)
//
// The implicit diffusion is unconditionally stable:
//   amplification factor = 1/(1 + μσ), always in [0,1] for μ≥0.
//   No CFL limit for diffusion. Only wave CFL matters.
//
// See docs/atmosphere_solver_analysis_and_patch_plan_20260319.md §8.

#include <cstdint>
#include <vector>

class AtmosphereSolver {
public:
    // Wave parameters
    float c          = 300.0f;   // wave speed (tiles/s)
    float damping    = 3.0f;     // wave velocity damping (1/s)
    float absorb_strength = 8.0f;// global scale on per-cell wave_absorb damping (4a)
    float transfer   = 0.5f;    // wave_p → atmosphere transfer rate (1/s)
    float feed_rate  = 200.0f;  // wave_source → wave_p feed rate (1/s)

    // Diffusion parameters
    float d_atm      = 50.0f;   // atmosphere diffusion coefficient
    int   gs_iters   = 8;       // Gauss-Seidel iterations for implicit diffusion

    // Boundary parameters
    float breach_rate = 5.0f;   // relaxation rate toward vacuum (1/s)

    // Source injection parameters
    float max_source_per_step = 0.5f;  // max wave_source fed per substep

    // Compute the maximum stable dt (wave CFL only — diffusion is implicit).
    float max_dt() const;

    // Advance ONE timestep of size dt.
    // Updates all fields in-place. Writes wind_x, wind_y.
    void step(
        float* wave_p,
        float* wave_v,
        float* wave_source,
        float* atmosphere,
        float* wind_x,
        float* wind_y,
        const bool* obstacles,
        const bool* is_wall,
        const bool* is_vacuum,
        const float* permeability,
        const float* wave_absorb,
        int h, int w,
        float dt
    ) const;

private:
    // Reused per-step scratch (GPU-prep: no per-step alloc; on CUDA a per-step
    // std::vector becomes a per-step cudaMalloc). Resized once, reused; `mutable`
    // so the const step() can use them as pure scratch (temperature_solver idiom).
    // Accessed in step() through `T* __restrict` locals — these buffers alias no
    // field pointer, so __restrict restores the fresh-local no-alias property
    // /fp:fast needs for bit-identical codegen.
    //   lap_      — wave Laplacian; FULLY overwritten each step before read.
    //   rhs_      — implicit-diffusion RHS; FULLY overwritten (copy of atmosphere)
    //               before read, inside the mu-gated block.
    //   vac_dist_ — sponge distance; default 255 IS read (cells that never reach
    //               0/1/2 stay 255), so it MUST be re-filled to 255 each step.
    mutable std::vector<float>   lap_;
    mutable std::vector<float>   rhs_;
    mutable std::vector<uint8_t> vac_dist_;
};
