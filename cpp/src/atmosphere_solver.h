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

    // --- Patch 2a: GS-residual diagnostic (read-only) --------------------
    // Linf norm of the implicit-operator residual (I - μΔ)atm - rhs over the
    // non-obstacle interior, normalized by max|atm|, measured INSIDE
    // diffuse_solve AFTER the GS sweeps but BEFORE the vacuum/sponge BC pass
    // (the BC pass mutates atmosphere post-solve and would contaminate it).
    // Answers "do gs_iters sweeps under-relax at this dt?". Nothing reads it
    // yet — pure instrumentation. Default 0 until the first diffuse_solve.
    // `mutable` so the const diffuse_solve()/step() can write it (it is pure
    // diagnostic state, not a solver output).
    mutable float last_gs_residual = 0.0f;
    float gs_residual() const { return last_gs_residual; }

    // Advance ONE timestep of size dt.
    // Updates all fields in-place. Writes wind_x, wind_y.
    //
    // Patch 2a: step() is now wave_substep() followed by diffuse_solve() —
    // kept as the single-substep convenience entry (and what the conservation
    // test drives). The engine's run_substeps splits these so the wave loops
    // at its CFL while the implicit diffusion solves ONCE per tick.
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

    // --- Patch 2a: the explicit-wave sub-steps (1-3) ----------------------
    // Feed wave_source -> wave_p, the explicit wave kick (+ per-cell absorb +
    // wave wall/vacuum BCs), then transfer the wave anomaly into atmosphere.
    // Runs `n_wave` times at the wave CFL dt. Mutates wave_p/wave_v/wave_source
    // and accumulates the per-substep anomaly transfer onto atmosphere.
    void wave_substep(
        float* wave_p,
        float* wave_v,
        float* wave_source,
        float* atmosphere,
        const bool* obstacles,
        const bool* is_wall,
        const bool* is_vacuum,
        const float* permeability,
        const float* wave_absorb,
        int h, int w,
        float dt
    ) const;

    // --- Patch 2a: the implicit diffusion + BCs + wind (4-7) --------------
    // u* = atmosphere, implicit Gauss-Seidel diffusion (μ = d_atm·dt; here dt
    // is the FULL tick sim_time — runs ONCE per tick, unconditionally stable),
    // the vacuum/sponge BC pass, then wind = -grad(atmosphere + wave_p).
    // Measures last_gs_residual after the sweeps, before the BC pass.
    void diffuse_solve(
        float* atmosphere,
        float* wave_p,
        float* wave_v,
        float* wave_source,
        float* wind_x,
        float* wind_y,
        const bool* obstacles,
        const bool* is_wall,
        const bool* is_vacuum,
        const float* permeability,
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
