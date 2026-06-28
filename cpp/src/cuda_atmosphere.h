#pragma once
// ============================================================================
// CUDA-S7 — AtmosphereSolver::diffuse_solve on the GPU (bit-identical).
// ============================================================================
//
// A faithful, bit-identical GPU port of AtmosphereSolver::diffuse_solve
// (atmosphere_solver.cpp ~269-601): the once-per-tick IMPLICIT atmosphere step —
// the Red-Black Gauss-Seidel pressure relaxation (residual form, per-cell Dinv),
// the vacuum BFS + sponge boundary pass, and the wind gradient. The LAST + hardest
// solver of the CUDA arc.
//
//   [μ-gate]  HOST: if (mu_q > MU_EPS_Q) run K0 + the GS; else skip them
//   K0  Dinv+RHS    rhs=atm; Dinv = reciprocal_q16(1 + mu*wsum); sentinel 0 on solid
//   K_GS red/black  for iter in 8: launch RED then BLACK over (x+y)&1==color cells
//                   acc=Σ4 mul_wide(mul_q16(mu_q,face_q), atm[nb]-ai); flux=narrow;
//                   resi=flux-(ai-rhs); inc=round_nearest((int64)resi*Dinv); atm+=inc
//   K_BFS1/K_BFS2   vacuum distance 1 then 2 (double-buffered, order-free gather)
//   K_SPONGE        per vac_dist tier: scale atmosphere/wave_v, zero wave_p/wave_v/
//                   wave_source per the CPU 0/solid/1/2 cases
//   K_WIND          p_total=atm+wave_p; wind = -shr_round0(grad(p_total), 1)
//
// The synced fields atmosphere, wave_p, wave_v, wave_source, wind_x, wind_y (int32
// Q16.16) come out byte-for-byte identical to the CPU diffuse_solve on every
// architecture (tol 0) — the point of S7. With S5 (wave) + this, the whole
// atmosphere/wave system is GPU.
//
// DETERMINISM crux:
//   * the GS increment uses round_nearest_q_dev (sign-symmetric — a toward-(-inf)
//     slip biases EVERY cell's relaxation by up to -1 LSB/sweep = a DC mass sink);
//   * the Red-Black schedule is TWO separate launches per iter (RED reads only the
//     frozen BLACK, and vice-versa) — order-free by colour;
//   * the BFS is double-buffered (each pass reads only the prior frozen level);
//   * Dinv is RECOMPUTED unconditionally (the CPU dinv_key_ cache is dropped — a
//     pure-function-of-inputs recompute is GPU-clean and identical);
//   * the perm bridge (quantize((double)min(perm_i,perm_n))) is a device double
//     cast made bit-identical by --fmad=false / /fp:strict (the S5/S4 idiom).
//
// last_gs_residual is a HOST-side float diagnostic (NOT synced, NOT in the digest);
// NOTHING reads it (verified: only the def_readonly binding + docs). The GPU entry
// does NOT compute it — it leaves the member as the CPU last set (default 0).
//
// Plain C++ declaration header (no CUDA types) so the .cpp TUs (bindings.cpp,
// physics_engine.cpp, compiled by cl.exe even in the CUDA build) can include it;
// cuda_atmosphere.cu provides the definitions. Compiled only when BREACH_CUDA.
#include <cstdint>

namespace breach_cuda {

// ONE AtmosphereSolver::diffuse_solve on the GPU — IN-PLACE on atmosphere /
// wave_p / wave_v / wave_source / wind_x / wind_y (h,w). Mirrors diffuse_solve
// exactly (the passes above). Because this is a FREE function (not a method on the
// solver), the solver's scalar dials are passed explicitly:
//   d_atm       — atmosphere diffusion coefficient (mu = d_atm*dt).
//   breach_rate — vacuum relaxation rate (1/s); the sponge factors fold it.
//   gs_iters    — Gauss-Seidel iterations (the CPU member, =8).
//   dt          — the FULL tick sim_time (NOT a substep dt; the implicit GS is
//                 unconditionally stable so it runs once per tick at the big dt).
// All quantized step constants are precomputed ON THE HOST in double, VERBATIM
// from the CPU top-of-function, and passed as scalar kernel args.
//
// permeability stays FLOAT (the per-face perm bridge, quantized on the device like
// S5's wave_lap). obstacles/is_wall/is_vacuum are the static masks.
//
// PERF NOTE (residency is S8): per-call H2D of the 6 mutated fields + masks + perm
// + the device scratch (d_rhs, d_dinv, d_vacdist double-buffer), and a D2H of the 6
// mutated fields, once per tick — deliberately deferred.
void diffuse_solve_gpu(
    int32_t* atmosphere,       // Q16.16 (h,w) — in/out (RB-GS + sponge sink)
    int32_t* wave_p,           // Q16.16 (h,w) — in/out (zeroed in the sponge; read for wind)
    int32_t* wave_v,           // Q16.16 (h,w) — in/out (scaled/zeroed in the sponge)
    int32_t* wave_source,      // Q16.16 (h,w) — in/out (clamped in the sponge)
    int32_t* wind_x,           // Q16.16 (h,w) — out (= -grad(atm+wave_p))
    int32_t* wind_y,           // Q16.16 (h,w) — out
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    const float* permeability, // FLOAT (h,w) per-face permeability bridge
    int h, int w, float dt,
    float d_atm, float breach_rate, int gs_iters);

// Backend selection (S7 gate + integration). When true, PhysicsEngine::
// run_substeps runs diffuse_solve on the GPU instead of the CPU
// AtmosphereSolver::diffuse_solve. Defaults false so the game + suite run on the
// CPU path unchanged until explicitly switched.
bool atmos_backend_is_cuda();
void set_atmos_backend_cuda(bool on);

}  // namespace breach_cuda
