#pragma once
// ============================================================================
// CUDA-S5 — the explicit damped-wave shockwave step on the GPU (bit-identical).
// ============================================================================
//
// A faithful, bit-identical GPU port of AtmosphereSolver::wave_substep
// (atmosphere_solver.cpp ~62-261): the explicit damped-wave shockwave substep —
//   K1 feed source        wave_p += feed; wave_source -= feed   (rate-limited)
//   K2 Laplacian gather    lap = narrow(Σ4 mul_wide(perm_face, wp[n]-wp))
//   K3 velocity kick       wave_v += narrow(c_sq_dt*lap - damp_dt*wave_v)
//   K4 pressure update     wave_p += mul_q16(wave_v, dt_q)
//   K5 absorption          k=(a<1)?1-a:0; wave_v=scale_mag(.,k); wave_p=scale_mag(.,k)
//   K6 wave BCs            =0 on obstacle|wall|vacuum
//   K7 mean_wp reduction   int64 atomicAdd of wave_p over the interior mask
//   K8 anomaly transfer    atmosphere += round_nearest((wave_p-mean_wp)*xfer_q)
//
// The synced fields wave_p, wave_v, wave_source (int32 Q16.16) AND atmosphere
// (modified by the one-sided anomaly transfer) come out byte-for-byte identical
// to the CPU wave_substep on every architecture — the point of S5. The
// determinism crux is K7: an int64 atomicAdd accumulator (integer + is order-free
// -> bit-identical regardless of thread/scheduler order, unlike a float atomicAdd)
// with the mean_round computed ON THE HOST after read-back (the exact CPU form).
//
// SCOPE: wave_substep ONLY. diffuse_solve (the implicit RB-GS + vacuum sponge +
// WIND gradient) is S7 and stays on the CPU — so this entry does NOT produce
// wind_x/wind_y. The integration gate stays valid (GS identical on the CPU in
// both wave-backend paths).
//
// Plain C++ declaration header (no CUDA types) so the .cpp TUs (bindings.cpp,
// physics_engine.cpp, compiled by cl.exe even in the CUDA build) can include it;
// cuda_wave.cu provides the definitions. Compiled only when BREACH_CUDA.
#include <cstdint>

namespace breach_cuda {

// ONE wave substep on the GPU — IN-PLACE on wave_p/wave_v/wave_source/atmosphere
// (h,w). Mirrors AtmosphereSolver::wave_substep exactly (the 8 passes above incl.
// the mean_wp reduction + the one-sided anomaly transfer). Because this is a FREE
// function (not a method on the solver), the solver's scalar dials are passed
// explicitly:
//   c                   — wave speed (tiles/s); c_sq = c*c folds into the kick.
//   damping             — wave velocity damping (1/s).
//   absorb_strength     — global scale on the per-cell wave_absorb damping.
//   transfer            — wave_p -> atmosphere transfer rate (1/s).
//   feed_rate           — wave_source -> wave_p feed rate (1/s).
//   max_source_per_step — per-substep cap on the source feed.
//   dt                  — the per-substep real dt (the wave CFL dt_actual).
// All quantized step constants are precomputed ON THE HOST in double, VERBATIM
// from the CPU top-of-function, and passed as scalar kernel args.
// permeability / wave_absorb stay FLOAT (the per-face / per-cell float bridges,
// quantized on the device like S4a's neighbor_q). obstacles/is_wall/is_vacuum are
// the static masks.
//
// PERF NOTE (residency is S8): per-call H2D of all fields + masks + perm/absorb
// and a D2H of the 4 mutated fields, n_wave× per tick — deliberately deferred.
void wave_substep_gpu(
    int32_t* wave_p,           // Q16.16 (h,w) — in/out (acoustic anomaly, signed)
    int32_t* wave_v,           // Q16.16 (h,w) — in/out (wave velocity, signed)
    int32_t* wave_source,      // Q16.16 (h,w) — in/out (injected energy, >= 0)
    int32_t* atmosphere,       // Q16.16 (h,w) — in/out (one-sided anomaly transfer)
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    const float* permeability, // FLOAT (h,w) per-face permeability bridge
    const float* wave_absorb,  // FLOAT (h,w) per-cell absorb bridge
    int h, int w, float dt,
    float c, float damping, float absorb_strength,
    float transfer, float feed_rate, float max_source_per_step);

// Backend selection (S5 gate + integration). When true, PhysicsEngine::
// run_substeps runs each wave substep on the GPU instead of the CPU
// AtmosphereSolver::wave_substep. Defaults false so the game + suite run on the
// CPU path unchanged until explicitly switched. diffuse_solve stays CPU either
// way (S7 is the GS port).
bool wave_backend_is_cuda();
void set_wave_backend_cuda(bool on);

}  // namespace breach_cuda
