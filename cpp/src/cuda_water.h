#pragma once
// ============================================================================
// CUDA-S3 — the water solver on the GPU (the pipe model, bit-identical).
// ============================================================================
//
// A faithful, bit-identical port of WaterSolver::step (water_solver.cpp): the
// pipe model — surface potential (§1) -> damped explicit velocity kick (§2,
// central difference, Neumann mirror) -> donor-cell upwind face fluxes (§3) ->
// per-face depth-delta dq (flux_to_dq) -> per-cell OUTFLOW LIMITER + the
// magnitude-first face scale -> divergence apply -> clamps (§4). The three
// SYNCED fields water_depth / flow_vx / flow_vy (int32 Q16.16) come out
// byte-for-byte identical to the CPU on every architecture — the whole point
// of S3.
//
// The transport core is pure-integer Q16.16; the only non-trivially-integer
// pieces are (1) the host-side scalar precompute, replicated in double exactly
// as the CPU does (so make_recip stays host-only); (2) the per-tile tilt
// `quantize(((double)x - cx) * dx)` which runs in double ON the device, made
// bit-identical by --fmad=false (no FMA contraction); and (3) the head FLOAT
// BRIDGE (k_p != 0) reading the host-dequantized atmosphere/wave_p in float.
// recip_mul + flux_to_dq use a 128-bit intermediate (nvcc takes the __int128
// branch); scale_mag (NOT mul_q16) does the per-face outflow scaling.
//
// Plain C++ declaration header (no CUDA types) so the .cpp TUs (bindings.cpp,
// physics_engine.cpp, compiled by cl.exe even in the CUDA build) can include
// it; cuda_water.cu provides the definitions. Compiled only when BREACH_CUDA.
#include <cstdint>

namespace breach_cuda {

// One water substep on the GPU — IN-PLACE on water_depth / flow_vx / flow_vy.
// Mirrors WaterSolver::step exactly. Because this is a FREE function (not a
// method on the solver), the solver's scalar dials (g, damping, dx, k_p,
// v_max, depth_eps) are passed explicitly alongside dt + tilt_x/tilt_y + h/w.
// The Q16.16 fields are int32_t (matching cuda_temperature.h's convention so
// the .cpp TUs need no fixed_point include). atmosphere / wave_p are the FLOAT
// head bridge (nullable -> no head term); floor_height nullable -> flat zero.
void water_step(
    int32_t* water_depth,        // Q16.16 (h,w) — in/out, CONSERVED
    int32_t* flow_vx,            // Q16.16 (h,w) — in/out, persistent velocity
    int32_t* flow_vy,            // Q16.16 (h,w) — in/out, persistent velocity
    const int32_t* floor_height, // Q16.16 (h,w) — nullable -> flat zero
    const float* atmosphere,     // FLOAT (h,w) — head bridge, nullable
    const float* wave_p,         // FLOAT (h,w) — head bridge, nullable
    const bool* solid,           // (h,w) static walls
    int h, int w, float dt,
    float tilt_x, float tilt_y,  // radians about grid centre (clamped on host)
    float g, float damping, float dx, float k_p, float v_max, float depth_eps);

// Backend selection (S3 gate + integration). When true, PhysicsEngine::
// step_water runs the per-substep water solver on the GPU instead of the CPU
// solver. Defaults false so the game + suite run on the CPU path unchanged
// until explicitly switched.
bool water_backend_is_cuda();
void set_water_backend_cuda(bool on);

}  // namespace breach_cuda
