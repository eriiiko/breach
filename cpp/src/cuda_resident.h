#pragma once
// ============================================================================
// CUDA-S8a — the shared launch-core header (residency, STEP B).
// ============================================================================
//
// docs/cuda_s8a_residency_spec_2026-07-19.md §4 STEP B. For each solver family
// we factor a NON-anonymous `*_launch_resident(...)` that takes DEVICE pointers
// + persistent scratch + scalars and does the kernel launches ONLY — no
// cudaMalloc, no cudaMemcpy (H2D/D2H), no cudaFree, no cudaDeviceSynchronize.
// Allocation and transfer are the caller's job:
//
//   * The existing per-call `*_step` wrappers keep working unchanged — they now
//     allocate + H2D, call the matching `*_launch_resident`, sync, D2H, free.
//     So the launch body is SHARED, never duplicated, and the live per-call GPU
//     path (+ the existing per-kernel gates) run the byte-for-byte same kernel
//     sequence and arithmetic as before.
//   * STEP D's `step_resident(...)` (physics_engine.cpp) owns the persistent
//     device fields (CuPy-backed, passed in as raw pointers) + persistent
//     scratch (lazily allocated, keyed by (h,w)) and calls these cores back to
//     back with ZERO mid-tick transfers.
//
// Bit-identity is the eventual gate (tol 0, Berlin): the launch cores move
// allocation/transfer OUT of the hot path; they change NO math. Kernel order,
// scalar precompute, and arithmetic are identical to the per-call path.
//
// Plain C++ declaration header (no CUDA types in the signatures — raw int32_t*/
// int64_t*/bool* device pointers) so the .cpp TUs (physics_engine.cpp,
// bindings.cpp, cl.exe-compiled) can include it. Each `*_launch_resident` is
// DEFINED in its family's .cu. Compiled only when BREACH_CUDA.
//
// NOTE: this header is being populated family-by-family across STEP B commits.
// Only the cores listed below are wired; the rest land in subsequent WIP checkpoints.
#include <cstdint>

class EOSSolver;

namespace breach_cuda {

// ---- water (cuda_water.cu) --------------------------------------------------
// One water substep, LAUNCH ONLY, in place on d_depth/d_vx/d_vy. Mirrors
// water_step's K1..K8 sequence exactly (same host scalar precompute, same
// kernels, same order). Scratch buffers (d_surface/d_dq_e/d_dq_s/d_scale =
// (h*w) int32; d_fx/d_fy = (h*w) int64) are caller-owned + persistent. No
// malloc/transfer/sync. d_floor / d_atm nullable (as in water_step).
void water_launch_resident(
    int32_t* d_depth, int32_t* d_vx, int32_t* d_vy,
    const int32_t* d_floor,   // nullable -> flat zero
    const int32_t* d_atm,     // nullable -> no head term
    const bool* d_solid,
    int32_t* d_surface, int32_t* d_dq_e, int32_t* d_dq_s, int32_t* d_scale,
    int64_t* d_fx, int64_t* d_fy,
    int h, int w, float dt, float tilt_x, float tilt_y,
    float g, float damping, float dx, float k_p, float v_max, float depth_eps);

// The whole water SUBSTEP LOOP, device-resident (S8a Path B FLOOR item 2). Runs
// water_launch_resident `n_sub` times back-to-back on the SAME persistent device
// buffers, with C++-owned persistent scratch (allocated once, keyed by (h,w) —
// NO per-substep cudaMalloc/H2D/D2H), one cudaDeviceSynchronize at the end. This
// is the exact substep loop PhysicsEngine::step_water runs, minus the transfer
// tax that today's per-call water_step pays on EVERY substep. d_depth/d_vx/d_vy
// are the persistent CuPy-owned water fields; d_floor/d_atm nullable (as in
// water_step); the caller (step_resident) owns the D2H that follows. Bit-identical
// to the per-call path (same shared launch core, same host scalar precompute).
void water_substeps_resident(
    int32_t* d_depth, int32_t* d_vx, int32_t* d_vy,
    const int32_t* d_floor, const int32_t* d_atm, const bool* d_solid,
    int h, int w, int n_sub, float wdt, float tilt_x, float tilt_y,
    float g, float damping, float dx, float k_p, float v_max, float depth_eps);

// ---- smoke (cuda_smoke.cu) --------------------------------------------------
// One smoke/trace step, LAUNCH ONLY, in place on d_gas. Mirrors smoke_step's
// K1..K4 sequence exactly (diffusion Laplacian -> diffuse apply -> D2D post-
// diffusion snapshot -> SL advect -> clamp). Scratch (d_lap/d_src = h*w int32)
// is caller-owned + persistent. No malloc/H2D/D2H/sync. d_amb nullable (space).
void smoke_launch_resident(
    int32_t* d_gas,
    const int32_t* d_wind_x, const int32_t* d_wind_y,
    const bool* d_obstacles, const bool* d_is_wall, const bool* d_is_vacuum,
    const float* d_perm, const bool* d_is_ambient,
    int32_t* d_lap, int32_t* d_src,
    int h, int w, float dt,
    float d_smoke, float wind_diffusion_scale, float advection_rate);

// The whole per-tick TRACE-PLANE LOOP, device-resident (S8a Path B FLOOR item 3).
// For each non-conservative gas plane: smoke_launch_resident (once per tick, on
// the final corrected wind) + the decay->inert_N2 credit (a device kernel,
// bit-identical to the CPU mul_q16 credit). Persistent scratch owned here (lap/
// src, keyed by (h,w)); one cudaDeviceSynchronize at the end; NO per-plane
// cudaMalloc/H2D/D2H. The all-zero-plane `.any()` skip is DROPPED — smoke_step
// on an all-zero plane is an arithmetic no-op (the EOS P6.5 device precedent), so
// processing every trace plane is bit-identical to the CPU skip. gas_conservative
// / gas_diffusion / gas_decay are the small (n_gases,) HOST columns (control-flow
// + the per-plane diffusion/decay scalars, exactly as run_substeps reads them).
void trace_smoke_resident(
    int32_t* d_gas_base,
    const int32_t* d_wind_x, const int32_t* d_wind_y,
    const bool* d_solid, const bool* d_is_vacuum, const float* d_perm,
    const bool* d_is_ambient,
    int h, int w, int n_gases, int inert_n2_idx,
    const bool* gas_conservative, const float* gas_diffusion,
    const float* gas_decay,
    float dt, float advection_rate, float wind_diffusion_scale);

}  // namespace breach_cuda
