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

}  // namespace breach_cuda
