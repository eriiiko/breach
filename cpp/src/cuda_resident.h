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
// the final corrected wind) + decay (a device kernel, bit-identical to the CPU
// mul_q16 shrink). P-T0 (energy-books arc, design §2.6 — the trace 0% ruling):
// the decay->inert_N2 credit this kernel used to pay is DELETED — decayed mass
// simply VANISHES, same as the CPU twin (physics_engine.cpp's run_substeps
// trace loop); `inert_n2_idx` stays a parameter for ABI/back-compat. Persistent
// scratch owned here (lap/src, keyed by (h,w)); one cudaDeviceSynchronize at
// the end; NO per-plane cudaMalloc/H2D/D2H. The all-zero-plane `.any()` skip is
// DROPPED — smoke_step on an all-zero plane is an arithmetic no-op (the EOS
// P6.5 device precedent), so processing every trace plane is bit-identical to
// the CPU skip. gas_conservative / gas_diffusion / gas_decay are the small
// (n_gases,) HOST columns (control-flow + the per-plane diffusion/decay
// scalars, exactly as run_substeps reads them).
void trace_smoke_resident(
    int32_t* d_gas_base,
    const int32_t* d_wind_x, const int32_t* d_wind_y,
    const bool* d_solid, const bool* d_is_vacuum, const float* d_perm,
    const bool* d_is_ambient,
    int h, int w, int n_gases, int inert_n2_idx,
    const bool* gas_conservative, const float* gas_diffusion,
    const float* gas_decay,
    float dt, float advection_rate, float wind_diffusion_scale);

// ---- kick + compression (cuda_kick_compression.cu) — S8a Path A ------------
// The per-tick scalar folds the kick/compression kernels consume, factored to
// ONE transcription (design §3.2.3): kick_scalar_folds computes them from the
// EOSSolver config floats through the IDENTICAL double expressions the CPU
// step() folds (/fp:strict host pass); consumed by BOTH the per-call
// eos_kick_compression entry and the resident launch core below.
struct KickScalarFolds {
    int32_t n_floor_q    = 0;
    int32_t t_min_q      = 0;
    int32_t t_max_phys_q = 0;
    int32_t u_max_q      = 0;
    int32_t gamma_m1_q   = 0;
    int32_t dt_q         = 0;
    int32_t inv_2dx_q    = 0;
    int32_t work_clamp_q = 0;
    int32_t absorb_dt_q  = 0;   // absorb_strength·dt (the §2.5 hoist's factor)
    int64_t Kdt_raw      = 0;   // (K·2^16)·dt at raw scale, 128-bit staged
};
KickScalarFolds kick_scalar_folds(
    float dt, float c_max, float dx, float adiabatic_index,
    float absorb_strength, float n_floor_solver, float t_min,
    float t_work_clamp, float t_max_phys, float u_max);

// The step-4 + step-4c tail, LAUNCH ONLY, on DEVICE pointers — K1 kick then
// K2 compression on one stream (the CPU pass boundary), no malloc, no
// transfer, no memset, no sync, no digest. d_ntot is the post-substep Dalton
// N_total plane; d_absorb_q the host-hoisted §2.5 absorb plane; d_cnt the
// 5-slot rail-counter buffer THE CALLER ZEROES each tick (design §3.2.5).
// d_amb/d_udamp nullable (space / no band). The per-call eos_kick_compression
// wraps this same core with its existing H2D/memset/D2H/digest flow — one
// kernel transcription, both paths.
void kick_compression_launch_resident(
    int32_t* d_wind_x, int32_t* d_wind_y, int32_t* d_temperature,
    const int32_t* d_p_new, const int32_t* d_ntot, const int32_t* d_absorb_q,
    const bool* d_solid, const bool* d_is_vacuum,
    const KickScalarFolds& folds, int32_t c_local_q,
    unsigned long long* d_cnt, int h, int w,
    const bool* d_is_ambient, const int32_t* d_sponge_udamp,
    // THERMAL-MASS AXIS, P-EOS: the medium mask K2 (compression work) skips its
    // T write on. The P2 device-fallback idiom applies — the caller passes
    // `d_thermal_solid ? d_thermal_solid : d_solid`, so the legacy path
    // allocates and copies nothing and is not a second code path. K1 (the kick)
    // never sees it: it writes u, never T.
    const bool* d_ts = nullptr);

}  // namespace breach_cuda
