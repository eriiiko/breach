#pragma once
// ============================================================================
// CUDA-S4a — the smoke transport solver on the GPU (bit-identical).
// ============================================================================
//
// A faithful, bit-identical GPU port of SmokeDynamics::step (smoke_dynamics.cpp):
// per-gas smoke transport on the precomputed wind field — wind-coupled diffusion
// (the permeability-weighted 4-neighbour Laplacian), then the INTEGER
// semi-Lagrangian wind advection (the sqrt-free DDA wall-clip march + the integer
// bilinear with the Newton-reciprocal renorm), then the clamp/zero-on-wall pass.
// The synced `gas` field (int32 Q16.16) comes out byte-for-byte identical to the
// CPU on every architecture — the point of S4a.
//
// NOT ported: SmokeDynamics::sink_hop (the breach-pull) — that is S4b. It reuses
// this same advection machinery but is a separate, smaller patch; with the smoke
// backend on, step() runs on the GPU and sink_hop() stays on the CPU in BOTH the
// reference and test paths, so the integration gate remains valid.
//
// The transport core is pure-integer Q16.16; the only float is (1) the host-side
// scalar precompute (dt_adv_q, replicated in double exactly as the CPU does) and
// (2) the per-cell DOUBLE wind^2 -> d_eff fold in the diffusion-apply kernel, made
// bit-identical to the CPU /fp:strict path by --fmad=false (no FMA contraction).
// The bilinear renorm uses reciprocal_q16_dev (a verbatim device port of the host
// Newton reciprocal). The permeability is a per-face float bridge (quantized per
// face exactly like the CPU's neighbor_q).
//
// Plain C++ declaration header (no CUDA types) so the .cpp TUs (bindings.cpp,
// physics_engine.cpp, compiled by cl.exe even in the CUDA build) can include it;
// cuda_smoke.cu provides the definitions. Compiled only when BREACH_CUDA.
#include <cstdint>

namespace breach_cuda {

// One smoke substep for ONE gas plane on the GPU — IN-PLACE on `smoke` (h,w).
// Mirrors SmokeDynamics::step exactly (diffusion -> SL advection -> clamp/zero).
// Because this is a FREE function (not a method on the solver), the solver's
// scalar dials are passed explicitly:
//   d_smoke              — this gas plane's base diffusion (set per-gas by the
//                          multi-gas dispatch; the engine writes smoke.d_smoke =
//                          gas_diffusion[gi] before each plane).
//   wind_diffusion_scale — the wind-coupled diffusion gain.
//   advection_rate       — the wind advection strength.
//   dt                   — the per-substep real dt (sim_time / n_smoke).
// wind_x/wind_y are Q16.16 int32 (the collapsed wind bridge). permeability is the
// per-face float bridge. obstacles/is_wall/is_vacuum are the static masks.
//
// PERF NOTE (residency is S8): this does a per-call H2D of the gas plane + wind +
// masks + permeability and a D2H of the gas plane. For the multi-gas substep
// loop that is N_substeps * N_gases transfers per tick — deliberately deferred;
// S8 makes the fields GPU-resident so only the final D2H remains.
void smoke_step(
    int32_t* smoke,            // Q16.16 (h,w) — in/out (one gas plane)
    const int32_t* wind_x,     // Q16.16 (h,w) = -d/dx(atmosphere + wave_p)
    const int32_t* wind_y,     // Q16.16 (h,w) = -d/dy(atmosphere + wave_p)
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    const float* permeability, // FLOAT (h,w) per-face permeability bridge
    int h, int w, float dt,
    float d_smoke, float wind_diffusion_scale, float advection_rate);

// Backend selection (S4a gate + integration). When true, PhysicsEngine::
// run_substeps runs each per-gas smoke transport substep on the GPU instead of
// the CPU SmokeDynamics::step. Defaults false so the game + suite run on the CPU
// path unchanged until explicitly switched. sink_hop ALWAYS runs on the CPU (S4b).
bool smoke_backend_is_cuda();
void set_smoke_backend_cuda(bool on);

}  // namespace breach_cuda
