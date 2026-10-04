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
// (An S4b paragraph here claimed sink_hop "now runs on the GPU" — false since
// EOS refactor P3 deleted the CPU twin. Removed with the kernel, audit Patch A
// / A9, 2026-08-04. This header now describes only the live smoke step.)
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

// Smoke transport v2 (#12, P2a): the per-call smoke_step (the isolated GPU SL
// step), its `set_smoke_backend` / `smoke_backend_is_cuda` dispatch flag and the
// `cuda_smoke_step` binding are DELETED -- the trace planes ride the bulk face
// flux inside the EOS substep loop (cuda_bulk_transport.cu stages 3b/3c) and the
// once-per-tick trace tail runs in shared host code after the dispatch.
//
// What remains in cuda_smoke.cu is ONLY what the device-RESIDENT path still
// calls: smoke_launch_resident + trace_smoke_resident (declared in
// cuda_resident.h) and their kernels -- the OLD semi-Lagrangian law, which the
// resident path keeps until P2b replaces it with the resident trace tail and
// deletes this file. It therefore no longer agrees with the CPU / chained law.
}  // namespace breach_cuda
