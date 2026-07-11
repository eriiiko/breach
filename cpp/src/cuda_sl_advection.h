#pragma once
// ============================================================================
// EOS P6.2 — fused 3-field semi-Lagrangian advection on the GPU (bit-identical)
// ============================================================================
//
// A faithful, bit-identical GPU port of the EOS solver's substep-loop advection
// chain (eos_solver.cpp, EOSSolver::step step 1a/1b/1f — design §3.2):
//   * the per-tick cmask build (sealed/breach/live corner+march table),
//   * per substep: src snapshot of (vx, vy, T) -> the FUSED per-cell backtrace
//     (one DDA wall-clip march + one Q16.16 bilinear weight set shared by all
//     three fields, exactly eos_backtrace_sample3_q) -> zero u on solid,
//     T := 0 on vacuum destinations.
// SL advection is a pure GATHER — each destination cell reads its backtraced
// source from the frozen snapshot and writes only itself — so there is no
// scatter hazard and bit-identity is direct (docs/eos_p6_gpu_alignment_review.md
// §1.4; device precedent: cuda_smoke.cu's backtrace class, S4a).
//
// The transport core is pure-integer Q16.16; the ONLY float is the host-side
// dt_s_q = quantize((double)dt / n_sub) scalar fold, replicated in double
// exactly as the CPU does (/fp:strict host pass), and the per-cell
// permeability <= 0 COMPARISON inside the cmask build (a comparison, not
// arithmetic — bit-exact by construction). The bilinear renorm uses
// reciprocal_q16_dev (the shared verbatim Newton reciprocal).
//
// Plain C++ declaration header (no CUDA types) so the .cpp TUs can include it;
// cuda_sl_advection.cu provides the definitions. Compiled only when BREACH_CUDA.
#include <cstdint>

namespace breach_cuda {

// The FULL substep-loop advection chain for one tick on the GPU — IN-PLACE on
// wind_x/wind_y/temperature (h, w). Mirrors eos_sl_advect_reference
// (eos_solver.cpp) exactly: cmask build, then n_sub x [snapshot -> fused
// advect]. n_sub is the schedule the real solver derived for this tick
// (EOSSolver::dbg_last_n_sub); dt is the full tick dt (dt_s folds on the host
// in double, exactly like the CPU). Returns the SAME chained FNV digest over
// (T, then wy, then wx) that EOSSolver::step stores in digest_advect at its
// last substep, computed host-side after the D2H (review §2.6: digests stay
// host-side in the per-call era).
//
// PERF NOTE (residency is S8): per-call H2D of 3 fields + masks + perm and a
// D2H of the 3 fields; n_sub kernel launches + 3*n_sub D2D snapshot copies.
// Deliberately unoptimized — P6's job is correctness + digest proof per
// kernel, not speed (review, executive verdict).
uint64_t eos_sl_advect(
    int32_t* wind_x,               // Q16.16 (h,w) — in/out (solver u.x)
    int32_t* wind_y,               // Q16.16 (h,w) — in/out (solver u.y)
    int32_t* temperature,          // Q16.16 (h,w) — in/out (ΔT above ambient)
    const bool* solid,
    const bool* is_vacuum,
    const float* dyn_permeability, // FLOAT (h,w) — cmask build only (<= 0 test)
    int h, int w, float dt, int n_sub);

// ---- EOS P6.5: device-pointer launchers for the chained eos.step dispatch --
// The P6.5 orchestrator (cuda_eos_step.cu) keeps u/T/gas DEVICE-RESIDENT
// across the whole substep loop, so it needs to launch the SAME two kernels
// the isolated entry above launches — on buffers it already owns on the
// device. These wrappers are defined in cuda_sl_advection.cu (same TU as the
// kernels — exactly ONE transcription of each kernel exists) and launch with
// the identical block/grid shape the isolated entry uses. All pointers are
// DEVICE pointers. Bit-identity: same kernels, same launch geometry; only the
// buffer residency differs (a transfer-boundary choice, not arithmetic).

// K0 — the per-tick cmask build (sealed/breach/live table), once per tick.
void sl_cmask_build_device(const bool* d_solid, const bool* d_vacuum,
                           const float* d_perm, uint8_t* d_cmask, int n);

// K1 — ONE fused advection substep: reads the frozen (src_vx, src_vy, src_t)
// snapshot, writes wind/temperature in place (solid u zeroed, vacuum T := 0).
// The caller owns the pre-substep D2D snapshot (the CPU's vx_src_/vy_src_/
// t_src_ copies) and the substep loop.
void sl_advect3_device(int32_t* d_wind_x, int32_t* d_wind_y,
                       int32_t* d_temperature,
                       const int32_t* d_src_vx, const int32_t* d_src_vy,
                       const int32_t* d_src_t,
                       const bool* d_solid, const bool* d_vacuum,
                       const uint8_t* d_cmask,
                       int32_t dt_s_q, int h, int w);

// Backend selection (P6.2 gate wiring, the surviving-backend idiom). EOS
// P6.5: now CONSUMED by the engine dispatch — PhysicsEngine::run_substeps
// routes eos.step to the GPU orchestration when this AND the other three
// EOS-kernel flags are on (breach_cuda::eos_step_backend_is_cuda(),
// cuda_eos_step.h). Defaults false.
bool sl_advection_backend_is_cuda();
void set_sl_advection_backend_cuda(bool on);

}  // namespace breach_cuda
