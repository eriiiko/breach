#pragma once
// ============================================================================
// EOS P6.3 — the multigrid Helmholtz pressure solve on the GPU (bit-identical)
// ============================================================================
//
// A faithful, bit-identical GPU port of EOSSolver::step's step 3 — the
// fixed-schedule V(nu1,nu2)xC multigrid solve of the SPD system
//   m_i·P_i + Σ_f g_f·(P_i − P_nb) = b_i   (F8 work scale)
// with the RB-GS smoother, residual-SUM restriction, piecewise-constant
// injection prolongation, the coarsest-level sweep block, and the
// vacuum-Dirichlet/solid zero on level 0 (eos_solver.cpp mg_run_solve_cpu —
// the routine the live path calls).
//
// SPLIT OF RESPONSIBILITY (docs/eos_p6_gpu_alignment_review.md §2.7): the
// PER-TICK hierarchy build (level-0 operator from p*/div_u/N̂ + the P_prev
// warm start, PC-Galerkin coarse operators, Q.32 diagonal reciprocals) stays
// on the HOST and runs through the SAME EOSSolver::mg_build_levels the CPU
// path calls — bit-identical because it IS the CPU build (one code path; the
// build depends on per-tick state, so it follows the tick cadence by
// necessity: level-0 m derives from p* and gE/gS fold the per-tick 1/N̂, and
// every coarse operator is a Galerkin sum of those). The binding hands the
// built levels to eos_mg_vcycle below as plain pointer views; this module
// uploads them and runs the ENTIRE iteration on the device. On-device build
// is the S8 endpoint (all gathers — review §1.2), not P6.
//
// DETERMINISM ARGUMENT (review §1.1/§1.2/§2.2):
//  * smoother — within one (sweep, color) pass every read is the cell's own
//    pre-update P (each cell written exactly once per pass), an
//    OPPOSITE-color neighbor (the 4-stencil flips parity; not written this
//    pass), or per-cell constants -> within-color updates are ORDER-FREE,
//    so one launch per color (the launch boundary = the CPU's color
//    boundary) reproduces the CPU's sequential sweep bit-for-bit;
//  * transfers — restriction (per-coarse-cell SUM of ≤4 child residuals)
//    and prolongation (per-fine-cell read of its ONE parent) are pure
//    gathers, single writer per cell;
//  * arithmetic — every coefficient×field product is the SAME 128-bit
//    staged multiply (mul128_shr_signed == the MSVC _mul128 hi:lo combine),
//    int64 accumulation, and (int32_t) narrow as the CPU; the device side
//    is DIVISION-FREE and FLOAT-FREE (all divides/reciprocals/quantize live
//    in the host build).
//  * fused coarse tail — see the .cu header; bit-identical by construction
//    (same arithmetic, same pass boundaries via __syncthreads, order-free
//    within each pass).
//
// Plain C++ declaration header (no CUDA types) so bindings.cpp can include
// it; cuda_mg_solve.cu provides the definitions. Compiled only when
// BREACH_CUDA.
#include <cstdint>

namespace breach_cuda {

// A pointer view of one host-built MG level (EOSSolver::MGLevel's arrays).
// All buffers are h*w; P is the level's initial value (level 0: the P_prev
// warm start; coarse levels: anything — the restriction kernel writes every
// coarse cell's P and b before any read of them, exactly like the CPU).
struct MGLevelHostView {
    int h = 0, w = 0;
    const uint8_t* excl = nullptr;   // 0 regular / 1 Dirichlet / 2 excluded
    const int64_t* m = nullptr;      // mass 1/aK (Q16.16 raw, clamped)
    const int64_t* gE = nullptr;     // east-face conductance (Q16.16 raw)
    const int64_t* gS = nullptr;     // south-face conductance (Q16.16 raw)
    const int64_t* recip = nullptr;  // diagonal reciprocal (2^48/d_raw, Q.32)
    const int64_t* b = nullptr;      // RHS at F8 work scale
    const int32_t* P = nullptr;      // initial P (q16 raw)
};

// Run the full pressure iteration on the device: the frozen V(nu1,nu2)xC
// schedule when use_multigrid && n_levels > 1, else flat_S RB-GS sweeps on
// level 0 (exactly mg_run_solve_cpu's branch), then the vacuum-Dirichlet/
// solid zero on level 0. Writes the solved level-0 P into p_out (h*w int32)
// and returns the host-side FNV digest over it — byte-for-byte the value
// EOSSolver::step stores in digest_helmholtz for identical inputs.
//
// launches_actual/launches_naive (optional, may be nullptr) report the
// solve's kernel-launch counts: actual = with the fused coarse-tail kernel
// (review §2.2, levels of ≤1024 cells run inside ONE single-block kernel);
// naive = the per-color-per-sweep count the same schedule would need without
// the tail fusion. Telemetry for the review §5 launch-cost check.
//
// PERF NOTE (residency is S8): per-call H2D of every level's 8 arrays and a
// D2H of level-0 P. Deliberately unoptimized — P6's job is correctness +
// digest proof per kernel, not speed (review, executive verdict).
uint64_t eos_mg_vcycle(
    const MGLevelHostView* levels, int n_levels,
    bool use_multigrid, int mg_cycles, int mg_nu1, int mg_nu2,
    int mg_coarsest_sweeps, int flat_S,
    int32_t* p_out,
    int* launches_actual, int* launches_naive);

// Backend selection (the P6.1/P6.2 surviving-backend idiom). NOTE: no engine
// dispatch site consumes this yet — EOS orchestration dispatch is P6.5 ("the
// big flip", review §4); until then the flag exists so the gate / tooling
// surface matches the other kernels and P6.5 has a switch to wire. Defaults
// false.
bool mg_solve_backend_is_cuda();
void set_mg_solve_backend_cuda(bool on);

}  // namespace breach_cuda
