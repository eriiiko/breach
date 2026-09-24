#pragma once
// ============================================================================
// ray-engine-v2 P4 — THE CUDA TWIN of the radiation sweep (radiation_sweep.h).
// docs/ray_engine_v2_design_v3_2026-09-15.md §8.2 ("one formulation, two
// backends"), §2.3 (the integer scheme in gather form), §10 (cost), §11 P4.
// ============================================================================
//
// The SAME sweep as RadiationSweep::run, integer for integer: the same Fleck
// pre-pass (radiation_sweep.h's FP_HD fleck_L_solid_q / fleck_f_q24, which
// call fixed_point.h's floordiv_q and signed-shift twin), the same E° lookup
// (emissive_table.h's FP_HD e_bucket_of), the same per-cell arithmetic of
// design §2.3, and the same checked-in per-ordinate constants — read on the
// host from RadiationSweep::ordinate_table, never re-typed here. Held to the
// CPU at tol 0 on all four planes, the Fleck plane and the stream telemetry by
// tests/cuda_radiation_sweep_check.py (+ its pytest wrapper), which chains it
// to the integer reference through gate 0.
//
// THE WAVEFRONT MECHANICS (design §8.2, row 29):
//   * Within an ordinate the GATHER form makes the order structural: a cell
//     reads its two upwind neighbours' stored outflow and recomputes the split
//     with the same shift, so no two threads ever write one outflow cell and
//     no atomic is needed inside an ordinate. The upwind cells of one
//     wavefront all lie on the previous one — a whole COLUMN (shear,
//     x-major), a whole ROW (shear, y-major), or an ANTI-DIAGONAL (step, the
//     Koch-Baker-Alcouffe skew) — so a launch per wavefront index is a
//     topological order of every ordinate's dependency DAG.
//   * Across ordinates the 16 run CONCURRENTLY, each with its own stored-
//     outflow plane (scratch (N, n_ordinates, h, w)), one launch per wavefront
//     INDEX covering all of them (blockIdx.y = the ordinate). Two ordinates
//     can reach one cell inside one launch, so the per-cell books (rad_net,
//     rad_flux, rad_amb, rad_fluence) are INTEGER atomics in the tree's idiom,
//     atomicAdd((unsigned long long*)p, (unsigned long long)v): unsigned wrap
//     IS the two's-complement signed add, modular addition is associative and
//     commutative, so the sum is the CPU's whatever order the hardware picks
//     (design §8.2; the headroom argument keeps every true sum in range).
//   * Launches per call: shear max(h, w) wavefronts, step h + w - 1, plus
//     three bookkeeping kernels (init, pre-pass, zero). §10 measures them.
//
// Credits (the published techniques this file implements, iron rule):
//   * The sweep itself — S_N discrete ordinates (Chandrasekhar, "Radiative
//     Transfer", 1950), the shear/step transport trade (Davis et al. 2012) and
//     the Fleck & Cummings 1971 emission damping — exactly as credited in
//     radiation_sweep.h; this twin changes WHERE they run, not what they are.
//   * The wavefront parallelisation of a discrete-ordinates sweep:
//     K.R. Koch, R.S. Baker, R.E. Alcouffe, "Solution of the first-order form
//     of the 3-D discrete ordinates equation on a massively parallel
//     processor", Trans. Am. Nucl. Soc. 65 (1992) 198-199 (LA-UR-91-4157) —
//     the "KBA" sweep: a wavefront's cells depend only on the previous
//     wavefront, so they run concurrently, with the ordinates pipelined
//     through the same steps. KBA decomposed a 3-D grid over processors; this
//     is the single-device 2-D case (the design's "KBA skew" is the step
//     transport's anti-diagonal). No full text exists to archive (OSTI holds
//     none); the entry is docs/papers/README_ray_engine_v2_2026-09-13.md #16.
//
// Plain C++ declaration header (no CUDA types in the signatures) so the .cpp
// TUs (physics_engine.cpp, bindings.cpp — cl.exe even in the CUDA build) can
// include it. Compiled only when BREACH_CUDA. The LAUNCH CORE
// (radiation_sweep_launch_resident) is declared in cuda_resident.h beside the
// other families' cores.
#include <cstdint>

namespace breach_cuda {

// ONE sweep on the GPU, the PER-CALL path (the cuda_resident.h `*_step`
// shape): allocate, copy in, launch the resident core, synchronize, copy out,
// free. HOST pointers throughout; this is what PhysicsEngine::step_tail
// dispatches at step 2b when the radiation backend is on, on the host mirror,
// on both the normal and the resident tick (design §8.2 "Residency").
//
// The arguments are RadiationSweep::run's, one for one, plus the ambient's
// second door:
//   amb_level         : int64 (h, w) — the per-cell ambient LEVEL, as run()
//                       takes it (0 <= amb <= e_table[0]). NULLABLE: when null
//                       the level is DERIVED per cell on the device from
//                       `is_vacuum` and `vac_level` — the twin of
//                       RadiationSweep::derive_ambient (a vacuum cell radiates
//                       against `vac_level`, every other cell against
//                       e_table[0]; vac_level < 0 means e_table[0], thermal v2
//                       R4) — so no ambient plane crosses the bus on the live
//                       path. amb_level == is_vacuum == nullptr is the uniform
//                       E°[0] configuration (the CPU binding's `None` door).
//   is_vacuum         : bool (h, w), nullable (read only when amb_level is null)
//   vac_level         : the vacuum cell's emissive level (read only when
//                       amb_level is null); > e_table[0] is rejected, exactly
//                       as derive_ambient rejects it.
//   rad_net/flux/amb/fluence : int64 (h, w), OVERWRITTEN (the core zeroes them
//                       on the device before the first wavefront), exactly
//                       run()'s contract. An ingress-rejected scene leaves the
//                       caller's planes UNTOUCHED (nothing is copied back).
//   fleck_out         : int32 (h, w), nullable — receives the Fleck plane (Q24)
//                       the pre-pass computed, RadiationSweep::fleck_plane()'s
//                       twin.
//   min_stream_out / max_stream_out : nullable — the smallest and largest
//                       per-ordinate stream seen at any cell, the twin of
//                       RadiationSweep::min_stream / max_stream.
//   gas, n_gases, heat_absorb_q16, n_bulk : THE GAS EXTINCTION (P5a) — run()'s
//                       four, host pointers, all or none. Only the ACTIVE
//                       gases (heat_absorb_q16 != 0) cross the bus: a zero
//                       coefficient adds exactly 0 to the density sum, so its
//                       plane is never uploaded — and with every shipped row at
//                       0.0 the live GPU sweep uploads no gas plane and no
//                       n_bulk at all, so P4's measured cost does not move.
// Throws std::invalid_argument on everything run() and derive_ambient() reject
// (an unsupported (n_ordinates, transport), k_leak_q outside [0, ONE],
// vac_level above e_table[0], a cell violating 0 <= a <= d <= ONE, an
// ambient level outside [0, e_table[0]], a partial gas group, n_gases outside
// [0, N_GAS_PLANES_MAX] or a heat_absorb_q16 outside [0, HEAT_ABSORB_Q_MAX]),
// and std::runtime_error on a CUDA error. Returns the number of kernel
// launches issued (design §10's count).
int radiation_sweep_step(
    const int32_t* temperature,
    const int32_t* heat_atten_q, const int32_t* dyn_heat_atten_q,
    const int32_t* heat_inv_shift, const bool* thermal_solid,
    const int64_t* e_table,
    const int64_t* amb_level, const bool* is_vacuum, int64_t vac_level,
    int32_t t_amb_q, int32_t k_leak_q,
    int transport, int n_ordinates, int h, int w,
    int64_t* rad_net, int64_t* rad_flux, int64_t* rad_amb, int64_t* rad_fluence,
    bool fleck_enabled,
    int32_t* fleck_out, int64_t* min_stream_out, int64_t* max_stream_out,
    const int32_t* gas = nullptr, int n_gases = 0,
    const int32_t* heat_absorb_q16 = nullptr, const int32_t* n_bulk = nullptr);

// The launch count radiation_sweep_launch_resident issues for one call of the
// given shape (the three bookkeeping kernels + one per wavefront index), or -1
// for an unsupported (n_ordinates, transport). A pure function of the shape.
int radiation_sweep_launch_count(int transport, int n_ordinates, int h, int w);

// Backend selection (the S1 idiom). When true, PhysicsEngine::step_tail runs
// the sweep on the GPU instead of RadiationSweep::run. Defaults false.
bool radiation_backend_is_cuda();
void set_radiation_backend_cuda(bool on);

// Dispatch-fired telemetry (the eos_step_cuda_calls idiom): how many times
// radiation_sweep_step has completed a sweep in this process. A gate that
// flips the backend and sees this move knows the GPU really ran — a silently
// CPU "GPU run" would make a bit-identity gate vacuous.
long long radiation_sweep_cuda_calls();

}  // namespace breach_cuda
