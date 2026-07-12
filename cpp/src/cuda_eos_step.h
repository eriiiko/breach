#pragma once
// ============================================================================
// EOS P6.5 — the full eos.step per-call GPU dispatch (bit-identical)
// ============================================================================
//
// The review's "big flip" row (docs/eos_p6_gpu_alignment_review.md §4, P6.5):
// one engine tick runs the WHOLE EOS chain on the device by CHAINING the four
// re-proven P6.1–P6.4 kernel surfaces, bit-identical to EOSSolver::step.
// PhysicsEngine::run_substeps dispatches here (instead of eos.step) when
// eos_step_backend_is_cuda() below is true.
//
// PASS-BOUNDARY MAP (what is device-resident vs host, and why each boundary
// is digest-neutral — the per-call-era layout; full residency is S8):
//
//  HOST pre-stage (tick-entry state, before any upload — verbatim step()
//  transcriptions, /fp:strict host):
//    * step 0  P_prev := P (pure copy);
//    * the per-tick scalar folds (identical double expressions);
//    * the four pre-substep reductions — c_LOCAL T-scan, max|u| (max_rad),
//      Dalton N_total, the K·|∇P|·dt/N̂ scan — and the n_sub ceil_div
//      (review §1.6: "n_sub derivation stays host-side"). Digest-neutral:
//      SAME host code on the SAME tick-entry bytes, no kernel involved.
//    * the donor-cell coeffE/S per-tick cache (verbatim host hoist, H2D once).
//
//  DEVICE substep loop (the NEW chaining this patch adds — u/T/bulk-gas stay
//  device-resident across ALL n_sub substeps; P6.1/P6.2's isolated entries
//  round-tripped per call):
//    * cmask built once on device (P6.2 K0);
//    * per substep: D2D src snapshot -> sl_advect3 (P6.2 K1) -> per
//      conservative plane the B1..B5 bulk-flux chain (P6.1) — the exact CPU
//      pass interleave on one stream (kernel order == CPU pass order). The
//      all-zero-plane skip is DROPPED (review §1.3: arithmetic no-op).
//    * step 1f zero-u-on-solid is subsumed by K1 (the proven P6.2 argument;
//      the bulk-flux kernels write only gas planes, never u/T).
//
//  D2H at the substep/solve boundary: wind_x/wind_y/temperature + the
//  conservative gas planes. FORCED host work lives here: the host digests
//  digest_advect/digest_bulk_flux (review §2.6), and the solve inputs —
//  step-2 Dalton sum, p*, div(u*) — must land in HOST buffers because
//  mg_build_levels (the P6.3 contract, review §2.7) is host-side. Digest-
//  neutral: bytes produced by the P6.2/P6.1-proven kernels, then verbatim
//  host code. (This is also why p*/div_u/the post-substep reductions are NOT
//  new device kernels in P6.5: their outputs must be on the host for the
//  level build regardless, so a device port would add un-gated kernel
//  surface for zero transfer savings in the per-call era.)
//
//  DEVICE solve: the proven P6.3 entry eos_mg_vcycle (host-built hierarchy
//  H2D, entire V-cycle iteration + level-0 zero on device, D2H of solved P +
//  host digest_helmholtz) — the blessed host round-trip.
//
//  DEVICE kick + compression: the proven P6.4 entry eos_kick_compression
//  (host Dalton rebuild + §2.5 absorb hoist inside, K1/K2 on device, D2H,
//  host digests, per-call rail counters) — accumulated into the solver's
//  cumulative counters exactly as step() would.
//
//  HOST post: step 5 P materialization (atmosphere := solved P), telemetry
//  (dbg_last_n_sub / dbg_last_c_local_q / the six digests / dbg probes).
//
// Plain C++ declaration header (no CUDA types) so physics_engine.cpp and
// bindings.cpp can include it; cuda_eos_step.cu provides the definitions.
// Compiled only when BREACH_CUDA.
#include <cstdint>

class EOSSolver;

namespace breach_cuda {

// True iff EVERY kernel surface the chained dispatch needs is flagged for
// the GPU (review §4 is silent on a master flag, so the four per-kernel
// flags are ANDed): sl_advection && bulk_flux && mg_solve &&
// kick_compression. This is THE dispatch predicate run_substeps consults.
bool eos_step_backend_is_cuda();

// Telemetry: how many engine ticks actually took the GPU eos.step path
// (incremented once per eos_step_cuda call that runs the chain). Lets the
// P6.5 gate prove the dispatch fired instead of silently comparing the CPU
// path against itself.
long long eos_step_cuda_calls();

// The full EOSSolver::step tick on the GPU — argument-for-argument the
// mirror of EOSSolver::step (eos_solver.h; `solver` supplies the config
// surface, the frozen MG schedule, mg_build_levels/mg_levels, and receives
// the digests / rail counters / telemetry through its mutable members,
// exactly as the CPU step() does). IN PLACE on atmosphere/p_prev/wind_x/
// wind_y/temperature/gas. Bit-identity is gated by
// tests/cuda_eos_step_check.py (the chained per-tick digest gate).
//
// PERF NOTE (residency is S8): per-tick H2D/D2H at the boundaries above and
// per-call uploads inside the P6.3/P6.4 entries. Deliberately unoptimized —
// P6's job is correctness + digest proof, not speed; the review is explicit
// that the per-call port will NOT beat the CPU at 160²x1 env.
void eos_step_cuda(
    const EOSSolver& solver,
    int32_t* atmosphere,
    int32_t* p_prev,
    int32_t* wind_x, int32_t* wind_y,
    int32_t* temperature,
    int32_t* gas, const bool* gas_conservative, int n_gases,
    const bool* solid, const bool* is_vacuum,
    const float* dyn_permeability, const float* dyn_wave_absorb,
    int h, int w, float dt);

}  // namespace breach_cuda
