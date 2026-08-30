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
#include <vector>

class EOSSolver;

namespace breach_cuda {

// ---- S8a Path A: the shared HOST pre-stage (ONE transcription) -------------
// (docs/cuda_s8a_path_a_impl_2026-07-21.md §3.2.2.) The verbatim step()
// pre-stage block eos_step_cuda has always run — the boundary_flux_ member
// reset (BOTH branches, incl. the space-map clear), the mirror
// p_prev := atmosphere copy (step 0 — load-bearing: the max_du grad-scan
// reads p_prev), the dbg probe, the per-tick scalar folds, the c_LOCAL /
// max|u| / Dalton / K·|∇P|·dt/N̂ scans and the n_sub ceil_div, the donor-cell
// coeffE/S cache, and the conservative-plane index list. Factored so the
// device-resident entry (cuda_eos_resident.cu) consumes the IDENTICAL bits;
// defined in cuda_eos_step.cu (the same /fp:strict host pass either way).
// Everything here reads TICK-ENTRY state — in the resident tick that is the
// authoritative numpy mirror (design §0).
struct EOSHostPrestage {
    // Per-tick scalars the mid/late stages consume (Q16.16 raw where noted).
    int32_t t_amb_q   = 0;   // quantize(T_AMB_K) — pstar
    int32_t s_eos_q   = 0;   // quantize(S_EOS) — pstar (phi_exp*k_temp_to_kelvin, frozen == 65536)
    int32_t c_q       = 0;   // quantize(C) — pstar
    int32_t inv_2dx_q = 0;   // quantize(1/(2·dx)) — div_u (+ kick per-call)
    // VELOCITY-CLAMP (P-V1, D2v2): c_LOCAL survives SOLELY as the n_sub/CFL
    // estimate's ceiling (D7's clip) — the kick no longer reads it, see
    // `cap2` below. == dbg_last_c_local_q.
    int32_t c_local_q = 0;
    int     n_sub     = 1;   // the substep schedule (== dbg_last_n_sub)
    int32_t dt_s_q    = 0;   // quantize(dt/n_sub) — the substep dt
    int32_t t_min_q   = 0;   // quantize(T_MIN) — P-E1's recovery clamp (§2.1.5)
    // Per-tick host planes (H2D'd once by the caller).
    std::vector<int32_t> coeffE, coeffS;   // donor-cell face-coeff cache
    // VELOCITY-CLAMP (P-V1, D2v2): the per-cell velocity-cap² plane (Q32.32
    // raw), folded from tick-entry T in the SAME scan as c_LOCAL above (the
    // coeffE/coeffS idiom — a HOST std::vector, H2D'd once by the caller;
    // NOT a device pointer, per this struct's CUDA-type-free contract).
    std::vector<int64_t> cap2;
    // Conservative-plane index list (the CPU's gi order preserved).
    std::vector<int> cons;
};

EOSHostPrestage eos_host_prestage(
    const EOSSolver& solver,
    const int32_t* atmosphere,
    int32_t* p_prev,
    const int32_t* wind_x, const int32_t* wind_y,
    const int32_t* temperature,
    const int32_t* gas, const bool* gas_conservative, int n_gases,
    const bool* solid, const bool* is_vacuum,
    const float* dyn_permeability,
    int h, int w, float dt,
    bool ambient_mode,
    // THERMAL-MASS AXIS, P-EOS: the per-medium THERMAL mask, mirroring
    // EOSSolver::step's `ts` fold (nullptr -> `solid`, today's behaviour).
    // VELOCITY-CLAMP (P-V1, D4): also gates the cap2 fold's ts->ambient rule.
    const bool* thermal_solid = nullptr);

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
// BC (boundary_conditions_spec_2026-07-19): the planetside AMBIENT ring mirror
// (B4). is_ambient/n_amb/p_amb/sponge_sigma/sponge_udamp default null/0 so a
// space map takes the byte-identical path. The SHIFT + ring→Dirichlet excl +
// σ-diagonal flow through the SHARED host-side mg_build_levels (args forwarded
// there); the reset+rail (bulk), the u/T widenings + u-damping (SL/kick), and
// the step-5 add-back are mirrored device/host-side here.
void eos_step_cuda(
    const EOSSolver& solver,
    int32_t* atmosphere,
    int32_t* p_prev,
    int32_t* wind_x, int32_t* wind_y,
    int32_t* temperature,
    // arc #54 §2.2 (P-G2): the conserved gas energy field, in/out — K1's KE
    // brackets and K3's face-flux step + recovery all read/write it here.
    int64_t* gas_energy,
    int32_t* gas, const bool* gas_conservative, int n_gases,
    const bool* solid, const bool* is_vacuum,
    const float* dyn_permeability, const float* dyn_wave_absorb,
    int h, int w, float dt,
    const bool* is_ambient = nullptr,
    const int32_t* n_amb = nullptr,
    int32_t p_amb = 0,
    const int32_t* sponge_sigma = nullptr,
    const int32_t* sponge_udamp = nullptr,
    // THERMAL-MASS AXIS, P-EOS (docs/thermal_mass_eos_ruling_2026-07-30.md §4
    // item 1): the per-medium THERMAL mask, mirroring EOSSolver::step's trailing
    // arg exactly. ONE static-shaped H2D per tick (the sponge-grid precedent);
    // nullptr -> today's behaviour byte-for-byte, with `d_ts` falling back to
    // `d_solid` so nothing is allocated or copied on that path.
    const bool* thermal_solid = nullptr);

}  // namespace breach_cuda
