#pragma once
// ============================================================================
// EOS P6.1 — the bulk donor-cell flux on the GPU (bit-identical).
// ============================================================================
//
// A faithful, bit-identical port of bulk_flux_transport_cached
// (bulk_transport.cpp) — the first-order upwind donor-cell conservative flux
// that moves the two BULK gas species (O2 / inert_N2) on the EOS solver's
// velocity field every substep (eos_solver.cpp:414, design §2.2 / §3.2 step
// 1d). Five stages, each one kernel, per conservative plane:
//
//   B1 face flux + dq   (donor-cell upwind, mul_wide -> flux_to_dq_dev —
//                        the CPU fuses water's K3+K4; so do we)
//   B2 outflow limiter  (out_sum > N -> scale = (N<<16)/out_sum, exact int64
//                        divide; FP_ONE otherwise)
//   B3 scale-apply      (scale_mag on the DONOR cell's factor, in place)
//   B4 divergence apply (gather-then-apply conservative +/- form)
//   B5 clamps           (N = 0 on solid AND vacuum; else max(N, 0))
//
// This is the precompute-then-gather pattern proven by cuda_water.cu K3-K8
// (design §7; docs/eos_p6_gpu_alignment_review.md §1.3): every kernel writes
// only its own cell / its own two faces, every cross-stage dependency sits on
// a kernel-launch boundary exactly where the CPU has a loop boundary — no
// atomics, no scatter, bit-identical by construction. Pure-integer Q16.16
// end to end (there is no float anywhere in the cached CPU entry).
//
// Deltas from water (all mechanical, review §1.3): per-face Q16.16 coefficient
// ARRAYS (coeffE/coeffS — face_permeability * dt_s, precomputed per tick by
// the caller; 0 == sealed face) instead of water's one scalar dt_over_dx_q;
// N conservative planes looped host-side (each plane independent — disjoint N,
// read-only wind/coeffs); the all-zero-plane skip stays a HOST-side scan of
// the host buffer (same early-exit scan as the CPU — and arithmetically a
// no-op either way, review §1.3).
//
// ENGINE DISPATCH WAITS FOR P6.5 (review §4, P6.1 row: "kernel-gate only") —
// eos_solver.cpp keeps calling the CPU entry unconditionally; the backend
// flag below exists so the P6.5 orchestration patch can wire the dispatch
// without touching this TU again.
//
// Plain C++ declaration header (no CUDA types) so the .cpp TUs can include
// it; cuda_bulk_transport.cu provides the definitions. Compiled only when
// BREACH_CUDA.
#include <cstdint>

namespace breach_cuda {

// One bulk donor-cell flux pass on the GPU — IN-PLACE on every
// `gas_conservative`-flagged plane of `gas`. Mirrors bulk_flux_transport_cached
// (bulk_transport.cpp) argument-for-argument — this is the entry the P6.5
// engine dispatch will call with the eos solver's per-tick coefficient cache:
//
// gas               : (n_gases, h, w) contiguous Q16.16 planes, mutated in place
// gas_conservative  : (n_gases,) — true for the bulk pair; traces untouched
// wind_x, wind_y    : (h, w) Q16.16 — the solver's cell-centred velocity
// solid             : (h, w) physics solid mask (N == 0 enforced there)
// is_vacuum         : (h, w) true vacuum — N zeroed (the deliberate breach sink)
// coeffE, coeffS    : (h, w) Q16.16 per-face coefficients (east face of i ->
//                     coeffE[i]; south -> coeffS[i]; 0 == sealed face),
//                     precomputed once per tick exactly as the CPU caller does
void bulk_flux_transport_cached(
    int32_t* gas,
    const bool* gas_conservative,
    int n_gases,
    const int32_t* wind_x,
    const int32_t* wind_y,
    const bool* solid,
    const bool* is_vacuum,
    const int32_t* coeffE,
    const int32_t* coeffS,
    int h, int w);

// Legacy-signature entry mirroring the CPU bulk_flux_transport (the pybind /
// P1-test path): hoists the per-face coefficient (min-perm quantize x dt) on
// the HOST with code VERBATIM from bulk_transport.cpp:47-66 (this .cu's host
// side compiles -Xcompiler=/fp:strict, same determinism floor as the CPU sim
// TUs), then forwards to the cached entry above — exactly the CPU structure.
void bulk_flux_transport(
    int32_t* gas,
    const bool* gas_conservative,
    int n_gases,
    const int32_t* wind_x,
    const int32_t* wind_y,
    const bool* solid,
    const bool* is_vacuum,
    const float* dyn_permeability,
    int h, int w,
    float dt);

// ---- EOS P6.5: device-pointer launcher for the chained eos.step dispatch ---
// One full B1..B5 chain for ONE conservative plane on caller-owned DEVICE
// buffers (the P6.5 orchestrator keeps u/T/gas device-resident across the
// substep loop and interleaves this with the SL-advection kernel exactly as
// the CPU substep loop interleaves the passes). Defined in
// cuda_bulk_transport.cu — the SAME anonymous-namespace kernels as the
// isolated entry, ONE transcription, identical launch geometry.
// d_dq_e/d_dq_s/d_scale are h*w int32 device scratch (fully written by
// B1/B2 before any read — no init needed). NOTE the all-zero-plane skip is
// the CALLER'S choice: the P6.5 chain runs every conservative plane
// unconditionally — arithmetically a no-op on an all-zero plane (review
// §1.3: zero fluxes, scale FP_ONE, unchanged N, clamp re-writes zeros).
// BC (boundary_conditions_spec_2026-07-19 §1/§5): the ambient ring reset lives
// in B5's clamp — d_is_ambient (nullptr = space), n_amb (this plane's reservoir
// value), and d_rail (this plane's int64 boundary_flux accumulator, signed
// atomicAdd). All null/0 -> the byte-identical space path.
// P-E1 (energy-books design §2.1.3): the two trailing int64 face planes, when
// non-null, ACCUMULATE the APPLIED (post-limiter, post-scale_mag) per-face dq
// — the CPU's stage-2c loop, as one kernel launched between B3 and B4. The
// caller owns them and zeroes them once before the plane loop (the CPU zeroes
// on entry because it owns the whole plane loop; same net contract).
// nullptr -> dormancy by branch, the byte-identical legacy chain.
void bulk_flux_plane_device(
    int32_t* d_N,
    const int32_t* d_wind_x, const int32_t* d_wind_y,
    const bool* d_solid, const bool* d_is_vacuum,
    const int32_t* d_coeffE, const int32_t* d_coeffS,
    int32_t* d_dq_e, int32_t* d_dq_s, int32_t* d_scale,
    int h, int w,
    const bool* d_is_ambient = nullptr, int32_t n_amb = 0,
    unsigned long long* d_rail = nullptr,
    int64_t* d_dqsum_e = nullptr, int64_t* d_dqsum_s = nullptr);

// ===========================================================================
// P-E1 — energy-conservative thermal transport, DEVICE twin (energy-books
// design §2.1). The kernel-for-loop transcription of
// bulk_flux_energy_transport_cached (bulk_transport.cpp): the same four pinned
// stages, the same per-cell expression order (face order E, W, S, N — L2-11:
// the gather form is single-writer-per-cell, so no atomics are needed for the
// energy itself; only the VALUE-SUM counters use int64 atomicAdd, which is
// order-free on two's complement).
//
// d_gas_planes : host array of `n_cons` DEVICE pointers, one per conservative
//                plane, in the caller's cons order (ascending gi — the CPU
//                sums in the same order; integer sums are order-free anyway).
// n_amb_cons   : host array of `n_cons` reservoir values (ambient maps only).
// d_rail       : host array of `n_cons` device rail pointers, or nullptr.
// Scratch, caller-owned, all h*w unless noted: d_e / d_nb / d_dqsum_e /
// d_dqsum_s (int64), d_dq_e / d_dq_s / d_scale (int32).
// d_ecnt       : 6 int64 slots, ACCUMULATED into (caller memsets per tick):
//                [0] e_ts_residual [1] e_wipe_sum [2] e_floor_sum (retired,
//                always 0) [3] n_active_flux [4] n_bulk_active_sum
//                [5] e_transport_net (arc #54 §2.8's closure-identity term).
//
// arc #54 (gas-energy conservation, design §2.7 row 1): `d_gas_energy` is now
// the LIVE conserved field this pass MOVES — priced off it directly
// (`price_face`, the exact `floordiv(dq*E,N)` split, bulk_transport.cpp),
// never rebuilt from `d_temperature` each call. `t_amb_raw` is the ambient
// K fold minted mass is priced at; `t_max_phys_q` bounds the MIRROR refresh
// only (the authoritative T_MAX_PHYS rail is the once-per-tick EOS recovery).
void bulk_flux_energy_transport_device(
    int32_t* const* d_gas_planes, int n_cons,
    int32_t* d_temperature,
    int64_t* d_gas_energy, int32_t t_amb_raw, int32_t t_max_phys_q,
    const int32_t* d_wind_x, const int32_t* d_wind_y,
    const bool* d_solid, const bool* d_is_vacuum, const bool* d_ts,
    const int32_t* d_coeffE, const int32_t* d_coeffS,
    int32_t t_min_q, int h, int w,
    int64_t* d_e, int64_t* d_nb, int64_t* d_dqsum_e, int64_t* d_dqsum_s,
    int32_t* d_dq_e, int32_t* d_dq_s, int32_t* d_scale,
    unsigned long long* d_ecnt,
    const bool* d_is_ambient = nullptr,
    const int32_t* n_amb_cons = nullptr,
    unsigned long long* const* d_rail = nullptr);

// Backend selection (P6.1 gate). EOS P6.5: now CONSUMED by the engine
// dispatch — PhysicsEngine::run_substeps routes eos.step to the GPU
// orchestration when this AND the other three EOS-kernel flags are on
// (breach_cuda::eos_step_backend_is_cuda(), cuda_eos_step.h). Defaults false
// so the game + suite run the CPU path unchanged until explicitly switched.
bool bulk_flux_backend_is_cuda();
void set_bulk_flux_backend_cuda(bool on);

}  // namespace breach_cuda
