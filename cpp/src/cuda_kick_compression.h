#pragma once
// ============================================================================
// EOS P6.4 — momentum kick + compression work on the GPU (bit-identical)
// ============================================================================
//
// A faithful, bit-identical GPU port of the EOS solver's post-solve tail
// (eos_solver.cpp, EOSSolver::step steps 4 + 4c — design §3.2, exactly what
// eos_kick_compression_reference replays on the CPU):
//   * K1 (kick): u -= dt·K·grad(P_new)/N̂ (the whole chain int64, the wide
//     Kdt bridge staged through 128-bit products) → absorption damping
//     u *= (1 − absorb·dt) (magnitude-first shrink) → the ±2^30 component
//     pre-clamp overflow guard → the |u| ≤ min(c_LOCAL, U_MAX) counted
//     magnitude clamp (scale-to-cap) → the ONE (int32_t) narrow at store;
//     u zeroed outside open-air.
//   * K2 (compression work): T -= (γ−1)·T·div(u_new)·dt on the corrected
//     velocity, factor clamped to ±T_WORK_CLAMP (counted), saturating add,
//     T_MIN floor + T_MAX_PHYS ceiling (counted, exclusive).
//
// Both passes are pure per-cell GATHERS (review §1.5): the kick reads only
// its own cell's u plus the solved-P plane (never written here) and writes
// its own u; the compression reads NEIGHBOR u (frozen — K1 completed grid-
// wide at the kernel boundary, the CPU's pass boundary) plus its OWN T and
// writes its own T. No scatter, no atomics on fields. The five rail counters
// are device atomics — pure +1 per engaging CELL, so the totals are
// order-free (= the number of engaging cells) and deterministic.
//
// The per-cell quantize(dyn_wave_absorb[i])·absorb_dt fold is hoisted to a
// host-side per-tick q16 plane (review §2.5 — same double math, same
// rounding, per-tick-constant input; the blessed P3 hoist class), so the
// kernels are float-free. All other scalar folds run on the /fp:strict host
// pass through the IDENTICAL double expressions step() uses.
//
// Plain C++ declaration header (no CUDA types) so the .cpp TUs can include
// it; cuda_kick_compression.cu provides the definitions. BREACH_CUDA only.
#include <cstdint>

namespace breach_cuda {

// The step-4 + step-4c tail for one tick on the GPU — IN PLACE on
// wind_x/wind_y/temperature (h, w). Argument-for-argument the mirror of
// eos_kick_compression_reference (eos_solver.h — semantics documented there):
// p_new is the solved pressure plane (== post-tick `atmosphere`), the gas
// planes rebuild step 2's Dalton N_total on the host verbatim, c_local_q is
// the solver's per-tick cap (EOSSolver::dbg_last_c_local_q), and the float
// scalars are the EOSSolver config members. Outputs: digest_velocity_out /
// digest_compression_out — byte-for-byte step()'s digest expressions,
// computed host-side after the D2H (review §2.6) — and counters_out[5] =
// { u_clamp_hits, u_max_hits, work_clamp_hits, energy_floor_hits,
// t_max_phys_hits } for THIS call.
//
// PERF NOTE (residency is S8): per-call H2D of 3 fields + P + N_total +
// absorb plane + masks, 2 kernel launches, D2H of 2 fields + counters.
// Deliberately unoptimized — P6's job is correctness + digest proof per
// kernel, not speed (review, executive verdict).
void eos_kick_compression(
    int32_t* wind_x,               // Q16.16 (h,w) — in/out (solver u.x)
    int32_t* wind_y,               // Q16.16 (h,w) — in/out (solver u.y)
    int32_t* temperature,          // Q16.16 (h,w) — in/out (ΔT above ambient)
    const int32_t* p_new,          // Q16.16 (h,w) — solved P (read-only)
    const int32_t* gas, const bool* gas_conservative, int n_gases,
    const bool* solid, const bool* is_vacuum,
    const float* dyn_wave_absorb,  // FLOAT (h,w) — host-hoisted to q16 (§2.5)
    int h, int w, float dt, int32_t c_local_q,
    float c_max, float dx, float adiabatic_index, float absorb_strength,
    float n_floor_solver, float t_min, float t_work_clamp,
    float t_max_phys, float u_max, float trace_mass_scale,
    uint64_t* digest_velocity_out, uint64_t* digest_compression_out,
    int64_t* counters_out /* [5] */,
    // BC (spec §1/§3): the ambient ring (nullptr = space) drives the velocity
    // zero + compression skip; the u-damping band grid sponge_udamp (nullptr =
    // off) is the rung-2 absorber applied after the absorb chain, magnitude-
    // first. All null -> the byte-identical space path.
    const bool* is_ambient = nullptr, const int32_t* sponge_udamp = nullptr,
    // THERMAL-MASS AXIS, P-EOS (docs/thermal_mass_eos_ruling_2026-07-30.md §4
    // item 3): step 4c skips its `temperature[i]` write where this mask is true
    // — the TemperatureSolver owns an object's T. The KICK is untouched (it
    // writes u, never T), so pressure/velocity/gas flow are unchanged. nullptr
    // -> `solid` on the device (the P2 fallback idiom) == pre-patch behaviour.
    const bool* thermal_solid = nullptr);

// Backend selection (the P6.1/P6.2 surviving-backend idiom). NOTE: no engine
// dispatch site consumes this yet — EOS orchestration dispatch is P6.5 ("the
// big flip", review §4); until then the flag exists so the gate / tooling
// surface matches the other kernels and P6.5 has a switch to wire.
// Defaults false.
bool kick_compression_backend_is_cuda();
void set_kick_compression_backend_cuda(bool on);

}  // namespace breach_cuda
