#pragma once
// ============================================================================
// CUDA-S8a Path A — the fully device-resident EOS tick (bit-identical)
// ============================================================================
//
// Design: docs/cuda_s8a_path_a_impl_2026-07-21.md (v2, post-critique).
// The resident sibling of eos_step_cuda (cuda_eos_step.h): the HOST pre-stage
// (all of EOS's global reductions — they consume tick-entry state, which the
// authoritative numpy mirror holds at entry) runs verbatim on the mirror via
// the shared eos_host_prestage; EVERYTHING after the substep loop runs on
// device — div_u / Dalton N_total / p* mid-stage kernels, the ON-DEVICE
// mg_build_levels port (per-cell + per-coarse-cell single-writer gathers,
// integer divides — "the S8 endpoint", cuda_mg_solve.h §2.7), the shared
// vcycle schedule (eos_mg_vcycle_resident), the shared kick/compression
// launch core, and the atmosphere store. ZERO mid-tick plane transfers; the
// only D2H is ~56 B of telemetry scalars (boundary_flux rail + rail counters).
//
// TELEMETRY GAPS (accepted, design §3.3): the six digest_* members, the
// dbg_probe_* T checkpoints, and the host solver.levels_ hierarchy are NOT
// maintained on this path (their only consumers drive the per-call path,
// which is unchanged). dbg_last_n_sub / dbg_last_c_local_q ARE set (host
// pre-stage), and boundary_flux_ + the five rail counters are maintained
// exactly as the per-call entry maintains them.
//
// Plain C++ declaration header (no CUDA types) so physics_engine.cpp and
// bindings.cpp can include it; cuda_eos_resident.cu provides the
// definitions. Compiled only when BREACH_CUDA.
#include <cstdint>
#include <string>

class EOSSolver;

namespace breach_cuda {

// One fully resident EOS tick. Host-mirror pointers feed the pre-stage
// (p_prev is WRITTEN host-side — the step-0 copy — and dyn_wave_absorb feeds
// the host absorb_q hoist); device pointers are the persistent CuPy-owned
// resident fields (GameMap.device_ptrs()) + the static ambient grids.
// d_is_ambient / d_sponge_sigma / d_sponge_udamp are nullptr on space maps /
// when off — dormancy BY BRANCH, byte-identical space path. The caller
// guarantees the mirror is current at entry (the design §2 invariant) and
// that the device copies of every input were uploaded from that mirror.
void eos_step_resident(
    const EOSSolver& solver,
    // host mirror (pre-stage + telemetry)
    const int32_t* atmosphere,
    int32_t* p_prev,
    const int32_t* wind_x, const int32_t* wind_y,
    const int32_t* temperature,
    const int32_t* gas, const bool* gas_conservative, int n_gases,
    const bool* solid, const bool* is_vacuum,
    const float* dyn_permeability, const float* dyn_wave_absorb,
    int h, int w, float dt,
    const bool* is_ambient, const int32_t* n_amb, int32_t p_amb,
    // device (persistent resident fields + static ambient grids)
    int32_t* d_atmosphere, int32_t* d_wave_p,
    int32_t* d_wind_x, int32_t* d_wind_y,
    int32_t* d_temperature, int32_t* d_gas_base,
    const bool* d_solid, const bool* d_is_vacuum,
    const float* d_dyn_permeability,
    const bool* d_is_ambient,
    const int32_t* d_sponge_sigma, const int32_t* d_sponge_udamp);

// Telemetry: how many ticks ran the resident EOS chain (the gate's
// vacuousness guard — proves the bracket is gone, not silently per-call).
long long eos_resident_calls();

// TEST-ONLY (gate PART 1c, design §7): run the HOST mg_build_levels and the
// DEVICE build (the same production kernels eos_step_resident launches) on
// identical inputs, byte-compare every level's excl/m/gE/gS/recip/b/P, and
// return the total mismatch count (0 == bit-identical). `report` (if
// non-null) receives a per-level per-array mismatch summary. Never touches
// the resident tick's persistent scratch.
long long eos_mg_build_parity(
    const EOSSolver& solver,
    const int32_t* pstar, const int32_t* div_u, const int32_t* n_total,
    const int32_t* p_prev,
    const bool* solid, const bool* is_vacuum,
    const float* dyn_permeability,
    int h, int w, float dt,
    const bool* is_ambient, int32_t p_amb,
    const int32_t* sponge_sigma,
    std::string* report);

}  // namespace breach_cuda
