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
// exactly as the per-call entry maintains them. Smoke transport v2 (#12, P2b):
// the trace planes ride stages 3b/3c on every substep here too, and the trace
// books (boundary_flux_'s trace slots = vent, trace_wipe_sum_) are stored from
// a second ~(16·n_gases) B D2H; the once-per-tick trace tail is
// PhysicsEngine::run_trace_tail_resident, which adds vent / sink / decay.
//
// Plain C++ declaration header (no CUDA types) so physics_engine.cpp and
// bindings.cpp can include it; cuda_eos_resident.cu provides the
// definitions. Compiled only when BREACH_CUDA.
#include <cstddef>
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
// #29 P5: the EOS stage's per-env books, (N, slots) unsigned long long blocks
// in the resident tick's arena (cuda_resident_tick.cu), laid out so ONE memset
// [zero_begin, zero_begin + zero_bytes) zeroes them all each tick; folded into
// each env's EOSSolver by the tick's one books D2H (resident_tail).
constexpr int EOS_BOOK_GAS_MAX = 16;                    // gas planes a batch may carry
constexpr int EOS_ECNT_SLOTS   = 6;                     // the P-E1 energy counters
constexpr int EOS_TCNT_SLOTS   = 4 * EOS_BOOK_GAS_MAX;  // trace vent | wipe | clip | neg
// The pre-stage's per-env slots (int64 in unsigned long long).
constexpr int EOS_PRE_MAX_DU    = 0;   // max K·|∇P|·dt/N̂ (Q16 raw)
constexpr int EOS_PRE_T_MAX_ABS = 1;   // max T_abs over open cells (0 = none above ambient)
constexpr int EOS_PRE_MAX_RAD   = 2;   // max |u|² (Q32.32)
constexpr int EOS_PRE_N_SUB     = 3;   // the env's substep count
constexpr int EOS_PRE_C_LOCAL   = 4;   // the env's c_LOCAL (Q16)
constexpr int EOS_PRE_FAIL      = 5;   // 1: the env selected a refused fold-table row
constexpr int EOS_PRE_SLOTS     = 8;
struct EosBooksDev {
    unsigned long long* kick = nullptr;      // (N, KICK_CNT_SLOTS)
    unsigned long long* flux = nullptr;      // (N, FLUX_CNT_SLOTS)
    unsigned long long* ecnt = nullptr;      // (N, EOS_ECNT_SLOTS)
    unsigned long long* rail = nullptr;      // (N, EOS_BOOK_GAS_MAX), by cons index
    unsigned long long* tcnt = nullptr;      // (N, EOS_TCNT_SLOTS), by trace position
    unsigned long long* pre = nullptr;       // (N, EOS_PRE_SLOTS)
    unsigned long long* nsub_max = nullptr;  // (1,) the batch's max n_sub
    void* zero_begin = nullptr;              // the contiguous span the tick zeroes
    size_t zero_bytes = 0;
};

// THE RESIDENT EOS STAGE over an (N, h, w) batch (S8a Path A; #29 P5 — design
// §4.2): every plane is the batch's (N, h, w) device buffer (env 0's address,
// the envs contiguous; gas (N, n_gases, h, w)); the pre-stage runs on the
// DEVICE (per-env reductions, the per-env fold into a FOLD-TABLE row, cap2 /
// absorb / coeffE / coeffS built from the resident planes); both substep loops
// run to the batch's max n_sub with per-env masks; the books land per env in
// `books` (no D2H here: the tick's one books D2H folds them). One counted 8-byte
// D2H of max(n_sub) per tick (P9 replaces it with a WHILE node). `solver` is the
// batch's config (a batch is homogeneous); `div_source_host` holds N entries,
// each an env's (h, w) blast-source plane or nullptr. An ambient map runs only
// at N = 1 (the sky bracket is the host's: ACCEPTED GAP). Returns max(n_sub).
int eos_step_resident(
    const EOSSolver& solver,
    int N, int h, int w, float dt,
    const bool* gas_conservative, int n_gases,
    const int32_t* n_amb, int32_t p_amb, bool ambient_mode,
    const int32_t* const* div_source_host,
    int32_t* d_atmosphere, int32_t* d_wave_p,
    int32_t* d_wind_x, int32_t* d_wind_y,
    int32_t* d_temperature, int32_t* d_gas_base,
    const bool* d_solid, const bool* d_is_vacuum,
    const float* d_dyn_permeability, const float* d_dyn_wave_absorb,
    const bool* d_is_ambient,
    const int32_t* d_sponge_sigma, const int32_t* d_sponge_udamp,
    const bool* d_thermal_solid,
    int64_t* d_gas_energy,
    const EosBooksDev& books);

// #29 P5: how many max(n_sub) reads the resident EOS made in this process (the
// counted per-tick sync; the capture test names it as the one exception).
long long eos_nsub_reads();

// TEST-ONLY (#29 P5, design §5.1's first check): the device pre-stage vs the
// host pre-stage at N = 1 on the same tick-entry planes (n_sub, c_LOCAL,
// dt_s_q, cap2, absorb_q, coeffE / coeffS, p_prev); returns the mismatch count
// (0 == bit-identical), the details in *report.
long long eos_prestage_parity(
    const EOSSolver& solver,
    const int32_t* atmosphere, const int32_t* wind_x, const int32_t* wind_y,
    const int32_t* temperature, const int32_t* gas,
    const bool* gas_conservative, int n_gases,
    const bool* solid, const bool* is_vacuum, const bool* thermal_solid,
    const float* dyn_permeability, const float* dyn_wave_absorb,
    int h, int w, float dt, std::string* report);

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
