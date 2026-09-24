#pragma once
// ============================================================================
// CUDA-S8a — the shared launch-core header (residency, STEP B).
// ============================================================================
//
// docs/cuda_s8a_residency_spec_2026-07-19.md §4 STEP B. For each solver family
// we factor a NON-anonymous `*_launch_resident(...)` that takes DEVICE pointers
// + persistent scratch + scalars and does the kernel launches ONLY — no
// cudaMalloc, no cudaMemcpy (H2D/D2H), no cudaFree, no cudaDeviceSynchronize.
// Allocation and transfer are the caller's job:
//
//   * The existing per-call `*_step` wrappers keep working unchanged — they now
//     allocate + H2D, call the matching `*_launch_resident`, sync, D2H, free.
//     So the launch body is SHARED, never duplicated, and the live per-call GPU
//     path (+ the existing per-kernel gates) run the byte-for-byte same kernel
//     sequence and arithmetic as before.
//   * STEP D's `step_resident(...)` (physics_engine.cpp) owns the persistent
//     device fields (CuPy-backed, passed in as raw pointers) + persistent
//     scratch (lazily allocated, keyed by (h,w)) and calls these cores back to
//     back with ZERO mid-tick transfers.
//
// Bit-identity is the eventual gate (tol 0, Berlin): the launch cores move
// allocation/transfer OUT of the hot path; they change NO math. Kernel order,
// scalar precompute, and arithmetic are identical to the per-call path.
//
// Plain C++ declaration header (no CUDA types in the signatures — raw int32_t*/
// int64_t*/bool* device pointers) so the .cpp TUs (physics_engine.cpp,
// bindings.cpp, cl.exe-compiled) can include it. Each `*_launch_resident` is
// DEFINED in its family's .cu. Compiled only when BREACH_CUDA.
//
// NOTE: this header is being populated family-by-family across STEP B commits.
// Only the cores listed below are wired; the rest land in subsequent WIP checkpoints.
#include <cstdint>

class EOSSolver;

namespace breach_cuda {

// ---- water (cuda_water.cu) --------------------------------------------------
// One water substep, LAUNCH ONLY, in place on d_depth/d_vx/d_vy. Mirrors
// water_step's K1..K8 sequence exactly (same host scalar precompute, same
// kernels, same order). Scratch buffers (d_surface/d_dq_e/d_dq_s/d_scale =
// (h*w) int32; d_fx/d_fy = (h*w) int64) are caller-owned + persistent. No
// malloc/transfer/sync. d_floor / d_atm nullable (as in water_step).
void water_launch_resident(
    int32_t* d_depth, int32_t* d_vx, int32_t* d_vy,
    const int32_t* d_floor,   // nullable -> flat zero
    const int32_t* d_atm,     // nullable -> no head term
    const bool* d_solid,
    int32_t* d_surface, int32_t* d_dq_e, int32_t* d_dq_s, int32_t* d_scale,
    int64_t* d_fx, int64_t* d_fy,
    int h, int w, float dt, float tilt_x, float tilt_y,
    float g, float damping, float dx, float k_p, float v_max, float depth_eps);

// The whole water SUBSTEP LOOP, device-resident (S8a Path B FLOOR item 2). Runs
// water_launch_resident `n_sub` times back-to-back on the SAME persistent device
// buffers, with C++-owned persistent scratch (allocated once, keyed by (h,w) —
// NO per-substep cudaMalloc/H2D/D2H), one cudaDeviceSynchronize at the end. This
// is the exact substep loop PhysicsEngine::step_water runs, minus the transfer
// tax that today's per-call water_step pays on EVERY substep. d_depth/d_vx/d_vy
// are the persistent CuPy-owned water fields; d_floor/d_atm nullable (as in
// water_step); the caller (step_resident) owns the D2H that follows. Bit-identical
// to the per-call path (same shared launch core, same host scalar precompute).
void water_substeps_resident(
    int32_t* d_depth, int32_t* d_vx, int32_t* d_vy,
    const int32_t* d_floor, const int32_t* d_atm, const bool* d_solid,
    int h, int w, int n_sub, float wdt, float tilt_x, float tilt_y,
    float g, float damping, float dx, float k_p, float v_max, float depth_eps);

// ---- smoke (cuda_smoke.cu) --------------------------------------------------
// One smoke/trace step, LAUNCH ONLY, in place on d_gas. Mirrors smoke_step's
// K1..K4 sequence exactly (diffusion Laplacian -> diffuse apply -> D2D post-
// diffusion snapshot -> SL advect -> clamp). Scratch (d_lap/d_src = h*w int32)
// is caller-owned + persistent. No malloc/H2D/D2H/sync. d_amb nullable (space).
void smoke_launch_resident(
    int32_t* d_gas,
    const int32_t* d_wind_x, const int32_t* d_wind_y,
    const bool* d_obstacles, const bool* d_is_wall, const bool* d_is_vacuum,
    const float* d_perm, const bool* d_is_ambient,
    int32_t* d_lap, int32_t* d_src,
    int h, int w, float dt,
    float d_smoke, float wind_diffusion_scale, float advection_rate);

// The whole per-tick TRACE-PLANE LOOP, device-resident (S8a Path B FLOOR item 3).
// For each non-conservative gas plane: smoke_launch_resident (once per tick, on
// the final corrected wind) + decay (a device kernel, bit-identical to the CPU
// mul_q16 shrink). P-T0 (energy-books arc, design §2.6 — the trace 0% ruling):
// the decay->inert_N2 credit this kernel used to pay is DELETED — decayed mass
// simply VANISHES, same as the CPU twin (physics_engine.cpp's run_substeps
// trace loop); `inert_n2_idx` stays a parameter for ABI/back-compat. Persistent
// scratch owned here (lap/src, keyed by (h,w)); one cudaDeviceSynchronize at
// the end; NO per-plane cudaMalloc/H2D/D2H. The all-zero-plane `.any()` skip is
// DROPPED — smoke_step on an all-zero plane is an arithmetic no-op (the EOS
// P6.5 device precedent), so processing every trace plane is bit-identical to
// the CPU skip. gas_conservative / gas_diffusion / gas_decay are the small
// (n_gases,) HOST columns (control-flow + the per-plane diffusion/decay
// scalars, exactly as run_substeps reads them).
void trace_smoke_resident(
    int32_t* d_gas_base,
    const int32_t* d_wind_x, const int32_t* d_wind_y,
    const bool* d_solid, const bool* d_is_vacuum, const float* d_perm,
    const bool* d_is_ambient,
    int h, int w, int n_gases, int inert_n2_idx,
    const bool* gas_conservative, const float* gas_diffusion,
    const float* gas_decay,
    float dt, float advection_rate, float wind_diffusion_scale);

// ---- kick + energy flux (cuda_kick_compression.cu) — S8a Path A ------------
// The per-tick scalar folds the kick kernel (K1) consumes, factored to ONE
// transcription (design §3.2.3): kick_scalar_folds computes them from the
// EOSSolver config floats through the IDENTICAL double expressions the CPU
// step() folds (/fp:strict host pass); consumed by BOTH the per-call
// eos_kick_compression entry and the resident launch core below.
//
// gas-energy conservation arc #54, P-G2: K2 (the old step-4c compression-work
// kernel) is DELETED — its dials (t_min/t_max_phys/t_work_clamp/u_max/
// gamma_m1/k_drag_heat_frac/c_v/n_work_ref) retire with it (D5/D10/D11). K1
// now carries the per-stage KE brackets (design §2.3) straight to
// `gas_energy`, so this struct shrinks to exactly what K1 reads.
struct KickScalarFolds {
    int32_t n_floor_q    = 0;
    // VELOCITY-CLAMP (P-V1, D3): u_max² (Q32.32) for the kick's cap_is_umax
    // test against the per-cell cap2_plane.
    int64_t u_max2_q32   = 0;
    int32_t inv_2dx_q    = 0;
    int32_t absorb_dt_q  = 0;   // absorb_strength·dt (the §2.5 hoist's factor)
    int64_t Kdt_raw      = 0;   // (K·2^16)·dt at raw scale, 128-bit staged
    // P-E3 (energy-books arc, design §2.8): the interior-drag scalar folds,
    // the SAME per-tick-not-per-cell idiom absorb_dt_q already uses.
    int32_t kd_q         = 0;   // quantize(k_drag * dt) — dormancy branches on this
    // drag-law v2 (docs/drag_law_v2_design_2026-08-23.md §2/§7): kd2_q beside
    // kd_q, plus the MANDATORY dormant-dial-guarded rad_dead_q32 = U0^2
    // (U0 = ceil(2^16/kd2_q)) — the calm-cell fast-path threshold stage Q's
    // divide skips below (0 when kd2_q == 0 — an unconditional ceil-divide
    // would be a divide-by-zero at the shipped config).
    int32_t kd2_q        = 0;   // quantize(k_drag2 * dt) — dormancy branches on this
    int64_t rad_dead_q32 = 0;   // U0^2, Q.32; 0 iff kd2_q == 0
    // arc #54 §2.1/§2.3: the derived specific-KE-to-ΔT constant, folded as
    // make_recip(1/k_ke) (a Q.32 wide reciprocal — see eos_solver.cpp's fold
    // comment for why the plain Q16.16 constant would lose ~6 bits).
    int64_t k_ke_recip_q32 = 0;
};
KickScalarFolds kick_scalar_folds(
    float dt, float c_max, float dx, float adiabatic_index,
    float absorb_strength, float n_floor_solver, float u_max,
    float k_drag, float k_drag2,
    // arc #54 §2.1: T_AMB_K, adiabatic_index and c_max fold k_ke — the
    // SAME double expression eos_solver.cpp's step() uses.
    float t_amb_k);

// The step-4 kick, LAUNCH ONLY, on DEVICE pointers (the CPU pass boundary),
// no malloc, no transfer, no memset, no sync, no digest. d_ntot is the
// post-substep Dalton N_total plane; d_absorb_q the host-hoisted §2.5 absorb
// plane; d_gas_energy the arc #54 conserved field (in/out — K1's KE brackets
// debit/credit it directly, design §2.3); d_cnt the KICK_CNT_SLOTS-slot
// counter buffer THE CALLER ZEROES each tick (design §3.2.5) — see
// cuda_kick_compression.cu's slot-map comment. d_amb/d_udamp nullable
// (space / no band). The per-call eos_kick_compression wraps this same core
// with its existing H2D/memset/D2H/digest flow — one kernel transcription,
// both paths.
constexpr int KICK_CNT_SLOTS = 15;
void kick_compression_launch_resident(
    int32_t* d_wind_x, int32_t* d_wind_y,
    int64_t* d_gas_energy,         // arc #54 §2.2/§2.3, in/out
    const int32_t* d_p_new, const int32_t* d_ntot, const int32_t* d_absorb_q,
    const bool* d_solid, const bool* d_is_vacuum,
    const KickScalarFolds& folds,
    const int64_t* d_cap2_plane,   // VELOCITY-CLAMP (P-V1, D2v2), (h,w), >= 0
    unsigned long long* d_cnt, int h, int w,
    const bool* d_is_ambient, const int32_t* d_sponge_udamp,
    // THERMAL-MASS AXIS, P-EOS: the medium mask the KE brackets export to
    // (ts cells carry no gas_energy — design §2.3 F5). The P2 device-fallback
    // idiom applies — the caller passes `d_thermal_solid ? d_thermal_solid :
    // d_solid`, so the legacy path allocates and copies nothing.
    const bool* d_ts = nullptr);

// ---- the face-flux energy step (cuda_kick_compression.cu K3) — P-G2 -------
// Replaces the retired step-4c compression work (design §2.4/§2.5, Kwatra
// eq. 3 with the eq. 15 face pressure). The per-tick scalar folds K3 and the
// once-per-tick recovery consume, ONE transcription (the KickScalarFolds
// precedent).
struct EnergyFluxScalarFolds {
    int32_t t_amb_q      = 0;   // quantize(T_AMB_K) raw
    int32_t t_min_q      = 0;   // quantize(T_MIN) raw
    int32_t t_max_phys_q = 0;   // quantize(T_MAX_PHYS) raw
    int32_t c_q          = 0;   // quantize(C) — p* = C·E, C = 1/T_AMB_K
    int32_t k_flux_q     = 0;   // the ONE per-sub-cycle flux constant (§2.4)
    int64_t flux_pu_cap  = 0;   // FLUX_MAG_CAP / k_flux_q — the int64 corner
    int     n_sub        = 1;   // the SAME substep schedule the kick/SL share
};
EnergyFluxScalarFolds energy_flux_scalar_folds(
    float dt, float dx, float adiabatic_index, float t_amb_k, float c_value,
    float t_min, float t_max_phys, int n_sub);

// The face-flux energy step + the once-per-tick recovery, LAUNCH ONLY, on
// DEVICE pointers — runs AFTER K_store_atm (the step-5 un-shift: the flux
// step consumes the ABSOLUTE solved pressure, design §2.4). d_atmosphere is
// the absolute p^{n+1} (post-unshift); d_gas_energy the conserved field,
// in/out; d_n_total the post-substep Dalton N; d_ts the export mask (walls to
// the energy step, design F4); d_e0/d_pcur/d_s_plane are caller-owned
// per-tick scratch, (h,w) int64/int32/int32 (design §2.5 — the two-pass
// gather shape, no atomics, no face buffer). d_cnt the FLUX_CNT_SLOTS-slot
// counter buffer the caller zeroes each tick.
constexpr int FLUX_CNT_SLOTS = 11;
void energy_flux_launch_resident(
    int64_t* d_gas_energy, int32_t* d_temperature,
    const int32_t* d_atmosphere,
    const int32_t* d_wind_x, const int32_t* d_wind_y,
    const int32_t* d_n_total,
    const bool* d_solid, const bool* d_is_vacuum, const bool* d_ts,
    const bool* d_is_ambient,
    const EnergyFluxScalarFolds& folds,
    int64_t* d_e0, int32_t* d_pcur, int32_t* d_s_plane,
    unsigned long long* d_cnt, int h, int w);

// ---- the radiation sweep (cuda_radiation_sweep.cu) — ray-engine-v2 P4 -------
// The sweep's LAUNCH CORE, born (N, h, w) with N = 1 today (RL-batch habits,
// docs/rl_env_arc_proposal_2026-08-27.md §A): every plane is (N, h, w), every
// per-env scalar is a device array of length N read by env index, and the
// per-env counters live in an (N, RADIATION_SWEEP_CNT_SLOTS) block. Launch
// only — no malloc, no transfer, no sync; the per-call radiation_sweep_step
// (cuda_radiation_sweep.h) wraps it.
//
// An ingress violation is not thrown from here (a device cannot, and a host
// check would be host-side gating, §A rule 4): the init and pre-pass kernels
// COUNT it into the env's slots, and every later kernel skips an env whose
// count is non-zero, so a rejected env's output planes are left exactly as the
// caller had them — RadiationSweep::run's contract, held by a per-env mask.
// The caller reads the slots after its sync (radiation_sweep_step throws).
//
// Slots, per env (int64, two's complement through the unsigned atomics):
//   0 cells violating 0 <= a <= d <= ONE          (a count)
//   1 cells whose ambient level is outside [0, E°[0]]   (a count)
//   2 scalar violations, a bitmask: 1 = k_leak_q outside [0, ONE],
//     2 = vac_level above E°[0] (only when the level is derived),
//     4 = a heat_absorb_q16 outside [0, RadiationSweep::HEAT_ABSORB_Q_MAX]
//         (P5a; the table is shared, so every env carries the bit)
//   3 min_stream   4 max_stream   (RadiationSweep's telemetry, via 64-bit
//                                   atomicMin / atomicMax — order-free)
// The core INITIALISES the block itself (its first kernel); the caller never
// zeroes it.
constexpr int RADIATION_SWEEP_CNT_SLOTS = 5;
constexpr int RS_SLOT_BAD_EXTINCTION = 0;
constexpr int RS_SLOT_BAD_AMBIENT    = 1;
constexpr int RS_SLOT_BAD_SCALARS    = 2;
constexpr int RS_SLOT_MIN_STREAM     = 3;
constexpr int RS_SLOT_MAX_STREAM     = 4;
constexpr int RS_BAD_K_LEAK      = 1;
constexpr int RS_BAD_VAC_LEVEL   = 2;
constexpr int RS_BAD_HEAT_ABSORB = 4;
//
//   d_temperature, d_heat_atten_q, d_dyn_heat_atten_q, d_heat_inv_shift,
//   d_thermal_solid : (N, h, w) — RadiationSweep::run's inputs
//   d_amb_level     : (N, h, w) int64, NULLABLE — run()'s ambient plane; null
//                     derives it per cell from d_is_vacuum (nullable too: all
//                     interior) and d_vac_level, derive_ambient's twin
//   d_e_table       : (E_TABLE_SIZE,) int64 — shared by every env (a pure
//                     function of the calibration dials, not per-env state)
//   d_vac_level, d_k_leak_q, d_t_amb_q : (N,) per-env scalars
//   THE GAS EXTINCTION (P5a, design v3 §6.3) — all three pointers or none
//   (none, with n_gases == 0: the pre-P5a sweep):
//   d_gas           : (N, n_gases, h, w) int32 Q16.16 — the gas planes
//   n_gases         : 0 <= n_gases <= RadiationSweep::N_GAS_PLANES_MAX (host)
//   d_heat_absorb_q16 : (n_gases,) int32 Q16 — shared by every env (config)
//   d_n_bulk        : (N, h, w) int32 Q16.16 — the bulk (O2 + N2) count
//   n_floor_q, recip_cv : THE GAS CURRENCY (P5b), two host scalars shared by
//                     every env (the temperature fold's config dials, like the
//                     heat_absorb table), required > 0 with the gas group —
//                     the Fleck pre-pass's gas arm prices every absorbing gas
//                     cell in them (radiation_sweep.h fleck_L_gas_q)
//   d_outflow       : (N, n_ordinates, h, w) int64 scratch (every cell is
//                     written before it is read; never needs clearing)
//   d_amb_m, d_ex_cell : (N, h, w) int64 scratch
//   d_f_q24         : (N, h, w) int32 — the Fleck plane (observable)
//   d_a_eff, d_d_eff : (N, h, w) int32 scratch — the extinction planes the
//                     wavefronts READ (the smoke term folded in, P5a)
//   d_rad_*         : (N, h, w) int64 — OVERWRITTEN (zeroed here first)
//   d_cnt           : (N, RADIATION_SWEEP_CNT_SLOTS) int64
// Throws std::invalid_argument on an unsupported (n_ordinates, transport),
// an n_env above 65535 (the wavefront grid's blockIdx.z carries the env), a
// partial gas group, an n_gases outside [0, N_GAS_PLANES_MAX] or a
// non-positive gas currency with the group — host control flow, checked
// before any launch. Returns the launch count.
int radiation_sweep_launch_resident(
    int n_env, int h, int w,
    const int32_t* d_temperature,
    const int32_t* d_heat_atten_q, const int32_t* d_dyn_heat_atten_q,
    const int32_t* d_heat_inv_shift, const bool* d_thermal_solid,
    const int64_t* d_amb_level, const bool* d_is_vacuum,
    const int64_t* d_e_table,
    const int64_t* d_vac_level, const int32_t* d_k_leak_q,
    const int32_t* d_t_amb_q,
    const int32_t* d_gas, int n_gases,
    const int32_t* d_heat_absorb_q16, const int32_t* d_n_bulk,
    int32_t n_floor_q, int64_t recip_cv,
    int transport, int n_ordinates, bool fleck_enabled,
    int64_t* d_outflow, int64_t* d_amb_m, int64_t* d_ex_cell, int32_t* d_f_q24,
    int32_t* d_a_eff, int32_t* d_d_eff,
    int64_t* d_rad_net, int64_t* d_rad_flux, int64_t* d_rad_amb,
    int64_t* d_rad_fluence,
    int64_t* d_cnt);

}  // namespace breach_cuda
