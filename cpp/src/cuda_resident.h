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
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

class EOSSolver;

namespace breach_cuda {

struct EnvSub;   // #29 P5 (cuda_resident_tick.h)

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
    float g, float damping, float dx, float k_p, float v_max, float depth_eps,
    // #29 P5: the batch (planes (N, h, w)) and the per-env active mask (N,)
    // uint8 — nullptr: every env runs; 0: that env does nothing.
    int n_env = 1, const uint8_t* d_active = nullptr);

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
    float g, float damping, float dx, float k_p, float v_max, float depth_eps,
    int n_env = 1, const uint8_t* d_active = nullptr);

// ---- the trace tail (cuda_bulk_transport.cu) — smoke transport v2 P2b ------
// (cuda_smoke.cu — smoke_launch_resident / trace_smoke_resident, the resident
// path's old semi-Lagrangian law — is DELETED: the trace planes ride the bulk
// face flux inside the EOS substeps, stages 3b/3c, on every path.)
//
// The once-per-tick TRACE TAIL's LAUNCH CORE (design §2.2-§2.4, §5
// "Resident"): stranded zeroing (sink) -> Jacobi magnitude-truncated face
// diffusion (vent onto vacuum / ring) -> ceil-rounded decay (its own kernel,
// over the shared rule in trace_decay.h). The kernel-for-loop transcription of
// trace_tail (bulk_transport.cpp — THE oracle). Born (N, h, w) with N = 1
// today (RL-batch habits §A): launch only — no malloc, no transfer, no sync;
// the core zeroes its own book block (its first kernel).
//
//   d_gas        : (N, n_gases, h, w) int32 Q16.16, in/out (only the planes
//                  whose is_trace fold is set are touched)
//   d_solid, d_is_vacuum : (N, h, w); d_is_ambient (N, h, w) NULLABLE (space)
//   d_perm       : (N, h, w) float — dyn_permeability (the face quantize is the
//                  CPU tail's, per face, on device)
//   d_scal       : (N, n_gases, TRACE_TAIL_SCAL) int32 per-env, per-gas folds
//                  [is_trace, dd_q, frac_q] (dt is a per-env scalar)
//   d_s0         : (N, n_gases, h, w) int32 scratch (the Jacobi snapshot)
//   d_cnt        : (N, n_gases, TRACE_TAIL_CNT_SLOTS) unsigned long long books
//                  [vent, sink, decay], OVERWRITTEN (integer atomicAdd)
// Returns the launch count.
constexpr int TRACE_TAIL_SCAL      = 3;
constexpr int TRACE_TAIL_CNT_SLOTS = 3;
constexpr int TRACE_TAIL_VENT  = 0;
constexpr int TRACE_TAIL_SINK  = 1;
constexpr int TRACE_TAIL_DECAY = 2;
int trace_tail_launch_resident(
    int n_env, int h, int w, int n_gases,
    int32_t* d_gas,
    const bool* d_solid, const bool* d_is_vacuum, const bool* d_is_ambient,
    const float* d_perm,
    const int32_t* d_scal,
    int32_t* d_s0,
    unsigned long long* d_cnt);

// (#29 P5: the N == 1 driver trace_tail_resident is DELETED — the batched
// tick's resident_trace_tail (cuda_resident_tick.h) is the one driver.)

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

// #31: the blast's divergence source comes off div(u*), LAUNCH ONLY, on DEVICE
// pointers, (N, h, w)-shaped (RL-batch habits; N == 1 today). The caller launches
// it only on a tick with a source. d_is_ambient nullable (space).
void div_source_launch_resident(
    int32_t* d_div_u, const int32_t* d_div_source,
    const bool* d_solid, const bool* d_is_vacuum, const bool* d_is_ambient,
    int N, int h, int w);

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
    const bool* d_ts = nullptr,
    // #29 P5: the batch -- every plane (N, h, w), the counters (N, cstride)
    // (0 = KICK_CNT_SLOTS). The kick runs once per tick on every env.
    int n_env = 1, size_t cstride = 0);

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
    unsigned long long* d_cnt, int h, int w,
    // #29 P5: the batch -- planes (N, h, w), counters (N, cstride) (0 =
    // FLUX_CNT_SLOTS); `es` the per-env fold-table rows (nullptr: unbatched,
    // the folds' own scalars and n_sub) and `n_cycles` the batch's max n_sub --
    // an env sits out every sub-cycle past its own n_sub.
    int n_env = 1, size_t cstride = 0, const EnvSub* es = nullptr,
    int n_cycles = 0);

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
//   P6a, APPENDED (the light channels; zero unless the light group rode):
//   5 cells violating 0 <= a_c <= d_c <= ONE on some light channel (a count)
//   6..8   light emit (R, G, B)      9..11  light absorb
//   12..14 light ring_in             15..17 light ring_out   (the light books,
//          exact integer sums: emit + ring_in == absorb + ring_out per channel)
//   18 light min_stream   19 light max_stream
// The core INITIALISES the block itself (its first kernel); the caller never
// zeroes it.
constexpr int RADIATION_SWEEP_CNT_SLOTS = 20;
constexpr int RS_SLOT_BAD_EXTINCTION = 0;
constexpr int RS_SLOT_BAD_AMBIENT    = 1;
constexpr int RS_SLOT_BAD_SCALARS    = 2;
constexpr int RS_SLOT_MIN_STREAM     = 3;
constexpr int RS_SLOT_MAX_STREAM     = 4;
constexpr int RS_SLOT_BAD_LIGHT_EXTINCTION = 5;   // P6a
constexpr int RS_SLOT_LIGHT_EMIT     = 6;         // P6a: + channel
constexpr int RS_SLOT_LIGHT_ABSORB   = 9;
constexpr int RS_SLOT_LIGHT_RING_IN  = 12;
constexpr int RS_SLOT_LIGHT_RING_OUT = 15;
constexpr int RS_SLOT_LIGHT_MIN_STREAM = 18;
constexpr int RS_SLOT_LIGHT_MAX_STREAM = 19;
constexpr int RS_BAD_K_LEAK      = 1;
constexpr int RS_BAD_VAC_LEVEL   = 2;
constexpr int RS_BAD_HEAT_ABSORB = 4;

// P6a: THE LIGHT GROUP on the device (RadiationSweep's LightChannels, one
// pointer per plane, (N, ...)-shaped like everything else here; a plain host
// struct of device pointers, so the declaration stays free of CUDA types).
// Shared by every env: the L° table and the two smoke columns (config, like
// heat_absorb). The light transport is a host scalar. With it the wavefronts
// become the step transport's ANTI-DIAGONALS for EVERY ordinate (h + w - 1
// launches): topological for step and shear alike, so heat rides them with
// its integers unchanged (the gather form). Without it (nullptr) the core is
// the heat-only one, launch for launch.
struct RadiationLightDev {
    const int32_t* d_light_atten_q     = nullptr;  // (N, h, w, 3) Q16
    const int32_t* d_dyn_light_atten_q = nullptr;  // (N, h, w, 3) Q16
    const int64_t* d_l_table           = nullptr;  // (3, E_TABLE_SIZE), shared
    const int32_t* d_light_absorb_q16  = nullptr;  // (n_gases, 3), shared, nullable
    const int32_t* d_light_glow_q16    = nullptr;  // (n_gases, 3), shared, nullable
    int transport = 0;                             // RadiationSweep::TRANSPORT_STEP
    // P6b: THE SKY and THE CONE EMITTERS (radiation_sweep.h LightChannels),
    // both nullable. The sky is the virtual ring's per-ordinate outflow, PER
    // ENV; the cones arrive as the host-built per-cell injection
    // (RadiationSweep::build_cone_injection -- the projection has one
    // implementation): an index plane (-1 off an emitter) and the slots it
    // points into, slots numbered across every env.
    const int64_t* d_sky        = nullptr;         // (N, n_ordinates, 3), nullable
    const int32_t* d_cone_index = nullptr;         // (N, h, w), nullable
    const int64_t* d_cone_inj   = nullptr;         // (n_slots, n_ordinates, 3)
    // G3 F4b: THE FIRE'S LIGHT FLICKER as the host-built per-cell Q16 plane
    // (RadiationSweep::build_emit_mult -- ONE where no row names the cell),
    // PER ENV; nullable (no flicker: the pre-F4b pre-pass, integer for integer).
    const int32_t* d_emit_mult  = nullptr;         // (N, h, w), nullable
    // scratch
    int64_t* d_l_outflow = nullptr;                // (N, n_ordinates, h, w, 3)
    int64_t* d_l_emit    = nullptr;                // (N, h, w, 3)
    int32_t* d_l_d       = nullptr;                // (N, h, w, 3) the d_c the loop reads
    int32_t* d_l_g       = nullptr;                // (N, h, w, 3) the glow coefficient
    // outputs, OVERWRITTEN (zeroed by the core first)
    int64_t* d_light_q      = nullptr;             // (N, h, w, 3)
    int64_t* d_light_flux_q = nullptr;             // (N, h, w, 2)
    int64_t* d_light_glow   = nullptr;             // (N, h, w, 3)
};
//
//   d_temperature, d_heat_atten_q, d_dyn_heat_atten_q, d_heat_inv_shift,
//   d_thermal_solid : (N, h, w) — RadiationSweep::run's inputs
//   d_amb_level     : (N, h, w) int64, NULLABLE — run()'s ambient plane; null
//                     derives it per cell from d_is_vacuum (nullable too: all
//                     interior) and d_vac_level, derive_ambient's twin
//   d_e_table       : (E_TABLE_SIZE,) int64 — shared by every env (a pure
//                     function of the calibration dials, not per-env state)
//   e_fine_bits     : THE TABLE'S CURRENCY (#78), a host scalar shared by every
//                     env like the table: 2^e_fine_bits of its integers per heat
//                     count, in [0, E_FINE_BITS]. The pre-pass's two Fleck arms
//                     convert out of it; the planes come out in it
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
// an e_fine_bits outside [0, E_FINE_BITS] (#78),
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
    const int64_t* d_e_table, int e_fine_bits,
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
    int64_t* d_cnt,
    const RadiationLightDev* light = nullptr);   // P6a: nullptr = heat only

// ---- combustion (cuda_combustion.cu) — #71, residency arc #29 P2 -----------
// The LAUNCH CORE of CombustionSolver::step (combustion.cpp — THE oracle):
// the pass-entry snapshot + scratch zeroing, Pass A (air cells: the reverse
// relaxation, the demand, the exact-integer split, the O2 debit, the donor's
// energy leaving with its O2, the H_bed deposit), Pass B (sources: the fuel
// payment, where the products and their energy land), Pass C (air cells: the
// soot / N2 / heat deposit into `gas_energy` through the railed seam). Born
// (N, h, w) with N = 1 today (RL-batch habits §A): launch only — no malloc, no
// transfer, no sync; the core zeroes its own scratch and book block (its first
// kernel). The per-call `combustion_step` (cuda_combustion.h) allocates,
// uploads, calls this, downloads; P4a's resident driver calls it on the
// persistent device fields.
//
// The scalars every env shares (config dials, baked on the host by
// `combustion_bake_scalars` with the CPU solver's own quantize / make_recip
// lines) travel by value; the per-env ones (`burn_cap_q` from the env's dt,
// `t_amb_q`) are (N,) device arrays. An env whose burn_cap_q <= 0 burns nothing
// (the CPU's early return, as a per-env mask).
struct CombustionScalars {
    int32_t o2_thresh_q = 0;
    int32_t x_ext_q = 0;
    int64_t recip_x_span = 0;
    int32_t x_degenerate = 0;          // bool
    int32_t x_n_floor = 0;
    int32_t p_ext_q = 0;               // o2_pressure::Factor, field for field
    int32_t p_full_q = 0;
    int32_t p_inv_span_q = 0;
    int32_t fire_T_ext_q = 0;
    int64_t recip_T_span = 0;
    int32_t hotf_cap_q = 0;
    int32_t H_bed_m_q = 0;
    int32_t H_bed_shift = 0;
    int32_t fuel_per_o2_q = 0;
    int32_t soot_yield_q = 0;
    int32_t H_fuel_m_q = 0;
    int32_t H_fuel_shift = 0;
    int64_t recip_cv = 0;
    int32_t n_floor_q = 0;
    int32_t t_max_phys_q = 0;
    int32_t w_hop_q[4] = {0, 0, 0, 0};  // combustion_draw::R_MAX + 1
};

// The host bake: every shared scalar, line for line the CPU step's own
// (quantize / make_recip / o2_pressure::bake in double, then integers).
CombustionScalars combustion_bake_scalars(
    float o2_thresh_burn, float o2_frac_ext, float o2_frac_full,
    float fire_T_ext, float fire_T_span, float hotf_cap,
    float H_BED_M, int H_BED_SHIFT, float fuel_per_o2, float soot_yield,
    float H_FUEL_M, int H_FUEL_SHIFT, float c_v, float n_floor_heat,
    float T_MAX_PHYS, int32_t p_ext_q, int32_t p_full_q);

// The draw geometry (combustion.h's `combustion_draw` tables) into the device's
// __constant__ banks: a host->symbol transfer, so the CALLER's job, once per
// process before the first launch (the per-call wrapper does it every call).
void combustion_upload_tables();

// The per-env book block, (N, COMB_CNT_SLOTS) unsigned long long, OVERWRITTEN
// (zeroed by the core). Signed books ride two's complement (integer atomicAdd
// is order-free modulo 2^64), read back as int64. Each slot is the
// CombustionSolver member of the same name.
constexpr int COMB_CNT_SLOTS = 15;
enum : int {
    COMB_HEAT_FLOOR_HITS  = 0,   // heat_floor_hits
    COMB_T_MAX_PHYS_HITS  = 1,   // t_max_phys_hits (both deposit sites)
    COMB_DEPOSIT_DROP     = 2,   // e_deposit_drop_sum
    COMB_TRACE_MINT       = 3,   // the soot mint's books (#88 P2) ...
    COMB_TRACE_CLIP       = 4,
    COMB_TRACE_NEG        = 5,
    COMB_E_DRAW           = 6,   // e_comb_draw_sum
    COMB_E_MINT           = 7,   // e_comb_mint_sum
    COMB_E_DELIVER        = 8,   // e_comb_deliver_sum
    COMB_E_SOOT_SHED      = 9,   // e_soot_shed_sum
    COMB_E_TS_PRODUCTS    = 10,  // e_ts_products_sum
    COMB_E_EXPORT         = 11,  // e_comb_export_sum
    COMB_E_HEAT           = 12,  // e_comb_heat_sum
    COMB_E_RAIL           = 13,  // e_comb_rail_sum (signed)
    COMB_E_SOLID_HEAT     = 14,  // e_comb_solid_heat_sum (signed)
};

//   d_gas           : (N, n_gases, h, w) int32 — only the o2 / inert_n2 /
//                     black_smoke planes are read or written
//   d_temperature, d_wall_hp : (N, h, w) int32, in/out
//   d_gas_energy    : (N, h, w) int64, in/out, NULLABLE (null = the pre-#54
//                     T-form deposit — the direct-binding path; then
//                     d_e_slot / d_e_dep may be null too)
//   d_fire, d_ignition_temp_q16 : (N, h, w) int32
//   d_flammable, d_solid, d_is_vacuum : (N, h, w) bool
//   d_is_ambient, d_thermal_solid : (N, h, w) bool, NULLABLE
//   d_heat_inv_shift : (N, h, w) int32, NULLABLE (the object deposit needs it
//                     AND d_thermal_solid)
//   d_heat          : (N, h, w) int32, NULLABLE (the H_bed deposit)
//   d_dem_acc       : (N, acc_depth, h, w) int32, NULLABLE (acc_depth >=
//                     slot_count(draw_r), host-checked)
//   d_perm_q        : (N, h, w) int32 Q16 — the permeability quantized once
//                     (the CPU's pass-entry quantize), NULLABLE (= 1.0)
//   d_fire_T_ext_plane, d_fuel_per_o2_plane, d_atmosphere : (N, h, w) int32,
//                     NULLABLE (the scalar fallbacks / g == FP_ONE)
//   d_burn_cap_q, d_t_amb_q : (N,) int32 per-env scalars
//   d_tsnap         : (N, h, w) int32 scratch
//   d_alloc         : (N, slot_count(draw_r), h, w) int32 scratch
//   d_dep           : (N, 5, h, w) int32 scratch
//   d_e_slot        : (N, slot_count(draw_r), h, w) int64 scratch (with energy)
//   d_e_dep         : (N, 5, h, w) int64 scratch (with energy)
//   d_cnt           : (N, COMB_CNT_SLOTS) unsigned long long, OVERWRITTEN
// Throws std::invalid_argument on a draw_r outside [1, R_MAX] or an acc_depth
// below slot_count(draw_r) (host control flow, before any launch). Returns the
// launch count.
int combustion_launch_resident(
    int n_env, int h, int w, int n_gases,
    int o2_idx, int inert_n2_idx, int black_smoke_idx, int draw_r,
    int32_t* d_gas, int32_t* d_temperature, int32_t* d_wall_hp,
    int64_t* d_gas_energy,
    const int32_t* d_fire, const int32_t* d_ignition_temp_q16,
    const bool* d_flammable, const bool* d_solid, const bool* d_is_vacuum,
    const bool* d_is_ambient, const bool* d_thermal_solid,
    const int32_t* d_heat_inv_shift,
    int32_t* d_heat, int32_t* d_dem_acc, int acc_depth,
    const int32_t* d_perm_q,
    const int32_t* d_fire_T_ext_plane, const int32_t* d_fuel_per_o2_plane,
    const int32_t* d_atmosphere,
    const int32_t* d_burn_cap_q, const int32_t* d_t_amb_q,
    const CombustionScalars& sc,
    int32_t* d_tsnap, int32_t* d_alloc, int32_t* d_dep,
    int64_t* d_e_slot, int64_t* d_e_dep,
    unsigned long long* d_cnt);

// ---- temperature (cuda_temperature.cu) — #29 P4a ----------------------------
// The LAUNCH CORE of TemperatureSolver::step's GPU twin (temperature_solver.cpp
// — THE oracle): the capacity planes, Pass 0a (the vacuum / ring wipe), Pass 1
// (the radiation fold, solid and gas, with the maximum-principle clamp, then
// the heat deposit), Pass 2 (conduction, gather into d_temp_new, the swap as a
// stream-ordered D2D copy, the parked gas-side face sums applied), and — when
// `solid_books` — the solid-books snapshot on the final temperature. Born
// (N, h, w) with N = 1 today: launch only, no malloc / transfer / sync; the
// core zeroes its own counter block (its first kernel). The per-call
// temperature_step allocates, uploads, calls this, downloads; the resident
// tick (cuda_resident_tick.cu) calls it on the persistent device fields.
//
// The counter block, per env, TEMP_CNT_SLOTS unsigned long long (signed books
// two's complement through the unsigned atomics): slots 0..16 are the C_*
// energy slots of cuda_temperature.h's energy_counters_out in that pinned
// order, then the T_MAX_PHYS rail count, the Pass-1 LOW rail count and the
// solid-books snapshot (Σ cap_real·T over thermal solids; written only with
// `solid_books`).
constexpr int TEMP_CNT_ENERGY      = 17;   // == TEMPERATURE_ENERGY_SLOTS (asserted)
constexpr int TEMP_CNT_HITS        = 17;
constexpr int TEMP_CNT_LOW_HITS    = 18;
constexpr int TEMP_CNT_SOLID_BOOKS = 19;
constexpr int TEMP_CNT_SLOTS       = 20;

// The dials every env shares, baked on the host with the CPU step's own
// quantize / make_recip lines (temperature_bake_scalars; refuses a
// rad_fine_bits outside [0, E_FINE_BITS], #78).
struct TemperatureScalars {
    int32_t c_v_q = 0;
    int64_t recip_cv = 0;
    int32_t n_floor_q = 0;
    int32_t t_max_phys_q = 0;
    int     no_face = 0;
    int     rad_fine_bits = 0;
};
TemperatureScalars temperature_bake_scalars(float c_v, float n_floor_heat,
                                            float t_max_phys, int no_face,
                                            int rad_fine_bits);

//   d_temperature   : (N, h, w) int32, in/out
//   d_temp_new      : (N, h, w) int32 scratch (Pass 2's gather target)
//   d_heat, d_heat_inv_shift : (N, h, w) int32;  d_face_shift (N, h, w, 4)
//   d_solid, d_ts, d_is_vacuum : (N, h, w) bool (d_ts = the thermal medium
//                     mask, or d_solid for the documented fallback)
//   d_is_ambient    : (N, h, w) bool, NULLABLE (space)
//   d_n_src         : (N, h, w) int32 — the bulk N (or the atmosphere proxy)
//   d_rad_net       : (N, h, w) int64, NULLABLE (no radiation fold)
//   d_rad_fluence + d_e_table : the clamp's pair, both or neither (NULLABLE)
//   d_gas_energy    : (N, h, w) int64, in/out, NULLABLE (the pre-#54 T-form);
//                     with it d_de_gas (N, h, w) int64 scratch is required
//   d_t_amb_q       : (N,) int32 per-env scalar (read with gas_energy)
//   d_cap_used, d_cap_real : (N, h, w) int64 scratch
//   d_cnt           : (N, TEMP_CNT_SLOTS) unsigned long long, OVERWRITTEN
// Returns the launch count.
int temperature_launch_resident(
    int n_env, int h, int w,
    int32_t* d_temperature, int32_t* d_temp_new,
    const int32_t* d_heat, const int32_t* d_heat_inv_shift,
    const int32_t* d_face_shift,
    const bool* d_solid, const bool* d_ts, const bool* d_is_vacuum,
    const bool* d_is_ambient,
    const int32_t* d_n_src, const int64_t* d_rad_net,
    const int64_t* d_rad_fluence, const int64_t* d_e_table,
    int64_t* d_gas_energy, int64_t* d_de_gas, const int32_t* d_t_amb_q,
    int64_t* d_cap_used, int64_t* d_cap_real,
    const TemperatureScalars& sc,
    unsigned long long* d_cnt, bool solid_books);

// ---- fire (cuda_fire.cu) — #29 P4a -----------------------------------------
// The LAUNCH CORE of FireSimulation::step's GPU twin (fire_simulation.cpp —
// THE oracle): P2 the signed logistic, P5 the wall burn-through with the
// destroyed collection, P6 the final clamp. Born (N, h, w) with N = 1 today.
// The CPU's P1 early exit (max(fire) < quantize(0.001) -> nothing happens)
// is, with `d_gate`, a per-env device gate: the core reduces max(fire) from 0
// into d_gate[env] and every later kernel skips a gated-off env, so its
// fields stay exactly as the CPU's early return leaves them (RL-batch habits:
// a per-env mask, never host gating). The per-call fire_step keeps its host
// exit and passes d_gate = nullptr.
//
// The destroyed walls come out per env as a counter d_counter[env] and the
// packed in-env linear indices d_destroyed_idx[env * h*w + k] in atomic
// ARRIVAL order; fire_destroyed_row_major sorts them into the CPU's row-major
// list (order is load-bearing: GameMap.destroy_wall is order-dependent).
struct FireScalars {
    int32_t max_fire_thresh_q = 0;
    int32_t k_grow_q = 0, k_die_q = 0, k_wind_fan_q = 0, k_wind_strip_q = 0;
    int32_t fire_T_ext_q = 0, I_min_q = 0, wall_damage_q = 0, INV_C = 0;
    int32_t temp_is_identity = 0;      // bool
    int64_t recip_temp_scale = 0, recip_fuel_ref = 0, recip_T_span = 0;
    int32_t x_ext_q = 0;
    int64_t recip_x_span = 0;
    int32_t x_degenerate = 0;          // bool
    int32_t o2f_cap_q = 0, hotf_cap_q = 0, X_N_FLOOR = 0;
    int32_t p_ext_q = 0, p_full_q = 0, p_inv_span_q = 0;   // o2_pressure::Factor
};
FireScalars fire_bake_scalars(
    float k_grow, float k_die, float fire_T_ext, float fire_T_span,
    float fuel_ref, float o2_frac_ext, float I_min,
    float k_wind_fan, float k_wind_strip, float wall_damage,
    float temp_scale, float I_cap_per_avail,
    float o2_frac_amb, float o2f_cap, float hotf_cap,
    int32_t p_ext_q, int32_t p_full_q);
// The per-env step constant (one quantize of the env's tick length).
int32_t fire_dt_q(float dt);
//   d_fire, d_wall_hp : (N, h, w) int32, in/out
//   d_atmosphere    : (N, h, w) int32, NULLABLE (the #7 factor's input)
//   d_gas, n_gases, o2_idx : the O2 plane is plane o2_idx of the env's
//                     (n_gases, h, w) block
//   d_n_total       : (N, h, w) int32 (the bulk N, the mole fraction's denominator)
//   d_temperature, d_wind_x, d_wind_y : (N, h, w) int32
//   d_is_wall, d_is_vacuum, d_flammable : (N, h, w) bool
//   d_fuel_recip (int64), d_fire_T_ext_plane (int32) : (N, h, w), NULLABLE
//   d_dt_q          : (N,) int32 per-env scalar
//   d_gate          : (N,) int32, NULLABLE (no device early exit)
//   d_counter       : (N,) int;  d_destroyed_idx : (N, h, w) int
// Returns the launch count.
int fire_launch_resident(
    int n_env, int h, int w,
    int32_t* d_fire, const int32_t* d_atmosphere,
    const int32_t* d_gas, int n_gases, int o2_idx,
    const int32_t* d_n_total, int32_t* d_wall_hp, const int32_t* d_temperature,
    const int32_t* d_wind_x, const int32_t* d_wind_y,
    const bool* d_is_wall, const bool* d_is_vacuum, const bool* d_flammable,
    const int64_t* d_fuel_recip, const int32_t* d_fire_T_ext_plane,
    const int32_t* d_dt_q, const FireScalars& sc,
    int32_t* d_gate, int* d_counter, int* d_destroyed_idx);
// Packed in-env linear indices (arrival order) -> the row-major (y, x) list.
std::vector<std::pair<int, int>> fire_destroyed_row_major(std::vector<int> idx, int w);

// ---- the water tail (cuda_water_tail.cu) — #29 P4a --------------------------
// The device twin of PhysicsEngine::step_water_tail (physics_engine.cpp — THE
// oracle): W5 the flash-boil vacuum sink (a pressure-keyed water -> steam
// puff, per cell), W3 the occupancy-transition EVACUATION (a flooding cell
// sheds (1 - 1/ratio) of every gas plane into its open neighbours, its bulk
// share carrying the donor's T_abs through the energy seam) with the flooded
// dyn_permeability seal, and the final before := water_depth copy. Born
// (N, h, w) with N = 1 today; launch only.
//
// W3 is SERIAL on the CPU (row-major, each cell reading neighbours an earlier
// cell already sealed or filled), and the device keeps that order: a per-cell
// kernel computes each cell's ratio, seal flag and evacuation fraction (pure
// functions of before / water_depth / solid); ONE BLOCK PER ENV compacts the
// evacuating candidates in row-major order and its thread 0 walks them
// serially -- the CPU loop restricted to the cells that can move gas, the
// seals of earlier cells read from the flags (a cell is sealed in its own
// iteration before it evacuates); a last per-cell kernel applies the seals
// and the copy. Float bridges (the dequantized depths, the f32 ratio, the
// double fractions) are the CPU's, operation for operation, under the TU's
// --fmad=false and the IEEE division and square-root flags.
//
// Books, per env, WT_CNT_SLOTS unsigned long long, OVERWRITTEN (zeroed by the
// core): the evacuation's export (e_water_evac_export_sum, the CPU's per-call
// value), the steam puff's trace mint, the trace clip and negative-hit books
// per gas plane (W5's steam slot and W3's trace receivers), and two activity
// counts (cells that boiled, cells that evacuated) for the gates.
constexpr int WATER_TAIL_GAS_MAX = 16;
constexpr int WT_EXPORT  = 0;
constexpr int WT_MINT    = 1;
constexpr int WT_N_BOIL  = 2;
constexpr int WT_N_EVAC  = 3;
constexpr int WT_CLIP    = 4;                               // + gas index
constexpr int WT_NEG     = WT_CLIP + WATER_TAIL_GAS_MAX;    // + gas index
constexpr int WT_CNT_SLOTS = WT_NEG + WATER_TAIL_GAS_MAX;

// The tail's scalars, folded on the host with the CPU tail's own casts.
struct WaterTailScalars {
    int32_t boil_p_thresh_q = 0;
    int32_t boil_amount_q = 0;
    float   steam_yield_f = 0.0f;
    float   ceiling_h_f = 0.0f, flood_eps_f = 0.0f, clip_lo_f = 0.0f, clip_hi_f = 0.0f;
};
WaterTailScalars water_tail_bake_scalars(
    float sim_time, double ceiling_h, double flood_eps, double ratio_cap,
    double boil_rate, double boil_p_thresh, double steam_yield);

//   d_water_depth, d_before : (N, h, w) int32, in/out
//   d_atmosphere    : (N, h, w) int32 (the boil gate; read only)
//   d_solid         : (N, h, w) bool
//   d_gas           : (N, n_gases, h, w) int32, in/out
//   d_perm          : (N, h, w) float dyn_permeability, in/out (the seal)
//   d_gas_energy    : (N, h, w) int64, NULLABLE (the pre-#54 path)
//   d_cons          : (n_gases,) bool gas_conservative, NULLABLE (with energy:
//                     required; without: every plane takes the plain add)
//   d_thermal_solid, d_is_vacuum, d_is_ambient : (N, h, w) bool, NULLABLE
//   d_t_amb_raw     : (N,) int32 per-env scalar
//   d_flag (uint8), d_frac, d_list (int32) : (N, h, w) scratch
//   d_cnt           : (N, WT_CNT_SLOTS) unsigned long long
// Throws std::invalid_argument on n_gases outside [1, WATER_TAIL_GAS_MAX] or
// a steam_idx outside the block. Returns the launch count.
int water_tail_launch_resident(
    int n_env, int h, int w, int n_gases, int steam_idx,
    int32_t* d_water_depth, const int32_t* d_atmosphere, const bool* d_solid,
    int32_t* d_gas, int32_t* d_before, float* d_perm,
    int64_t* d_gas_energy, const bool* d_cons,
    const bool* d_thermal_solid, const bool* d_is_vacuum, const bool* d_is_ambient,
    const int32_t* d_t_amb_raw, const WaterTailScalars& sc,
    uint8_t* d_flag, int32_t* d_frac, int32_t* d_list,
    unsigned long long* d_cnt,
    // #29 P5: the per-env active mask (N,) uint8 — nullptr: every env runs;
    // 0: that env's fields and books are untouched (its book block is zeroed).
    const uint8_t* d_active = nullptr);

// ---- the stamp kernel (cuda_stamp.cu) — #29 P4b ----------------------------
// Slot 6's FIELD REBUILD on the device: the five stamp outputs from the living
// units' footprint rows. The device twin of PhysicsEngine::stamp_units
// (physics_engine.cpp — THE oracle). Born (N, h, w), N = 1 today; launch
// only — no malloc, no transfer, no sync.
//
//   d_permeability, d_wave_absorb : (N, h, w) float   static baselines (read)
//   d_heat_atten_q                : (N, h, w) int32   static Q16 (read)
//   d_light_atten_q               : (N, h, w, 3) int32 static Q16 (read)
//   d_obstacles                   : (N, h, w) bool    (written)
//   d_dyn_perm, d_dyn_wabs        : (N, h, w) float   (written)
//   d_dyn_heat                    : (N, h, w) int32   (written)
//   d_dyn_light                   : (N, h, w, 3) int32 (written)
//   d_rows   : (n_rows, STAMP_ROW_COLS) int32, in the host's stamp order:
//              [env, y, x, perm bits, wabsorb bits, heat_q, l0, l1, l2]
//   d_winner : (N, h, w) int32 scratch (the last row to touch each cell)
// Returns the launch count.
constexpr int STAMP_ROW_COLS = 9;
int stamp_launch_resident(
    int n_env, int h, int w, int n_rows,
    const float* d_permeability, const float* d_wave_absorb,
    const int32_t* d_heat_atten_q, const int32_t* d_light_atten_q,
    bool* d_obstacles, float* d_dyn_perm, float* d_dyn_wabs,
    int32_t* d_dyn_heat, int32_t* d_dyn_light,
    const int32_t* d_rows, int32_t* d_winner);

// ---- the device edit queue (cuda_edit_queue.cu) — #29 P4c ------------------
// Slot 6b's FieldEdit FLUSH on the device (residency design §4.8): the host
// enqueues, sorts and packs exactly as the CPU flush orders its work
// (simulation/field_edit.py pack_for_device: one EDIT row per edit at the
// door, one TILE row per tile of its region, in the CPU's applied order); the
// device applies every env's tiles in that order, serially per env, with the
// per-field skip masks, the integer contribution (amount_q x weight x mult x
// scale over den, rounded once half away), the keyed noise (Philox-4x32-10 on
// (tick, edit key, FIELD_NOISE, tile) under the env's seed: draws.
// noise_mult_q16's law), the per-field combines and the gas-energy seam
// (deposit + mirror refresh), every book into the env's arena block. The
// CPU flush (field_edit.apply_field_edit) is THE oracle.
//
// The field ORDINAL is the rank of the field's name in the lexicographic
// order of field_edit.FIELD_POLICY (the pack refuses a binding whose table
// differs). water_depth is never packed (the host flushes such a tick).
enum : int {
    EDQ_F_ATMOSPHERE = 0, EDQ_F_FIRE = 1, EDQ_F_GAS = 2, EDQ_F_GAS_ENERGY = 3,
    EDQ_F_HEAT = 4, EDQ_F_SMOKE = 5, EDQ_F_WATER_DEPTH = 6, EDQ_F_WAVE_SOURCE = 7,
    EDQ_N_FIELDS = 8,
};
// An EDIT row (int64): the columns of field_edit.E_*.
constexpr int EDQ_EDIT_COLS = 16;
enum : int {
    EDQ_E_ENV = 0, EDQ_E_FIELD = 1, EDQ_E_SLICE = 2, EDQ_E_MODE = 3,
    EDQ_E_AMOUNT = 4, EDQ_E_DEN = 5, EDQ_E_NOISY = 6, EDQ_E_LOW = 7,
    EDQ_E_KEY = 8, EDQ_E_CLAMP_FLAGS = 9, EDQ_E_LO = 10, EDQ_E_HI = 11,
    EDQ_E_O2_LO = 12, EDQ_E_O2_HI = 13, EDQ_E_N2_LO = 14, EDQ_E_N2_HI = 15,
};
// A TILE row (int64): [edit row (batch-global), tile r*w+c, weight_q, scale].
constexpr int EDQ_TILE_COLS = 4;
// The per-env scalars (int64, (N, EDQ_ENV_COLS)): the draws' tick word, the
// Philox key (seed lo, seed hi), the seam's ambient (gas_t_amb_raw).
constexpr int EDQ_ENV_COLS = 4;
// The books, per env (unsigned long long slots of the arena): the trace
// deposit / clip books per gas slice (GameMap.trace_deposit_books /
// trace_clip_books), the gas-energy seam channels field_edit / _wave / _atm
// (each its net and its count of non-zero deposits: the host books a channel
// key on its first non-zero delta), the noise draws (MatchDraws.drawn), a
// failure count (a result the host flush would refuse: an int32 store out of
// range, an int64 energy overflow, a negative trace result) and the tiles
// applied (not skipped).
constexpr int EDQ_GAS_MAX = 16;
enum : int {
    EDQ_B_DEP = 0, EDQ_B_CLIP = EDQ_GAS_MAX,
    EDQ_B_GE_FE = 2 * EDQ_GAS_MAX, EDQ_B_GE_FE_N, EDQ_B_GE_WAVE, EDQ_B_GE_WAVE_N,
    EDQ_B_GE_ATM, EDQ_B_GE_ATM_N, EDQ_B_NOISE, EDQ_B_FAIL, EDQ_B_TILES,
    EDQ_SLOTS,
};
// The launch core: (N, h, w) planes (gas (N, n_gases, h, w)); d_env_off
// (N + 1) tile offsets per env; d_env (N, EDQ_ENV_COLS); d_contrib (n_tiles)
// int64 scratch; d_books (N, EDQ_SLOTS), zeroed by the caller. is_ambient may
// be nullptr (a space map). Launch only. Returns the launch count.
int edit_queue_launch_resident(
    int n_env, int h, int w, int n_gases, int o2_idx, int n2_idx,
    int n_tiles, uint32_t salt,
    const int64_t* d_edits, const int64_t* d_tiles,
    const int64_t* d_env_off, const int64_t* d_env,
    int32_t* d_gas, int32_t* d_temperature, int64_t* d_gas_energy,
    int32_t* d_fire, int32_t* d_heat,
    const bool* d_solid, const bool* d_is_vacuum, const bool* d_is_ambient,
    const bool* d_thermal_solid, const bool* d_flammable,
    int64_t* d_contrib, unsigned long long* d_books);
// TEST-ONLY (the gate's noise / contribution check): K1 on host arrays -- the
// contribution of every tile row (rows as above, env scalars (n_env,
// EDQ_ENV_COLS)), so the device's noise multiplier and rounding are held to
// draws.noise_mult_q16 and unit_fixed.div_round record by record. Allocates
// and copies: never on the hot path.
void edit_queue_contrib_probe(const int64_t* edits, int n_edits, const int64_t* tiles,
                              int n_tiles, const int64_t* env, int n_env, uint32_t salt,
                              int64_t* out);

}  // namespace breach_cuda
