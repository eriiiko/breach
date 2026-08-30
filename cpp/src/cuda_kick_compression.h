#pragma once
// ============================================================================
// EOS P6.4 — momentum kick on the GPU (bit-identical); P-G2 — the face-flux
// energy step's GPU twin (gas-energy conservation arc #54, design §2.4/§2.5).
// ============================================================================
//
// gas-energy conservation arc #54 (design §5): K2, the old step-4c
// compression-work kernel (T -= (γ−1)·T·div(u_new)·dt, ±T_WORK_CLAMP-railed)
// is DELETED. Its job is replaced by two device passes:
//   * K1 (kick, this file): u -= dt·K·grad(P_new)/N̂, absorption damping,
//     the B3c sponge band, the |u| <= per-cell cap2_plane[i] counted clamp,
//     staged drag L/Q — now carrying the per-stage KINETIC-ENERGY BRACKETS
//     (design §2.3) straight into `gas_energy` (debit on the ∇p kick and the
//     drag heat credit; absorb/sponge/clamp EXPORT or DESTROY, counted,
//     never heat — D6).
//   * K3 (the face-flux energy step, this file): Kwatra's conservative
//     flux-form energy update on the solved ABSOLUTE pressure (post step-5
//     un-shift) and the corrected velocity, sub-cycled n_sub times with the
//     increment-form pressure refresh (design §2.4) and a donor-only
//     positivity rail (design §2.4, two gather passes — no atomics, no face
//     buffer, design §2.5), followed by the once-per-tick recovery (§2.6).
//
// Every arithmetic step is a VERBATIM device transcription of the CPU loop
// (eos_solver.cpp EOSSolver::step, steps 4/6/7): the (int64_t)(Pn[ir]-Pn[il])
// int32-subtract-then-widen, the staged mul128_shr chains, the magnitude-
// first absorption/sponge shrink, the ±2^27 KE_SAFE guard (load-side +
// unconditional post-∇p), the counted per-cell cap2_plane[i] scale-to-cap,
// and K3's canonical-orientation face pricing + donor-only rail. Rail
// counters are device atomicAdds — pure order-free int64 sums, so the totals
// equal the CPU's sequential sums exactly.
//
// Plain C++ declaration header (no CUDA types) so the .cpp TUs can include
// it; cuda_kick_compression.cu provides the definitions. BREACH_CUDA only.
#include <cstdint>

namespace breach_cuda {

// KICK_CNT_SLOTS-slot layout (cuda_resident.h), THIS call's counters (not
// accumulated — the caller adds/assigns into the solver's own members):
//   0 u_clamp_hits            1 u_max_hits
//   2 retired (work_clamp_hits, always 0)
//   3 retired (energy_floor_hits — moved to §2.6 recovery, always 0)
//   4 retired (t_max_phys_hits — moved to §2.6 recovery, always 0)
//   5 ke_drag_removed         6 e_drag_heat_sum
//   7 retired (e_drag_drop_sum, always 0)
//   8 retired (e_drag_rail_clipped, always 0)
//   9 e_kick_ke_sum          10 e_absorb_export_sum
//  11 e_sponge_export_sum    12 e_clamp_destroyed_sum
//  13 rad_clip_hits          14 e_ts_ke_sum
//
// The step-4 kick for one tick on the GPU — IN PLACE on wind_x/wind_y and
// (design §2.3) `gas_energy`. p_new is the solved pressure plane (pre step-5
// un-shift — the SAME operand the CPU kick reads); the gas planes rebuild
// step 2's Dalton N_total on the host verbatim; cap2_plane is the per-cell
// (h,w) Q32.32 velocity-cap² plane (VELOCITY-CLAMP, P-V1, D2v2). Outputs:
// digest_velocity_out (byte-for-byte step()'s digest_velocity expression,
// computed host-side after the D2H) and counters_out[KICK_CNT_SLOTS] (the
// slot map above) for THIS call.
void eos_kick_compression(
    int32_t* wind_x,               // Q16.16 (h,w) — in/out (solver u.x)
    int32_t* wind_y,               // Q16.16 (h,w) — in/out (solver u.y)
    int64_t* gas_energy,           // arc #54 §2.2/§2.3 (h,w) — in/out
    const int32_t* p_new,          // Q16.16 (h,w) — solved P (read-only)
    const int32_t* gas, const bool* gas_conservative, int n_gases,
    const bool* solid, const bool* is_vacuum,
    const float* dyn_wave_absorb,  // FLOAT (h,w) — host-hoisted to q16 (§2.5)
    int h, int w, float dt, const int64_t* cap2_plane,   // D2v2 (h,w), >= 0
    float c_max, float dx, float adiabatic_index, float absorb_strength,
    float n_floor_solver, float u_max,
    // P-E3 (energy-books arc, design §2.8): interior drag + heat
    // counterparty. k_drag default 0.0 -> dormant. k_drag2 (drag-law v2):
    // the quadratic term, same dormancy idiom (kd2_q).
    float k_drag, float k_drag2,
    // arc #54 §2.1: T_AMB_K — folds the derived k_ke constant (with
    // adiabatic_index/c_max above).
    float t_amb_k,
    uint64_t* digest_velocity_out,
    int64_t* counters_out /* [KICK_CNT_SLOTS] */,
    // BC (spec §1/§3): the ambient ring (nullptr = space) drives the velocity
    // zero; the u-damping band grid sponge_udamp (nullptr = off) is the
    // rung-2 absorber applied after the absorb chain, magnitude-first.
    const bool* is_ambient = nullptr, const int32_t* sponge_udamp = nullptr,
    // THERMAL-MASS AXIS, P-EOS: the medium mask the KE brackets export to
    // (design §2.3 F5 — ts cells carry no gas_energy). nullptr -> `solid` on
    // the device (the P2 fallback idiom).
    const bool* thermal_solid = nullptr);

// FLUX_CNT_SLOTS-slot layout (cuda_resident.h), THIS call's counters:
//   0 p_face_floor_hits    1 p_face_ceil_hits    2 flux_sat_hits
//   3 e_energy_floor_sum   4 e_work_export_sum
//   5 e_ts_work_sum        6 e_wall_work_probe_sum
//   7 energy_floor_hits (recovery T_MIN)   8 t_max_phys_hits (recovery)
//   9 e_rail_sum          10 e_wipe_sum
//
// The face-flux energy step (design §2.4/§2.5) + the once-per-tick recovery
// (design §2.6), for one tick on the GPU — IN PLACE on `gas_energy` and
// `temperature` (the mirror). `atmosphere` is the ABSOLUTE solved pressure
// (post step-5 un-shift — pinned: this call must run AFTER that add-back);
// wind_x/wind_y are the corrected (post-kick/absorb/sponge/cap/drag)
// velocity; n_total the post-substep Dalton N. n_sub is the SAME substep
// count the kick/SL advection share (design §2.4 D8).
void eos_energy_flux(
    int64_t* gas_energy,           // arc #54 §2.2 (h,w) — in/out
    int32_t* temperature,          // (h,w) — in/out (the mirror, §2.6)
    const int32_t* atmosphere,     // Q16.16 (h,w) — ABSOLUTE p^{n+1}
    const int32_t* wind_x, const int32_t* wind_y,
    const int32_t* n_total,
    const bool* solid, const bool* is_vacuum,
    int h, int w, float dt, int n_sub,
    float dx, float adiabatic_index, float t_amb_k, float c_value,
    float t_min, float t_max_phys,
    int64_t* counters_out /* [FLUX_CNT_SLOTS] */,
    const bool* is_ambient = nullptr,
    const bool* thermal_solid = nullptr);

// Backend selection (the P6.1/P6.2 surviving-backend idiom).
bool kick_compression_backend_is_cuda();
void set_kick_compression_backend_cuda(bool on);

}  // namespace breach_cuda
