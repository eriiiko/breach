#pragma once
// Donor-cell conservative flux transport for the BULK gas species
// (EOS refactor P1 — docs/eos_refactor_design.md §2.2, decisions log #11).
//
// The two CONSERVATIVE bulk species (O2 / inert_N2, simulation/gases.py) move
// by first-order upwind donor-cell flux on the solver's wind field, using the
// SAME pattern (gather-once wide flux + per-cell outflow limiter) as
// WaterSolver::step's donor-cell block (water_solver.cpp) — every subtraction
// has a matching addition, so mass is conserved to the integer LSB and a
// sealed room's O2+N2 total never drifts (the P1 gate: exact conservation
// over 1000 ticks). This is P1: purely ADDITIVE, riding TODAY's wind
// (gmap.wind_x/wind_y, already computed by AtmosphereSolver::diffuse_solve
// earlier this tick) — no solver change, nothing yet consumes N_O2/N_N2.
//
// Non-conservative TRACE gas planes (gas_conservative[gi] == false) are left
// completely untouched here — they stay on the existing per-gas
// semi-Lagrangian loop (PhysicsEngine::run_substeps), which in turn SKIPS the
// conservative planes so the two transport schemes never both touch the same
// plane (see the run_substeps body).
#include <cstdint>

// gas               : (n_gases, h, w) contiguous Q16.16 density planes, mutated in place
// gas_conservative  : (n_gases,) — true for the bulk pair (O2 / inert_N2); every
//                     other (trace) plane is skipped entirely
// wind_x, wind_y    : (h, w) Q16.16 — the solver's cell-centred velocity-like
//                     field (AtmosphereSolver's wind, this tick's fresh value)
// solid             : (h, w) — the physics solid mask (permeability <= 0); a
//                     solid cell always holds N == 0 (enforced by the final clamp)
// is_vacuum         : (h, w) — true vacuum; N is zeroed there every tick (mass
//                     legitimately leaves the system at a breach — a deliberate
//                     sink, not a conservation bug)
// dyn_permeability  : (h, w) — the live per-tick face-permeability field; gates
//                     flux exactly like the smoke/atmosphere stencils
//                     (face = min(perm_self, perm_neighbor))
// h, w              : grid dimensions
// dt                : the FULL tick length (seconds, sim_time); dx == 1 tile
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

// EOS P3 micro-opt (BIT-IDENTITY-PRESERVING): the per-face coefficient
// coeff_q = mul_q16(quantize(min(perm_i, perm_j)), quantize(dt)) is constant
// across a tick's substeps (perm and dt_s do not change within a tick), yet
// the legacy entry recomputed it per face per plane per substep (float min +
// quantize x2 planes x n_sub). This entry takes the two PRECOMPUTED per-face
// coefficient arrays (east face of i -> coeffE[i]; south face -> coeffS[i];
// 0 == sealed face — quantize-to-0 and the face_f<=0 gate collapse to the
// same zero flux the legacy path produced) and reuses internal scratch
// across calls. Arithmetic per face is IDENTICAL to the legacy entry — the
// caller hoists the loop-invariant computation, nothing more. The legacy
// entry remains for the pybind/P1-test path and now forwards here.
// BC (boundary_conditions_spec_2026-07-19 §1 "N — sink becomes clamp, per
// substep"): the three trailing params add the planetside AMBIENT ring reset.
// ALL default nullptr -> the legacy/space path is byte-identical (dormancy BY
// BRANCH, spec §5): every ambient branch is gated on is_ambient != nullptr.
//   is_ambient    : (h, w) — the ambient reservoir mask (nullptr = space map)
//   n_amb         : (n_gases,) — the per-plane ambient N value the ring clamps
//                   to (conservative planes only; 0 elsewhere). nullptr = space.
//   boundary_flux : (n_gases,) int64 rail — accumulates Σ(N_pre_reset − N_amb)
//                   per conservative plane, per substep (§5). nullptr = no rail.
// P-E1 (energy-books arc, design §2.1.3): the two trailing int64 face planes,
// when non-null, are ZEROED here and then accumulate the APPLIED (post-limiter,
// post-scale_mag) per-face dq summed over the conservative planes — exactly the
// flux the mass books move, which is what the thermal energy rides. dqsum_e[i]
// is the east face of i (positive = i -> i+1); dqsum_s[i] the south face
// (positive = i -> i+w). Sign(dq) == sign(v_face) per face across all planes
// (donor N >= 0, coeff > 0, scale_mag preserves sign), so the donor identity
// the energy pass reads off the sign is the SAME donor the mass flux chose.
// Default nullptr -> byte-identical legacy path.
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
    int h, int w,
    const bool* is_ambient = nullptr,
    const int32_t* n_amb = nullptr,
    int64_t* boundary_flux = nullptr,
    int64_t* dqsum_e = nullptr,
    int64_t* dqsum_s = nullptr);

// ===========================================================================
// P-E1 — energy-conservative thermal transport (energy-books arc, design
// energy_transport_design_2026-08-16.md v2.1 §2.1). THE core law change:
// thermal energy rides the conservative donor-cell mass fluxes; temperature
// is recovered as e / n_bulk at each endpoint. Replaces the retired
// semi-Lagrangian T copy (T-WRITE SITE 1/2 — the measured mint).
//
// One substep = four pinned stages (the CUDA twins in cuda_bulk_transport.cu
// transcribe these loops kernel-for-loop; per-cell expression order is the
// contract — L2-11):
//   1. e build      e[i] = n_bulk_pre[i] * T[i], int64 exact, PARTICIPATING
//                   cells only (!solid && !ts && !vacuum && !ring — design
//                   §2.1.2: a ts tile's temperature[] is the OBJECT's T, so
//                   e there would be bogus; excluded cells carry e = 0).
//   2. mass flux    bulk_flux_transport_cached above, accumulating the
//                   applied per-face dq sums (dqsum_e/dqsum_s).
//   3. e apply      gather-form (each cell edits only its own e — the CUDA
//                   no-atomics shape): for each face with dq != 0, the
//                   PARTICIPATING donor is debited dq * T_raw[donor] (its
//                   recovered T stays exactly invariant as mass leaves); a
//                   PARTICIPATING receiver is credited the same amount when
//                   the donor participates. ts-face rule (d), per-face
//                   (design §2.1.4): air->ts debits the donor at its OWN T
//                   into the SIGNED counter e_ts_residual (counted
//                   destruction); ts->air (and vacuum/ring->air) arrives
//                   carrying ZERO relative energy; ts->ts moves nothing.
//                   Mass into vacuum/the ambient ring takes its energy with
//                   it uncounted (the §4/§5 boundary channels, unchanged).
//   4. recovery     participating cells: n_new = post-flux bulk sum (>= 0,
//                   post-clamp). n_new < N_EPS (1 raw): T := 0, residual ->
//                   SIGNED e_wipe_sum. Else T = floordiv(e, n_new) toward
//                   -inf by the pinned idiom `q = e/n; if (e%n && e<0) --q`
//                   (n >= 1; -n < e < 0 recovers T = -1 — stable); T_MIN
//                   clamp counted in ENERGY units (e_floor_sum, a CREATOR —
//                   R3). Quiescent cells (no face traffic) rebuild EXACTLY.
//                   Non-participating: solid and ts cells are never written
//                   (T-WRITE guard upheld); vacuum/ring cells keep their
//                   per-substep T := 0 wipe (moved here verbatim from the
//                   retired SL write; semantics unchanged).
//   Counters n_active_flux / n_bulk_active_sum (design §2.5): per substep, a
//   participating cell with ANY nonzero touching face dq is ACTIVE; the §7
//   transport bound is scaled by the measured active fraction (L2-10).
//
// Range (design §2.1.2/§3): per-cell n_bulk < 2^30 raw (debug-asserted at
// the e build), |T| <= T_MAX_PHYS raw < 2^31 => |e| < 2^61; recovered T is a
// mass-weighted mix of donor T's (convexity) so the int32 narrow is safe.
// ===========================================================================
struct BulkEnergyCounters {
    int64_t e_ts_residual = 0;      // signed rule-(d) air->ts debits
    int64_t e_wipe_sum = 0;         // signed N_EPS wipe residuals
    int64_t e_floor_sum = 0;        // energy CREATED by the T_MIN recovery clamp
    int64_t n_active_flux = 0;      // active-cell count (substep-accumulated)
    int64_t n_bulk_active_sum = 0;  // sum n_bulk_new (raw) over active cells
};

// thermal_solid_ts: the resolved medium mask (caller passes its `ts` fold —
// thermal_solid ? thermal_solid : solid; never nullptr). e_scratch / dqsum_e /
// dqsum_s: caller-owned (h*w) int64 scratch (overwritten here). t_min_q: the
// caller's quantize(T_MIN) fold. Counters ACCUMULATE (+=) — caller resets per
// tick. The trailing ambient args mirror bulk_flux_transport_cached.
void bulk_flux_energy_transport_cached(
    int32_t* gas,
    const bool* gas_conservative,
    int n_gases,
    int32_t* temperature,
    const int32_t* wind_x,
    const int32_t* wind_y,
    const bool* solid,
    const bool* is_vacuum,
    const bool* thermal_solid_ts,
    const int32_t* coeffE,
    const int32_t* coeffS,
    int32_t t_min_q,
    int h, int w,
    int64_t* e_scratch,
    int64_t* dqsum_e,
    int64_t* dqsum_s,
    BulkEnergyCounters& cnt,
    const bool* is_ambient = nullptr,
    const int32_t* n_amb = nullptr,
    int64_t* boundary_flux = nullptr);
