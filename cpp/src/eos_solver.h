#pragma once
// EOSSolver — the compressible Kwatra pressure-evolution solver.
//
// N. Kwatra, J. Su, J.T. Grétarsson, R. Fedkiw, "A Method for Avoiding the
// Acoustic Time Step Restriction in Compressible Flow", J. Comput. Phys. 228
// (2009) 4146-4161.
//
// NAMED DEVIATION FROM THE PAPER (docs/eos_refactor_design.md §3.1): Kwatra
// advects pressure itself (p_a); we derive p* = C*N_adv*T_adv from the
// ADVECTED (N,T) state via the ideal-gas EOS instead of carrying a separate
// pressure state — a consistent O(dt) choice that guarantees P can never
// drift from (N,T) (the paper itself notes the method is EOS-agnostic).
// Energy is carried as T with an explicit compression-work term rather than
// a conservative E (a named §3.3-class simplification).
//
// EOS refactor P3, design v2.2 (docs/eos_refactor_design.md §3). Replaces
// AtmosphereSolver::wave_substep + ::diffuse_solve. `atmosphere` is the
// stored P (materialized ONCE per tick, step 5); `wind_x`/`wind_y` are the
// solver's OWN velocity state u.
//
// v2.2 D-A (unit consistency — adopted 2026-07-10 after P3's gate measured
// the N·c² transplant at ~64,000× the pressure scale): the operator/RHS
// coefficient is the EXACT ideal-gas identity rho*c^2 = gamma*P, evaluated
// per cell as (γ·p*)_cell — in P's own units by construction. ONE unit-
// bridge constant K = c_amb²/γ (a WIDE int64 — it does not fit q16) lives in
// the momentum kick u -= dt·K·grad(P)/N̂ and in the CFL estimate's ∇P term.
// c is state-derived (c ∝ √T); the velocity cap is PER-CELL
// (VELOCITY-CLAMP, P-V1, design v3): cap²_cell = c_amb²·t_abs_cell/T_AMB,
// folded from tick-entry T alongside c_LOCAL in the same scan. c_LOCAL
// itself survives solely as the n_sub/CFL estimate's ceiling (never a stale
// ambient constant there either).
// The per-tick system is LINEAR ((γ·p*) frozen at the advected value);
// near-vacuum rows degenerate to identity (correct Dirichlet physics);
// N_FLOOR_SOLVER applies ONLY to the face 1/N̂ divide.
//
// v2.2 D-B: the pressure solve is a FIXED-SCHEDULE MULTIGRID V-cycle
// (P3's gate measured point-RB-GS needing S≈128 at ambient coupling —
// unaffordable; MG carries room-scale influence in one cycle). RB-GS is the
// smoother at every level; coarse operators are RE-DISCRETIZED from
// fine-informed face conductances (restriction of the fine faces' perm/N̂;
// harmonic-class averaging — the arithmetic-N̂ face IS the harmonic mean of
// the two cells' 1/N conductivities); coarse Dirichlet rule: a coarse cell
// is vacuum iff ALL children are vacuum (straddlers stay regular); transfer
// operators are fixed integer stencils (full-weighting restriction = 4-child
// average via >>2; bilinear prolongation (9,3,3,1)/16 via >>4, TRUNCATING
// toward -inf — the named rounding rule). Truth lives on the fine grid.
// Schedule (nu1, nu2, cycles, coarsest sweeps) pinned at the MG measurement
// gate, FROZEN thereafter — never adaptive.
//
// v2.2 D-C: the smoother's flux accumulator stays WIDE (int64) until the
// final per-cell store; the per-cell diagonal reciprocal is a wide Q.32
// integer divide precomputed once per cell per tick per level (NOT the
// q16 reciprocal_q16, whose range the diagonal exceeds — §3.4 rule 1).
//
// TICK ORDER (design §3.2):
//   0. P_prev := P
//   1. advection substeps (SL u self-advect, SL T, donor-cell O2/N2),
//      N_SUB_MAX-capped; NO compression work here (moved to step 4c —
//      the pre-solve placement double-counted against the RHS div term)
//   2. p* := C·N_total·(T + T_AMB_K)
//   3. PRESSURE SOLVE: multigrid V-cycles (or flat RB-GS when
//      use_multigrid=false — the measurement-gate A/B reference)
//   4. u -= dt·K·grad(P)/N̂ (whole chain int64; |u| clamped to c_LOCAL;
//      narrowed once at store); absorption damping; zero outside open-air
//   5. P := P_new stored once (the `atmosphere` alias)
//   6. FACE-FLUX ENERGY STEP (gas-energy conservation arc #54, design §2.4 —
//      REPLACES step 4c): n_sub sub-cycles of a two-pass (rail, apply) gather
//      over the 4 faces, on the ABSOLUTE solved p^{n+1} and the stored u_new,
//      applying each face with opposite signs to its two cells so Σ_region ΔE
//      telescopes to 0 exactly. Kwatra eq. 3 with the eq. 15 face pressure.
//   7. RECOVERY (§2.6): T := floordiv(E, N) − T_AMB once per tick over the
//      whole accountable set; T_MIN / T_MAX_PHYS rails clamp the mirror AND
//      write gas_energy back ONLY when a rail binds (counted, e_rail_sum).
//
// Step 4c (T ← T(1±w), the per-cell temperature-form compression work) is
// DELETED at P-G1a: reversible per cell, but the books quantity Σ N_i T_i did
// not telescope, which is issue #54's +121/−20 sealed-box signature.

// ---------------------------------------------------------------------------
// THERMAL-MASS AXIS, P-EOS (docs/thermal_mass_eos_ruling_2026-07-30.md — the
// Fable ruling that answers docs/thermal_mass_eos_escalation_2026-07-30.md).
//
// THE GOVERNING RULE:
//   On `thermal_solid` tiles (`thermal_mass > 0`; GameMap.thermal_solid),
//   `temperature[]` is OWNED by the TemperatureSolver (deposit-convert /
//   conduct / COOL_SHIFT). Every other system is a READER. The EOS reads T
//   (for p* = C·N·T_abs) and NEVER writes it there.
//
// Consequences inside this solver (ruling §4 items 1-4):
//   * step-1b (semi-Lagrangian sample): the `temperature[i]` write is SKIPPED
//     on thermal_solid tiles (A1 — "the gas now at i came from upstream" is a
//     fluid-parcel claim; the OBJECT at i did not come from upstream), and a
//     thermal_solid tile is an OCCLUDER to the T backtrace (A2 — the eos-side
//     analog of the temperature solver's `gas_wall_at`; sampling a 1300 K
//     crate as a source would be a free-energy channel, since SL copies
//     without debiting).
//   * step-4c (−P∇·u compression work): the `temperature[i]` write is SKIPPED
//     on thermal_solid tiles (A1 — the object does not compress).
//   * `cmask` is UNTOUCHED: pressure, velocity and gas flow are unchanged, so
//     `permeability` / shield-but-not-seal survives verbatim (ruling §4 item 4
//     / escalation trigger 5). The T occlusion therefore rides a SECOND,
//     T-ONLY mask — occluding the shared fused march would have moved the
//     VELOCITY self-advection, which item 4 forbids. (P-E1 RETIRED that second
//     mask with the SL T sample itself; the bullet above it is likewise
//     historical — step 1b no longer writes temperature at all. Step 4c's
//     A1 skip and the p* reading below are unchanged and still live.)
//   * p* = C·N·T[i] keeps reading the OBJECT temperature on a crate tile
//     (A3, hot pore gas — the decision stands; gate (f) is its tripwire).
//
// `thermal_solid == nullptr` falls back to `solid` — today's behaviour
// byte-for-byte (the legacy / space-map path), and on any FURNITURE-FREE map
// `thermal_solid == solid` elementwise (build addendum D4), so the fallback is
// not a second code path in practice.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <vector>

// THERMAL-MASS AXIS: does the thermal medium actually DIVERGE from the gas
// medium anywhere on this map — i.e. is some tile thermally solid AND a LIVE
// gas cell to the EOS cmask? The predicate is exactly `thermal_solid[i] &&
// cmask[i] != 0`, re-expressed on the cmask's own definition
// (`sealed <=> solid || dyn_permeability <= 0`) so the CPU step and both CUDA
// hosts can share ONE transcription and can never disagree about whether the
// T-only occlusion mask is live. False (the furniture-free / nullptr case) =>
// every T sample takes the fused mask, i.e. the pre-patch code path, bit for
// bit. The float touch is a COMPARISON only — no float arithmetic.
inline bool eos_thermal_occludes(const bool* thermal_solid, const bool* solid,
                                 const float* dyn_permeability, int n) {
    if (thermal_solid == nullptr) return false;
    for (int i = 0; i < n; ++i) {
        if (thermal_solid[i] && !(solid[i] || dyn_permeability[i] <= 0.0f))
            return true;
    }
    return false;
}

class EOSSolver {
public:
    // ---- PINNED / config constants (design §8/§9, decisions log) ----------
    // c_max == c_amb: the AMBIENT sound speed dial (Erik: 300 m/s). v2.2:
    // K = c_amb²/γ is derived from this each tick; the true local sound
    // speed is state-derived (c ∝ √T) and exceeds this at a hot core.
    float c_max = 300.0f;
    // dx: the level's physical tile size (metres), lazy-bound by the caller
    // from gmap.tile_size_m (the WaterSolver.dx precedent).
    float dx = 0.333f;
    // S: flat RB-GS sweep count, used ONLY when use_multigrid == false (the
    // MG measurement gate's A/B reference path — not the shipped mechanism).
    int   S = 8;
    // ---- multigrid schedule (v2.2 D-B) — frozen at the MG gate ------------
    bool  use_multigrid = true;
    int   mg_nu1 = 2;              // pre-smooth sweeps per level
    int   mg_nu2 = 2;              // post-smooth sweeps per level
    int   mg_cycles = 2;           // V-cycles per tick — FROZEN at the MG gate
                                   // (2026-07-10). With the P_prev warm start
                                   // C=2 is durably stable over 300 ticks on
                                   // BOTH E2Es (water worst-dev 0.0066 atm,
                                   // vent overshoot 0.0005); C=1's vent dev
                                   // (0.042) was too marginal to freeze.
                                   // (Pre-warm-start history: cold-start-from-
                                   // p* needed C=4; C=3 was measurably
                                   // unstable — see the gate doc.)
    int   mg_coarsest_sweeps = 32; // RB-GS sweeps at the coarsest level
    int   mg_min_dim = 1;          // coarsen the pyramid all the way (the DC /
                                   // room-bulk mode is solved EXACTLY at 1×1)
    // -----------------------------------------------------------------------
    // N_SUB_MAX — RE-PINNED 16 -> 8 (Erik, 2026-07-10, decisions log #14):
    // measured exactly as stable as 16 on both E2Es (300 ticks, worst-dev
    // 0.010/0.0006), and the sustained-sonic-venting regime pins n_sub at
    // the cap for the whole post-breach phase, so the cap IS the perf dial.
    int   N_SUB_MAX = 8;
    float CFL_ADV = 0.5f;
    // N_FLOOR_SOLVER: applies ONLY to the face 1/N̂ divide (design §3.1
    // property 2 — never to the outer γ·p* coefficient, whose vanishing at
    // vacuum IS the desired Dirichlet degeneracy).
    float N_FLOOR_SOLVER = 1e-3f;
    // T_AMB_K / C mirror [physics.temperature_scale].eos_t_amb_k (ruling 6:
    // EOS pressure calibration is a deliberate 290 K exception to the unified
    // kelvin_ambient map — see docs/temperature_scale_unification_design_
    // 2026-08-13.md §2/§3c). Fold path for both is double -> float -> quantize
    // (struct default here, in double at config load, then requantized to q16
    // per-tick in eos_solver.cpp) -- do not "fix" one backend's rounding
    // without the other; c_q is still 226 after the float32 round-trip.
    float T_AMB_K = 290.0f;
    float C = 1.0f / 290.0f;
    // S_EOS mirrors [physics.temperature_scale]: phi_exp * k_temp_to_kelvin,
    // value-frozen to 1.0 exactly this arc (P-K3) — the slope MECHANISM exists
    // (t_abs = (S_EOS_q16 * T >> 16) + t_amb_q in eos_solver.cpp/cuda_eos_*)
    // but at the frozen identity the EOS is byte-identical to pre-P-K3. Same
    // double -> float -> quantize fold idiom as T_AMB_K/C above.
    float S_EOS = 1.0f;
    float adiabatic_index = 1.4f;   // γ (compile-time-class constant; config echo)
    float absorb_strength = 8.0f;
    float T_MIN = -289.0f;
    // T_WORK_CLAMP / n_work_ref RETIRED (gas-energy conservation arc #54,
    // design D11, P-G1a): step 4c — the per-cell temperature-form compression
    // work whose rate rail and trust gate these two dials were — is GONE,
    // replaced by the conservative face-flux energy step (§2.4). Their job is
    // done structurally: telescoping (no temperature unbacked by energy) plus
    // the flux step's own donor-only positivity rail. `renderer/cold_overlay.
    // py` keeps its own COLD_N_MIN_FRAC constant (HUMAN-TEST ruling
    // 2026-08-21 stands). Do not re-add them: a rate rail on a conservative
    // flux would break the exact ± cancellation the whole arc rests on.
    // k_drag / k_drag_heat_frac (energy-books arc, design §2.8, NEW patch
    // P-E3): interior momentum drag WITH a heat counterparty — the mechanism
    // that gives the Helmholtz storm an honest grave. Per-tick, in the step-4
    // kick loop, AFTER the |u| cap and BEFORE the store: component-wise
    // magnitude-first shrink u *= (1 - kd_q), kd_q = quantize(k_drag*dt)
    // folded once per tick (the absorb precedent); the removed kinetic energy
    // deposits into the SAME cell's T (a collocated-grid shear-heating
    // placement). k_drag default 0.0 -> the mechanism ships SILENT (dormancy
    // BY BRANCH on the QUANTIZED kd_q, not the float — see the .cpp).
    // k_drag_heat_frac default 1.0 (RULING R2, Erik 2026-08-17): full deposit
    // keeps the conservation oracle EXACT through every gate; Erik sweeps the
    // fraction at P-E5 (physical-air anchor ~=0.0014 — Q16 game units put
    // air's heat capacity ~700x below physical, c_v=1 by convention). Any
    // non-deposited remainder is the counted, named e_drag_drop_sum channel.
    float k_drag = 0.0f;
    // k_drag2 (drag-law v2, docs/drag_law_v2_design_2026-08-23.md, issue #4
    // P1): the QUADRATIC term of the two-term law F = -k1*u - k2*|u|*u
    // (k1 = k_drag above, independent dial). Implicit discretization
    // u <- u/(1+k2*dt*|u|), applied in the SAME kick-loop drag block
    // immediately after the linear shrink, before the energy booking (see
    // the .cpp). Default 0.0 -> dormant (branch on the quantized kd2_q, not
    // this float -- the k_drag idiom). Lands dormant this arc (P1/P2);
    // dial-turning is P3, HUMAN-TEST.
    float k_drag2 = 0.0f;
    // k_drag_heat_frac RETIRED (arc #54, design D5/§2.1, P-G1a): the drag
    // deposit is no longer a DIAL-scaled fraction of ΔKE divided by the
    // convention c_v — it is the derived unit-bridge constant
    // k_ke = γ(γ−1)·T_AMB_K/(2·c_max²) applied to the SAME Δ(|u|²) bracket
    // (eos_solver.cpp's k_ke_recip_q32 fold). 0.0014 was the hand-rolled
    // stand-in for 1/c_v_phys ≈ 0.0018; the P-E5 `k_drag_heat_frac = 1.0`
    // detonation cannot recur because the constant is DERIVED, not dialled.
    // c_v (energy-books arc, design §2.8): the SAME gas heat-capacity
    // constant TemperatureSolver::c_v prices its ΔT=ΔE/(N*c_v) deposits with
    // ([physics.thermal] c_v — physics_runner.py binds both from the ONE
    // config key). The drag deposit's ΔT = k_drag_heat_frac*ΔE_cell/c_v needs
    // NO per-cell N divisor (ΔE_cell is already specific — a velocity, not a
    // momentum), so this is EOSSolver's own copy of the same load-time
    // constant rather than a cross-solver reference. Default 1.0 mirrors
    // TemperatureSolver's own default.
    float c_v = 1.0f;
    // T_MAX_PHYS (v2.4 as-built amendment, PROVISIONAL pending Erik's P5
    // review — eos-p3fix-thermal-ceiling, decisions.md #16): a COUNTED
    // physical-maximum rail on the T FIELD itself, applied as a saturating
    // clamp at every T write path (here: step 4c; also TemperatureSolver
    // Pass 1 and CombustionSolver's deposit — each path with its own hit
    // counter, the T_MIN-floor/work-clamp counter idiom). Default 16000
    // (K above ambient) ≈ 2× the design's stated 9000 K extreme — it
    // cannot clip legitimate physics. WHY: the measured B4/B7 runaway is
    // (a) step 4c's multiplicative T·(1−k) update — the ±T_WORK_CLAMP rail
    // bounds the per-tick RATE, never the VALUE, so a persistent
    // compression pocket compounds ~1.5×/tick to the format ceiling —
    // coupled with (b) Pass-1's ΔT=ΔE/(N·c_v) reciprocal dividing the heat
    // deposit by a collapsing near-vacuum N. The physically honest story:
    // a near-vacuum cell's temperature is thermodynamically ill-defined,
    // and real gas would equilibrate such a spike away almost instantly —
    // the cap stands in for that missing fast equilibration. Bounds the
    // runaway regardless of driving term (compression, the reciprocal,
    // anything future).
    float T_MAX_PHYS = 16000.0f;
    // U_MAX (v2.4, PROVISIONAL — same review): defense-in-depth velocity
    // rail; the step-4 store clamp caps |u| at min(c_LOCAL, U_MAX).
    // 1000 m/s is far above any legitimate game wind (ambient c = 300;
    // even a T_MAX_PHYS-hot core's c_LOCAL is only ~2250) — it exists so
    // no future T-side change can push stored velocities back into the
    // int64/narrow overflow regime (see step 4's overflow guard).
    float U_MAX = 1000.0f;
    // trace_mass_scale RETIRED (energy-books arc, P-T0, design §2.6 — the
    // 0% ruling): traces left the Dalton sum entirely rather than keep the
    // half-citizenship (2% pressure weight, zero thermal weight) that fed
    // both the storm audit's pressure pump and the round-1 energy-mint
    // class. N_total is now exactly n_bulk (the gas_conservative pair) —
    // see the Dalton-sum sites in eos_solver.cpp/cuda_eos_step.cu/
    // cuda_eos_resident.cu/cuda_kick_compression.cu, which no longer read a
    // trace weight at all (not wired to 0.0 — the trace planes are skipped
    // outright). Full-citizenship recipe for a future plane: design §2.6.

    // --- MEASUREMENT-ONLY diagnostic (MG gate; never a ship path) --------
    // debug_pstar_from_prev = true replaces step 2's p* = C*N*T_abs with
    // p* = P_prev — the paper's own "pressure is its own evolved state"
    // structure (un-advected; adequate for quiescent-scenario diagnosis).
    // Isolates whether residual slow growth is caused by the named
    // derive-p*-from-(N,T) deviation re-coupling the acoustic loop through
    // the EXPLICIT bulk-N transport. Diagnostic evidence only.
    bool debug_pstar_from_prev = false;

    // --- debug telemetry -----------------------------------------------
    mutable int64_t energy_floor_hits = 0;
    mutable int64_t u_clamp_hits = 0;      // |u| clamped (c_LOCAL or U_MAX)
    // work_clamp_hits: RETIRED with step 4c (arc #54, D10/D11) — the member
    // and its counters_out[2] slot are KEPT so the positional unpacks in the
    // gates/tools do not renumber, and it is ALWAYS 0 from P-G1a on.
    mutable int64_t work_clamp_hits = 0;
    // t_max_phys_hits / energy_floor_hits now count the ONCE-PER-TICK recovery
    // rails (design §2.6), not step 4c's per-cell rails. tests/test_air_
    // boundary.py:820's `t_max_phys_hits == 0` STOP survives verbatim.
    mutable int64_t t_max_phys_hits = 0;
    mutable int64_t u_max_hits = 0;        // clamps where U_MAX (not c_LOCAL)
                                           // was the binding cap (v2.4)
    // --- arc #54 P-G1a rail/telemetry hit counters (design §2.3/§2.4) ------
    //   rad_clip_hits      the ±2^27 component guard (load-side clamp AND the
    //                      post-∇p guard) engaging in the kick loop.
    //   p_face_floor_hits  a per-cell p floored at 0 before eq. 15 (§2.4 F15).
    //   flux_sat_hits      a face magnitude saturated (the int64 corner).
    mutable int64_t rad_clip_hits = 0;
    mutable int64_t p_face_floor_hits = 0;
    //   p_face_ceil_hits   the sub-cycle pressure refresh clamped at the
    //                      physical ceiling C·N·(T_MAX_PHYS + T_AMB) — §2.2's
    //                      own E bound, expressed as the pressure it implies
    //                      (see the fold site for why the increment form
    //                      needs it in the INFLOW direction).
    mutable int64_t p_face_ceil_hits = 0;
    mutable int64_t flux_sat_hits = 0;

    // --- P-E0 energy-bracket counters (energy-books arc, design §2.5) ----
    // Law-independent brackets over S = Σ n_bulk·T on the step-4c skip-set
    // complement (gas cells: !solid, !ts, !vacuum, !ring); n_bulk = the
    // gas_conservative pair summed as int64; T = raw game-T (Q16.16).
    //   eth_transport_delta   = Σ_substeps [S after step-d flux − S at the
    //                           substep transport-block entry]  (HEAD bracket;
    //                           moves to after-recovery at P-E1)
    //   eth_compression_delta = S after − S before the step-4c loop.
    // Pure instrumentation, digest-inert; RESET at step() entry (per tick,
    // the boundary_flux_ idiom) so each read is that tick's delta. These are
    // the counters P-E1's ≤ 0 transport gate measures (design §7).
    mutable int64_t eth_transport_delta = 0;
    // eth_compression_delta RETIRED (arc #54 §2.8, P-G1a): step 4c is gone and
    // the flux step's contribution to Σ_region E is STRUCTURALLY 0 by
    // telescoping, so this bracket can no longer detect anything. The member
    // and its `int` type are KEPT (test_destroy_wall_conserves_mass.py:501's
    // type assert is MECHANICAL, gate 6 STOPs) and it is always 0.
    mutable int64_t eth_compression_delta = 0;

    // =====================================================================
    // arc #54 (gas-energy conservation) — THE ABSOLUTE ENERGY COUNTERS
    // (design §2.3/§2.4/§2.6/§2.8). All int64, all in the field's own Q32
    // currency (E = N_raw · T_abs_raw; dequant = raw / 65536²), all RESET at
    // step() entry (the boundary_flux_ / P-E1 per-tick idiom).
    //
    // THE CLOSURE IDENTITY (§2.8), exact in int64 across ONE EOSSolver::step:
    //
    //   Δ Σ_accountable gas_energy ==
    //         e_entry_resync_sum          (RETIRED at P-G1b; structurally 0)
    //       + e_transport_net_sum         (§2.7 row 1, all substeps)
    //       - e_wipe_sum                  (the N_EPS wipe's destruction)
    //       - e_kick_ke_sum               (∇p kick KE debit)
    //       + e_drag_heat_sum             (structural drag heat, D5)
    //       - e_work_export_sum           (boundary face work, §2.4)
    //       + e_rail_sum                  (§2.6 T_MIN / T_MAX_PHYS rails)
    //
    // The face-flux term itself contributes EXACTLY 0 to that sum over the
    // accountable set (every interior face is applied with opposite signs to
    // its two cells, from identical int64 inputs — §2.5). That is the arc.
    //
    // NOT in the identity, by construction (they never touch gas_energy):
    // e_absorb_export_sum / e_sponge_export_sum / e_clamp_destroyed_sum /
    // e_ts_ke_sum are EXPORTS (D6: numerical dampers and rails export or
    // destroy KE, never heat the gas); e_ts_work_sum / e_wall_work_probe_sum
    // are the D4 accepted-gap PROBES; e_energy_floor_sum is the suppressed
    // (not destroyed) transfer of the positivity rail.
    // =====================================================================
    // RETIRED at P-G1b and structurally 0 (the D10 "retired and zero"
    // convention): the entry re-sync it measured is deleted and `gas_energy`
    // is now the cross-tick truth. Kept as a term so every Python
    // transcription of the identity — the five §6 benches, test_e1_hot_rail,
    // tools/storm_ledger — reads the same seven names it did at P-G1a, and so
    // that a future re-introduction has a booked home instead of being
    // invisible.
    mutable int64_t e_entry_resync_sum = 0;    // RETIRED (P-G1b), always 0
    mutable int64_t e_transport_net_sum = 0;   // Σ over accountable cells of de
    mutable int64_t e_kick_ke_sum = 0;         // ∇p kick bracket (debit, ≥/≤0)
    mutable int64_t e_absorb_export_sum = 0;   // dyn_wave_absorb (exported)
    mutable int64_t e_sponge_export_sum = 0;   // B3c band (exported)
    mutable int64_t e_clamp_destroyed_sum = 0; // velocity cap (destroyed)
    mutable int64_t e_drag_heat_sum = 0;       // drag L+Q (deposited; ex
                                               // e_drag_deposit, slot 6)
    mutable int64_t e_ts_ke_sum = 0;           // thermal_solid cells' brackets
    mutable int64_t e_work_export_sum = 0;     // vacuum/ring OUTFLOW faces
    mutable int64_t e_ts_work_sum = 0;         // PROBE: work lost at ts faces
    mutable int64_t e_wall_work_probe_sum = 0; // PROBE: D4 wall-stencil term
    mutable int64_t e_energy_floor_sum = 0;    // positivity-rail shortfall
    mutable int64_t e_rail_sum = 0;            // §2.6 recovery rails (signed)
    mutable int64_t e_retire_sum = 0;          // P-G1b seam (declared, 0 here)

    // --- P-E1 energy-transport counters (design §2.1.5/§2.5) -------------
    // The one-way guard terms of the new conservative transport law, all in
    // ENERGY units (raw Q16.16², dequant = raw / 65536²) and all int64.
    // Accumulated over the tick's substeps; RESET at step() entry.
    //   e_ts_residual     SIGNED — rule (d) air->ts debits: the relative energy
    //                     gas sheds when it transits a thermal_solid face
    //                     (counted DESTRUCTION; signed because sub-ambient gas
    //                     carries negative relative energy).
    //   e_wipe_sum        SIGNED — residual e destroyed by the N_EPS wipe
    //                     (n_bulk_new < 1 raw count -> T := ambient).
    //   e_floor_sum       energy CREATED by the T_MIN clamp on recovery (a
    //                     CREATOR under R3 — floors destroy, rails may create,
    //                     both counted).
    //   n_active_flux     count of (cell, substep) pairs with ANY nonzero
    //                     touching face dq — the ACTIVE-FLUX fraction §7's
    //                     truncation bound is scaled by (L2-10); quiescent
    //                     cells rebuild T exactly and lose nothing.
    //   n_bulk_active_sum Σ n_bulk_new (raw) over exactly those cells.
    mutable int64_t e_ts_residual = 0;
    mutable int64_t e_wipe_sum = 0;
    mutable int64_t e_floor_sum = 0;
    mutable int64_t n_active_flux = 0;
    mutable int64_t n_bulk_active_sum = 0;

    // --- P-E3 interior-drag oracle counters (design §2.8) -----------------
    // Both n-WEIGHTED (raw ΔT is not comparable to KE) and PER-TICK (the
    // P-E1 reset-at-step()-entry idiom, not P-E2a's accumulate idiom — see
    // the as-built for why: the drag identity is checked ONE tick at a time,
    // like the transport gate, not diffed against a run total). Raw
    // Q16.16^2 (dequant = raw / 65536^2), the SAME "N*T" currency as
    // eth_transport_delta / e_floor_sum / e_ts_residual.
    //   ke_drag_removed     = Sigma n_bulk*(|u_old|^2 - |u_new|^2) over the
    //                         drag loop (structurally >= 0 — the
    //                         magnitude-first shrink guarantees it, no
    //                         clamp/signed term needed).
    // arc #54 P-G1a (D5/D11): `e_drag_deposit`, `e_drag_drop_sum` and
    // `e_drag_rail_clipped` are RETIRED — there is no heat FRACTION any more
    // (the whole ΔKE is deposited, at the derived k_ke), no c_v divide, and no
    // T_MAX_PHYS rail AT the deposit site (the rail runs once per tick in the
    // §2.6 recovery). The one drag energy counter is `e_drag_heat_sum` above.
    // `ke_drag_removed` SURVIVES as the raw KE oracle; the P-E3 identity is
    // restated in absolute currency (tests/test_p_e3_drag.py, rewritten):
    //     e_drag_heat_sum == Σ_cells N_i · trunc_k_ke(Δ|u|²_i)
    // with the SAME per-cell truncation the kick applies (no c_v factor).
    mutable int64_t ke_drag_removed = 0;

    // --- P6.2 telemetry: the substep count the last step() actually ran ---
    // (design §3.2 step 1's n = ceil(dt/dt_adv), N_SUB_MAX-capped). Exposed so
    // the P6 per-kernel digest gates can reconstruct the substep-loop inputs
    // (n_sub is derived from max|u|/∇P/T state that only the solver sees) and
    // replay the isolated kernel on the SAME schedule. Pure telemetry — no
    // arithmetic reads it.
    mutable int dbg_last_n_sub = 0;

    // --- P6.4 telemetry: the c_LOCAL velocity cap the last step() derived ---
    // (q16 raw). c_LOCAL = c_amb·sqrt(T_max_abs/T_AMB) is computed from the
    // PRE-advection temperature scan (§3.2 v2.2), so an isolated replay of the
    // step-4 kick cannot re-derive it from post-tick state. Exposed (the
    // dbg_last_n_sub idiom) so the P6.4 digest gate can feed the isolated
    // kick+compression replay the exact per-tick cap. Pure telemetry — no
    // arithmetic reads it.
    mutable int32_t dbg_last_c_local_q = 0;

    // --- six sub-kernel digest checkpoints (§3.4.6) ---------------------
    mutable uint64_t digest_advect      = 0;
    mutable uint64_t digest_bulk_flux   = 0;
    mutable uint64_t digest_pstar       = 0;
    mutable uint64_t digest_helmholtz   = 0;   // post-solve P (MG or flat)
    mutable uint64_t digest_velocity    = 0;
    mutable uint64_t digest_compression = 0;

    // --- DEBUG probe (temporary instrumentation, eos-p3fix-thermal-ceiling
    // investigation, decisions.md #16): per-tick T checkpoints at ONE traced
    // cell, root-causing which step-1-vs-4c term drives the thermal ceiling.
    // dbg_probe_idx = -1 disables (one branch/tick, no other cost). Values
    // are RAW Q16.16 counts (a Kelvin-delta == raw/65536.0).
    int dbg_probe_idx = -1;
    mutable int32_t dbg_T_pre_advect       = 0;   // T at step-1 entry
    mutable int32_t dbg_T_post_advect      = 0;   // T after the SL substep loop
    mutable int32_t dbg_T_post_compression = 0;   // T after step 4c

    // BC (boundary_conditions_spec_2026-07-19): the planetside AMBIENT ring.
    // is_ambient/n_amb/p_amb/sponge_sigma default null/0 so the space path is
    // byte-identical (dormancy BY BRANCH — spec §5; every ambient edit inside
    // step() is gated on is_ambient != nullptr):
    //   is_ambient    : (h,w) ring mask (nullptr = space map)
    //   n_amb         : (n_gases,) per-plane ambient N for the ring clamp
    //   p_amb         : effective pin P_amb (raw q16) — the shift trick's shift
    //   sponge_sigma  : (h,w) level-0 diagonal sponge mass (B3b σ; nullptr = off)
    //   sponge_udamp  : (h,w) k(d) velocity-damping band coefficient, Q16 in
    //                   [0,FP_ONE) (B3c rung 2, the real absorber; nullptr = off)
    void step(
        int32_t* atmosphere,
        int32_t* p_prev,
        int32_t* wind_x, int32_t* wind_y,
        int32_t* temperature,
        // arc #54 (design §2.2/§6 P-G1a): `gas_energy`, the (h,w) int64
        // CONSERVED gas thermal energy field — the exact unshifted product
        // N_raw·T_abs_raw on the accountable set, 0 elsewhere. TRANSITIONAL
        // at P-G1a: step() RE-SYNCS it from the (N, T) mirror at entry (T is
        // still the cross-tick truth) and REFRESHES the mirror from it at
        // exit (the §2.6 recovery). nullptr is NOT accepted — every caller
        // passes GameMap.gas_energy.
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
        // THERMAL-MASS AXIS, P-EOS (see the file header): the per-medium
        // THERMAL mask (`thermal_mass > 0`, GameMap.thermal_solid). Steps 1b
        // and 4c skip their `temperature[i]` write where it is true, and the
        // step-1b T backtrace treats those tiles as occluders. `cmask` — hence
        // pressure/velocity/gas flow — is UNTOUCHED. Default nullptr ->
        // fall back to `solid`, the pre-patch behaviour byte-for-byte (the
        // documented back-compat idiom this signature already uses for the BC
        // args).
        const bool* thermal_solid = nullptr) const;

    // BC (spec §5): the boundary_flux rail — per-conservative-plane int64 sum of
    // the ring mass exchange, accumulated at the reset site per substep. NOT
    // folded into any digest (absence-transparent; zero golden re-baseline).
    // Sized n_gases in ambient mode, empty otherwise. PUBLIC mutable (the
    // digest/counter telemetry pattern) so the CUDA path (eos_step_cuda) writes
    // it exactly as the CPU step() does; the getter gives the read-only view.
    mutable std::vector<int64_t> boundary_flux_;
    const std::vector<int64_t>& boundary_flux() const { return boundary_flux_; }

    // ---- one multigrid level (v2.2 D-B) ---------------------------------
    // (EOS P6.3: struct made public — unchanged fields — so the CUDA binding
    // can view the host-built hierarchy; levels_ itself stays private.)
    // All coefficient fields rebuilt every tick (p*/N/perm change per tick).
    // excl: 0 = regular equation cell; 1 = Dirichlet (vacuum, P pinned 0);
    //       2 = excluded (solid — no equation; faces into it carry g = 0,
    //       the Neumann mirror).
    // v2.2-final level form (adopted at the MG measurement gate after the
    // nonsymmetric row-scaled form's deep pyramid was MEASURED divergent —
    // more cycles made it worse, the signature of an amplifying coarse
    // correction): the row is divided by aK_i, giving the SYMMETRIC
    // POSITIVE-DEFINITE system  m_i·P_i + Σ_f g_f·(P_i − P_nb) = m_i·rhs_i
    // with m_i = 1/aK_i (the near-vacuum degeneracy now reads as a HUGE
    // mass pinning P→rhs — same solution, stable row). Coarsening is
    // exactly-variational piecewise-constant Galerkin: masses SUM,
    // face conductances SUM (interior faces cancel), residuals SUM,
    // prolongation is piecewise-constant injection (the exact transpose)
    // — the two-grid correction is an energy-norm projection and cannot
    // amplify, at ANY pyramid depth.
    struct MGLevel {
        int h = 0, w = 0;
        std::vector<uint8_t> excl;
        std::vector<int64_t> m;       // per-cell mass 1/aK — int64 Q16.16 raw, clamped
        std::vector<int64_t> gE, gS;  // face conductance (sums on coarsening) — Q16.16 raw
        std::vector<int64_t> recip;   // per-cell diag reciprocal (2^48/d_raw)
        std::vector<int32_t> P;       // solution / correction (q16 raw)
        std::vector<int64_t> b;       // RHS m·rhs at the F8 work scale ((raw·raw)>>8)
        std::vector<int64_t> res;     // residual scratch (F8 scale)
    };

    // ---- EOS P6.3: the pressure solve, split into its two internal stages
    // (PURE CODE MOTION out of step() — identical arithmetic, identical
    // order; step() now calls these two back-to-back). The split exists so
    // (1) the CUDA binding can run the SAME host-side per-tick hierarchy
    // build and hand the built levels to the device V-cycle, and (2) the
    // standalone CPU replay entry (mg_solve_reference below) can drive the
    // SAME internal routines the live path uses — zero drift by construction.
    //
    // ---- S8a Path A: the MG-build per-tick scalar folds, EXPORTED ---------
    // (docs/cuda_s8a_path_a_impl_2026-07-21.md §3.2.3, critique blocker A-B1.)
    // The four folds mg_build_levels hoists — n_floor_q, gamma_q, dt_q and the
    // 5-op double expression Kdt2dx2_raw (with its std::max(dx,1e-6) floor) —
    // are the ONLY nontrivial double arithmetic feeding the operator build.
    // The device-resident build (cuda_eos_resident.cu) must consume the
    // IDENTICAL bits, so there is exactly ONE transcription: this method, in
    // this /fp:strict MSVC TU. mg_build_levels itself calls it (pure code
    // motion — CPU bytes unchanged, pinned by the existing CPU goldens).
    // Caller guards dt > 0 (the mg_build_levels degenerate early-out).
    struct MGScalarFolds {
        int32_t n_floor_q   = 0;   // q16
        int32_t gamma_q     = 0;   // q16
        int32_t dt_q        = 0;   // q16
        int64_t Kdt2dx2_raw = 0;   // K·dt²/dx² at Q16.16 raw
    };
    MGScalarFolds mg_scalar_folds(float dt) const;

    // mg_build_levels: level count (fixed by grid size), the level-0 build
    // (m/gE/gS/b/excl + the P_prev warm start), the exactly-variational PC
    // Galerkin coarse hierarchy, and the per-level Q.32 diagonal reciprocals.
    // PER-TICK by necessity: level-0 m derives from p* and gE/gS fold the
    // per-tick 1/N̂, and every coarse operator is a Galerkin sum of those —
    // so the whole pyramid follows the tick cadence (review §2.7). The
    // per-tick scalar folds (n_floor_q, gamma_q, dt_q, Kdt2dx2_raw) are
    // recomputed here from the SAME double expressions step() used to hoist
    // (/fp:strict TU — identical bits). Returns n_levels; 0 on degenerate
    // input (n <= 0 or dt <= 0), in which case levels_ is untouched.
    // BC: is_ambient/p_amb/sponge_sigma default null/0 so the existing callers
    // (cuda_eos_step.cu, eos_mg_solve_reference) compile unchanged and run the
    // byte-identical space path. The live eos.step() passes the real values:
    // the SHIFT (subtract p_amb from rhs + warm start), the ring→Dirichlet excl,
    // and the σ-sponge (B3b) all live here, all gated on is_ambient != nullptr.
    int mg_build_levels(
        const int32_t* pstar, const int32_t* div_u, const int32_t* n_total,
        const int32_t* p_prev,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability,
        int h, int w, float dt,
        const bool* is_ambient = nullptr,
        int32_t p_amb = 0,
        const int32_t* sponge_sigma = nullptr) const;

    // mg_run_solve_cpu: the fixed-schedule V-cycles (or the flat RB-GS
    // reference path), the vacuum-Dirichlet/solid zero on level 0, and the
    // digest_helmholtz checkpoint. Operates on the levels_ mg_build_levels
    // just built. No-op for n_levels <= 0.
    void mg_run_solve_cpu(int n_levels) const;

    // Read-only view of the built hierarchy (the CUDA binding flattens this
    // into device uploads; level-0 P is the warm start / solved P).
    const std::vector<MGLevel>& mg_levels() const { return levels_; }

    // EOS P6.3 gate telemetry — read-only views of the per-tick solve-input
    // caches AS LEFT by the last step() (nothing after the solve writes
    // them): pstar_/div_u_ are the step-2/RHS fields, n_total_ is the
    // step-2 (post-substep) Dalton sum the face conductances folded. The
    // digest gate reads these after a real engine tick to reconstruct the
    // EXACT solve inputs (p_prev is already engine-visible). Pure telemetry.
    const std::vector<int32_t>& dbg_pstar_cache()   const { return pstar_; }
    const std::vector<int32_t>& dbg_div_u_cache()   const { return div_u_; }
    const std::vector<int32_t>& dbg_n_total_cache() const { return n_total_; }

private:
    mutable std::vector<MGLevel> levels_;

    // Reused per-tick scratch (house pattern: no per-tick alloc).
    mutable std::vector<int32_t> n_total_;
    // VELOCITY-CLAMP (P-V1, design v3, D2v2): the per-cell velocity-cap²
    // plane (Q32.32 raw), folded once per tick alongside c_LOCAL in the same
    // scan (eos_solver.cpp:397-427) — the kick trusts it verbatim (D5).
    mutable std::vector<int64_t> cap2_plane_;
    mutable std::vector<int32_t> vx_src_, vy_src_, t_src_;
    mutable std::vector<int32_t> pstar_;
    mutable std::vector<int32_t> div_u_;
    // per-tick caches for the substep loop (micro-opt, bit-identity-neutral):
    mutable std::vector<uint8_t> cmask_;              // sealed/breach/live table
    // (The THERMAL-MASS-AXIS T-ONLY corner/march mask `tcmask_` lived here and
    // is RETIRED at P-E1 — energy-books design §2.1.1. Its ONLY consumer was
    // the step-1b temperature sample, and that sample is gone: temperature now
    // rides the conservative energy books, where a thermal_solid tile is
    // excluded structurally by ts-face rule (d) rather than by an occluder
    // mask. `cmask_` above is untouched, so velocity / pressure / gas flow are
    // bit-identical.)
    mutable std::vector<int32_t> coeffE_, coeffS_;    // donor-cell face coeffs
    // P-E1 (design §2.1.2): the TRANSIENT energy accumulator + the applied
    // per-face dq planes. int64, scratch only — NOT synced state, never
    // digested, rebuilt from (n_bulk, T) at the top of every substep (R4).
    mutable std::vector<int64_t> e_scratch_, dqsum_e_, dqsum_s_;
    // arc #54 §2.7 row 1: the transport's PRE-flux bulk-N denominator plane
    // (e_scratch_ becomes its energy sibling — see bulk_transport.h).
    mutable std::vector<int64_t> n_pre_;
    // arc #54 P-G1a scratch (design §2.4/§2.5). All rebuilt every tick, never
    // synced, never digested. RL-batch habits: CPU keeps (h,w); the device
    // twin (P-G2) allocates (N,h,w) with N=1.
    //   e0_      E at the ENERGY-PASS entry — the increment-form pressure
    //            refresh's one fixed baseline (§2.4 F2/R3-#6);
    //   pcur_    the per-sub-cycle refreshed cell pressure, materialized into
    //            a PLANE so pass A and pass B read byte-identical operands;
    //   s_plane_ the donor-only positivity rail's per-cell Q16 scale (F3/F13).
    mutable std::vector<int64_t> e0_;
    mutable std::vector<int32_t> pcur_;
    mutable std::vector<int32_t> s_plane_;
};

// ---------------------------------------------------------------------------
// P-M4b (mass-books arc) — THE energy-books sum, extracted so there is exactly
// ONE implementation of the accountable set.
//
// S = Σ n_bulk·T over the step-4c skip-set complement, i.e. every cell that is
// NOT (solid || thermal-solid || vacuum || ambient-ring); n_bulk = the
// gas_conservative planes summed as int64; T = raw game-T (Q16.16). Units are
// raw Q16.16² (dequant = raw / 65536²) — the same currency as
// eth_transport_delta / e_wipe_sum / e_floor_sum.
//
// WHY IT LIVES HERE. This was a `[&]`-capturing lambda local to
// EOSSolver::step, so the only way to read the books from Python was to
// re-implement the four-flag skip-set on the Python side — which would drift
// from this file the first time the skip-set changed, and a books instrument
// that silently disagrees with the books is worse than none. Both callers now
// go through this one function: step()'s per-tick brackets
// (eth_transport_delta / eth_compression_delta) and the `eos_energy_books_sum`
// binding the mass-books gates measure Δ(books) across a destruction with.
//
// PURE INSTRUMENTATION — nothing in the sim path reads S, no digest folds it.
// The extraction is byte-for-byte the lambda's arithmetic in the lambda's loop
// order; it moves no behaviour (gated by the canonical-scenario digest).
//
// `thermal_solid == nullptr` falls back to `solid` — the same back-compat
// idiom EOSSolver::step's `ts` uses, and the reason the fallback is HERE and
// not at the call site: the binding gets it for free.
// `is_ambient == nullptr` means space map -> the ring term is dormant BY
// BRANCH, exactly as `ambient_mode` gates it inside step().
//
// arc #54 P-G1a (design §2.8): the sum is now READ OFF THE FIELD —
//     S = Σ_accountable ( gas_energy_i − n_bulk_i · T_AMB_raw )
// which is the SAME quantity (gas_energy ≡ n_bulk·(T + T_AMB_raw) whenever
// the mirror is current) in the SAME units, computed as an int64 per-cell
// DIFFERENCE then summed — never as two absolute sums (§2.2's overflow rule).
// `gas_energy == nullptr` falls back to the pre-#54 `n_bulk · temperature`
// form, byte-identically, so a caller without the field (a bare unit test)
// still measures the same books.
int64_t eos_energy_books_sum(
    const int32_t* gas, const bool* gas_conservative, int n_gases,
    const int32_t* temperature,
    const bool* solid, const bool* is_vacuum,
    int n,
    const bool* is_ambient = nullptr,
    const bool* thermal_solid = nullptr,
    const int64_t* gas_energy = nullptr,
    int32_t t_amb_raw = 0);

// ---------------------------------------------------------------------------
// EOS P6.2 — standalone CPU reference for the SL-advection substep loop
// (docs/eos_p6_gpu_alignment_review.md §4 row P6.2). Replays EXACTLY the
// step-1a/1f chain of EOSSolver::step for a GIVEN substep count: the per-tick
// cmask build, then n_sub x [src snapshot -> fused backtrace -> u write /
// zero-u-on-solid], IN PLACE on wind_x/wind_y. Calls the SAME file-local
// eos_backtrace_sample3_q the real solver uses (one routine, zero drift).
//
// P-E1 (energy-books arc, design §2.1.1 — CONTRACT CHANGE, authorized rewrite
// Appendix A): **this reference is now u-ONLY.** The SL sample's `.t` slot is
// RETIRED — temperature is transported by the conservative energy books
// (bulk_flux_energy_transport_cached), not by a semi-Lagrangian copy — so
// `temperature` here is a READ-ONLY src slot for the fused sampler (its `.t`
// result is discarded, exactly as in step(); `.vx`/`.vy` do not depend on it)
// and the returned digest is the chained FNV over (wind_y, wind_x) ALONE.
// It is therefore NO LONGER comparable to `digest_advect`, which now hashes
// (wx, wy, T-after-recovery) and is taken AFTER the flux call (§2.1.6). The
// gate it still serves is CPU-reference vs GPU-twin bit-identity on u.
// ---------------------------------------------------------------------------
// BC: is_ambient defaults nullptr — existing test callers stay byte-identical;
// when supplied, the ring is a still-boundary breach corner (cmask).
// THERMAL-MASS AXIS: `thermal_solid` is RETIRED here with the T sample — the
// A2 T-only occluder mask had no consumer once the `.t` slot went (design
// §2.1.1). The parameter is kept for ABI/back-compat and ignored.
uint64_t eos_sl_advect_reference(
    int32_t* wind_x, int32_t* wind_y, int32_t* temperature,
    const bool* solid, const bool* is_vacuum,
    const float* dyn_permeability,
    int h, int w, float dt, int n_sub,
    const bool* is_ambient = nullptr,
    const bool* thermal_solid = nullptr);

// ---------------------------------------------------------------------------
// EOS P6.4 — standalone CPU reference for the post-solve tail: the step-4
// momentum kick (u -= dt·K·grad(P_new)/N̂ → absorption damping → the
// per-cell cap2_plane counted magnitude clamp, with the ±2^30 component
// pre-clamp overflow guard) AND the step-4c compression work
// (T -= (γ−1)·T·div(u_new)·dt with the ±T_WORK_CLAMP rail, T_MIN floor and
// T_MAX_PHYS ceiling, all counter-tracked) — docs/eos_p6_gpu_alignment_review.md
// §4 row P6.4. Replays EXACTLY EOSSolver::step's chain on the given step-4-entry
// state, IN PLACE on wind_x/wind_y/temperature.
//
// "EXACTLY" IS NOW TRUE AGAIN (audit Patch A / A6, 2026-08-04). It was false
// from B3c until then: the sponge velocity-damping band step() applies after
// the absorb chain was never added here, so the ambient/planetside path was
// structurally OUTSIDE this gate's coverage — and since the CUDA twin
// (cuda_kick_compression.cu:168) does have the band, a lockstep failure there
// would have blamed the GPU for a drifted CPU reference. Pass `sponge_udamp`
// (with `is_ambient`) to replay it; both default nullptr, which reproduces the
// pre-A6 behaviour byte for byte.
//
// VELOCITY-CLAMP (P-V1, design v3): the contract INVERTS from the pre-P-V1
// shape above — the cap is now fully derivable from the replay's own inputs.
// `cap2_plane` is folded from the SAME tick-entry T (== `temperature` on
// entry, the P6.4 replay's own t0 state) via formula A, so a caller with the
// step-4-entry state in hand can reconstruct the exact plane itself (no
// hidden `dbg_last_c_local_q` telemetry dependency the way the old scalar
// cap needed — the P6.4 replay used to be the ONE place that couldn't see
// the pre-advection T scan; now it can, because the plane rides the same t0
// the caller already has).
// Inputs:
//   * p_new           — the solved pressure plane the kick differentiates
//                       (== L0.P after the vacuum/solid zeroing == the post-tick
//                       `atmosphere`, which step 5 copies verbatim);
//   * gas planes      — the step-2 Dalton sum N_total (P-T0, design §2.6:
//                       n_total ≡ n_bulk, the gas_conservative pair summed
//                       at full weight; trace planes contribute nothing) is
//                       recomputed here verbatim: it is the kick's 1/N̂ input;
//   * cap2_plane      — the per-cell Q32.32 velocity-cap-squared plane (D5:
//                       the kick TRUSTS it verbatim, no re-min against U_MAX
//                       here — the caller/scan owns floor/ts/min policy;
//                       HARD CONTRACT: every entry must be >= 0, or a
//                       divide-by-zero is reachable inside the clamp branch);
//   * scalar params   — the EOSSolver config members, folded to q16/int64
//                       through the IDENTICAL double expressions step() uses.
// Outputs: the SAME chained FNV digest step() stores in digest_velocity, plus
// the NINE counters FOR THIS CALL in
// counters_out[9] = { u_clamp_hits, u_max_hits, 0, 0, 0,
//                     ke_drag_removed, e_drag_heat_sum, 0, 0 }
// arc #54 P-G1a, D10: the LAYOUT IS KEPT so no positional unpack renumbers.
// Slot 2 (work_clamp_hits), slots 3/4 (the step-4c energy_floor / T_MAX_PHYS
// rails) and slots 7/8 are RETIRED-AND-ZERO: step 4c is gone from this
// reference, so it no longer writes `temperature` at all and
// *digest_compression_out is the digest of the UNCHANGED temperature plane
// (the once-per-tick recovery that now owns those rails lives in step(), not
// in this isolated tail replay). Slot 5 stays the raw KE oracle; slot 6 is
// the arc's one drag energy counter. All ENERGY slots are int64 sums in the
// gas_energy Q32 currency, not hit counts.
// Test entry only — the live path remains EOSSolver::step.
// ---------------------------------------------------------------------------
void eos_kick_compression_reference(
    int32_t* wind_x, int32_t* wind_y, int32_t* temperature,   // in/out
    // arc #54 P-G1a: the (h,w) int64 gas_energy field the KE brackets debit /
    // credit (§2.3). IN/OUT. May be nullptr — then the brackets are computed
    // and COUNTED exactly as in step(), but nothing is stored (the pre-#54
    // callers' shape, kept so a gate can measure the brackets alone).
    int64_t* gas_energy,
    const int32_t* p_new,                                     // solved P (q16)
    const int32_t* gas, const bool* gas_conservative, int n_gases,
    const bool* solid, const bool* is_vacuum,
    const float* dyn_wave_absorb,
    int h, int w, float dt, const int64_t* cap2_plane,        // D2v2 (h,w) Q32.32, >= 0
    float c_max, float dx, float adiabatic_index, float absorb_strength,
    float n_floor_solver, float t_min,
    float t_max_phys, float u_max,   // t_work_clamp param RETIRED (arc #54 D11)
    // P-E3 (design §2.8): interior drag + its heat counterparty. k_drag
    // default 0.0 -> the mechanism is dormant (branch on the QUANTIZED kd_q,
    // not these floats — see the .cpp). k_drag2 (drag-law v2, design
    // docs/drag_law_v2_design_2026-08-23.md): the quadratic term, same
    // dormancy idiom (kd2_q).
    // arc #54 P-G1a (D5): `k_drag_heat_frac` and `n_work_ref` are RETIRED
    // (there is no heat fraction and no trust gate any more). `c_v` stays in
    // the signature as the config echo — it is NOT read by the drag deposit
    // (that is the derived k_ke now).
    float k_drag, float k_drag2, float c_v,
    // ambient K — folded the SAME A7-floored expression as the CPU live path.
    // arc #54: now LOAD-BEARING in the KE brackets too (k_ke is derived from
    // c_max, adiabatic_index and this).
    float t_amb_k,
    uint64_t* digest_velocity_out, uint64_t* digest_compression_out,
    int64_t* counters_out /* [9] */,
    const bool* is_ambient = nullptr,    // BC: ring u ≡ 0 (defaults off)
    // THERMAL-MASS AXIS, P-EOS: step-4c skips its T write on thermal_solid
    // tiles (the kick is untouched). Default nullptr -> `solid` -> pre-patch.
    const bool* thermal_solid = nullptr,
    // B3c sponge velocity-damping band (audit Patch A / A6) — the static k(d)
    // coefficient plane step() applies immediately after the absorb chain.
    // Only read when is_ambient != nullptr (dormancy by branch, as in step()).
    // Default nullptr reproduces the pre-A6 reference exactly.
    const int32_t* sponge_udamp = nullptr);

// ---------------------------------------------------------------------------
// EOS P6.3 — standalone CPU reference for the multigrid pressure solve
// (docs/eos_p6_gpu_alignment_review.md §4 row P6.3; the eos_sl_advect_reference
// pattern). Replays EXACTLY step 3 of EOSSolver::step for GIVEN solve inputs by
// calling the SAME two internal routines the live path calls
// (solver.mg_build_levels + solver.mg_run_solve_cpu — one code path, zero
// drift): per-tick hierarchy build (level 0 from pstar/div_u/n_total/p_prev +
// masks/perm, PC-Galerkin coarse levels, Q.32 diagonal reciprocals), the
// frozen V(nu1,nu2)xC schedule (or flat RB-GS when solver.use_multigrid is
// false), the vacuum-Dirichlet/solid zero, and the digest_helmholtz FNV over
// the solved level-0 P. Writes the solved P (== step 5's atmosphere
// materialization == the step-4 kick's Pn input) into p_out (h*w int32) and
// returns the digest — so a gate that reconstructs the solve inputs from a
// real tick can assert  reference == EOSSolver.digest_helmholtz, then hold
// the GPU V-cycle to the identical bytes. Takes the solver instance for the
// config surface (dx/c_max/gamma/N_FLOOR_SOLVER + the frozen MG schedule).
// Test entry only — the live path remains EOSSolver::step.
// ---------------------------------------------------------------------------
uint64_t eos_mg_solve_reference(
    const EOSSolver& solver,
    const int32_t* pstar, const int32_t* div_u, const int32_t* n_total,
    const int32_t* p_prev,
    const bool* solid, const bool* is_vacuum,
    const float* dyn_permeability,
    int h, int w, float dt,
    int32_t* p_out);
