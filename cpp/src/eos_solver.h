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
// c is state-derived (c ∝ √T); the per-tick velocity cap is
// c_LOCAL = c_amb·sqrt(T_max_abs/T_AMB), never a stale ambient constant.
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
//   4c. compression work ONCE, post-correction, on div(u_new); T_MIN floor
//      + T_WORK_CLAMP rail, both counter-tracked
//   5. P := P_new stored once (the `atmosphere` alias)

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
//     T-ONLY mask (`tcmask_` below) — occluding the shared fused march would
//     have moved the VELOCITY self-advection, which item 4 forbids.
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
    float T_WORK_CLAMP = 0.5f;
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
    // trace_mass_scale (P3 integration constant, FLAGGED for Erik): the
    // Dalton sum N_total = Σ N_i must weight the TRACE planes by the molar
    // mass of a full-opacity cloud relative to ambient — the trace fields
    // are [0,1] OPACITY tracers, not molar densities, and an unweighted sum
    // makes a 0.6-opacity teargas cloud a +60% pressure bomb that blasts
    // itself apart in one tick (measured). 0.02 == "a fully opaque cloud
    // carries 2% of ambient molar density" — keeps the design §2.1 premise
    // ("the bulk pair carries ~99% of N_total") true by calibration.
    float trace_mass_scale = 0.02f;

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
    mutable int64_t work_clamp_hits = 0;   // step-4c factor rail engagements
    mutable int64_t t_max_phys_hits = 0;   // step-4c T_MAX_PHYS rail (v2.4)
    mutable int64_t u_max_hits = 0;        // clamps where U_MAX (not c_LOCAL)
                                           // was the binding cap (v2.4)

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
    mutable int64_t eth_compression_delta = 0;

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
    mutable std::vector<int32_t> vx_src_, vy_src_, t_src_;
    mutable std::vector<int32_t> pstar_;
    mutable std::vector<int32_t> div_u_;
    // per-tick caches for the substep loop (micro-opt, bit-identity-neutral):
    mutable std::vector<uint8_t> cmask_;              // sealed/breach/live table
    // THERMAL-MASS AXIS, P-EOS: the T-ONLY corner/march mask — `cmask_` with
    // every thermal_solid tile forced to 0 (sealed/occluder). Built beside
    // cmask_ and used ONLY by the step-1b temperature sample; `cmask_` itself
    // (velocity, and therefore pressure and gas flow) is untouched. Left empty
    // and unread when the two masks cannot differ (eos_thermal_occludes).
    mutable std::vector<uint8_t> tcmask_;
    mutable std::vector<int32_t> coeffE_, coeffS_;    // donor-cell face coeffs
};

// ---------------------------------------------------------------------------
// EOS P6.2 — standalone CPU reference for the fused SL-advection substep loop
// (docs/eos_p6_gpu_alignment_review.md §4 row P6.2). Replays EXACTLY the
// step-1a/1b/1f chain of EOSSolver::step for a GIVEN substep count: the
// per-tick cmask build, then n_sub x [src snapshot -> fused 3-field backtrace
// (vx, vy, T) -> zero-u-on-solid / T:=0-on-vacuum], IN PLACE on
// wind_x/wind_y/temperature. Calls the SAME file-local eos_backtrace_sample3_q
// the real solver uses (one routine, zero drift), and returns the SAME chained
// FNV digest step() stores in digest_advect at its last substep — so a gate
// that reconstructs step-1-entry state + n_sub can assert
// eos_sl_advect_reference(...) == solver.digest_advect, then hold the GPU port
// to the identical bytes. The interleaved bulk flux (step 1d, P6.1) neither
// reads nor writes u/T, so replaying the advection substeps back-to-back is
// arithmetically identical to the real interleaved loop. Test entry only —
// the live path remains EOSSolver::step.
// ---------------------------------------------------------------------------
// BC: is_ambient defaults nullptr — existing test callers stay byte-identical;
// when supplied, the ring is a still-boundary breach corner (cmask) and T:=0.
// THERMAL-MASS AXIS, P-EOS: `thermal_solid` defaults nullptr — existing test
// callers stay byte-identical; when supplied the replay applies the SAME
// skip-write + T-only-occluder rules step() does (one code path, zero drift).
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
// min(c_LOCAL, U_MAX) counted magnitude clamp, with the ±2^30 component
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
// Inputs:
//   * p_new           — the solved pressure plane the kick differentiates
//                       (== L0.P after the vacuum/solid zeroing == the post-tick
//                       `atmosphere`, which step 5 copies verbatim);
//   * gas planes      — the step-2 Dalton sum N_total (bulk planes at full
//                       weight, trace planes × trace_mass_scale) is recomputed
//                       here verbatim: it is the kick's 1/N̂ input;
//   * c_local_q       — the per-tick state-derived cap the solver computed
//                       PRE-advection (EOSSolver::dbg_last_c_local_q);
//   * scalar params   — the EOSSolver config members, folded to q16/int64
//                       through the IDENTICAL double expressions step() uses.
// Outputs: the SAME chained FNV digests step() stores in digest_velocity /
// digest_compression, plus the five rail counters FOR THIS CALL in
// counters_out[5] = { u_clamp_hits, u_max_hits, work_clamp_hits,
// energy_floor_hits, t_max_phys_hits } (the solver's members are cumulative;
// a gate compares per-tick deltas). Counter semantics are the solver's own:
// ONE increment per engaging CELL (the |u| clamp is a magnitude event, not
// per-component; the 4c rails are an exclusive if/else-if chain). Test entry
// only — the live path remains EOSSolver::step.
// ---------------------------------------------------------------------------
void eos_kick_compression_reference(
    int32_t* wind_x, int32_t* wind_y, int32_t* temperature,   // in/out
    const int32_t* p_new,                                     // solved P (q16)
    const int32_t* gas, const bool* gas_conservative, int n_gases,
    const bool* solid, const bool* is_vacuum,
    const float* dyn_wave_absorb,
    int h, int w, float dt, int32_t c_local_q,
    float c_max, float dx, float adiabatic_index, float absorb_strength,
    float n_floor_solver, float t_min, float t_work_clamp,
    float t_max_phys, float u_max, float trace_mass_scale,
    uint64_t* digest_velocity_out, uint64_t* digest_compression_out,
    int64_t* counters_out /* [5] */,
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
