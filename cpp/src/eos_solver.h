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

#include <cstdint>
#include <vector>

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
    float T_AMB_K = 290.0f;
    float C = 1.0f / 290.0f;
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

    // --- P6.2 telemetry: the substep count the last step() actually ran ---
    // (design §3.2 step 1's n = ceil(dt/dt_adv), N_SUB_MAX-capped). Exposed so
    // the P6 per-kernel digest gates can reconstruct the substep-loop inputs
    // (n_sub is derived from max|u|/∇P/T state that only the solver sees) and
    // replay the isolated kernel on the SAME schedule. Pure telemetry — no
    // arithmetic reads it.
    mutable int dbg_last_n_sub = 0;

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

    void step(
        int32_t* atmosphere,
        int32_t* p_prev,
        int32_t* wind_x, int32_t* wind_y,
        int32_t* temperature,
        int32_t* gas, const bool* gas_conservative, int n_gases,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability, const float* dyn_wave_absorb,
        int h, int w, float dt) const;

private:
    // ---- one multigrid level (v2.2 D-B) ---------------------------------
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
    mutable std::vector<MGLevel> levels_;

    // Reused per-tick scratch (house pattern: no per-tick alloc).
    mutable std::vector<int32_t> n_total_;
    mutable std::vector<int32_t> vx_src_, vy_src_, t_src_;
    mutable std::vector<int32_t> pstar_;
    mutable std::vector<int32_t> div_u_;
    // per-tick caches for the substep loop (micro-opt, bit-identity-neutral):
    mutable std::vector<uint8_t> cmask_;              // sealed/breach/live table
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
uint64_t eos_sl_advect_reference(
    int32_t* wind_x, int32_t* wind_y, int32_t* temperature,
    const bool* solid, const bool* is_vacuum,
    const float* dyn_permeability,
    int h, int w, float dt, int n_sub);
