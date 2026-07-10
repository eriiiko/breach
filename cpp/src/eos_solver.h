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
// EOS refactor P3 (docs/eos_refactor_design.md §3, §8 patch P3). Replaces
// AtmosphereSolver::wave_substep + ::diffuse_solve. `atmosphere` becomes a
// zero-copy alias of the solved P (materialized ONCE per tick, step 5);
// `wind_x`/`wind_y` become the solver's OWN velocity state `u` (self-advected
// + pressure-corrected here, not merely derived from a static gradient).
//
// TICK ORDER (design §3.2 — the load-bearing spec this implements exactly):
//   0. P_prev := P (kept copy, for the ripple transient + the digest gate)
//   1. ADVECTION SUBSTEPS, n = ceil(dt/dt_adv), integer-ceil discipline,
//      capped at N_SUB_MAX (accuracy cap, not a stability cap — SL is
//      unconditionally stable and the donor-cell limiter rate-caps
//      gracefully):
//        a. u  <- semi-Lagrangian self-advection
//        b. T  <- semi-Lagrangian advection (open-air mask)
//        c. (traces do NOT substep here — advected once/tick by the caller,
//           PhysicsEngine::run_substeps, on the final u)
//        d. bulk O2/N2 <- donor-cell conservative flux on u (bulk_transport.h)
//        e. T -= (gamma-1)*T*div(u)*dt_s   (compression work, T-floored,
//           every floor hit increments `energy_floor_hits`)
//        f. zero u on solid; N floored/zeroed at solid+vacuum by (d)'s own
//           clamp (bulk_transport.cpp) — the OCCUPANCY-TRANSITION evacuation
//           rule (§2.2, cell LEAVING open-air) is a WRITER concern (W3 water
//           displacement / door-close / wall-spawn), not this per-tick pass.
//   2. p* := C * N_total * (T + T_AMB_K)          (wide mul, §3.4)
//   3. HELMHOLTZ SOLVE (once per tick, S fixed sweeps, RB-GS, wide int64 —
//      §3.4): Neumann mirror at solid, Dirichlet P=0 at vacuum, face
//      coefficients permeability-scaled.
//   4. u -= dt*grad(P_new)/N_hat; u *= (1 - absorb*dt) (dyn_wave_absorb);
//      zero u outside open-air.
//   5. P := P_new, stored ONCE (the `atmosphere` alias).
//
// Six sub-kernel digest checkpoints (§3.4.6) let the P3 gate assert
// determinism across two identical runs without a single end-of-tick hash
// hiding a compensating-error pair. CPU-sequential order-dependent hash
// (NOT order-free) — sufficient for the CPU-lockstep gate; a future GPU port
// (P6, out of scope here) needs its own order-free reduction.

#include <cstdint>
#include <vector>

class EOSSolver {
public:
    // ---- PINNED constants (docs/eos_refactor_decisions.md; design §8/§9) --
    // c_max: config/Erik's call (the old wave_c=66 was a perf compromise
    // Kwatra retires — see docs/eos_refactor_decisions.md 2026-07-10).
    float c_max = 300.0f;
    // dx: the level's physical tile size (metres), bound by the caller from
    // gmap.tile_size_m — the SAME lazy-bind idiom as WaterSolver::dx. The
    // design's c_max=300 m/s and the microbench's overflow budget (design
    // §3.4 "k_f <= 2c^2dt^2/dx^2 ~= 11,180 at c=300") are BOTH derived at
    // the physical dx=1/3 m, not the tile-unit dx=1 convention the older
    // wave/diffusion solvers used — this solver follows the design's
    // physical-units convention (a deliberate, flagged departure from
    // AtmosphereSolver's "c in tiles/s" convention).
    float dx = 0.333f;
    // S: Helmholtz RB-GS sweep count. Starts at 8 per the design's "start 8,
    // gate may pin <=16" instruction; the P3 convergence gate (measured on
    // the stress scenarios at c=300 vs a high-sweep reference) pins the
    // FROZEN value — never adaptive once pinned. See docs/eos_p3_solver_gate.md.
    int   S = 8;
    // N_SUB_MAX: advection substep cap. LOCKED AT 16 (not a stability bound —
    // SL is unconditionally stable; this trades front-resolution on a blast's
    // wildest tick for a bounded frame cost). docs/eos_refactor_decisions.md
    // 2026-07-10 (commit c109d4e) overrides the raw microbench recommendation
    // of 512 with this reasoned 16.
    int   N_SUB_MAX = 16;
    float CFL_ADV = 0.5f;
    // N_FLOOR_SOLVER: solver-INTERNAL N floor only; gameplay N (suffocation,
    // combustion) reads the real unfloored field. §3.4 overflow budget.
    float N_FLOOR_SOLVER = 1e-3f;
    // T_AMB_K: T is ambient-RELATIVE Kelvin; T_abs = T + T_AMB_K for the EOS.
    float T_AMB_K = 290.0f;
    // C: p* = C * N_total * T_abs. Calibrated (P1 §2.1) so ambient P == 1.0
    // at the Q16.16 level; config default mirrors [physics.eos] C = 1/t_amb_k.
    float C = 1.0f / 290.0f;
    // gamma: ideal-gas adiabatic index for the compression-work term
    // T -= (gamma-1)*T*div(u)*dt_s. 1.4 == diatomic (O2/N2 air) — a
    // reasonable engineering default; TUNING DIAL (not measured/gated here).
    float adiabatic_index = 1.4f;
    // absorb_strength: global scale on the per-cell dyn_wave_absorb damping
    // applied to u in the correction step (D4 — unit/material shockwave
    // shielding). Mirrors AtmosphereSolver::absorb_strength.
    float absorb_strength = 8.0f;
    // T_MIN: floor on the RELATIVE T field (T_abs >= ~1 K). Every floor hit
    // increments `energy_floor_hits` (the named 4th energy sink, D3).
    float T_MIN = -289.0f;

    // --- debug telemetry -----------------------------------------------
    mutable int64_t energy_floor_hits = 0;

    // --- six sub-kernel digest checkpoints (§3.4.6) ---------------------
    // A cheap FNV-1a-style running hash over the named buffer's post-stage
    // state, recomputed every step() call. Read via the accessors below.
    mutable uint64_t digest_advect      = 0;   // after 1a/1b (u,T self/SL advect)
    mutable uint64_t digest_bulk_flux   = 0;   // after 1d (donor-cell O2/N2)
    mutable uint64_t digest_pstar       = 0;   // after step 2 (p* materialization)
    mutable uint64_t digest_helmholtz   = 0;   // after step 3 (the RB-GS solve)
    mutable uint64_t digest_velocity    = 0;   // after step 4 (velocity correct)
    mutable uint64_t digest_compression = 0;   // after 1e, LAST substep (compression work)

    // One full Kwatra tick, per §3.2 above.
    //
    //   atmosphere        : Q16.16 int32 (h,w) — P. Read (last tick's P, for
    //                        p* is NOT read here directly — p* comes from
    //                        N,T) and WRITTEN (materialized once, step 5).
    //   p_prev             : Q16.16 int32 (h,w) — the repurposed `wave_p`
    //                        buffer (design's aliasing precedent extended:
    //                        atmosphere->P, wave_p->P_prev, same pattern).
    //                        WRITTEN at step 0 with a copy of `atmosphere`
    //                        as it stood BEFORE this tick's solve.
    //   wind_x/wind_y      : Q16.16 int32 (h,w) — u. READ+WRITTEN (self-
    //                        advected in 1a, pressure-corrected in 4).
    //   temperature         : Q16.16 int32 (h,w) — T, ambient-RELATIVE.
    //                        READ+WRITTEN (advected 1b, compression-worked 1e).
    //   gas                 : Q16.16 int32 (n_gases,h,w) contiguous. The two
    //                        `gas_conservative`-flagged planes (O2/N2) are
    //                        donor-cell transported each substep (1d); every
    //                        other plane is untouched here (traces advect
    //                        once/tick, the CALLER's job — see file header).
    //   solid               : the physics obstacle mask (== is_wall == obstacles).
    //   is_vacuum           : true vacuum (Dirichlet P=0).
    //   dyn_permeability    : per-tick face permeability (gates flux + the
    //                        Helmholtz face coefficient coherently, per §3.4).
    //   dyn_wave_absorb     : per-cell shockwave absorption (D4).
    //   h, w, dt            : grid dims, full tick length (seconds).
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
    // Reused per-tick scratch (GPU-prep: no per-tick alloc; house pattern).
    mutable std::vector<int32_t> n_total_;      // Σ gas planes, post-substep
    mutable std::vector<int32_t> vx_src_, vy_src_, t_src_;   // SL backtrace snapshots
    mutable std::vector<int32_t> pstar_;        // p* RHS source term
    mutable std::vector<int32_t> dinv_;         // per-cell Helmholtz Dinv, rebuilt every tick
    mutable std::vector<int32_t> p_new_;        // the Helmholtz solve's working P
    mutable std::vector<int32_t> div_u_;        // divergence(u) scratch (compression work + RHS)
};
