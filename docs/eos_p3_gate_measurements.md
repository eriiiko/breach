# EOS P3 gate measurements — stability matrix + double-count fix verification

> Executed 2026-07-10 (P3 build, branch `eos-p3-solver`, post compression-work
> relocation per the corrected design §3.2 step 4c). Companion to
> `eos_p3_microbench_results.md` (the pre-build microbenches). Scenario grids
> are 16×16 hull-ringed rooms at tile_size_m = 1/3, dt = 1/24 s, driven through
> the REAL `Simulation.step()`; `amp` = max over interior of |P − 1 atm|.

## 1. The compression-work double-count fix — VERIFIED on its home turf

Hot-cell scenario (sealed room, +400 K patch on 2×2 cells, c=66, S=64 — the
stable solver regime): P transient peaks at **0.050 atm** on tick 0 and decays
monotonically to ~0.009 by tick 79; T cools smoothly 400→367 K via expansion
work; **zero** energy-floor / work-clamp / u-clamp hits. The pre-fix in-loop
compression work double-counted against the Helmholtz RHS's div(û*) term
(≈(2γ−1)-vs-γ over-response); the relocated once-per-tick post-correction form
(step 4c) shows no oscillatory response. The fix is correct and stays.

## 2. BUT the pressure instability SURVIVES the fix — and it is provably a
##    DIFFERENT mechanism

Decisive observation: in the water-displacement scenario **T ≡ 0 everywhere**
(no heat source), so the compression-work term — old placement or new — was
**identically zero** there. Its blow-up trace is bit-identical before and
after the fix. Two further mechanisms measured:

### 2a. Stability matrix (amplitude at ticks 5/10/20/40; `--` = already >1e3)

| scenario | S   | c_max | amp@t5    | amp@t10   | amp@t20   | amp@t40   | outcome |
|----------|-----|-------|-----------|-----------|-----------|-----------|---------|
| water    | 8   | 66    | 5.3e-01   | 2.2e+03   | --        | --        | BLEW t10 |
| water    | 16  | 66    | 3.2e-03   | 4.8e-02   | 1.3e+01   | --        | BLEW t25 |
| water    | 64  | 66    | 8.1e-04   | 4.4e-04   | 2.6e-03   | 2.6e-03   | stable |
| water    | 512 | 66    | 6.0e-04   | 6.7e-04   | 4.5e-03   | 7.6e-03   | stable |
| water    | 8   | 300   | --        | --        | --        | --        | BLEW t4 |
| water    | 16  | 300   | 3.3e+04   | --        | --        | --        | BLEW t5 |
| water    | 64  | 300   | 2.5e-03   | 6.0e-02   | --        | --        | BLEW t13 |
| water    | 128 | 300   | (held 80 ticks, final 3.6e-03)          | marginal-stable |
| water    | 256 | 300   | (held 80 ticks, final 9.8e-02)          | marginal |
| water    | 512 | 300   | 4.5e-03   | 4.4e-03   | 3.8e-03   | 2.0e-02   | slow creep to ~8e-2 @150t |
| vent     | 8   | 66    | 4.3e+00   | --        | --        | --        | BLEW t8 |
| vent     | 16  | 66    | 5.1e-01   | 1.3e+00   | 3.6e+03   | --        | BLEW t20 |
| vent     | 64  | 66    | 5.8e-01   | 6.3e-01   | 6.1e-01   | 3.7e-01   | BLEW t51 |
| vent     | 512 | 66    | 6.4e-01   | 7.1e-01   | 7.1e-01   | 5.7e-01   | stable (settling) |
| vent     | 8   | 300   | --        | --        | --        | --        | BLEW t3 |
| vent     | 512 | 300   | --        | --        | --        | --        | BLEW t3 |

Reading: stability requires S ≥ ~64 at c=66 and S ≥ ~128 (marginal) at c=300
in the water case; the venting case at c=300 diverges in 3 ticks at ANY sweep
count. **The design's frozen S ≤ 16 is not achievable with point RB-GS at
either wave speed.** M2's per-sweep cost (0.238 ms × 1.5 wide factor) prices
S=128 at ~46 ms/tick — the whole-tick budget is 20.75 ms. Point-GS cannot
converge an operator whose face coupling is c²dt²/dx² ≈ 1409 (c=300; 68 at
c=66) in a fixed low sweep count: the identity term no longer dominates, so
the solve is effectively a Poisson problem needing global information
propagation (~grid-diameter sweeps). §3.4's "diagonal dominance ⇒ the
fixed-sweep GS guarantee is intact" conflates GS *asymptotic convergence*
(true) with *convergence in 8 sweeps* (false at this stiffness — the shipped
d_atm diffusion the 8-sweep habit came from has μ·w ≈ 8/face, 175× softer).
An under-converged solve leaves the grid-scale RHS component unbalanced, and
the scheme degenerates toward an EXPLICIT acoustic update at CFL c·dt/dx =
37.5 — amplification per tick up to c²dt²k² ≈ 1.4e4, matching the observed
~×700/tick growth.

### 2b. The vent@300 any-S divergence: a UNIT/IMPEDANCE inconsistency in the
###     spec transplant (a second spec-level issue, flagged for the design doc)

The paper's Helmholtz RHS is p_a − ρc²Δt∇·u* with everything SI: there
ρc² = γp, so the two RHS terms are the SAME order (~1e5 Pa each). Our
transplant sets p* on the 1.0-atm calibration but takes N·c² with c = 300 m/s
⇒ N·c² = 90,000 while p* ≈ 1 — the γ-ratio (ρc²/p = 1.4 in any consistent
unit system) is broken by ~64,000×. Consequences, both observed:

- a physically modest venting velocity (u ≈ 0.06 m/s at a fresh breach)
  produces an RHS excursion of N·c²·dt·div(u) ≈ −300 atm — the pressure
  response to divergence is ~4 orders too stiff relative to the pressure
  scale, so P swings past ±300 atm within 2 ticks even under an essentially
  exact solve (S=512), then saturates the Q16.16 field ceiling (±32768);
- symmetrically the momentum kick u −= dt·∇P/N̂ is missing the
  P_unit/ρ_unit conversion (≈ c²/(γP) ≈ 84,437 at ambient in SI), so
  velocities are far too small for the pressure field they ride — until the
  huge P swings drive them to the c_max clamp anyway.

Consistent-unit form (proposed for the design doc, needs Erik/orchestrator
sign-off): the pressure-evolution/RHS/LHS coefficient is (ρc²) expressed in
the SAME atm units as p*, i.e. **γ·p\*** per cell (≈1.4 at ambient — and
per-cell, the paper's own (ρc²)ⁿ⁺¹ evaluation); the momentum correction gains
the constant unit factor κ_u = c²/(γ·P_amb/N_amb) ≈ 84,437·(dx-units), giving
real m/s velocities. NOTE: the face-coefficient PRODUCT is unchanged
(κ_p·κ_u = c² — the §3.4 overflow budget still holds), so **this fix does NOT
by itself cure 2a's sweep-count problem**; it cures the amplitude blow-through
(P stays ~1-atm-scaled, the Q16.16 ceiling and the GS flux-narrow overflow
stop being reachable in ordinary scenarios).

### 2c. A third, smaller defect: the GS flux-narrow int32 ceiling

`flux = narrow(Σ mul_wide(face_k, ΔP))` wraps when the REAL value k_f·ΔP
exceeds Q16.16's ±32768: at c=300 (k_f ≈ 1409·perm·ratio) any neighbor P
difference beyond ~23 atm corrupts the sweep arithmetic and turns a marginal
oscillation into an explosion. The formal overflow-stress-sweep gate would
flag exactly this line. Fix trivially available (keep the accumulator wide
through the resi·Dinv product), but pointless until 2a/2b decide the
operator's final scaling.

## 3. Digest determinism gate (stable regime)

Water scenario, c=66, S=64, two identical 40-tick runs: all six sub-kernel
digests (advect / bulk-flux / p* / Helmholtz / velocity / compression)
bit-identical at every tick. PASS (CPU lockstep).

## 4. Where this leaves the P3 gates

BLOCKED on a design decision (fixed-sweep point-GS cannot meet the pinned
S ≤ 16 at c = 300; candidate resolutions: multigrid/FFT-class solver [new
design work, contradicts the frozen-S pin], a much lower c_max [contradicts
the c=300 pin], or accepting a large S with its cost [contradicts the perf
budget]). The venting/water E2Es, S-pinning, stress probes, perf bench,
behavioral-parity bakes, and the golden re-baseline are all downstream of
that decision and were NOT forced.


---

# ADDENDUM — v2.2 (D-A + D-B + D-C) build and the MG measurement gate
> 2026-07-10, same branch, after the design v2.2/v2.2-final merge.

## A. D-A verified
The γ·p* coefficient + wide-K kick + state-derived c_LOCAL eliminated the
3-tick unit-driven divergence in every scenario (venting included, any S).

## B. D-B: the MG gate's own findings (two corrections beyond the spec)
1. **The spec'd nonsymmetric operator + averaged coarse ops + bilinear
   prolongation is DIVERGENT on deep pyramids** (measured: error ×7/cycle at
   a breach; more cycles = worse — an amplifying, non-variational coarse
   correction). Adopted: symmetric (mass + face-Laplacian) row form ×
   exactly-variational PC-Galerkin transfers (masses/conductances/residuals
   SUM; PC-injection prolongation). FLAGGED spec deviation.
2. **Galerkin Dirichlet anchor:** straddler coarse cells must fold their
   regular-child→vacuum face conductances into the coarse DIAGONAL; without
   it the vent case amplifies at any cycle count. With it, the dedicated
   breach-adjacent-to-coarse-boundary test shows NO odd-vs-even alignment
   degradation (0.229 vs 0.237 atm one-tick error @C=2).
3. Single-tick convergence (16² vent, vs a C=64 deep reference):
   C=1: 0.44 → C=2: 0.24 → C=4: 0.084 atm (~×0.55/cycle).
4. **Warm start from the previous tick's solved P** buys ~2 cycles: the
   durably-stable schedule drops from V(2,2)×C=4 (cold start; C=3 was
   UNSTABLE at 19.9 atm worst-dev) to **V(2,2)×C=2 — FROZEN**, full pyramid
   (mg_min_dim=1; the room-bulk/DC mode is solved exactly at 1×1),
   coarsest = 32 sweeps. 300-tick durability at the frozen schedule:
   water worst-dev 0.0066 atm; vent overshoot 0.0005; u_clamp 0 (sealed) /
   ~3-per-tick breach-adjacent (physical choked flow); V(1,1) and C=1
   measured too marginal.

## C. Two further unit-calibration seams (found by the suite, fixed, FLAGGED)
1. **trace_mass_scale = 0.02** (new [physics.eos]-class constant): traces
   are [0,1] OPACITY tracers, not molar densities — an unweighted Dalton sum
   made a 0.6-teargas cloud a +60% pressure bomb that blast-scattered itself
   in one tick (measured). 0.02 keeps §2.1's "bulk carries ~99%" premise
   true by calibration. NEEDS Erik's sign-off as a design amendment.
2. **Trace advection unit conversion** (engine-owned): the SL displacement is
   now u[m/s]·dt/dx tiles — the config advection_rate (900, old-wind-scale)
   is DEAD at P3 (a raw 900 gave 326-tile/tick displacements and ×5 mass
   duplication); wind_diffusion_scale (50, old-wind-units²) is DISABLED
   pending P5 recalibration (it would explode the un-substepped forward-Euler
   diffusion at m/s wind scales).

## D. Perf gate (M1 160², 300 ticks, real Simulation.step, frozen schedule)
| config                     | p50   | p99   | max   | gate ≤20.75 |
|----------------------------|-------|-------|-------|-------------|
| N_SUB_MAX=16 (pinned)      | 25.0  | 33.4  | 36.1  | FAIL |
| N_SUB_MAX=8                | 15.8  | 24.8  | 32.1  | FAIL |
| N_SUB_MAX=4                | 13.1  | 22.0  | 24.9  | FAIL (6% over) |

Substep caps 8 and 4 are MEASURED as stable as 16 (both E2Es, 300 ticks:
worst-dev 0.010/0.0006 at cap 4). The cost driver is SUSTAINED sonic venting
(u = c at the breach ⇒ the u_est cliff pins n_sub at the cap for the whole
post-breach regime — not the "wildest 1-2 blast ticks" the cap was priced
for). Already done: fused 3-field SL march (one DDA+bilinear serves vx/vy/T;
41.6→24.6 ms venting median), zero-displacement + all-open-corner fast
paths. The remaining gap needs an N_SUB_MAX re-pin (Erik's constant) and/or
~2× micro-optimization of the substep+solve inner loops — REPORTED, not
silently retuned. Solver-only split at 160² venting: substeps(16) ≈ 16.5 ms,
MG C=2 ≈ 3.2 ms, fixed overhead ≈ 4 ms.

## E. Remaining gates at the frozen config — all PASS
- Digest determinism: two identical 40-tick runs, all six digests
  bit-identical every tick.
- Hot-cell thermal: peak amp 0.017 atm, clean decay, zero clamp/floor hits.
- 9-grenade stack (48²): held 120 ticks, peak tick 18.5 ms; the sealed room
  ends over-pressured (~2.1 atm max) from the deposited bulk N — physical.
- O2-tank rupture (200×N + 2000 K): held, final dev 0.024 atm, 7 work-clamp
  hits, no overflow.
- Full suite: 619 passed / 5 skipped (2 = the P5 k_push-recalibration skips,
  attributed) / 10 cuda-deselected. Golden re-baselined ONCE at patch end:
  aggregate 2bab9702 → f7b8becd (perfield baseline regenerated + 8 check
  scripts + the perfield tool updated in the same commit).

## F. Behavior deltas for the merge review (feel/tuning class, P5)
- apply_wave_push: grad(P) transients no longer reach the old knockdown
  radii at shipped k_push (no knockdown at d=7; the two calibration tests
  are skip-attributed pending P5).
- Pressure steps now drive PHYSICAL wind speeds (a 0.1-atm step ≈
  hurricane-scale; sonic at a breach). Old scenes that painted sustained
  pressure imbalances get violent (correct) responses.
- A sealed burning room ends over-pressured (bulk-N deposits are conserved
  mass now) — the fire/explosion pressure economy is real.
- Fire wind-coupling dials (k_wind_fan/strip) read m/s magnitudes now —
  P5 recalibration listed.


---

# FINAL LAP — cap re-pin (16→8, blessed), bit-identical micro-opts, final perf
> 2026-07-10, decisions log #14 executed.

## Micro-opt pass (ALL verified BIT-IDENTITY-PRESERVING at fixed cap:
## the six-digest trajectory over 40-tick water+vent runs is byte-identical
## before/after, plus the standing two-run determinism check passes)
- max|u|: one isqrt of the max radicand (monotone ⇒ max∘sqrt = sqrt∘max).
- CFL ∇P pass and the momentum kick skip the per-cell Newton reciprocal at
  exactly-zero gradient (du ≡ 0 there).
- Donor-cell face coefficients (min-perm quantize × dt_s — constant within a
  tick) hoisted to a per-tick cache; `bulk_flux_transport_cached` entry added
  (same per-face arithmetic, evaluated once instead of per plane × substep),
  with reused thread_local scratch; the legacy entry forwards through the
  same hoist (pybind/P1-test path unchanged in behavior).
- The SL sample's corner/march predicates re-expressed as a per-tick
  sealed/breach/live byte table (same classification the original
  float/bool chain produced).
- absorb·dt quantize hoisted out of the per-cell kick loop.
- The dead single-field backtrace sampler deleted (the fused version had
  replaced every call site).

## Final perf (M1 160², 300 ticks, real Simulation.step, shipped defaults:
## N_SUB_MAX=8, MG V(2,2)×C=2 warm-started)
p50 = 15.0 ms   p99 = 22.5–24.0 ms (run-to-run OS noise band)   max = 28.3 ms
**GATE (p99 ≤ 20.75): FAIL — but with a sharp shape:** p97 = 17.97 ms PASSES;
the p99 tail is exactly the five explosion-DETONATION ticks (22.4–26.0 ms
each: the FieldEdit disc flush + the blast's first solver response +
n_sub spiking to the cap). Steady-state (quiet, venting, water, smoke) all
sit ≤ ~18 ms. Per the stop-band instruction this is REPORTED for Erik's
renegotiation lever (decisions #13 names two graceful levers), not pushed
further here. Candidate next steps if the gate must hold as-written:
amortize the event-tick response (the detonation tick is also the baseline's
own worst tick), or the ambient-c/K dial.

## Final E2E durability at the SHIPPED configuration (300 ticks)
- water: worst-dev 0.0058 atm, N exactly conserved (196.0), zero
  u-clamp/work-clamp/floor hits.
- vent: worst overshoot 0.0004 atm, N drains 196→2.9 and flow stops;
  u_clamp 186 (breach-adjacent choked flow, <1/tick avg), work_clamp 12,384
  (the venting expansion-cooling rail at breach cells — expected), floors 0.
- Two-run six-digest determinism: PASS.

## Final golden
Aggregate `493645d3…` (the 16→8 re-pin legitimately moved trajectories from
the same-day `f7b8becd`; history annotated in the perfield tool). Committed
perfield baseline regenerated + reproducibility re-run matches; the 8
cuda_s*_check scripts + the perfield tool updated in the same commit.
Suite: 619 passed / 5 skipped / 10 cuda-deselected — GREEN.
