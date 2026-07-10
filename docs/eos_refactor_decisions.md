# EOS refactor — decisions log & resume notes

> **Status:** living decisions record for the pressure-model refactor — adopt **rung B**
> (genuine compressible ideal gas, `P = C·N·T`). Companions: `eos_research_report.md` (the
> A/B/CFL evidence), `eos_refactor_interaction_map.md` (code-grounded interaction inventory).
> Decisions taken 2026-07-08 → 07-09 (Erik + Claude).
>
> **RESUME PROTOCOL (Erik's request):** next session, *quickly re-confirm the LOCKED decisions
> below* (he was a little distracted when they were made), then continue with the OPEN items.

## LOCKED / working decisions

1. **Adopt rung B** — genuine compressible ideal gas (Kwatra semi-implicit), `P = C·N·T`. Defer is
   off the table; A-vs-B resolved to **B**. Why: real ideal-gas fidelity (rung A's `N` is
   static/decorative — it advects everything *but* N, so `P ∝ T`; rung B advects a genuinely
   *conserved* `N`); **native breach-venting**; real acoustics (feeds sound-ML) + a chemistry
   substrate; determinism path credible (its hardest kernel is the already-proven spike0b GS
   Poisson); Q16.16 range fine; cost only ~1.3× rung A and comfortably realtime (18% of budget).
   *(Quick explicit re-confirm next session.)*
2. **Merge `wave_p` into one unified `P`.** It existed *only* as a numerical workaround — the IMEX
   split ran implicit-diffusion (bulk) and explicit-wave (acoustic) on two fields because those two
   PDEs fight on one field. A compressible solver produces bulk equilibration AND acoustic waves
   from ONE pressure, so the split is obsolete — **the merge is implied by choosing rung B** (rung A
   would keep it hybrid; rung B unifies). The transient-buffet(wave_p)/sustained-dome(atmosphere)
   distinction consumers rely on survives as the single field's *time evolution* (front passes,
   dome lingers) — more physical. **Work item:** rewrite `apply_wave_push` to read `grad(P)` +
   recalibrate `k_push`.
3. **Delete `sink_hop` + the geometric venting hacks.** Venting emerges natively from `−∇P` toward a
   true-vacuum (N=0) cell under rung B. *(Erik's blessing given.)* Also generalize breach→vacuum
   beyond today's edge-hull-only rule.
4. **O2 gates on real `N`**, not `atmosphere`-as-a-pressure-proxy. Combustion reads gas
   quantity/species density, not P.
5. **`P` is materialized once per tick** into a stored Q16.16 field — right after the (N,T) update
   and *before any consumer* — for determinism (all consumers see an identical P) + CUDA
   cleanliness. *(Claude's recommendation; Erik to confirm.)*
6. **Purify the water pressure-head float bridge.** Today it dequantizes `atmosphere+wave_p` → float
   → `×k_p` → requantize. No reason for float (k_p is a fractional coeff → a `mul_q16` by a quantized
   k_p). **Fold into the refactor** (the head term is rewritten to read the derived integer P anyway).
   No urgency (already deterministic via the quantize-pin), but do it — pure integer > pinned bridge.
7. **Temperature = ONE unified field, masked per-medium passes** *(OPEN-A LOCKED 2026-07-09)*. A single
   Q16.16-Kelvin `temperature` array covers gas + solid cells. Implementation:
   - **gas rules** (semi-Lagrangian advect + compression-work `−P∇·u` + combustion/radiation sources,
     NO decay-to-ambient) run on the **open-air mask**;
   - **solid rules** (existing convert/conduct-owned-below/cool) run on the **solid mask**;
   - **conduction is ONE whole-grid pass** (the existing `conductivity`-keyed stencil) — give air a small
     nonzero conductivity and it does air↔air, solid↔solid, AND the solid↔air interface exchange
     automatically. *That interface exchange is the primary energy sink for sealed rooms — free here,
     explicit code in a two-field design; that + Dalton keeping gas-T single is why one field won.*
   - **Energy exits physically** now (not the old phantom decay-to-0 + per-tick heat-clear): (1) hot gas
     **vents to vacuum** (N,T advect out a breach), (2) **conducts into the ship's thermal mass** (then
     hull cells radiate to space via the reinterpreted `cool_shift_vacuum`), interior solids conduct to
     the adjacent gas instead of decaying to a fake ambient.
   - **Kelvin locked** (fits Q16.16: fire ≤ ~9000 K « 32767; 1/65536 K precision). No rescale.
   - Numerical work item: gas heat deposit → `ΔT = ΔE/(N·c_v)` is a **÷ by dynamic per-tile N** — a
     per-tile per-tick fixed-point reciprocal (the *proven* spike0b GS-reciprocal class), not the solid's
     free bit-shift.
   *(Bonus, still true: `temperature_design_proposal.md` §5 rejected an air-temp field as "one consumer,
   not worth it" — the EOS dissolves that; gas T co-derives P.)*
8. **Everything involved is ALREADY Q16.16 fixed-point** (gas, atmosphere, wave, heat, temperature,
   water, fire). The refactor **rearranges** fixed-point fields (N+T→P), it does not convert
   float→fixed. Big de-risk. (A few float bridges remain at boundaries — item 6 is one; purify where
   the refactor touches them.)
9. **Process = the design-gate practice.** design doc → independent adversarial critique (distinct
   lenses) → resolve blockers on paper (iterate v2/v3) → patch plan → autonomous build (**CPU + CUDA
   in lockstep** — both wave & diffusion already have bit-identical GPU mirrors, so it's a double
   implementation) with the determinism/bit-identity gates. **Nothing is built until the doc survives
   critique.**

## Peripheral decisions locked (2026-07-09)

- **2.5D z-levels DEFERRED** — `128×128×Z` (Z=2–4) is the natural next phase after the flat-2D EOS
  lands (or never — "we'll see"). It's the clean home for z-buoyancy AND it retires the permeability
  fudge (furniture solid at low z, open above → smoke pours over). Out of scope for the first patch.
- **Molar mass DROPPED (for now)** — with no z-layer there's no buoyancy for it to drive, and per
  Avogadro every gas contributes equally per particle to pressure. Returns only *with* 2.5D.
- **Multi-gas = Dalton sum + threshold chemistry** — `P = C·T·Σ N_i` over the existing `(N_i,h,w)`
  Q16.16 gas fields; chemistry (e.g. two gases mix → react) from per-gas density + T thresholds.
  **No lit search** (game-adequate falls out; only realistic multi-species diffusion / combustion
  kinetics would warrant one — not now).
- **Units:** must **block/absorb shockwaves** (teammate shielding — a kept gameplay requirement → units
  are partial obstacles/absorbers in the P solver, the wave-absorb mechanic carries over); must **NOT
  block water** (kept — Erik's call); **gas-permeability is droppable** (the smoke-seeps-past fudge, no
  longer needed).

## v2 design calls (LOCKED 2026-07-09, Erik nodded all three — Fable session)

10. **True-Kwatra pressure evolution** — the Helmholtz RHS carries the advected absolute
    pressure `p* = C·N·T` (not a divergence-only correction). Native venting/expansion/
    water-push fall out; `div_target`, `K_EXPAND`, `W_DISPLACE_GAIN` deleted. The float
    prototype demotes to a shape reference (its projection-form numerics do NOT carry over).
    Root cause of round-1 blocker B1: the prototype implemented a projection scheme, not Kwatra.
11. **Bulk O2/N2 transport = donor-cell conservative flux** (the shipped water-solver
    pattern + outflow limiter, CUDA precedent `cuda_water.cu`). Global mass-renormalization
    deleted. Flux form IS continuity ⇒ dilation (B2) included; sealed rooms airtight by
    construction (D1 dead).
12. **Combustion products conserve N_total** — non-soot fraction of consumed O2 credited to
    `inert_N2` ("burnt products"), **and (v2.1) trace decay likewise returns its mass to
    `inert_N2`** — conservation holds through the full burn-then-decay cycle. Sealed-room
    pressure never fake-drains (D2 fully dead).

**Round-2 critique verdict (2026-07-09, two targeted critics):** 14/16 round-1 findings
FIXED outright, 2 PARTIALLY — both closed by the same-day **v2.1** pass (occupancy-transition
mass rule; complete WRITER migration in P3 incl. W3→evacuation, plume→T shim, destroy_wall
N-seeding; decay→N2; permeability scales k_f; ρ̂:=N_total; amortized wide divide;
CFL_ADV≤0.5; napkin-cost-model + Kwatra-verification as P3 design-gate deliverables).
**The design doc (v2.1) has survived critique. Build may proceed per the §8 patch graph —
P1 and P2 are additive and parallel-safe; P3 requires its own design-gate first.**

Design doc v2: `docs/eos_refactor_design.md` (supersedes v1 in place; v1→v2 changelog at top
maps every round-1 blocker/decision to its fix). Round-1 critique: `..._design_critique.md`.

13. **Perf gate stays p99 ≤ 25% of the tick (20.75 ms @160², CPU reference path) — but
    renegotiation is EXPLICITLY on the table if needed** (Erik, 2026-07-10). Two graceful
    fallback levers exist before any physics compromise: raising the gate percentage, and
    the ambient-c dial (`K`). Note: GPU improves this in two stages — P6 kernel ports
    (modest at small grids; transfer-bound) and S8 residency/CUDA-graphs (the real
    multiplier, deliberately scheduled before big training runs). The gate's rationale
    (shared GPU w/ render+NN, training throughput ∝ 1/tick-cost, bigger-map headroom)
    survives both stages.

14. **P3 as-built amendments BLESSED (Erik, 2026-07-10):** variational/Galerkin MG
    transfers (the spec's re-discretized pick was measurably divergent — deviation
    adopted wholesale); `trace_mass_scale=0.02` (traces are opacity, not molar density);
    engine-owned trace advection in physical units (`advection_rate` config dead);
    **N_SUB_MAX re-pinned to 8** + targeted substep micro-opt (Erik chose this over gate
    renegotiation). Key clarifications recorded: N conservation is EXACT (zero drift,
    LSB-level, forever); the 0.0066 atm figure is a stationary solver band on the
    per-tick-DERIVED P, not accumulating error; perf numbers are the C++ engine, not
    Python.

15. **P3 MERGED TO MAIN (`b7b114c`, 2026-07-10).** Perf gate RESTATED intent-preservingly
    (Erik): **steady-state p97 ≤ 25% (passes: 21.6%) + event-tick max ≤ 50% (passes: 34%)**
    — p99 conflated event frequency with code speed (5 detonations/300 ticks land AT p99
    by construction). Golden `493645d3`. Suite 619 green.

16. **P4 MERGED (`396323d`, 2026-07-10 evening, auto-merge-on-green — no HUMAN-TEST marker
    on P4).** Combustion on real O2: four emergent payoffs green as E2Es; conservation
    bit-exact through burn+decay; trace `decay` column finally APPLIED (mass→inert_N2);
    fire/ignition O2 gates read `gas[O2]` (thresholds rescaled to the 0.21-ambient scale);
    Feldman/O'Brien/Arikan credited in `combustion.cpp`. Suite 627 green, golden
    `7eeb41d4`, perf p97 17.17 ms ✓. Constants are `[physics.combustion]` dials for P5.
    **Flagged for P5's feel pass:** a strongly-seeded sealed-room fire can drive T near
    the Q16.16 ceiling — PRE-EXISTING P3 coupling (plume→T shim × compression work),
    reproduced with combustion off; documented in
    `test_thermal_spike_is_pre_existing_not_a_p4_regression`.

17. **P5 HUMAN-TEST PASSED — v2.4 stack BLESSED, P5.1 launched (Erik, 2026-07-11).**
    Erik reviewed the final re-bake (`tools/eos_p5_out/index.html`, main @ `f646056`)
    and **blessed all four provisional v2.4 items as shipped**: the T_MAX_PHYS/U_MAX
    counted rails, the absorption-∝-density radiant deposit, the O2-gate hot-zone
    rescale (0.01/0.03/0.01), and the T_FLAME_MAX≈2000 shim limiter. His verdict on the
    bake itself: B3's native breach-vent is the arc's money shot, everything green.
    Two outcomes from his review:
    - **P5.1 — stoichiometric fuel consumption** (design §5 v2.5 amendment): combustion
      consumes `wall_hp` at ember scale (`fuel_per_o2` dial, default 0.7), **clamped at
      1 LSB — smolder never destroys** (Erik: charred walls survive as "easy prey for
      almost anything else — more emergent cool stuff"); the no-fuel gate moves to
      `hp ≤ 1 LSB` so charred embers extinguish. Completes Erik's original fire-lifecycle
      vision (die → ember → wind re-ignites → burn out); the ember state is EMERGENT
      (I=0 ∧ T≥ign ∧ hp>floor), no new state machine. Trio timings will move by design;
      golden re-baselines once.
    - **B2's undamped Helmholtz ring flagged** (Erik's eye: rooms shouldn't oscillate
      forever). Diagnosis: real two-room Helmholtz mode (measured 15-tick period,
      rock-steady, from the B2 trace), unphysically undamped because the momentum update
      has no viscosity/drag and walls are rigid. Remedy identified but NOT launched: a
      `k_drag` per-tick velocity decay dial (`u *= 1−k_drag`, one deterministic Q16.16
      multiply, ~0.02–0.05) — queued as a §9 feel-pass item (it's the §D "artificial
      acoustic damping for feel" item, now with a measured motivation).
    - Related vision note for later arcs: temperature→blackbody-glow rendering is now
      UNBLOCKED by the unified-T substrate (the parked smoke/blackbody planning-day
      topic); glow would visually mark re-ignition sites since ignition ≡ fuel∧O2∧T.

## RESUME POINT (Erik paused 2026-07-10, resuming tonight, same PC)

**Where we are:** P1+P2+P3 all merged — the compressible EOS engine is LIVE on main
(native venting, conserved mass, unified temperature, deterministic, suite green).
**Next in order:** (1) **P4 — combustion on real O2** (design §5: own pass, conservative
products incl. trace-decay→N2, both o2_thresholds, the four emergent payoffs as its gate);
(2) **P5 — combined bake-off HUMAN-TEST** (Erik's eyes; k_push/k_p recalibration + the
trace-advection/wind_diffusion_scale feel items parked there); (3) **GPU end-state
alignment review** (scheduled, see OPEN below) BEFORE (4) P6 CUDA ports (incl. MG kernels
+ retirement of cuda_wave/cuda_atmosphere); (5) P7 cleanup + canon fold-back (+ the
post-EOS doc consolidation from the roadmap). The P3 worktree/branch can be cleaned up
(merged). CUDA gates remain suspended via EOS_P6_PENDING until P6.

## OPEN — to decide next

- **GPU END-STATE ALIGNMENT REVIEW — after P3 lands, BEFORE P6 starts** (Erik, 2026-07-10:
  "the C++/CPU step is intermediate; step back and check alignment with the end goal").
  Audit every P3-introduced primitive against GPU residency (S8) + batched RL training:
  MG hierarchy coarse-tail strategy (truncation depth; CUDA-graphs launch collapse;
  batching-across-envs makes coarse levels large again), smoother choice (RB-GS vs a
  Chebyshev–Jacobi swap — the most GPU-native variant, contained change), wide-int64 ops
  on device, transfer stencils, per-cell reciprocals. Context: the V-cycle's only real GPU
  awkwardness is the serialized tiny-level launch tail (~0.5 ms/tick naive, ~nil under
  CUDA graphs; batched training cures it structurally); CPU path is PERMANENT as the
  bit-identity reference, not throwaway.

- **B — "what is N" largely RESOLVED (2026-07-09, Erik keen).** The bulk air becomes **two explicit
  species: O2 + inert-N2**, with the traces (smoke/poison/…) on top, and `N = Σ(all species)` (Dalton,
  `P = C·T·N`). **O2 is tracked as its own gas** — Claude's feasibility call: **very doable and cheap**
  (just another slice in the existing `(N_species,h,w)` Q16.16 gas array; rides the proven per-gas
  fixed-point transport; ≈ +1 bulk field vs today's single `atmosphere`). It's the `O2-gates-on-N`
  decision made real, and it makes a pile of gameplay **emergent, not scripted**: fire self-extinguishes
  as it eats local O2; breach → O2 vents → fire dies; **O2-tank rupture → O2-rich pocket → fireball**;
  inert/CO2 flood smothers fire + suffocates; units suffocate in low-O2. Strongly endorsed.
  Still open (N-session, no lit search): **combustion stoichiometry** (O2 consumed → smoke/heat yield)
  + suffocation tuning — balancing, not feasibility.
- **C. Remaining `§D` items** in the interaction map (P-materialization contract details; whether to add
  an artificial acoustic-damping-for-feel term; recalibrating `k_push`/`k_p`; etc.).

## Resume plan (next session)

1. **Quickly re-confirm LOCKED 1–9** (Erik wanted a brief review — he was distracted).
2. Walk **OPEN-A** (temperature details), then schedule **OPEN-B** (the N / multi-gas dedicated session).
3. Then: full **design doc → adversarial critique → resolve → patch plan → CPU+CUDA build**.
4. Housekeeping: consolidate EOS design docs onto the refactor branch when cut; **delete dead
   `wave_solver.*`**; fix the 6 stale doc spots (interaction map §0).
