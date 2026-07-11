# EOS refactor design doc — adversarial critique (round 1)

> **Status:** design-gate step 2 output. Four *independent* critics (physics/numerics, determinism/
> fixed-point, integration/regression, performance/patch-plan), each code-grounded, each forbidden from
> fanning out. Synthesized 2026-07-09. Reviews `docs/eos_refactor_design.md` (the v1 draft). **The
> flow STOPPED here per Erik's scope — draft + one critique, no build.**
>
> **Verdict:** the draft is a *strong* first pass — the architecture is sound and the critics
> independently validated the load-bearing choices (see §Sound). But they found **7 genuine blockers**
> that make the doc **not yet build-ready**: two are *physics* gaps that touch rung B's core promises,
> two are *fixed-point* correctness cliffs, two are *patch-sequencing* errors, one is a *cost-reality*
> gap. All are fixable on paper — this is exactly what the gate is for. **Recommend a design-doc v2
> addressing the blockers, then a second critique pass before any build.**

## BLOCKERS (must fix before building)

**B1 — "native venting" is NOT delivered by the design as written. [physics] — and it touches the
core reason we chose rung B.** The Helmholtz RHS (`rhs = dt·c²·(div(u*) − div_target)`) has sources
only from *existing* velocity divergence + the `div_target` heuristic (thermal expansion + water). **There
is no term that responds to a static N/P gradient.** A quiescent room breached at a corner → `rhs=0`,
`p≡0` is the exact solution → *zero* induced flow. So "delete `sink_hop`, venting is native" fails for
pure decompression — the prototype already found this for thermal expansion but never generalized the
fix. Compounding it: `wind = −∇P` (raw gradient, drives smoke) and the solver's momentum-corrected
`(vx,vy)` (transports the gameplay-critical `N_O2`) are *different quantities* — so **smoke could stream
out of a breach while `N_O2` stays put**, breaking suffocation *and* splitting cosmetic from physical.
→ *v2 must add an explicit `−∇P/N` forcing term and reconcile `wind` vs `(vx,vy)`.*

**B2 — `N` has no continuity/dilation term; "genuinely compressible" is half-true. [physics]** Continuity
says `DN/Dt = −N·∇·u` — a converging flow should raise *density*, not only temperature. But `N` is
advected as a plain passive scalar (`Dφ/Dt=0`); only `T` gets the adiabatic update. So a blast/O2-tank
shows compression purely as a T-spike, and `N`'s own compressibility — the defining feature of rung B —
is structurally absent. → *v2 must add `N −= N·div(u)·dt`, or explicitly document N as an incompressible
tracer riding a compressible T/velocity field (a real, nameable simplification, not a silent one).*

**B3 — the Helmholtz coefficient `c²/N` overflows Q16.16 on the breach-venting path. [determinism]**
As `N→0` at a breach, `coef = c_max²/N` reaches ~1.4e7 with the prototype's constants — **~450× past
Q16.16's ~32768 ceiling.** This is overflow-to-garbage, on the headline gameplay path. And the "reuse
the proven spike0b GS-reciprocal" claim **does not transfer** — the existing atmosphere divisor is a
*bounded small-integer* set; this one is *per-cell and unbounded*. *(This qualifies "Q16.16 range is
fine" — true everywhere except here.)* → *Patch-2's design-gate must derive a wider intermediate
(int64 through the sweep, narrow only at the final quotient) + a Helmholtz-specific N-floor.*

**B4 — the bit-identity gate would show GREEN on the B3 overflow. [determinism]** The failure is
*overflow* (a correctness cliff), not float cancellation — and an overflowed integer is still
*bit-identical* CPU↔CUDA, so the digest gate passes while the physics is silently wrong on both
platforms (the worst failure mode for this project). → *Patch-2's gate needs an explicit
overflow/saturation stress sweep (drive N to floor across a breach), not just accuracy-vs-double.*

**B5 — Patch 2 breaks the game for ~3 patches. [integration + performance — found independently by two
critics].** Patch 2 retires `wave_p`/`atmosphere` and deletes `wave_solver.*`, but their consumers —
`apply_wave_push` (unit knockback), the water head term, the ripple splash, `recorder.py`'s field list +
blowup trigger, and `test_wave_absorption.py` — aren't migrated until Patch 5. With no compatibility
alias, the game won't run, the recorder crashes tick 1, and tests fail for the entire 2→5 window,
violating the "each patch independently runnable/testable" rule. → *v2: keep `wave_p`/`atmosphere` as
live aliases onto `P` through patch 4, OR fold the consumer rewiring into patch 2.*

**B6 — Patch 2 is not independent of Patch 3. [integration]** Patch 2's solver needs a gas-`T` field
(materialize P from N,T; the compression-work update) — but gas has no temperature field until Patch 3.
§8's "run 2 and 3 concurrently on separate worktrees" is therefore unsafe. → *v2: fold a minimal gas-T
stub into patch 2, or merge/reorder 2+3.*

**B7 — the "comfortably realtime ~1.3×" cost claim isn't evidenced for the real architecture.
[performance].** (a) The bakelog's own worst case is already **S5 max 71 ms of the 83 ms budget (85%)** —
bare solver, Python, *no* combustion/temperature/species/CUDA. The "18%" was a cross-scenario *mean*.
(b) The "1.3× rung A" was measured advecting **one** scalar N, not **seven** Q16.16 species inside the
substep loop — and rung B collapses four independently-tuned substep cadences (wave/diffusion/smoke/
sink) into one `|u|`-driven count applied to all of them. (c) The Helmholtz solve runs **40 sweeps vs
the shipped kernel's 8** — 5× the cost it claims to "just reuse," unbudgeted. → *v2: model multi-species
substep cost before Patch 1 locks the transport contract; report p99/max not mean; re-derive/justify the
sweep count; give Patch 2 an explicit ms/tick acceptance gate.*

## DECISIONS NEEDED (design-level, for Erik + v2)

- **D1 — the O2/N2 conservation scheme needs real design** *(raised by all four lenses)*: global
  mass-renormalization can **leak density into sealed, *unconnected* rooms** (breaks the airtight/
  suffocation invariant); its divide is a grid-wide int64 sum that **doesn't fit `reciprocal_q16`**
  (needs a Door-3 pinned double-divide); it's **undefined if a room fully vents** (`ΣN_after→0`); and its
  **cadence is ambiguous** (once/tick vs inside the ≤64 substep loop — affects both cost and conservation
  drift). Likely answer: per-connected-component (or local-flux) conservation + a Door-3 divide.
- **D2 — combustion is a net `N_total` sink** *(physics)*: it destroys `N_O2` but credits only decaying
  `black_smoke` at `soot_yield<1` (CO2/H2O unmodeled) → a sealed room that burns shows *permanently lower*
  baseline pressure from bookkeeping, on top of the intended O2-depletion. → make the product conservative
  (route mass to inert-N2) or accept + document.
- **D3 — compression-work stability + a hidden 4th energy sink** *(physics)*: `T −= (γ−1)T·div(u*)·dt`
  is forward-Euler, unstable if `(γ−1)|div u|dt→1` (exactly at breach fronts); the prototype's `T=1K`
  floor silently vanishes energy outside the three declared exits. → substep the term, and name the
  floor as accounted (or remove the need).
- **D4 — unit shockwave-shielding (`wave_absorb`) has no slot in Kwatra's split** *(integration)*:
  today's per-cell acoustic-energy-removal doesn't obviously map onto self-advection→Helmholtz→T. It's
  gameplay-critical (teammate shielding) with a regression test — needs concrete placement, not assertion.
- **D5 — substep count `n=ceil(dt/dt_adv)` is the historical 1-ULP-desync bug class** *(determinism)*:
  must go through the proven `smoke_cliff_count` integer-ceil discipline, and `max|u|` must use
  `sqrt_q16`/`sqrt_q16_dev`, not native `sqrtf`.
- **D6 — the ripple-splash raw-`wave_p` reader is uninventoried** *(integration)*: a naive
  `wave_p→P` repoint makes every wet tile splash from the standing baseline every tick. Needs a
  delta-from-baseline decision.
- **D7 — pin the GPU backend to CPU for the patch-2→6 window** *(integration)*: backends are
  runtime-switchable; flipping wave/atmosphere to the stale GPU kernels mid-refactor crashes/desyncs.
- **D8 — add a combined-system bake-off gate** *(performance)*: the A-vs-B decision was gated on a GIF
  bake-off; the *assembled* system (temperature + real-O2 combustion + migrated consumers) needs an
  equivalent S1–S5 bake vs the pre-refactor baseline before commit — §8 currently has no such gate.
- **D9 — Patch 6's CUDA surface is under-scoped** *(performance)*: beyond the 3 named kernels it also
  needs a new velocity-advection kernel, a 7-species advection **+ mass-renorm global reduction** (a
  genuinely new CUDA primitive class for this codebase), and `cuda_wave.cu` retirement.

## MINORS
- `find_burst_walls` "solid contributes 0" needs a one-line confirmation of furniture's `N` state.
- O2-tank-rupture N-spike: range-check once §9 calibration lands.
- Combustion heat divisor `N_total` needs a floor independent of the TBD `o2_thresh`.
- No shock-capturing (fixed-sweep GS smooths discontinuities) — fine for a game, but *name* it as a
  fidelity limit so no one later expects Rankine–Hugoniot behavior.
- Patch 2 should carry *per-sub-kernel* bit-identity checkpoints (advect / div_target / Helmholtz /
  velocity-correct / T-update), not one end-of-tick digest that could hide a compensating-error pair.

## SOUND — independently validated (the reassuring column)
- `|u|`-only advection CFL decoupled from `c_max` is faithful Kwatra 2009 — not a novel risk.
- The Helmholtz operator is **strictly diagonally dominant** (identity + Laplacian), so fixed-sweep
  Gauss-Seidel convergence is *guaranteed* regardless of sweep count — the fixed-sweep discipline is safe.
- Dalton `P = C·N_total·T` is dimensionally/structurally fine.
- Once-per-tick stored-`P` materialization is the right contract (one writer, many readers → clean CUDA).
- Patch 1 (additive species plumbing, gated on legacy-species bit-identity) is well-scoped.
- Combustion's `ΔT=ΔE/(N·c_v)` reciprocal **is** safe to reuse (its `N` is bounded away from 0 by the
  ignition gate) — a clean contrast to B3's unbounded case.
- `reciprocal_q16`/`sqrt_q16` already have verbatim, S6/S7-proven CUDA device mirrors.
- Fixed sweeps confirmed everywhere — no adaptive/tolerance loop.
- Fire plume→feed-T and O2-gate→`N_O2` remaps are accurately characterized and low-risk.

## Recommended next step
A **design-doc v2** that: adds the `−∇P/N` venting force (B1) + the N-dilation term (B2); pins the
Helmholtz fixed-point representation + overflow gate (B3/B4); re-derives the patch graph so every patch
is runnable/testable (B5/B6); and models the real multi-species cost with hard ms/tick gates (B7) — then
resolves D1–D9. Then a second, shorter critique pass. **No build until v2 survives it.**
