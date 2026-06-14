# Resolution architecture — research findings & proposal

**Status:** process artifact (research night, 2026-06-14). Produced by an 8-agent workflow: 6 parallel
facet-research agents → synthesis → **adversarial critique**. The critique materially changed the
conclusion, so this doc presents the *reconciled* answer, not the first synthesis. Feeds the
PhysicsEngine-unification and CUDA planning.

---

## TL;DR — the adversarial pass changed the answer

**Three verdicts are solid and survive:**
1. **Keep Gauss–Seidel; do *not* adopt multigrid.** Our implicit operator is `(I − μΔ)` — a *screened/
   Helmholtz* problem, diagonally dominant, so GS already converges essentially resolution-independently.
   Multigrid's `O(N)` win is an argument about *pure* Poisson, where it doesn't bite here — and
   destructible walls are multigrid's documented worst case.
2. **FFT is off the table** — exactly because of destructible geometry (your memory was right).
3. **"Subcycle the wave only when active" is the immediate win** — ~10 lines, decision-free.

**The headline reversal:** *per-system* multi-resolution (a separate fine grid per field, with bespoke
inter-grid transfers) is **over-engineered**. The simpler, equivalent answer is **ONE uniform finer grid
(engine/01 §2's `base/k` knob — i.e. halve `tile_size_m`) with the wave kept coarse as a strided view.**
It delivers exactly your goal (fine atmosphere/smoke/light, coarse wave) with *none* of the five
hand-verified transfers, *no* second coordinate vocabulary, *no* canon violation, and it sidesteps the
fatal gap the per-system design hit (the fire step — below).

**The real cost center isn't resolution at all** — it's the **Q16.16 fixed-point migration** of the
atmosphere/wave/smoke solvers. That's a multi-week project and the true prerequisite for determinism
(and therefore CUDA). Resolution is, in the critic's words, "a small bow on top."

**And three concrete code findings worth acting on regardless of the resolution decision** (the
adversarial agent read the actual solvers):
- `stamp_units` is a **full-field Python rebuild every tick**, not the O(1) edit-seam the synthesis
  assumed — a farm-scale cost *now*, at k=1.
- **Smoke diffusion is explicit forward-Euler** (CFL-bound), *not* unconditionally stable — so refining
  smoke is **not** free.
- The **fire step reads/writes five fields at one index** — it dictates that those fields share a
  resolution (which "one grid" gives for free, and which the per-system design couldn't express).

---

## What's solid (survives the critique)

### Gauss–Seidel, not multigrid
The implicit atmosphere operator is `(I − μΔ)`, not `−Δ`. The identity term makes it diagonally dominant
(condition number `O(1 + 8μ)`), so the current **fixed 8 GS sweeps** are a cheap, stable equilibrium
smoother that's fine at this size and *doesn't* degrade with resolution the way pure-Poisson GS would.
Multigrid is rejected for three Breach-specific reasons: (1) wrong operator for the `O(N)` argument; (2)
destructible thin walls hit multigrid's "small-islands" coarsening failure (a wall that vanishes when
the mask is coarsened → pressure leaks across a barrier the fine grid considers solid), and a hierarchy
must rebuild its coarsened operators on *every* `destroy_wall`; (3) GPU coarse-level under-occupancy
breaks the many-parallel-instance farm.

**The multigrid-as-coupling-machinery hypothesis I floated is rejected** — and the reasoning is good:
multigrid restriction/prolongation move an *algebraic residual of one operator* to accelerate its own
convergence; the wave→atmosphere coupling is a *physical energy transfer between two PDEs* with a
conservation law. They share the words and nothing load-bearing. Worse, a true multigrid restriction is
a weighted **float average** — order-dependent, i.e. exactly the bit-exactness we can't lose.
*Reserve option:* if measurement ever shows 8 GS sweeps under-relax at the target resolution, add a
**single** two-grid correction (2:1 only, where MIN-restriction is still mask-safe) for the diffusion
solve *alone* — never as the cross-system coupler. **Measure before building it** (Decision 6).

### FFT — confirmed off the table
FFT/DST/DCT diagonalize the Laplacian only on a *separable rectangular* domain; arbitrary interior
Neumann walls destroy that separation. The only correctness workaround (capacitance-matrix/embedding)
needs a dense `p×p` correction over the `p` boundary cells, **rebuilt on every topology edit** — and a
wall-dense ship has `p` in the hundreds-to-thousands, so it's big *and* volatile. Independently fatal:
cuFFT is float-internal and bit-reproducible only on the *same* GPU model, violating cross-machine
determinism. Not even viable as a multigrid bottom solver (needs a wall-free coarsest grid). **Your
recollection was exactly right.**

### Subcycle-the-wave-when-active — the immediate win (with a farm caveat)
Today `n = ceil(sim_time / dt)` runs the full wave+diffusion+smoke loop ~16× **every tick, even in a
dead-calm room**. The fix (~10 lines, Python-only, `PhysicsRunner.step`): gate on
`wave_active = wave_source.any() or |wave_p|.max() > εₚ or |wave_v|.max() > εᵥ`. If inactive: skip the
wave kick, run the (unconditionally stable) implicit diffusion **once** at full `dt`, advect smoke once
on that settled wind. A calm ship collapses **~16 substeps → 1**. If active: subcycle only the wave at
its CFL `dt`, diffusion still once. This **decouples the wave's substep budget from the diffusion's** —
exactly the "worst of both worlds" fix we discussed.
**Caveat (don't lose this):** on the GPU farm, per-instance gating breaks SIMT uniformity — some lanes
subcycle a blast, some don't → warp divergence; the alternative (per-*batch* max-substep) defeats the
savings. So the gate is a **single-game / dev-loop win** and roughly a wash on the farm. Land it now;
don't cite it as evidence the architecture scales.

**BUILD FINDING (2026-06-15) — the gate is NOT a decision-free standalone; it's a dt-policy problem.**
A first attempt (wave-active gate + dead-wave snap-to-zero + smoke-diffusion-CFL floor) **broke
breach-venting** (`tests/test_smoke_sink_pull.py::test_breached_room_clears`): a "calm" venting room
stopped clearing. Root cause: **the substep count is overloaded.** The smoke **sink-pull** (the
drain-toward-breach mechanism) is **capped at 1 cell/substep**, so its drain *rate* is silently coupled
to the substep count — the old ~18 wave-CFL substeps were doing double duty (wave CFL **and** 18 cells/
tick of venting drain). Collapsing to 1 substep cut the drain 18×. So at least **four** distinct needs
ride one number: wave CFL, implicit-diffusion stability (needs 1), smoke explicit-diffusion CFL, and the
**sink-pull drain rate**. Decoupling them is exactly the unification's "coherent dt policy" pillar — so
the gate is **folded into the unification**, not shipped standalone. (A venting-detection hack that
preserves the old count was rejected: it hard-wires the drain rate to the arbitrary wave-CFL count and
the unification redoes it anyway. The proper fix likely decouples the sink-pull cap from the substep
count — a smoke-solver change — so the drain rate is a tuned constant independent of how the wave
substeps.) The dead-wave snap-to-zero (prevents the explicit kick amplifying a sub-eps residual at the
big dt) and the smoke-diffusion-CFL floor are both correct and carry forward into that work.

---

## The reframe: one finer grid, not per-system grids

The critic's strongest move, and I think it's right: **default to one uniform finer grid + a coarse
wave, and make per-system multi-resolution *prove* it buys more before we build it.**

- Your actual goal — fine atmosphere/smoke/light, coarse wave — is delivered by running the whole sim at
  `base/2` (halve `tile_size_m`, which **engine/01 §2 already supports**) and keeping the wave as a
  **coarse strided view** (`row>>1, col>>1`) of that one grid. The wave stays cheap; everything visual
  goes fine.
- This keeps engine/01 §1's "**exactly one grid**" intact — there's still one grid, just finer. The
  per-system design re-introduced a *second coordinate vocabulary* (`frow/fcol` beside `row/col`), which
  is precisely the dual-grid structure that killed the old dead design (spawn points landing ⅓ across the
  map — that was vocabulary drift, not numerics, and `coords.py` discipline is the *same* firewall the
  dead design thought it had).
- It **deletes the five bespoke conservative fixed-point transfers**, the masking-redistribution rule,
  and — critically — the fire-step crossing problem (below), which the per-system design never solved.

**Open mechanic (a sub-decision, not a blocker):** the coarse wave can be either a strided *view* of the
one fine grid, or run fine-but-subcycled. Either way it's one allocation, not two grids.

**My recommendation:** treat "uniform `base/2` + coarse/strided wave" as the **default**, and only
revisit per-system grids if we can articulate a concrete thing it buys that the uniform knob doesn't.
As written, the research couldn't name one.

---

## The real work: Q16.16 fixed-point migration (the actual project)

The critic is right that this was understated. Moving atmosphere/wave/smoke/wind to fixed-point is **not
a prerequisite step — it's the multi-week project**, and resolution is the bow on top:
- The implicit GS update is a **per-cell division** (`(rhs + μ·nb)/(1 + μ·wsum)`) every sweep, every
  tick — fixed-point division is the single nastiest op to make bit-exact *and* fast on GPU.
- The `mean_wp = sum/count` **global float reduction** (`atmosphere_solver.cpp`) is already a latent
  cross-machine desync, and more cells make it worse; a deterministic fixed-point reduction must agree
  bit-for-bit across CPU-scalar, CPU-SIMD, and CUDA warp-shuffle — three *structurally different*
  summation orders. This is the deepest determinism trap and it's real.
- The **two-seeded-sim determinism harness is specified but not yet written** (engine/ml/01) — so there's
  currently **no CI guard** that would catch a non-deterministic transfer. That harness is a hard
  prerequisite before any of this; without it the determinism claims are unfalsifiable.

This reorders the runway: the fixed-point migration *is* the prep for CUDA, and resolution rides on top
of it — consistent with your "fixed-point first, tested, then CUDA" instinct, just bigger than it looked.

---

## Code findings to act on regardless of resolution

1. **`stamp_units` is a per-tick full-field rebuild** (`gamemap.py`: `dyn_permeability[:] = permeability`
   + a Python loop over every unit footprint, *every tick*) — for `dyn_permeability` **and**
   `dyn_light_atten` (×3 RGB) **and** `dyn_wave_absorb`. This is a farm-scale cost **now**, at k=1.
   Moving it to C++/edit-triggered is a bigger decision-free farm win than the subcycle gate, and a
   prerequisite for any resolution change.
2. **Smoke diffusion is explicit Euler** (`smoke[i] += d_eff·dt·lap[i]`), CFL-bound `dt < dx²/(2D)`, with
   a *wind-dependent* `D` — so a strong shockwave wind at finer resolution can checkerboard the smoke.
   Refining smoke needs implicit diffusion or an explicit bound; "free to refine" is false for the
   diffusion half.
3. **The fire step crosses five fields at one index** (`fire_simulation.cpp` reads temperature/atmosphere/
   wind, writes smoke + atmosphere-plume at flat index `i`). A one-way footprint *reduction* cannot
   express fire *writing* a fine smoke ring and plume bump. This is why those fields must share a
   resolution — which "one grid" delivers for free and per-system grids could not.

---

## Recommended sequence (revised by the research)

1. ~~Subcycle-when-active gate (standalone)~~ — **tried, reverted (2026-06-15): not decision-free.** The
   substep count is overloaded (the sink-pull drain rate is coupled to it), so this is a **dt-policy
   problem folded into the unification** (#3), not a standalone first step. See the BUILD FINDING above.
2. **Move `stamp_units` to C++ / edit-triggered** — decision-free farm win, prerequisite for resolution.
3. **PhysicsEngine unification** — the glue→C++, the **coherent dt policy** (decouple wave CFL /
   diffusion stability / smoke-diffusion CFL / sink-pull drain rate — incl. the now-folded
   subcycle-when-active + dead-wave snap + the GS-residual instrumentation), the shared stencil, and #2.
4. **Q16.16 fixed-point migration + the two-seeded determinism harness** — *the* project; prerequisite to
   determinism and CUDA.
5. **Resolution** — adopt the uniform `base/2` + coarse/strided wave; measure whether it needs anything
   beyond the `tile_size_m` knob *before* building per-system machinery.
6. **CUDA.**

---

## Open decisions for Erik

- **D-A (the big one): per-system multi-res vs uniform `base/2` + coarse wave.** Recommendation: default
  to uniform; make per-system prove its worth. Do you agree, or is there a per-system benefit you want
  that I should have the panel stress-test?
- **D-B: is smoke sim-affecting?** Still your open call (poison damage path). Doesn't change the
  recommended sequence; does decide whether smoke gets a fixed-point read-path.
- **D-C: does determinism need to survive *different* GPUs (a teammate's card), or only same-GPU /
  single-machine replay?** Integer atomics + integer transfers are bit-exact across any CUDA GPU; **float
  solves are not guaranteed across architectures even with pinned order.** This decides whether a
  float-internal solve is *ever* admissible for lockstep, or whether the whole sim path must be
  fixed-point. It's the most consequential determinism question and it's genuinely yours.
- **D-D: resolution `k`** (2 vs 4) — measured against the *parallel-instance count*, not single-game
  headroom (k=2 is 4× field memory per instance). Defer until we have the farm sizing.
