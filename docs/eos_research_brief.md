# EOS / ideal-gas reframe — research brief (roadmap Phase 1.1)

> **Authored 2026-07-05** (Erik + Claude, last-Fable-day planning session). This doc is the
> **input** to the Phase-1.1 deep-research pass — execute on fresh tokens (Opus, deep-research
> workflow), passing this brief as the refined question. The report it produces feeds Phase 1.2
> (the rung A/B prototypes) and the Phase 1.3 adopt/defer decision.
>
> Read-first siblings: the reframe itself — `blackbody_smoke_and_rendering_brainstorm.md` **§0**;
> determinism scope — `determinism_without_fixedpoint_research.md`; plan — `roadmap_2026-07.md`.

---

## 1. What we are deciding

Breach may repurpose its pressure field via the ideal gas law: stop storing pressure as primary
state; store **N** (particle/mass density, the conserved quantity — the existing per-gas
`gmap.gas` slices promoted to real mass), add **one** new field, a shared gas temperature **T**,
and derive **P = C·N·T** every step. Two ambition rungs are on the table:

- **Rung A — "Darcy-EOS refit":** conservative N-flux + derived `P = C·N·T` + advected gas T,
  quasi-static wind (no momentum field). Moderate cost; no real curl. ≈ "everything downstream
  stays the same."
- **Rung B — full compressible with momentum:** the real rewrite — velocity is state, baroclinic
  vorticity exists, mushroom clouds and shock steepening are physics, not garnish.

The research question: **which numerical scheme(s) should each rung use**, what do they cost at
Breach's scale, where do they break, and what will they visibly buy us — so Erik can make the
1.3 call (adopt A / adopt B / defer) on evidence instead of vibes.

## 2. The system as it exists (facts the research must respect)

- **Grid:** 2D top-down tiles, 1 tile = **1/3 m**. Current test ship ≈ 50×120 tiles; `--res 2/3/4`
  multipliers exist; think "order 10⁴–10⁵ cells", not film-sim 512²+.
- **Tick:** 83 ms (~12 Hz sim). Current per-tick dt policy (`engine/05` §2): explicit **wave**
  substeps at CFL ~1.67 ms (≈50/tick — the engine already sustains this comfortably), **implicit
  diffusion once** (RB Gauss-Seidel), smoke semi-Lagrangian on the once-solved wind, K sink-hops.
- **Fields today:** `atmosphere` (slow pressure-ish), `wave_p` (acoustic overlay), wind =
  −∇(atmosphere + wave_p); `gas` = (n_gases, h, w) **passive tracers** (no mass, no back-reaction);
  `heat` = per-tick radiation deposit (cleared every tick); `temperature` = **solids only**
  (air has no temperature today — that is the field the reframe adds).
- **Compute:** every solver is C++ **and** CUDA-ported, bit-identical (tag `cuda-breached`,
  2026-07-04). GPU path is per-call (naive transfers) and still fast; the S8 GPU-residency pass
  (`docs/cuda_s8a_residency_spec.md`) is parked as the optimize-hard step. **There is real
  performance headroom** — Erik: "the simulations are crazy fast."
- **Qualitative-physics-as-differentiator:** Breach does not compete on graphics; it competes on
  *qualitatively different things happening* than in other games. Weight criteria accordingly.

## 3. Fixed constraints (decided — not up for re-litigation by the research)

1. **Heat is a strict one-way channel** (decision #2, blackbody doc §8, DECIDED 2026-07-05):
   gas/soot absorbs heat and **never re-radiates it**; heat ray-sources are combustion/weapon
   events only, O(burning tiles) not O(plume). Hot gas emits **light**: per-tile in-march glow
   (free) + brightest-K promoted ray sources. Any scheme must feed a T that drives the §6.2
   emission LUT; none may assume smoke-reradiated heat.
2. **Determinism scope** (determinism report + blackbody §0.9): the chaotic fluid runs **float,
   local, non-authoritative**. Only thin scalar crossing-quantities are authoritative/integer
   (ignition flags, shockwave impulse on units, water heat-sink, HP deltas). Same-box ML training
   needs only pinned RNG + reduction order. So: schemes are NOT gated on integerizability — but
   clean extractability of those few crossings is a criterion, and scheme-internal nondeterminism
   that breaks *same-box* replay (see Q10) is a real problem.
3. **2D top-down now, 2.5D later:** the K-layer stack (blackbody §0.5) is the intended future;
   schemes should extend to K stacked layers with per-column buoyancy exchange without redesign.
4. **Water/steam phase change is owned by the water effort.** The EOS only needs steam as a
   source/sink term with latent-heat hooks; do not design the phase change.
5. **Gameplay heat transport stays radiation-only by default.** The reframe gives air a real T,
   which *tempts* contact/convective ignition — that is open question Q5, to be answered with
   precedent, not assumed.
6. **Realtime on consumer GPUs** (RTX 3070 / RTX 1000 Ada class), leaving budget for the game +
   NN inference. A scheme that only works at film-render budgets is out.

## 4. Candidate schemes (seed list — the research should add, merge, or kill candidates)

1. **Kwatra et al. — semi-implicit compressible flow** (removes the acoustic time-step
   restriction via a pressure solve; the graphics-community standard for explosion-scale
   compressible effects). The prime rung-B candidate on paper.
2. **Feldman–O'Brien suspended-particle explosions / divergence-controlled incompressible** —
   incompressible solve + prescribed divergence sources at combustion; the classic
   film/games explosion *hack*. Cheap curl; no real acoustics. Possibly "rung A.5".
3. **Explicit compressible Euler + artificial viscosity** (von Neumann–Richtmyer lineage) —
   honest shocks, brutal acoustic CFL (see Q1); simplest math, most substeps.
4. **Thermal / double-distribution Lattice-Boltzmann** — fully local stencils (GPU-native,
   embarrassingly parallel, even integer-friendly), but stability at extreme T gradients and
   the compressible-regime fit need scrutiny.
5. **Low-Mach / anelastic / pseudo-incompressible approximations** (atmospheric-CFD lineage) —
   filter sound entirely, keep buoyancy + baroclinic torque. Cannot do blast fronts — but a
   **hybrid** (low-Mach thermodynamics + the existing tuned `wave_p` acoustic field kept as-is
   for blasts) may be exactly rung A's sweet spot. Evaluate the hybrid explicitly.
6. **Weakly-compressible with a capped/tuned sound speed** (WCSPH-style stiffened EOS on a grid)
   — accept slower-than-real shocks to loosen CFL; the current engine already runs a tuned
   c ≈ 200 m/s (see Q1), so precedent exists in-house.
7. **"Stable-Fluids-compressible" baseline** — semi-Lagrangian advect (N, T, u), derive P, kick
   u by −∇P, no acoustic sub-solve. The naive scheme the first rung prototypes resembled;
   include it as the control everything else must beat.

## 5. Questions the report MUST answer

- **Q1 — the acoustic CFL, quantified at our numbers.** Worked example to beat: hot core
  2500 K → c = √(γR_s·T) ≈ 1000 m/s; dx = 1/3 m ⇒ explicit dt ≤ 0.33 ms ⇒ **~250 substeps per
  83 ms tick** (ambient 347 m/s ⇒ ~86). The shipped wave solver runs ~50 substeps/tick at a
  game-tuned c ≈ 200 m/s without strain, so brute force is not *obviously* fatal on GPU — but
  per-substep cost of momentum+energy updates differs per scheme. For each candidate: substep
  count at ambient and at hot-core, per-substep cost shape, and whether c can be treated as a
  tunable dial (game-speed sound) without wrecking the visual payoff.
- **Q2 — stability under game abuse:** a multi-thousand-K energy dump into 1–4 tiles, hard
  vacuum boundaries (hull breach), sealed rooms, doors slamming (topology changes mid-run).
  Which schemes survive without clamps that kill the look?
- **Q3 — field unification:** does `atmosphere + wave_p` collapse into one (N, u, T) system, or
  is keeping the acoustic overlay separate (hybrid, candidate 5) the better engineering? What do
  comparable engines do?
- **Q4 — combustion source terms:** concrete burn-rate/energy-release/yield formulations used in
  film & games (fuel + O₂ → soot + steam + energy), and how they drive (N, T) without mass or
  energy blow-ups. Numbers, not just shapes.
- **Q5 — ignition pathway precedent:** with a real gas T, do shipped games/sims let hot gas
  ignite by contact/convection, or keep radiation-only ignition? Consequences either way
  (gameplay: fire spreading through hot air pockets vs our current ray-based spread).
- **Q6 — tracers become mass:** the existing per-gas slices N_i as the conserved mass (Dalton:
  shared T and u, per-gas N_i, per-gas molar mass M_i as a constant driving buoyancy/settling).
  What breaks when smoke stops being passive? Momentum of emitted gas at nozzles (flamethrower)?
- **Q7 — boundaries & venting:** vacuum outrush, conservation in sealed rooms, and whether the
  EOS genuinely fixes the known lingering-haze/venting weakness (the current sponge/sink-hop
  hacks — engine/04 §4) rather than reproducing it.
- **Q8 — CUDA shape:** stencil locality, atomics, memory traffic per cell per substep; fit with
  the existing RB-GS and semi-Lagrangian kernel patterns; ×K cost for future 2.5D layers.
- **Q9 — visual evidence at COARSE grids:** find footage/papers demonstrating baroclinic curl,
  mushroom caps, and the fireball cooling arc at game-ish resolutions (≤256²), not 512²+ film
  sims. Which schemes' good looks survive coarsening?
- **Q10 — same-box determinism traps:** scheme-internal nondeterminism (adaptive iteration
  counts, data-dependent convergence tests, atomics-order-sensitive reductions in float) that
  would break same-box replay even with pinned RNG — flag per scheme (cross-links Q1 if adaptive
  local time-stepping is proposed).

## 6. Evaluation criteria (weights)

| Criterion | Weight | Notes |
|---|---|---|
| Realtime cost at Breach scale, CUDA-shaped | 30 % | Q1/Q8. Hard gate below the weights: fails realtime ⇒ out. |
| Visual payoff at coarse game grids | 25 % | Q9. Curl, cooling arc, geometry interaction — the point of the exercise. |
| Stability under game abuse | 20 % | Q2. Hard gate: needs look-killing clamps ⇒ out. |
| Integration fit (fields, tick order, 2.5D, existing kernels) | 15 % | Q3/Q6/Q7/Q8. |
| Determinism-scope fit (crossing extraction, same-box replay) | 10 % | Q10 + constraint 2. |

## 7. Deliverables (the report)

1. **Comparison table** — schemes × criteria, scored, with the two hard gates applied.
2. **Recommendation** — one scheme per rung (A and B), with runner-ups and the reasoning that
   would flip the choice.
3. **Worked CFL/substep math** at Breach's numbers (Q1) per recommended scheme.
4. **Pseudocode sketch** per recommended scheme, shaped for a numpy prototype using our field
   names (N_i, T, u, P; sources: combustion, breach, door).
5. **Risk register** — top 5 risks with mitigations (e.g. "hot-core CFL blows budget → cap c").
6. **Annotated reading list** — max ~10 items that actually matter, one line each on why.

## 8. Prototype scenario spec (Phase 1.2 — pre-agreed; the research report should sanity-check it)

Judged **by eye** (Erik), rendered as GIF A/Bs — rung A vs rung B vs the current engine, same
geometry and seeds. Float numpy, non-deterministic, in-engine-shaped (our field names). Real ship
geometry, **not** empty space (explicit correction of the first-round GIFs, which were
free-space explosions and undersold both rungs):

- **S1 — corridor blast:** detonation at one end of a long corridor; expect a planar front,
  reflection at the far door, venting behaviour.
- **S2 — room + door jet:** blast in a sealed room with one open door; expect overpressure,
  a jet through the doorway, room-scale circulation (rung B) vs none (rung A).
- **S3 — breach to vacuum:** hull breach on a smoke-filled compartment; expect sustained outrush,
  fire starvation as N drains, honest venting (the current model's known weak spot).
- **S4 — open-bay fireball over standing smoke:** ignition inside an existing smoke cloud;
  expect the expansion to visibly push/shape the smoke (Erik: "if the hot air expands I suspect
  it will look nice on the smoke") and the cooling arc to read (fireball → plume → soot).
- **S5 — water displacement pushes smoke (tilted ship):** a ship tilted a few degrees, a
  ruptured water container releasing a flooding front across the deck, and a **non-uniformly
  distributed** smoke cloud already in the compartment. Expect the rising water to shrink the
  air column, raise local pressure, and drive a wind that visibly **pushes and deforms the smoke**
  ahead of the flood — the dynamic-air-motion effect Erik is chasing. This tests the
  **already-shipped** water↔air coupling (engine/07 §5.1: W3 volume displacement + W4 pressure
  head, live at `k_p=0.5`, `ceiling_h=2.5 m`) and, critically, whether the EOS reframe makes that
  push read **dramatically** rather than subtly. Needs the tilt + `floor_height` + `water_depth`
  fields the harness must carry anyway; non-uniform smoke seeding is the point (a flat blob shows
  nothing being pushed).

## 9. Reading seeds (the research finds the rest)

- Kwatra, Su, Grétarsson, Fedkiw — avoiding the acoustic time-step restriction in compressible
  flow (JCP 2009).
- Feldman, O'Brien, Arikan — Animating Suspended Particle Explosions (SIGGRAPH 2003).
- Bridson — *Fluid Simulation for Computer Graphics* (2nd ed.) — the graphics-side baseline for
  every candidate above.
- Thermal/double-distribution LBM survey (pick a modern one; GPU implementations exist).
- Atmospheric low-Mach lineage: anelastic (Ogura–Phillips), pseudo-incompressible (Durran).
- Shipped-game precedent sweep: Noita (falling-everything engine talks), Teardown, Frostpunk,
  Company of Heroes (non-deterministic effects over synced sim) — for Q5/Q9 precedent.
