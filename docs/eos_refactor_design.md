# EOS refactor — design document (rung B, compressible ideal gas)

> **Status:** DRAFT for design-gate review (adversarial critique, then Erik). Not canon,
> not a build order. Companions (read before this, they hold the locked decisions this
> doc turns into an architecture): `docs/eos_refactor_decisions.md` (LOCKED 1–9 +
> peripheral decisions), `docs/eos_refactor_interaction_map.md` (code-grounded
> read/write inventory, §D risk list), `docs/eos_research_report.md` §4 (Kwatra
> pseudocode + CFL story). Working reference: `prototypes/eos/scheme_rung_b.py` +
> `eos_core.py` on branch `eos-prototype` (float numpy spike, self-verified stable
> at 128²; this doc treats it as an implementation reference, not a spec).
>
> **Per the design-gate practice (decision 9): nothing is built until this doc
> survives critique.** No engine code changes in this pass.

---

## 1. Goal & scope

**Adopt rung B** — a genuine compressible ideal gas — as Breach's pressure model,
replacing the current two-field `atmosphere + wave_p` IMEX scheme.

Two primary fields carry the physics; everything else is derived or downstream:

- **Gas temperature `T`** (Q16.16 Kelvin, unified across gas and solid — decision 7).
- **Particle density `N`** (Q16.16, per-species, `(N_species, h, w)` — decision "B" /
  OPEN-B resolution).

Pressure is **derived, not stored-as-primary**: `P = C · N_total · T` (Dalton sum over
species), materialized once per tick into a stored field (decision 5). Downstream
consumers (wind, water head, burst walls, unit push) **read** derived `P`; upstream
writers (explosions, fire, water displacement, breach venting) **feed** `N` or `T`,
never `P` directly (interaction map §B: "feeding pressure stops existing").

**Fixed-point discipline:** everything involved is *already* Q16.16 (decision 8) —
`atmosphere`, `wave_p`/`wave_v`, `gas` (all 5 species), `heat`, `temperature`, `water_depth`,
water's `flow_vx/vy`. This refactor **rearranges** existing fixed-point fields into a new
equation structure; it does not introduce float→fixed conversion anywhere. That is the
single biggest de-risking fact about this refactor and it should be treated as a hard
constraint on every patch below: if a patch is reaching for a float intermediate that
isn't one of the three already-known bridges (§7), that's a red flag, not a shortcut.

### Explicitly OUT of scope

- **2.5D z-layers.** Deferred (decisions, "peripheral"). The flat-2D EOS lands first,
  or the ship never grows z at all ("we'll see"). Z is the natural home for
  molar-mass buoyancy and retires today's permeability fudge (furniture solid-low/open-high)
  — but neither is needed to ship rung B flat.
- **Molar-mass buoyancy.** Deferred *with* z (no z-layer ⇒ no buoyancy axis for it to
  drive; per Avogadro every gas contributes equally per particle to pressure at fixed
  T, so a single flat layer has nothing for molar mass to differentiate).
- **Species-realistic combustion kinetics / diffusion.** Decision: "no lit search" —
  game-adequate stoichiometry (§5) is deliberately under-researched; only a switch to
  *realistic* multi-species kinetics would warrant literature work, and that is not this
  patch's goal.
- **Through-wall wave transmission** (engine/04 §5, "4b" — already deferred pre-EOS,
  unaffected by this refactor either way).
- **Rewriting the fire intensity feedback logistic** (engine/06 §5, stage 1). It keeps
  its shape; only its `o2` term's *input* changes (mean `atmosphere` → mean `N_O2`, §5).

---

## 2. Field architecture

### 2.1 Species set

Extend the existing `gmap.gas` dense `(N_species, h, w)` Q16.16 array
(`src/simulation/gases.py`, currently `N_GASES = 5`: `white_smoke, black_smoke, poison,
teargas, fuel_gas`) with **two new bulk species**, prepended or appended as new
contiguous ids (exact id placement is a patch-1 detail, not architectural):

| Species | Role | Conservative? | Notes |
|---|---|---|---|
| **O2** | bulk breathable oxygen | Yes (transport-conserved) | combustion sink (§5), suffocation gate |
| **inert_N2** | bulk inert filler | Yes (transport-conserved) | today's "the rest of the atmosphere"; smothers fire, dilutes O2 |
| `white_smoke`, `black_smoke`, `poison`, `teargas`, `fuel_gas` | existing traces | No (decay-permitted, per `gases.py`'s loaded-but-unapplied `decay` column) | unchanged role; ride on top of the bulk as before |

This is the OPEN-B resolution from the decisions log: O2 is "just another slice in the
existing `(N_species,h,w)` Q16.16 gas array" — ≈ +1 bulk field over today's single
`atmosphere` scalar (N2 is the second). No new array shape class, no new dtype, no new
transport kernel class — the per-gas semi-Lagrangian advection Breach already runs
per-slice (`PhysicsRunner.step`'s per-gas loop) is the transport primitive for O2/N2 too,
with one change: **O2 and N2 must not decay** (§2.2 — decision 2 in the interaction map:
"Gas as *conserved* N is a real fight, not a coefficient").

`N_total = Σ_i N_i` over **all seven** species (Dalton — decision "Multi-gas = Dalton sum
+ threshold chemistry"). `P = C · N_total · T`. The existing five traces continue to
contribute to `N_total` (they are real gas mass, just non-conservative in bookkeeping) —
this is a deliberate simplification already blessed by the Dalton decision, not a new one.

### 2.2 The conservation fight (interaction map §D-2, made concrete)

Today's smoke advection is **deliberately non-conservative**: the semi-Lagrangian sampler
loses mass to interpolation truncation every step ("accepted Q-S2-1"). That's fine for a
visual tracer; it is **not** fine for O2/N2, because:

- Sealed-room suffocation math depends on `N_O2` actually going down only when something
  consumes it (combustion) or it's transported out (a breach), not from silent numerical
  leakage.
- `P = C·N_total·T` must track real physical pressure, so `N_total`'s bulk component
  (O2+N2) needs the same conservation guarantee the rung-B research report assumes
  (`§4`'s conservative-advection framing, `N_i, N·u, N·E`).

**Resolution for this design:** the bulk species (O2, N2) get a **mass-corrected**
semi-Lagrangian step — advect, then rescale the post-advection field by
`(Σ N_before / Σ N_after)` over the open-air mask each tick (a global renormalization,
not per-cell — cheap, one reduction + one scalar multiply, and it is exactly what a
fixed-sweep-count discipline wants: no adaptive correction, one deterministic pass).
This is *not* full finite-volume conservative transport (the report's `N·u, N·E`
triple) — it is the cheapest fix that makes the *conserved-in-aggregate* property hold,
which is what suffocation/fire-starvation/breach-venting gameplay actually needs. The
prototype (`eos_core.py`, `scheme_rung_b.py`) does **not** implement even this — it
advects a single scalar `N` as a plain semi-Lagrangian tracer with no renormalization.
Flagged explicitly as **Open question 1** (§10): whether global renormalization is
sufficient or whether local flux-form conservation is needed for gameplay to read right
(a big sealed room breached at one corner should show O2 draining asymmetrically, which
global renormalization cannot represent — see §10).

### 2.3 What's deleted / merged

- **`wave_p` merges into unified `P`.** It existed only as a numerical workaround — the
  IMEX split ran implicit-diffusion (bulk `atmosphere`) and explicit-wave (`wave_p`) on
  two fields because those two PDEs fight on one field (engine/04 §4). Rung B's
  semi-implicit split produces **both** bulk equilibration and acoustic waves from the
  *same* `(N, T)` → `P` derivation (decision 2). The transient-buffet vs sustained-dome
  behavioral distinction that `apply_wave_push` and water's head term rely on survives
  as **the single field's time evolution** (a blast front passes through P as a fast
  transient, the post-blast dome lingers as P's slow relaxation) — not as two separate
  fields.
- **`wave_v`, `wave_source`** are subsumed into the Kwatra solver's own state (§3) —
  there is no separate explicit-wave velocity/staging buffer; momentum `(vx, vy)` and the
  Helmholtz pressure correction replace them.
- **`sink_hop` deleted** (decision 3). Breach venting emerges natively from `−∇P` toward
  a true-vacuum (`N=0`) cell — the geometric BFS hack existed only because the old
  scheme's wind died as pressure equalized (engine/04 §4's "lingering-haze" discussion).
  Under rung B a breach sustains a real density gradient until the room actually empties.
  Also **generalize breach→vacuum beyond today's edge-hull-only rule** (interaction map
  §B, "the EOS would generalize it") — any destroyed wall exposing `is_vacuum=True` vents
  natively, not just edge-hull tiles.
- **Dead `wave_solver.{cpp,h}` deleted** — already-verified dead code (no caller, no
  binding), superseded by `AtmosphereSolver` even before this refactor (interaction map §0).
- **`atmosphere` is reinterpreted, not deleted.** Today's `gmap.atmosphere` (Q16.16,
  1.0 = standard atm) becomes the **bulk density read** — either it *is* `N_total` under
  a fixed `C`, or it is retired in favor of reading `N_total`/`P` directly through the
  same accessor name for compatibility. This is a naming/API decision for patch 1, not
  an architectural fork — flagged as **Open question 2** (§10).

---

## 3. The compressible solver (Kwatra semi-implicit)

One solver, one primary state per tick, matching the research report §4 and the working
prototype (`scheme_rung_b.py`).

### 3.1 Tick order

```
0. P materialized from LAST tick's (N, T) — see §3.4 (scheduling contract).
1. EXPLICIT advection substeps, at the |u|-CFL (not |u|+c):
     for n = ceil(dt_tick / dt_adv) substeps, dt_adv = CFL_ADV·dx/(max|u|+eps):
       advect (vx, vy) on themselves (self-advection, §3.3)
       advect each N_i (bulk: renorm-corrected; traces: as today, decaying)
       advect T
       zero (vx,vy,N) on solid; T -> ambient on solid
2. Prescribed div_target source (thermal expansion + water displacement) —
   folded into the acoustic solve's RHS so a resting gas with no existing
   momentum still starts moving (§3.2).
3. IMPLICIT acoustic solve: fixed-sweep Red-Black Gauss-Seidel Helmholtz/Poisson
   for pressure correction p'  (REUSE the existing RB-GS kernel class).
4. Velocity correction: (vx,vy) -= dt·grad(p')/N ; zero outside open-air.
5. Compression-work temperature update: T -= (gamma-1)·T·div(u*)·dt  (adiabatic).
6. P = C · N_total · T  — materialized ONCE, stored, BEFORE any consumer (§3.4).
7. Combustion pass reads this tick's P/N/T (§5).
8. Downstream consumers run (§6): wind, water head, burst walls, unit push, fire O2 gate.
```

Advection is substepped at the `|u|`-only CFL (Kwatra's entire point — decoupling from
the acoustic speed `c`); the implicit Helmholtz solve is **one fixed-sweep pass per
tick**, not per substep (it does not need to track the fast advective motion, only the
slow-relative-to-advection acoustic response) — this matches the prototype's
`self.last_substeps` accounting (advection-only) and its separately-fixed
`PRESSURE_SWEEPS = 40`.

### 3.2 The div_target source (a finding from the prototype, carried forward)

The prototype's module docstring documents an empirically-found gap: a **purely
reactive** Helmholtz RHS (`dt·c_max²·div(u*)`, responding only to existing velocity
divergence) produces **zero** motion for a resting gas with no initial kick — e.g. hot
gas sitting still post-blast, or water compressing air from rest. The fix folds rung A's
own prescribed-divergence recipe (thermal-expansion term `K_EXPAND·(T−T_ambient)/T_ambient`
+ water-displacement term `W_DISPLACE_GAIN·d(free_height)/dt`) into the RHS as
`div_target`, so `rhs = dt·c_max²·(div(u*) − div_target)`. This is carried into the
design as the acoustic solve's **source term**, not an optional extra — without it,
combustion heat and water displacement (§5, §6) would have no mechanism to actually
pressurize anything. Reusing rung A's already-tuned constants (rather than inventing new
ones) is explicitly the prototype's choice and this doc adopts it as the patch-1 starting
point (subject to the TBD retuning list, §9).

### 3.3 Momentum representation — a carried simplification, flagged

The prototype advects **velocity** `(vx, vy)` as a self-advected field (semi-Lagrangian
on itself), not the conservative triple `(N_i, N·u, N·E)` the research report's
pseudocode sketches. Rationale (from the prototype's own docstring, endorsed here):
dividing momentum by `N` to recover velocity is least stable exactly where the flow is
most dynamic (breach fronts, near-vacuum `N`) — advecting velocity directly sidesteps
that division. This is a real physics simplification (it is not literally conservative
momentum transport) carried forward as the patch-1 default; flagged as **Open question 3**
(§10) — whether it holds up under Breach's abuse cases (multi-grenade stacks, O2-tank
rupture fireballs) or needs the full conservative form later.

### 3.4 Materialization contract (decision 5, resolved into a concrete rule)

`P` is computed **once per tick**, immediately after step 6 above (post velocity-correction
and temperature update, using *this* tick's final `N, T`), and **stored** in a field (not
recomputed per-consumer). Every consumer in step 8 reads that one stored `P` — this is
what makes wind, water head, burst-wall scan, and unit push all see an *identical* P
within a tick (interaction map §D-5), and what keeps a CUDA port clean (one kernel writes
`P`, every other kernel only reads it, no consumer racing a partial update).

Reused kernel: the Helmholtz solve is the **same RB-GS pattern** already proven
bit-identical cross-machine for the atmosphere diffusion pass (`atmosphere_solver.cpp`'s
`gs_iters`-sweep red-black loop, the "spike0b/S7 class") and for water's ripple/flow
solve. It is generalized with a variable coefficient `c_max²/N` and an added identity
term (report §4, prototype's `_solve_helmholtz`) — structurally the same fixed-sweep,
red-black, Neumann-mirror-at-solid, Dirichlet-zero-at-vacuum kernel Breach already ships,
not a new numerical method.

---

## 4. Unified temperature field

Per decision 7 (OPEN-A LOCKED 2026-07-09): **one** Q16.16-Kelvin `temperature` array
spans gas *and* solid cells, replacing today's split between the solid-only
`temperature` field and the phantom "air has no temperature" design (engine/06 §1).

### 4.1 Masked passes, one field

- **Gas rules** (open-air mask): semi-Lagrangian advect (§3.1 step 1) + compression-work
  `−P∇·u` (§3.1 step 5) + combustion/radiation sources (§5) — **no decay-to-ambient**.
  Gas energy conserves/advects/expands; it does not exponentially relax to a fake
  ambient the way solids do (interaction map §C: "cooling is physically WRONG for gas").
- **Solid rules** (solid mask): the existing convert (heat→temperature via
  `log2(thermal_mass)` bit-shift) → conduct (`face_shift` stencil) → cool
  (`cool_shift`/`cool_shift_vacuum`) pipeline, **unchanged** — `temperature_solver.cpp`'s
  current three-stage pass keeps its bit-shift arithmetic exactly as shipped
  (interaction map §C: "keep the working solid path as-is").
- **Conduction is ONE whole-grid pass**, unchanged in *mechanism*: the existing
  `conductivity`-keyed stencil (`temperature_solver.cpp`, no solid/air branch today) has
  **no solid/air branch to begin with** — it is keyed purely on the per-tile
  `conductivity` cache. Giving air a small nonzero `conductivity` value (today: `0.0`,
  engine/06 §2 table) makes the *same* stencil do air↔air, solid↔solid, **and** the
  solid↔air interface exchange, all for free. This interface exchange is called out in
  the decisions log as "the primary energy sink for sealed rooms — free here, explicit
  code in a two-field design" — it is the single biggest reason the unified-field
  decision won over keeping gas T separate.

### 4.2 Energy exits physically

Today's solid-only pipeline decays toward a phantom `ambient = 0` every tick (a
non-physical sink) and clears `heat` each tick (a one-way per-tick deposit, not an
accumulator). Under the unified field, energy must exit through **real** channels:

1. **Vent to vacuum** — hot `(N, T)` advects out through a breach exactly like every
   other field (§3.1 step 1); no separate mechanism.
2. **Conduct into the ship's thermal mass** — gas cells adjacent to solid conduct through
   the same whole-grid pass (§4.1); interior solids no longer decay to a fake ambient,
   they conduct to the adjacent gas.
3. **Hull radiates to space** — hull cells (already `is_vacuum`-adjacent via the exposed
   mask) use the existing `cool_shift_vacuum` path (`temperature_solver.cpp`, distinct
   from the interior `cool_shift`) as the terminal radiative sink. This is a
   **reinterpretation** of an existing mechanism (vacuum-exposed solids already cool
   faster via `cool_shift_vacuum`), not a new one.

The old "decay to ambient=0 + per-tick heat-clear" model is retired for gas; `heat`
itself (the ray-deposit buffer) keeps its existing per-tick-clear-after-both-readers
contract (engine/06 §"Resolved" notes) since that's a deposit buffer, not a state field —
unaffected by this refactor.

### 4.3 The GS-reciprocal class (a new numerical primitive, not a new class of risk)

The solid conversion pass is a free bit-shift because `thermal_mass` is a **fixed
power-of-two constant per material**. Gas "mass" under rung B is `N` — **dynamic,
per-tile, per-tick**. The gas heat deposit `ΔT = ΔE / (N·c_v)` is therefore a genuine
divide by a runtime value, which cannot be a compile-time shift. This is explicitly
flagged in the decisions log as needing "a per-tile per-tick fixed-point reciprocal (the
*proven* spike0b GS-reciprocal class), not the solid's free bit-shift" — i.e., this is
not a new numerical risk, it is an instance of a technique Breach already has proven
(the same reciprocal-multiply pattern water's solver already uses for `dt/dx` and
`2·dx` denominators, `water_solver.cpp`'s `make_recip`/`recip_mul`). Patch-level work
item, not open research.

---

## 5. Combustion on real O2

A **combustion pass**, its own class/function, run at a defined tick phase **after** the
field core (§3.1 step 7 — after `P` is materialized, so combustion reads a settled tick's
state, and its heat/O2-consumption feeds *next* tick's field update rather than racing
this tick's).

**Per candidate tile** (flammable material or `fire`-intensity > 0, per today's
`FireSimulation` — the ignition/spread *scaffolding* in engine/06 §5 is unchanged, only
its O2 *input* changes):

```
read:  N_O2[tile]  (local oxygen density, NOT atmosphere/P proxy)
       fuel         (wall_hp-derived, as today's F = clamp01(wall_hp/fuel_ref))
       T[tile]

if N_O2 > o2_thresh AND T >= ignition_temp AND fuel > 0:
    consume:  N_O2   -= burn_rate · dt        (integer decrement, saturating at 0)
    yield:    N_black_smoke += burn_rate · soot_yield · dt
              T            += burn_rate · H_fuel / (c_v · N_total) · dt   (heat deposit,
                                                                            §4.3's reciprocal)
```

This **replaces** the current O2 gate, which reads mean `atmosphere` over open neighbours
as a pressure-as-oxygen proxy (engine/06 §5 stage 1's `P` / `o2 = smoothstep(P_min,
P_full, P)` term, fed by `apply_temperature_ignition`'s atmosphere-threshold gate). Under
rung B, `N_O2` is a **real** species density (§2.1) — the gate becomes a genuine oxygen
read, not a proxy (interaction map §D-4, "a genuine density read — combustion should gate
on N, not P").

**Rates are TBD/tunable** (§9) — this design fixes the *structure* (consume O2 → yield
heat + soot, gated by threshold-O2 ∧ ignition-T ∧ fuel-present), not the constants,
matching the research report's Q3 answer ("mine the structure, tune the numbers by eye")
and the decisions log's explicit deferral ("combustion stoichiometry ... still open,
no lit search").

### Emergent payoffs this structure is designed to produce (not scripted)

- **Fire self-starves** as it eats local O2 — a sealed room's fire dims as `N_O2` drops,
  independent of fuel remaining.
- **Breach vents O2 → fire dies** — the same `−∇P`/`−∇N` outrush that vents pressure
  (§2.3) also drains the local O2 supply a fire needs; venting a room starves any fire in it.
- **O2-tank rupture → fireball** — a ruptured O2 tank is a local `N_O2` spike; any ignition
  source in that pocket burns hot and fast because the O2 gate is wide open, exactly the
  "one level up" payoff the rung-B adoption decision cites as its headline reason
  (decisions log item 1).
- **Inert flood smothers + suffocates** — dumping inert_N2 into a room dilutes local
  `N_O2` below `o2_thresh` for both combustion (extinguishes fire) and unit environmental
  tolerance (suffocates units) — one mechanism, two consumers.

---

## 6. Downstream consumers on derived P

Per the interaction map §A ("the bulk of downstream rides a derived P unchanged — the
low-risk half of your dream holds"):

| Consumer | Change required | Detail |
|---|---|---|
| **wind** `= −∇P` | Scheduling only | Reads the once-materialized `P` (§3.4); no formula change beyond `P` now being `C·N_total·T` instead of `atmosphere+wave_p`. |
| **water pressure-head** (W4, `water_solver.cpp`) | **Purify the float bridge** (decision 6) | Today: dequantize `atmosphere+wave_p` → float → `×k_p` → requantize (`water_solver.cpp` lines ~112–137, explicitly commented "FLOAT BRIDGE until S2"). Rewrite to read the derived **integer** `P` directly: `k_p` becomes a quantized Q16.16 coefficient, the head term becomes a `mul_q16(kp_q, P[i])` — pure integer, no dequantize/requantize round-trip. `water_solver.cpp` needs no *structural* change (it already isolates the head term behind the `head_on`/`kp_f` gate) — only the read source and the arithmetic mode. |
| **find_burst_walls** | None (formula-transparent) | Already reads a pressure spread vs `burst_threshold` — reads the derived `P` differential exactly as it read `atmosphere`'s. |
| **apply_wave_push** (units) | **Rewrite to read `grad(P)`** + recalibrate `k_push` | Today reads `grad(wave_p)` specifically (the transient field) via `reduce_grad(gmap.wave_p, tiles)` in `exchange.py`. Under the merge (§2.3), there is no separate transient field — `apply_wave_push` reads `grad(P)` on the unified field, and its buffet-vs-dome character now comes from *P's own time evolution* (a blast is a fast transient in P, the dome is P's slow relaxation) rather than from reading a dedicated zero-mean field. **`k_push` needs recalibration** (§9) — the transient P gradient right after a blast will differ in magnitude/shape from today's dedicated `wave_p` gradient. |
| **O2-gates-fire** | Reads real `N_O2` | See §5 — replaces the `atmosphere`-as-O2-proxy read entirely. |
| **temperature cooling (vacuum-exposure)** | None | Already a read-only compare against a pressure-like field; works unchanged on derived `P`. |

**Units remain partial obstacles/absorbers to the pressure wave** (teammate shielding —
kept per the peripheral decisions: `dyn_wave_absorb`'s per-material + per-unit absorption,
engine/04 §2.7 "4a", carries forward unchanged in spirit — units still damp the acoustic
component of `P`'s Helmholtz solve, same mechanism, generalized to the new solver's
coefficient field). **Units stay transparent to water** (kept — Erik's call, peripheral
decisions) — no change to water's solid mask.

---

## 7. Determinism / fixed-point plan

- **All new/changed fields are Q16.16**, matching every existing field this refactor
  touches (`N` per species, `T`, `P`, `vx/vy` if kept as a persistent field rather than
  solver-local). No float intermediate is introduced that isn't one of the three
  **already-known, already-scoped** bridges:
  1. Water's head-term float bridge (§6) — **purified by this refactor**, not carried
     forward.
  2. `atm_f_`/`wave_p_f_` float mirrors (interaction map §D-7) — rebuilt every tick today
     for water's still-float head term; **retired** once §6's purification lands (their
     sole consumer goes away).
  3. Any genuinely render-only channel (light, smoke_glow) — untouched by this refactor,
     stays float per the existing float/fixed split (engine/06 §3's stated principle).
- **Fixed GS sweep counts everywhere** — never a convergence tolerance. This is already
  Breach's rule for the atmosphere RB-GS pass (`gs_iters`, a solver-side constant, not
  config-adaptive) and the prototype's `PRESSURE_SWEEPS = 40`; the Helmholtz solve
  inherits it unchanged. (Research report risk #4 / Q10: "the one real trap is adaptive
  iteration counts" — a hard rule, not a judgment call, per tick.)
- **The per-tile reciprocal** (§4.3) is the one genuinely new fixed-point primitive this
  refactor needs, and it is a **known, proven class** (water's `make_recip`/`recip_mul`),
  not new territory.
- **CPU + CUDA in lockstep** (decision 9). Wave and diffusion already have bit-identical
  GPU mirrors (`cuda_wave.cu`, `cuda_atmosphere.cu`, plus `cuda_temperature.cu`,
  `cuda_water.cu`, `cuda_smoke.cu`, `cuda_fire.cu` for the other touched systems). **Every
  new or changed kernel in this refactor is a double implementation**, gated by the
  existing field-digest / xarch bit-identity harness (the same harness that resolved the
  `cuda-breached` finding). This is explicitly called out as a *larger migration surface
  than the docs implied* before the interaction-map pass (interaction map §D-6) — the
  patch decomposition (§8) treats CPU-then-CUDA-port as separate gated steps per solver
  component, not a single combined patch, precisely because of this.
- **The species mass-renormalization** (§2.2) is itself a determinism-sensitive step: it
  must be a single deterministic reduction (sum over the grid) + one scalar multiply pass,
  computed in a fixed order (or as an associative/commutative integer sum, which addition
  over Q16.16 int32 is) — flagged so the patch that implements it treats the reduction's
  determinism with the same care as the existing `heat` buffer's saturating-add discipline
  (engine/06 §3).

---

## 8. Patch decomposition

A build sequence sized for the autonomous-patch-workflow (plan once, execute per-patch
with design-gates on the risky steps). Each patch is CPU-only unless stated; CUDA ports
are pulled into their own gated patches per §7's "larger migration surface" flag, not
bundled into the CPU patch that introduces the kernel.

1. **Field-core infra.** Add `O2`, `inert_N2` species ids to `gases.py`
   (`N_GASES` 5→7); wire the mass-renormalization step (§2.2) into the existing per-gas
   transport loop, gated so it applies *only* to the two bulk species (traces keep
   today's decay-permitted behaviour, unchanged). No solver-behavior change yet — this
   patch is purely additive plumbing + a config/material-table row for O2/N2 initial
   densities. **Gate:** existing gas transport tests unaffected (bit-identical on the
   5 legacy species); new species round-trip through save/load and field-edit tooling.

2. **Compressible P solve (CPU).** Port the prototype's Kwatra split (§3.1–3.3) from
   `scheme_rung_b.py`'s float32 numpy into the C++ solver, in Q16.16, reusing the RB-GS
   kernel pattern (§3.4). This is the **highest-risk patch** — float→fixed-point port of
   a Helmholtz solve with a variable `c_max²/N` coefficient and per-cell reciprocals.
   **Gate: design-gate on the fixed-point derivation before implementation** (the
   float32-precision notes in the prototype's `_solve_helmholtz` docstring — the
   solid-face coefficient mirroring trick to avoid catastrophic cancellation — need to be
   re-derived for integer arithmetic, not assumed to carry over). Replaces
   `AtmosphereSolver`'s wave+diffusion steps; `wave_p`/`wave_v`/`wave_source` retired
   (§2.3). Delete dead `wave_solver.{cpp,h}` in this patch (housekeeping already flagged,
   zero-risk).

3. **Unified temperature field.** Extend `temperature` to the open-air mask with gas
   rules (§4.1), wire the whole-grid conduction pass to a nonzero air `conductivity`,
   retire the ambient-decay-to-0 for gas cells, wire `cool_shift_vacuum` as the hull
   radiative sink (§4.2). Implement the GS-reciprocal gas heat deposit (§4.3). **Gate:**
   sealed-room energy balance sanity check (a heated sealed room should conduct into
   walls and hull-radiate, not vanish or persist forever) — an E2E scenario, not just a
   unit test, per Erik's bug-fix-starts-with-E2E-repro standard.

4. **Combustion / O2 consumer.** New combustion pass (§5) reading `N_O2` instead of the
   `atmosphere`-as-proxy gate; rewire `apply_temperature_ignition`'s O2 test and
   `FireSimulation`'s `o2 = smoothstep(...)` term (engine/06 §5 stage 1) to read `N_O2`.
   **Gate:** the four emergent payoffs (§5) demonstrated in isolated scenarios
   (self-starving fire, breach-kills-fire, O2-rupture-fireball, inert-flood-suffocation)
   — a feel-check, tunable-constants-still-TBD (§9), but the *mechanism* must visibly work.

5. **Downstream-consumer migration.** `water_solver.cpp` head-term purification (§6,
   integer `mul_q16(kp_q, P)`); `apply_wave_push` rewired to `grad(P)` with `k_push`
   recalibration parked as a tuning pass (§9); `find_burst_walls` and wind need no code
   change beyond reading the new `P`'s storage location. **Gate:** water-head and
   unit-push behavioral parity check against pre-refactor gifs/replays where feasible
   (same spirit as the Phase-1.2 A/B/control bake-off), since these are the two consumers
   with a real formula change.

6. **CUDA ports.** One gated sub-patch per touched kernel (Helmholtz solve, unified
   temperature conduction, combustion pass if it's grid-parallel), each validated against
   the CPU path via the field-digest / xarch bit-identity harness before being considered
   done — mirroring how wave/diffusion's CUDA mirrors were proven (`cuda-breached`
   resolution). **Gate: bit-identical CPU/GPU digests**, non-negotiable per decision 9.

7. **Cleanup.** Remove the retired float bridges (`atm_f_`/`wave_p_f_` mirrors, §7);
   remove `sink_hop` and its BFS rebuild machinery (§2.3); fix the six stale doc spots
   flagged in the interaction map §0; fold this design doc's *as-built* shape back into
   the living `docs/architecture/engine/04_atmosphere_and_pressure.md` and
   `06_temperature_and_fire.md` chapters (per the post-EOS doc-consolidation plan already
   noted for this arc). **Gate:** none functional — a docs/cleanup patch, but sequenced
   last deliberately so the "as-built" write-up reflects what actually shipped, not what
   was planned.

Patches 2 and 3 are independent of each other in principle (P-solve vs temperature) but
patch 4 (combustion) depends on both being in place (it reads `N_O2` from patch 1/2 and
`T`/ignition from patch 3), and patch 5 depends on patch 2 (P must exist to migrate
consumers onto it). Patches 2 and 3 could run concurrently on separate worktrees per
Erik's standard multi-agent practice if desired; combustion and cleanup cannot.

---

## 9. TBD / tuning list (feel-gated, post-build)

Explicitly deferred to after-build tuning, not pre-build research (per the decisions
log's "no lit search" calls and the research report's "mine the structure, tune the
numbers by eye" verdict):

- **Combustion stoichiometry** — `burn_rate`, `o2_thresh`, `H_fuel` (heat yield),
  `soot_yield` (§5). Structure is fixed; numbers are Erik's feel-pass.
- **`k_push` recalibration** (§6) — the unified `P`'s transient-gradient magnitude near a
  blast will not match today's dedicated `wave_p` gradient scale; needs a fresh
  calibration pass against the existing knockdown-threshold tuning (`config.toml`
  `[exchange]`, calibrated 2026-07-05).
- **`k_p` recalibration** (§6) — water's pressure-head coefficient, same reasoning: the
  derived `P`'s scale/dynamics differ from `atmosphere+wave_p`'s.
- **Conductivity of air** — engine/06 §2's material table currently ships air at `0.0`
  (a hard insulator by design); the unified-field decision requires a **small nonzero**
  value, and "small" is a feel constant, not a derived one (too high and it behaves like
  the old air-temperature-advects-everything rejection the decisions log explicitly
  avoided; too low and the solid↔air interface exchange the decision banks on doesn't
  actually fire).
- **Cooling / radiation rates** — `cool_shift_vacuum` is reused (§4.2), but its rate was
  tuned for the old solid-only phantom-ambient model; the unified field's real
  conduct-then-radiate energy path may want a different rate.
- **`c_max`** — the prototype's tunable sound-speed dial (research report Q1: "yes,
  unambiguously" a dial). Prototype default 120 m/s; Breach's actual grid-feel constant
  is unset. Interacts with the Helmholtz solve's conditioning at the fixed sweep count
  (§3.4's docstring note: `c_max` trades sharpness, not substep count) — a genuine
  feel/performance tuning axis, not a physics constant.
- **`o2_thresh` for unit suffocation** vs. **`o2_thresh` for combustion** — the decisions
  log flags "suffocation tuning" as a separate open balancing question from combustion's
  own threshold; they may want different values (a unit can tolerate thinner air than a
  flame needs) — worth an explicit tuning pass rather than assuming one constant serves both.

---

## 10. Open questions & risks for Erik's review

Things this doc could not fully resolve on paper — flagged for the adversarial critique
and Erik, not silently decided:

1. **Is global mass-renormalization (§2.2) enough, or does gameplay need local
   conservative flux transport?** The cheap fix (rescale the whole open-air mask by one
   global ratio each tick) guarantees *aggregate* O2/N2 conservation but cannot represent
   *where* the loss/gain physically happened — a room breached at one corner should show
   O2 draining from that corner outward, and a global renormalization applied
   grid-wide could paper over local artifacts (e.g., numerical diffusion inventing O2 in
   a sealed side-room while the breached room's corner is where mass "should" have left
   from). The prototype doesn't even implement the cheap fix yet (it advects `N` as a
   plain non-conservative tracer). **This needs a decision before patch 1**, because it
   changes the per-gas transport loop's contract for exactly two species.

2. **Does `atmosphere` survive as a name/API, or is it fully retired in favor of
   `N_total`/`P`?** (§2.3.) A naming decision with real blast radius — every existing
   `gmap.atmosphere` read site (field-edit tooling, renderer, recorder, the
   `atmosphere_fixed` boundary helpers) needs to either keep working against a
   reinterpreted field or get migrated. Not resolved here; needs an inventory pass before
   patch 1 locks the field-core shape.

3. **Is velocity self-advection (§3.3) good enough, or does it need to become real
   conservative momentum `(N·u)`?** The prototype's own docstring flags this as a
   deliberate simplification chosen for stability at breach fronts, not as a
   physically-validated equivalence. It has only been exercised at 128² in the
   float-numpy spike across 5 scenarios (S1–S5) — it has not been stress-tested against
   Breach's actual abuse cases (multi-grenade stacks, an O2-tank rupture fireball, a
   Woodward–Colella-style interacting-blast scenario the Kwatra paper itself uses as a
   stability benchmark, research report §6 Q4). Recommend treating this as a specific
   thing the patch-2 gate's testing should probe, not assume from the paper's general
   stability claims.

4. **Float32→Q16.16 port risk on the Helmholtz solve is real and not fully paper-derivable.**
   The prototype's `_solve_helmholtz` docstring documents a *float32-specific*
   catastrophic-cancellation trap (the solid-face coefficient mirroring trick) that was
   found empirically, not derived first. Porting to fixed-point removes float rounding
   error but introduces integer-specific failure modes of its own (reciprocal precision
   at extreme `c_max²/N_floor` ratios, overflow margins on `beta·coef` products at
   Q16.16's range). This doc flags patch 2 as needing its own focused fixed-point
   derivation pass (§8, patch 2's gate) — it is explicitly **not** claiming the
   prototype's numerics carry over unchanged, only that the *algorithm shape* does.

5. **The `wave_p`-merge changes `apply_wave_push`'s statistical character, not just its
   scale.** Today `apply_wave_push` reads a *zero-mean* field by construction (`wave_p`'s
   design invariant) — `reduce_grad`'s footprint-edge-line-mean-difference trick and the
   "integer-exact no-op when `gx==gy==0`" fast path (engine's exchange.py docstring) both
   lean on that zero-mean property to make "quiet field" cheap and exact to detect. Under
   the merge, `P` is *not* zero-mean (it carries the standing atmospheric baseline). The
   gradient itself (`∇P`) is still well-defined and still zero in a uniform-pressure
   region, so the no-op fast path likely survives unchanged — but this should be
   explicitly re-verified against the merged field's actual behavior, not assumed by
   analogy, since it's exactly the kind of invariant a merge silently breaks.

6. **Combustion pass tick-ordering vs. multi-substep advection.** §3.1 places combustion
   *after* the full advection+acoustic tick (step 7, reading the settled `P`). But
   advection itself runs `n` substeps (up to 64, prototype's `MAX_SUBSTEPS`) inside one
   tick. Is once-per-tick combustion (matching today's `FireSimulation`, which already
   runs once per tick at full `sim_time` on the settled atmosphere, engine/06 §"Built") the
   right cadence, or should a fast-burning O2-rich pocket (the fireball emergent payoff,
   §5) see intermediate substep states? This doc assumes once-per-tick (matching current
   fire's cadence, lowest risk) but flags it as a real behavioral choice, not an
   obviously-forced one.

7. **No coarse-grid visual evidence yet for the *combined* system.** The research
   report's central gap (§9, "no ≤256² 2D real-time evidence exists") was about rung B's
   curl/mushroom-cap visual payoff in isolation; the Phase-1.2 bake-off (which produced
   the `prototypes/eos/out/*.gif` artifacts on the `eos-prototype` branch) tested the bare
   solver, not this doc's *full* stack (unified temperature + real O2 combustion + the
   downstream-consumer rewrites). The visual/feel payoff of the *complete* system as
   designed here is still unmeasured — the patch gates (§8) are the plan to close that
   gap incrementally, but it's worth naming explicitly that this design doc is a paper
   exercise on top of a partial (solver-only) prototype, not a validated full-stack spike.

8. **`inert_N2` initial density and total-N budget.** Today's single `atmosphere` scalar
   (1.0 = standard atm) has no species breakdown. Splitting it into O2 + N2 (§2.1) needs
   a real-world-inspired but game-tuned initial ratio (~21%/78% is the obvious real-air
   anchor, but Breach's `P = C·N_total·T` calibration constant `C` and the fire-ignition
   thresholds were tuned against the old single-scalar `atmosphere` — the split needs to
   preserve today's *ambient pressure and fire-ignition feel* at initialization, which is
   a calibration step, not just "use real air's ratio." Not resolved here; flagged for
   patch 1.
