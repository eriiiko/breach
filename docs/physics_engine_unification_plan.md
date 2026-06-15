# PhysicsEngine unification — plan (draft for panel review)

**Status:** draft, pre-panel. The first of the CUDA-prep patches. Grounded in
`docs/physics_field_interaction_map.md`, `docs/resolution_architecture_proposal.md`, and the locked
decisions (D-A uniform grid, D-C full fixed-point cross-GPU). **Scope is float-and-CPU only** — this
patch changes *structure*, not arithmetic; the Q16.16 migration is a separate later patch that overlays
onto the surface this one creates.

---

## 1. Goal & scope

**Goal:** collapse the eight separate C++ solvers + the Python orchestration glue into **one C++
`PhysicsEngine` that owns the grids and exposes a single `step(dt)`** — the migration boundary the CUDA
port plugs into. Per engine/02: *"the physics engine **contains** the grid owner,"* so this is not a new
owner competing with `GameMap` — `GameMap` stays the interface (`gmap.<field>`), and the engine becomes
the physical home those fields will later migrate onto the GPU from.

**In scope:**
- A C++ `PhysicsEngine` holding the solver instances + binding the grids; one `step(dt)` entry point.
- Move the **field→field glue** now in Python (`physics_runner.py`) into C++: W3 displacement, W5
  flash-boil, the per-gas transport loop, the substep derivation, the fire heat-source build.
- The **coherent dt policy** (§6) — decouple the four overloaded substep needs, incl. the folded
  subcycle-when-active and the sink-pull-cap decoupling.
- **`stamp_units` → C++**, delta-based (§7).
- The **shared stencil** consolidation (§8).
- **Cleanups:** remove the vestigial `AtmoDiffusion`; `dt_scale → advection_rate` (§9).
- **Fixed-point-aware structure** (§10) so the later Q16.16 overlay is mechanical.

**Out of scope (explicitly):** the Q16.16 arithmetic migration; GPU residency / CUDA kernels; the
resolution change (uniform `base/k`); per-field format tuning. Those are downstream patches.

**Guiding constraint:** **behavior test-identical where the change is structural** (Phases A, B, E, the
cleanups), and **behavior-changed-with-feel-test where the change is the dt policy** (Phase C) — flagged
explicitly, never silent. The 336-test suite is the regression guard; the determinism and
conservation/dormancy tests are the hard gates.

---

## 2. Grounding — where physics lives today

The **game tick** (`Simulation.step`, Python — actors, FieldEdit flush, structural edits) wraps the
**physics tick** (`PhysicsRunner.step`, Python — the unification target). The physics tick is *not* pure
solver calls: real numpy physics lives in the orchestrator —
- `cast_fire_heat` (builds the row-major fire source list, calls the raycaster per source → `heat`),
- `_step_water` (W3 isothermal displacement onto `atmosphere`, W5 flash-boil → steam, the flooded-cell
  `dyn_permeability` seal),
- the **per-gas loop** (binds each gas's diffusion onto the shared solver, steps each slice),
- the **substep derivation** (`n = ceil(sim_time / atmos.max_dt())`),
- `_step_ripple` (visual-only).

The eight solvers (`atmosphere`, `smoke`, `fire`, `temperature`, `raycaster`, `water`, + the vestigial
`atmo_diffusion`, `wave_solver`) are **stateless** and bind to the numpy arrays zero-copy. Per engine/02:
*"the model is already in place on the CPU; CUDA only moves the memory."* So the unification is mostly a
**re-home + re-sequence**, not an algorithm rewrite.

---

## 3. The target structure

```
Simulation.step()  [Python — actors, FieldEdit flush, destroy_wall, unit damage, ignition, heat clear]
   │  (unchanged except: one call instead of the inline PhysicsRunner orchestration)
   ▼
PhysicsEngine.step(gmap_fields, dt)   [C++ — owns solvers + the per-tick order + the field→field glue]
   ├─ cast_fire_heat            (fire sources → heat)         ← moved into C++
   ├─ step_water + W3 + W5 + seal                              ← moved into C++
   ├─ dt-policy: derive substeps (§6)                          ← new, C++
   ├─ for each substep: wave+diffusion ; per-gas smoke         ← shared stencil (§8)
   ├─ step_ripple              (visual-only)
   ├─ fire feedback            → returns destroyed walls
   └─ heat→temperature+conduction+cooling
```

`GameMap` keeps allocating + owning the field arrays (this patch); the engine **binds** to them (today's
zero-copy pybind pattern, one bind at construction, honoring the in-place-write discipline). The
`gmap.<field>` read interface is **untouched** — no caller changes, exactly as engine/02 requires. (When
GPU residency lands later, only the *home* of those arrays moves, inside `GameMap`/the engine; callers
still don't change.)

---

## 4. What moves to C++, what stays Python

| Logic | Today | Target | Why |
|---|---|---|---|
| Per-tick solver order | Python (`PhysicsRunner.step`) | **C++** | the one `step(dt)` boundary |
| W3 displacement, W5 boil, flooded seal | Python numpy | **C++** | field→field physics; the stable foundation |
| Per-gas transport loop | Python | **C++** | field→field; batched stencil on GPU later |
| Substep derivation / dt policy | Python | **C++** | §6, the coherent policy |
| Fire heat-source build | Python (row-major + per-source cast) | **C++** | field-driven; `update_from_fire` C++ path already exists |
| `stamp_units` field writes | Python full-field rebuild | **C++ (delta)** | §7 |
| **Reading the `Unit` list** (footprints, opacity) | Python | **stays Python** | actors are CPU/Python (engine/02); the engine gets a *footprint delta list*, not the unit objects |
| FieldEdit enqueue (weapons/fire) | Python | **stays Python** | actor-driven writes; the flush already lands before physics |
| `destroy_wall`, burst, ignition, unit heat-damage | Python (`Simulation.step`) | **stays Python** | topology + actor/threshold logic; engine/13 carve-out |

The split rule (engine/02): **field→field → C++; actor→field → the boundary stays Python, only the
field-write half is C++** (the engine receives a delta list of footprint coefficient-writes, computed
Python-side from the unit objects).

---

## 5. Phased build (each phase: implement → build → pytest → commit/revert, per the house workflow)

- **Phase A — the skeleton (behavior test-identical).** Create the C++ `PhysicsEngine` class holding the
  existing solver instances; move the per-tick *order* into its `step(dt)`, still calling the same solver
  methods on the same `gmap` arrays. Python `PhysicsRunner` becomes a thin shim that calls
  `engine.step(...)`. **No physics changes** — the bytes match. Gate: full suite green incl. the
  determinism + dormancy bit-identity tests; both smoke `--auto` runs exit 0.
- **Phase B — glue → C++ (behavior test-identical).** Port W3/W5/the flooded seal/the per-gas loop/the
  fire-source build into the engine, one at a time, each behavior-preserving (the existing
  conservation + dormancy tests are the guard). The numpy and C++ must agree bit-for-bit in float
  (verify with the dormancy-style A/B harness).
- **Phase C — the coherent dt policy (BEHAVIOR CHANGE — feel-test gated).** §6. This is the one phase
  that changes the felt result (calm-room substep collapse, the sink-pull decoupling). Land behind a
  config flag, **Erik feel-tests** the calm air + breach-venting before it becomes default. New
  regression tests pin: venting still clears, sealed room still conserves, the GS residual stays bounded.
- **Phase D — shared stencil (behavior test-identical or near).** §8. One Laplacian/gather kernel
  parameterized by field+operation, serving wave + atmosphere-diffusion + smoke-diffusion. Bit-identical
  if the stencil math is preserved; if consolidation changes a rounding/iteration detail, treat as a
  flagged behavior change.
- **Phase E — `stamp_units` → C++ delta (behavior test-identical).** §7.
- **Cleanups** (anytime after A): remove `AtmoDiffusion` (vestigial — `PhysicsRunner` uses the IMEX
  `AtmosphereSolver`, not it); `dt_scale → advection_rate` (§9, behavior-preserving fold or flagged).

Phasing rationale: **A and B are pure structure** (safe, test-identical) and establish the surface; **C
is the only real behavior change** (isolated, feel-tested); **D/E are optimizations** on the established
surface. The panel should challenge whether C belongs in this patch at all or should be its own.

---

## 6. The coherent dt policy (folds in the subcycle finding)

Today **one substep count `n = ceil(sim_time / wave_CFL)` is overloaded** across four needs (the
2026-06-15 build finding): (i) the explicit wave CFL, (ii) implicit diffusion stability (wants 1), (iii)
the smoke explicit-diffusion CFL, (iv) the **sink-pull drain rate** (capped 1 cell/substep, so coupled
to `n`). Decouple them:

- **Wave:** substep at its CFL **only when active** (`wave_source`/`wave_p`/`wave_v` above an ε dial).
  When inactive, **snap the sub-ε acoustic field to exactly 0** (so the explicit kick can't amplify a
  residual at the big dt) and take **one** big implicit-diffusion step.
- **Implicit diffusion:** one step (unconditionally stable). Instrument the **post-sweep GS residual**
  here — the measurement that settles Decision 6 (two-grid correction yes/no). *(Needs a small C++ hook
  to expose the residual.)*
- **Smoke diffusion:** its own CFL floor (`dt < dx²/(4·d_eff)`, `d_eff = d_smoke·(1+wds·|wind|²)`) —
  ≈1 at rest, tightens only under strong wind (where the wave already dominates).
- **Sink-pull drain — the key decouple:** lift the **1-cell-per-substep cap off the substep count.** Make
  the drain a **tuned rate constant** (cells/second) realized independent of `n` — e.g. a per-tick drain
  budget applied in one pass, or a cap expressed per-`dt` not per-substep. Then a calm venting room
  drains at the same feel with `n=1`, and the wave-CFL number stops secretly setting the venting speed.

Net substep count: `n = max(n_wave_if_active, 1, n_smoke_cfl)` — and the sink-pull no longer contributes
to `n` at all. **Open for the panel:** the exact sink-pull decoupling (per-tick budget vs per-dt cap),
and whether the smoke-diffusion floor should instead become *implicit* smoke diffusion (the fine-res
fix) now or later.

---

## 7. `stamp_units` → C++, delta-based

Today (`gamemap.py`): full-field reset (`dyn_permeability[:] = permeability`, ×3 for `dyn_light_atten`,
+ `dyn_wave_absorb`) **every tick**, then a Python loop over unit footprints. The reset is the dominant
cost and the farm-killer (whole fields re-uploaded per tick). Target:
- The engine holds the static base caches; per tick it receives a **delta list** from Python: for each
  unit that *moved*, `(cleared_footprint_cells, new_footprint_cells, coefficients)`; for each *destroyed
  wall*, the patched tile. Python still reads the `Unit` objects (actors stay CPU) and emits the delta;
  C++ applies it — **only touched cells change**, no whole-field reset.
- The unit's stamped coefficients: `dyn_light_atten` (RGB opacity), `dyn_permeability` (0.5 default),
  `dyn_wave_absorb`. The `obstacles` mask stays walls-only (units are soft).
- This *is* engine/02's "deltas-up" seam, pre-built on the CPU so the GPU port inherits it.

**Open for the panel:** the clear-old-footprint bookkeeping (track each unit's previous footprint;
handle death/spawn/teleport); whether a moved-this-tick set is cheaper than diffing footprints.

---

## 8. Shared stencil

Wave, atmosphere-diffusion, and smoke-diffusion each re-implement their own 2D grid traversal. Per the
CUDA plan §7, consolidate into **one stencil/gather kernel parameterized by field + operation** (the
`face = min(perm[self], perm[n])` gather is already shared in spirit). On CPU a modest win; on GPU it's
the difference between one well-occupied kernel and several mediocre ones. Must preserve the per-system
boundary handling (Neumann at walls, the `dyn_permeability` face gather).

---

## 9. Cleanups

- **Remove `AtmoDiffusion`** — `PhysicsRunner` uses the IMEX `AtmosphereSolver`; `AtmoDiffusion` (separate
  explicit Euler) is unreferenced. Delete + rebuild; no behavior change.
- **`dt_scale → advection_rate`** — express "how hard smoke rides the wind" as the `advection_rate`
  coefficient on the real dt, drop the timestep-faking `dt_scale` (the ~6.25× double-scale). Likely
  behavior-preserving (`new advection_rate = old advection_rate × old dt_scale`) **iff** `dt_scale` only
  scaled advection — verify against `smoke_dynamics.cpp` whether it also scaled diffusion; if so,
  separate the two and re-tune diffusion with Erik's eye.

---

## 10. Fixed-point-aware structure (prep, not the migration)

Structure the C++ so the later Q16.16 overlay is mechanical, without doing it now:
- **Isolate the arithmetic** behind typedef'd scalars + inline ops (`add/mul/div/scale`), so swapping
  `float → fixed` is a type change, not a rewrite. (The `heat`/`temperature` Q16.16 lane is the existing
  template.)
- **Pin reduction sites** (e.g. `mean_wp`) behind a single helper, so the float→integer (order-
  independent) swap is one place — and flag that `mean_wp` is a *known* determinism hazard today.
- **No new float-only idioms** in the moved glue (avoid FMA-dependent expressions, transcendentals where
  avoidable) so the migration doesn't fight the structure.
- Keep render-only buffers (`light_rgb`, `smoke_glow`, `ripple`) clearly separated — they **stay float**
  forever (not sim state, never synced).

---

## 11. Gates

- **Bit-identity** (Phases A, B, D-if-claimed, E): the dormancy-style A/B harness (run old vs new on the
  same seed, assert 0-ULP field deltas) — extend it to the unified path.
- **Determinism:** the existing same-seed determinism test stays green (per-machine). *(The cross-machine
  two-seeded harness is owed to the fixed-point patch, not this one.)*
- **Conservation/dormancy:** the sealed-hull conservation + water/smoke dormancy tests are the
  non-negotiable guards.
- **Phase C:** new tests pin venting-clears, sealed-conserves, GS-residual-bounded; **+ Erik feel-test.**
- Both `tests/test_main_smoke.py --auto` runs (default + `unhcr_vessel_2`) exit 0 each phase.

---

## 12. Open questions for the panel

1. **Does Phase C (dt policy) belong in this patch**, or should the unification be pure structure (A, B,
   D, E) and the dt policy + sink-pull decouple be its own feel-test-gated patch right after? (Lean:
   split it out — keep the unification test-identical, isolate the one behavior change.)
2. **The sink-pull decouple mechanism** — per-tick drain budget vs per-`dt` cap? Which preserves the
   venting feel with `n` decoupled?
3. **Glue ownership granularity** — does the fire heat-source build move to C++ now (it's field-driven
   but currently Python-row-major-deterministic), or stay Python until the raycaster port?
4. **Shared-stencil scope** — consolidate now (Phase D) or defer to the CUDA port where it actually pays?
   Risk of churn vs the determinism guarantee.
5. **`stamp_units` delta vs keep-full-rebuild-but-move-to-C++** — is the delta bookkeeping worth it on
   CPU, or does the C++ full-rebuild suffice until the GPU port forces deltas?
6. **Fixed-point-aware vs fixed-point-now** — is structuring-for-later enough, or does touching every
   solver twice (unify-float then migrate-fixed) waste enough effort to justify doing them together?

---

## 13. Risks

- **Glue→C++ float bit-identity** — the W3/W5 numpy math must reproduce exactly in C++ (float order, the
  `np.where`/clip idioms). Mitigate: port one coupling at a time behind the A/B harness.
- **Phase C feel regression** — the dt policy changes calm-room and venting dynamics; the whole reason
  it's feel-test-gated behind a flag.
- **Touching solvers twice** (unify-float, then fixed-point) — accepted for reviewability unless the
  panel argues to bundle (Q6).
- **Scope creep** — six sub-systems in one patch. The phasing + the "split Phase C out" option (Q1) are
  the controls; the panel should prune hard.
