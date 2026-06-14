# Physics field × system interaction map

**Status:** foundation artifact (process doc, not canon). Built 2026-06-14 from the code
(`simulation.py`, `physics_runner.py`, `cpp/src/bindings.cpp`) and reconciled against canon
(`architecture/engine/02_state_and_ownership.md`, `13_field_edit.md`).

**Why this exists.** Three patches sit ahead of us — **PhysicsEngine unification**, **fixed-point
determinism**, and the **CUDA port** — and all three are statements *about this map*: what the unified
`step()` touches, which fields must go fixed-point, and which fields go GPU-resident. The map is the
shared ground truth; it is also the first thing the expert/adversarial panel verifies. Every read/write
direction below is taken from the pybind signatures (`mutable_unchecked` = **write**, `unchecked`/const =
**read**), so it is mechanical, not remembered.

---

## 1. The two-level tick

The **game tick** (`Simulation.step`) wraps the **physics tick** (`PhysicsRunner.step`). The physics tick
is the unification target; the game tick around it is actor/logic (stays CPU per engine/02 — "GPU owns
the fields, CPU owns the actors").

```
Simulation.step()  — the GAME tick (CPU/Python; actors + structural edits)
 1  clear tick_events
 2  projectiles → on detonation: apply_explosion (enqueue FieldEdits + immediate wall_hp/destroy_wall)
                                 apply_blast_damage (units) · add_explosion_smoke (enqueue)
 3  player movement (read precomputed path)
 4  process_shooting              (RNG)
 5  zombie AI
 6  stamp_units  →  WRITES dyn_permeability, dyn_wave_absorb, obstacles, dyn_light_atten
 6b EditQueue.flush  →  WRITES smoke/atmosphere/wave_source/fire/heat  (stable-sorted, single RNG consumer)
 ── 7  PhysicsRunner.step(gmap, dt)  ───────────────────────────  the PHYSICS tick (unification target)
 │   a  cast_fire_heat        raycaster:  fire,gas,attens → heat
 │   b  _step_water           water.step (substeps) + W3 displacement + W5 boil + flooded seal (Python)
 │   c  IMEX loop ×n:         atmos.step ; then per-gas smoke.step
 │   d  _step_ripple          water.step_ripple  (visual-only)
 │   e  fire.step             → fire, atmosphere(plume), smoke, wall_hp ; returns `destroyed`
 │   f  temperature.step      heat → temperature (+conduction +cooling)
 ── ────────────────────────────────────────────────────────────
 9  destroy burn-through walls (destroy_wall)           ← from `destroyed`
 9b find_burst_walls → destroy_wall  (over-pressure relief)
 9c apply_environmental_damage   heat → unit HP
 9d apply_temperature_ignition   temperature + atmosphere → fire
 -- recorder.record · heat.fill(0)  (end-of-tick clear, after every heat reader)
 10 advance tick / phase-boundary explosives
```

Note the physics tick is **not** pure solver calls: steps **b** (W3/W5) and the per-gas loop in **c** are
real numpy physics living in the orchestrator, and the *write* of several fields (`atmosphere`, `fire`,
`smoke`) happens in **both** the FieldEdit flush (6b, Python) and the solvers (7, C++). That split is the
spine of the unification decision (§5).

---

## 2. Field inventory — writers, readers, class, home

Class ⇒ fixed-point scope: **SIM** = crosses a gameplay threshold → needs determinism (fixed-point
candidate); **RENDER** = perceptual only → stays float (canon-exempt); **STRUCT** = int/bool source of
truth; **COEF** = static/dynamic coefficient projection. Home per engine/02's CPU↔GPU seam.

| Field | dtype | Written by | Read by | Class | GPU/CPU home |
|---|---|---|---|---|---|
| `material` | int8 | `destroy_wall`/edits (CPU) | everything (caches) | STRUCT | CPU source + GPU mirror |
| `wall_hp` | f32 | fire.step, explosion (Py) | fire, destruction sweep | **SIM** (≤0 → destroy) | GPU-written |
| `is_vacuum` | bool | destroy_wall | atmos, smoke, fire, temp | STRUCT mask | GPU |
| `flammable` | bool | cache (table) | fire, ignition | COEF | GPU |
| `conductivity`/`face_shift` | f32/i32 | cache | temperature | COEF (static) | GPU |
| `atmosphere` | **f32** | atmos.step, fire(plume), **W3**(Py), FieldEdit | atmos, smoke, fire, temp, water, ignition, burst | **SIM** (O2/burst/vacuum) | GPU |
| `wave_p`,`wave_v` | **f32** | atmos.step | atmos, water, ripple | **SIM** (burst, wind) | GPU |
| `wave_source` | **f32** | atmos.step, FieldEdit | atmos.step | **SIM** | GPU |
| `wind_x`,`wind_y` | **f32** | atmos.step (= −∇p) | smoke, fire | **SIM** (derived) | GPU |
| `smoke` / `gas[N]` | **f32** | smoke.step, fire, FieldEdit, **W5**(Py) | ray march, fire | **SIM?** (see §4) | GPU |
| `fire` | **f32** | fire.step, ignition(Py), FieldEdit | fire, raycaster(heat), ignition | **SIM** (spread/dmg/I_min) | GPU |
| `solid` | bool | cache | atmos, smoke, water, temp, LoS | COEF mask | GPU |
| `permeability` | f32 | cache | (via dyn) | COEF static | GPU |
| `dyn_permeability` | f32 | stamp_units, W3 seal | atmos, smoke | COEF dynamic | GPU (units = delta-up) |
| `wave_absorb`/`dyn_wave_absorb` | f32 | cache / stamp_units | atmos | COEF | GPU |
| `obstacles` | bool | stamp_units | atmos, smoke | COEF mask | GPU (delta-up) |
| `light_atten` | f32×3 | cache | raycaster | COEF static | GPU |
| `dyn_light_atten` | f32×3 | stamp_units | raycaster | COEF dynamic | GPU (delta-up) |
| `heat` | **Q16.16 i32** | raycaster, FieldEdit | temperature, unit-dmg | **SIM — DONE** | GPU (atomicAdd) |
| `temperature` | **Q16.16 i32** | temperature.step | fire, ignition | **SIM — DONE** | GPU |
| `water_depth` | **f32** | water.step, W5, sources, FieldEdit | water, ripple, W3, boil | **SIM** (displace/boil/wade) | GPU |
| `flow_vx`,`flow_vy` | **f32** | water.step | water.step | **SIM** | GPU |
| `floor_height`,`tilt` | f32 | static/level | water.step | COEF static | GPU |
| `ripple`,`ripple_v` | f32 | step_ripple | render | **RENDER** (visual-only, canon) | GPU or render-side |
| `light_rgb`,`light_dir`,`smoke_glow` | f32 | raycaster | renderer | **RENDER** (canon-exempt) | render-only |
| `light_map` | f32 | update_from_fire | renderer (+ stealth?) | RENDER (legacy, phasing out) | render-only |

---

## 3. Per-solver read/write sets (from the pybind signatures)

| Solver call | WRITES | READS |
|---|---|---|
| `atmos.step` | wave_p, wave_v, wave_source, atmosphere, wind_x, wind_y | obstacles, solid, is_vacuum, dyn_permeability, dyn_wave_absorb |
| `smoke.step` (per gas) | gas[i] | wind_x, wind_y, sink_x, sink_y, obstacles, solid, is_vacuum, dyn_permeability |
| `water.step` | water_depth, flow_vx, flow_vy | floor_height, atmosphere, wave_p, solid, tilt |
| `water.step_ripple` | ripple, ripple_v | water_depth, wave_p, solid |
| `fire.step` | fire, atmosphere, smoke, wall_hp | temperature, wind_x, wind_y, solid, is_vacuum, flammable |
| `temperature.step` | temperature | heat, heat_inv_shift, face_shift, solid, is_vacuum, atmosphere |
| `raycaster.cast_*` (heat) | heat (+ scratch light_rgb/dx/dy) | fire-built sources, gas, gas_absorption, gas_scatter, light_atten, heat_atten |

---

## 4. Cross-field couplings (the edges, not the nodes)

These are why the orchestrator is more than solver calls — each is a field-to-field dependency that pins
tick order:

1. **fire → heat → temperature → {ignition, unit-damage}.** Fire cast (7a) deposits heat *this tick*;
   temperature.step (7f) converts it; ignition (9d) and unit-damage (9c) read the result. Order-rigid.
2. **wind = −∇(pressure).** `atmos.step` derives `wind_x/y` and the smoke advection + fire feedback read
   it — so a blast's pressure gradient fans fire and pushes smoke. Atmos must run before smoke/fire.
3. **W3 — water → atmosphere (isothermal P·V), Python.** Water depth compresses the air column;
   `_step_water` scales `atmosphere` *before* the IMEX loop so diffusion doesn't pre-smear it. Plus the
   **flooded-cell seal**: `dyn_permeability → 0` so air/smoke can't cross a submerged tile.
4. **W5 — water → steam (gas), Python.** Low-pressure water boils into `gas[white_smoke]`.
5. **fire → atmosphere (plume).** `fire.step` deposits a self-limiting own-tile overpressure (pushes
   smoke outward).
6. **burst — atmosphere differential → destroy_wall (topology).** Over-pressure relief (9b).
7. **FieldEdit flush (6b).** Grenade/explosion/laser/gas deposits land as a stable-sorted edit list into
   smoke/atmosphere/wave_source/fire/heat *before* the solvers — the canonical write primitive (engine/13).
8. **stamp_units.** Units project onto dyn_permeability/dyn_wave_absorb/obstacles/dyn_light_atten — the
   "actor → field" delta the GPU seam treats as deltas-up.

---

## 5. Fixed-point scope — the answer to "how many systems?"

Determinism rule (canon, engine/02): **fixed-point where a value crosses a discrete gameplay threshold;
float where continuous and perceptual.**

- **Already done (Q16.16):** `heat`, `temperature`. (2)
- **Definitely need conversion — 3 evolving sim-field solvers:**
  1. **AtmosphereSolver** — `atmosphere` + the wave fields (`wave_p/v/source`) + derived `wind`. Crosses
     thresholds: O2-proxy (`P_min`), burst (`burst_threshold`), vacuum.
  2. **FireSimulation** — `fire` intensity. Crosses `I_min` extinguish + ignition; spreads and damages.
  3. **WaterSolver** — `water_depth` + `flow` (the **ripple stays float** — visual-only). Crosses the
     boil threshold; drives displacement and wading.
  These three pull their **coefficient inputs** (`permeability`, `wave_absorb`, masks) into the
  fixed-point domain too (the gather `face = min(perm[self],perm[n])` must be integer-exact).
- **Borderline — 1, and it's a DESIGN CALL: smoke/gas (SmokeDynamics).** Smoke needs determinism *only if
  a gameplay decision thresholds on it.* Today `has_los` reads `solid` (walls only) — so smoke looks
  **render-only right now** and could stay float. But the **stealth-through-smoke** intent would make it
  sim-affecting → fixed-point. **This is a question for Erik + the panel, not a fact to read off the
  code.** It changes the count by one whole solver.
- **Partially done — the ray engine.** `heat` output is already Q16.16; the RGB light is render-only
  (float, exempt). The only open thread is a *scalar light-for-stealth/LoS* value if that path is used.
- **The coupling glue (Python numpy) also needs deterministic treatment:** W3 displacement ratio, W5
  boil, `stamp_units` dyn-field rebuild, and the float-field FieldEdits. Float is deterministic *on one
  machine*; cross-machine bit-identity is what fixed-point buys. These live at the unification boundary
  (§6), so **the fixed-point scope and the unification scope overlap here** — a thing to decide together.

**Headline:** **3 solvers definitely (atmosphere, fire, water), +1 by design choice (smoke), the ray
engine mostly done, + the Python coupling glue. Two (heat, temperature) already done; the render buffers
and ripple are permanently exempt.**

---

## 6. Open questions for the panel (and Erik)

1. **Smoke determinism** — is smoke sim-affecting (stealth/LoS thresholds on it) or render-only? Sets
   whether SmokeDynamics is in the fixed-point patch. *(Erik's call.)*
2. **Where the coupling glue lives** — W3/W5/`stamp_units`/FieldEdit-float are sim-affecting numpy. Do
   they move to C++ in the unification (cleaner fixed-point surface), or stay Python as CPU-side
   between-tick coupling (Erik's flexible model, but then they need deterministic fixed-point in Python)?
   This couples the **unification order** with the **fixed-point scope**.
3. **Patch order** — fixed-point-first (Erik's lean: per-solver, independent, test then CUDA) vs
   unification-first (gives one `step()` surface to convert + test fixed-point against, and settles Q2).
   The entanglement in Q2 is the deciding factor.
4. **Wind in fixed-point** — `wind = −∇p` is a derived field read by two solvers; derive it in the
   integer domain or keep a quantization boundary?
5. **Coefficient quantization** — `permeability`/`wave_absorb`/`dyn_*` are float coefficients feeding the
   gather stencils; pick their fixed-point representation (radix point) so `min`/`face` math stays exact.
