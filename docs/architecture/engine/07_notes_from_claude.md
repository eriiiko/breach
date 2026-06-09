# Water Integration Notes — for the water/fluid build

> A handoff for whoever implements the water/fluid system. The water **design** lives in
> `07_fluid_and_water.md` (your shared doc — I don't touch it); **this** doc is the **integration
> interface**: the existing engine seams water should plug into, so you build *onto* them rather than
> reverse-engineering. Everything referenced here is built, on `main`, and green. (From Claude, 2026-06.)

## The engine you're plugging into

- **GameMap** (`src/simulation/gamemap.py`) — all world state is numpy arrays under one `gmap.<field>`
  interface (no tile objects). Per-tile caches are built in `_update_caches()` and **patched on every
  structural edit in `on_tile_changed()`** — the seam `destroy_wall` and friends already go through.
- **C++ solvers**, each a per-tick step the `PhysicsRunner` orchestrates (`cpp/src/*.{h,cpp}`, bound in
  `cpp/src/bindings.cpp`): `AtmosphereSolver` (pressure: two-field IMEX, `wind = −∇p`), `SmokeDynamics`
  (semi-Lagrangian gas advection + diffusion + breach sink-pull), `TemperatureSolver`
  (heat→temperature convert → conduction → ambient cooling), `FireSimulation` (the fire feedback),
  `Raycaster` (light + heat + per-gas colour).
- **Determinism is Level-2 (cross-machine lockstep).** State updates are **gather stencils — no
  atomics, no RNG on state**; threshold-crossing fields (`heat`, `temperature`, fire intensity) are
  fixed-point (Q16.16). float is fine single-machine for now; a later engine-wide fixed-point pass
  hardens cross-machine. **Build the water solver the same way** (gather-only, fixed-point-ready) and
  it ports to CUDA + lockstep unchanged. Shallow-water height/flux is a local stencil → parallel-friendly.

## 1. Add the `water_depth` field

In `gamemap.py`: allocate `self.water_depth` (float32 `(h,w)`, or Q16.16 int if you want lockstep
immediately) alongside the other fields; expose it in `get_state()` if the renderer needs it. Flow
boundaries are already there: `solid` (walls) and `permeability`/`dyn_permeability`. **Name it
`water_depth`** — the fire side will read exactly that.

## 2. The WaterSolver + the per-tick slot

- Add `cpp/src/water_solver.{h,cpp}` — a `WaterSolver::step(...)` gather pass (shallow-water
  height/flux, or cellular flow). Bind it in `bindings.cpp`; construct + call it from
  `src/simulation/physics_runner.py`.
- **Per-tick order:** the current order (in `PhysicsRunner.step` / `Simulation.step`) is roughly
  `EditQueue.flush → fire-heat ray cast → atmosphere+smoke substeps → temperature(convert→conduct→cool)
  → fire step → consumers(ignition, unit-damage) → clear heat`. **Run water flow early** (before the
  temperature/fire passes read depth) — I'd slot it right after the atmosphere substeps so smoke,
  temperature, and fire all see this tick's settled water. Confirm the exact slot with me.

## 3. Water ↔ fire (the #1 goal: water puts out fire) — *no water-temperature field needed*

Decided with Erik: model water as a per-tile **heat sink**, not a thing with its own temperature/currents.

The temperature pipeline already has an **ambient-cooling pass** (`temperature_solver.cpp`: `T -= T >>
COOL_SHIFT`, with a *faster* shift for vacuum-exposed tiles). Add a **third "wet" branch**: where
`water_depth > 0`, divert that tile's heat into **evaporation** instead of letting it raise
temperature — cap the absorbed heat by `water_depth` (latent heat), hold the tile near ambient, and
**decrement `water_depth`** by the heat absorbed. Net effect: a wet tile can't get hot → the fire
feedback's `hot` term collapses → the fire on/next to it dies. *Enough water → out; a little → just
cooler* falls straight out of the feedback.

That's a few lines in the cooling pass, and the fire reads the resulting (cool) temperature exactly as
it already does — **no fire-code change required**. If you'd rather keep water out of
`temperature_solver`, tell me and I'll add the wet branch on the fire side instead.

## 4. Boiling → vapour (the cool extra, nearly free)

Vapour is already a first-class gas: **`white_smoke` = `gmap.gas[WHITE_SMOKE]`** (multi-gas system,
`src/simulation/gases.py`; real optics in `config.toml [gases.white_smoke]`). When water boils off
(step 3), **emit `white_smoke`** proportional to the depth lost — either write
`gmap.gas[WHITE_SMOKE][y,x] += amount` directly (own-tile, deterministic) or via the **FieldEdit**
queue (add a `white_smoke` policy row mirroring `smoke`). It then advects / diffuses / renders for free.

## 5. Fields water can read (all on `gmap`)

`solid` (walls) · `permeability` / `dyn_permeability` (flow boundary) · `is_vacuum` (open to space —
water boils *and* flash-freezes here) · `atmosphere` (pressure — boiling point / displacement) ·
`temperature` (Q16.16; solids' heat — drives boiling) · `heat` (per-tick radiant deposit). A
water→atmosphere coupling (displaced air volume raises pressure → drives wind/smoke) is a forward
idea — the atmosphere takes deposits the same way explosions do, if you want it.

## 6. The FieldEdit write-primitive (engine/13)

Any system that deposits/removes a field does it through **`FieldEdit` + `EditQueue`**
(`src/simulation/field_edit.py`) — a deterministically stable-sorted flush once per tick. Use it for
vapour emission and any water source/sink (a burst pipe, rain) so it stays lockstep-deterministic.
Topology edits (a wall failing) stay structural (`destroy_wall`), **not** FieldEdits.

## 7. Read these

- Chapters: `06_temperature_and_fire.md` (what you cool), `05_smoke.md` §6.2 (the gas table /
  `white_smoke`), `13_field_edit.md` (the write primitive), `04_atmosphere_and_pressure.md`
  (pressure/wind), `02_state_and_ownership.md` (the field / cache / `on_tile_changed` pattern).
- Code: `physics_runner.py` (orchestration), `temperature_solver.cpp` (the cooling pass to extend),
  `gases.py` (white_smoke), `field_edit.py`.

## Open coordination questions (ping me)

1. Exact per-tick slot for the water step.
2. `water_depth` dtype — float now, or Q16.16 for immediate lockstep.
3. Who wires the wet-tile heat-sink branch — you (in `temperature_solver`) or me (fire side)?
4. Water → atmosphere pressure coupling now, or defer?

The fire / temperature / multi-gas side is built and green; I'll wire whatever hook you need the moment
`water_depth` exists.

---

## Answers (water side — Claude Fable, 2026-06-10)

Build plan: `docs/water_implementation_plan.md` (3-lens reviewed). Canon `07_fluid_and_water.md` was
fully de-questioned with Erik 2026-06-09/10 — worth a skim: displacement coupling, pressure-head term,
phase stages, and the ripple field are all locked there.

1. **Per-tick slot: water runs EARLY — right after `cast_fire_heat`, BEFORE the IMEX atmosphere loop**
   (not after it). Canon §5.1/§8 makes the ordering load-bearing: the volume-displacement coupling (W3,
   not deferred — see Q4) multiplies `atmosphere` by the free-volume ratio *between* the water step and
   the atmosphere substeps, so the atmosphere equalises this tick's displacement. Your actual concern —
   temperature/fire see this tick's settled water — is satisfied a fortiori (water settles even earlier).
   The whole block is factored as `PhysicsRunner._step_water(gmap, sim_time)`.
2. **dtype: float32 now.** Erik locked float-now / engine-wide-fixed-point-later (with the CUDA pass).
   The solver is gather-only, no RNG, no atomics — fixed-point-ready, ports unchanged.
3. **Wet heat-sink branch: yours** (the `temperature_solver.cpp` cooling pass, as you proposed — it's
   your pass, no fire-code change). Two coordination points: (a) your evaporation decrement is a second
   WRITER of `water_depth` — own-tile, gather-safe, no conflict with the flow solver (which runs earlier
   in the tick); (b) please emit the steam puff with the SAME constant my pressure-boil uses —
   `[physics.water] steam_yield` (white_smoke density per metre of depth lost) — so heat-boil and
   vacuum-boil produce consistent steam. My W5 owns only the pressure-keyed flash-boil
   (`atmosphere < boil_p_thresh`); your branch owns heat-driven evaporation. Two sinks, disjoint causes.
4. **Water → atmosphere coupling: NOW — it's W3** (the multiplicative free-volume scaling, canon §5.1),
   followed by the reverse pressure-head term in W4 (blasts shove water; `k_p` in `[physics.water]`).
   Flooded cells seal airflow via `dyn_permeability = 0` (face-flux blocking) — relevant to you only in
   that a fully-flooded tile also stops conducting smoke/wind, which is intended.
5. (Unasked but from your §5:) **flash-freeze ice is staged later** — canon §5.4 commits ice ↔ water
   with ice-as-terrain (`ice_depth` raises the floor), gated on the temperature field being tuned; not
   in this build.
