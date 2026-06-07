# State & Ownership (GameMap)

**Depends on:** [Grid & Coordinates](01_grid_and_coordinates.md)

This chapter defines how Breach represents the world's state, who owns it, and how
that ownership survives the move of hot fields onto the GPU. It is the foundation
every other engine chapter builds on: the material system, the ray engine, the
temperature pass, and the renderer all reach the world through the single contract
described here.

---

## What the system is

There is one logical owner of world state: the **`GameMap`** interface
(`src/simulation/gamemap.py`). Every system that reads or writes the world does so
through `gmap.<field>`. `GameMap` sits inside the **`Simulation`** facade
(`src/simulation/simulation.py`), which owns the rest of the world — the unit list,
projectiles, the RNG, and the turn/phase clock — and exposes the only API the
outside world uses: `apply_action`, `step`, `get_state`.

State splits into two kinds, and the split is the spine of the whole design:

- **Field state (grids).** Dense, array-shaped, touched every tick: `material`,
  `wall_hp`, `atmosphere`, the wave fields, wind, `smoke`, `fire`, the attenuation
  fields, and the ray buffers (`light_rgb`, `light_dir`, `heat`, `smoke_glow`).
  These are the GPU-resident half when CUDA lands.
- **Entity / logic state.** Small, branchy, serial: the unit list, orders,
  projectiles, the RNG, the turn/phase counters, and the pathfinding / walkability
  predicates. This stays CPU/Python, owned by `Simulation`.

> **Rule of thumb:** the GPU owns the world's *fields*; the CPU owns the world's
> *actors*.

The fields are **numerical arrays plus a material-property table** — there is **no
grid of tile objects**. This is the most important structural decision in Breach
and the one most likely to surprise a reader coming from object-oriented
tile-engine designs, so it gets its own section below.

---

## Authoritative shape: arrays + a material table, not tile objects

The structural truth of the world is a small set of numerical arrays, each shaped
to the level's grid:

- `material` (`int8`) — the material id per tile (`MAT_AIR`, `MAT_HULL`,
  `MAT_WOOD`, `MAT_DOOR`, `MAT_STEEL`, `MAT_GLASS`). This is the editable source of
  truth that structural edits mutate.
- `wall_hp` (`float32`), `is_vacuum` (`bool`) — the other pieces of structural
  state. Future `liquid_type` / `liquid_depth` join here.

**Every per-material constant lives in one place:** the `MaterialTable`
(`src/simulation/materials.py`), indexed by material id. It holds `hp`,
`flammable`, `passable`, `conductivity`, `ignition_temp`, the per-channel
`light_atten` triple, `heat_atten`, and the acoustic columns. The table is built
from the `[materials]` section of `config.toml`, so **adding a material is one
config row plus one CSV mapping in the level loader — no code change.**

Everything else a tile "has" is a **derived cache**: a projection of the table
indexed by the `material` grid (`table.hp[gmap.material]`, etc.). Caches are not a
second source of truth; they are a fast lookup recomputed from the two things that
are.

**Why no tile objects.** An object-per-tile grid is the obvious design and it is
the wrong one for Breach. Its three classic selling points — extensibility, free
serialization, mixed type+continuous state — all evaporate once a property table
exists: extensibility is a table row, serialization of arrays is trivial and far
cheaper, and the table already mixes typed and continuous columns. Worse, tile
objects actively fight the two things Breach does most: (1) **per-tick bulk field
mutation** — every physics solver is a numpy/C++ array operation, and an array of
Python objects cannot cross zero-copy into a pybind11 kernel; and (2) the **ML
pipeline**, which wants the world as tensors, not as a pointer-chasing object
graph. Arrays are the representation the simulation, the C++ physics, and the
neural net all already want.

**Rare unique-tile state** — a named door, a hackable terminal — does *not*
motivate a tile-object grid. It goes in a **sparse side-structure keyed by
`(x, y)`**: a dict over the handful of special tiles, never an object per cell.

> This supersedes the older "tile objects are the authoritative state, arrays are a
> prototype cache" framing in `architecture.md` §5. Arrays + table is the canon
> target, not a stepping stone — the tile-object layer is not coming.

---

## What a cell blocks: see the Material chapter

How a cell interacts with each system is **per-system coefficients**, not a single occlusion flag —
`light_atten` for light/vision, the wave coefficients for pressure, `permeability` for gas/smoke
flow, `passable` for movement. That model (and why `is_wall` is retired) is owned by the **Material
chapter**; State only needs its one ownership consequence: **`passable` (walkability) is a CPU-only
predicate and never goes GPU-side**, while the coefficient fields the kernels read (`light_atten`,
the permeability/wave boundaries, `obstacles`) are GPU-resident.

---

## Field inventory

Allocated up front in `GameMap.__init__`, sized from the loaded level's grid (the
CSV decides world size — not a fixed config resolution):

| Field | Type | Role |
|---|---|---|
| `material` | `int8` | Material id per tile — the structural source of truth |
| `wall_hp` | `float32` | Current wall HP (table-derived; depleted by fire/heat) |
| `is_vacuum` | `bool` | Vacuum boundary mask |
| `flammable` | `bool` | Can ignite (derived) |
| `conductivity` | `float32` | Thermal conductivity (derived; consumed by ch. temperature) |
| `atmosphere` | `float32` | Air pressure (1.0 interior air, 0.0 wall/vacuum) |
| `wave_p`, `wave_v`, `wave_source` | `float32` | Pressure-wave fields (explosions) |
| `wind_x`, `wind_y` | `float32` | Wind velocity field |
| `smoke` | `float32` | Smoke density |
| `fire` | `float32` | Fire intensity |
| `solid` | `bool` | Static solidity (`permeability <= 0`); replaces the retired `is_wall`. Movement hard-stop + LoS basis |
| `permeability` | `float32` | Static gas/smoke flux coefficient per material (0 = sealed wall, 1 = open air) |
| `dyn_permeability` | `float32` | Live flux coefficient: static `permeability` with unit footprints (partial — soft bodies) combined in, rebuilt each tick |
| `wave_absorb` | `float32` | Static per-material wave-energy damping; units add to `dyn_wave_absorb` |
| `dyn_wave_absorb` | `float32` | Live wave damping (material + unit footprints), rebuilt each tick — units absorb blasts |
| `obstacles` | `bool` | Wave/flow boundary mask (walls + unit footprints), rebuilt each tick. Now sourced from `permeability == 0`, not the old occlusion flag |
| `light_atten` | `(h,w,3) float32` | **Static** per-channel light attenuation (table projection) |
| `dyn_light_atten` | `(h,w,3) float32` | **Dynamic** attenuation: static combined with unit opacity, rebuilt each tick |
| `light_rgb` | `(h,w,3) float32` | Summed light colour reaching each tile (ray output) |
| `light_map` | `float32` | Legacy scalar light field (kept during the RGB migration) |
| `heat` | `int32` (Q16.16) | Per-tick heat deposit from the ray march — sim-affecting |
| `smoke_glow` | `(h,w,3) float32` | Light absorbed by smoke, for god-rays (render-only) |

Two distinctions in that table are load-bearing:

**Static vs. dynamic attenuation.** `light_atten` is the material table's
attenuation projected onto the grid — a structural cache, rebuilt only on a
structural edit. `dyn_light_atten` is the live field the ray march actually reads:
each tick it is reset to the static field and then unit opacity is combined in.
`total_atten = material(static) combined with dynamic(live)`. An occluder can only
*add* opacity, never remove it (the combine is a per-channel `max`).

**Fixed-point heat.** `heat` is a Q16.16 fixed-point `int32` — 16 integer bits, 16
fractional, so `1.0` energy is `65536` raw counts. It is integer specifically so
that many rays depositing into one cell is **order-independent** (integer `+=`
commutes; `atomicAdd` on the GPU is deterministic), which gives cross-machine and
future-lockstep determinism. The render-only colour buffers (`light_rgb`,
`smoke_glow`, `light_dir`) stay float: there is no downstream threshold, and
fixed-point would band the near-dark gradients and fight the HDR sum. The
principle: **fixed-point where a value crosses a discrete gameplay threshold;
float where it is continuous and perceptual.**

**In-place discipline.** The per-tick buffers (`dyn_light_atten`, `heat`,
`smoke_glow`) are allocated once and written **in place**, never reassigned. A C++
(and later GPU) view binds to the buffer's memory once; reassigning the numpy array
would silently leave that view pointing at stale memory. This is a hard rule for
any code that touches these fields.

---

## Caches and the single structural-edit seam

All derived caches are rebuilt from the table in `_update_caches`, called once at
construction. After that, structural edits never rebuild the whole grid — that
would not scale when a firestorm melts many walls in one tick. Instead every
structural edit funnels through one seam:

```text
on_tile_changed(fy, fx):
    re-read material[fy, fx]
    patch permeability/solid, light_atten, wave_absorb, flammable, wall_hp, conductivity  # O(1)
```

`destroy_wall` is the canonical caller. It sets `material` to air, calls
`on_tile_changed` to patch the static caches, and then handles the
atmosphere/vacuum semantics that are *edit-specific* and therefore the caller's job,
not the cache seam's:

- **Hull tile on the map edge** → a true hull breach: mark `is_vacuum`, and fill
  the cell with the neighbour mean of `atmosphere` rather than hard-zeroing, so the
  relaxation boundary condition drains it smoothly instead of opening with an
  artificial vacuum pulse.
- **Interior wall or interior hull** → fill with the neighbour mean of
  `atmosphere`. This preserves real pressure differentials (a wall between a
  high- and low-pressure room still produces an equalization rush) while avoiding a
  spurious vacuum spike when two equal-pressure rooms are joined.

A **config hot-reload** (`reload_material_table`) re-reads the table and rebuilds
only the table-derived caches, snapshotting and restoring the live
`atmosphere`/`obstacles` the running sim produced.

---

## Units as a per-tick world edit: `stamp_units`

Units live in `Simulation.units` — a Python list of `Unit` instances, each a live
instance of a data-driven `SpeciesDef` (definition vs. instance; base vs. effective
stats; faction owned by the mission, not the unit). They are *actors*, not fields,
so they are never baked into the grid permanently. Instead they are **projected**
onto two fields once per tick by `stamp_units`, which does two outputs in one pass:

1. **The gas/wave coefficient fields** — `dyn_permeability` and `dyn_wave_absorb`, plus the boolean
   `obstacles` mask, all read by the wave and smoke solvers. A living unit is now a **soft** patch,
   not a full wall: its footprint gets a *partial* `dyn_permeability` (default 0.5, per-unit hook +
   `[physics] unit_permeability`), so smoke and air **seep past a body** rather than reflecting off
   it, and it adds to `dyn_wave_absorb` (`[physics] unit_wave_absorb`) so it **absorbs blasts**
   instead of mirroring them. Units still stamp into `obstacles` (movement hard-stop) and cast light
   shadows. The C++ atmosphere + smoke solvers gather flux via `face = min(perm[self], perm[neighbor])`
   over `dyn_permeability`; with the current materials the behaviour is identical to the old boolean
   boundary. (Wave transmission *through* walls — 4b — is still deferred.)
2. **`dyn_light_atten`** = the static `light_atten` with each living unit's opacity
   combined in per channel (`max`). A unit's opacity comes from an optional
   `unit.light_atten` (default `[1,1,1]` = a full shadow), so the design already
   admits a creature that passes green light or an aquarium that tints — for free,
   because the field is RGB.

`stamp_units` reads footprints through `unit.occupied_tiles()`, depending only on
the footprint contract, not on any storage representation. When a tile transitions
from blocked to free (a unit moved away or died), it is filled with the neighbour
mean of `atmosphere` — the same anti-vacuum-pulse rule `destroy_wall` uses.

---

## Ownership and the CPU↔GPU seam

The logical owner is the `GameMap` interface; the **physical home of each field is
per-field and migratable.** Callers always reach state through `gmap.<field>` and
**must never assume who allocated a field or where it lives.** This single
discipline makes the eventual GPU migration a localized, mechanical change *inside*
`GameMap` — no caller changes.

When CUDA lands, the split is:

- **Live physics fields** → GPU-resident; the GPU copy is authoritative and the CPU
  takes read-only snapshots.
- **`material`** → CPU keeps the editable source of truth (`destroy_wall` runs
  there) and pushes deltas to a GPU mirror the kernels read.
- **`wall_hp`** → GPU-resident and GPU-written (depleted on-GPU by thermal failure
  and explosions); CPU edits issue a delta and read it back in the field snapshot.
- **Entity/logic state** → stays CPU: branchy and serial, not array-shaped.

The C++ holds `GameMap`'s grids alongside the stateless solvers — "GameMap owns the
fields" and "the physics engine holds the grids" unify, because the physics engine
*contains* the grid owner. The C++ solvers are already stateless and bind to the
numpy arrays zero-copy (see `PhysicsRunner.step`, which passes `gmap.wave_p`,
`gmap.obstacles`, etc. straight into the pybind kernels), so the model is already in
place on the CPU; CUDA only moves the memory.

**The per-tick seam (cheap):**

- **Up — tiny deltas:** moved unit footprints rebuild `obstacles` and re-stamp
  `dyn_light_atten`; destroyed walls patch `material` / `wall_hp`, one delta each.
- **Down — one snapshot per frame:** the field buffers, once, serving both the
  renderer and any CPU-side field reads (LoS, stealth, damage sampling).

**Freshness split.** The once-per-frame download may be **one tick stale for the
renderer** — invisible on screen. The **simulation must read current-tick values**:
its field reactions (ignition, wall failure, unit heat damage) read what the GPU
just computed, on-GPU, before any staleness is introduced. Headless training has no
"frame" and always reads current-tick values. *Render may read stale; the sim never
does.*

VRAM is not the constraint. The full field set is ~100–150 B/tile — about 17 MB at
240×480, ~150 MB at 1000×1000 (a few percent of an 8 GB card). The limit is
compute/bandwidth, not memory; even hundreds of parallel training instances fit.

---

## Render buffers are conditional

The ray buffers split by nature. **`heat` is a simulation field — always
allocated** (the sim reads it). The **visual byproducts — `light_rgb`,
`light_dir`, `smoke_glow` — exist only when a renderer is attached.** Headless
training computes `heat` (plus a scalar light intensity for stealth/LoS) and skips
the render-only colour channels. This is a *flag* ("produce render byproducts?"),
**not** a sim→render dependency: the sim never reads a render buffer. (See the ray
engine and lighting chapters for the full split.)

---

## The Simulation facade: the one mutation point

`GameMap` holds the fields; `Simulation` owns everything else and is the **only**
place game state is mutated. The renderer reads `sim.get_state()` — a lightweight
`SimState` snapshot holding *references* into the live arrays, never deep copies —
and never writes back. AI rollouts call `sim.step()` in a loop and never touch
pause.

`get_state()` returns `gmap`, `units`, `projectiles`, `tick`, `phase`, `paused`.
The tick loop in `step()` has a load-bearing order: clear events → projectiles →
player movement → shooting → zombie AI → **re-stamp obstacles** → physics step →
phase/round-boundary explosives → process fire burn-through walls → advance tick.
The `stamp_units` call sits before the physics step precisely so the solvers see
the new unit positions this tick.

**Serialization and replay.** World state serializes trivially because it is
arrays: `np.save` the field set plus the entity/logic state. Replay is **initial
seed + input/command log**, not per-tick state — and it is reproducible
cross-machine because the sim is fixed-point deterministic where it matters (the
`heat` deposit, the temperature field). A single seeded `numpy.random.Generator` on
`sim.rng` is plumbed through every known nondeterminism site, so the same seed plus
the same inputs gives a bit-identical trajectory on any machine.

---

## Implementation status

**Built and matching this chapter:**

- Arrays + `MaterialTable` as the authoritative shape — no tile objects anywhere.
  `material` is `int8`; all per-material constants come from the table
  (`src/simulation/materials.py`); caches are table projections built in
  `GameMap._update_caches`.
- Walkability predicate `is_passable` / `is_passable_block` (`AIR`/`DOOR`), CPU-only.
- **`is_wall` retired.** `GameMap.solid` (= `permeability <= 0`) replaces it everywhere in Python;
  the flow boundary and `has_los` now read `solid`. The C++ solvers keep a now-vestigial `is_wall`
  *parameter* (fed `gmap.solid`) — removing that param + rebuild is a pending follow-up.
- **Per-material `permeability` column** built and consumed: `GameMap.permeability` /
  `dyn_permeability` caches; the C++ atmosphere + smoke solvers gather flux via
  `face = min(perm[self], perm[neighbor])`. Behaviour-identical for the current materials.
- **Soft units.** `stamp_units` writes unit footprints as a *partial* `dyn_permeability` (default
  0.5, per-unit hook + `[physics] unit_permeability`) so gas/air seep past a body; units still
  hard-stop movement and cast light shadows.
- **Units absorb blasts.** `wave_absorb` / `dyn_wave_absorb` caches (material `wave_absorb` + units
  via `[physics] unit_wave_absorb`); the C++ wave update damps per cell by it. Energy-out only —
  open air is bit-identical.
- **Over-pressure wall failure.** `MaterialTable.burst_threshold` column +
  `GameMap.find_burst_walls(max_pops)`; `Simulation.step` (after fire burn-through) destroys walls
  holding a pressure differential above their `burst_threshold`, capped by `[physics]
  burst_max_per_tick`, gated by `[physics] burst_enabled`.
- The full field inventory above is allocated in `GameMap.__init__`, sized from the
  level grid.
- Static vs. dynamic attenuation (`light_atten` / `dyn_light_atten`), with the
  per-channel-`max` combine and the per-unit `light_atten` opacity hook, rebuilt in
  `stamp_units`.
- Fixed-point Q16.16 `heat` (`int32`), float render buffers, and the
  in-place-write discipline (buffers allocated once, written via `[:]` / indexed
  assignment, never reassigned).
- The single structural-edit seam `on_tile_changed`, with `destroy_wall` owning the
  hull-breach / neighbour-mean atmosphere semantics; `reload_material_table` for
  config hot-reload.
- `Simulation` as the sole mutation point; `get_state()` returning a reference-only
  `SimState`; the load-bearing tick order with `stamp_units` before the physics
  step; seeded `sim.rng` plumbed through the nondeterminism sites.

**Designed, not yet built:**

- **GPU residency and the CPU↔GPU seam.** Today every field is a CPU numpy array
  and the C++ solvers bind to them zero-copy via `PhysicsRunner`; nothing is
  GPU-resident yet. The ownership contract (`gmap.<field>`, per-field migratable
  home, deltas-up / snapshot-down, the render/sim freshness split) is fixed so the
  migration is internal to `GameMap`, but the migration itself is future work.
- **GPU-written `wall_hp` via thermal failure.** `wall_hp` is currently depleted on
  the CPU by the C++ fire solver (`PhysicsRunner.step` returns burn-through tiles
  the `Simulation` feeds to `destroy_wall`). The "temperature crosses a material
  threshold → HP depletes on-GPU" path belongs to the temperature chapter and is
  not built.
- **Render-conditional allocation.** The contract says headless skips the
  render-only colour buffers, but `GameMap.__init__` currently allocates
  `light_rgb` / `light_dir` / `smoke_glow` unconditionally. Harmless today (they are
  small and the headless path simply does not write them), but the conditional-flag
  allocation is not yet wired.
- **Save/load and replay.** `get_state()` is shaped to be pickle-friendly and the
  seed+input-log replay model is fixed, but no save/load or replay code exists yet.

**Gaps / loose ends:**

- **`light_dir` is referenced but not allocated.** The ownership and ray-engine
  designs list `light_dir` (the per-tile light-direction field the shader reads) as
  part of the buffer set, but `GameMap` does not allocate it yet — it lands with the
  ray-engine work that produces it.
- **Legacy scalar `light_map` coexists with `light_rgb`.** Kept deliberately during
  the RGB migration (fire raycaster output, unit/smoke tinting). It is redundant
  once the RGB path fully replaces it and should be removed then.
- **`level_loader` / level-editor table conformance.** The level pipeline was
  prototyped before the mature material table and partly reinvents it; bringing it
  fully onto the table (a new material map plus minor glue) is outstanding.
- **Sparse side-structures for unique tiles** (named doors, terminals) are specified
  but not yet present — no level needs one today.
- **`fy/fx` naming.** Some `GameMap` methods still take `(fy, fx)` parameters (e.g.
  `on_tile_changed`). The canon convention (Grid chapter) is `(row, col)` for indices and `(x, y)`
  for tile/metre coords; a rename to match is owed — cosmetic, not behavioural.
