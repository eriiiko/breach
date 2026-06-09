# Material System

**Depends on:** [grid](01_grid_and_coordinates.md), [state](02_state_and_ownership.md)

Every tile in Breach is made of a **material**: air, hull, wood, door, steel, glass.
A material is not an object — it is a small integer id stored in the `material` grid, plus a
row in a **property table** that maps that id to every physical constant the tile needs.
This is the foundation the rest of the engine stands on: the ray march, the temperature
field, the wave solver, pathfinding, and destruction all derive their per-tile behaviour by
indexing the same table with the same `material` grid.

The design has one governing rule: **no per-tile objects**. World state is numerical arrays
(`material`, `wall_hp`, the physics fields) and the material-property table. A tile's
properties are never stored on the tile; they are looked up from the table. Adding a material
is one config row and one CSV-code mapping — never a code edit to a propagation system.

## The property table

The table lives in `config.toml` as a **named-key dictionary**, one `[materials.<name>]`
section per material. The section name is the material's config key; its position in id order
(declared in `simulation.materials.MATERIAL_NAMES`) gives it its integer id. The shape:

```toml
[materials.hull]
hp            = 300          # structural HP; 0 = not a destructible wall
flammable     = false        # can ignite
passable      = true|false   # unit walkability (independent of occlusion — see below)
# --- optics (the ray march) ---
light_atten   = [1.0, 1.0, 1.0]  # per-channel RGB attenuation; 1 = opaque, 0 = clear
heat_atten    = 1.0          # heat-ray attenuation (scalar)
# --- thermal (temperature & heat) ---
conductivity  = 50.0         # high = heat spreads fast along the material (metal)
ignition_temp = 0.0          # catches fire at this temperature; 0 = non-flammable
# --- acoustics (pressure-wave boundary) ---
wave_reflect  = 0.9          # fraction of shockwave energy bounced back
wave_absorb   = 0.1          # damped; for now reflect + absorb = 1 (transmit = sound through the wall = deferred 4b)
# --- flow (gas + smoke share one permeability) ---
permeability  = 0.0          # 0 = sealed (wall); 1 = open (air); units write partial values
# --- structural ---
blast_resist  = 0.0          # blast resistance
burst_threshold = 0.0        # over-pressure differential a wall holds before it bursts (0 = never)
```

The full shipped set — **one table**, columns grouped by the **system** that reads each
(structural · fire · optics/rays · acoustic wave · gas-flow · movement):

| id · key | hp | blast | flammable | ignition | conductivity | `light_atten` | `heat_atten` | wave R / A | `permeability` | passable |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 `air`   | 0   | 0    | no  | –   | 0    | `[0,0,0]`       | 0   | 0 / 0     | 1  | yes |
| 1 `hull`  | 300 | high | no  | –   | 50   | `[1,1,1]`       | 1   | 0.9 / 0.1 | 0  | no  |
| 2 `wood`  | 60  | 0    | yes | 300 | 0.15 | `[1,1,1]`       | 1   | –         | 0  | no  |
| 3 `door`  | 40  | 0    | no  | 280 | 0.3  | `[1,1,1]`       | 1   | –         | 0¹ | yes |
| 4 `steel` | 200 | high | no  | –   | 45   | `[1,1,1]`       | 1   | –         | 0  | no  |
| 5 `glass` | 15  | 0    | no  | –   | 1    | `[0.1,0.1,0.1]` | 0.3 | –         | 0  | no  |

¹ door `permeability` is 0 only while closed; the dynamic-door system flips it open. Wave:
`reflect + absorb = 1` for now — *transmit* (sound through a wall) is the deferred 4b extension.

This single table **is** the per-system coefficient set — each column is consumed by exactly one
system (the next section spells out which). **Consumed today:** `light_atten` (the ray march's
RGB channels), `heat_atten` (the ray march's independent heat channel — engine/06 §1),
`permeability` (the gas/smoke flux boundary), `wave_absorb` (per-cell wave damping), and
`burst_threshold` (over-pressure wall failure). Still stored-and-waiting: `wave_reflect`/transmit
waits for the through-wall 4b extension. Values are illustrative — tuned in the lighting demo and a
future wave test; `–` means "not yet tuned".

### Why a named-key table and not flat arrays

An earlier design stored materials as parallel flat arrays
(`[hp, reflectivity, absorption, flammable, passable]`), one array per material indexed by
column number. That format conflated two physically distinct things under one "absorption"
number — *optical* absorption (how much light a tile eats) and *acoustic* absorption (how
much shockwave energy it damps) — and made adding a column a positional, error-prone edit.

The named-key form fixes both. Optics and acoustics are **separate columns** with separate
meanings, and a new property is a new named key, not a new array slot whose index every
reader must agree on. The "two places" failure mode — the same id list duplicated across
files — is also gone: `MAT_*` ids live in exactly one module (`simulation.materials`) and are
re-exported everywhere else.

## How the table becomes per-tile arrays

`MaterialTable` (in `simulation.materials`) reads the config and builds one numpy array per
column, indexed by material id. Scalar columns (`hp`, `conductivity`, …) become 1-D arrays;
`light_atten` is an `(N, 3)` RGB array. The table validates that ids are **contiguous 0..N-1**
so an id-indexed array has no gaps.

Per-tile state is then a **single fancy-index lookup**: `table.hp[gmap.material]` projects the
HP column across the whole grid in one vectorised operation. `GameMap._update_caches()` does
exactly this to build the derived caches:

```python
self.permeability = table.permeability[material]           # gas/smoke flux coefficient
self.solid        = self.permeability <= 0                 # solidity mask (replaces is_wall)
self.light_atten  = table.light_atten[material]            # (h, w, 3) static optics
self.wave_absorb  = table.wave_absorb[material]            # per-cell wave damping
self.flammable    = table.flammable[material]
self.wall_hp      = table.hp[material]
self.conductivity = table.conductivity[material]
```

This is why there are no tile-objects: every "property of a tile" is a column of the table
read through the `material` index. A C++/CUDA port keeps the same shape — the table is a few
small constant arrays uploaded once, and the per-tile caches are array projections.

## Interaction is per-system — a coefficient, not one flag

A tile has no single "is it solid" property. *What* a tile blocks depends on *which* system is
asking, and the answers are independent: a **grill** stops a unit but passes light, gas, and fluid; a
pane of **glass** stops a unit and passes light (dimmed); a **hull plate** stops everything. No one
boolean can carry that, so each transport system reads **its own coefficient**, and the table holds
one column (or set) per system — **the master table above is grouped exactly this way**: `light_atten`
for light (and vision, which reads its aggregate), `heat_atten` for heat, `wave_reflect`/`wave_absorb`
for the pressure wave, `permeability` for gas + smoke flow, `passable` for movement, `conductivity`
for electricity (and fluid will reuse a permeability when it exists). The grill is the proof that one
mask cannot serve all of them: it passes light, gas, and fluid yet stops a unit, so its columns
disagree — `permeability = 1`, `light_atten = [0,0,0]`, `passable = no`.

The unifying rule is how a **unit** sits in those columns: a unit is *partial* in every system — it
dims light, soaks heat (and so catches fire), absorbs a shockwave, slows but does not seal smoke —
with exactly **one** exception. **Movement** is the only interaction where a unit is a hard, binary
blocker (the `passable` stamp); everything else is a coefficient.

**`is_wall` is retired.** It used to mean "occludes" — derived from `light_atten > 0` — and three
different systems leaned on that one accidental flag at once (the smoke/pressure boundary, the vision
blocker, the wall hard-stop). Each now reads the coefficient it actually needs: light and vision →
`light_atten`, the gas wave → the wave coefficients, gas flow → permeability, movement → `passable`.
No system asks "is this a wall?" — it asks "how much does this stop *me*?"

The door is why `passable` and the flow coefficients must stay separate: a **closed door** is
impermeable to gas (it stops smoke *now*) yet `passable` (a unit traverses it — it opens). Two
systems, two answers, one tile — a single boolean could never hold both. (There is currently no
open/closed door state in code; a door is always passable and always opaque until the dynamic-door
system lands.)

### A unit is a mobile material patch

Because every interaction is a coefficient, a unit needs no bespoke physics: it is **matter that
moves**, carrying the same coefficient set a material does and writing it into the *dynamic* copy of
each field every tick — exactly as the static material table fills the static fields. The shipped
instances are **light** (`dyn_light_atten`), **gas/smoke flow** (a partial `dyn_permeability`, so a
body is *soft* — air and smoke seep past it), and the **pressure wave** (`dyn_wave_absorb`, so a body
absorbs a blast instead of mirroring it); heat follows the same per-tick stamp once its solver is
built. The lone non-coefficient interaction, movement, is the hard `passable` stamp (a unit's
footprint marks its cells un-enterable).

## Per-channel attenuation: static × dynamic

The ray march does not read a binary "blocked" flag. It reads a **per-channel RGB attenuation**
and multiplies the ray's surviving light by `(1 - atten)` on each channel as it crosses a tile.
This single mechanism subsumes both the old `block_light` boolean and the wall hard-stop:

| material | `light_atten` | behaviour in the march |
|----------|---------------|------------------------|
| opaque wall | `[1,1,1]` | ray dies (`1 - 1 = 0` survives) — exactly the old hard block |
| air | `[0,0,0]` | passes untouched |
| glass | `[0.1,0.1,0.1]` | passes, dimmed to 90% |
| tinted window | `[0.9,0.9,0.1]` | "blocks 2 of 3 colours" for free — survivor is tinted |

Because the coefficient is RGB, colour tinting is free: an unequal triple lets a tile pass some
wavelengths and block others. This is the same multiply smoke already uses; the only difference
is where the multiplier comes from (a table value instead of a smoke density).

The attenuation the march actually reads is the product of two fields:

```
total_atten[channel] = material_atten[channel]  (static)
                     × dynamic_atten[channel]   (live, recomputed each tick)
```

- **Static (`light_atten`)** — the material column projected onto the grid. It is a
  *structural-change cache*: it changes only when a tile's material changes (a wall is
  destroyed), so it is rebuilt in `_update_caches()` and patched per-tile in `on_tile_changed`,
  never recomputed per tick.
- **Dynamic (`dyn_light_atten`)** — a live per-channel field rebuilt **every tick** in
  `stamp_units()`. It starts as a copy of the static field, then each living unit raises the
  opacity over its footprint via **per-channel MAX** (an occluder can only *add* opacity, never
  remove it). A unit's opacity comes from an optional `unit.light_atten` (default `[1,1,1]` =
  fully opaque, casting a solid shadow), so a future creature could pass green light or an
  aquarium could tint blue-green — per colour, for free.

`stamp_units()` produces the live coefficient fields in one pass: `dyn_permeability` and
`dyn_wave_absorb` (gas/smoke flux + wave damping, read by the wave and smoke physics), the
`obstacles` mask, and `dyn_light_atten` (read by the ray march). Units now write *partial* gas
coefficients — a reduced `dyn_permeability` (soft body) and an added `dyn_wave_absorb` — so a person
slows-not-seals smoke and soaks a blast without reflecting it, exactly as the interaction model
above requires. The march reads `dyn_light_atten` **read-only** — units occlude rays by being
stamped into the field
before the march, never by the kernel writing units. This keeps the march a deposit-only pass
over a frozen world, which is what makes the eventual CUDA port (one thread per ray, read-only
world) mechanical.

What attenuation does **not** cover is **direction-changing** optics — refraction, prisms,
mirrors. Those are a deferred entity-re-emission pattern (secondary rays spawned outside the
kernel), never in-kernel ray forking. Attenuation is straight-through dimming and tinting only.

## Cache invalidation

The table-derived caches are patched incrementally on each structural edit through the single
`on_tile_changed` seam — never an O(grid) rebuild, which would not scale when a firestorm melts many
walls per tick. That seam, and `destroy_wall`'s edit-specific atmosphere/vacuum handling, are owned
by the **State & Ownership chapter**; materials only adds that the caches patched there are the table
projections (`light_atten`, `flammable`, `wall_hp`, `conductivity`, …) — one per-tile delta, exactly
what the CPU→GPU upload would push.

## Config hot-reload

The table is rebuildable at runtime. After a `config.toml` edit, `reload_material_table()`
re-reads `[materials]`, rebuilds `MaterialTable`, and re-derives the static caches while
preserving the live `material`/vacuum grids and the running atmosphere/obstacles state. This
lets values be tuned in the lighting demo without a restart. (When CUDA lands, the GPU
material mirror re-syncs here.)

## Material ids are open-ended

The table makes the material set open: a new material is one `[materials.<name>]` row (which
fixes its id by declaration order) plus one CSV-code mapping in `level_loader`. There is no
fixed cap and no propagation system to edit.

The CSV-code mapping is deliberately *not* the same as the material id. CSV codes are the
authoring convention in a level's `tilemap.csv`; `level_loader.materials_from_tilemap` translates
them to ids:

| CSV code | material |
|----------|----------|
| 0 | outer space → vacuum (`MAT_AIR` + vacuum flag) |
| 1 | `MAT_HULL` |
| 2 | `MAT_WOOD` |
| 3 | `MAT_DOOR` |
| 4..8 | interior-air decoration variants → `MAT_AIR` |

Keeping codes and ids distinct is what lets existing levels use codes 4–8 as floor-decoration
variants (all air) without colliding with steel/glass — those materials exist in the table but
have no CSV code yet, because no shipped level places them. A dedicated code is added the first
time a level needs them.

## Deferred columns

Two columns are named in the design but not in the shipped table, because their consumers do not
exist yet:

- **`emissivity`** — for hot-tile emission (tiles above a glow threshold becoming light
  sources, powering the wind→fire→firestorm loop). Added when that chapter lands.

It is listed here so the schema is understood as intentionally incomplete, not forgotten.

Specular reflection is deliberately **not** a material column. Mirrors and prisms are the explicit
Tier-3 entity re-emission pattern (secondary rays we spawn *outside* the kernel — see "direction-
changing optics" above), so reflectivity lives on that entity, never on the material the march reads.
A per-material `light_reflect` would imply automatic in-march bouncing, which the CUDA contract
(one thread per ray, no forking) rules out.

---

## Implementation status

**Built and shipped:**

- **Named-key property table** — `config.toml [materials.*]` in the documented format, read by
  `MaterialTable` (`src/simulation/materials.py`) into per-id numpy arrays. Contiguity validated;
  dict-of-dicts accepted for tests.
- **All six materials** present as rows: air, hull, wood, door, steel, glass.
- **Single source of `MAT_*` ids** in `simulation.materials`, re-exported by `gamemap.py` and
  used by `level_loader.py` — the "two places" duplication is gone.
- **Table → per-tile caches** via fancy-index in `GameMap._update_caches()`: `permeability`,
  `solid`, `light_atten`, `wave_absorb`, `flammable`, `wall_hp`, `conductivity`.
- **`is_wall` retired.** `GameMap.solid` (= `permeability <= 0`) replaces it everywhere in Python;
  the flow boundary (`obstacles`) and `has_los` now read `solid`/`permeability`, not the old
  light-occlusion accident. (The C++ solvers keep a now-vestigial `is_wall` *parameter*, fed
  `gmap.solid`; removing it + rebuild is a pending follow-up.)
- **`is_passable` / `is_passable_block`** as the separate walkability predicate (AIR + DOOR).
- **`permeability` column consumed** — `GameMap.permeability` / `dyn_permeability`; the C++
  atmosphere + smoke solvers gather flux via `face = min(perm[self], perm[neighbor])`. Default is
  sealed iff the material occludes light; behaviour-identical for the current materials.
- **`wave_absorb` column consumed** — `GameMap.wave_absorb` / `dyn_wave_absorb`; the C++ wave update
  damps per cell by it (`wave_absorb_strength`). Energy-out only; open air bit-identical.
- **`burst_threshold` column consumed** — over-pressure wall failure: `GameMap.find_burst_walls`
  destroys walls holding a pressure differential above their `burst_threshold` (see ch.04).
- **Soft units in the gas fields.** `stamp_units` writes unit footprints as a *partial*
  `dyn_permeability` (default 0.5, per-unit hook + `[physics] unit_permeability`) and an added
  `dyn_wave_absorb` (`[physics] unit_wave_absorb`) — soft body, absorbs blasts; movement still hard.
- **Per-channel RGB attenuation, consumed for real.** The C++ raycaster
  (`cpp/src/raycaster.cpp`) reads `light_atten` per channel and applies `(1 - atten)` to the
  ray's RGB; opaque `[1,1,1]` kills the ray, glass `[0.1,…]` dims it, asymmetric triples tint.
- **Scalar `heat_atten`, consumed for real (engine/06 §1).** The same march carries an
  INDEPENDENT 4th channel — a scalar heat survival attenuated per tile by `(1 - heat_atten)`,
  exactly as each RGB channel is attenuated by `light_atten`. The per-tile `GameMap.heat_atten`
  cache is the table column projected onto the grid (built in `_update_caches`, patched in
  `on_tile_changed` — same seam as `light_atten`). Heat and light occlusion now diverge: a
  heat-shield (light-clear, heat-opaque) blocks heat while passing light, smoked glass the
  converse. Static material heat only — no `dyn_heat_atten` (units blocking heat is a later
  refinement). The ray terminates on the aggregate of all four channels {R,G,B,heat}.
- **Static × dynamic split.** `dyn_light_atten` is rebuilt in `stamp_units()` as static atten
  combined per-channel (MAX) with each unit's opacity (default opaque), allocated once and
  written in-place; the march reads it read-only. Per-unit `light_atten` hook exists.
- **Incremental cache seam** — `on_tile_changed(fy, fx)` patches all static caches O(1);
  `destroy_wall` routes through it.
- **Config hot-reload** — `reload_material_table()` rebuilds table + static caches, preserving
  live grids.
- **Open material set** — CSV-code → id mapping in `level_loader.materials_from_tilemap`, with
  codes distinct from ids.

**Designed but not yet built:**

- **The light cast still runs renderer-side.** The march is invoked from
  `renderer/lighting.py:LightingPass`, not from the sim step. The buffers and read-only
  contract are already correct, but the planned relocation of the cast into the
  `Simulation`/PhysicsEngine step has not happened. This is a ray-engine concern, not a material
  concern, but it means the material attenuation is consumed by a render-time call today.
- **Door open/closed state** — both predicates treat every door identically; no occlusion flip.
- **Remove the vestigial C++ `is_wall` parameter** (+ rebuild). The Python side is fully off
  `is_wall`; the solvers still take an `is_wall` argument (fed `gmap.solid`) that should be dropped.
- **`wave_reflect`, `blast_resist` columns** are in the schema but **consumed by nobody yet**.
  The per-material *reflective* wave boundary (the lossy mirror; through-wall 4b) wires into
  `wave_reflect` later. (`heat_atten` is now consumed by the ray march's heat channel — see
  the "Built and shipped" list above; `conductivity`/`ignition_temp` feed the temperature solver.)
- **`emissivity`** column — not in the table; deferred with hot-tile emission. (Specular reflection is
  not a material column at all — it is the Tier-3 entity re-emission pattern.)

**Gaps and inconsistencies to be aware of:**

- **Door flammability mismatch with the prose design.** The design illustrates doors as
  `flammable = true` (`ignition_temp = 280`), but the shipped `config.toml` sets door
  `flammable = false` to preserve the legacy behaviour where the old hardcoded cache made only
  wood flammable. The door's `ignition_temp` is set to 280 regardless. This is a deliberate,
  documented hold (see the comment in `config.toml`), to be flipped when igniting doors becomes
  an intended gameplay change.
- **Glass default permeability via occlusion.** `permeability` is optional in config and defaults
  to *sealed iff the material occludes light* (`MaterialTable.occludes()`), so the shipped glass row
  (`light_atten = [0.1,0.1,0.1]`, occludes) defaults to `permeability = 0` — sealed to gas/smoke
  while still transmitting most light in the march. No shipped level places glass, so this is latent;
  when a level does, glass will correctly seal gas yet pass dimmed light. A grill (passes gas) would
  declare `permeability = 1` explicitly to override the occlusion-derived default.
- **Scalar `light_map` lingers** alongside `light_rgb` during the RGB migration; the scalar field
  is a legacy render-tint path, not part of the material table.
