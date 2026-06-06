# Grid & Coordinates

**Depends on:** (none — foundation)

This chapter fixes the spatial substrate every other system stands on: the single grid the world is
discretised into, the two coordinate systems used to address it (array indices and metres), the map
between them, and what each cell stores. It is also **where the physics happens** — every solver
(atmosphere, smoke, fire, the wave equation, the ray march) reads and writes its fields on this one
grid, in place. Get this right and every downstream system — physics fields, the ray engine, units,
pathfinding, rendering — addresses the same cells the same way.

---

## 1. One grid

Breach has **exactly one grid**. Every spatial quantity lives on it — material, wall health,
atmosphere, smoke, fire, wind, the per-channel light-attenuation field, the heat buffer, unit
footprints, line-of-sight. There is no second, coarser grid: a unit is not on a different grid from
the smoke it walks through; it occupies a block of the *same* cells the smoke diffuses across.

Every field is a numpy array of shape `(h, w)` (or `(h, w, 3)` for the RGB light / attenuation
fields), all indexed identically. So one wall mask, one Laplacian, and one set of boundary
conditions serve all of them, and the emergent chains — a breach vents atmosphere → smoke is pulled
out → fire starves near the breach — fall out of independent solvers reading and writing the *same*
cells, with no cross-system glue.

> **Why one grid.** An earlier design ran units on a coarse 1 m grid above a finer 1/3 m physics
> grid. It is abolished: the two vocabularies (`coarse`/`fine`, `cx/cy` vs `fx/fy`) drifted until
> the same numbers meant different things in different files — the worst case put HUD-read spawn
> points a third of the way across the map — and every unit↔field interaction paid a translation
> tax. One grid removes both the tax and the ambiguity at once.

---

## 2. Tile size is a per-level knob, bounded by geometry

`tile_size_m` — the physical size of a tile — is **not a global constant**. It is a property of the
loaded level and a deliberate **fidelity/performance knob**. The physics is grid-resolution-limited
(pressure gradients, smoke plumes, and blast fronts are only as sharp as the cells they live in), so
finer tiles buy sharper physics at more compute. A 1 m tile smears a decompression front across a
doorway; 1/3 m resolves it. Today 1/3 m is cheap; as systems are added, the knob lets a level trade
resolution for headroom without touching any system's code.

The knob is bounded from above by **geometry, not by the material**: a level's walls and features
define the coarsest tile size at which every edge still lands exactly on a tile boundary. The sim
may run at that size **or any integer subdivision of it** — you can always halve a tile and stay
aligned, never coarsen it without smearing geometry across cells. So a level declares its base
(coarsest-exact) resolution and the sim runs at `base / k`.

> *Forward refinement:* a finer scheme — each material declaring a minimum feature size, with a
> level's required resolution = the GCD of the features present — is possible later. Today, alignment
> is treated as a level-layout property.

Because the size is variable, **the physics must derive its SI constants from it.** Diffusion rates,
wave speed, and the CFL-bounded `dt` are all functions of `dx = tile_size_m`; a solver that assumes
1/3 m is silently wrong at any other resolution. The scaling is read from the level, not hardcoded.
This is the substance of "variable tile size" — the knob only *means* anything once the constants
follow it.

The same caveat reaches **gameplay** quantities expressed in tiles. A unit speed of "one tile every
N ticks" is really a physical speed of `tile_size_m · (ticks_per_s) / N`, so it is just as
resolution-dependent as a diffusion rate: speeds — and any tiles-per-tick rate — are defined in m/s
and converted through `tile_size_m`, never frozen as raw tiles. Where that conversion lives for units
is the Units chapter's call.

Grid **dimensions** likewise come from the level: `GameMap` allocates every field at the tilemap
CSV's `(h, w)` (the test vessel is 120 × 75 tiles = 40 m × 25 m). There are no
`map_w`/`map_h`/`fine_w`/`fine_h` config knobs — those belonged to the dead dual-grid.

---

## 3. Two coordinate systems and the map between them

Two systems, each with one job:

- **Array indices `(row, col)` — integers.** For addressing fields: `field[row, col]`. Row-major,
  matching numpy.
- **World position — metres, floats.** For anything physically meaningful: forces, damage-falloff
  radii, distances, the location of a thing in the world.

The map between them is fixed by a **canon origin** and the **scale**:

- **Origin:** the upper-left corner of tile `(row=0, col=0)` is world `(0, 0)`.
- **Axes:** `col` / `x` increases to the right; `row` / `y` increases **downward** — the same
  direction as the array and the screen.
- **Scale:** `tile_size_m` (per level).

So `x_m = col · tile_size_m`, `y_m = row · tile_size_m`, and the inverse **floors**:
`row = floor(y_m / tile_size_m)`. Flooring (not rounding) defines tile membership — a position lands
in the cell whose box contains it; `(3.7, 2.1)` tiles belongs to cell `(row=2, col=3)`.

> **Why y-down, not y-up.** The usual reason to anchor the origin lower-left with y increasing
> upward is that physics formulas with gravity read naturally that way. Breach is **top-down** — the
> simulated plane is the floor seen from above; there is no gravity in it, so that benefit does not
> exist. A y-up world frame over a y-down array would force a flip (`y_m = (H − row)·s`) at *every*
> index↔metre boundary — a permanent, silent bug source — bought for nothing. One direction
> everywhere is worth more than a textbook orientation we never exploit. Rotational sign is fixed by
> a single documented convention: a positive angle turns **clockwise on screen** (because y points
> down).

**Tile units** survive as a convenience **alias of metres** — `tile = metres / tile_size_m` —
because "move one tile", "footprint 3", "a wall 5 tiles away" are the natural currency of a tile
game and keep `tile_size_m` out of gameplay code. Game logic reads and stores tile coordinates
(float); the metre reading is taken at the physics boundary; the index reading at array access. All
three describe the same point.

These conversions live in `src/simulation/coords.py` — the origin, the scale, and the
index/tile/metre maps — and nowhere else. `coords.py` knows **nothing about pixels**: the tile→pixel
mapping is the renderer's camera, so the simulation's spatial logic carries no display assumption and
runs headless (for testing and NN self-play) with no renderer present.

```python
tile_to_index(x, y)            -> (row, col)   # floor
index_to_tile(row, col)        -> (x, y)
tile_to_meters(x, y)           -> (mx, my)     # × tile_size_m, origin (0,0)
meters_to_tile(mx, my)         -> (x, y)
tile_distance_m(x1,y1,x2,y2)   -> float
```

> *Per-level origin:* each level is its own `(0,0)` today. A *shared* world frame — placing a ship
> and an EVA exterior band in one space — is where a per-level world-offset will earn its keep.
> Deferred until the exterior band exists.

---

## 4. What the grid stores about each cell

World state is **arrays plus a material-property table**, not a grid of tile objects: a cell's
properties are looked up by projecting the per-material table through the `material` grid, not stored
per-cell.

How a cell interacts with each system is **not one "is it solid" flag — it is a per-system
coefficient.** A grill passes light, gas, and fluid but stops a unit; glass stops a unit and passes
light (dimmed); a hull plate stops everything. No single mask can express that, so each transport
system reads its own coefficient — light/heat → attenuation, the gas wave → reflect/absorb/transmit,
gas flow → permeability, and so on — and both static materials and dynamic units write them. That
model (the master table of system → coefficient, and the rule that **a unit is a mobile material
patch**, partial in every system but one) is the subject of the **Material chapter**. The grid only
fixes that every such field shares this one grid and is indexed identically.

The single interaction that *is* a grid-level boolean is **`is_passable`** — walkability for units
and pathfinding: a cell a unit may occupy (walkable material, not already occupied). Movement is the
one place occupancy is hard and binary; every other "blocking" is a coefficient in which a unit is
only partial. (The full field set and the coefficient model live in the State and Material chapters.)

---

## 5. Units on the grid: the interface, not the convention

A unit is larger than a tile and occupies a block of cells, so the grid needs to know *which* cells.
But **how a unit defines its anchor and shape is the units chapter's business, not this one.** The
grid depends only on a small method interface:

```python
unit.occupied_tiles() -> list[(x, y)]   # the cells this unit covers now
unit.occupies(tile)   -> bool
```

Everything grid-side goes through it: collision, line-of-sight, hit-detection, and the per-tick
**stamp** (`GameMap.stamp_units`) that paints living units' `occupied_tiles()` into `is_solid` (units
act as walls for flow) *and* into the dynamic attenuation field (units cast shadows) in one pass.
Because the contract is a method, a unit of any size — a rigid 3×3 human, a larger body, or a future
articulated segment-chain that reshapes every tick — satisfies it with no change to any consumer.

What the grid deliberately does **not** fix: whether a unit's position is its top-left tile, its head
tile, or several anchors; the footprint→size rules; and lifestate such as zombification-as-a-state.
Those are unit-design decisions and live in the units chapter.

---

## 6. Logical position is discrete; smooth motion is render-only

Game state is **tile-discrete**: a unit's logical position is its current tile, movement hops
tile-to-tile at the unit's cadence, and the simulation never depends on a sub-tile position. The
float in a unit's `(x, y)` exists so the **renderer** can interpolate between the previous and current
tile for smooth motion; that interpolation is purely visual and never feeds back into logic
(pathfinding, collision, and combat read the integer tile). Keeping interpolation in the render layer
is what makes ticks deterministic regardless of frame rate — the same ticks produce the same discrete
states, which is what lets the sim run headless.

---

## 7. The conversion discipline (summary)

One habit: **stay in tile units; convert only at the boundary that needs another representation.**

```
                 floor (origin 0,0, y-down)
   tile (x, y) ───────────────────────────►  (row, col)   →  field[row, col]
       │
       │  × tile_size_m
       ▼
    metres                                   →  forces, damage falloff, distances
       │
       │  (renderer camera; never in coords.py)
       ▼
    pixels                                   →  drawing only
```

- **Indexing a field?** `tile_to_index` (or a unit's integer tile).
- **A physical quantity?** Metres via `coords.py`.
- **Drawing?** The renderer's camera maps tiles to pixels. The sim never does.

---

## Implementation status

Audited against the shipped code, not the prose above.

**Built and in use:**

- **One grid, per-level sizing.** `GameMap.__init__` allocates every field from
  `level_data.tilemap.shape`; no `fine_w`/`fine_h`/`map_w`/`map_h`. Fields are `(h, w)` or `(h, w, 3)`,
  all indexed identically.
- **`tile_size_m` is per-level** (`LevelData.tile_size_m`, default 0.333), not a global config key.
- **The canon origin/axes already match the code** — `tile_to_meters` multiplies from `(0,0)` with
  y-down; this chapter formalises what the code already does (no flip to undo).
- **`coords.py`** has all five helpers with floor (not round) semantics in `tile_to_index`.
- **Coord cleanup complete.** `Unit` stores float `x`/`y` as source of truth, exposes integer
  `tile_x`/`tile_y`, carries `footprint` (default 3); the `cx/cy/fx/fy/fxf/fyf` vocabulary and
  `CFG.display.coarse` are gone; spawn entries use physics-tile `x`/`y`.
- **Walkability + stamp.** `is_passable` / `is_passable_block` exist; the per-tick stamp paints unit
  footprints into the solid mask and the dynamic attenuation field; `occupied_tiles()` / `occupies()`
  exist and drive it.

**Designed — owed by the code (this chapter is now canon; code catches up):**

- **Drop `is_wall`.** Today `is_wall` = "occludes light" (derived from `light_atten`), and both the
  solid mask (`obstacles`) and `has_los` depend on it. Build the flow boundary (`is_solid`) from a
  dedicated material solidity column, route every consumer off `is_wall`, and remove it.
- **Vision via attenuation.** `has_los` is Bresenham-on-`is_wall` today; it should march the
  attenuation field (smoke dims sight; infravision on the heat channel) behind the same `has_los(a,b)`
  interface.
- **Physics constants from tile size.** Audit `physics.py` / the solvers: confirm diffusion, wave
  speed, and `dt` derive from `dx = tile_size_m` rather than assuming 1/3. Required before the
  tile-size knob is real.
- **Material `blocks_flow` / `walkable` columns.** Add explicit solidity and walkability properties
  to the material table so `is_solid` / `is_passable` derive from intent, not from the
  light-attenuation accident. (Shared with the Material chapter.)
- **`coords.py` adoption.** Helpers are under-used — many call sites floor inline or open-code metre
  conversions; route them through `coords.py` as the single seam.

**Not built (forward):**

- **Variable footprints** — the machinery parameterises on `footprint`, but every spawn is 3 today;
  the size→footprint rule (2/3/4), articulated/segment-chain bodies, and facing-rotated footprints
  are unbuilt.
- **Geometry-alignment validation** — levels do not yet declare/enforce a base resolution.
- **Per-level world-offset** — for a shared world frame (the EVA exterior band).
- **Legacy `light_map`** (scalar) still coexists with `light_rgb` during the RGB migration.
