# Grid & Coordinates

**Depends on:** (none — foundation)

This chapter defines the single spatial substrate every other system stands on:
the grid the world is discretised into, the one coordinate system used to address
it, and the rules that connect tile units to matrix indices, to metres, and to the
screen. Get this right and every downstream system — physics fields, the ray
engine, units, pathfinding, rendering — addresses the same cells the same way.

---

## 1. One grid

Breach has **exactly one grid: the physics-tile grid.** Every spatial quantity in
the game lives on it — material, wall health, atmosphere, smoke, fire, wind, the
per-channel light-attenuation field, the heat-deposit buffer, unit footprints,
line-of-sight checks. There is no second, coarser grid layered on top.

This is the load-bearing decision of the whole spatial design, and it was reached
the hard way. An earlier design ran units on their own coarse 1 m × 1 m grid sitting
above a finer 1/3 m physics grid. That dual-grid scheme is **abolished**. It created
a permanent translation tax — every interaction between a unit and a field had to
multiply or divide by the coarse factor — and the two vocabularies (`coarse`/`fine`,
`cx`/`cy` vs. `fx`/`fy`) drifted until the same numbers meant different things in
different files. The most visible failure: cursor coordinates read off the on-screen
HUD were physics-tile coordinates, but the unit constructor interpreted them as
coarse coordinates and multiplied by three, so spawn points copied from the HUD
landed a third of the way across the map.

The single grid removes the tax and the ambiguity at once. A unit is not on a
different grid from the smoke it walks through; it occupies a block of the *same*
tiles the smoke diffuses across.

### Eriks comments: Im not sure i like we open the first paragraph of the first document with a description of what the grid is not.
Im happy we went through this, and reading this whole document, there are a few things I really dont like.

1. The tile is deliberatley small - 1/3 metre :  The original intention was to make the dimensions of the tile a variable, i had no clue how many tiles i could afford simulating, i had no idea how expensive everytthing would be, difffusion and wave equation combined with raycasting etc. Right now , it seems to be extremely cheap- and we can reduce the size of the  tiles  - but who knows what happens as we add more systems.

I would really like to discuss how much of a redesign it would be to re-write this whole document the way i initially intended- I suspect it's not that bad, because the strucutre is all ready right-. we have aset of tiles, and that's what decides everything basuically.
maps generated will have to know which tile resolution they go for, since it determines at what distances you can put walls from each other, so they'll ahve requirements on the tile dimensions. Perhaps each mat needs to define it's "biggest smallest" unit, of which every geometry must be a multiple - if u get what i mean. This is what the section ### Tile size is per-level, not global should be replaced with - it's not even one size per level, but levels have a requirement of biggest smallest tile kind of (since you can alwasy divide the tiles and make them smaller, thats why im talking about the biggest tile that is small enough to accuratley align with the geometry with the level).

2. The coodinate system - i would actually prefer if we had one coorinate system which is in meters, and one tile system which has indcies, like rows and columsn
And a map between them
We could introduce a canon origin between the tiles and coorinate system - for example the outer corner of the tile in the last row of the first column could be origo. (setting origo in the lower left corner) - and use that as a standard. so each level - ship or whatever, has it's own coorinate system anchored to the canon origo like this.

I would like to discuss with Cladue before committing to this- beacuae im not sure if it might cause more problems than it solves to have the axis go in different directions like this - im absolutely oipen to have the coorinates align with the row col system, but i wonder if the x axis should align with rows and y with cols in that case -so we at least maintain a right handed system (i dont know what u call that in english, but it's something about the axis x,y,z should align with the index finger, middle finger and the thumb of the right hand, this is a property that seems nice to preserve? We should discuss the different possibilities.)

that would separate the roles iof the two - if u want to talk elements in an array, you use the indicies (i,j) and if you want a location (in meters) you can use coorinates. The mapping between them is trivial to produce, it needs the origo (which we've all ready established) and the scale of the tiles.

coorinates are floats, the indicies are ints ofc.

### Why a fine grid

The tile is deliberately small — **1/3 metre**. The physics is the reason. The
atmosphere, smoke, and wave solvers are grid-resolution-limited: pressure gradients,
smoke plumes, and blast fronts are only as sharp as the cells they live in. A 1 m
tile would smear a decompression front across a doorway; a 1/3 m tile resolves it.
Units are large relative to a tile (a baseline human is 3×3 tiles = 1 m²) precisely
so the *physics* can be fine while a *unit* still spans enough cells to interact with
spatial structure — to be pushed unevenly by a wind gradient, to cast a believable
shadow, to block a corridor.

---

## 2. The coordinate system

A position is a single **`(x, y)` pair in tile units**. That is the only coordinate
system in game state. `x` increases to the right, `y` increases downward; the origin
`(0, 0)` is the top-left tile.

The same `(x, y)` pair has three readings, reached by explicit conversion and never
by a competing storage format:

| Reading | How | Used for |
|---|---|---|
| **Float tile position** | the value itself | the source of truth on a unit; smooth motion; render interpolation |
| **Integer matrix index `(row, col)`** | `int(y), int(x)` (floor) | indexing a physics array `field[row, col]` |
| **Metres** | `x * tile_size_m, y * tile_size_m` | any physically meaningful quantity — forces, damage falloff radii |

The integer index is `row` first, `col` second, because the field arrays are numpy
and row-major: `field[row, col]` is `field[int(y), int(x)]`. Flooring (not rounding)
defines tile membership — tile position `(3.7, 2.1)` belongs to cell
`(row=2, col=3)`. A unit standing anywhere inside a cell indexes that cell.

These conversions are centralised in `src/simulation/coords.py` so they exist in
exactly one place. The rest of the code stays in tile units; it only reaches for a
conversion at the boundary where another representation is genuinely required —
matrix access, or a metre-valued physics term.

```python
def tile_to_index(x, y)   -> (row, col):   return int(y), int(x)      # floor
def index_to_tile(row, col) -> (x, y):     return float(col), float(row)
def tile_to_meters(x, y, tile_size_m)      -> (mx, my)
def meters_to_tile(mx, my, tile_size_m)    -> (x, y)
def tile_distance_m(x1, y1, x2, y2, tile_size_m) -> float
```

### Tile size is per-level, not global

`tile_size_m` is a property of the loaded level (`LevelData.tile_size_m`, read from
the level's TOML; currently 1/3 m), **not** a global config constant. The renderer
and simulation read it from the level. This is why metre conversions are isolated to
`coords.py`: if a level ships at a different physical resolution, only the
metre-conversion call sites are affected — everything expressed in tile units is
unchanged, because tile units are the stable currency.

### Grid dimensions come from the level

The grid is sized from the level's tilemap CSV, not from a config key. `GameMap`
reads `level_data.tilemap.shape` and allocates every field at `(h, w)` to match. The
CSV decides the world size; the test vessel is 120 × 75 tiles (40 m × 25 m). There
are no `map_w`/`map_h`/`fine_w`/`fine_h` config knobs — those belonged to the dead
dual-grid scheme and are gone.

### Eriks comment on this: this is all fine - but the physics depend on SI units, so wherever we decide to store the scaling, and the level is a fine choice - the physics needs to read off that scaling so it sets the simulation constants accordingly.

---

## 3. How fields use the grid

Every physics field is a numpy array of shape `(h, w)` (or `(h, w, 3)` for the
per-channel light and attenuation fields), addressed by integer `(row, col)`. They
all share the same grid, so a single wall mask, a single Laplacian, and a single set
of boundary conditions serve all of them — the emergent chain (breach vents
atmosphere → smoke is sucked out → fire starves near the breach) falls out of
independent solvers reading and writing the *same* cells, with no cross-system glue.

Access is always through the `gmap.<field>` interface — `gmap.atmosphere`,
`gmap.material`, `gmap.is_wall`, `gmap.light_atten`. World state is **arrays plus a
material-property table**, not a grid of tile objects: a cell's properties are looked
up by projecting the per-material table through the `material` grid, not stored
per-cell. (The table and the field set are the subject of the State and Material
chapters; this chapter only fixes that they are all indexed identically.)


### Eriks comment. this following section is not great - We should now design exactly these things - i think we dont have an occlusion mask any more , do we? we have attenuation intead, dont we? -let's settle it together. i guess wee need a map that tells units if htey can walk there or not , and that should be is_passable, i like that i guess. is_wall should only be true if it is a wall, probably this will have to go? or at least we should have one more look at it.
Two grid-level predicates are worth naming here because they are distinct and easy to
conflate:

- **`is_wall`** — the **occlusion** mask. A cell occludes if it stops light, smoke,
  pressure, and vision. Doors occlude (they are visually solid) and so are marked
  `is_wall`, even though a unit may walk through them.
- **`is_passable(row, col)`** — the **walkability** predicate. A cell is passable if
  a unit may stand on it (air and doors). This is a separate query, not the negation
  of `is_wall`, precisely so a door can occlude *and* be walkable.

Keeping occlusion and walkability separate at the grid level is what lets a single
material (the door) participate correctly in both the physics boundary and the
pathfinding graph without a special case.

---

## 4. Unit occupancy on the grid 

### Eeriks commetns: This following section might need one more look - we decided that units are not just one tile, they have a "footprint" which can take any form- hence they dont have 1 coorinate, however, it might bve very convenient for htem to have 1 coorinate - and we could assign it the top left if you think that is a clear chouice- im not super sure it is the best choice, perhaps they'll need many? we could also assign the tile where the units head is as its position - i think this is kind of more in the units design doc, and that if we try to lock this down now- we are perhaps not doing ourselves a favour- i know we have units in the game, so we need some convention - but those units are more prototypes than anything else. it's a good question if we should add units to this docuemnt or if we should write the units doc next, after this.

A unit's position is one `(x, y)` tile-unit pair — by convention the **top-left tile
of its footprint**. A unit is larger than a tile, so it occupies a block of cells,
and the grid needs to know which ones.

**Footprint is a property of the unit, not a global constant.** Each unit carries
`footprint` (the side length of its square footprint in tiles, default 3 for a
baseline human). No simulation code hardcodes "3" outside the unit constructor's
default; physics, pathfinding, and input all read `unit.footprint`. This is the seam
through which the future variant system introduces larger and smaller bodies (the
sketched rule: footprint 2 / 3 / 4 by unit size) without touching any consumer.

The grid never reads a unit's footprint storage directly. The contract is a **method
interface**:

```python
unit.occupied_tiles() -> list[(tile_x, tile_y)]   # the cells this unit covers now
unit.occupies(tile)   -> bool
```

For a rigid unit this is `anchor + each offset` from the unit's offset list (the
species default for a 3×3 human; a generated square block otherwise). Routing
occupancy through methods rather than a fixed field is deliberate: it lets a future
articulated body — a snake or worm carried as a segment-chain deque whose footprint
reshapes every tick — satisfy the same interface with no change to any consumer.
Collision, line-of-sight, hit-detection, and the per-tick stamp all call
`occupied_tiles()`; none of them knows or cares how the shape is stored.

The clearest consumer is the per-tick stamp. Each tick, `GameMap.stamp_units` walks
the living units' `occupied_tiles()` and paints them onto the grid: into the
`obstacles` mask (units act as walls for the wave and smoke physics) and, in the same
pass, into the dynamic light-attenuation field (units cast shadows). One method call,
two grid outputs — and because it is driven by the footprint interface, a unit of any
size or shape stamps correctly.

---

## 5. Logical position is discrete; smooth motion is render-only

Game state is **tile-discrete**. A unit's logical position is its current tile;
movement hops tile-to-tile at the unit's tick cadence (a baseline marine takes one
tile every few ticks). The simulation never depends on a sub-tile position.

The float in `unit.x` / `unit.y` exists so the **renderer** can interpolate between
the previous tile and the current tile for smooth on-screen motion. That
interpolation is purely visual and never feeds back into logic — pathfinding,
collision, and combat all read the integer tile (`unit.tile_x`, `unit.tile_y`,
which floor the float). Keeping interpolation entirely in the render layer is what
preserves determinism: the same sequence of ticks produces the same sequence of
discrete states regardless of frame rate.

The screen mapping itself — tile coordinates to pixels — is the renderer's concern,
handled by its camera. `coords.py` deliberately knows nothing about pixels; it stops
at tiles, indices, and metres. This keeps the simulation's coordinate logic free of
any display assumption, which is what allows the sim to run headless (for testing and
for neural-network self-play) with no renderer present at all.

---

## 6. The conversion discipline (summary)

The whole system reduces to one habit: **stay in tile units; convert only at the
boundary that needs another representation.**

```
                 floor (int x, int y)
   tile (x, y) ─────────────────────────►  (row, col)   →  field[row, col]
       │
       │  × tile_size_m
       ▼
    metres                                 →  forces, damage falloff, distances
       │
       │  (renderer camera; not in coords.py)
       ▼
    pixels                                 →  drawing only
```

- **Indexing a field?** `tile_to_index` (or a unit's `tile_x`/`tile_y`).
- **Computing a physical quantity?** Convert to metres via `coords.py`.
- **Drawing?** The renderer's camera maps tiles to pixels. The sim never does.

Everything else — positions, footprints, orders, paths, spawn entries — is stored
and reasoned about in tile units. That single currency is what keeps the one-grid
model honest.

---

## Implementation status

Audited against the shipped code, not the prose above.

**Built and in use:**

- **`src/simulation/coords.py`** — all five helpers (`tile_to_index`,
  `index_to_tile`, `tile_to_meters`, `meters_to_tile`, `tile_distance_m`) exist as
  described, with floor (not round) semantics in `tile_to_index`.
- **One grid, per-level sizing.** `GameMap.__init__` allocates every field from
  `level_data.tilemap.shape`; there is no `fine_w`/`fine_h`/`map_w`/`map_h`. Fields
  are `(h, w)` or `(h, w, 3)`, all indexed identically.
- **`tile_size_m` is per-level**, read from the level TOML into
  `LevelData.tile_size_m` (default 0.333), not a global config constant.
- **The coord cleanup is complete.** `Unit` stores float `x` / `y` as the source of
  truth, exposes integer `tile_x` / `tile_y` properties, and carries
  `footprint` (default 3). The `cx`/`cy`/`fx`/`fy`/`fxf`/`fyf` vocabulary and the
  `CFG.display.coarse` global are gone. Spawn entries in `level.toml` use `x` / `y`
  in physics-tile coordinates.
- **Occlusion vs. walkability split.** `GameMap.is_wall` (occlusion, doors included)
  and `GameMap.is_passable` / `is_passable_block` (walkability, air + doors) are
  distinct, exactly as specified.
- **Footprint interface.** `Unit.occupied_tiles()` / `occupies()` exist and are the
  path `stamp_units` uses; footprint is read from `unit.footprint`, not hardcoded.
- **Discrete-state / render-interpolation split.** Logic reads `tile_x` / `tile_y`;
  the float `x` / `y` is the interpolation source. The renderer owns the
  tile→pixel mapping; `coords.py` has no pixel code.

**Designed, not yet built:**

- **Variable footprints.** The machinery handles arbitrary `footprint` correctly
  (`occupied_tiles()`, `stamp_units`, `is_passable_block` all parameterise on it),
  but every spawned unit is footprint 3 today. The size→footprint threshold rule
  (2 / 3 / 4 by unit size) is a routed idea, not implemented; the test level marks
  the intended ogryn zombie with a `# TODO` rather than a larger footprint.
- **Articulated / deforming footprints.** The `occupied_tiles()` method contract is
  shaped to absorb a segment-chain (snake) or deforming body with no consumer change,
  but no such unit exists; only the rigid square case is implemented.
- **Facing-rotated footprints.** `occupied_tiles()` applies no rotation — fine for
  the symmetric 3×3 default, deferred for any asymmetric rigid shape.

**Gaps / inconsistencies:**

- **`coords.py` conversion helpers are under-used.** Most call sites still floor
  inline (`int(y), int(x)`) or read `unit.tile_x` directly rather than calling
  `tile_to_index`; metre conversions in physics are largely open-coded rather than
  routed through `tile_to_meters` / `tile_distance_m`. The helpers are the intended
  single seam, but adoption is partial — a tidy-up target, not a correctness bug.
- **`architecture.md` §3 is stale.** It still describes integer-only unit positions
  with `(fx, fy)` naming and presents `map_w`/`map_h` as live config. This chapter
  supersedes it: positions are float `x` / `y`, the dual-grid vocabulary is gone, and
  grid size comes from the level CSV.
- **`light_map` (scalar) coexists with `light_rgb`.** A legacy scalar light field is
  still allocated alongside the per-channel field during the RGB migration; it is a
  field-set concern (the State/Material chapters), noted here only because both are
  grid-shaped arrays on the same grid.
