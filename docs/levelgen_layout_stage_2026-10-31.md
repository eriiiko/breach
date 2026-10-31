# Levelgen: the layout stage (stage 1.5) -- from a resolved recipe graph to rectangles

> **Status**: design, revision 1 (2026-10-31 night, lead `level-design-lead`,
> Claude alone; Erik on a break). A computational-method design, so the
> lead's call under the mandate; every game-feel consequence is listed in
> §9 for Erik's veto. Critique pass pending. Serves #136 / #133 inside
> milestone 11 (#131). Design only: no code.
>
> Reads: `docs/levelgen_cycle_types_and_recipes_2026-10-31.md` (rev 2, whose
> §5.1 names this stage as the largest unbuilt piece), the research digest
> strand (b) (`docs/research/levelgen_litsearch_2026-10-29.md`: B6 Mirahmadi
> / Merrell, B7 Buchsbaum, "Recommendation for Breach (strand b)"),
> `docs/breach_levelgen_design_v0.1.md` §5 (the embedding session's parked
> decisions) and `tools/gen_ship.py` (`plan_graph`, `embed`, `validate`).

## 0. The problem in one paragraph

Stage 1 (the grammar) now produces a GRAPH: rooms with size ranges and
shape hints, declared edges by kind, departments with a gradient wish,
cycle types with legs. Stage 2 (`embed`) stamps rooms that already carry
`x, y, w, h`. Today the gap is filled by `plan_graph`, which does it the
other way round: it cuts rectangles first (band, spawn slab, BSP) and
chooses edges FROM the geometry. The layout stage turns the grammar's graph
into rectangles inside a rigid hull such that every declared edge is a
shared wall long enough for its opening, the ship is mirrored, and
`validate`'s gates pass. The research pass's answer for ships is
**subdivision, not search** (hull as input; lay out one half and reflect;
chains along cuts; repair, then reroll). This document makes that concrete
as a CONSTRUCTIVE guillotine layout: a fixed skeleton, a small set of
placement primitives, one bounded repair, and reroll.

## 1. Contract

**Input** (from the grammar + resolver, the QUOTIENT half: left wing + axis
band):

- the hull: a size preset (`SIZES`: W, H, band width range, section count
  range, `max_room`) and, later, a silhouette (§8); v1 the rectangle with
  the two insets `plan_graph` draws today;
- rooms: `id`, `shape` (`room` / `corridor` / `vestibule`), size ranges
  (`{w, h}` or `{across, along}`), `width` class, `kind` (`spawn` /
  `section` / `room`), tags (resolved), `department`, `cell`, and for each
  department its `placement` (`wing` / `axis`), `entry`, `toward`;
- declared edges `(a, b, kind)`; cycle-type applications with their legs
  (which rooms are a leg, which hubs they join, the leg's width class);
- the spawn request (`max_per_team`, zombies) and the archetype's spawn
  exits count.

**Output**: a `ShipGraph` exactly as `embed` consumes it today: `Room`s
with `x, y, w, h` and `mirror`, `Edge`s with `at`, `hull`, `mirror_x`. So
`embed`, `validate` and `generate`'s retry loop are UNCHANGED; the layout
stage replaces `plan_graph` as the producer of the graph's geometry, and
`plan_graph` itself becomes the v1 archetype's fallback producer (kept for
`ship_demo`'s byte-identity until the first recipe ship replaces it).

**Guarantee**: every declared edge is realised as a contact of length
>= its opening (`DOOR_W` + 2 clearance for a door, `AIRLOCK_W` + 2, `GLASS_W`
+ 2); every room is inside the hull and >= its minimum size; the left wing
and the axis band are laid out once and reflected. **Not guaranteed**:
that a draw fits; area is the one failure mode (B7), and it rerolls (L8).

## 2. The skeleton (fixed per archetype, v1: the freighter)

The hull interior is split, as `plan_graph` does, into the AXIS BAND
(x-centred, width `wm` drawn even from the preset's range, sections along y)
and two WINGS, mirrored. The wing is then given a fixed skeleton:

```
 hull
 +----------------------------------------------------------+
 |  S  |  dept row A (upper)       | dept | dept |          |
 |  P  +----------------------------------------------------+  axis
 |  A  |          wing passageway  (wide corridor, one room) |  band
 |  W  +----------------------------------------------------+  (sections)
 |  N  |  dept row B (lower)       | dept |          |      |
 +----------------------------------------------------------+
```

- **Spawn slab**: at the outer end of the wing, sized by `_spawn_grid` /
  `_extent` exactly as today; the slab is a room of kind `spawn`.
- **Wing passageway**: ONE corridor room (`wide`), running along x from the
  slab's inner wall to the axis band, at a y the layout draws (centre ±
  jitter within the recipe's range). It is the wing's walk spine. It joins
  the slab by the spawn's exits (§5) and the axis band's facing section by a
  door, or by the airlock pair when that section is in another cell.
- **Department rows**: the strips above and below the passageway. Each row
  is a 1D sequence of DEPARTMENT RECTANGLES along x, ordered by the
  gradient: `toward = "spawn"` departments at the slab end, `toward =
  "axis"` at the band end, the rest between, order drawn. Every department
  rectangle spans its row's full depth (y) and touches the passageway along
  its full x-extent.
- **Axis band**: sections along y as today; an `axis` department takes one
  or more consecutive sections (engineering: the engine room is the hub
  section, its ring and satellites §4.4).

The skeleton is the archetype's: a station fragment or a cavern (#145) has
another skeleton (or none, and the other embedding family). Only the
freighter's is designed here.

## 3. Placement primitives

Six operations, all guillotine cuts of a rectangle; everything below is
built from them. Walls are one tile; "contact of n" means the two air
rectangles share a wall run of >= n tiles.

| Primitive | Does | Used for |
|---|---|---|
| **strip** | cuts a corridor of width `across` from one side of a rectangle, full length | passageway, a department's stub, a hull-side service corridor |
| **pack** | fills one side of a strip with a 1D sequence of rooms along it, each room's extent along the strip drawn in its range, walls between; every room touches the strip along its whole extent | rooms off a stub; departments off the passageway (at the row level the "rooms" are department rectangles) |
| **end room** | the room at the far end of a strip, spanning the strip's `across` plus its two pack sides | the mess at the end of the stub; a hub at the end of a corridor |
| **satellite** | splits a packed room's slab along the strip's direction: parent nearest the strip, satellite behind it, contact = the parent's extent | galley off the mess, a pocket off a bunk, a secret annex |
| **two-stub** | cuts TWO parallel stubs from one department rectangle with an end room spanning both | the retreat loop inside a department (§4.1) |
| **hull strip** | a corridor of width `across` along the hull wall over several department rectangles (their back rooms shrink by `across` + 1) | the flank pair's narrow leg (§4.2) |

Rules the primitives keep: a room is never below its minimum in either
extent; a `narrow` strip is entered only through its long sides; a contact
that carries a declared edge is at least the opening plus clearance; a
contact that carries no declared edge is a blank steel wall (B7 (i): gaps
and blank contacts are allowed, so the layout only has to realise DECLARED
edges, never forbid undeclared touches).

## 4. The department packer

For a `wing` department the packer receives its rectangle (x-extent from
the row packing, y-extent = the row depth) and lays out:

1. **the entry stub**: a `strip` of `across` (drawn) cut from the
   rectangle's passageway side inward, full depth; its passageway end gets
   the entry door;
2. **the end room** (if the recipe marks one: the mess) at the stub's far
   end;
3. **pack** the remaining rooms on the stub's two sides in recipe order
   (required first, then optional by their rolled presence; `count`
   instances split across the two sides), each room's `along` extent drawn
   in its range so the side's sum fits the stub's length (draw midpoints,
   scale to fit, jitter, floor at minimums; if the minimums do not fit:
   reroll the draw, L8);
4. **satellites** behind their parents;
5. **declared edges**: each `(a, b, kind)` is checked to be a contact of
   the needed length; `at` is drawn inside the shared run with clearance.

The hub-perimeter feasibility check of the recipe doc (§3.1) is exactly
"the minimum pack of step 3 fits the maximum stub"; the loader runs it on
the recipe, the packer on the draw.

### 4.1 Retreat loop (edge_to_cycle on passageway-department)

`two-stub`: the department rectangle gets two parallel stubs, each with
its own entry door on the passageway, and the end room spans both. The
4-cycle is passageway - stub_1 - end room - stub_2 - passageway. Rooms pack
on the outer sides of both stubs and between them if the gap allows a room
(else the gap is the end room's neck). Width demand: 2·across + 2 room
depths (+ an inner depth) + walls; the row packing is told this demand.

### 4.2 Flank pair (edge_to_cycle between the slab and the axis band)

The `wide` leg is the passageway itself. The `narrow` leg is a `hull strip`
of `across` = 3 along the row's hull wall, from above the slab to above the
last department before the band, entered through its long side: a door
down into the slab (the slab touches it along its top) and a door down into
one department room near the band (that room's back wall), or into the
axis section if the strip reaches the band and the band's section touches
it along its top (then the strip ends at the band wall, which is its end
wall: no door there). The departments under the strip lose `across` + 1 of
depth, which the row depth must afford (checked before the cut). `light =
"dim"` is the strip's lamp preset; `fuel` is refused on it (recipe doc
§1.2).

### 4.3 Airlock bypass (parallel on the passageway-section wall)

The passageway's end wall is shared with an axis section. If that section's
department is `cell = "own"`, the edge is an airlock pair (its vestibule
carved on the passageway's side by `_airlock`, as today) and the bypass puts
a `breakable` run on the SAME wall: the wall must be >= `AIRLOCK_W` + 2 +
`DOOR_W` + 2 (= 13) long, which fixes the passageway's minimum `across` at
13 when the bypass is applied there; otherwise the bypass is applied on a
department-room-to-section wall instead (a room packed at the band end of a
row touches the band along its depth). The layout tries the passageway
first, then the row's band-end rooms, then rejects the application (not the
ship).

### 4.4 Maintenance ring (ring on an axis hub)

The hub is a section of the axis band. The ring rooms are cut INSIDE the
band around the hub: the band's width `wm` is split as ring depth + 1 + hub
+ 1 + ring depth, the hub's section is extended by one ring depth + 1 fore
and aft; so the ring is 2 side rooms (each split into 1-2 along y: that
gives 4 or 6 ring rooms with the fore and aft rooms) plus the fore and aft
rooms, which sit ON the axis and are their own twins. The hub joins the
odd-numbered ring rooms (recipe doc §0.1), the ring rooms join their
neighbours. Band width demand: 2·6 + 2 + hub minimum; the archetype's band
range must afford it or the loader refuses the ring on that preset.

### 4.5 Pocket (leaf)

A `satellite` behind a room, or a packed room on a stub side, with its one
door. A spawn with one exit is a pocket by construction.

### 4.6 Duct link

No geometry. `_build_vents` already places a site per role; the only layout
duty is that each linked room has a steel wall site (a `_Reject` today).

## 5. Spawn exits (the archetype parameter)

The slab's exits are its declared edges: 1 = one door onto the passageway;
2 = passageway + one department room of a row (the slab touches the first
department rectangle of each row along the slab's inner wall above / below
the passageway); 3 = passageway + both rows' first rooms. A `hull strip`
door counts as an exit when the flank pair is applied.

## 6. Row packing and area

The wing's length (slab inner wall to band) is shared by each row's
departments. Each department states its x-demand range from its own
packing (stub `across` + two room depths + walls, or the two-stub demand).
The row packer draws each department's x-extent in its range so the row's
sum equals the wing length exactly: midpoints scaled to fit, jittered,
floored at minimums; filler between departments is given to the
departments (never a void: a void would be a sealed box with no vents,
which `validate`'s compartment gate would count as a lost room). If the
sum of minimums exceeds the wing length, the draw rerolls. The archetype
loader runs the same check on recipe minimums versus the preset's smallest
wing: a recipe that can NEVER fit is refused at load, by name.

## 7. Repair, then reroll (L8)

Constructive placement makes almost every declared edge a contact by
construction (a packed room touches its strip along its whole extent). The
misses are satellites whose parent is shorter than the needed opening and
rows whose jitter left a contact one or two tiles short. ONE bounded repair
pass, Merrell's wall slide restricted to our cuts: for each short contact,
slide the nearest guillotine cut by up to `REPAIR_SLIDE` (3) tiles toward
lengthening it, if every room on both sides stays >= its minimum and no
other declared contact drops below its opening; mirrored by construction
since the half is repaired before reflection. No swap move in v1 (identity
swaps break the gradient order). If a contact is still short: reject the
draw (`_Reject`), the attempt rerolls under `MAX_ATTEMPTS`, as today. Never
re-enter the grammar (L8).

## 8. Hull silhouettes (later, the research pass Q1)

v1 keeps the rectangle with insets. A silhouette library (notched,
stepped bow) decomposes a hull into a few rectangles (bow, midbody, stern)
that become the top-level cuts: the skeleton then runs per rectangle with
the passageway continuous through them. Named here so the skeleton is
written to take a LIST of hull rectangles from the start; not v1.

## 9. Feel consequences for Erik (vetoable)

- Every freighter wing has ONE wide passageway with departments as side
  streets off it. Variety comes from the departments, their order, the
  stub geometry, the cycle types and the band; the passageway itself is
  always there. (A second skeleton, e.g. a ring passageway, is another
  archetype.)
- The narrow flank runs along the HULL: breaching it from outside is
  breaching the shortcut. Probably good.
- Spawn exits 1-3 as stated in §5 (the ruling: default 2).
- The axis band carries engineering and its ring; the wings never do.

## 10. The triangle rule, read against the layout (a nuance for the recipe doc)

The recipe doc §1.5 forbids a 3-cycle of DECLARED edges. Two notes so the
rule is applied as meant:

- it is the DECLARED edges that form the contact graph the rule runs on,
  never every physical touch (two bunks side by side share a wall without an
  edge; that is fine and inevitable);
- the theorem (B7) forbids only FILLED triangles; a triangle of three
  rectangles meeting at a T (a room spanning two consecutive corridor
  sections, both declared) is not filled and is realisable. The grammar
  keeps the strict rule because its moves are unguided; the LAYOUT's own
  primitives never create a triangle the layout cannot realise. If an
  archetype ever wants sectioned corridors with rooms spanning two
  sections, the rule gets a named exception ("the T") then, not now.

## 11. Costs

| Piece | Where | Size |
|---|---|---|
| the skeleton + six primitives + row packer + department packer | new module `tools/ship_layout.py` (stage 1.5), pure, seeded, numpy-free integer geometry | the arc's main patch, ~600 lines + property tests (every primitive: contacts realised, minimums kept, inside the rectangle; the mirror; the reroll) |
| cycle-type realisations §4.1-4.5 | same module, one function per type keyed by the cycle file's `move` | ~250 lines |
| the repair pass | same module | ~80 lines |
| `generate` grows a `recipe=` path: grammar -> resolver -> layout -> `embed` -> `validate`; `plan_graph` stays for the recipe-less path | `tools/gen_ship.py` | small; `ship_demo` byte-identical gate |
| the recipe loader + resolver | the implementing chat's `136-recipe-loader` | separate patch; this stage consumes its output |

Determinism: every draw from the one `rng` in a fixed order (rows, then
departments in gradient order, then rooms in recipe order); no set
iteration; mirrored by reflection, never by a second draw.

## 12. Open

- The stub's `across` range and the passageway's: recipe fields or preset
  fields? Draft: the department recipe gives the stub's; the ARCHETYPE
  recipe gives the passageway's (it is the archetype's skeleton).
- Whether a department may be denied a row (too deep for its rooms) and go
  to the other row: v1 rows have equal depth, so no.
- The axis band's own packer when an axis department is not engineering
  (the garden dome as an axis department): v1 band packing = sections as
  today, one department per section group.

## Systems

**(a) Existing canonical systems this design uses**: Level generators
(`gen_ship.embed` / `validate` / `generate`'s retry unchanged; `Room`,
`Edge`, `ShipGraph` as the output contract; `_spawn_grid` / `_extent` for
the slab; `_airlock` for the vestibule; `EDGE_KINDS` widths for the
openings); Spawn rooms (the slab is the `[[spawn_room]]`, placement
unchanged); Determinism gates (`ship_demo` byte-identical while
`plan_graph` stays the recipe-less producer); the Recipe library and
Resolver (the recipe doc rev 2) as the input.

**(b) New systems this design creates** (draft rules, into CLAUDE.md when
the code exists):

- **Layout stage** -- `A recipe ship's geometry is produced ONLY by
  tools/ship_layout.py (stage 1.5): a fixed per-archetype skeleton, six
  guillotine primitives (strip, pack, end room, satellite, two-stub, hull
  strip), one bounded wall-slide repair, then reroll; it emits the
  ShipGraph embed consumes and never stamps a tile. A new way to place a
  room is a primitive here, never a search or a second layouter.`
- **Declared-edge contact** -- `The layout realises every DECLARED edge
  as a contact of at least its opening plus clearance and forbids no
  undeclared touch; the grammar's no-triangle rule runs on declared edges.`
