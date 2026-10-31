# Levelgen: the layout stage (stage 1.5) -- from a resolved recipe graph to rectangles

> **Status**: design, **revision 2** (2026-10-31 night, lead
> `level-design-lead`, Claude alone; Erik on a break). Revision 1's
> adversarial critique (`levelgen_layout_stage_2026-10-31_critique.md`,
> Opus, 20 findings, 2 blockers, every number recomputed from `SIZES`) is
> folded in. Revision 2 is NOT re-critiqued; Erik may call one. A
> computational-method design, the lead's call under the mandate; every
> game-feel consequence is in §10 for Erik's veto. Serves #136 / #133 inside
> milestone 11 (#131). Design only: no code.
>
> **What revision 2 changed.** The skeleton gained a **LOBBY** at the band
> end of each wing (the slab's twin at the other end): every cross-structure
> edge now lands on an axiom room the grammar declares, so the layout adds
> no edge and picks no room. **Bounded steel voids are allowed** (the hull
> interior is steel; a gap is a thick wall, invisible to every gate), which
> is where the slack between hull-fixed rectangles and recipe size RANGES
> goes. The flank pair's narrow leg runs along the hull from the slab to the
> lobby, entered from above into both. The spawn slab spans the wing's
> height. Engineering is entered by a mirrored pair of airlocks from the two
> lobbies (vestibules on engineering's side), never from the passageway, so
> bots always meet through the main cell; an axis-to-axis cell crossing is a
> bulkhead or a breakable wall (airlock parity). The band's cuts are
> conditioned on the lobby and the ring. Two rng streams. A worked example
> with the real numbers (medium and large). The retreat loop is realised on
> a department's entry edge (two stubs); the recipe doc is amended for it
> and for `attach` / `exit` room keys and the quarters' stub range.
>
> **Amends** (dated lines land in the named documents in the same patch):
> recipe doc rev 2 §1.1 / §1.5 (the contact graph is the DECLARED edges),
> §1.3 (flank pair hubs = slab and lobby; retreat loop = two stubs on an
> entry edge), §3.1 (stub `along = [14, 24]`), §3.2 (`attach`, `exit`),
> §5.1 (constructive skeleton instead of a weighted slicing treemap; no swap
> move); archetype doc §1 (`on` classes, axiom rows, `sizes`).
>
> Reads: the recipe doc, the research digest strand (b) (B6, B7), v0.1 §5
> and `tools/gen_ship.py` (`plan_graph`, `embed`, `_airlock`, `validate`).

## 0. The problem in one paragraph

Stage 1 (the grammar) produces a GRAPH: rooms with size ranges and shape
hints, declared edges by kind, departments with a gradient wish, cycle types
with legs. Stage 2 (`embed`) stamps rooms that already carry `x, y, w, h`.
Today `plan_graph` fills the gap the other way round: rectangles first,
edges chosen FROM the geometry. The layout stage turns the grammar's graph
into rectangles inside a rigid hull such that every DECLARED edge is a
shared wall long enough for its opening, the ship is mirrored, and
`validate` passes. The research pass's answer for ships is **subdivision,
not search**. This document makes it concrete as a CONSTRUCTIVE guillotine
layout: a fixed skeleton whose rooms are the archetype's axiom, a small set
of placement primitives, elastic extents with bounded steel voids, one
bounded repair, and reroll.

## 1. Contract

**Input** (from the grammar + resolver, the QUOTIENT half: left wing + axis
band): the size preset (`SIZES`: W, H, band range, section range); the
graph: `Room`s with `kind`, `shape` hint, size ranges, `width` class,
`attach`, resolved tags (`dept`, `cell`, `exit`, `hull_door`, ...), their
department's `placement`, `entry`, `toward`; `Edge`s `(a, b, kind)`; the
cycle-type applications (type, legs, the rooms each leg made); the spawn
request and the spawn exits count. The axiom rooms (§2) are IN the graph,
with their roles.

**Output**: a `ShipGraph` exactly as `embed` consumes it: `Room`s with
`x, y, w, h`, `kind`, `tags`, `mirror`; `Edge`s with `at` set for every
edge; the graph's `hull` and `mirror_x`. `embed`, `validate` and
`generate`'s retry loop are UNCHANGED; `plan_graph` stays the recipe-less
producer (`ship_demo` byte-identical).

**Guarantee**: every declared edge is a contact of length >= its opening
plus the layout's clearance (one tile each side: `DOOR_W` + 2 = 6,
`AIRLOCK_W` + 2 = 7, `GLASS_W` + 2 = 4; the clearance is the LAYOUT's
convention, as `plan_graph`'s candidate filter is today, not
`_opening_tiles`', which has none); every room is inside the hull and
within its size range (soft maxima only where §3 says so); the half is laid
out once and reflected; the layout adds no edge, picks no room, drops no
application: a miss is `_Reject`. **Not guaranteed**: that a draw fits.
Area, the airlock's vestibule depth, a vent or hull-door site and the
symmetry gate can all reject, and a rejection rerolls (L8).

## 2. The skeleton IS the axiom (L3)

The freighter SKELETON is code in the layout stage; the archetype file
NAMES it (`skeleton = "freighter"`) and lists its axiom DEPARTMENTS; the
skeleton's own rooms and edges below come with it (*[archetype rev 2]*,
critique 17; a new skeleton is layout code first, a new ship kind on an
existing skeleton is a file). The skeleton supplies these quotient
rooms with fixed roles; the layout maps each role to a primitive and fails
loudly on an unknown role:

| Axiom role | Node | Where | Size |
|---|---|---|---|
| `slab` | the spawn room, `kind = "spawn"` | the outer end of the wing, spanning the wing's full height (minus a hull strip when one is applied) | x: `_extent(c)` + 0..2 as today; formation centred |
| `passageway` | ONE `wide` corridor, `shape = "corridor"` | along x from the slab to the lobby, at a y drawn inside the slab's range (§6) | across: the archetype's range |
| `lobby` | ONE `wide` room | the band end of the wing, spanning the wing's height (minus the strip) | x: the archetype's range (default 8-10) |
| `section[i]` | axis-band rooms, `kind = "section"`, self-twins | along y in the band | conditioned cuts (§5) |
| the horde | `zombie` tag on the MAIN-cell section nearest the centre, never an `own`-cell one | | |

```
 hull
 +----------------------------------------------------------------+
 |  S  | dept row A (upper)       : dept     |  L  |               |
 |  L  +--------------------------------------+  O  |  axis band   |
 |  A  |      wing passageway (one wide room) |  B  |  (sections)  |
 |  B  +--------------------------------------+  B  |               |
 |     | dept row B (lower)       : dept     |  Y  |               |
 +----------------------------------------------------------------+
```

Declared axiom edges: slab-passageway (door), passageway-lobby (door),
lobby-section (door, to a MAIN-cell section the lobby faces), and the
engineering entry: lobby-engine room (airlock pair, a mirrored PAIR, one
per wing, vestibule carved on ENGINEERING's side). The departments' entry
edges are stub-passageway. Spawn exits (§7) and cycle types (§4) add
edges the GRAMMAR declares on these roles; the layout only realises them.

Two rules the skeleton guarantees for the bots (#142, airlocks not yet
walkable): the lobby always faces a main-cell section, so the two wings
meet through the main cell with no airlock on the way; and the layout runs
a "bot-walkable between spawns" A* with airlock outer doors CLOSED as its
own acceptance, before `validate`.

The skeleton is the archetype's; a station fragment or a cavern (#145) has
another or none.

## 3. Placement primitives and elastic extents

Seven operations, all guillotine cuts of a rectangle; walls one tile;
"contact of n" = a shared wall run of >= n tiles.

| Primitive | Does | Used for |
|---|---|---|
| **strip** | a corridor of `across` from one side of a rectangle, running `along` | passageway, a stub, a hull strip |
| **pack** | fills one long side of a strip with a 1D sequence of rooms, each room's `along` drawn in its range, walls between; every room touches the strip along its whole `along` | rooms off a stub |
| **end room** | the room at a strip's far end, its wall across the strip's end; width >= the strip's `across` + 2 so the door fits, centred on the strip; the pack sides' last rooms (or steel) fill the corners | the mess at the end of the stub |
| **behind** | splits a room's slab perpendicular to the strip: parent nearest the strip, the child behind it, contact = the parent's `along` (>= 6) | galley behind the mess, a pocket, a secret annex |
| **two-stub** | two parallel stubs in one department rectangle, each with its own door on the passageway, the end room spanning both; the gap between the stubs is a packed room if >= 6 wide, else steel | the retreat loop (§4.1) |
| **hull strip** | a corridor of `across` along the hull wall over the slab, row A and the lobby, which all lose `across` + 1 of height | the flank pair's narrow leg (§4.2) |
| **ring** | an even ring of rooms around a hub section inside the band (§4.4) | the maintenance ring |

**Elastic extents (critique 1).** The hull fixes the skeleton; recipes give
ranges. Where slack goes, in order:

1. a strip's `along` and a pack side's depth absorb slack WITHIN their
   ranges (a department rectangle's depth is the row's; the stub's `along`
   is the row depth minus the end room);
2. a pack side has ONE depth = `max(w_min)` over its rooms; each room is
   deepened up to its `w_max`; what remains behind a shallower room is
   **steel** (a thick back wall);
3. the row's x-slack goes between departments as thick walls, at most
   `VOID_MAX` (3) tiles each; more slack than that rerolls the draw ("sum
   of maxima < length" is a reroll too);
4. a department too shallow for its row keeps its rooms at their maxima
   and the rest of the depth is steel against the hull.

Steel voids are legal: `embed` fills the hull interior with steel and stamps
air only in rooms (critique 17), so a void is no compartment, no vent, no
gate's business. They are kept BOUNDED (`VOID_MAX`, and only at the hull
side or between departments) because a thick wall breaches and conducts
differently and reads oddly on the map; a 1-3 tile service space does not.

Rules the primitives keep: a room never below its minimum; a `narrow` strip
entered only through its long sides; a declared contact >= its opening + 2;
an undeclared touch is a blank wall (B7 (i)); a `hull_door` room is placed
as a back room against the hull with a run >= its door and never under the
hull strip, pre-checked (`hull_door_site`'s rule) so a miss is `_Reject`,
not `GraphSchemaError`.

## 4. The department packer

For a `wing` department the packer gets its rectangle (x-extent from the
row packing §6, depth = the row's) and lays out, in this order:

1. **the stub** (`entry`): a strip of `across` (drawn) from the passageway
   side inward; its x-position in the rectangle is CENTRED unless the
   department has an `exit` room (then the stub sits so that room lands at
   the slab end, §7) or the rectangle is narrow (then flush with one side
   and the stub packs ONE side: x-demand `across + depth + 1`); its
   passageway end gets the entry door, `at` centred on the stub;
2. **the end room** (`attach = "end"`, at most one) at the stub's far end;
   the stub's `along` = row depth minus the end room's `h` minus 1, drawn
   so both stay in range (else reroll);
3. **pack** the `attach = "side"` rooms (the default) on the stub's one or
   two sides in recipe order, required first, then rolled optionals;
   `count` instances alternate sides starting with the slab side; each
   room's `along` drawn in its range so the side's sum fits the stub's
   `along` exactly (midpoints scaled, jittered, floored at minima; walls
   between; a last gap < 6 is steel);
4. **behind** rooms (`attach = "behind:<room>"`) split their parent's
   slab; the side's depth must be >= parent `w_min` + 1 + child `w_min`
   (the loader checks it against the preset's row depth);
5. **declared edges**: every `(a, b, kind)` is a contact of >= opening + 2
   by construction (a packed room touches its stub along its whole
   `along` >= 6; a behind room touches its parent along >= 6); `at` is
   drawn inside the run minus one tile each side.

For an `own`-cell wing department (the hard profile's armory) the entry is
an airlock pair: the stub is room `a` (its `along` >= 9 =
`VESTIBULE_D + 1 + FORMATION_FOOTPRINT` always holds for a stub) and its
`across` must be >= 7; the loader refuses an `own` department whose stub
range cannot.

**Loader twin.** The recipe loader runs this packer's arithmetic at the
recipe's MINIMA against the smallest allowed preset: two-bin packing of the
side rooms (not the rev-2 perimeter sum, which is necessary only), the end
room against the row depth, the behind depths. A recipe that cannot pack is
refused by name and size.

### 4.1 Retreat loop (edge_to_cycle on a department's entry edge)

The grammar replaces passageway-stub by passageway-stub_1-END and
passageway-stub_2-END, where END is the department's end room (the mess)
and stub_2 is a new corridor twin of the stub; the department's side rooms
keep their edges to stub_1 (= the authored `entry`) or are reassigned to
stub_2 by the grammar's application (alternating). The layout's `two-stub`
realises it: both stubs with their own passageway door, the end room
spanning both. x-demand `2·across + 2·depth + gap + 4`. A department without
an end room cannot take the loop (the grammar's precondition).

### 4.2 Flank pair (edge_to_cycle on slab-lobby)

The hubs are the slab and the lobby. The `wide` leg is the passageway
(one room, the axiom's). The `narrow` leg is a **hull strip** of `across` =
3 along row A's hull wall from above the slab to above the lobby; the slab,
row A's departments and the lobby lose 4 tiles of height; the strip is
entered from its long side by a door DOWN into the slab (contact = the
slab's x-extent >= 8) and a door DOWN into the lobby (contact = the lobby's
x-extent >= 8). Both anchors are axiom rooms: nothing is picked. `light`
dim; `fuel` refused. Row A's depth after the strip must still take its
departments (checked before the cut; else reroll).

### 4.3 Airlock bypass (parallel on lobby-engine room)

The lobby's band wall is shared with engineering's hub section. The axiom
edge is the airlock pair (vestibule carved on engineering's side, so the
lobby needs no depth); the bypass adds a `breakable` run on the SAME wall.
Wall demand along y: `1 + AIRLOCK_W + 1 + 1 + DOOR_W + 1 = 13` (one steel
tile each side of the outer door for the airlock's wall mounts, one tile
clearance each side of the breakable; critique 8), plus the two `at`s set
explicitly and non-overlapping: the airlock centred on the hub's y-range,
the breakable toward the slab-side end. The lobby spans the wing height
(>= 24 on medium), so the demand is on the HUB SECTION's height (§5).
Until `CELL_CROSSING_KINDS` gains `breakable` (recipe doc §5.2) the row
cannot be stamped; it is a prerequisite patch.

### 4.4 Maintenance ring (ring on the engine room)

Inside the band: `wm` is split as ring depth (6) + 1 + hub + 1 + ring depth
(6): **`wm >= 20`**, so only the large preset can draw it, and the `wm` draw
is CONDITIONED on the application (drawn from `[20, band_w.hi]` even). The
hub section's height is extended by 7 fore and aft for the fore and aft
ring rooms, which sit ON the axis (self-twins). The two side columns are
split into 1 or 2 rooms each along y, so the ring has 4 or 6 rooms (**`rooms
> 6` is refused in v1**); the side rooms are a mirror PAIR inside the band
(the band is laid out once and reflected like the wing; axis rooms are
self-twins, side rooms twins). Numbering: r_1 = fore, clockwise. The hub
joins the odd-numbered ring rooms by doors; ring neighbours join by one kind
per application. **The lobbies' engineering airlocks then land on the ring
SIDE rooms** (the grammar's application rewrites the lobby-hub edge to
lobby-r_side when it applies the ring; both are engineering's cell, so L5
holds). The hub is then 6-10 wide: a closet-sized engine room (§10).

### 4.5 Pocket (leaf)

A `behind` room or a packed side room with its one door; a one-exit slab is
a pocket by construction.

### 4.6 Duct link

No geometry. `_build_vents` needs a steel wall site per (room, role); a
miss is `_Reject` today and stays one.

## 5. The band packer: conditioned cuts (critique 5)

In order: (1) pin the lobby-facing interval: the main-cell section the
lobbies face spans at least the lobby's y-centre ± 4 (a 6-tile contact is
then guaranteed; the lobby's whole height touches the band, so the cut
just must not split the contact into two short runs); (2) pin the
engineering hub section: height >= 13 when the bypass may be applied, and
the ring's fore/aft extension when the ring is; (3) split the remaining
y-ranges with `_split_1d` at `MIN_SECTION` into the remaining `k` sections;
(4) tag `zombie` on the main-cell section nearest the centre (never the
hub; `check_request`'s capacity rule is re-derived for this packer on the
recipe path). An axis-to-axis cell crossing (the hub's fore and aft walls to
main sections) is a **bulkhead or a breakable wall**: an axis-axis opening
is centred, never drawn, and must be of EVEN width (`DOOR_W` 4, `GLASS_W`
2); `AIRLOCK_W` = 5 cannot be mirrored on an even grid, so the loader
refuses an airlock on an axis-axis edge by name (critique 6). Engineering's
cell is entered only through the lobbies' airlock pair.

## 6. Rows: assignment, packing, the passageway

- **Row depth**: the wing's height minus the passageway's `across` minus 2
  walls, split at the passageway's y; the passageway's y is drawn so that
  each row's depth >= the maximum of its departments' minimum depth demand
  (stub `along_min` + end room `h_min` + 1, or the side rooms' `along`
  packing, whichever is larger), and so that the slab keeps >= 7 tiles on
  each side of the passageway door for exits (§7). With a hull strip, row A
  is 4 shallower; "equal depth" is NOT a rule.
- **Assignment**: departments in gradient order (`toward = "spawn"` first,
  `"axis"` last, the rest in recipe order; ties in file order) are dealt to
  rows alternately starting with row A, then moved by first-fit if a row's
  minimum x-demand overflows its length. The archetype's loader requires
  >= 2 wing departments at minimum count so neither row is empty.
- **Row length** = wing air width − slab − 1 − lobby − 1. Each department's
  x-extent is drawn in its demand range (`across + 2·depth + 2`, or the
  one-sided `across + depth + 1`, or two-stub) so the row's sum plus
  `VOID_MAX` walls fits exactly; midpoints scaled, jittered, floored at
  minima; voids between departments <= 3; a department's slab-end wall is
  the slab's inner wall (exits, §7) and its lobby-end wall the lobby's
  (blank).

## 7. Spawn exits (the archetype parameter, Erik's 1-3, default 2)

Exits are declared by the GRAMMAR on roles: 1 = slab-passageway; 2 = plus
slab-`exit` room of row A's first department; 3 = plus row B's. A
department recipe marks at most one side room `exit = true` (the loader
refuses a second); the packer pins that room at the slab end of the slab
side of the stub, so it touches the slab along its `w` (>= 6). If the first
department of a row has no `exit` room, that exit is not available and the
grammar draws from the exits the ship can realise (never adds a door to a
bunk). A hull-strip door into the slab counts as an exit when the flank
pair is applied. Contact arithmetic: the slab spans the wing height, so
with the passageway at y_p the slab offers >= 7 tiles above and below it
by the §6 draw rule.

## 8. Repair, then reroll (L8)

Constructive placement makes every declared edge a contact by construction
except the cross-structure ones, where jitter can leave a run short: the
slab-exit contacts, the strip's two doors, the lobby-section contact and
the bypass wall. ONE bounded repair pass, Merrell's wall slide restricted
to our cuts: slide the nearest guillotine cut by up to `REPAIR_SLIDE` (3)
tiles toward lengthening the short run, if every room on both sides stays
>= its minimum and no other declared contact drops below its opening + 2;
on the half, before reflection. No swap move (identities carry the
gradient order and the `exit` pin; amendment to the recipe doc §5.1). Still
short: `_Reject`; the attempt rerolls under `MAX_ATTEMPTS`; never re-enter
the grammar.

## 9. Worked example (the numbers, critique 2)

Constants: `BAND` 3, `DOOR_W` 4, `AIRLOCK_W` 5, `MIN_ROOM` 6, formation
pitch 5 / gap 2, `max_per_team` 6 (c = 2, r = 3: slab x 8-10).

| | medium (100 × 70) | large (160 × 110) |
|---|---|---|
| interior air w / h | 92 / 58-62 | 152 / 98-102 |
| band `wm` (recipe path) | 12-14 (conditioned 20-22 is impossible: **no ring on medium**) | 14-22; 20-22 with a ring |
| wing air width `(92 − wm)/2 − 1` | 38-39 | 64-68 |
| slab x / lobby x | 8-10 / 8-10 | 8-10 / 8-10 |
| **row length** | 16-21 | 42-50 |
| passageway `across` | 6-8 | 6-8 |
| **row depth** `(h − across − 2)/2` | 24-27 | 44-47 |
| one-sided department x-demand (`6 + 6..8 + 1`) | 13-15: **one department per row** | |
| two-sided (`6 + 2·(6..8) + 2`) | 20-24: fits only at the row's top | 20-24: **two per row** |
| crew quarters depth (stub 14-24 + mess 9-12 + 1) | 24-37: fits at the bottom of its ranges | fits |

So a **medium** freighter carries two departments per wing (quarters +
cargo, each mostly one-sided), a **large** one four; the **small** preset is
not a recipe ship (it stays `plan_graph`'s arena; archetype doc `sizes`).
The quarters' stub range is amended to `along = [14, 24]` so the department
fits a medium row (recipe doc §3.1; revision 1's `[20, 24]` could not).

## 10. Feel consequences for Erik (vetoable)

- Every freighter wing: slab -- one wide passageway -- lobby, departments as
  side streets. Variety is in the departments, their order, the stubs, the
  cycle types and the band. A second skeleton is another archetype.
- The spawn room spans the wing's height: a long room with the formation
  centred (not today's compact slab).
- The narrow flank runs along the HULL between the two big rooms: breaching
  it from outside is breaching the shortcut.
- Medium ships are small inside: two departments per wing; large ships are
  the full freighter. Recipe ships are medium and large only.
- With a ring, the engine room is a 6-10 wide closet in a ring of service
  rooms; without one it is a whole band section.
- Thick steel walls (1-3 tiles) appear behind back rooms and between
  departments.
- Long rooms (a 50-tile passageway) are lit by ONE lamp of range 16 today:
  their ends are dark until the per-length lamp rule lands (§11).

## 11. Costs

| Piece | Where | Size |
|---|---|---|
| skeleton (axiom roles), seven primitives, elastic extents, row packer, department packer, band packer, exits | new `tools/ship_layout.py` (stage 1.5), pure, seeded, integer geometry | ~800 lines + property tests (per primitive: declared contacts realised with clearance, minima kept, inside the rectangle, voids bounded; the mirror; the reroll; the bot A*) |
| cycle-type realisations §4.1-4.5 | same module, one function per `move` | ~250 lines |
| repair pass | same module | ~80 lines |
| the recipe path's `check_request` (skeleton arithmetic, zombie capacity) and the loader's feasibility twin (§4) | `ship_layout.py`, called by the loader | ~100 lines (a coupling: the loader imports the layout's arithmetic, never a copy) |
| `generate(recipe=)`: grammar (own stream) -> resolver -> layout -> `embed` -> `validate`; `ShipPlan` carries the recipe hash; schema errors on the recipe path are bugs, site misses are `_Reject` | `tools/gen_ship.py` | small; `ship_demo` byte-identical gate |
| prerequisites from other patches | `CELL_CROSSING_KINDS` + `breakable` (recipe doc §5.2); the grammar executor (archetype doc §7); the recipe loader + resolver (the implementing chat) | |
| stage-2 extension: lamps per length (one lamp per 12 tiles of a room's long side, centred on each span) | `embed`'s lights | small; a feel item (§10) |

**Determinism (critique 14).** Two streams: the GRAMMAR draws once from
`SeedSequence([seed, 0])` (so the resolved graph is the (recipe hash, seed)
function the recipe doc promises); layout + `embed` draw from
`SeedSequence([seed, 1, attempt])` per attempt, as `generate` does today
(the recipe-less path keeps `SeedSequence([seed, attempt])` untouched).
Ordered draw list, one stream, per attempt: insets; `wm` (conditioned);
passageway `across`; lobby x; slab x jitter; band cuts (pinned, then
`_split_1d`); passageway y; row assignment first-fit (deterministic, no
draw); per row in A, B order: department x-extents (scaled midpoints, one
jitter draw per department in gradient order), void split; per department
in that order: stub `across`, stub x (if free), end room `h`, side rooms'
`along` (one jitter per room in recipe order), behind splits, every `at`
(in edge order `(a, b)`); hull strip door `at`s; bypass `at`s; ring side
splits. No set iteration; the mirror is a reflection.

## 12. Open

- The lobby's x range: archetype field (draft 8-10) or derived from the
  wing (10 %)? Draft: a field.
- Whether a medium ship should drop the lobby to buy a second department
  per row (then the engineering airlock needs another anchor). Draft: no;
  medium is the two-department ship.
- Hull silhouettes (research Q1): v1 keeps the rectangle with insets; the
  skeleton takes a LIST of hull rectangles from day one so a notched hull
  later means more rows, not a rewrite.

## Systems

**(a) Existing canonical systems this design uses**: Level generators
(`embed` / `validate` / `generate` unchanged; `Room`, `Edge`, `ShipGraph`
as the output; `_spawn_grid` / `_extent`; `_airlock` with `a` =
engineering's section; `hull_door_site`'s rule pre-checked; `EDGE_KINDS`
widths; `_split_1d`); Spawn rooms (the slab is the `[[spawn_room]]`,
placement unchanged); Determinism gates (`ship_demo` byte-identical); the
Recipe library, Resolver, Fixtures (recipe doc) and the Archetype recipe
(archetype doc) as the input.

**(b) New systems this design creates** (draft rules, into CLAUDE.md when
the code exists):

- **Layout stage** -- `A recipe ship's geometry is produced ONLY by
  tools/ship_layout.py (stage 1.5): the archetype's axiom rooms as a fixed
  skeleton, seven guillotine primitives, elastic extents with bounded steel
  voids, one wall-slide repair, then reroll; it emits the ShipGraph embed
  consumes, adds no edge, picks no room and never stamps a tile. A new way
  to place a room is a primitive here, never a search or a second
  layouter.`
- **Declared-edge contact** -- `The layout realises every DECLARED edge as
  a contact of its opening plus one tile each side and forbids no
  undeclared touch; the grammar's invariants run on declared edges.`
- **Axis parity** -- `An opening between two axis (self-twin) rooms is
  centred and even-width; an axis-axis cell crossing is a bulkhead or a
  breakable wall, never an airlock.`
