# Levelgen: cycle types and the department recipe format

> **Status**: design, **revision 2** (2026-10-31 evening, lead
> `level-design-lead`). Revision 1 was written in the afternoon session with
> Erik in the loop; the adversarial critique
> (`levelgen_cycle_types_and_recipes_2026-10-31_critique.md`, Opus, 20
> findings, every "exists" claim checked against `tools/gen_ship.py`) is
> folded in here. Serves #136 (adjacency grammar) inside milestone 11 (#131).
> Design only: no code. Follows the grammar session
> (`docs/breach_levelgen_design_v0.1.md` §9, family B) and the vocabulary doc
> (`docs/levelgen_zone_vocabulary_2026-10-29.md`, Parts 0 and B). The file
> keeps its name so links hold.
>
> **What revision 2 changed, in one paragraph.** The no-triangle rule now
> runs on the CONTACT graph over every edge kind, and the retreat loop and
> the maintenance ring were reshaped to obey it. The duct link lost its
> "direction" (the plenum is well mixed and every vent runs at one rate) and
> its leash. The airlock bypass is legal under an amended L5 (Erik's ruling,
> §4). The missing LAYOUT stage is named and costed. The width ruling is
> scoped to floor area; a narrow corridor is entered from its long sides.
> Recipe names resolve to `ROOM_TAGS` values through a named RESOLVER with
> preset tables; "room role" is renamed **fixture** (the tag `role` already
> means something else). `physics.air` became `pressure`; a gas mix is a NEW
> mechanism for Erik's ruling. The move library is stated honestly (six
> moves, face insertion unused in v1) with a phase table. The crew-quarters
> example no longer claims a loop it does not contain and has a feasibility
> check against its own ranges.

Family B's grammar is a fixed move library in code driven by three kinds of
DATA: department recipes, cycle types and an archetype recipe. This document
defines the first two. The archetype recipe is the next session.

## 0. The pieces, at four scales

The things a ship is built from, smallest first. Writing a new ship type
means writing data at scales 1.5-4, never code.

| Scale | Piece | What it is | Status |
|---|---|---|---|
| 1 | **Edge kind** | how two rooms join: door, open arch, window, breakable wall, bulkhead, airlock pair, vent-only, entity door, locked door | in code, `gen_ship.EDGE_KINDS` (#133) |
| 1.5 | **Fixture** | a room with a physics signature of its own, reusable across departments (the aquarium, the O2 store, the galley stove) | this doc, §3.3 (revision 1 called it "room role"; renamed, critique 11) |
| 2 | **Cycle type** | a shape of 0-6 rooms with a tactical meaning, the authored object of the moves | this doc, §1 |
| 3 | **Department** | a room cluster with one function and a physics signature | prose in the vocabulary doc Part B; format this doc, §3 |
| 4 | **Archetype recipe** | the build order for a whole ship: (phase, move or cycle type or department, count range) | next session |

### 0.1 The move library (critique 13, 14)

v0.1 §9 said "the four L4 moves". Honestly counted, v1 has **six**, and one
of L4's four is unused:

| Move | L4 | What it does to the graph | Planarity |
|---|---|---|---|
| `subdivide` | move 1 | A-B becomes A-C-B | safe |
| `edge_to_cycle` | move 2 | A-B becomes A-C-B plus A-D-B; **A-B is removed** (so no triangle) | safe |
| `leaf` | move 3 | adds A-N, N new | safe |
| `ring` | new (critique 13) | a hub H gets an EVEN ring of n rooms r_1..r_n around it, ring neighbours joined, H joined to the odd-numbered ring rooms only (so H-r_i-r_(i+1) is never a triangle) | safe: a wheel minus alternate spokes |
| `parallel` | new (critique 17) | a second opening of another kind on the SAME shared wall of A and B (a multi-edge; no new room) | safe: nothing crosses |
| `duct` | the L4 vent exception | joins two rooms of one cell on one duct; no wall edge | not a wall edge |
| ~~`face`~~ | move 4 | insertion into an enclosed empty face | **unused in v1**: the grammar keeps no combinatorial embedding; the ring covers the one wanted case |

Every move records provenance (rule id + application index) on what it
makes, and the invariants of §1.5 run after every application.

## 1. Cycle types

### 1.1 Three graphs, not one (Erik, 2026-10-31: "a loop in the walking path and a loop in the air path are not the same"; critique 2)

A ship has three graphs over the same rooms:

- the **contact graph**: rooms that share a wall. EVERY row of `EDGE_KINDS`
  is a contact (all are `must-share-wall`, and stage 2 refuses an edge whose
  rooms share no wall). The rectangular-layout guarantee lives here.
- the **walk graph**: contacts a unit can pass (door, open arch, entity
  door; a breakable wall after it is broken; an airlock pair in sequence).
- the **air graph**: anything gas can pass: open arches always; doors,
  windows and breakable walls only once broken or opened (a plain `door` is
  gas-tight, `MAT_DOOR`); plus every DUCT (a supply and a return on one duct
  join their rooms for gas with no wall between them).

A cycle type names a shape in the walk graph, the air graph, or both. A loop
in the walk graph is tactics (two ways to the same place). A loop in the air
graph is physics (smoke and teargas travel it). The duct link (§1.3) is the
pure air shape; the pocket is the pure walk shape.

### 1.2 Width classes (Erik's ruling, 2026-10-31; scoped by critique 8)

A marine's footprint is 3x3 tiles (one metre at the 1/3 m tile). Three tiles
is single file: two marines cannot pass. Rulings:

- **Every room and corridor FLOOR is at least 6 tiles across** (two
  abreast), the `wide` class. "We don't want only corridors where two
  marines can't even meet and bypass each other, that'll be frustrating to
  play." The ruling is about floor area. **Openings stay what they are**:
  a door is `DOOR_W` = 4, an airlock `AIRLOCK_W` = 5, both single file,
  like real doors (Claude's scoping, vetoable, §4).
- **ONE exception**: the flank pair's service leg may be 3 tiles across,
  the `narrow` class. Single file is the price of the dark shortcut; that
  price is what makes it a choice and not a free second corridor. Erik: "3
  tiles wide for flank pair is no problem."
- A narrow corridor is **one room, entered from its long sides**: a 4-tile
  door cannot be cut in a 3-tile end wall, so the legs that join it meet its
  long sides. The loader refuses `fuel` on a narrow room (a cluster against
  its wall would block it).
- Corridors are rooms (the research pass, #132): one node type, told apart
  by `shape = "corridor"`. **gen_ship has no corridors today** (the middle
  band's sections are ordinary rooms, and nothing carves a corridor in a
  wing): long thin rooms are placed by the LAYOUT stage (§5.1), and `embed`
  needs no change for width.

### 1.3 The first six types (Erik: "I'd love to have all of them in", 2026-10-31)

Each type has: the move that realises it, its contact / walk / air shape,
the edge kinds its legs may use, the width class per leg, the phases it may
be applied in, and one tactical sentence. Fields are the same for every type
so the grammar reads one format (§2).

| Type | Move | Shape | Air | Phase | Tactic |
|---|---|---|---|---|---|
| **Retreat loop** | `edge_to_cycle` | A-B is REPLACED by A-C-B and A-D-B: a 4-cycle, all legs walkable, plain doors or arches, `wide` | follows the walk graph through arches only; doors are gas-tight | 2 | A hub with two ways back toward your own end (the grammar reads the department's `toward` gradient): you can always fall back without being cornered. |
| **Flank pair** | `edge_to_cycle` | two corridors between the same hubs: one `wide`, `light = "lit"`, 1-3 rooms; one `narrow`, `light = "dim"`, exactly ONE room entered from its long sides | each leg may carry its own duct | 2 | The slow team takes the dark way round, single file. |
| **Airlock bypass** | `parallel` across a cell boundary | A and B share ONE wall carrying two openings: the airlock pair (its vestibule carved on A's side, as `_airlock` does today) and a `breakable` segment. No new room. | the breakable leg joins the two cells for gas and units once demolished; until then the boundary holds | 2 | You cannot walk around the airlock; a charge on the wall makes a second door, and the cells breathe together. |
| **Maintenance ring** | `ring` | an even ring of 4, 6 or 8 small rooms around a hub; ring neighbours joined by doors or arches (one kind per application); the hub joined to the odd-numbered ring rooms by doors; all `wide` | the ring's own duct (supply + return per ring room) or, with arches, one compartment | 2 | Service closets around the engine room: a fire in one smokes the whole ring, and the ring is a flank around the hub. |
| **Duct link** | `duct` | NO wall edge: the two rooms need not touch; precondition: both in one cell | one duct carrying both rooms, each with supply + return (the sealed-room rule): a shared, well-mixed plenum | 4 | A grenade in one room comes out of the grille in the other, and some of it comes back. What a poison grenade exploits. |
| **Pocket** | `leaf` | one room, one opening, `wide`; the opening is a `door`, `entity_door` or `open` arch | through a door: sealed, its air arrives and leaves only by its vents, if any (supply + return or none); through an arch: gas that enters POOLS, nothing flows through | 1 (inside a department) or 2 (ship level) | Safe room and trap at once: nobody flanks you, nobody relieves you. Its contents wait for #135. |

Notes:

- The pocket and the duct link are "cycle types" by file format, not by
  topology; the name of the scale-2 piece stays `cycle` because the
  grammar's authored object is a shape with legs, and a shape may have one
  leg or none.
- **The duct link has no direction** (critique 4-6). The plenum is well
  mixed and every vent gen_ship writes runs at one rate (`Q_CIRC`), so two
  rooms on one duct gas each other equally. The honest air shape is "shared
  plenum". A per-role RATE CLASS (a named preset, never a number in the
  recipe) that would make a link one-sided is a later knob and a named
  stage-2 extension (§5.2); it is not in v1. There is no leash: ducts have
  no geometry, so two rooms on one duct behave the same at any distance.
  The v1 "leash" is the research pass's: same cell, and a wall site in each
  room (already a `_Reject` in `_build_vents`).
- A room on TWO ducts (its department's and a link's) needs the `vent` tag
  to become a list (§5.2). Inside one department the link is simply the
  department's own duct (B.1's heads-to-bunks trick IS the quarters duct);
  the interesting link is cross-department, and that is what the list
  extension buys.
- The airlock bypass's breakable leg is a secret path in the sense of
  `EDGE_KINDS` (it excuses an unreached compartment only when every room
  behind it is `secret`); when the far side is an ordinary cell reached by
  its airlock, the breakable leg is simply a second way, after demolition,
  for gas and units. A `vent_only` leg across the boundary (gas without
  demolition) is NOT in v1: it needs the damper entity (#49) and the
  archetype flag `allow_cross_cell_leak` (L5's leak-path clause); the
  format accepts the kind and the loader refuses it naming #49.
- ACCEPTED GAP for v1 matches: the bots' A* does not route through an
  airlock's closed outer door (#142), so in a bot match the bypass's walk
  leg is unused until #142 lands.
- **Spawn room exits are an archetype parameter, not a type**: the
  archetype recipe hands the GRAMMAR how many exits a spawn room has, range
  1-3, default 2 (Erik: "spawns with two exits are perhaps even more
  interesting, and three exits are even more interesting than that"). A
  one-exit spawn is a pocket. (`[[spawn_room]]` rows carry no exits; the
  exits are graph edges the grammar makes.)

### 1.4 Not a cycle type

The **aquarium** (Erik asked): a sealed room full of water behind a window
edge onto a corridor; burst the glass and the water floods the corridor. That
is a ROOM with a physics signature, scale 1.5, not a loop shape. It is a
fixture (§3.3) used by the pool / aquarium hall department (vocabulary B.7).
Its `water` key is in the format now; the level layer EXISTS
(`water_init.npy`, written through `level_lib`), the per-room STAMP in
gen_ship does not (§5.2).

### 1.5 The invariants, and the graph each runs on (critique 2, 14)

Evaluated after every move application; a failure names the rule.

| Invariant | Graph | Source |
|---|---|---|
| **No 3-cycle of contacts** (any edge kinds). A wanted three-room corner is a prefab (#133). | contact | research pass B7, Q6: triangle-free planar => a rectangular layout exists |
| Every cell-boundary crossing is an airlock pair, a bulkhead or a breakable wall (L5 as amended, §4); never a door, arch or window | contact | L5 |
| A duct never crosses a cell; a room whose every contact is gas-tight carries both a supply and a return on each of its ducts, or none | air | vocabulary §0.4-0.5 |
| Mirror: every wing room and edge has its twin; axis rooms are self-twins | all | vocabulary §0.3 |
| Every room reachable in the walk graph from both spawns (a `secret` room through a secret-path edge) | walk | `validate` |

## 2. The cycle file

`levels/recipes/cycles/<name>.toml`. One cycle type per file. A leg is the
chain of rooms (possibly none) between the two anchors A and B; a leg's
`edges` lists the kinds its openings may use, drawn per application.

```toml
[cycle]
name = "flank_pair"
move = "edge_to_cycle"          # subdivide | edge_to_cycle | leaf | ring | parallel | duct
phase = [2]                     # phases an archetype recipe may apply it in; one ENTRY applies it in one phase
tactic = "Two ways between the same hubs; the slow team takes the dark way round."

[[leg]]
walk = true
rooms = [1, 3]                  # inclusive range of rooms on this leg
width = "wide"                  # wide (floor >= 6) | narrow (floor 3; one room; entered from its long sides)
light = "lit"
edges = ["door", "open"]        # edge kinds this leg's openings may use

[[leg]]
walk = true
rooms = [1, 1]
width = "narrow"
light = "dim"
edges = ["door"]
```

The retreat loop is the same file with two `wide` legs of 1-2 rooms each.

The airlock bypass (no rooms; two openings on one wall):

```toml
[cycle]
name = "airlock_bypass"
move = "parallel"
phase = [2]
crosses_cell = true             # the ONLY type allowed to; its walk opening must be an airlock pair (L5)
tactic = "You cannot walk around the airlock; a charge on the wall makes a second door."

[[leg]]
walk = true
rooms = [0, 0]
edges = ["airlock"]

[[leg]]
walk = false
rooms = [0, 0]
edges = ["breakable"]           # vent_only accepted by the schema, refused by the loader until #49 (damper) lands
```

The maintenance ring:

```toml
[cycle]
name = "maintenance_ring"
move = "ring"
phase = [2]
rooms = [4, 8]                  # even; an axis hub puts its fore and aft ring rooms ON the axis
tactic = "Service closets around the engine room: one fire smokes the ring, and the ring flanks the hub."

[[leg]]                         # ring neighbour to ring neighbour
walk = true
edges = ["door", "open"]        # one kind per application

[[leg]]                         # hub to the odd-numbered ring rooms
walk = true
edges = ["door"]
```

The duct link has no `[[leg]]` and one `[[duct]]`:

```toml
[cycle]
name = "duct_link"
move = "duct"
phase = [4]
tactic = "A grenade in one room comes out of the grille in the other, and some of it comes back."

[[duct]]
roles = ["supply", "return"]    # each room carries both (the sealed-room rule); one shared plenum
filter = "derelict"             # a [filters.*] row by name
```

The pocket:

```toml
[cycle]
name = "pocket"
move = "leaf"
phase = [1, 2]
tactic = "Safe room and trap at once."

[[leg]]
walk = true
rooms = [1, 1]
width = "wide"
edges = ["door", "entity_door", "open"]
```

**Loader rules**: `move` a row of §0.1 (never `face` in v1); every `edges`
entry a row of `EDGE_KINDS`; `width` a known class, `narrow` only with
`rooms = [1, 1]`; `filter` a `[filters.*]` row; each phase within 1-4;
`crosses_cell` refused on any type whose walk legs are not all `airlock`;
a `walk = false` leg uses a kind that CAN pass gas, through a duct
(`vent_only`) or once broken (`breakable`, `window`), and the loader says
which; `ring` needs an even `rooms` range. Refusal by name, at load, before
any ship is planned.

## 3. The department file

`levels/recipes/departments/<name>.toml`. The six fields of the vocabulary
doc §0.2 (rooms, adjacency, cell, ducts, physics signature, secrets) become
tables. **A recipe never contains a physics number**: where it needs one it
NAMES a preset or config row, so Erik tunes physics in config.toml and a
recipe cannot smuggle a value in. The RESOLVER (§3.4) turns names into the
`ROOM_TAGS` values gen_ship already checks.

### 3.1 Crew quarters, the worked example (vocabulary B.1; critique 1, 9)

Revision 1 claimed an internal retreat loop; the edge list was a star around
the stub, a tree. The quarters are a tree on purpose (one entry, bunks and
mess open onto the stub), so the claim is gone. A retreat loop is a
SHIP-level type the archetype recipe applies between departments' entries.

```toml
[department]
name = "crew_quarters"
placement = "wing"              # wing = a mirror pair, one per wing | axis = one, on the mirror line
cell = "main"                   # main = the accommodation cell (both wings + the spine band, v1's one big cell) | own = its own cell (L5 on every boundary edge)
entry = "stub"                  # the room that joins the ship's spine
toward = "spawn"                # spawn | axis: the gradient it wants (vocabulary §0.3)
wish_near = ["medbay"]
wish_far = ["engineering"]

[[room]]
id = "stub"
shape = "corridor"              # room | corridor | vestibule
required = true
width = "wide"
size = { across = [6, 8], along = [20, 24] }   # corridors: (across, along); the layout stage rotates
vent = { duct = "quarters", roles = ["return"] }

[[room]]
id = "bunk"
shape = "room"
count = [2, 3]                  # instances; a required room has count [1, 1]
size = { w = [6, 8], h = [6, 8] }             # rooms: (x, y) extents in tiles
fuel = "furniture"              # a fuel class (§3.4)
light = "dim"
vent = { duct = "quarters", roles = ["supply"] }

[[room]]
id = "heads"
required = true
size = { w = [6, 6], h = [6, 8] }
vent = { duct = "quarters", roles = ["return", "supply"] }   # a sealed box carries both

[[room]]
id = "mess"
required = true
size = { w = [9, 12], h = [9, 12] }
fuel = "furniture"

[[room]]
id = "galley"                   # the kitchen
p = 0.7                         # optional room, probability of presence
size = { w = [6, 9], h = [6, 9] }
fixture = "galley"              # §3.3: a stove on from tick 0 (Erik's ruling), kindling
vent = { duct = "galley", roles = ["supply", "return"] }

[[room]]
id = "laundry"
p = 0.3
size = { w = [6, 6], h = [6, 8] }

[[edge]]
a = "stub"
b = "bunk"
kind = "open"                   # every bunk instance; the quarters breathe as one compartment

[[edge]]
a = "stub"
b = "mess"
kind = "open"

[[edge]]
a = "stub"
b = "heads"
kind = "door"

[[edge]]
a = "mess"
b = "galley"
kind = "door"

[[edge]]
a = "stub"
b = "laundry"
kind = "door"

[[duct]]
name = "quarters"
filter = "derelict"             # passes teargas: the heads' air comes out over the sleepers

[[duct]]
name = "galley"
filter = "derelict"             # its shut-off is #49

[secret]
kind = "breakable"
host = "bunk"
weight = 1                      # a weight in the SHIP's secret draw (the archetype's budget), not a probability

[physics]
temp = "ambient"                # a [temps.*] preset or "ambient"; resolves to the temp tag (temp_init.npy EXISTS)
pressure = "ambient"            # a [pressures.*] preset or "ambient"; resolves per room into air_init.npy (EXISTS: pressure only)
```

**Feasibility (critique 9).** Every edge is a contact, so every room that
joins the stub needs its own run of the stub's perimeter: at least 6 floor
tiles plus a wall each, the mess 9. The loader computes the department's
MAXIMUM contact demand on each hub (here 3 bunks + mess + heads + laundry =
3·7 + 10 + 7 + 7 = 45 tiles) and refuses a recipe whose demand exceeds the
hub's MAXIMUM perimeter less its entry (here 2·24 + 2·8 − 6 = 58). A draw
whose demand exceeds the DRAWN hub rerolls the draw (L8), as every other
area failure does. Revision 1's ranges (stub `along` down to 12, count up to
4) failed this at count 3.

### 3.2 Field reference

| Table / key | Meaning | Checked at load |
|---|---|---|
| `department.placement` | `wing` or `axis` | one of the two |
| `department.cell` | `main` or `own` | `own` forces every boundary edge to `airlock`, `bulkhead` or `breakable` (L5 as amended) |
| `department.entry` | a room id | exists, `required` |
| `department.toward`, `wish_near`, `wish_far` | placement wishes for the grammar (#136); routed-freely in L9 terms | department names exist in the library |
| `[[cycle]]` (optional) | `{ type, a, b }`: a cycle type applied INSIDE the department between two of its rooms, in phase 1 (critique 1, reading b) | type exists, `a`/`b` room ids exist, the type's phases include 1 |
| `room.id`, `shape`, `required` / `count` / `p` | identity, shape hint, multiplicity | exactly one of `required`, `count`, `p` |
| `room.width` | `wide` / `narrow`, corridors only | `narrow` only on a corridor, never with `fuel` |
| `room.size` | rooms `{ w, h }` (x, y extents); corridors `{ across, along }` (the layout stage rotates) | each min >= 6 for a room; a corridor's `across` within its width class |
| `room.fixture` | a fixture (§3.3) | fixture exists |
| `room.fuel` | a fuel class (§3.4) | class exists |
| `room.light` | a light preset name: `dim` / `lit` / `dark` / `harsh` (§3.4) | preset exists |
| `room.vent` | `{ duct, roles }` or a LIST of them (§5.2) | duct declared; a sealed room carries both roles or none, per duct (vocabulary §0.4) |
| `room.water` | `full` / `pool` (the aquarium, the pool; §1.4) | format only until the per-room stamp lands (§5.2); the loader refuses to stamp it, naming the extension, never drops it |
| `edge.a`, `b`, `kind` | adjacency by edge kind | both ids exist, `kind` a row of `EDGE_KINDS`, no 3-cycle of contacts (§1.5) |
| `duct.name`, `filter` | a duct within the department's cell | `filter` a `[filters.*]` row; a duct never crosses a cell (§0.5) |
| `secret.kind`, `host`, `weight` | a hidden annex (vocabulary Part C); a `vent_only` secret rides the host's duct with both roles | `kind` a secret-path edge kind; the ship's secret budget (the archetype's) draws among departments by weight |
| `physics.temp`, `pressure` | presets by name | preset exists; `temp` never sub-ambient (parked physics ruling); a GAS MIX (O2-rich, poison) is NOT in v1: no mechanism exists (critique 12; Erik's ruling, Part E style) |

Everything the loader refuses, it refuses by name, before planning. A key
whose mechanism is not built yet (`water`, a future `air`) is ACCEPTED by the
schema and REFUSED at stamp time naming the missing extension, never dropped.

### 3.3 Fixtures (scale 1.5; critique 11)

A fixture is a room with a physics signature of its own that several
departments reuse. Revision 1 called it a "room role"; `ROOM_TAGS["role"]`
already means "the room's role in its department" (what this recipe calls
`id`), so the reusable signature is a **fixture**. It is a row in
`levels/recipes/fixtures.toml`, naming presets and config rows, never
numbers:

```toml
[galley]
heat_source = "stove"           # a [heat_sources.*] preset: heat_per_s, radius, max_temp, death_payload
heat_source_enabled = true      # Erik 2026-10-31: on from tick 0; "perhaps we'll look over this later"
fuel = "kindling"

[o2_store]
hazard = "o2_store_rupture"     # a [hazards.*] preset: payload, trigger_temp, fuse_s

[aquarium]
water = "full"
edges_out = ["window"]          # it meets the ship only through glass

[magazine]
hazard = "magazine_cookoff"
```

A department's `room.fixture` pulls the fixture's fields in; a field set on
the room overrides the fixture's. **The galley's stove and its kindling
(critique 19)**: a stove on from tick 0 beside kindling lights every galley
at a seed-independent time unless `[heat_sources.stove].max_temp` is below
kindling's `ignition_temp`. Erik ruled FUSE (§4): the loader refuses a stove preset whose `max_temp`
is below kindling's `ignition_temp`.

### 3.4 The resolver and its presets (critique 10, 12, 16)

The resolver is the ONE door from recipe names to `ROOM_TAGS` values. It
reads preset tables in config.toml, on the `[payloads.*]` pattern,
placeholders for Erik to tune:

| Recipe key | Preset table | Resolves to (`ROOM_TAGS` today) |
|---|---|---|
| `light` | `[levelgen.lights.*]` (`dim`, `lit`, `dark`, `harsh`: rgb + level) | `light = ((r, g, b), level)` |
| `fuel` | `[levelgen.fuel.*]` (a material mix + a cluster count range; `furniture` = today's 60/40 furniture/wood, `kindling` = kindling) | `fuel = n` today; the tag gains a `material` or class (§5.2) |
| `temp` | `[temps.*]` (degrees above ambient) | `temp = float > 0` |
| `pressure` | `[pressures.*]` (atm) | a per-room value gen_ship composes into `air_init.npy` (§5.2) |
| `heat_source` | `[heat_sources.*]` (the four keys) + `enabled` | the `heat_source` tag; it gains `enabled` (§5.2) |
| `hazard` | `[hazards.*]` (`payload`, `trigger_temp`, `fuse_s`) | the `hazard` tag |
| `cell` | numbering by the grammar: `main` = 0, each `own` department the next int | `cell = int` |

**Reproducibility.** The resolved stage-1 graph (names already turned into
values) is what the (recipe hash, seed) pair reproduces: the hash covers the
recipe files AND the resolved preset values, so retuning `[heat_sources.stove]`
changes the hash, as it changes the ship.

## 4. Rulings (Erik, 2026-10-31, each scoped to its case)

| Ruling | Scope |
|---|---|
| Walk and air loops are different things; both are named. | the cycle format (§1.1) |
| Every room and corridor floor >= 6 tiles across; the flank pair's service leg may be 3. | width classes (§1.2) |
| All six first types are in. | §1.3 |
| The galley's stove is ON from tick 0; every rolled galley starts warm. "A little strange if every kitchen is always warm", to be looked over once ships play. | the `galley` fixture (§3.3) |
| Spawn-room exits: 1-3, default 2, an archetype parameter. | archetype recipe (next session) |
| **L5 amended: a breakable wall is a legal third kind of cell-boundary crossing**, a deliberate weak point beside the airlock pair / bulkhead and the per-archetype vent leak path. "I accept it absolutely." | L5 (v0.1 §9 carries the dated line); the airlock bypass (§1.3) |
| **The galley is a FUSE** (Erik, 2026-10-31: "the fuse is all right for now, we can always change it later"): `[heat_sources.stove].max_temp` sits ABOVE kindling's `ignition_temp`, so every rolled galley lights itself a fixed time after the start; the loader refuses a stove preset that could not. | the `galley` fixture (§3.3) |



Claude's calls, vetoable: recipes under `levels/recipes/`; recipes name
presets and config rows and never carry a physics number; the word
**fixture**; openings stay `DOOR_W` 4 / `AIRLOCK_W` 5 (single file) while
floors are >= 6; the move library of §0.1 with `face` unused; `cell = "main"`
= both wings + the spine band as ONE cell in v1 (engineering the second);
the duct link symmetric in v1 (a rate class later); a `vent_only` bypass leg
waits for #49; the aquarium's `water` key enters the format before its
stamp; "corridor" is a shape hint, not a node type; the pocket and the duct
link share the cycle file format; `[[cycle]]` inside a department is an
INSTRUCTION (reading b), applied in phase 1.

## 5. Costs and the open list

### 5.1 The layout stage (critique 7): the largest piece, named

`embed` takes rooms that already carry `x, y, w, h` and only stamps them;
today every geometric decision is made in `plan_graph` (band, spawn slab,
BSP, then edges chosen FROM the geometry). A recipe yields the reverse: a
topology with size ranges and contact edges. Turning that into rectangles
where every contact is at least `DOOR_W` long is the open layout problem the
research pass answered for ships: **a slicing subdivision on the quotient
half (one wing + the axis band) with the recipe's size hints as weights and
chains laid along cuts so consecutive rooms share a wall; mandatory and
prefab rooms placed first; a bounded Merrell repair (slide a cut, swap two
leaves, mirrored); then reroll the seed (L8), never re-enter the grammar.**
This is **stage 1.5, "layout"**, between the grammar (stage 1) and `embed`
(stage 2). It is its own design session (v0.1 §5, the embedding session)
and its own issue; nothing in this document is buildable without it.

### 5.2 Stage-2 extensions this document needs (patches on `gen_ship`)

| Extension | Why |
|---|---|
| `vent` tag becomes a list of `{duct, roles}`; `_build_vents` places a site per (room, duct, role); sealed-room rule per duct | duct link across departments (§1.3) |
| `fuel` tag gains a class / material; `compartments()` derives "contents" from the material table (mobility > 0: furniture, wood, kindling) instead of its literal `(MAT_AIR, MAT_FURNITURE, MAT_WOOD)` | kindling today would SPLIT its room and fail the compartment gate (critique 10) |
| `heat_source` tag gains optional `enabled` | the galley fixture |
| `CELL_CROSSING_KINDS` gains `breakable` | L5 as amended |
| a per-room `water` stamp into `water_init.npy` | the aquarium, the pool (the layer exists, the stamp does not) |
| a per-room `pressure` composed into `air_init.npy` (ambient elsewhere) | the `pressure` preset (the file exists, gen_ship never writes it) |
| the resolver + the preset tables `[levelgen.lights.*]`, `[levelgen.fuel.*]`, `[temps.*]`, `[pressures.*]`, `[heat_sources.*]`, `[hazards.*]` | §3.4 |
| a `rate` class on a vent role (later, not v1) | a one-sided duct link |
| a per-room GAS MIX initial field (NEW mechanism: a per-species level field through the loader door and the energy seam) | an O2-rich pharmacy; Erik's ruling first |

### 5.3 Phase table (critique 14)

| Phase (L6) | What is applied | Invariants re-run |
|---|---|---|
| 1 grow | the axiom; departments expand (their rooms, edges, ducts, their own `[[cycle]]` rows) | §1.5 all |
| 2 patterns | ship-level cycle types between departments' rooms; spawn exits | §1.5 all |
| 3 terminalise | fixtures resolved, `secret` drawn from the archetype's budget by weight | reachability |
| 4 annotate | `duct` links; the resolver; tags written | the air-graph invariants (sealed-room, duct-in-one-cell), then stage 1.5 |

A cycle file lists the phases it MAY be applied in; an archetype recipe
entry applies it in ONE phase (L6's "a rule declares its phase" read as the
allowed set; the application is single).

### 5.4 Open for the next session

- The archetype recipe: phases, budgets per phase, which cycle types and
  departments at which counts, spawn exits, the secret budget (one mirrored
  pair per ship, Part F ruling 4), target room counts.
- The layout stage's design session (§5.1).
- Dead-end contents (#135): what makes a pocket worth entering.
- Whether a department recipe may contain another department (a two-level
  nonterminal) or stays flat. Draft: flat for v1.

## Systems

**(a) Existing canonical systems this design uses** (CLAUDE.md table):
Level generators (`gen_ship.EDGE_KINDS` is scale 1; `ROOM_TAGS` is what the
resolver writes to; stage 2 `embed` + `validate` stamp the laid-out graph;
the recipe loader and the resolver are new stage-1 inputs, the layout stage
is new stage 1.5, never a parallel generator); Vent / duct system and
Filter table (`duct.filter` names a `[filters.*]` row; one rate per vent
today); Timed charge with `trigger = "heat"` and Heat source entity (the
fixtures' `hazard` and `heat_source`); Payload executor (`[payloads.*]`
rows by name); Level data layer (`temp_init.npy`, `air_init.npy`,
`water_init.npy` through the one writer); Material tables (a fuel class is
a mix of furniture / wood / kindling rows); Config (the preset tables are
new config tables on the `[payloads.*]` pattern); Spawn rooms (exits are
graph edges the grammar makes; the writer is unchanged); Starting a fire
(the stove delivers heat; it never writes `fire`).

**(b) New systems this design creates** (draft rules, written into
CLAUDE.md only when the code exists):

- **Recipe library** -- `A ship's pieces are data under levels/recipes/:
  cycles/ (a shape with legs: move, phases, legs with walk / rooms / width /
  light / edge kinds, or a duct), departments/ (rooms, edges by kind, ducts,
  secret, physics by preset name, optional internal cycles), fixtures.toml
  (a room's physics signature by preset names). The loader refuses an
  unknown name at load; a recipe never carries a physics number; a key
  whose mechanism is unbuilt is accepted and refused at stamp time by name,
  never dropped.`
- **Resolver** -- `Recipe names become ROOM_TAGS values through one
  resolver reading the preset tables; the (recipe hash, seed) pair hashes
  the RESOLVED graph.`
- **Width classes** -- `Every room and corridor floor is wide (>= 6 tiles,
  two marines abreast); narrow (3) exists only on a one-room corridor a
  cycle type declares narrow, entered from its long sides, never with fuel;
  openings stay DOOR_W / AIRLOCK_W. Erik's ruling 2026-10-31, scoped.`
- **Three graphs** -- `The grammar keeps the contact, walk and air graphs;
  the no-triangle rule runs on the contact graph over every edge kind, L5
  on contacts crossing a cell, the vent rules on the air graph.`
- **Move library** -- `Six moves in code (subdivide, edge_to_cycle, leaf,
  ring, parallel, duct), each with precondition, phases and provenance; a
  new shape is a cycle file on these moves, never a seventh move without a
  design line.`
- **Fixture** -- `A room with its own physics signature reused across
  departments is a row in fixtures.toml; a department's room pulls it in by
  name and may override a field.`

## As built (2026-10-31, branch `136-recipe-loader`): the loader and the resolver

Built against revision 2: `tools/levelgen_recipes.py` (loader + resolver),
the recipe files under `levels/recipes/` (`cycles/` x6,
`departments/crew_quarters.toml` = §3.1 verbatim, `fixtures.toml` = §3.3
verbatim), the §3.4 preset tables in config.toml (`[heat_sources.stove]`,
`[hazards.o2_store_rupture]`, `[hazards.magazine_cookoff]`,
`[temps.hot_engineering]`, `[levelgen.lights.{lit,dim,dark,harsh}]`,
`[levelgen.fuel.{furniture,kindling}]`; every value a PLACEHOLDER), and
`tests/test_levelgen_recipes.py`. `gen_ship` is untouched; `ship_demo` is
byte-identical.

- **Home**: `tools/`, beside `gen_ship.py`, its one consumer (a stage-1
  input of the offline generator; the door's vocabulary, `EDGE_KINDS` and
  `ROOM_TAGS`, is gen_ship's). Import-light: stdlib at import, gen_ship and
  config imported lazily by the door.
- **The resolver** is `resolve_room` / `resolve_department`: recipe room ->
  `ROOM_TAGS` values (`dept`, `role` = the recipe id, `cell` (`main` = 0,
  `own` = the number the grammar passes), `light`, `fuel` (a cluster count
  drawn from the class's range by the caller's generator), `vent`,
  `heat_source`, `hazard`, `temp`), each passing gen_ship's own check, plus
  a sha256 `digest` over the recipe files AND the resolved values. It
  resolves per RECIPE room: expanding into instances (phase 1, §5.3) and
  the layout (stage 1.5, §5.1) are not built.
- **Pending, never dropped**: a key whose §5.2 mechanism is unbuilt is
  resolved and kept in `ResolvedRoom.pending` with the extension it waits
  for; `stamp_tags()` refuses it by name. Today that is the galley's
  kindling (the fuel tag has no material), a vent list, a disabled heat
  source, a pressure preset, water. So crew quarters resolves fully except
  the galley's fuel.
- **Refusals by name** (file + field): unknown fields; a number where a
  preset name belongs; a missing preset / config row (filters, lights, fuel
  classes and their materials, heat sources and their death payload,
  hazards and their payload, temps, pressures); an edge kind outside
  `EDGE_KINDS`; width class; narrow only as one room and never with fuel;
  room size floors (rooms >= 6; a corridor's `across` within its class);
  exactly one of required / count / p, their ranges (count 1..12); unknown
  cycle types, fixtures, rooms; a department `[[cycle]]` whose type lacks
  phase 1; the `face` move (unused, §0.1); an odd ring; a non-walk leg that
  cannot pass gas (the message says how each kind passes); `vent_only`
  across a cell (#49); `crosses_cell` without airlock walk legs; a 3-cycle
  of contacts; the sealed-room rule per duct; a fixture's `edges_out`; the
  §3.1 feasibility bound; the galley FUSE (a heat source capped at or below
  its fuel's lowest `ignition_temp`); an `own` cell resolved without the
  grammar's number.

Deviations and amendments (dated 2026-10-31, as built):

- `wish_near` / `wish_far` may name one of the eight departments vocabulary
  Part B blessed before its recipe exists (`BLESSED_DEPARTMENTS`), not only
  library departments: §3.1's crew quarters wishes for the medbay and
  engineering, neither written yet, so the §3.2 rule as stated would refuse
  the worked example. The pool / aquarium hall is `pool_hall` there.
- The retreat loop's file (§2 says "the same file with two wide legs of
  1-2 rooms each") has `edges = ["door", "open"]` on both legs, from §1.3.
- `[levelgen.fuel.*]` is `clusters = [lo, hi]` + `mix = {material =
  fraction}`; the resolver writes today's `fuel = n` only for the mix
  gen_ship stamps (60/40 furniture / wood, `TODAYS_FUEL_MIX`), anything
  else is pending. `[temps.*]` is `above_ambient`, `[pressures.*]` `atm`
  (no row shipped), `[levelgen.lights.*]` `rgb` + `level` (`lit` ==
  gen_ship's lamp).
- `[heat_sources.stove].max_temp` = 320 (above kindling's 280: the fuse).
- The feasibility bound is checked for EVERY room against its own edges
  (the entry less one 6-tile run), at each neighbour's maximum count and
  least floor side + 1, as §3.1 computes it for the stub; the secret annex
  is not counted (its size is the archetype's).
- `cell = "own"`'s L5 rule has nothing to check at load (a recipe has no
  boundary edges; the entry's join is the grammar's).
- The hazard / water fixtures load and resolve; no shipped department uses
  them.

Not touched (§5): the archetype recipe, the layout stage (§5.1), every §5.2
stage-2 extension, the phase-1 expansion into room instances, dead-end
contents (#135), nested departments.
