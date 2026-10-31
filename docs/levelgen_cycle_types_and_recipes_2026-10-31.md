# Levelgen: cycle types and the department recipe format

> **Status**: design, 2026-10-31 (afternoon session, lead `level-design-lead`;
> Erik in the loop). Serves #136 (adjacency grammar) inside milestone 11
> (#131). Design only: no code this session. Follows the grammar session
> (`docs/breach_levelgen_design_v0.1.md` §9, family B) and the vocabulary doc
> (`docs/levelgen_zone_vocabulary_2026-10-29.md`, Parts 0 and B). Critique
> pass pending.

Family B's grammar is four fixed moves in code driven by three kinds of
DATA: department recipes, cycle types and an archetype recipe. This document
defines the first two. The archetype recipe is the next session.

## 0. The pieces, at four scales

The things a ship is built from, smallest first. Writing a new ship type
means writing data at scales 2-4, never code.

| Scale | Piece | What it is | Status |
|---|---|---|---|
| 1 | **Edge kind** | how two rooms join: door, open arch, window, breakable wall, bulkhead, airlock pair, vent-only, entity door, locked door | in code, `gen_ship.EDGE_KINDS` (#133) |
| 1.5 | **Room role** | a room with a physics signature of its own, reusable across departments (the aquarium, the O2 store, the galley) | this doc, §3.3 |
| 2 | **Cycle type** | a shape of 1-6 rooms with a tactical meaning, the authored object of the L4 moves | this doc, §1 |
| 3 | **Department** | a room cluster with one function and a physics signature | prose in the vocabulary doc Part B; format this doc, §3 |
| 4 | **Archetype recipe** | the build order for a whole ship: (phase, move or cycle type or department, count range) | next session |

The four L4 moves (edge subdivision, edge-to-cycle, leaf attachment, face
insertion) are the hands that place the pieces.

## 1. Cycle types

### 1.1 Two graphs, not one (Erik, 2026-10-31: "a loop in the walking path and a loop in the air path are not the same")

A ship has two graphs over the same rooms:

- the **walk graph**: rooms joined by edges a unit can pass (door, open
  arch, entity door; a breakable wall after it is broken; an airlock pair in
  sequence);
- the **air graph**: rooms joined by anything gas can pass (every walk edge
  when open, plus windows after they burst, plus every DUCT: a supply and a
  return on one duct join their rooms for gas with no walkable path at all).

A cycle type names a shape in one or both. A loop in the walk graph is
tactics (two ways to the same place). A loop in the air graph is physics
(smoke and teargas travel it). The duct link (§1.3) is the pure air cycle;
the pocket is the pure walk shape with an air consequence (gas pools where
nothing flows through).

### 1.2 Width classes (Erik's ruling, 2026-10-31)

A marine's footprint is 3x3 tiles (one metre at the 1/3 m tile). Three tiles
is single file: two marines cannot pass. Rulings:

- **Every walkable path is at least 6 tiles wide** (two abreast), the
  `wide` class. "We don't want only corridors where two marines can't even
  meet and bypass each other, that'll be frustrating to play."
- **ONE exception**: the flank pair's service leg may be 3 tiles, the
  `narrow` class. Single file is the price of the dark shortcut; that price
  is what makes it a choice and not a free second corridor. Erik: "3 tiles
  wide for flank pair is no problem."
- A room (not a corridor) is never below 6 in either dimension.

`gen_ship.embed` today has one corridor width; the width class is a per-room
field it will read (§3.2 `width`).

### 1.3 The first six types (Erik: "I'd love to have all of them in", 2026-10-31)

Each type has: the move that realises it, its walk shape, its air shape, the
edge kinds its legs may use, the width class per leg, the L6 phase it may
appear in, and one tactical sentence. Fields are the same for every type so
the grammar reads one format (§2).

| Type | Move | Walk shape | Air shape | Phase | Tactic |
|---|---|---|---|---|---|
| **Retreat loop** | edge-to-cycle | A-B and A-C-B, all legs walkable, plain doors or arches, `wide` | follows the walk graph (one compartment when arches) | 2 | A room with two doors back toward your own end: you can always fall back without being cornered. |
| **Flank pair** | edge-to-cycle | two corridors between the same hubs: one `wide`, `light = "lit"`; one `narrow`, `light = "dim"`, 1-3 rooms long | each leg may carry its own duct | 2 | The slow team takes the dark way round, single file. |
| **Airlock bypass** | edge-to-cycle across a cell boundary | ONE walk leg: an airlock pair (L5); the other leg is NOT walkable: a `vent_only` link or a `breakable` wall | the second leg joins the two cells for gas (through the duct's filter, or once the wall is broken) | 2 | You cannot walk around the airlock; smoke and teargas can. |
| **Maintenance ring** | face insertion | a ring of 3-6 small rooms around a hub, each joined to the hub by a door and to its ring neighbours by doors; `wide` or `narrow` per recipe | the ring's own duct loops every ring room (supply + return each, or the ring rides one compartment) | 2 | Service closets around the engine room: a fire in one smokes the whole ring, and the ring is a flank around the hub. |
| **Duct link** | none (the L4 vent exception, under the embedder's leash) | NO walk edge: the two rooms need not touch | one duct: room A `return`, room B `supply` (each a sealed box carries both roles, §3 rule) | 4 | The toilet-to-bunks trick as a type: a grenade in A comes out of the grille in B. What a poison grenade exploits. |
| **Pocket** | leaf attachment | one room, one door, `wide` | no through-flow: gas that enters POOLS; a sealed pocket carries supply + return or no vents at all | 1 or 2 | Safe room and trap at once: nobody flanks you, nobody relieves you. Its contents wait for #135. |

Notes:

- The pocket and the duct link are "cycle types" by file format, not by
  topology; the name of the scale-2 piece stays `cycle` because the
  grammar's authored object is a shape with legs, and a shape may have one
  leg or none.
- A duct link's two rooms are both "sealed boxes" unless an arch joins them
  to a compartment; the sealed-room vent rule (vocabulary §0.4) applies
  per room, so room A carries return + a small supply and room B supply +
  a small return. The authored intent is the DIRECTION (A's air lands in
  B); the small reverse role is the pressure safety valve.
- The airlock bypass's `breakable` leg is a secret path in the sense of
  `EDGE_KINDS` (it excuses an unreached compartment only when every room
  behind it is `secret`); when the far side is an ordinary cell reached by
  its airlock, the breakable leg is simply a second way for gas and, after
  demolition, for units.
- **Spawn room exits are an archetype parameter, not a type**: the archetype
  recipe says how many exits a spawn room has, range 1-3, default 2 (Erik:
  "spawns with two exits are perhaps even more interesting, and three exits
  are even more interesting than that"). A one-exit spawn is a pocket.

### 1.4 Not a cycle type

The **aquarium** (Erik asked): a sealed room full of water behind a window
edge onto a corridor; burst the glass and the water floods the corridor. That
is a ROOM with a physics signature, scale 1.5, not a loop shape. It is a room
role (§3.3) used by the pool / aquarium hall department (vocabulary B.7).
Its `water` tag is in the format now although the embedder cannot stamp it
yet (vocabulary Part E lists the per-room water group as unbuilt), so the
format is complete before the mechanism.

**Corridors are rooms** (the research pass, #132): one node type in the
graph. A corridor differs by its `shape` hint (§3.2) and by being the thing
the width rule applies to; never a second node type.

## 2. The cycle file

`levels/recipes/cycles/<name>.toml`. One cycle type per file.

```toml
[cycle]
name = "flank_pair"
move = "edge_to_cycle"          # edge_subdivision | edge_to_cycle | leaf | face | duct
phase = [2]                     # L6 phases it may be applied in
rooms_per_leg = [1, 3]          # inclusive range, per walkable leg
tactic = "Two ways between the same hubs; the slow team takes the dark way round."

[[leg]]
walk = true
width = "wide"                  # wide (>= 6) | narrow (3)
light = "lit"
edges = ["door", "open"]        # edge kinds this leg may use between its rooms

[[leg]]
walk = true
width = "narrow"
light = "dim"
edges = ["door"]
```

The airlock bypass:

```toml
[cycle]
name = "airlock_bypass"
move = "edge_to_cycle"
phase = [2]
rooms_per_leg = [1, 1]
crosses_cell = true             # the ONLY type allowed to; its walk leg must be an airlock pair (L5)
tactic = "You cannot walk around the airlock; smoke and teargas can."

[[leg]]
walk = true
edges = ["airlock"]

[[leg]]
walk = false
edges = ["vent_only", "breakable"]   # the grammar picks one per application
```

The duct link has no `[[leg]]` and one `[[duct]]`:

```toml
[cycle]
name = "duct_link"
move = "duct"
phase = [4]
tactic = "A grenade in the first room comes out of the grille in the second."

[[duct]]
from_role = "return"            # the room the air is drawn from
to_role = "supply"              # the room it lands in
leash = 12                      # max tiles between the two rooms (L4 vent exception; the embedder must route it)
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
width = "wide"
edges = ["door"]
```

Loader rules: `move` and every `edges` entry must be a known move / a row of
`EDGE_KINDS`, `width` a known class, `filter` a `[filters.*]` row, `phase`
within 1-4; `crosses_cell` is refused on any type whose walk legs are not
all `airlock`; a `walk = false` leg must use a gas-passing kind
(`vent_only`, `breakable`, `window`). Refusal by name, at load, before any
ship is planned.

## 3. The department file

`levels/recipes/departments/<name>.toml`. The six fields of the vocabulary
doc §0.2 (rooms, adjacency, cell, ducts, physics signature, secrets) become
tables. **A recipe never contains a physics number**: where it needs one it
NAMES a config row (a `[payloads.*]`, `[filters.*]`, `[materials.*]` row, a
heat-source or temperature preset), so Erik tunes physics in config.toml and
a recipe cannot smuggle a value in.

### 3.1 Crew quarters, the worked example (vocabulary B.1)

```toml
[department]
name = "crew_quarters"
placement = "wing"              # wing = a mirror pair, one per wing | axis = one, on the mirror line
cell = "wing"                   # wing = rides its wing's cell | own = its own cell (L5: airlocks/bulkheads on every boundary edge)
entry = "stub"                  # the room that joins the ship's spine
toward = "spawn"                # spawn | axis: the gradient it wants (vocabulary §0.3)
wish_near = ["medbay"]
wish_far = ["engineering"]
cycles = ["retreat_loop"]       # cycle types this department contains internally (here: stub + two bunks through arches)

[[room]]
id = "stub"
shape = "corridor"              # room | corridor | vestibule
required = true
width = "wide"
size = { w = [6, 8], h = [12, 20] }
vent = { duct = "quarters", roles = ["return"] }

[[room]]
id = "bunk"
shape = "room"
count = [2, 4]                  # how many instances; a required room has count [1, 1]
size = { w = [6, 9], h = [6, 9] }
fuel = "furniture"              # a fuel cluster class (gen_ship's furniture stamps)
light = "dim"
vent = { duct = "quarters", roles = ["supply"] }

[[room]]
id = "heads"
required = true
size = { w = [6, 6], h = [6, 9] }
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
role = "galley"                 # a room role (§3.3): heat source "stove", enabled from tick 0 (Erik's ruling, 2026-10-31)
fuel = "kindling"
vent = { duct = "galley", roles = ["supply", "return"] }

[[room]]
id = "laundry"
p = 0.3
size = { w = [6, 6], h = [6, 9] }

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
p = 0.1

[physics]
temp = "ambient"                # a temperature preset name, or "ambient"
air = "ambient"                 # a gas-mix preset name, or "ambient"
```

### 3.2 Field reference

| Table / key | Meaning | Checked at load |
|---|---|---|
| `department.placement` | `wing` or `axis` | one of the two |
| `department.cell` | `wing` or `own` | `own` forces every boundary edge to `airlock` or `bulkhead` (L5) |
| `department.entry` | a room id | exists, `required` |
| `department.toward`, `wish_near`, `wish_far` | placement wishes for the grammar (#136); routed-freely in L9 terms | department names exist in the library |
| `department.cycles` | cycle types realised inside the department | cycle files exist |
| `room.id`, `shape`, `required` / `count` / `p` | identity, shape hint, multiplicity | exactly one of `required`, `count`, `p` |
| `room.width` | `wide` / `narrow`, corridors only | `narrow` only on a corridor |
| `room.size` | `{ w = [min, max], h = [min, max] }` in tiles | each min >= 6 for a room |
| `room.role` | a room role (§3.3) | role exists |
| `room.fuel` | a fuel cluster class | class exists |
| `room.light` | `dim` / `lit` / `dark` / `harsh` (the `ROOM_TAGS` `light` tag) | known value |
| `room.vent` | `{ duct, roles }` | duct declared; a sealed room carries both roles or none (vocabulary §0.4) |
| `room.water` | `full` / `pool` (the aquarium, the pool; §1.4) | format only until the per-room water group lands |
| `edge.a`, `b`, `kind` | adjacency by edge kind | both ids exist, `kind` a row of `EDGE_KINDS`, no 3-cycle of door edges |
| `duct.name`, `filter` | a duct within the department's cell | `filter` a `[filters.*]` row; a duct never crosses a cell (§0.5) |
| `secret.kind`, `host`, `p` | a hidden annex (vocabulary Part C) | `kind` a secret-path edge kind; the ship's secret budget is the archetype's |
| `physics.temp`, `air` | presets by name | preset exists; `temp` never sub-ambient (parked physics ruling) |

Everything the loader refuses, it refuses by name, before planning.

### 3.3 Room roles (scale 1.5)

A room role is a room with a physics signature of its own that several
departments reuse. It is a row in `levels/recipes/roles.toml`, naming config
rows, never numbers:

```toml
[galley]
heat_source = "stove"           # a heat-source preset: heat_per_s, radius, max_temp live in config.toml
heat_source_enabled = true      # Erik 2026-10-31: on from tick 0; "perhaps we'll look over this later"
fuel = "kindling"

[o2_store]
hazard = "o2_store_rupture"     # a [payloads.*] row on a heat-triggered charge
trigger = "heat"

[aquarium]
water = "full"
edges_out = ["window"]          # it meets the ship only through glass

[magazine]
hazard = "magazine_cookoff"
trigger = "heat"
```

A department's `room.role` pulls the role's fields in; a field set on the
room overrides the role's. The heat-source preset and the temperature / air
presets are new config.toml tables (`[heat_sources.*]`, `[temps.*]`,
`[airs.*]`), placeholders for Erik to tune, the same pattern as
`[payloads.*]`.

## 4. Rulings this session (Erik, 2026-10-31, each scoped to its case)

| Ruling | Scope |
|---|---|
| Walk and air loops are different things; both are named. | the cycle format (§1.1) |
| Every walkable path >= 6 tiles; the flank pair's service leg may be 3. | width classes (§1.2) |
| All six first types are in. | §1.3 |
| The galley's stove is ON from tick 0; every rolled galley starts warm. "A little strange if every kitchen is always warm", to be looked over once ships play. | the `galley` role (§3.3) |
| Spawn-room exits: 1-3, default 2, an archetype parameter. | archetype recipe (next session) |

Claude's calls, vetoable: recipes under `levels/recipes/`; recipes name
config rows and never carry a physics number; a department may name its
internal cycle types; the aquarium's `water` tag enters the format before
its mechanism; "corridor" is a shape hint, not a node type; the pocket and
the duct link share the cycle file format.

## 5. Open for the next session

- The archetype recipe: phases, budgets per phase, which cycle types and
  departments at which counts, spawn exits, the secret budget (one mirrored
  pair per ship, Part F ruling 4), target room counts.
- Dead-end contents (#135): what makes a pocket worth entering.
- The duct link's leash: a number the embedder's router sets, not this doc.
- Whether a department recipe may contain another department (a two-level
  nonterminal) or stays flat. Draft: flat for v1.

## Systems

**(a) Existing canonical systems this design uses** (CLAUDE.md table):
Level generators (`gen_ship.EDGE_KINDS` is scale 1; `ROOM_TAGS` carries
`light`, `fuel`, `vent`, `hazard`, `heat_source`, `hull_door`; stage 2
`embed` + `validate` stamp whatever the recipe resolves to; the recipe
loader is a new stage-1 input, never a parallel generator); Vent / duct
system and Filter table (`duct.filter` names a `[filters.*]` row); Timed
charge with `trigger = "heat"` and Heat source entity (the roles' `hazard`
and `heat_source`); Payload executor (`[payloads.*]` rows by name); Level
data layer (presets `temp` / `air` resolve to `temp_init.npy` /
`air_init.npy` through the one writer); Material tables (a fuel cluster
class is furniture / kindling rows); Config (`[heat_sources.*]`,
`[temps.*]`, `[airs.*]` are new config tables on the `[payloads.*]`
pattern); Spawn rooms (exits become a parameter the archetype passes to the
existing writer).

**(b) New systems this design creates** (draft rules, written into
CLAUDE.md only when the code exists):

- **Recipe library** -- `A ship's pieces are data under levels/recipes/:
  cycles/ (a shape with legs: move, phase, legs with walk/width/light/edge
  kinds, or a duct), departments/ (rooms, edges by kind, ducts, secret,
  physics by preset name), roles.toml (a room's physics signature by config
  row names). The loader refuses an unknown name at load; a recipe never
  carries a physics number.`
- **Width classes** -- `Every walkable path is wide (>= 6 tiles, two
  marines abreast); narrow (3) exists only on a corridor a cycle type
  declares narrow (the flank pair's service leg). Erik's ruling 2026-10-31.`
- **Two graphs** -- `A cycle type names a shape in the walk graph, the air
  graph, or both; the grammar evaluates its invariants (no door 3-cycle,
  L5, the sealed-room vent rule) on the right graph.`
- **Room role** -- `A room with its own physics signature reused across
  departments is a row in roles.toml; a department's room pulls it in by
  name and may override a field.`
