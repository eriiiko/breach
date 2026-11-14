# Critique: `levelgen_cycle_types_and_recipes_2026-10-31.md`

Reviewer: adversarial critic (Opus, read-only), 2026-10-31.
Document: `.claude/worktrees/136-cycle-types-recipes/docs/levelgen_cycle_types_and_recipes_2026-10-31.md`.
Checked against: `docs/breach_levelgen_design_v0.1.md` (L1-L9, §9), `docs/levelgen_zone_vocabulary_2026-10-29.md` (§0, B.1, B.8, Part E),
`docs/research/levelgen_litsearch_2026-10-29.md` (strand a/b recommendations, B7, ten questions), `tools/gen_ship.py` (main checkout; identical
to the worktree's copy), `src/simulation/vent_system.py`, `config.toml`, `tools/air_paint.py`, CLAUDE.md rows "Level generators" and "Vent / duct system".

Counts: 3 BLOCKER, 12 MAJOR, 5 MINOR.

## Findings

### 1. BLOCKER -- the worked example does not contain the cycle type it declares

> `cycles = ["retreat_loop"]       # cycle types this department contains internally (here: stub + two bunks through arches)`

**Evidence.** The example's five `[[edge]]` rows are stub-bunk, stub-mess, stub-heads, mess-galley, stub-laundry (§3.1). That is a star
around `stub` plus a pendant galley: a TREE. No bunk-bunk, bunk-mess or other second path exists, so there is no loop of any kind, walk or
air. "Stub + two bunks through arches" is two edges out of one hub, not a cycle. The department also has exactly one `entry`, so "two doors
back toward your own end" (the retreat loop's tactic, §1.3) cannot hold either. Vocabulary B.1 never asks for a loop inside the quarters.

The deeper problem: the format never says what `department.cycles` DOES. Reading (a): a label the loader checks against the explicit
`[[edge]]` list (then this example fails its own check). Reading (b): a generator instruction, the grammar applies the cycle type's move
inside the department (then it must say which rooms are A and B, how cycle legs bind to recipe room ids, and whether the explicit edges are
a skeleton the move rewrites). Two implementers will pick different readings.

**Fix.** Decide (a) or (b) and write it in §3.2. If (a): state the check ("the listed edges contain an instance of each named type's
shape") and give the example a real loop, which needs four rooms (e.g. stub - bunk_1 - mess - bunk_2 - stub, bunks flanking the mess);
adding bunk_1--mess alongside stub--bunk_1 and stub--mess is a triangle (finding 2). If (b): add a `[[cycle]]` table with `type`, `a`, `b`
room ids. Either way, drop the retreat loop from crew quarters unless the department gets a second exit: a loop with one way out is not a
retreat.

### 2. BLOCKER -- the no-triangle rule is written for the wrong edge set, and two of the six types violate it

> `edge.a, b, kind | ... | no 3-cycle of door edges` (§3.2);
> Retreat loop: `A-B and A-C-B` (§1.3);
> Maintenance ring: `a ring of 3-6 small rooms around a hub, each joined to the hub by a door and to its ring neighbours by doors` (§1.3)

**Evidence.** The rule exists for one reason (litsearch B7, ~lines 443-460; ten questions Q6): a planar graph has a rectangular layout iff it
has no filled triangle, and the L4 moves stay triangle-free only "provided edge->cycle *replaces* A-B rather than keeping it alongside
A-C-B". That theorem is about CONTACTS. In `gen_ship.EDGE_KINDS` every row (door, entity_door, open, window, bulkhead, airlock, vent_only,
breakable, locked) is `L9_MUST_SHARE_WALL` (gen_ship.py:254-276), and `_opening_tiles` raises `_Reject("the rooms share no wall")` for every
kind (gen_ship.py:627-633). So every edge kind is a contact and must count; restricting the rule to `door` lets an `open` / `window` /
`breakable` triangle through, and the crew quarters' all-arch cluster is where that will happen first.

The retreat loop as tabled ("A-B and A-C-B") KEEPS A-B: the triangle A-B-C, the exact case the research says to forbid. The maintenance
ring as tabled is a WHEEL: hub-r_i-r_(i+1) is a door triangle for every adjacent pair of ring rooms, 3 to 6 triangles per ring, all violating
the document's own rule, which v0.1 §9 says is "evaluated after every application".

**Fix.** Restate the invariant as "no 3-cycle of must-share-wall edges (every `EDGE_KINDS` row today); a wanted three-room corner is a
prefab", evaluated on the CONTACT graph, not the walk graph. §1.1 should say which invariant runs on which graph (walk, air, contact); it
lists none, although the draft "Two graphs" rule claims it does. Redefine the retreat loop as L4's edge->cycle literally: A-B becomes A-C-B
plus A-D-B (a 4-cycle). Redefine the maintenance ring so the hub touches ring rooms only at alternate positions (or the ring is entered at
1-2 points), or declare it a prefab with its triangles inside.

### 3. BLOCKER -- the airlock bypass's non-walk leg is illegal under L5, vocabulary §0.5 and the code, and the document does not say it reopens them

> `ONE walk leg: an airlock pair (L5); the other leg is NOT walkable: a vent_only link or a breakable wall`; `crosses_cell = true # the ONLY type allowed to`

**Evidence.** L5: "Every edge crossing a cell boundary must be an airlock-pair or sealed bulkhead -- never a plain door"; L5 also: "a vent
crossing a cell boundary is a deliberate leak path -- forbidden or allowed per-archetype". Vocabulary §0.5: "A duct never crosses a cell
boundary" (stated as the corrected SOLAS rule). The code enforces all of it: `CELL_CROSSING_KINDS = ("airlock", "bulkhead")`
(gen_ship.py:279) refuses a cross-cell `vent_only` or `breakable` edge (gen_ship.py:588); a `vent_only` edge needs both rooms on ONE duct
(gen_ship.py:578-586); a duct whose rooms lie in two cells is refused (gen_ship.py:560-566). The vent_only leg fails three checks, the
breakable leg one, and a broken breakable leg is a hole between two cells at runtime, which is what L5 exists to prevent. Litsearch Q8:
"cross-cell ducts are archetype-opt-in only and carry a damper entity"; the damper is #49, unbuilt.
Also: CLAUDE.md "Level generators" notes bots cannot path through an airlock's closed outer door (#142), so in a v1 match the type's only
walk leg is unwalkable by every bot.

**Fix.** Either (i) say in §4 that this type REOPENS vocabulary §0.5 and uses L5's per-archetype leak-path clause; name the archetype flag
(e.g. `allow_cross_cell_leak`); add to Systems the code changes (`CELL_CROSSING_KINDS` gains `vent_only` / `breakable` only under that flag,
the duct-spans-cells refusal gains the same exemption) and the damper (#49) as prerequisite or accepted gap; or (ii) keep L5 / §0.5 intact and
make the gas leg a cross-cell `window` that leaks only by DAMAGE (still a reopening of L5 for windows, a smaller one). This is a
physics/objective decision: Erik's ruling, recorded in §4.

### 4. MAJOR -- the duct link's "direction" cannot exist under the fixed-circulation plenum

> `one duct: room A return, room B supply (each a sealed box carries both roles, §3 rule)`;
> `The authored intent is the DIRECTION (A's air lands in B); the small reverse role is the pressure safety valve.`;
> `from_role = "return"` / `to_role = "supply"`

**Evidence.** `vent_system.py` docstring (~lines 20-50): every return extracts its own fixed per-tick quantum into ONE well-mixed plenum;
the duct distributes across ALL its supply vents in proportion to each vent's fixed rate. With A = {return, supply} and B = {supply,
return} on one duct, A's air enters the plenum and is split between A's supply and B's supply by their rates; B's air goes the same way. At
equal rates the link is symmetric: half of A's grenade comes back into A, and B gasses A as much as A gasses B. Nothing makes a role "small":
gen_ship writes every vent at the constant `Q_CIRC = 16.0` (gen_ship.py:116, 1372) and the `vent` tag is only `{duct, roles}`
(gen_ship.py:304-317); no per-vent rate exists in the tag, the recipe or the cycle file. And `from_role` / `to_role` (one role per room)
contradicts the both-roles rule the same row invokes.

**Fix.** Drop "direction" as the authored intent, or make it real: a per-role rate on the `vent` tag (named as a `[vent_rates.*]`-style
preset, since recipes carry no numbers) plus the `_build_vents` extension, both in Systems. State the honest air shape: "A and B share a
well-mixed plenum; each receives the other's gas at a share set by the supply rates". Replace `from_role` / `to_role` with per-room role sets.

### 5. MAJOR -- a duct link between rooms that already ride a department duct is unexpressible: one `vent` tag per room

> `Duct link | none (the L4 vent exception ...) | NO walk edge: the two rooms need not touch | one duct ...`

**Evidence.** `ROOM_TAGS["vent"]` is ONE `{duct, roles}` per room (gen_ship.py:376-379, `_check_vent`). A bunk already on `quarters` cannot
also be on a link duct. Today two rooms "share a duct" only by both being on the same department duct, i.e. the link is possible only INSIDE
one department, where the department's own duct already does it (the "toilet-to-bunks trick" IS B.1's quarters duct). The interesting case,
cross-department (galley gas out in the medbay), needs a room on two ducts. Separately, the duct must stay inside one cell
(gen_ship.py:560-566); the document does not state that precondition for this type.

**Fix.** Name the stage-2 extension in Systems: the `vent` tag becomes a list of `{duct, roles}` (decide whether the sealed-room rule is per
duct or per room); `_build_vents` places a site per (room, duct, role). State "both rooms in one cell" as a grammar precondition.

### 6. MAJOR -- the duct leash is a number with no meaning while ducts are abstract; the document both sets it and disowns it

> `leash = 12  # max tiles between the two rooms (L4 vent exception; the embedder must route it)` vs §5: `The duct link's leash: a number the embedder's router sets, not this doc.`

**Evidence.** There is no duct router. `_build_vents` (gen_ship.py:1328-1374) only picks a steel wall tile per (room, role); the duct is a
`duct` entity with no geometry and the plenum is abstract (vent_system.py). Two rooms 80 tiles apart on one duct behave exactly like two
adjacent ones. L4's leash was defined so "the embedder can always route them through walls", meaningful only for physical ducts (the parked
z-layer). Litsearch Q8 already answered it: "the 'nearby' leash is 'same cell, wall site exists' while ducts are abstract". The file format
pins `12` while §5 says the number is not this document's: self-contradiction.

**Fix.** Remove `leash` from the v1 cycle file; define the v1 leash as Q8 does (same cell + a wall site in each room, the latter already a
`_Reject` in `_build_vents`). If a distance cap is wanted for gameplay, express it in graph hops and label it a design knob, not a routing
limit. Delete "the embedder must route it".

### 7. MAJOR -- the pipeline has no layout stage; "embed stamps whatever the recipe resolves to" hides the largest piece of work

> `stage 2 embed + validate stamp whatever the recipe resolves to; the recipe loader is a new stage-1 input` (Systems a);
> `gen_ship.embed today has one corridor width; the width class is a per-room field it will read` (§1.2)

**Evidence.** `embed` takes a `ShipGraph` whose `Room`s already carry `x, y, w, h` (gen_ship.py:415-433) and only stamps them. Every
geometric decision is made in `plan_graph` (gen_ship.py:1003-1146: band, spawn slab, BSP, then Kruskal over shared walls; edges are chosen
FROM geometry, not the reverse). A recipe yields topology + size ranges + must-share-wall edges; turning that into rectangles where every edge
is a contact at least `DOOR_W` long is the open layout problem (v0.1 §5 "Layout algorithm", parked; litsearch strand b: slicing treemap +
Merrell repair + wall-graph corridors). It is neither built nor named here. And gen_ship has no corridor at all: `corridor` in
tools/gen_ship.py matches only the L9 constant; the middle band's "sections" are ordinary rooms. "One corridor width" is false. Flank pair,
maintenance ring and the quarters' stub all need the layout stage to place long thin rooms.

**Fix.** Add to Systems (b) "Recipe layout (stage 1.5)": graph + size hints -> rectangles on the quotient half, mirrored; reroll on failure
(L8); pointer to the embedding session. Correct §1.2 to "gen_ship has no corridors; corridors are rooms placed by the layout stage; `embed`
needs no change for width".

### 8. MAJOR -- the narrow class does not fit the door width, and "every walkable path >= 6" is contradicted by every door

> `ONE exception: the flank pair's service leg may be 3 tiles`; `Every walkable path is at least 6 tiles wide (two abreast)`; narrow leg `edges = ["door"]`, `rooms_per_leg = [1, 3]`

**Evidence.** `DOOR_W = 4` (gen_ship.py:109); `AIRLOCK_W = 5` with a 5-deep vestibule (gen_ship.py:153); `open` / `breakable` / `entity_door`
also `DOOR_W` (gen_ship.py:254-273). A narrow corridor's end wall is 3 tiles: a 4-tile opening cannot be cut there (`_opening_tiles`:
"opening leaves lo..hi", gen_ship.py:637-638), and v1's door candidates need a shared run >= `DOOR_W + 2` = 6 (gen_ship.py:1081). So a narrow
leg of 2-3 corridor rooms joined end to end by doors is unrealisable; only side-entry geometry works. Conversely, every door (4) and airlock
(5) is narrower than "every walkable path >= 6": two marines never pass abreast through ANY opening, so the ruling as worded is false on day
one. Also `_place_fuel` puts 1-3-deep clusters against walls (gen_ship.py:1175-1209); one in a 3-wide corridor blocks it and the A* gate
rerolls.

**Fix.** Scope the ruling to floor area ("rooms and corridors; openings are `DOOR_W` 4 / `AIRLOCK_W` 5") and confirm with Erik that
openings stay single-file. Make the narrow leg ONE corridor room (`rooms_per_leg = [1, 1]`), or add a 3-wide opening kind as a new
`EDGE_KINDS` row. Loader: refuse `fuel` on a `narrow` room.

### 9. MAJOR -- the worked example is geometrically infeasible at the top of its own ranges

> stub `size = { w = [6, 8], h = [12, 20] }`; bunk `count = [2, 4]`, 6-9 square; mess 9-12; heads 6 x 6-9; laundry 6 x 6-9; every listed edge has `stub` as an end except mess-galley

**Evidence.** Every edge is must-share-wall (finding 2). Stub perimeter at most: two long sides of 20 + two ends of 8, one side or end
consumed by the `entry` join to the spine. Rooms along a side need >= 6 floor + 1 wall each: a 20-tile side seats three 6-rooms, or the mess
(>= 9) plus one 6-room. At count 4 with laundry: 4 bunks + mess + heads + laundry = 7 rooms needing roughly 6x7 + 10 = 52 tiles of contact
against at most ~48, before the entry. At stub h = 12 it fails at count 3. Some draws pass, many reroll; the document neither notices nor
gives the grammar a way to grow the stub.

**Fix.** Co-constrain `count` and `size`: the loader computes the minimum contact length of the department's hub star and refuses a recipe
whose minimum exceeds the hub's maximum perimeter. Or restructure the example (bunks off a second corridor, or the mess as a second hub,
which also gives finding 1 its loop).

### 10. MAJOR -- the recipe's room fields do not map onto ROOM_TAGS' value types; the resolver and its presets are unnamed

> `room.light | dim / lit / dark / harsh (the ROOM_TAGS light tag)`; `fuel = "furniture" # a fuel cluster class (gen_ship's furniture stamps)`; `fuel = "kindling"`; `heat_source_enabled = true`; `hazard = "o2_store_rupture"`, `trigger = "heat"`; `cell = "wing"`

**Evidence (gen_ship.py:370-406).** `light` is `((r, g, b), level)` numbers, not a name. `fuel` is `int >= 0`, a cluster COUNT; the material
is drawn inside `_place_fuel`, 60 % furniture / 40 % wood (gen_ship.py:1185). There is no class, and `MAT_KINDLING` is never stamped. Worse,
`compartments()` counts only `MAT_AIR, MAT_FURNITURE, MAT_WOOD` as room contents (gen_ship.py:1715): a kindling cluster would be STRUCTURE
and split its room, failing the compartment gate. `temp` is a per-room `float > 0`; the recipe has a department-level preset name. `hazard`
requires `{payload, trigger_temp, fuse_s}` (gen_ship.py:319-325); the role supplies a payload and `trigger = "heat"` only. `heat_source`
requires EXACTLY `{heat_per_s, radius, max_temp, death_payload}` (gen_ship.py:328-337): an `enabled` key is refused. `cell` is an int.
Systems (a) says ROOM_TAGS "carries" these, implying a pass-through.

**Fix.** Add a "Recipe resolver" to Systems (b): recipe names -> tag values, with the preset tables named: light presets (`dim` / `lit` /
`dark` / `harsh` -> rgb, level; render data, decide where they live), fuel classes (class -> material mix + count range), hazard presets
(`trigger_temp`, `fuse_s`), heat-source presets (the four keys), cell numbering. Name the stage-2 extensions: the `fuel` tag gains a material
or class; `compartments()` contents include `MAT_KINDLING` (better: derive "contents" from the material table, not a literal list); the
`heat_source` tag gains an optional `enabled`.

### 11. MAJOR -- "role" now means two things (one word, one meaning)

> `Room role | a room with a physics signature of its own, reusable across departments (the aquarium, the O2 store, the galley)`; `room.role | a room role (§3.3)`

**Evidence.** `ROOM_TAGS["role"] = "str: the room's role in its department"` (gen_ship.py:381), i.e. what this recipe calls `id` (`bunk`,
`heads`). v0.1 §4: "one word, one meaning ... if the existing level generator uses a word with a different meaning, either the meanings are
reconciled or one side is renamed"; the vocabulary doc renamed "zone" to "department" for exactly this.

**Fix.** Rename one side: the reusable physics signature becomes e.g. `fixture` / `signature`, or the existing tag takes the new meaning
and the recipe `id` goes into a new tag. List it in §4's vetoable calls.

### 12. MAJOR -- the `air` preset has nothing to resolve onto

> `air = "ambient" # a gas-mix preset name`; Systems (a): `presets temp / air resolve to temp_init.npy / air_init.npy through the one writer`

**Evidence.** `air_init.npy` is int32 Q16.16 ATMOSPHERES, pressure only, and "governs the WHOLE map once it exists" (`tools/air_paint.py`
docstring, lines 4-54; `tools/gen_wind_studio.py:24`). It carries no gas mix. gen_ship writes no `air_init.npy`. A per-room gas-mix seed
(O2-rich pharmacy) needs a new per-species level field through the loader door and the energy seam (`reseed_gas_energy`): a level-format
change with digest consequences.

**Fix.** Either restrict `physics.air` to "ambient" in v1 (stated), or rename it `pressure` and map it honestly onto `air_init.npy`. List a
gas-mix initial field as a NEW mechanism needing Erik's ruling (Part E style), not an existing one.

### 13. MAJOR -- face insertion's preconditions on the quotient graph are missing, and the maintenance ring is not L4's face insertion

> `Maintenance ring | face insertion | a ring of 3-6 small rooms around a hub`; `Service closets around the engine room`; `wide or narrow per recipe`; `or the ring rides one compartment`

**Evidence.** L4 move 4 is "add a cycle of rooms inside an enclosed EMPTY face". A ring AROUND an existing hub is not insertion into an empty
face; it is a wheel (finding 2). "Face" requires the grammar to carry a combinatorial embedding (rotation system); neither v0.1 §9 nor this
document says the family-B grammar maintains faces. On the quotient graph (wing + axis band) the engine room is an AXIS room (vocabulary
B.2): a ring around it straddles the mirror, so "apply every wing move to the twin by construction" (v0.1 §9) would double it or need
self-mirrored ring rooms; an odd ring (3, 5) is impossible unless a ring room sits on the axis. Engineering is its own cell, so the ring must
also stay inside it. "Narrow per recipe" contradicts §1.2 (narrow only on the flank pair's leg; a room never < 6) and the draft Width rule.
"The ring rides one compartment" contradicts "joined ... by doors" (doors bound compartments).

**Fix.** State the move's preconditions: the grammar keeps a planar embedding; a face is enclosed and bounded by rooms of one cell; an
axis-straddling face takes an even ring or an axis ring room; a wing face is inserted in the quotient and mirrored. Fix the ring's edge kinds
(doors or arches, not "or"), drop "narrow", and decide whether the hub touches every ring room (then it is a prefab, finding 2).

### 14. MAJOR -- phase assignments are under-specified against L6, and a fifth move appears undeclared

> `move = "edge_to_cycle" # edge_subdivision | edge_to_cycle | leaf | face | duct`; Duct link `Phase 4`; Pocket `1 or 2`; `cycles = ["retreat_loop"]` against `phase = [2]`

**Evidence.** v0.1 §9: "the grammar is a FIXED MOVE LIBRARY in code -- the four L4 moves"; this document's own opening says "four fixed
moves". `duct` is a fifth move in code, unacknowledged. L6 phase 4 is "Annotate (emit the full annotation schema)"; a duct link changes the
air graph and duct membership, which the invariants (sealed-room vent rule, duct-in-one-cell) must re-check: a structural rewrite in the
annotation phase. L6: "a rule declares its phase" (singular), yet `phase = [1, 2]`. The document never says in which phase a DEPARTMENT
expands. If departments are phase-1 nonterminals (v0.1 §4.1: SECTOR -> ZONE -> rooms), a department's internal retreat loop runs in phase 1
against its file's `phase = [2]`; if departments terminalise in phase 3, phase-2 cycle types cannot attach to department rooms that do not
exist yet.

**Fix.** Add a phase table: phase 1 departments expand (their internal cycles either exempt from the cycle file's `phase` or applied in
phase 2 explicitly); phase 2 ship-level cycle types; phase 3 roles / terminals; phase 4 ducts and annotations, with phase 4 re-running the
vent invariants. Declare `duct` as the fifth move (the L4 vent exception) in §0 and in v0.1 §9's wording. One phase per rule, or amend L6
explicitly to allow a set.

### 15. MAJOR -- the pocket's air shape is wrong for the edge kind it allows

> `Pocket | leaf attachment | one room, one door, wide | no through-flow: gas that enters POOLS; a sealed pocket carries supply + return or no vents at all`

**Evidence.** The pocket's file permits only `edges = ["door"]`. A plain `door` is `MAT_DOOR`, permeability 0, gas-tight (vocabulary §0.4;
`EDGE_KINDS["door"]` bounds a compartment). No gas enters through it. The only gas path is its vents, and supply + return on a duct IS
through-flow (vent_system.py's plenum). So "gas that enters pools" is true of no pocket the format can build. Same slip in the retreat loop:
"Air shape: follows the walk graph" is false for plain doors, true only for arches.

**Fix.** Either allow `open` / `entity_door` on the pocket's leg and say pooling happens only then, or rewrite the air shape: "sealed: gas
arrives and leaves only through its vents, if any". Fix the retreat loop's air column to "follows the walk graph through arches only; doors
are gas-tight".

### 16. MINOR -- reproducibility (L7) now depends on config values the recipe hash does not cover

> `A recipe never contains a physics number: where it needs one it NAMES a config row`; v0.1 §9 `Reproducibility stays (recipe hash, seed)`

**Evidence.** gen_ship bakes numbers into the written level: `heat_source` and `hazard` tags become entity fields (`heat_per_s`,
`trigger_temp`, ...; gen_ship.py:1277-1326). Retuning `[heat_sources.stove]` changes the generated level for the same (recipe hash, seed).

**Fix.** One sentence: the hash covers recipes + the RESOLVED preset values (hash the resolved stage-1 graph), or the level carries preset
names resolved at load (a bigger change).

### 17. MINOR (ambiguity) -- the airlock bypass's leg structure

> `rooms_per_leg = [1, 1]`; `[[leg]] walk = true edges = ["airlock"]`; `edge-to-cycle across a cell boundary`

**Evidence.** Edge->cycle turns A-B into A-C-B + A-D-B. With one room per leg the walk leg has TWO edges (A-C, C-B): which is the airlock,
and what kind is the other (the leg lists only `airlock`)? Is C the airlock vestibule (which gen_ship carves itself and is not a graph room,
gen_ship.py:782-853) or a real room? On the gas leg, is D in A's cell or B's, and which of A-D / D-B is the `vent_only` edge? Two implementers
will build different ships.

**Fix.** Either `rooms_per_leg = [0, 0]` (the legs are single edges; then A and B must share two walls, or the gas leg is a duct, not an
edge), or list the per-edge kinds positionally (`edges = ["door", "airlock"]`) and name each new room's cell.

### 18. MINOR -- the loader's "gas-passing kind" list is not gas-passing

> `a walk = false leg must use a gas-passing kind (vent_only, breakable, window)`

**Evidence.** `breakable` (`MAT_WOOD`) and `window` (`MAT_GLASS`) block gas until destroyed (vocabulary §0.4). Only `vent_only` passes gas
intact, and only through a shared duct.

**Fix.** Rename the category ("can pass gas: through a duct, or once broken") and say whether a leg that passes gas only after damage
satisfies the airlock bypass's tactic sentence.

### 19. MINOR -- the galley role pairs a from-tick-0 heat source with kindling without a stated cap

> `heat_source = "stove"`, `heat_source_enabled = true`, `fuel = "kindling"`

**Evidence.** CLAUDE.md "Starting a fire": heat starts fires; `apply_temperature_ignition` lights a tile at its ignition temperature
(config.toml `[materials.kindling]`). A stove on from tick 0 beside kindling ignites every galley at a seed-independent time unless
`[heat_sources.stove].max_temp` is below kindling's ignition temperature. Possibly wanted ("every kitchen a fuse"), possibly a bug; the
document does not say. (`enabled` is also a refused tag key today, finding 10.)

**Fix.** State the intent and the constraint Erik tunes against: "stove `max_temp` < kindling `ignition_temp`" or the opposite, deliberately.

### 20. MINOR -- small ambiguities two implementers would resolve differently

- `size = { w, h }` vs `width`: is `w` the x-extent or the corridor's cross-section? A stub laid horizontally swaps them. Say "(short, long)
  for corridors, (x, y) for rooms", or let the layout rotate.
- `room.size` "each min >= 6 for a room" leaves corridor sizes unchecked against their `width` class.
- `department.cell = "wing"` on a `placement = "axis"` department: which wing? Say whether the two wings + axis band are one cell in v1.
- `[secret] p = 0.1` per department vs "the ship's secret budget is the archetype's" (§3.2) and "one mirrored pair per ship" (§5): who
  wins when three departments each roll 0.1?
- `[secret] kind = "vent_only"` requires the hidden room on the HOST's duct with both roles (vocabulary B.8); the format has no field for it.
- Spawn exits "a parameter the archetype passes to the existing writer": `[[spawn_room]]` rows carry `team, x, y, w, h, max_per_team,
  placement` (CLAUDE.md "Spawn rooms"); exits are a graph property, nothing in the writer receives them. The archetype passes them to the
  grammar, not the writer.
- "Back toward your own end" (retreat loop): ends exist only relative to a spawn; say the grammar reads the `toward` gradient.

## What the document gets right (keep through revision)

- The two-graph split (walk vs air) is the right frame and Erik's ruling; it is what makes the invariants placeable (finding 2 only asks it
  to name a third, the contact graph, for the triangle rule).
- "A recipe never contains a physics number; it names config rows": the correct boundary for Erik's physics authority (needs finding 16's
  hash sentence, not reversal).
- Refusal by name, at load, before planning: matches the `ROOM_TAGS` / `EDGE_KINDS` refusal style already in gen_ship.
- Corridors are rooms, never a second node type (litsearch Q3).
- The aquarium as a room role, not a cycle type; format before mechanism, explicitly marked format-only.
- The scale table (edge kind -> role -> cycle -> department -> archetype) gives the archetype session a defined input.
- The width ruling and its single exception, with Erik's words and scope; keep the exception narrow.
- Rulings scoped per case; Claude's calls listed separately as vetoable.
