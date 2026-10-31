# Levelgen department vocabulary — the ideas list and the first eight departments

*Dated capture, 2026-10-29. The content of #27 (vocabulary page). Design-only:
nothing here is built. Erik opened the talk in a parallel design chat; this
document is its output. **Revision 2 (2026-10-29 night)** folds in the
adversarial critique (`levelgen_zone_vocabulary_2026-10-29_critique.md`, Opus,
17 findings, every one checked against the code): the mirror policy is now
stated, the edge types are a table that maps to tiles, gates and L9, the duct
rule no longer contradicts itself, the doorless room's air runs the right
way, every "exists today" claim that was false is marked as new work, and the
cold rooms are parked on a physics session. The file keeps its original name
so links hold; the WORD has changed, see §0.1.*

Part A is the UNLIMITED ideas list (append freely, never prune). Part B is the
"implement next" set Erik blessed wholesale ("perfect start, and not that
simple frankly"). Part C is the secrets rule. Part D is the materials
question. Part E is the honest cost table the assets session and the patch
plan shop from. Part F is the agenda for the design session with Erik.

**Feeds:** #27 (this page), #133 (prefab pieces: a department's rooms become
pieces), #134 (vent rules: every recipe names its supply and return), #136
(ship-design adjacency), #137 (pressure cells). **Reads:** the research digest
`docs/research/levelgen_litsearch_2026-10-29.md` (#132), strand (d) above
all, and `docs/breach_levelgen_design_v0.1.md` (L1–L9). **Relates:** #120
(`tools/gen_ship.py`, whose stage 1 these departments replace and whose
`embed` + `validate` they reuse, extended as named in Part E), #48/#49
(vents; vent states), #135 (terminals), #60 (props).

---

## 0. The framing: a department is a recipe, not a room

### 0.1 The word

Erik's word in the talk was "zone", and "zone" is the natural word. But
`zone` is already a LOCKED term in the code: the two intangible painted
entities `breach_site` / `extraction_zone` in `entities/zones.py` +
`zones.npy` (level editor v3 §5, 2026-07-18; ~47 references). Design v0.1 §4
says one word, one meaning. This document therefore uses **department** (a
ship's own word: the engineering department, the medical department) for the
room cluster, and leaves `zone` to the painted entities. **Erik's call**
(Part F, item 1): keep "department", or rename the two entities and take
"zone" back. In conversation either word is fine; in code and docs, one.

### 0.2 The recipe

A **department** is a cluster of rooms with one function (a medbay is an
operating theatre, a ward, a pharmacy, a morgue and a scrub room). In
generator terms a department is a stage-1 object: it expands into rooms under
rules. A **room** is what the embedder stamps (#133's piece). A department is
described by six fields:

| Field | What it says | Who consumes it |
|---|---|---|
| **Rooms** | required rooms; optional rooms with a probability; size ranges in tiles | grammar → embedder |
| **Adjacency** | which rooms touch and by which EDGE TYPE (§0.4); what the department wants next to it on the ship, and where it sits relative to the mirror axis and the spawn ends (§0.3) | grammar (#136) |
| **Cell** | does the department sit inside ONE pressure cell (default) or is it its own cell (then every edge on its boundary is an airlock pair or a bulkhead, L5) | #137 |
| **Ducts** | which rooms carry a supply vent, which a return vent, on which duct; a duct never crosses a cell boundary | #134 |
| **Physics signature** | ambient air (gas mix, temperature), fuel load and where it sits, water, light level, hazards that are PAYLOADS waiting (O2 bottles, halon dump, dust) | embedder annotations → entities, props, `[[light]]` rows |
| **Secrets** | may this department carry a hidden annex, and of which kind (Part C) | grammar |

The physics signature is the whole point. The sim already does gas, O2, fire,
water, vents, light and materials; a department that only looks different is
wallpaper, a department whose AIR is different plays different. Every row in
Part A carries its physics hook for that reason, and Part E says honestly
which hooks exist today.

### 0.3 The mirror (critique finding 1)

`gen_ship` mirrors the ship left↔right, seats the two spawn teams at the two
ENDS, and `validate` refuses a non-mirrored grid. This is a match-fairness
property, and the RL milestone depends on it (a memorised ship is the enemy;
an unfair ship is worse). **Draft policy (Erik's call, Part F item 2):**

- Departments come in **mirror pairs** (one per wing, reflected) or as
  **axis departments** (one, straddling the mirror line, reflected onto
  itself). Both are compatible with the existing symmetry gate as is.
- Engineering is the first axis department. The digest's end-to-end gradient
  (D10: reactor at one end, stores between it and quarters) becomes an
  **axis → wing** gradient: engineering on the axis, cargo as the stores
  buffer, quarters toward the spawn ends.
- Secrets come in mirror pairs too (both teams get the same hidden annex), or
  sit on the axis. One secret per ship means one per wing, mirrored.
- An asymmetric ship (option c in the critique: mirror the skeleton, let the
  dressing differ) is an ARCHETYPE for later, and needs the symmetry gate to
  compare structure rather than materials. Not v1.

### 0.4 Edge types (critique findings 2, 10)

Every adjacency in a recipe uses one of these. The table maps each to what
stage 2 stamps, whether `compartments()` sees a boundary, the L9 strength,
and whether it exists. L9's strengths are `must-share-wall` (the two rooms
are physically adjacent), `may-become-corridor`, `routed-freely`.

| Edge type | Stamps | Gas | Compartment boundary? | L9 | Status |
|---|---|---|---|---|---|
| **door** (the plain, gas-tight door) | `MAT_DOOR` tile in the shared wall | blocks (permeability 0); units walk through | yes | must-share-wall | EXISTS (`gen_ship` `Edge.kind="door"`) |
| **entity door** (closes, opens, wireable, lockable later) | a `door` entity; open stamps `MAT_AIR`, closed `MAT_DOOR_CLOSED` | passes when open | yes when closed | must-share-wall | runtime EXISTS (A6); `gen_ship` never emits one → stage-2 extension |
| **open doorway / arch** | a `MAT_AIR` gap in the shared wall | passes always | NO — merges the two rooms into one compartment | must-share-wall | NEW gate change: `compartments()` must count graph rooms joined by open edges as one compartment and check room count against the graph's compartment count, not its room count |
| **window** | `MAT_GLASS` | blocks until burst (threshold 1.0) | yes | must-share-wall | EXISTS (`Edge.kind="window"`) |
| **bulkhead** (a cell boundary with no opening) | steel/hull wall, nothing cut | blocks | yes | must-share-wall | EXISTS (it is a wall) |
| **airlock pair** (a cell boundary you can cross) | two entity doors + a small room + an `airlock_controller` entity | sealed in sequence | yes | must-share-wall | runtimes EXIST (A6, `airlock_controller`); stage-2 extension to emit the L5 recipe |
| **vent-only** | no opening; a supply and a return on a duct the two rooms share | gas only, through the duct and its filter | yes | must-share-wall | runtime EXISTS; stage-2 extension (multi-duct builder, finding 11) + reachability gate change (finding 4) |
| **breakable wall** | a `MAT_WOOD` segment in a steel/hull wall (wood only: solid, mobility 0; furniture is walkable and leaks) | blocks until destroyed | yes — but `compartments()` today counts wood as room contents, so the structure mask must come from the graph's edge kinds | must-share-wall | NEW: gate change (structure mask) + reachability accepts "reachable through a breakable edge" |
| **locked door** | an entity door + a `terminal` | as entity door | yes | must-share-wall | BLOCKED on #135 (`terminal` has no behaviour) |
| **department-to-department wish** ("quarters next to medbay") | nothing; a placement preference | — | — | routed-freely | grammar (#136) |

Rule of thumb from the critique: **a room whose every edge is gas-tight is a
sealed box.** The vent system has no pressure feedback (fixed circulation,
no makeup controller), so a sealed room with a supply only is pumped up until
its weakest membrane bursts, and with a return only it is drained. Therefore
**every sealed room carries BOTH a supply and a return, or neither.** "Returns
through the doorway" is legal only across an open-doorway edge, and that edge
needs the compartment gate change.

### 0.5 Ducts and cells (critique findings 8, 9)

- **A duct never crosses a cell boundary.** Several ducts per cell are fine
  (the spine's duct, a galley's exhaust, a department's own loop). This is the
  SOLAS rule stated correctly; "one cell = one duct system" in revision 1 was
  wrong.
- **A department that is its own cell joins the ship only by airlock pairs or
  bulkheads** (L5, not reopened). That is expensive in a small ship, so in v1
  only ENGINEERING is its own cell; every other department rides its wing's
  cell. The medbay's isolation room shows the L5 recipe at room scale.
- **Dampers, shut-offs, "closed on alarm" (ZEBRA):** vent `state` is reserved
  for #49 (v1 forces open); closing an entity door on a smoke sensor is
  wire-able today (`sensors` → `gate_or` → door `close`) but only for entity
  doors, which stage 2 does not yet emit. Every damper in a recipe below is
  marked "#49".

---

## Part A — the unlimited list

One line per idea: **name** — rooms it implies · *physics hook* (marked
**[session]** where the hook needs physics that does not exist today, see
Part E). Append at will; a line here costs nothing and commits nobody.

### A.1 Ship essentials

- **Crew quarters** — bunks, heads, showers, mess, galley, laundry, lounge · *heads return → bunk supply on one duct whose filter passes teargas (teargas in the heads comes out over the sleepers); galley grease fire; laundry steam.*
- **Engineering** — engine room, reactor room, coolant pool, fuel bunker, pump room, workshop, switchboard · *hot ambient [session: initial-T level field]; heavy fuel; a reactor that leaks heat when shot [session: heat-source entity]; coolant pool is water.*
- **Life support** — O2 generator, CO2 scrubber, air plant, water cistern · *the trunk of every duct; kill it and the ship's vents stop; cistern floods on burst.*
- **Bridge / CIC / comms / navigation** — glass, consoles · *terminals that unlock doors [#135]; glass is light 0.1 / heat 0.3.*
- **Cargo** — container hold, cold storage, hazmat bay, livestock pens · *crates = fuel clusters; see A.6.*
- **Medbay** — operating theatre, ward, isolation / quarantine with own airlock, pharmacy, scrub room, morgue · *O2 bottle store as a fire accelerant INSIDE the medbay [new payload effect]; quarantine beds as pressure cells.*
- **Hangar / airlock chain / EVA suit room** — *hull-side; a depressurisation valve is a legitimate fire response (MPLM, D7).*
- **Drone bay** — racks, charging, a repair gantry · *drones repair the hull (Erik; drones are a special unit, out of this arc).*
- **Corridor spine** — *its own duct (the pressurised-stair rule); the pressure sandwich as a map feature.*

### A.2 Military

- **Armory** — racks, ammo magazine, explosives bunker · *cook-off chain [heat trigger on `timed_charge`].*
- **Firing range** — *long sightline, nowhere to hide.*
- **Barracks**, **war room**, **brig** and **panopticon cellblock**, **interrogation room**.
- **Missile tubes + missile magazine** · *the biggest payload row on the ship.*
- **Repair bay with half-dismantled mechs** · *props, cover.*

### A.3 Luxury and civil

- **Casino with a vault** · *the vault is a secret-chamber variant behind a locked door [#135].*
- **Disco with a fog machine** · *a steam source entity (the steam row exists).*
- **Bordello wing** — numbered rooms off a corridor.
- **Spa and swimming pool** · *the water sim as a level feature.*
- **Ballroom**, **theatre with backstage and trapdoors** (trapdoor = secret annex; needs a vertical seam the engine lacks — parked), **cinema**, **promenade with shopfronts**.
- **Restaurant kitchen** · *grease fire; galley exhaust with its own shut-off [#49].*
- **Cigar lounge, sealed** · *the air is already smoke: a room that starts with trace.*
- **Chapel with organ**, **hotel floor**, **kindergarten and school**, **library** (paper = fuel), **museum of looted artefacts**, **zoo**.

### A.4 Science vessels

- **Aquarium hall** — a tank big enough for the octopus (#104–#112) · *glass walls, water.*
- **Terrarium / botanical garden under glass** · *foliage burns; procedural trees as decoration and fuel (NOT as walls or cover: foliage is walkable and see-through; a blocking hedge is a new material row).*
- **Hydroponics** · *high CO2 / low O2 by design — Erik 2026-10-29: NO CO2 gas row; represent as a low-O2 air seed instead (bulk O2 fraction is state already).*
- **Animal holding pens**, **clean lab with airlock**, **cryo bay with sleepers**, **specimen vault**, **observatory dome**, **insect protein farm**, **mushroom farm in the dark** (light 0), **bioreactor / sewage plant** (methane = the existing flammable gas row, `fuel_gas`, which has never been placed in a level).

### A.5 Dark

- **Morgue** [session: cold], **crematorium** (a furnace = heat-source entity).
- **Human experimentation ward**, **torture chamber behind a false wall** (breakable-wall edge), **devil-worship chapel with a desecrated altar**, **ossuary made of the crew**, **taxidermy room**.
- **The Eyes Wide Shut mansion** — masks, ballroom, a corridor of locked bedrooms [#135].
- **The Black Lodge** — red curtains, zigzag floor; a room that is a MIRROR COPY of another room on the ship (the embedder already mirrors; an axis room is its own mirror).
- **The smokers' room** — the air is already poison (a poison-row seed).
- **The sealed deck** — frozen since the last breach, bodies in place; vacuum and dark today, cold [session].

### A.6 Rooms you would not expect, and why they are fun

- **Server farm with halon suppression** · *fire triggers a gas dump that replaces O2: the fire dies and so does whoever is inside. Heat trigger + a bulk-gas payload effect injecting inert N2 (both Part E mechanisms) — no new gas row.*
- **Grain / flour store** · *a dust row that is explosive at concentration — Erik wants DUST (grey, settles faster than smoke) as a gas row anyway. New row + a "flammable trace" rule the fuel_gas row already has.*
- **Paint shop / solvent store** · *fuel_gas vapour drifts along the ducts to a far room before it finds a flame. Exists, never tried.*
- **Cold storage / meat locker** (Erik: "I especially love the cold storage") · *a cold THERMAL SOLID that just keeps its temperature (Erik's ruling). **[session]**: solids have a low rail at ambient and the emission table starts at ambient, so sub-ambient needs a physics ruling (Part F item 3). No fog and no sinking smoke: the sim is 2D without buoyancy, and condensation is the water arc's.*
- **Livestock pens** (Erik likes) · *animals as props now, non-player units later; straw as fuel; methane = fuel_gas.*
- **Distillery / still** · *fuel_gas vapour + a heat source; the brewing crew's secret.*
- **Oxygen candle store** · *a fire that brings its own O2, so sealing the room does not kill it — the O2 payload effect, triggered by heat.*
- **A ship inside the ship** · *an older hull swallowed by the newer one; a second hull material row (needs the migration, Part D), its own dead air.*
- **A room with no door** · *the cheapest secret; see B.8 for the version that actually works.*
- **Hall of mirrors** · *vision-oracle trickery, no physics cost.*
- **Flooded deck** · *a corridor you wade through while something swims below.*
- **Hedge labyrinth in a garden dome** · *needs a blocking, flammable hedge material row (foliage is walkable); after the migration.*
- **Bowling alley / ice rink** · *absurd on a warship; a long open sightline.*
- **Quarantine ward with airlocked beds** · *each bed a pressure cell (the L5 recipe at bed scale).*
- **Memorial wall of the dead crew** · *names of units from a previous match, free once the replay path (#126–#128) exists.*

---

## Part B — the first eight departments (blessed 2026-10-29)

The simple start: departments that reuse what exists and need few new
mechanisms. Build order = list order; crew quarters first because it
exercises supply/return and the open-doorway gate change end to end. Each
recipe uses the fields of §0.2 and the edge types of §0.4. "Extension" means a
named stage-2 extension from Part E; "[session]" means Erik's physics ruling
first.

### B.1 Crew quarters (mirror pair, one per wing)

- **Rooms:** stub corridor (required) · bunk rooms ×2–4 (required) · heads
  (required) · mess (required) · galley (optional 0.7) · laundry (optional 0.3).
- **Adjacency:** bunk rooms and mess join the stub by **open doorways** (one
  compartment: the quarters breathe as a unit); the heads joins the stub by a
  **door**; the galley joins the mess by a door. Wish: medbay adjacent;
  engineering far (toward the axis), with cargo between.
- **Cell:** the wing's cell.
- **Ducts:** ONE quarters duct, filter `derelict` or a teargas-passing row:
  heads = RETURN + a small SUPPLY (a sealed room carries both); bunk rooms =
  SUPPLY; the stub carries the compartment's RETURN. So the heads' air is
  recirculated into the bunks: teargas thrown into the heads comes out over
  the sleepers. Galley, when present: its own small duct with a shut-off [#49].
- **Physics:** ambient air; fuel = bunks and lockers (furniture clusters as
  gen_ship stamps today); galley = a heat-source entity [Part E] + kindling;
  lights one lamp per room, dim in bunks (per-room light annotation, Part E).
- **Secrets:** a breakable-wall annex behind a bunk room, 0.1.
- **Needs:** open-doorway gate change; multi-duct builder; per-room light tag.

### B.2 Engineering (axis department, its own cell)

- **Rooms:** engine room (required, large, on the axis) · fuel bunker
  (required) · workshop (optional 0.6) · pump room (optional 0.4) ·
  switchboard (optional 0.5).
- **Adjacency:** the engine room is the hub; the fuel bunker joins it by a
  **door** (gas-tight: the bunker is a sealed box with its own supply+return
  or none); workshop and switchboard by doors. The department joins each
  wing's spine by an **airlock pair** (L5) — one per wing, mirrored.
- **Cell:** its own cell (the only one in v1 besides the wings).
- **Ducts:** its own duct, separate from accommodation (D2): engine room
  supply + return; bunker none (sealed, its air is its fuel's).
- **Physics:** hot ambient [Part E: initial-temperature level field, seeded
  through `reseed_gas_energy`]; the engine = a heat-source entity that
  deposits heat every tick and, when its hp goes, delivers IGNITE_HEAT (the
  Starting-a-fire rule) [Part E]; the bunker = dense kindling clusters, the
  biggest fire load on the ship; light harsh and warm.
- **Secrets:** the still, 0.2 (a breakable-wall annex off the workshop).
- **Needs:** airlock-pair emission; heat-source entity; initial-T field.

### B.3 Medbay (mirror pair)

- **Rooms:** ward (required) · operating theatre (required) · pharmacy with
  the O2 store (required) · scrub room (optional 0.5) · morgue (required) ·
  isolation room with its own airlock pair (optional 0.4).
- **Adjacency:** the ward is the hub off the spine by a **door**; theatre and
  scrub room join the ward by **open doorways** (one clean compartment); the
  pharmacy by a **door** (a **locked door** once #135 lands); the morgue by a
  **door**; isolation by an **airlock pair** (the L5 recipe at room scale).
  Wish: quarters adjacent, engineering far.
- **Cell:** the wing's cell; the isolation room is a cell of its own inside it.
- **Ducts:** the ward compartment: theatre = SUPPLY, ward = RETURN; the
  pharmacy and the morgue are sealed boxes: supply + return each, or none;
  isolation: its own supply + return [dampers #49].
- **Physics:** clean air; fuel light (beds, cabinets); **the O2 store is a
  heat-triggered bulk-gas payload** [Part E: both mechanisms] that injects
  pure O2 — the sustain law is linear in the mole fraction above ambient, so
  a fire next door roars (the one property test on this,
  `test_eos_p4_combustion.py::test_e2e_3_o2_rich_pocket_intensifies_burn`,
  says it does); the morgue is a plain cool room in v1 and a cold thermal
  solid after the physics session [session]; light bright white in the
  theatre, dim blue in the morgue.
- **Secrets:** the experimentation ward behind the morgue (breakable wall), 0.15.
- **Needs:** open doorway; multi-duct; airlock pair; heat trigger; O2 payload.

### B.4 Cargo hold (mirror pair; the stores buffer between engineering and quarters)

- **Rooms:** main hold (required, the largest room in the wing) · side bays
  ×1–3 (optional 0.6 each) · cold storage (optional 0.4, [session]) · hazmat
  bay (optional 0.3) · livestock pens (optional 0.3).
- **Adjacency:** the hold on the hull; the bays join the hold by **open
  doorways** (arches: one compartment); the hazmat bay by a **door** (sealed
  box); the hold joins the spine by a **door**. A cargo door to space is an
  **entity door** in the hull (A6; stage 2 must emit it) — the department's
  depressurisation valve (D7).
- **Cell:** the wing's cell.
- **Ducts:** hold compartment: supply + return on the cell's cargo duct
  (cargo spaces separately ventilated, D2); hazmat bay: none (sealed).
- **Physics:** fuel = container rows (furniture clusters in lines, the cover
  geometry); hazmat = a heat-triggered payload releasing `fuel_gas` (exists:
  `emit_gas` on a trace row) — a solvent store that finds a flame later; pens
  = straw fuel + animals as props; cold storage [session]; light low, pooled.
- **Secrets:** the ship inside the ship, 0.1 — a sealed older-hull section
  behind the hold's back wall, breakable-wall edge; its second hull row needs
  the migration (Part D).
- **Needs:** open doorway; entity door in hull; heat trigger.

### B.5 Armory (mirror pair)

- **Rooms:** armory (required) · magazine (required) · firing range
  (optional 0.4) · explosives bunker (optional 0.3).
- **Adjacency:** the armory joins the spine by a **door** (a **locked door**
  after #135); the magazine joins the armory by a **door**; the bunker joins
  the magazine by a **door**; the range is a long room off the armory by an
  **open doorway**.
- **Cell:** the wing's cell (a magazine of its own cell is an archetype for
  later; in v1 its doors are plain).
- **Ducts:** armory + range compartment: supply + return; magazine and bunker
  sealed, none.
- **Physics:** **the magazine is a heat-triggered `timed_charge`** [Part E]
  firing a `[payloads.*]` cook-off row when its tiles pass a threshold; the
  bunker chains a bigger row with a fuse delay; fuel otherwise light; light
  bright.
- **Secrets:** none; 0.
- **Needs:** heat trigger on `timed_charge` (the ONE new mechanism B.5 adds).

### B.6 Garden dome (mirror pair, or axis)

- **Rooms:** dome (required, large; a rectangle in v1) · potting room
  (optional 0.5) · pump / water room (optional 0.4).
- **Adjacency:** the dome joins the spine by a **door**; the potting room by
  an **open doorway**. A **window** wall on the corridor side so a fire in the
  dome is seen from the spine. The dome need not touch the hull (finding 17: a
  one-tile glass hull membrane sits at its burst limit and blows on the first
  fire; whether that is a FEATURE is Part F item 5).
- **Cell:** the wing's cell.
- **Ducts:** its own duct within the cell: supply + return.
- **Physics:** fuel = procedural trees and foliage (`MAT_FOLIAGE`, `propgen`,
  `static_props`) as DECORATION AND FUEL, not walls (foliage is walkable and
  see-through; the hedge maze waits for a blocking hedge row, Part D); water =
  a pond tile group (optional 0.5); light: lamps. There is NO daylight: the
  ship's boundary sky is `[light.sky.space]`, dark. A sun or a lit star-field
  for a dome is Part F item 5.
- **Secrets:** none in v1.
- **Needs:** nothing new beyond the common extensions; the cheapest of the
  eight.

### B.7 Pool / aquarium hall (mirror pair)

- **Rooms:** hall with the tank or pool (required) · pump room (required,
  small) · changing rooms (pool variant, optional 0.6) · viewing gallery
  (aquarium variant, optional 0.6, behind a **window** wall).
- **Adjacency:** the hall joins the spine by a **door**; the pump room behind
  the tank by a **door**; changing rooms by **open doorways**.
- **Cell:** the wing's cell.
- **Ducts:** hall compartment: supply + return; pump room sealed, none.
- **Physics:** water (the water sim as a level feature; tank walls are glass
  `MAT_GLASS`, bursting them floods the hall); the octopus's home later
  (#104–#112); fuel light; light: lamps now, caustics with #97.
- **Secrets:** the flooded deck below as the pump room's annex, 0.1 (a
  breakable wall into a water-filled room).
- **Needs:** nothing new beyond the common extensions.

### B.8 The doorless room (the first secrets rule) — corrected

Not a department: a GRAMMAR RULE any department may apply. A room with no door
edge, joined to its host by exactly one of:

- a **breakable-wall** edge (wood, in a steel or hull wall), or
- a **vent-only** edge: the hidden room carries a SUPPLY and a RETURN on the
  HOST's duct, whose filter passes smoke (`derelict`, never `hepa_basic`,
  which scrubs smoke to zero). The host's return pulls the host's smoke into
  the duct and the hidden room's supply deposits it there; the hidden room's
  return keeps it from over-pressurising (finding 3: revision 1 had the vent
  the wrong way round and behind a HEPA filter).

What the player sees: the breakable wall is a wood segment in a steel wall,
readable to a sharp eye and to the tile inspector; the vent-only room shows
as a wall thicker than it should be and, when the host burns, as a room that
is full of smoke when it is finally cut open. The hook "watch the smoke go
into the wall" from revision 1 is weak (a return grille is a return grille);
it is dropped.

Gates: the hidden room is airtight by construction (the airtight gate passes
as is). Two gate changes are needed and named in Part E: `compartments()`
must take its structure mask from the graph's edge kinds (today it counts
wood as room contents and would merge host and annex), and the reachability
gate must accept "reachable through a breakable or vent-only edge" for a
room tagged `secret`. What is inside is the host department's secret row.

---

## Part C — secrets as a rule, not a room type

Any department may carry ONE hidden annex, at its own probability. The annex
is reached by exactly one of: a breakable wall (B.8), a vent-only edge (B.8),
a locked door (#135's budget), a trapdoor under a stage or floor plate (needs
a vertical seam the engine does not have; parked). The horror comes from
CONTRAST, so the annex's content is drawn from the DARK list (A.5) attached to
a benign host: the torture chamber behind the kindergarten, the altar under
the chapel, the lodge behind the hotel's last door, the experimentation ward
behind the morgue. Secrets are budgeted per ship (one or two, and under the
mirror policy that means a mirrored pair, §0.3), never per department, so
they stay secret. Whether that budget is #135's interactivity budget or its
own is Part F item 4.

---

## Part D — materials: the table is already unbounded (Erik: "rows suffice")

Erik's question (2026-10-29): "isn't it quite easy to create such a material
list on the fly? … could we have materials that is the whole parameter
space?" and his ruling the same night: **rows suffice; design for 127
materials; bump int8 to int16 only if we ever fill it.**

**The system is already built that way.** `MaterialTable`
(`src/simulation/materials.py`) is a per-id column store: every property
(hp, flammable, mobility, thermal mass, ignition temperature, fuel, heat and
light attenuation per channel, permeability, burst threshold, cover exposure,
wave coefficients) is a numpy column indexed by the tile's material id, and
every derived cache is one fancy-index lookup `column[material]`. Adding a
material is one `[materials.<name>]` row in config.toml. A row IS a point in
the parameter space. What we do NOT want is a continuous per-tile parameter
(property planes instead of an id): it multiplies digested state by the
number of properties for no gameplay gain. Note a material row carries no
TEMPERATURE: temperature is state, so a "cold room" is an initial-temperature
level field plus whatever the rails allow (Part F item 3), not a row.

**Why ten is the limit today, precisely** (critique finding 14 corrected my
revision-1 account): `gmap.material` is `int8` (127 ids, fine). The v2
tilemap CSV writes a tile's material id literally plus ONE reserved
non-material code `SPACE_CODE = 9` (`level_loader.py`). `MAT_FOLIAGE = 9`
already equals it — harmless only because foliage is a prop stamp, never
painted. An id 10 would not collide with 9 at all; the block is the CLAUDE.md
rule "an 11th material needs Erik's SPACE_CODE-migration ruling FIRST",
written so the clash is cleaned up before the table grows past it. **Erik has
now given that ruling** ("we can't be limited to 10 materials … no big deal"),
so the migration is a patch — BUILT the same night on branch `27-space-code-migration` (five commits off main `de1f461`; suite 3908 passed, no golden moved, bake goldens unchanged; awaiting Erik's look and the lead session's merge):

- SPACE moves to a code outside the id range. `255` does not fit the int8
  plane; use `-1`, cast explicitly at the loader (finding 15).
- Scope is ~26 files, not one: `level_loader.py`, `level_lib.py`,
  `map_editor.py` (13 sites), `level_edit_common.py`, `bake_level_art.py`
  (the golden-gated art path — its goldens must be CHECKED, not assumed
  neutral), `make_tileset.py`, `align_level_art.py`, the three generators,
  `eos_p5_bake.py`, and ~15 tests that pin 9.
- Every shipped CSV migrates in place through a `level_lib` writer (the
  `migrate_tilemap_v2.py` pattern), level format version bumps to v3 so an old
  CSV is refused rather than misread.
- The field digest does not move (material planes are unchanged), so the
  GOLDEN_AGGREGATE stays; the bake goldens are re-run and compared.

**The migration IS on Part B's critical path** (finding 14): the old-hull
row (B.4's secret), a blocking hedge (A.6), a cold solid (if the physics
session wants a distinct row), a stronger dome glazing (B.6) are all ids ≥ 10.
B.1, B.5, B.6, B.7 and B.8 need no new row and can start before it; B.3 and
B.4 keep their full signature only after it. Two real costs scale with row
COUNT, both small: the N×N face table (`materials.py` ~l.1322) and the
thin-rows rulings (`THIN_LIMIT_M` / `LUMPED_CRITERION_EXEMPT`, PROVISIONAL
until Erik's material design session) that any new flammable thin row must
pass.

---

## Part E — honest cost of the first eight

Revision 1 called several new mechanisms "cheap data rows" (finding 13). This
table is the corrected one. Three columns: what EXISTS and is reused as is;
what is a NEW PATCH on an existing system (named, mechanical, mine to build);
what needs a SESSION with Erik (physics or art).

### E.1 Common stage-2 extensions every department needs (patches on `gen_ship.embed` / `validate`)

| Extension | What | Gate change? |
|---|---|---|
| **Edge kinds** | `Edge.kind` grows from `{door, window}` to §0.4's set; each kind stamps as the table says | — |
| **Open-doorway compartments** | `compartments()` labels graph rooms joined by open edges as one compartment and the room-count gate compares against the graph's compartment count | yes |
| **Structure mask from the graph** | `compartments()` takes walls from edge kinds, so a wood breakable segment is structure, not contents | yes |
| **Secret reachability** | the reachability gate accepts a `secret`-tagged room reached only through a breakable or vent-only edge | yes |
| **Multi-duct vent builder** | `_build_vents` emits one duct per department (ids `duct_<dept>_<n>`), a unique vent id per room, a per-duct filter name, a room may carry both roles | — |
| **Entity doors and airlock pairs** | stage 2 emits `door` entities (and an `airlock_controller` + two doors + a vestibule for an L5 edge) where the edge kind says so; today it stamps only `MAT_DOOR` | — |
| **Per-room annotations** (finding 16) | the tag schema a department writes per room: `dept`, `role`, `cell`, `light` (rgb, level), `fuel` density, material overrides (glass wall), water group, props list; how departments coexist with the spawn rooms and the zombie section `validate` A*-checks | — |

Status (2026-10-31, #133 patch 1, branch `133-stage2-edges`):

- **Edge kinds**: DONE 01537e9, `gen_ship.EDGE_KINDS` (this §0.4 table as code: stamps, width, bounds, secret path, L9, pending); door / window / open / bulkhead / vent_only / breakable stamped, entity_door / airlock / locked accepted and refused by `embed` until the next patch.
- **Open-doorway compartments**: DONE 01537e9, `ShipGraph.compartment_of` + `validate(graph=)` (`_graph_compartment_reasons`: count and room-to-compartment match against the graph's compartments).
- **Structure mask from the graph**: DONE 01537e9, `gen_ship.compartments(tm, graph)` (a bounding edge's opening tiles are structure, so a breakable's wood is a wall).
- **Secret reachability**: DONE 01537e9, `gen_ship._excused_secrets` (an all-`secret` compartment joined to a reached one by a breakable / vent_only edge); airtight gate untouched (`tests/test_gen_ship_edges.py`, 1838ce8).
- **Multi-duct vent builder**: not in this patch (next).
- **Entity doors and airlock pairs**: not in this patch (next).
- **Per-room annotations**: DONE 01537e9 in part, `gen_ship.ROOM_TAGS` (spawn_team, zombie, fuel, vent, dept, role, cell, light, secret; unknown keys refused); `light` replaces the fixed lamp, dept / role / cell / secret / light ride `Embedding.room_tags`. Material overrides, water group and props list not yet.

Status (2026-10-31, #133 patch 2, branch `133-stage2-vents-doors`):

- **Multi-duct vent builder**: DONE b81237f, `gen_ship._build_vents` (one `duct_<name>` per duct the `vent` tags `{"duct", "roles"}` name, filter from `ShipGraph.ducts`, default `hepa_basic`; ids `vent_<duct>_<room>_<role>`, v1's string shorthand keeps `duct_main` / `vent_<role>`); `ShipGraph._check_ducts_and_cells` refuses a duct across cells, a sealed room with one role (dict form; v1's shorthand pair is exempt, it is the finding-2 lone supply / lone return) and a `vent_only` edge off one duct (`tests/test_gen_ship_vents_doors.py`, 64100ad, incl. the B.8 smoke-arrival test).
- **Entity doors and airlock pairs**: DONE b81237f, `EDGE_KINDS` `entity_door` (an A6 `door` entity, emitted OPEN: the bots never open a door, #142) and `airlock` (`gen_ship._airlock`: a 5x5 vestibule carved on room `a`'s side, levels/airlock_demo's two doors, pump, plate, probe, deciders, controller and 15 wires); the gates count each vestibule as its own compartment and walk through the airlock's outer door (its controller opens it for any unit inside); an edge across cells must be an airlock or a bulkhead (L5). `locked` still waits on #135.

### E.2 New mechanisms (patches on existing canonical systems; none is a parallel system)

| Mechanism | Extends | Used by | Note |
|---|---|---|---|
| **Heat trigger on `timed_charge`** | `entities/charge.py` + `timed_charge_system.py`: `trigger = "heat"`, `trigger_T`, a synced fired-latch; the countdown becomes the fuse delay | B.3 O2 store, B.4 hazmat, B.5 magazine, halon room, O2 candles | NEW host-side tick logic in slot 9e → needs a ruling against RL-batch habit §A (Part F item 6). Not a reuse of `apply_temperature_ignition` (per-material threshold, per-tile armed plane, lights fires only). **DONE a2924be** (Erik ruled §A accepted 2026-10-31): `trigger` / `trigger_temp` (game degrees above ambient, Q16 at load) in `entities/charge.py`, the latch + fuse in `timed_charge_system.sweep_timed_charges`; `fired` digested on heat charges only; `[payloads.magazine_cookoff]` / `hazmat_release` placeholders; gen_ship `hazard` room tag (`tests/test_timed_charge_heat.py`) |
| **Bulk-gas payload effect** | `payloads.py`: a `[payloads.*]` column executed through the gas pump primitives with an explicit per-gas vector (`inject_gas_n_vec` takes `add_vec`; `inject_gas_n` is FIXED 21/79 and cannot enrich); energy booked through the seam as minted at the deposit's T | B.3 O2 store, halon (N2), O2 candles | `emit_gas`'s FieldEdit REFUSES conservative rows, so this is the only honest path to O2/N2; own closure test  **DONE 0ea3c22**: `payloads.inject_bulk_gas` (step 4b of `execute_payload`) through `inject_gas_n_vec` with a pure-slice vector, FLAT over the `bulk_radius` disc minus the vent aperture guard, vent rails; columns `bulk_gas` / `bulk_amount` (bulk N per tile, 1.0 = a tile of ambient air) / `bulk_radius` / `bulk_temp` (game degrees above ambient), refused at the `PayloadTable` door (trace gas, bulk `gas_species`, non-positive / sub-LSB amount, orphan columns); books `payload_bulk_gas` / `payload_bulk_gas_rail` (seam channels, exact across-tick closure); `[payloads.o2_store_rupture]` / `[payloads.halon_dump]` PLACEHOLDERS (`tests/test_bulk_gas_payload.py`) |
| **Heat-source entity** | a new entity class + a slot-9e runtime that deposits heat each tick through the temperature solver's deposit path; on hp → 0 delivers IGNITE_HEAT (Starting-a-fire rule) | B.1 galley, B.2 engine, crematorium, still | New class, not a row; same RL §A ruling. **DONE 41c7634 + f901af6** (Erik ruled §A accepted 2026-10-31): `entities/heat_source.py` (`heat_per_s` real heat/s -> one Q16 per-tick quantum at load, `radius` beyond its own tile, `enabled`, `hp` in the wall-hp unit, `death_payload`, `max_temp` cap in game degrees, 0 = none); `heat_source_system.sweep_heat_sources`, LAST in 9e after the timed charges, deposits through `payloads.deposit_heat` (the solver's counted deposit, closure exact) and skips while its tile's mirror reads >= `max_temp`; DEATH HOOK: entities have no hp path, so `execute_payload` appends every blast disc to `GameMap.blast_log` (drained at 9e, empty between ticks) and a source takes the wall law's damage from it, chained deaths in one sweep; on hp 0 its `death_payload` runs once -- `[payloads.engine_death]` PLACEHOLDER (mid blast + `heat_amount` 8520 over radius 3 = >= IGNITE_HEAT out to 2 tiles); synced rows `enabled` + runtime `hp` + `alive`; gen_ship `heat_source` room tag on the shared `_floor_site` (`tests/test_heat_source.py`). FOUND (not fixed): a `pressure > 0` blast beside a thermal solid breaks the closure identity by a constant on its own (base 8da03d6, no source) |
| **Initial-temperature level field** | level format: a per-tile T seed written by `level_lib`, read at the loader door, applied through `reseed_gas_energy` after the air seed | B.2 hot engineering, cold rooms later | `air_init.npy` is atmosphere only today. **DONE d42dfc1 + 0b881df**: optional `temp_init.npy` (int32 Q16.16 game degrees above ambient, shape == tilemap, by presence, format v3 unbumped), refused at the loader door (`_parse_temp_init_grid`: dtype, shape, sub-ambient -- Part F item 3 --, above `T_MAX_PHYS`); one writer `level_lib.write_temp_init_npy` (all-zero deletes: 0 is ambient); seeded in `GameMap._seed_temp_init` after the air seed and `refresh_gas_energy` -- gas through `seed_gas_temperature`, thermal solids on their own `temperature`, anything else refused; `--res` replicates it; gen_ship `temp` room tag (floor air only; no tag, no file) (`tests/test_temp_init.py`). FOUND (not fixed): a hot thermal solid seeded in ambient air drives its neighbours sub-ambient on tick 1 and walls into the low rail |
| **Cargo door in the hull** | an entity door whose far side is space, emitted by stage 2 | B.4 | the airtight gate must accept a closed entity door in the hull |
| **SPACE_CODE migration** | Part D | B.3/B.4 full signature, A.6 rows | ruled; a patch |

### E.3 Per department

| Department | Reuses as is | E.1/E.2 it needs | Session (Erik) |
|---|---|---|---|
| B.1 Crew quarters | furniture/kindling clusters, vents runtime, lamps, `MAT_DOOR` | open doorway, multi-duct, annotations, heat-source (galley) | props (bunks, lockers, tables) |
| B.2 Engineering | kindling, heat deposit, drag/wave, lamps, `airlock_controller` runtime | airlock pair emission, heat-source, initial-T field | engine/reactor/pipe props |
| B.3 Medbay | doors, vents runtime, `inject_gas_n_vec`, lamps, glass | open doorway, multi-duct, airlock pair, heat trigger, bulk-gas payload; locked door blocked on #135 | cold morgue (physics: sub-ambient), bed/cabinet props, white light look |
| B.4 Cargo | furniture rows in lines, `fuel_gas` + `emit_gas`, lamps | open doorway, cargo entity door, heat trigger; old-hull row after the migration | cold storage (physics), crate props |
| B.5 Armory | payloads, explosion, `timed_charge`, `MAT_DOOR` | heat trigger; locked door blocked on #135 | rack/crate props |
| B.6 Garden dome | `MAT_FOLIAGE`, `propgen`, `static_props`, glass, water | multi-duct, annotations | a sun/star sky for a dome (ruling); hedge row after the migration; dome-bursts-under-fire as feature? |
| B.7 Pool / aquarium | water sim, glass, `MAT_DOOR` | multi-duct, annotations | caustics (#97), octopus later |
| B.8 Doorless room | airtight gate, vents runtime, `MAT_WOOD`, `derelict` filter | structure mask, secret reachability, multi-duct | none |

**What can start before any ruling:** E.1 in full, the SPACE_CODE migration,
B.6, B.7, B.8, B.1 without its galley, B.5. **What waits on Part F:** the
heat trigger and heat-source entity (item 6), the cold rooms (item 3), the
dome's sky (item 5).

---

## Part F — agenda for the design session with Erik (one question at a time)

1. **The word.** "department" for the room cluster (this doc), or rename the
   two painted `zone` entities and take "zone" back? Draft: department.
2. **Mirror policy.** Departments as mirror pairs + axis departments (§0.3),
   secrets mirrored. Draft: yes for v1; asymmetric dressing is a later
   archetype with a structural symmetry gate.
3. **Sub-ambient temperature** (cold storage, morgue, the sealed deck). Today
   solids have a low rail at ambient (a hit is RED in every gate) and the
   emission table's domain starts at ambient. Options: (a) a per-level
   ambient so the cold room IS ambient and the ship is warm; (b) extend the
   rails and E° below ambient to a new floor. This is a physics-model ruling
   and a radiation-reference change (the integer reference first). Draft: (a)
   is cheaper and keeps the reference; (b) is the real thing.
4. **Secrets budget.** Own budget (one mirrored pair per ship) or part of
   #135's interactivity budget? Draft: own.
5. **The dome.** No daylight exists (space sky is dark). A lit sky for a dome
   boundary, or lamps only? And is a hull-glass dome that bursts under fire a
   feature or a bug? Draft: lamps in v1; corridor-side window wall, not hull
   glass.
6. **New host-side tick logic.** The heat trigger (on `timed_charge`) and the
   heat-source entity are new slot-9e tick logic, which the RL-batch habit §A
   says to avoid. Draft: both ride the existing entity runtimes' pattern and
   are small; accept, noting they join the resident port like the rest of 9e.
7. **Dust.** Erik wants a dust gas row (grey, settles faster than smoke); a
   gas row = one int32 plane per level, transported every tick on both
   backends, digested (goldens regenerate). Draft: yes, after the first
   departments play, because the grain store and the explosion's dust both
   want it.
8. **High-O2 demo.** Run the O2-rich pocket as a studio scene once the
   bulk-gas payload exists, and watch it together before the medbay recipe is
   final.

---

## Systems

**(a) Existing canonical systems this design uses** (CLAUDE.md table):
Level generators (`tools/gen_ship.py` stage 2 `embed` + `validate`, through
`level_lib`); Vent / duct system (every supply and return is a vent entity on
a duct; `[filters.*]` by name — `derelict` where smoke must pass); Gas pump
primitives (`inject_gas_n_vec` for the bulk-gas payload); Payload executor +
Timed charge (the heat trigger extends `timed_charge`, the cook-off and
hazmat rows are `[payloads.*]`); Starting a fire (every igniter delivers
HEAT — the heat-source entity included); Gas energy seam (every injected or
seeded N·T is booked); Material / gas tables (every new wall, hedge, hull
variant = a row after the migration; no new gas row in Part B); Prop tile
stamps + Prop generator + Static props (the dome's trees, every prop); Spawn
rooms + Match config (unchanged); Entity system (door, airlock_controller,
sensors, gates for ZEBRA wiring); Frame lights (a lamp per room is a
`[[light]]` row; `sky_for_level` is the dome's door if item 5 says yes);
Wall-destruction render seam (the breakable wall's reveal); Level data layer
(one writer; the CSV migration and the initial-T field go through it);
Determinism gates (the field digest does not move for Part B; the bake
goldens are re-run on the migration).

**(b) New systems this design creates, for others to reuse** (draft rules,
written into CLAUDE.md only when the code exists):

- **Department recipe** — `A department is a data row (rooms, adjacency by
  edge type, cell, ducts, physics signature, secrets) the grammar expands and
  the embedder stamps; never a per-department code path.` (#27 → #133)
- **Edge-type table** — `Every room adjacency is one of the edge kinds in
  the vocabulary's table, each mapped to what it stamps, whether it bounds a
  compartment and its L9 strength; a new way two rooms join is a new row
  there, never an inline stamp.` (§0.4 → `gen_ship.Edge.kind`)
- **Sealed-room vent rule** — `A room whose every edge is gas-tight carries
  both a supply and a return or neither; a duct never crosses a cell
  boundary.` (§0.4–0.5, #134)
- **Heat trigger** — `A stored hazard that goes off when its tiles get hot
  is a timed_charge with trigger = "heat": one latch, one threshold, one
  execute_payload — never a per-hazard check.` (E.2)
- **Bulk-gas payload** — `A payload that adds or removes BULK gas (O2, N2)
  goes through the gas pump primitives with an explicit per-gas vector and
  books through the energy seam; emit_gas stays the trace path.` (E.2)
- **Heat-source entity** — `A thing that is hot every tick (engine, stove,
  furnace) is a heat_source entity depositing through the temperature
  solver; when destroyed it delivers IGNITE_HEAT.` (E.2)
- **Secret annex rule** — `A hidden room is a grammar rule on its host
  department (breakable-wall edge or vent-only edge on the host's own duct),
  budgeted per ship, mirrored, content drawn from the dark list; the
  compartment labeller reads structure from edge kinds and the reachability
  gate counts a secret edge as a path.` (B.8, Part C)
- **SPACE_CODE migration** — `The tilemap CSV's reserved SPACE code lives
  outside the material id range (-1, cast at the loader); material ids are
  unbounded rows; level format v3.` (Part D)
