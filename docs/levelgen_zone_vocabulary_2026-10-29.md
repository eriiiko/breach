# Levelgen zone vocabulary — the ideas list and the first eight zones

*Dated capture, 2026-10-29. The content of #27 (vocabulary page). Design-only:
nothing here is built. Erik opened the zone talk in a parallel design chat;
this document is its output. Part A is the UNLIMITED ideas list (append
freely, never prune). Part B is the "implement next" set Erik blessed
wholesale the same evening ("perfect start, and not that simple frankly").
Part C is the secrets rule. Part D is the materials question. Part E is the
asset-cost view that the assets session will shop from.*

**Feeds:** #27 (this page), #133 (prefab pieces: a zone's rooms become
pieces), #134 (vent rules: every recipe below names its supply and return),
#136 (ship-design adjacency: every recipe names its neighbours), #137
(pressure cells: every recipe says which cell it lives in). **Reads:** the
research digest `docs/research/levelgen_litsearch_2026-10-29.md` (#132),
strand (d) above all, and `docs/breach_levelgen_design_v0.1.md` (L1–L9).
**Relates:** #120 (`tools/gen_ship.py`, whose stage 1 these zones replace
and whose `embed` + `validate` they reuse), #48 (vents), #60 (props).

---

## 0. The framing: a zone is a recipe, not a room

A **zone** is a cluster of rooms with one function (a medbay is an operating
theatre, a ward, a pharmacy, a morgue and a scrub room). In generator terms a
zone is a stage-1 object: it expands into rooms under rules. A **room** is
what the embedder stamps (#133's piece). A zone is described by six things:

| Field | What it says | Who consumes it |
|---|---|---|
| **Rooms** | required rooms; optional rooms with a probability; size ranges in tiles | grammar → embedder |
| **Adjacency** | which rooms touch, what edge joins them (joiner door / gas-tight door / bulkhead / window / vent only / breakable wall), what the zone wants next to it on the ship | grammar (#136) |
| **Cell** | does the zone sit inside ONE pressure cell (default) or span several (an archetype choice) | #137 |
| **Ducts** | which rooms have a supply vent, which a return vent, and which rooms return through an always-open doorway into a room that has one; does the zone own its duct system or ride the cell's | #134 |
| **Physics signature** | ambient air (gas mix, temperature), fuel load and where it sits, water, light level, hazards that are PAYLOADS waiting (O2 bottles, halon dump, dust) | embedder annotations → entities, props, `[[light]]` rows |
| **Secrets** | may this zone carry a hidden annex, and of which kind (Part C) | grammar |

The physics signature is the whole point. The sim already does gas, O2, fire,
water, vents, light and materials; a zone that only looks different is
wallpaper, a zone whose AIR is different plays different. Every row in Part A
carries its physics hook for that reason.

Two rules from the digest's strand (d) that every recipe obeys:

- **One cell = one duct system = one alarm zone = one door-readiness group**
  (SOLAS II-2, D2). A duct that crosses a cell boundary is a sabotage-map
  choice, never a default.
- **Every room has a supply and a return**; the return is its own return vent
  or an always-open doorway into a room that has one (ISS IMV, D6/D7, and
  Erik's toilets-intake / quarters-exhaust rule). The corridor spine gets its
  own duct so it can be held clear while a side cell burns (D2, D8).

---

## Part A — the unlimited list

One line per idea: **name** — rooms it implies · *physics hook*. Append at
will; a line here costs nothing and commits nobody.

### A.1 Ship essentials

- **Crew quarters** — bunks, heads, showers, mess, galley, laundry, lounge · *heads intake → quarters exhaust (teargas in the heads comes out over the sleepers); galley grease fire; laundry steam.*
- **Engineering** — engine room, reactor room, coolant pool, fuel bunker, pump room, workshop, switchboard · *hot ambient, heavy fuel, a reactor that leaks heat when shot; coolant pool is water.*
- **Life support** — O2 generator, CO2 scrubber, air plant, water cistern · *the trunk of every duct; kill it and the ship's vents stop; cistern is a water store that floods on burst.*
- **Bridge / CIC / comms / navigation** — glass, consoles · *terminals that unlock doors; glass is light 0.1 / heat 0.3.*
- **Cargo** — container hold, cold storage, hazmat bay, livestock pens · *crates = fuel clusters; see A.6 for cold storage and pens.*
- **Medbay** — operating theatre, ward, isolation / quarantine with own airlock, pharmacy, scrub room, morgue · *O2 bottle store as a fire accelerant INSIDE the medbay; quarantine beds as pressure cells.*
- **Hangar / airlock chain / EVA suit room** — *hull-side, depressurisation valve legitimate (MPLM fire response, D7).*
- **Drone bay** — racks, charging, a repair gantry · *drones repair the hull (Erik; the drone design lives on the Lenovo, out of this arc); a drone is a special unit, later.*
- **Corridor spine** — *its own duct (the pressurised-stair rule); the pressure sandwich as a map feature.*

### A.2 Military

- **Armory** — racks, ammo magazine, explosives bunker · *cook-off chain through `[payloads.*]`.*
- **Firing range** — *long sightline, nowhere to hide.*
- **Barracks**, **war room**, **brig** and **panopticon cellblock**, **interrogation room**.
- **Missile tubes + missile magazine** · *the biggest payload row on the ship.*
- **Repair bay with half-dismantled mechs** · *props, cover.*

### A.3 Luxury and civil

- **Casino with a vault** · *the vault is a secret-chamber variant with a terminal door.*
- **Disco with a fog machine** · *a literal gas source entity (steam row).*
- **Bordello wing** — numbered rooms off a corridor.
- **Spa and swimming pool** · *the water sim as a level feature.*
- **Ballroom**, **theatre with backstage and trapdoors** (trapdoor = secret annex), **cinema**, **promenade with shopfronts**.
- **Restaurant kitchen** · *grease fire, galley exhaust with its own shut-off (D2).*
- **Cigar lounge, sealed** · *the air is already smoke; a room that starts with trace.*
- **Chapel with organ**, **hotel floor**, **kindergarten and school**, **library** (paper = fuel), **museum of looted artefacts**, **zoo**.

### A.4 Science vessels

- **Aquarium hall** — a tank big enough for the octopus (#104–#112) · *glass walls, water.*
- **Terrarium / botanical garden under glass** · *foliage material burns; procedural trees as cover.*
- **Hydroponics** · *high CO2, low O2 BY DESIGN: fine for plants, bad for breathing; a gas row.*
- **Animal holding pens**, **clean lab with airlock**, **cryo bay with sleepers**, **specimen vault**, **observatory dome**, **insect protein farm**, **mushroom farm in the dark** (light 0), **bioreactor / sewage plant** (methane: a flammable gas row).

### A.5 Dark

- **Morgue**, **crematorium** (a furnace: a heat source entity).
- **Human experimentation ward**, **torture chamber behind a false wall**, **devil-worship chapel with a desecrated altar**, **ossuary made of the crew**, **taxidermy room**.
- **The Eyes Wide Shut mansion** — masks, ballroom, a corridor of locked bedrooms.
- **The Black Lodge** — red curtains, zigzag floor; a room that is a MIRROR COPY of another room on the ship (the embedder already mirrors).
- **The smokers' room** — the air is already poison.
- **The sealed deck** — frozen since the last breach, bodies in place; vacuum, cold, dark.

### A.6 Rooms you would not expect, and why they are fun

- **Server farm with halon suppression** · *fire triggers a gas dump that replaces O2: the fire dies and so does whoever is inside. One payload row + one gas row.*
- **Grain / flour store** · *a trace gas explosive at concentration; a spark becomes a dust explosion. New gas row, existing explosion path.*
- **Paint shop / solvent store** · *flammable vapour drifts along the ducts to a far room before it finds a flame.*
- **Cold storage / meat locker** (Erik: "I especially love the cold storage") · *a COLD SINK: breach it and fog rolls out, smoke sinks instead of rising; the cold is a thermal-solid row with low T.*
- **Livestock pens** (Erik likes) · *animals as non-player units later; straw as fuel; methane.*
- **Distillery / still** · *ethanol vapour + a heat source; the brewing crew's secret.*
- **Oxygen candle store** · *a fire that brings its own O2, so sealing the room does not kill it.*
- **A ship inside the ship** · *an older hull swallowed by the newer one; different materials, its own dead air.*
- **A room with no door** · *only vents lead in; found by cutting a wall or following the smoke. Airtight by definition, so it falls straight out of the airtight gate. The cheapest secret.*
- **Hall of mirrors** · *vision-oracle trickery, no physics cost.*
- **Flooded deck** · *a corridor you wade through while something swims below.*
- **Hedge labyrinth in a garden dome** · *procedural trees as walls, and foliage burns.*
- **Bowling alley / ice rink** · *absurd on a warship; a long open sightline.*
- **Quarantine ward with airlocked beds** · *each bed a pressure cell.*
- **Memorial wall of the dead crew** · *names of units from a previous match, free once the replay path (#126–#128) exists.*

---

## Part B — the first eight zones (blessed 2026-10-29)

The simple start: zones that reuse what exists (materials, props, water,
vents, trees, the fire studio) and need almost no new assets. Build order =
list order; crew quarters first because it exercises the supply/return rule
end to end. Each recipe uses the six fields of §0.

### B.1 Crew quarters

- **Rooms:** corridor stub (required) · bunk rooms ×2–4 (required) · heads
  (required) · mess (required) · galley (optional 0.7) · laundry (optional 0.3).
- **Adjacency:** bunk rooms and mess open on the stub by joiner doors; the heads
  off the stub by a gas-tight door; the galley off the mess. The zone wants to
  sit far from engineering with stores between (D10) and next to the medbay.
- **Cell:** one cell, shared with the corridor spine it hangs off.
- **Ducts:** heads = RETURN (intake), bunk rooms = SUPPLY (exhaust into the
  room); mess returns through its open doorway to the stub, which returns to the
  heads. Galley on a SEPARATE duct with a shut-off (D2) when present.
- **Physics:** ambient air; fuel = bunks and lockers (furniture clusters, as
  gen_ship stamps today); galley = one heat source + grease (kindling row);
  lights: one lamp per room, dim in bunks.
- **Secrets:** a hidden annex allowed behind a bunk-room wall (0.1).

### B.2 Engineering

- **Rooms:** engine room (required, large) · fuel bunker (required) · workshop
  (optional 0.6) · pump room (optional 0.4) · switchboard (optional 0.5).
- **Adjacency:** the engine room is the hub; the fuel bunker off it by a
  bulkhead door; the workshop and switchboard off the engine room by gas-tight
  doors. Sits at one END of the ship, stores and tanks between it and the
  quarters (D10).
- **Cell:** its own cell. Its boundary doors are ZEBRA-class (closed on alarm).
- **Ducts:** own duct system, separate from accommodation (D2). Engine room
  supply + return; bunker returns through the engine room's door.
- **Physics:** hot ambient (engine room seeded above ambient through the energy
  seam at load); fuel bunker = dense kindling/furniture clusters, the biggest
  fire load on the ship; one static heat source entity in the engine room
  (the engine) that becomes a fire when its hp goes; light: harsh, warm.
- **Secrets:** the still (A.6) as the annex, 0.2.

### B.3 Medbay

- **Rooms:** ward (required) · operating theatre (required) · pharmacy with the
  O2 store (required) · scrub room (optional 0.5) · morgue (required, cold) ·
  isolation room with its own two-door airlock (optional 0.4).
- **Adjacency:** the ward is the hub off the spine; the theatre off the scrub
  room off the ward; the pharmacy off the ward by a locked door (terminal);
  the morgue off the ward by a gas-tight door; isolation off the ward through
  its airlock. Wants the quarters next door and engineering far away.
- **Cell:** one cell; the isolation room is its own cell inside it (the L5
  airlock recipe; the first in-zone pressure cell).
- **Ducts:** theatre and pharmacy = SUPPLY; the ward = RETURN (the theatre
  and pharmacy return through their doorways into the ward); the morgue is
  gas-tight, so it has its own small supply + return. Isolation: its own
  supply + return with dampers (D2, D6).
- **Physics:** clean air; fuel light (beds, cabinets); the O2 store is a
  `[payloads.*]` row waiting (ruptured bottles = an O2 injection through
  `inject_gas_n`, which makes any nearby fire roar); the morgue is a cold
  thermal solid (the cold-storage precedent, A.6); light: bright white in the
  theatre, dim blue in the morgue.
- **Secrets:** the experimentation ward (A.5) behind the morgue, 0.15.

### B.4 Cargo hold

- **Rooms:** main hold (required, the largest room on the ship) · side bays
  ×1–3 (optional 0.6 each) · cold storage (optional 0.4) · hazmat bay
  (optional 0.3) · livestock pens (optional 0.3).
- **Adjacency:** the hold on the hull (things dock there, D4/D6), with a cargo
  airlock or a sealed cargo door to space; bays off the hold by open arches
  (no door) or roller doors (gas-tight when closed). Wants to sit between
  engineering and the quarters as the "stores" buffer (D10).
- **Cell:** one cell. The cargo door to space is the zone's depressurisation
  valve (D7).
- **Ducts:** hold supply + return on its own duct (cargo spaces are separately
  ventilated, D2); bays return through their arches.
- **Physics:** fuel = container rows (furniture clusters in lines, the sweet
  cover geometry); cold storage = cold thermal solid + fog on breach; hazmat =
  one `[payloads.*]` row (a solvent vapour release, flammable gas row, later);
  pens = straw fuel + animals as props now, units later; light: low, pooled.
- **Secrets:** the ship inside the ship (A.6), 0.1 — a sealed older hull
  section as a second material row behind the hold's back wall.

### B.5 Armory

- **Rooms:** armory (required) · magazine (required) · firing range (optional
  0.4) · explosives bunker (optional 0.3).
- **Adjacency:** the armory off the spine by a locked door (terminal); the
  magazine off the armory by a bulkhead door; the bunker off the magazine by a
  second bulkhead door; the range a long room off the armory.
- **Cell:** its own small cell (a magazine is subdivided for exactly this
  reason).
- **Ducts:** supply + return on the cell's duct; dampers on the boundary.
- **Physics:** the magazine is a `timed_charge` / `[payloads.*]` cook-off: a
  heat threshold on the magazine's tiles fires its payload row (this is the
  one NEW mechanism in the eight, "heat-triggered payload"; it reuses
  `apply_temperature_ignition`'s threshold shape). The bunker chains a bigger
  row. Fuel otherwise light; light: bright.
- **Secrets:** none by default (a magazine has no hidden rooms); 0.

### B.6 Garden dome

- **Rooms:** dome (required, large, round-ish footprint accepted as a rectangle
  for v1) · potting room (optional 0.5) · pump / water room (optional 0.4).
- **Adjacency:** the dome off the spine by a gas-tight door (its air is its
  own); potting room off the dome; wants a hull side (a sky boundary → the
  `[light.sky.*]` door gives it daylight through glass).
- **Cell:** its own cell.
- **Ducts:** supply + return on its own duct (its air is enriched / humid; a
  hydroponics variant swaps the air to the high-CO2 mix, A.4).
- **Physics:** fuel = procedural trees and foliage (`MAT_FOLIAGE`, `propgen`,
  `static_props`) — the one zone where the fuel is the walls; the dome's hull
  tiles are glass (light 0.1 / heat 0.3), so a fire here is seen from the
  corridor; water = a pond tile group (optional 0.5); light: sky through the
  glass by day, lamps by night.
- **Secrets:** the hedge labyrinth variant (A.6) IS the dome at 0.2.

### B.7 Pool / aquarium hall

- **Rooms:** hall with the tank or pool (required) · pump room (required,
  small) · changing rooms (pool variant, optional 0.6) · viewing gallery
  (aquarium variant, optional 0.6).
- **Adjacency:** the hall off the spine by a joiner door; the pump room behind
  the tank by a gas-tight door; gallery alongside the tank behind glass.
- **Cell:** the cell it hangs off.
- **Ducts:** humid air; supply + return on the cell's duct; the pump room is
  the natural return.
- **Physics:** water (the water sim as a level feature; the tank walls are
  glass; bursting them floods the hall); the octopus's home (#104–#112, later);
  fuel light; light: caustic pool light later (#97 water redo), lamps now.
- **Secrets:** the flooded deck (A.6) below as the pump room's annex, 0.1.

### B.8 The doorless room (the first secrets rule)

Not a zone: a GRAMMAR RULE any zone may apply. A room with no door edge at
all, joined to its host room by exactly one `vent only` edge (a return vent on
the hidden side, a supply on the host's), or by a `breakable wall` edge (a
wood/furniture-row wall segment in a hull or steel wall). It is airtight by
construction, so `validate`'s airtight gate passes as is; the A* reachability
gate must learn that a room reachable only through a breakable edge counts as
reachable (the one gate change). Smoke finds it: a fire in the host room sends
soot through the vent into the hidden room and a player who watches the smoke
go into a wall knows. What is inside is the host zone's secret row (the still,
the ward, the altar, the hoard).

---

## Part C — secrets as a rule, not a room type

Any zone may carry ONE hidden annex, at the zone's own probability. The annex
is reached by exactly one of: a breakable wall (B.8), a vent-only edge (B.8),
a terminal-locked door (#135's budget), a trapdoor under a stage or a floor
plate (needs a vertical seam the engine does not have; parked). The horror
comes from CONTRAST, so the annex's content is drawn from the DARK list
(A.5) attached to a benign host: the torture chamber behind the kindergarten,
the altar under the chapel, the lodge behind the hotel's last door, the
experimentation ward behind the morgue. The generator budgets secrets per
ship (one or two), never per zone, so they stay secret.

---

## Part D — materials: the table is already unbounded

Erik's question (2026-10-29): "isn't it quite easy to create such a material
list on the fly? … could we have materials that is the whole parameter space?"

**Answer: yes, and the system is already built that way.** `MaterialTable`
(`src/simulation/materials.py`) is a per-id column store: every property
(hp, flammable, mobility, thermal mass, ignition temperature, fuel, heat and
light attenuation per channel, permeability, burst threshold, cover exposure,
wave coefficients) is a numpy column indexed by the tile's material id, and
every derived cache is one fancy-index lookup `column[material]`. Adding a
material is one `[materials.<name>]` row in config.toml. A zone that wants a
"meat-locker wall" (steel optics, cold thermal mass) or an "old hull" (hull
hp, more permeable) writes a row with those values; nothing in the sim knows
or cares how many rows exist. A row IS a point in the parameter space, so the
space is fully available; what we do NOT want is a continuous per-tile
parameter (per-tile property planes instead of an id) — that multiplies
digested state by the number of properties for no gameplay gain. Rows it is.

**The "budget" is two small artefacts, not a design limit:**

1. `gmap.material` is `int8` (127 ids). Fine for a long time; `uint8` or
   `int16` is a digest-spec version bump when the day comes.
2. The v2 tilemap CSV writes a tile's material id LITERALLY, with ONE reserved
   non-material code `SPACE_CODE = 9` (`level_loader.py`), which collides with
   `MAT_FOLIAGE = 9` (harmless today because foliage is never in a CSV; it is
   a prop stamp). An 11th material needs a code that cannot collide.
   Cleanest fix: move SPACE to a code outside the id range (e.g. `-1` or
   `255`) with a one-shot migration of every shipped CSV through
   `tools/migrate_tilemap_v2.py`'s pattern, written through `level_lib`, and
   the editor's palette key (`align_level_art.py`, key 9) re-pointed. One
   patch, golden-neutral (the material planes do not change), level format
   version bump to v3 so an old CSV is refused rather than misread.

Two real costs scale with row COUNT, both small: the per-pair face table
(`materials.py` ~l.1322, N×N int32) and the thin-rows rulings (a new
flammable thin row must pass `THIN_LIMIT_M` / `LUMPED_CRITERION_EXEMPT`,
which are PROVISIONAL until Erik's material design session). Erik's ruling
2026-10-29: "we can't be limited to 10 materials, so we'll need to rework it
at some point, no big deal" — the migration is accepted work, timing open;
the eight zones in Part B need at most two new rows (a cold thermal solid and
an old-hull variant), so it is not on their critical path.

---

## Part E — asset cost of the first eight

What exists today vs what is new, so the assets session has a shopping list.

| Zone | Exists (reuse) | New, cheap (data rows) | New, needs a session |
|---|---|---|---|
| Crew quarters | furniture/kindling clusters, vent pair (gen_ship), lamps, doors | galley heat-source entity row | bunk/table/locker props |
| Engineering | kindling, heat deposit, wave/drag, lamps | hot-ambient seed annotation, engine entity row | engine/reactor/pipe props |
| Medbay | doors, vents, `inject_gas_n`, lamps | O2-store payload row, cold thermal-solid material row, locked-door terminal (#135) | bed/cabinet/table props, white light look |
| Cargo | furniture rows in lines, cargo door = `seal_tiles` | cold-storage row (shared with morgue), hazmat payload row | crate/container props (Kenney-class OBJ) |
| Armory | payloads, explosion, `timed_charge` | **heat-triggered payload** (the one new mechanism), magazine payload rows | rack/crate props |
| Garden dome | `MAT_FOLIAGE`, `propgen`, `static_props`, glass, sky door, water | — | none for v1 (trees exist) |
| Pool / aquarium | water sim, glass | tank-wall row (glass today) | caustics later (#97), octopus later |
| Doorless room | airtight gate, vents, wood wall | A* reachability through a breakable edge (gate change) | none |

Material rows needed in total for Part B: a cold thermal solid (morgue, cold
storage, meat locker) and possibly an old-hull variant — within today's
budget if foliage keeps id 9 and SPACE moves (Part D).

---

## Open questions for Erik (one at a time, in session)

1. Does a zone's hidden annex count against #135's interactivity budget
   (terminals/buttons), or is secrets a separate small budget? (My draft:
   separate, one or two per ship.)
2. Cold storage as a THERMAL SOLID row (cold walls that chill the air) or as
   a seeded cold gas region (which warms back over minutes)? (Draft: thermal
   solid, so it keeps chilling; it also gives the "fog on breach" for free.)
3. The hydroponics variant's "high CO2 / low O2" air: is CO2 a new gas row
   (`[gases.co2]`, non-flammable, non-breathable) worth adding now, or does
   the garden dome stay at ambient air for v1? (Draft: ambient for v1.)

---

## Systems

**(a) Existing canonical systems this design uses** (from CLAUDE.md):
Level generators (`tools/gen_ship.py` stage 2 `embed` + `validate`, through
`level_lib`); Vent / duct system (every supply and return above is a vent
entity on a duct, `[filters.*]` by name); Gas pump primitives (the O2-store
payload injects through `inject_gas_n`); Payload executor + Timed charge
(magazine cook-off, hazmat release); Starting a fire (every igniter delivers
HEAT); Material / gas tables (every new wall, cold solid, gas = a row); Prop
tile stamps + Prop generator + Static props (the dome's trees, every prop);
Spawn rooms + Match config (unchanged; zones are not spawn rooms); Frame lights
(a lamp per room is a `[[light]]` row; the sky through the dome's glass is
`sky_for_level`); Wall-destruction render seam (the breakable wall's reveal);
Level data layer (one writer); Integer reference / determinism gates (a zone
is data, so no digest moves unless a new field is added).

**(b) New systems this design creates, for others to reuse** (draft rules,
written into CLAUDE.md only when the code exists):

- **Zone recipe** — `A zone is a data row (rooms, adjacency, cell, ducts,
  physics signature, secrets) the grammar expands and the embedder stamps;
  never a per-zone code path.` (#27 → #133)
- **Heat-triggered payload** — `A stored hazard that goes off when its tiles
  get hot is ONE mechanism: a threshold on the tile's temperature firing a
  [payloads.*] row through execute_payload, the shape of
  apply_temperature_ignition — never a per-hazard check.` (B.5; also the
  O2 store, the hazmat bay, the solvent store, the halon room.)
- **Secret annex rule** — `A hidden room is a grammar rule on its host zone
  (breakable-wall edge, vent-only edge, or terminal door), budgeted per ship,
  content drawn from the dark list; validate's reachability gate counts a
  breakable edge as a path.` (B.8, Part C)
- **SPACE_CODE migration** — `The tilemap CSV's reserved SPACE code lives
  outside the material id range; material ids are unbounded rows.` (Part D;
  lands with the first 11th material, not before.)
