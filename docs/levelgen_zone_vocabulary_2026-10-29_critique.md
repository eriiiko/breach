# Critique — levelgen zone vocabulary (2026-10-29)

*2026-10-29, critic: Opus. Adversarial pass on
`docs/levelgen_zone_vocabulary_2026-10-29.md` against CLAUDE.md, the locked
decisions L1–L9 (`docs/breach_levelgen_design_v0.1.md`), `tools/gen_ship.py`
and the code. Part A (the unlimited list) is not critiqued line by line.
Research digest not read (other branch); D-references to it are taken on
trust.*

---

## 1. BLOCKER — Mirror symmetry is never mentioned, and the recipes break it

- **Section:** §B.2, §B.4 (D10 "engineering at one END … stores between …
  quarters"), Part C ("one or two secrets per ship"), every single-instance zone.
- **Claim:** zones are realisable by gen_ship's `embed` + `validate`.
- **Code:** gen_ship mirrors the wing left↔right (`gen_ship.py:11-23`,
  `ShipGraph.mirror_x`), puts the two spawn rooms at the two ENDS
  (`gen_ship.py:15-19`, `383`), and `validate` refuses a non-mirrored grid
  (`gen_ship.py:667-668`). A one-ended engineering, a single main hold, a
  single garden dome, or one secret per ship is either duplicated by the mirror
  or fails the symmetry gate. The ends are already taken by the spawns.
- **Fix:** decide, before the session, one of: (a) zones come in mirror pairs
  plus axis zones only (a single engineering lives on the axis, not at an end;
  D10's end-to-end gradient becomes axis→wing); (b) mirror is dropped and
  match fairness is guaranteed some other way (reachability distances
  balanced), which is an RL-fairness call for Erik; (c) mirror applies only
  to the gameplay skeleton and zone dressing is allowed to be asymmetric
  (requires changing the symmetry gate to compare structure, not materials).
  Then say for each recipe where it sits relative to the axis and the spawns.

## 2. BLOCKER — "returns through an open doorway" cannot exist: every gen_ship door is gas-tight, and an open doorway merges compartments

- **Section:** §0 second rule; B.1 (mess → stub → heads), B.2 (bunker via the
  engine-room door), B.3 (theatre, pharmacy via doorways into the ward), B.4
  (bays via arches).
- **Code:** gen_ship cuts doors as `MAT_DOOR` (`gen_ship.py:582`), the legacy
  door, "walkable-but-flow-solid" (`materials.py:36-40`; `[materials.door]`
  has no `permeability` key → 0). So every gen_ship door is already the
  "gas-tight door"; there is no gas-passing "joiner door". An actual open
  doorway / arch is `MAT_AIR` (or an entity door with `initial_state="open"`,
  which also stamps `MAT_AIR`, `door.py:66`), and `compartments()` then labels
  the two rooms as ONE compartment (`gen_ship.py:638-644`), so the room-count
  gate refuses the plan (`gen_ship.py:697-698`).
- **Consequence:** with flow-solid doors, a room that "returns through its
  doorway" is a sealed SUPPLY-only (or no-vent) room. Vents have no pressure
  feedback ("no makeup controller, no pressure sensing at all",
  `vent_system.py` module doc; fixed `q_circ` accrual) — a supply-only sealed
  room is pumped up while the return room is drained until its aperture is
  empty. Steel holds that, but door (`burst_threshold 2.0`) and glass (1.0)
  membranes pop on `spread > t` (`gamemap.py:2364`). The §0 rule as written
  produces over-pressure, not circulation.
- **Fix:** add an edge kind "open doorway" and change the compartment gate to
  count graph rooms joined by open edges as one compartment (a gate change,
  named); or require every sealed room to carry its own return (no
  doorway-returns until a gas-passing door material exists — which would be a
  new material row, see finding 14). State which.

## 3. BLOCKER — B.8: the vent runs the wrong way and the filter eats the smoke

- **Section:** B.8 "a return vent on the hidden side, a supply on the host's …
  a fire in the host room sends soot through the vent into the hidden room".
- **Code:** a RETURN vent EXTRACTS from its room and a SUPPLY vent DEPOSITS
  (`vent_system.py` doc, steps 1–2). With the return in the hidden room, air
  and soot flow hidden → host — the opposite of the hook. And gen_ship's only
  duct is `hepa_basic` (`gen_ship.py:96`), which scrubs `smoke = 1.0`
  (`config.toml [filters.hepa_basic]`): no soot ever reaches any supply.
  Also a hidden room with a single vent and no other opening is the finding-2
  pump-runaway (it is filled or emptied without limit).
- **Fix:** host = RETURN, hidden = SUPPLY, on a duct whose filter passes smoke
  (`derelict`), and pair it with a second path (a return in the hidden room
  on the same duct, or a leaky breakable wall) so the hidden room does not
  over-pressurise. Re-check that the reveal is visible (soot arriving in a
  sealed room is invisible to a player outside it; what the player sees is
  smoke going INTO the host's return grille).

## 4. MAJOR — B.8 gate change is understated; the breakable-wall variant fails a different gate, and furniture is not a wall

- **Section:** B.8, Part E row "Doorless room", Part C draft rule.
- **Code:** (a) vent-only variant: the hidden room is its own compartment,
  unreachable by a 3x3 unit → "compartments: 1 not reachable"
  (`gen_ship.py:699-711`). The doc names a gate change only for the breakable
  edge. (b) breakable-wall variant with wood: `compartments()` treats
  `MAT_WOOD` as room CONTENTS (`gen_ship.py:644`), so host + hidden merge into
  one compartment → the room-count gate fails (`697-698`) while reachability
  passes vacuously. (c) "furniture-row wall segment": furniture has
  `permeability 0.5` and `mobility 400` — it is walkable (units clamber over
  it) and not solid, so in a hull wall it is a vacuum leak (`find_leaks`,
  `gen_ship.py:663-666`) and in a steel wall it is a door, not a secret.
- **Fix:** breakable wall = wood only (solid, `mobility 0`); `compartments()`
  must take the structure mask from the graph's edge kinds (or count a
  breakable-edge region separately); reachability must accept "reachable via a
  breakable or vent-only edge". Two gate changes, plus the compartment
  labeller. Say so in Part E.

## 5. MAJOR — Cold storage / morgue as a "cold thermal-solid row" does not work today, and neither does fog or sinking smoke

- **Section:** A.6 cold storage, B.3 morgue, B.4 cold storage, Part D, Part E,
  Open question 2.
- **Code:** a material row has no temperature column (temperature is state).
  A thermal solid below ambient is lifted back to 0 by the low rail
  (`temperature_solver.cpp:383-388`), which is documented as "a COUNTED
  DIAGNOSTIC … must be INERT in every gate scenario (a hit inside a gate run is
  a RED)" (`:366-370`). The emissive table's domain starts AT ambient (`E°`
  floors at bucket 0, `config.toml:557-559`), so sub-ambient emission is not
  modelled. There is no condensation (no fog), and the sim is 2D with no
  gravity and no buoyancy (`config.toml:380-381`), so "smoke sinks instead of
  rising" has no meaning; `ceiling_gases` is a fixed per-species render layer
  (`config.toml:1637`). OQ2's alternative (a seeded cold gas region) is
  representable (gas `T_MIN = -292`, `config.toml:789`) but "warms back over
  minutes" is wrong — nothing relaxes to ambient (Pass 3 deleted) — and the
  walls it chills by conduction go sub-ambient and hit the same rail.
- **Fix:** move cold storage to "needs a session" (a physics session with
  Erik: sub-ambient solids = the low rail's role, the E° table below bucket 0,
  an initial-temperature level field). Drop "fog" and "smoke sinks" from the
  recipes, or park them on the water boil/freeze arc. B.3 and B.4 must stand
  without the cold.

## 6. MAJOR — The O2 store cannot be built on `inject_gas_n`, and no payload can add O2 today

- **Section:** B.3 physics, Part E Medbay, Systems (a) "Gas pump primitives".
- **Code:** `inject_gas_n` injects at the FIXED 21/79 O2/N2 mix
  (`gamemap.py:3149-3152`): it raises pressure but not the O2 mole fraction,
  so a nearby fire does not "roar" (the sustain law is linear in X, enrichment
  only above `o2_frac_amb`, `fire_simulation.cpp:161-171`; the pressure factor
  is already 1 above `p_full_atm = 0.5`). Payloads add gas only through
  `emit_gas` → a FieldEdit on the `gas` policy (`payloads.py:59-90`), which
  REFUSES a conservative slice (`field_edit.py:614-630`); O2 is conservative
  (`config.toml:1302`). And `inject_gas_n*` is the slot-9e actuator path, not
  a payload path (CLAUDE.md Gas pump primitives row).
- **Fix:** name it honestly as a new payload effect: a `[payloads.*]` bulk-gas
  column executed by `execute_payload` through `inject_gas_n_vec` with an
  O2-only vector (the energy seam books it as minted at ambient, or at the
  bottle's T), with its own closure test. Move it from "cheap" to "new
  mechanism", next to the heat trigger it depends on.

## 7. MAJOR — "Heat-triggered payload reuses apply_temperature_ignition's threshold shape" is not a reuse

- **Section:** B.5, Part E Armory, Systems (b) draft rule.
- **Code:** `apply_temperature_ignition` (`combat.py:468-493`) compares the
  tile's T against a PER-MATERIAL `ignition_temp_q16`, keeps a PER-TILE synced
  `ignition_armed` plane, and seeds `fire`. A cook-off needs a PER-HAZARD
  threshold, a fire-once latch per hazard that is synced (ENTITY_SECT
  digest), a location, and an `execute_payload` call in a tick slot — which is
  `timed_charge`'s shape (`entities/charge.py:36-60`: x, y, payload, enabled;
  digested countdown), not ignition's. A second, parallel "hazard system"
  would violate "extend, never parallel".
- **Fix:** specify it as an extension of the `timed_charge` entity (a
  `trigger = "heat"` + `trigger_T` field; the countdown becomes the fuse
  delay), run in the same slot-9e runtime. Flag the RL-batch §A habit ("no new
  host-side tick logic") for a ruling — this is new host-side tick logic.

## 8. MAJOR — L5 violated: plain doors on cell boundaries

- **Section:** B.2 ("its own cell. Its boundary doors are ZEBRA-class"), B.5
  (own cell, off the spine "by a locked door"), B.6 (own cell, off the spine
  "by a gas-tight door"), B.4 (own cell, joins not stated).
- **Rule:** L5: "Every edge crossing a cell boundary must be an airlock-pair or
  sealed bulkhead — never a plain door." A door that closes on alarm, a locked
  door and a gas-tight door are all plain doors.
- **Fix:** every own-cell zone joins its neighbours by an airlock pair (L5
  recipe, as B.3's isolation room does) or a bulkhead; or the session reopens
  L5 explicitly (e.g. "a ZEBRA door counts as a bulkhead"). Not drift.

## 9. MAJOR — "One cell = one duct system" is contradicted by the recipes; dampers and shut-offs do not exist

- **Section:** §0 rule 1 vs §0 rule 2 (spine gets its own duct) + B.1 (quarters
  share the spine's cell; galley on a separate duct), B.3 (morgue and
  isolation with their own ducts inside the medbay cell).
- **Code:** dampers / shut-offs are vent `state` "closed", reserved for #49;
  "v1 forces open behavior regardless of the authored state"
  (`entities/vents.py:22-23, 48-50`). ZEBRA closure on alarm is wire-able
  from existing parts (smoke sensor → `gate_or` → door `close` input,
  `sensors.py`, `nodes.py`, `door.py` INPUTS) but only for ENTITY doors, which
  gen_ship never emits.
- **Fix:** restate rule 1 as "a duct never crosses a cell boundary" (several
  ducts per cell allowed), or make the spine its own cell. Mark every
  damper/shut-off as depending on #49, and ZEBRA on entity doors + wires in
  stage 2.

## 10. MAJOR — The edge-type vocabulary is not expressible by stage 2, and is not mapped to L9

- **Section:** §0 Adjacency field; every recipe's Adjacency line.
- **Code:** `Edge.kind ∈ {"door", "window"}` (`gen_ship.py:156-163`, `582`).
  Of the doc's types, only "gas-tight door" (= `MAT_DOOR`) and "window" exist.
  Missing: joiner door (no gas-passing door material), bulkhead (an entity
  door? steel?), vent-only, breakable wall, arch, roller door, locked/terminal
  door (`terminal` is format-reserved, no behavior, `entities/controls.py:1-7`).
  L9 fixes the per-edge contract as `must-share-wall | may-become-corridor |
  routed-freely`; the doc never says which of its edges is which.
- **Fix:** a table: each edge type → tile/material or entity it stamps, its
  L9 adjacency strength, whether `compartments()` sees it as a boundary, and
  "exists / stage-2 extension / blocked on #135 or #49".

## 11. MAJOR — Stage 2's vent builder supports one duct and one vent per role

- **Section:** every Ducts line; Part E Crew quarters ("vent pair (gen_ship)" as
  existing).
- **Code:** `_build_vents` builds ONE duct `duct_main` with ids
  `f"vent_{role}"` (`gen_ship.py:549-563`): a second supply or return collides
  on id; each room carries one `vent` tag. The runtime itself supports many
  vents per duct and many ducts.
- **Fix:** name the extension: per-zone duct ids, a unique vent id per room,
  a per-duct filter, a room may carry both roles.

## 12. MAJOR — "zone" already means something else in the code

- **Section:** title, §0.
- **Rule:** design v0.1 §4: "one word, one meaning. If the existing level
  generator uses a word (e.g. 'zone') with a different meaning, either the
  meanings are reconciled or one side is renamed."
- **Code:** `src/simulation/entities/zones.py`: `breach_site` /
  `extraction_zone`, painted into `zones.npy` (`level_lib.write_zones_npy`).
- **Fix:** pick a different word for the room cluster (e.g. "department",
  "suite") or rename the entity zones, recorded in the vocabulary page.

## 13. MAJOR — Part E puts several new mechanisms in the "cheap data row" column

- **Rows that are wrong:**
  - "galley heat-source entity row", "engine entity row": no heat-source
    entity CLASS exists (entities: airlock_controller, decider, gates,
    filter, pump, light, timed_charge, sensors, door, prop, vents, zones).
    This is a new class + runtime that deposits heat every tick, plus "becomes
    a fire when its hp goes" (must deliver heat, Starting-a-fire rule).
  - "hot-ambient seed annotation": the level format has no initial-
    temperature field (`air_init.npy` is atmosphere only,
    `level_loader.py:700-705`); this is a level-format extension through
    `level_lib` + the loader door + a seam-booked init (`reseed_gas_energy`).
  - "O2-store payload row" (finding 6), "cold thermal-solid material row"
    (finding 5), "locked-door terminal (#135)" (terminal has no behavior).
  - Cargo "cargo door = `seal_tiles`": `seal_tiles` is a topology seam; a door
    to space is an entity door (A6), which gen_ship does not emit.
  - Garden dome "sky door … none for v1": the sky is per boundary TYPE and
    `[light.sky.space]` is `overcast = [0,0,0]` (`config.toml:1575-1576`,
    `frame_lights.py:292-304`). A ship (boundary "space") gets no daylight
    through its glass. "Trees as walls / cover / hedge labyrinth" also fails:
    foliage is walkable (`mobility 1000`), see-through (`light_atten 0`) and
    gives no cover (`cover_exposure 1.0`), and `PROP_MATERIAL_CHOICES =
    ("foliage",)` (`entities/prop.py:35`). A blocking hedge is a NEW material
    row.
- **Fix:** move these to "new, needs a patch" (or a session) and add a sun /
  starlight source for the dome as an open question.

## 14. MAJOR — Part D: the migration IS on Part B's critical path

- **Section:** Part D last paragraph, Part E footnote.
- **Claim:** "at most two new rows … so it is not on their critical path."
- **Rule/code:** CLAUDE.md Material/gas tables: "an 11th material needs
  Erik's SPACE_CODE-migration ruling FIRST". Ten ids exist (0–9,
  `materials.py:29-71`). The cold solid (if it survives finding 5), the old
  hull and the blocking hedge (finding 13) are ids 10–12. Part E itself says
  "within today's budget if … SPACE moves". The doc contradicts itself.
- **Fix:** say plainly: any new row in Part B needs the migration first (or
  re-use an existing row). Also the stated reason is imprecise: id 10 does
  not collide with SPACE 9; the collision is foliage's (it already cannot be
  painted). Name the real blocker the CLAUDE.md rule guards.

## 15. MINOR — Part D's migration scope and "golden-neutral"

- **Code:** `SPACE_CODE` is referenced in ~26 files, not one key in
  `align_level_art.py`: `map_editor.py` (13), `level_edit_common.py`,
  `bake_level_art.py` (the golden-gated art path), `make_tileset.py`,
  `gen_explosion_studio.py`, `gen_fire_smoke_studio.py`, `eos_p5_bake.py`,
  `gen_ship.py` (+ its CSV legend line `862`), and ~15 tests that pin 9.
  `255` does not fit the `int8` material plane (`gamemap.py:285`,
  `level_loader.py:1415-1417` casts to int8): use `-1` or state the cast.
  "Golden-neutral" holds for the field digest (the planes do not change) but
  the bake goldens must be checked, not asserted.

## 16. MINOR — Stage-2 annotations the recipes need are not named

- **Code:** `Room.kind ∈ {spawn, section, room}`; `embed` reads only tags
  `fuel`, `vent`, `spawn_team`, `zombie` (`gen_ship.py:135-147`, `568-606`);
  lights are one fixed `LAMP_RGB` per room (`gen_ship.py:602-604`).
- **Missing:** the tag schema a zone writes per room (zone id, room role,
  cell id, light colour/level, fuel density, material overrides such as glass
  hull, water group, props), and how zones coexist with the two spawn rooms and
  the zombie section that `validate` A*-checks (`gen_ship.py:683-687`).
- **Fix:** a Systems (b) row for "stage-2 room annotations", with the list.

## 17. MINOR — A glass dome in the hull is a one-tile membrane at its burst limit

- **Section:** B.6 "the dome's hull tiles are glass".
- **Code:** glass `burst_threshold 1.0`, pops on `spread > t`
  (`gamemap.py:2364`); a one-tile membrane between ~1 atm and vacuum sits
  at the threshold, so the first fire or vent overpressure blows the dome.
  "A fire here is seen from the corridor" needs a glass wall on the corridor
  side, not in the hull.
- **Fix:** decide whether that is the feature (a dome that fails under fire)
  or give the dome a stronger glazing row (another material, finding 14).

---

## Overall verdict

The eight recipes cannot go to the design session as written. The framing
(zone = recipe with a physics signature) is sound, but three things are
BLOCKERS for every recipe: the mirror (finding 1) is never addressed although
stage 2 and the gates enforce it; the supply/return topology rests on gas
passing through doorways that are gas-tight in gen_ship and would merge
compartments if opened (finding 2), with a pump-runaway when it doesn't; and
B.8's signature hook runs backwards through a filter that removes the smoke
(finding 3). Several physics signatures are claimed to exist that do not (cold
solids, fog, sinking smoke, O2 enrichment via `inject_gas_n`, daylight in
space, trees as walls), and the "heat-triggered payload" is a new entity
trigger, not a reuse. Before the session: settle the mirror policy, add an
edge-type table that maps to stage-2 tiles, gates and L9, fix L5 on the
own-cell zones and the one-cell-one-duct contradiction, rename "zone",
rewrite Part E with the true costs (heat-source entity, O2 payload effect,
heat trigger on `timed_charge`, level initial-T field, multi-duct builder, two
gate changes), and move cold storage to a physics session with Erik. With
those changes, B.1, B.5, B.7 and a fixed B.8 are realisable with named
stage-2 extensions; B.3, B.4 and B.6 lose part of their signature until the
physics and material work lands.
