---
paths:
  - "src/simulation/verb_gates.py"
  - "src/simulation/charges.py"
  - "src/simulation/flash_blind.py"
  - "src/simulation/combat.py"
  - "src/simulation/weapons.py"
  - "src/simulation/loadouts.py"
  - "src/simulation/simulation.py"
  - "src/control_direct.py"
---

# Hand verbs: reload, throw, blind, strike, light, plant, and their gates

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.

## Magazines
- A reload DISCARDS the magazine in the gun with every round in it and loads a full one: rounds are never moved between magazines. `magazines` is the total carried, the one in the gun included; 0 or absent = untracked = the infinite auto-reload. The count belongs to the loadout's OWN weapon.
- The law is these functions and nothing else (the gate, spend and the verb all call `reload_begin` / `mag_settle`); a weapon with `mag_size` 0 touches none of the state. A stall is never interrupted and does not stop movement. HUDs read `mag_status`, never the fields.
- `_reload` asks `verb_gates.can_reload` FIRST and writes nothing when it says no; the lazy settle is the next trigger pull's or `mag_status`'s.
- A reload that STARTS appends ONE render-only `ReloadEvent`, from `reload_begin` alone. Bots do not call the verb (they auto-reload).

## Grenade kinds
- A new grenade is a hand-grenade `[ammo.*]` row APPENDED to the family (the throw verb's `kind` is its index in config order, frag = 0), never a verb. The door (`loadouts._grenade_counts`) refuses an unknown or non-hand-grenade row and a negative or non-integer count by name. `Unit.has_grenade` is the FRAG slot.
- The fuse is INTEGER ticks, converted once at the caller (`weapons.fuse_ticks_from_seconds`). THE WIND-UP: the grenade leaves the hand after the unit's throw wind-up, along the facing it has THEN; the count is spent when it LEAVES; a unit that dies or is stunned in the wind-up throws nothing and keeps its count.
- Whether a throw would start is `verb_gates.can_throw`. The selected kind is CONTROL state (`DirectControl.grenade_kind`), never sim state.
- THE FLIGHT (#215). A thrown grenade never passes a wall (Erik, 2026-11-17). Its flight is a chain of LEGS on `combat.Projectile` (`start_*` / `target_*` the CURRENT leg, `leg_tick`, `bounces`): the first leg is the throw, from the HAND to the target. A THROW THAT MEETS NO WALL DETONATES ON THE TILE AND TICK IT DID BEFORE #215 (its target is unchanged: the centre tile's point at the reach, so an aimed point is where a free throw lands; only its per-tick positions moved with the start); a change that moves an unobstructed throw's detonation is a gameplay change for Erik, never a refactor.
- WHERE IT STARTS (Erik, 2026-11-18): `combat.release_point` -- the thrower's Q16 centre plus `[weapons.hand_grenade] release_forward_m` / `release_side_m` (measured on the marine's throwing hand, the LEFT one: the side is negative) turned by the throw's vector, the piece centre -> hand walked so a grenade never starts inside or beyond a wall. Every continuous throw starts there; the WEGO order path keeps its legacy centre start.
- ONE walk: `combat.flight_walk` asks every tile the piece flown ENTERS through `combat.flight_blocked`: off the grid, `solid`, or a `[materials.*] blocks_thrown` row (derived = sealed to gas, i.e. solid; the grille sets it: Erik, 2026-11-18). What blocks a thrown thing is that column, never a per-material if; the rounds' march does NOT read it (a round through a grille is undecided). Asked against the map as it stands at the slot's start. Never a second walk, never `has_los` (sight), never the leg's target alone.
- A bounce is `Projectile.bounce` only (mirror in the face met, length = distance left x `bounce_restitution`, rounded once in Q16; a leg under a quarter tile or a wall after `bounce_max` rests it); the dials are `[weapons.hand_grenade] bounce_restitution` / `bounce_max`, refused outside 0..1 / below 0 at the table's load.
- `_update_grenades` flies EVERY grenade before ANY detonates (the device's shape); a fuse-out first resolves the rest of its flight at once (`settle`), then detonates at `combat.detonation_tile` (never a blocked tile), approach = `approach_tile()` (the current leg's start). The WEGO lob rides the same law.
- A change to `flight_blocked` / `flight_plane`, `flight_walk`, `release_point`, `detonation_tile`, `Projectile.update_position` / `bounce` / `fly` / `settle` lands in `cpp/src/cuda_unit_lib.cuh` (the walk, the release point), `cpp/src/cuda_unit_verbs.cu` (the throw's start) and `cpp/src/cuda_unit_objects.cu` in the SAME patch (`.claude/rules/device-units.md`, Grenade and charge slots).
- ACCEPTED GAPS (reported to Erik on #215): the observation's grenade velocity reads the current leg (so a bounce shows as the new leg's velocity); the release offsets are metres, but the marine is drawn 6 tiles tall whatever the tile, so the hand matches the picture exactly only at the 0.333 m tile, and a left-handed (render-only) unit still releases from the left; a grenade that never left the legacy painted door (`MAT_DOOR`) its thrower stood in (the centre's own tile is never asked), or under a door closed on it, detonates on the first free 4-neighbour in the order N, S, W, E.

## Flash blinding
- Through the `blind_*` columns (door `weapons._blind_columns`), applied by `apply_flash_blind` from `execute_payload`: never a per-grenade check. Every living unit of ANY team, the thrower included, in range with `GameMap.has_los`; a wall protects fully; a longer blinding is never shortened. Integer, no draw.
- ACCEPTED GAP: the blinded player's screen is not whited out and an agent's observation is not masked (the obs door is P8b's).

## Melee verb
- A blow is a melee-archetype weapon ROW named by the loadout, never a branch. Slot 4a consumes every latch and records on the SAME `attack_effects` as the shots, decided against the slot-start state; hit, stun and push all land at 4c.
- It strikes the ENEMY (another team, zombies included, never a teammate) adjacent and nearest its facing within the row's arc, ties to the lower id. The blow is `melee_blow`, the order path's own, with no facing snap (the arc is the aim). A whiff spends no cadence; the latch is consumed either way. A zombie's bite stays its own law.
- Integer, no draw: the loadouts door refuses a melee row with `crit_chance > 0`. Slot 4a asks `verb_gates.can_melee(..., at_tick_start=False)`. Bots do not call it.

## Flashlight bit
- ON by default; `set_flashlight` takes the OFF sense, so a zero policy row never darkens a unit; hashed ONLY while off.
- NOTHING in the sim reads it (light as game state is #81): the frame-light assembly draws a cone from it, and `carried_flashlight_cones` is the ONE choice of who is lit. A new unit light is a cone of that assembly, never a second path.

## Planted charge
- A planted charge is a `charges.PlantedCharge` and nothing else: never a timed_charge entity, a projectile or a second payload path. It fires through `charges.tick` with exactly ONE driver per path: the conductor's slot 9e on the continuous path, OnePhaseWEGO's timeline where the ruleset `drives_units`.
- The verb clamps its integer fuse to the plant row's range and the tap records the clamped value. Whether a plant would start is `verb_gates.can_plant`; a throw and a melee blow cannot start during a plant; death or a stun cancels it (nothing planted, count kept). The charge hurts everyone in range, the planter included, and credits the planter.
- The table holds `CHARGE_CAPACITY` live charges; a plant into a full one is refused and counted at THE door `charges.place`. Live charges hashed only while one exists (`CHARGE_SCHEMA` rows). Integer, no draw. Bots do not plant.
- ACCEPTED GAPS: the planter may walk during its plant; under TwoPhaseWEGO (legacy) `det_tick` does not follow the round rewind.

## Verb gates
- Whether a hand verb would act if issued now is ONE pure read per verb (no write, settle, event, draw, noise or cool-down), called by the verb at its consumption and by the policy's masks at the START of the tick: a mask never carries its own copy of a decision. A new hand verb brings its gate in the same patch.
- Possession is no term, and the TARGET is never a term. Gates are EXACT at both moments except for events that can only FORBID, so gate False at tick start ⇒ the verb does nothing (the one-sided promise); gate True may still do nothing. `can_plant` reads a wind-up as busy only if `throw_release_tick > tick`; `can_throw` is decided before the advances; `can_melee` takes the moment as a required keyword.
- THE TWO OVERSIGHTS ARE KEPT (design 11, Erik's to rule on): a stunned unit can reload and open a door, and a reload ignores a wind-up and a plant. They are pinned by tests, so a ruling is a deliberate test change.
- No gate for the trigger or the flashlight. `door_in_reach` is the use verb's own geometry per door.
