---
paths:
  - "src/simulation/match.py"
  - "src/simulation/match_session.py"
  - "src/simulation/ruleset.py"
  - "src/simulation/team_picture.py"
  - "src/simulation/unit_orders.py"
  - "src/simulation/room_graph.py"
  - "src/simulation/orders.py"
  - "src/simulation/intents.py"
  - "src/simulation/action_registry.py"
  - "src/simulation/timeline.py"
  - "src/control_source.py"
  - "src/control_policy.py"
  - "src/control_direct.py"
  - "src/control_gamepad.py"
  - "src/control_match.py"
  - "src/control_zombies.py"
  - "src/control_onephase.py"
  - "src/match_replay.py"
  - "main.py"
  - "tools/run_match.py"
  - "renderer/follow_camera.py"
---

# Match: config, spawn, controllers, replay, control, the team's knowledge

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.

## Match config
- The ONLY way a match is parameterised: never argv scans, never per-test dicts. Test fixtures write it through `level_lib.write_match`, which re-parses through the loader's door first. The door refuses an unknown objective, loadout, profile or controller and `n` over the room's `max_per_team`. The seed is `Simulation(seed=)` and nothing else draws.
- `sight = "cone" | "all_round"` (#195 1a) is the match's ONE sight rule, the same for every seat; `match.check_sight` is THE door (also for a run file's `[run] sight`). It is written and recorded only when not the default, so every existing match.toml keeps its bytes and an old recording reads as cone.
- `MatchRules` ends on fewer than two marine teams alive (zombies are no team) or the time limit.

## Spawn rooms
- Where a match's teams stand is the LEVEL's decision (Erik, 2026-10-29): one room per team. v1 has ONE placement rule, `formation`; a new rule is a new `placement` value, refused until it exists. The loader refuses a room off the map, too small for its `max_per_team`, or with a cell off open floor.
- `Simulation.spawn_match(spec)` is the ONE match spawn (a fresh sim only); `[[spawn]]` stays the authored single-player path and a match never reads it. A recipe ship's spawn rooms are `gen_ship.centred_spawn_box` boxes.

## Match controller registry
- A controller that drives a team is a registry row, called once per tick as `handle_frame(sim, None)`, so it must work with no renderer. The bots and the horde register here, never a second control path; duplicate names are refused.
- In a match every unit has ONE driver and it is a controller: `create_match_controllers(sim, spec)` is the one builder (both runners use it): the team rows, then the zombie controller after `sim.hand_zombies_to_controller()`, from which the conductor stops driving the zombies. `zombies` names every living zombie through `Simulation.drive_zombies` (refused on a game path), so a match plays bit for bit as the conductor did (`tests/test_zombie_control.py`).
- `commander` is added by the builder for every team whose row names a `DOOR_CONTROLLERS` controller (a door driver with NO brain of its own) on a level with a room graph; a replay re-issues its recorded orders instead.
- `bots_v1` is kept UNCHANGED as a second opponent (a ladder of opponents, Erik): never tuned, its law pinned bit for bit (`tests/test_bots_v1.py`).
- `ControlSource.bind_match(sim, spec)` is the one way a controller learns its match's spec. The registry is separate from `--control`'s human schemes.

## Match recording / replay
- Replay is RE-SIMULATION (#126): level + spec + seed + every intent per tick + periodic `(field_digest, tick_digest)` checkpoints. The field recorder is NOT the replay path.
- ONE tap: every intent entry point hands its call, normalised as the method normalises it, to the attached log BEFORE acting. Never log inside a controller. A new intent entry point records itself through the tap in the same patch (a row in `INTENT_KINDS`). `apply_action` orders are refused at record time.
- The replay is the controller `replay`, the match's ONE controller; it verifies each checkpoint (`ReplayDivergence` names the tick) and its header refuses another level, seed or spec (`ReplayRefused`).
- A format change, a changed argument's MEANING included, bumps `FORMAT_VERSION`; an old recording is refused by its header, never migrated.
- Watching is `main.py --replay PATH [--seek TICK]`, THE only playback path: a divergence pauses, never crashes; seeking is re-simulation; changed tiles repatch through the wall seam. The checkpoint digest stays in `tests/`: never a second digest.
- The follow camera (`renderer/follow_camera.py`, pure, raylib-free) is THE camera of every watched match. Every `[replay]` key is read through `match_replay.replay_key`, THE refusing door.

## MatchSession + the `policy` controller
- THE steppable match: `run_match` and the farm worker are its callers, never a second loop body.
- A policy drives units ONLY as this controller. Its action row's columns are GENERATED from `INTENT_KINDS` minus `EXCLUDED_KINDS` (`drive_zombies`; `order_intent`, the commander's verb); a new verb needs a `VERB_RULES` row (hold or latch) or it is refused. A zero row does nothing (the flashlight's OFF sense, the plant gated nonzero).

## Control seams
- Phase decisions via Ruleset; schemes are ControlSource subclasses; verbs are registry rows; durations live in Timeline only.
- WEGO (`TwoPhaseWEGO`, `--control wego`) is LEGACY; the main line is continuous control. New designs sample, gate and tune on the continuous path; WEGO's locked decisions constrain nothing new, and a WEGO regression is a one-line finding, never a design driver (Erik, 2026-10-31; incident: #140's wind law).

## Direct control
- It reads the pad AND the keyboard + mouse every frame and uses whichever was touched last. A device is a HALF that turns one frame into a `DeviceFrame` and never calls the facade; `DirectControl._emit` is the one frame → intents path. A new device is a half, never a second scheme.
- Keys are DATA: `[controls.keys]` through the refusing door `key_table`. A new verb is a row of the table plus a `DeviceFrame` field, never a key read in main.py; a bound-but-verbless action is listed in `INERT_ACTIONS`. The bindings are Erik's (2026-11-10).
- The scheme owns the camera keys; the mouse is ignored while `GameRenderer.pointer_captured()`.

## Team picture
- What a team knows of the enemy lives in ONE place: per marine team a frozen `Record` per enemy, plain values, never a live unit. Switched on only for a match; updated ONCE per tick at slot 11, never on demand (`update` refuses a skipped tick, `records()` a stale picture). Not synced, not hashed, in no digest or recording.
- A record is NEVER refreshed from the live unit. It is kept `[team_picture] memory_s` unless the team SAW the death (a kill out of sight, a frag's included, is not seen), so a record outlives an unseen death or conversion.
- The fire lanes come from state, never a controller's memory: `fire_lanes` through `draw_lanes`, THE fire-lane rasteriser.
- The TRACKERS (motion, last hit, the damage window, bars, death tick, the object tracker) are members of the picture on the same slot-11 driver, never memory on demand. `bars` is THE bar of the reward's health: never a second tracker. The observation reads them only through `obs_spec_v1.TeamKnowledge`.
- The bots, the observation and the aim mask read it: never a second picture.

## Order
- Set ONLY through `Simulation.order_intent`, the commander's verb: refused by name at the door BEFORE the tap, out of every policy row, recorded, LATCHED (the last call per unit per tick wins) and written at slot 1b'. It names a PLACE on the room graph (through `room_graph.place_at`), never a tile.
- Cleared at death, at conversion and when its tick comes; NOT when its target dies. Hashed only while held.
- INFORMATION ONLY until rung 4: nothing in the game reads an order for a decision; a reader is a design line first (`tests/test_orders.py` scans `src/` for the column names).
- A new kind of order is a code of `ORDER_WHAT`; a new commander is a caller of the verb, never a writer of the columns. The module is `unit_orders` (`simulation.orders` is WEGO's vocabulary).
- ACCEPTED GAP: under TwoPhaseWEGO (legacy) `order_until` does not follow the round rewind.

## Room graph
- An opening is every edge a unit can walk now or later (the `EDGE_KINDS` rows that `keep_clear`). Its id is its RANK under (lower room, higher room, first tile row-major), never a counter, so a recorded order means the same opening in a replay.
- Orders, the commander and the bots' room goals read THIS map, never a second.
- What is passable NOW is worked out, never stored (`opening_states`, `joined_pairs`): host-side reads cached per map, nothing synced or hashed, nothing in the tick calls them.
- At BASE resolution (a `--res`-scaled map is refused). A level without a graph has `rooms = None` and is refused for training.
- ACCEPTED GAP: a hole joins two rooms but has no id, so an order cannot name it.

## main.py
- New launch flags are `_parse_*` helpers, never argv scans in subsystems. `--replay PATH` takes level, spec and seed from the recording and refuses `--control`, `--warp`, `--res` and a disagreeing `--level`.

## Headless match runner
- No light, no recorder, no renderer import. Tests and benches call `run_match`, never a second loop (#119 P2).
