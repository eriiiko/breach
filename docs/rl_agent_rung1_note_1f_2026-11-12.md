# Rung 1, patch 1f: the order (the note)

2026-11-12. Written for the orchestrator of rung 1 (#195) from the plan
(`docs/rl_agent_rung1_plan_2026-11-11.md` §3, §5 "1f"), the design
(`docs/rl_agent_design_2026-11-11.md` 5, 5.1-5.7, 6.7, 9, 10), the roadmap's
"Order" row (`docs/rl_agent_roadmap_2026-11-11.md` §7) and the critique
(lens 3 M4), checked against the code at main `6c8416e8` (P3a, 1b, 1d, 1e
merged). Same shape as the 1d and 1e notes. **CHOICE:** marks where the
design is silent. **STOP:** would mark a design/code contradiction; there is
none (see the end).

**Design:** 5.1 (what / where / until + an optional priority target), 5.3
(the first commander is the bots' team layer), 5.4 (information only until
rung 4), 5.5 (the format is the named spec "rooms"), 5.7 (unit state, one
verb, out of the soldier's columns, recorded, applied in the tick, hashed only
while set, written only for a door-driven team), 9 (four columns). **Why now:**
1g puts the order in the observation (5.2) and 1j's door-driven bots need it.

## What the code does today (the facts the build rests on)

- **Unit state is the schema.** `Unit.__slots__` is generated from
  `src/simulation/unit_schema.py` (`slot_names` `:676`); a column is a
  `_c(name, kind, block, default, enc, dec, ...)` row of `COLUMNS` (`:469`),
  its default applied by `apply_defaults` (`:699`), hashed by
  `digest_record` (`:722`) always or -- the default, `DIGEST_NONDEFAULT` --
  only while its encoding differs from the default's. The blocks run identity
  ... `# flags` (`:548`), `# intent state` (`:550`). `REFERENCE_ONLY` (`:635`)
  already holds WEGO's `orders` and `current_order_type`: the new names must
  not collide.
- **The verbs and the tap.** Every intent entry point records itself on the
  tap (`self._intent_log.record(self.tick, kind, args)`) and then latches on
  the unit (`plant_charge_intent` `simulation.py:1127-1160`). A verb that
  refuses an argument by name does so BEFORE the tap (`throw_grenade_intent`'s
  kind, `:1108`), so a recording never holds a refused call. The horde's verb
  `drive_zombies` (`:637-653`) is the precedent for a verb that is not a
  soldier's: it latches on the SIM (`self._zombie_requests`, `:451`), not on
  a unit, and is excluded from the policy's columns
  (`control_policy.EXCLUDED_KINDS`, `control_policy.py:77`).
- **Recording.** `match_replay.INTENT_KINDS` (`match_replay.py:104-117`),
  `FORMAT_VERSION = 7` (`:98`, history above it); the replay feeds a kind back
  by `getattr(sim, kind)(*args)` (`:534`). `control_policy.POLICY_KINDS` is
  `INTENT_KINDS` minus `EXCLUDED_KINDS` (`:79`); `_check_rules` refuses a
  `VERB_RULES` row for an excluded kind, so an excluded kind needs no row.
- **The tick.** `_step_pre_physics`: 1a `unit_impulse.integrate_all`
  (`simulation.py:1994`), 1b `_consume_direct_intents` (`:2001`; possessed
  units only, dormant on an empty set), 1c exposure (`:2006`) ... ;
  `_step_post_physics`: 10 `self.tick += 1` (`:2369`), `ruleset.on_tick_end`
  (`:2372`, which runs `convert_marines_to_zombies` every tick under
  `ContinuousRealtime` / `MatchRules`, `ruleset.py:280`), 11 the team picture
  (`:2379`).
- **Death and conversion.** The one death is `damage.apply_packet`'s
  alive -> dead branch, which already clears the impulse and horde state
  (`damage.py:410-421`). The conversion `ai_zombie.convert_marines_to_zombies`
  clears them again (`ai_zombie.py:401-432`). A converted unit was dead
  first.
- **The room graph (1b).** `LevelData.rooms` (`level_loader.py:1175`) is a
  `RoomGraph` (`:1665`): `rooms` (ids 0..n-1), `openings` (id = rank,
  `OpeningRecord.tiles` row-major `(y, x)`), `grid` int32, -1 outside every
  room (walls, voids, an opening's tiles, an airlock's vestibule). Base
  resolution only (`room_graph._check_shape`, `room_graph.py:50`). Today only
  `levels/ship_demo` carries one; every test box built by `level_lib` without
  `write_room_graph` (`level_lib.py:273`) has `rooms = None`.
- **The team layer (1d).** `control_bots.TeamBrain.update` (`control_bots.py:337`)
  sets `play` (`fall_back` / `hold` / `bound`, `PLAYS` `:283`), `target` (a
  live unit or None, `:376-378`) and `objective` (`_objective` `:399`: the
  TOP-LEFT tile `(x, y)` of a footprint body -- the own spawn room's centre
  cell, the hold anchor, the target's or a known marine's tile, the enemy
  spawn room's centre), every `team_period_ticks` (6). It is run only by
  `BotTeam.handle_frame` (`:879-903`), and reads `bots` only for
  `weapon_of`, `arrived` (True for a unit with no `_BotState`, `:1327`),
  `room_goal` / `enemy_room_goal`, `topology_key`, `tile_ok`.
- **The match's controllers.** `control_source.create_match_controllers`
  (`control_source.py:305-330`): one per team row (`row.controller`), then the
  horde's. `policy` is registered but not nameable (`:262`, `:267`): only the
  batched env puts it in a spec (`rl_env.EnvRunner.reset`, `rl_env.py:515-518`),
  and the spec is what a recording stores. `MatchSession.tick`
  (`match_session.py:97-108`) calls every controller, then `step()`.

## Build

### 1. The four fields (four rows of the schema)

A new block `# order (#195 1f)` in `COLUMNS`, between `# flags` and
`# intent state` (`unit_schema.py:548-550`), four plain `KIND_INT` rows
(`_enc_int`, `_dec_identity`, digest `DIGEST_NONDEFAULT`):

| column (= digest key) | meaning | default | codes |
|---|---|---|---|
| `order_what` | the kind | `0` (`ORDER_NONE`) | 0 none, 1 go to, 2 hold, 3 clear, 4 fall back to (`ORDER_WHAT = ("none", "go_to", "hold", "clear", "fall_back_to")`, the index is the code) |
| `order_where` | the place | `-1` (`NO_PLACE`) | a room id `r` (0 <= r < `ORDER_OPENING_BASE`) or an opening `ORDER_OPENING_BASE + o` |
| `order_until` | the first tick it no longer holds | `-1` (`NO_LIMIT`, until told otherwise) | an absolute `sim.tick` |
| `order_target` | the priority target | `-1` (`NO_TARGET`) | a unit id |

- `ORDER_OPENING_BASE = 1 << 16` (CHOICE: one int tells a room from an
  opening by range, independent of the level's room count; the door refuses a
  level with 65 536 rooms or more, which nothing comes near).
- Names `order_*`, never `order`/`orders` (WEGO's `orders` is
  `REFERENCE_ONLY`). Integer codes, no host enum (CHOICE: as `throw_kind`).
- **Hashed only while set:** a unit without an order holds all four defaults
  and adds no key. While an order is held, `order_what` is always present;
  `order_until` / `order_target` only when not "no limit" / "no target"
  (absence = the sentinel; no digest version moves, `UNIT_SECT_V2` unchanged).
- The schema's own tests accept them as they stand: `_check_unit` round-trips
  ints; `test_a_column_at_its_default_adds_no_key`'s `_sample` (default + 5)
  covers an int column (`tests/test_unit_schema.py:104`, `:172-196`). No
  `_SAMPLES` row is needed. (This is why the latch is NOT a fifth unit
  column, see 2.)
- The constants and the law live in a new module `src/simulation/orders.py`
  (CHOICE: one place, as `exposure.py` / `horde.py`): the codes, the door
  `check(sim, u, what, where, ticks, target_id)`, `apply_requests(sim)`,
  `expire(sim)`, `clear(u)`, and `place_name(graph, where) -> str` (a
  readable "room 3" / "opening 7" for messages and tools).

### 2. The commander's verb

`Simulation.order_intent(unit_id, what, where, ticks, target_id)` (CHOICE:
the name -- `*_intent` = latched and consumed inside the tick, like the hand
verbs; `apply_action` / `Order` are WEGO's and stay untouched).

- `ticks`: integer ticks the order holds, `0` = until told otherwise (every
  caller converts seconds once at its own door,
  `weapons.fuse_ticks_from_seconds`); `order_until = sim.tick + ticks`, or
  `NO_LIMIT` for 0. The verb records `ticks` (the relative value), so a replay
  issuing it at the same tick gets the same `order_until`.
- `what = 0` (none) CLEARS the unit's order; then `where` must be `-1`,
  `ticks` 0 and `target_id` -1 (CHOICE: a priority target needs an order).

**The door** (`orders.check`, called first, before the tap; `ValueError`
naming the argument, `TypeError` for a bool or a non-integer as
`plant_charge_intent` does, `simulation.py:1147-1150`):

| refused | message names |
|---|---|
| the level has no room graph (`sim.level.rooms is None`), or its grid is not the map's shape (a `--res`-scaled map) | "the order names a place on the room graph; level X has none" |
| `unit_id` names no unit; the unit is dead; the unit is a zombie | the unit and why |
| `what` outside 0..4 | the value and `ORDER_WHAT` |
| `what != 0` and `where` neither a room id `< len(rooms)` nor `ORDER_OPENING_BASE + o` with `o < len(openings)` | the value, the counts |
| `ticks < 0` | the value |
| `target_id` not -1 and not a living unit of ANOTHER team than the ordered unit (a zombie on `ZOMBIE_TEAM` is an enemy) | the target and why (none / dead / a teammate) |
| a non-none field with `what = 0` | which |

Every `what` takes a room or an opening (CHOICE: the design's table pairs
none). The door does not ask who calls (a seat: scripted, network, human).

**The latch, on the sim** (CHOICE, the `drive_zombies` precedent): after the
door and the tap, `self._order_requests[unit_id] = (what, where, until,
target)`; a second call for the same unit in one tick REPLACES the first
(CHOICE: the last word of the commander). `_order_requests = {}` is set beside
`_zombie_requests` (`simulation.py:451`) in `_reset_internal`, so `reset()`
drops it. Why not a unit column: the critique's M4 needs a commander's
`handle_frame` to change no unit record outside the intent block
(`tests/test_bots.py:529-565`); a sim-level dict is outside every record,
needs no optional codec and no `_SAMPLES` row, and is empty between ticks.
The verb does NOT possess the unit (`_possessed_ids` untouched, as
`melee_intent`, `:1214-1237`): an order is not a soldier's verb.

**Recording.** `INTENT_KINDS["order_intent"] = ("int", "int", "int", "int",
"int")` (CHOICE: `target_id` an `int` with -1 for none, the schema's own
encoding, not `int?`), inserted before `drive_zombies` (`match_replay.py:116`);
`control_policy.EXCLUDED_KINDS = ("drive_zombies", "order_intent")` (`:77`), no
`VERB_RULES` row. `FORMAT_VERSION = 8` with the history line:

> v8 (#195 1f, 2026-11-12): the `order_intent` kind (the commander's order:
> what, where, ticks, priority target), and a match with a team driven
> through the door carries the bots' commander, whose orders are hashed unit
> state -- a v7 recording could not name the verb and is refused.

Amend the tap's lists: `attach_intent_log`'s docstring (`simulation.py:626`)
and the "thirteen intent entry points" count in `CLAUDE.md` (now fourteen).

**Applied inside the tick, slot 1b'.** `orders.apply_requests(self)` is called
right after `self._consume_direct_intents()` (`simulation.py:2001`), with its
own ordering comment (god-file policy: one call line): if the dict is empty,
return (dormant); else, in unit-id order, for each request whose unit is alive
and not a zombie, write the four columns (or the four defaults for `what = 0`),
and drop every request (a dead unit's too: nothing stays latched -- 1e's
finding of a latch left set forever does not arise here). At 1b the unit was
already checked at the door in this tick's controller pass, before any slot
ran, so the alive check is defence only.

**Cleared** (`orders.clear(u)` writes the four defaults):

- **at death:** in `damage.apply_packet`'s alive -> dead branch, beside
  `horde.clear` (`damage.py:419-421`), so the order goes on the tick the unit
  dies, at whatever slot kills it;
- **at turning into a zombie:** in `convert_marines_to_zombies`, beside
  `_horde.clear(u)` (`ai_zombie.py:429`) -- redundant after the death clear,
  kept as the house rule ("zeroed at death and at conversion");
- **when its time runs out:** `orders.expire(self)` right after
  `self.tick += 1` (`simulation.py:2369`), before `on_tick_end`: every living
  unit with `order_what != 0` and `order_until != NO_LIMIT` and
  `order_until <= self.tick` is cleared. So between ticks an order held always
  has `order_until > sim.tick` (CHOICE: expiry after the tick advance, not at
  1b, so a controller or the observation never sees an order with no time
  left). An order of `ticks = d` issued at tick T holds for exactly the d
  steps T .. T + d - 1. One integer compare per unit per tick.
- **not** when the priority target dies (CHOICE: the target id is
  information; a reader checks the unit; the commander rewrites). Accepted
  gap, as the planted charge's: under `TwoPhaseWEGO` (legacy) the absolute
  `order_until` does not follow the round rewind.

**Information only.** Nothing in `src/simulation/` reads `order_*` for a
decision in 1f (design 5.4): only `orders.py` writes them, `damage.py` /
`ai_zombie.py` clear them, `unit_schema.py` declares them. The policy decoder
and the observation read them later (1g, rung 4).

### 3. The bots' team layer as the commander, only for a door-driven team

**How a team is known to be driven through the door** (CHOICE, the simplest
honest one): the match spec's own team row. `control_source.DOOR_CONTROLLERS =
frozenset({"policy"})` -- the controllers that drive a team through the action
spec (1j adds its door-driven bots by name). `create_match_controllers`
(`control_source.py:305`), after the team rows and before the horde, appends
`create_controller("commander", team)` for every team `i` whose
`spec.teams[i].controller in DOOR_CONTROLLERS`, **when the level has a room
graph** (`sim.level.rooms is not None`; without one there is nothing an order
could name, and the RL door's refusal of a graphless level is 1g's). The spec
is in every recording, and a replay builds only the `replay` controller
(`control_source.py:321-323`), which re-issues the recorded orders: a recorded door-driven match
replays without a commander running. A match.toml cannot name `policy`, so
**no existing match, run_match, the tournament, main.py's watched match or a
bots-alone farm run ever builds a commander or writes an order.**

**The commander:** `control_bots.BotCommander(BotTeam)`, registered as
`"commander"` with `nameable=False` (`_NOT_NAMEABLE`, `:267`). Its
`handle_frame` runs only the TEAM layer of `BotTeam.handle_frame`
(`:879-896`): once per tick; nothing when the team has no living marine;
`brain.observe(sim)`; `brain.update(self, sim, mine, vis)` on the same cadence
(`team_period_ticks`). No unit layer, no unit verb, no `_BotState`, so
`arrived` is True for every unit (CHOICE: the Bound halves then swap on
`unit_period_ticks` / `bound_max_s` alone; the halves are not part of the
order). On each tick the brain updated it translates (design 5.3):

- **what:** `PLAY_ORDER = {"fall_back": "fall_back_to", "hold": "hold",
  "bound": "go_to"}`, a data row per `PLAYS` row, checked at import (a play
  without a row is refused by name: a fourth play must name its order).
  `clear` is not written by the bots.
- **where:** `brain.objective` is a body's top-left tile `(x, y)`; the place
  is read at the body's CENTRE tile `(y + fp // 2, x + fp // 2)`, `fp =
  mine[0].footprint` as `_objective` uses (CHOICE). The read is one new
  function of 1b's module, `room_graph.place_at(graph, y, x)`: `grid[y, x]` if
  >= 0, else the lowest-id opening whose rectangle holds the tile
  (`ORDER_OPENING_BASE + o`), else `NO_PLACE` -- the one map from tile to
  place, never a second.
- **A tile on no room and no opening, or no objective:** the commander writes
  `what = none` (clears the order) (CHOICE: no order is more honest than a
  stale one or a guessed room; the case is rare: every objective is a tile a
  body stands on).
- **target:** `brain.target.id`, or -1.
- **ticks:** 0, until told otherwise (CHOICE: the team layer has no
  duration; it re-decides every 6 ticks).
- **to whom:** every living non-zombie unit of the team, the same order
  (CHOICE: one team order; per-unit orders are rung 6's), in unit-id order,
  issued ONLY when it differs from the four fields the unit holds now (read
  from the unit) -- so the recording holds a change, not a repeat every 6
  ticks.

It draws nothing, writes nothing but through `order_intent`, and imports no
renderer: the `test_bots.py:529` property applies to it unchanged.

**Note for 1j (not a STOP):** if 1j's door-driven bots are in
`DOOR_CONTROLLERS`, their team gets a second, commander-only brain whose Bound
swap reads `arrived = True`; 1j's note decides whether its controller writes
its own brain's orders instead (and is then left out of the commander rule).

## Gate

1. **Bots alone play and hash as before.** Commit A, before any source
   change, takes the references on main's code to a dump outside the repo
   (`C:/tmp/breach_1f/before.json`, never committed): the
   `tools/bot_tournament.py --json` records (outcome, team, ticks,
   `field_digest`, `tick_digest`) of `bots` vs `bots_v1` on `ship_demo`
   (it has the graph) for seeds 1-4, with and without `--horde`, under
   `--sight cone` and `--sight all_round`. After the patch the same command
   gives the same records, field for field. The 60 1d pins are not a gate
   (step (ii) moved them by design).
2. `tools/match_pin_check.py` EQUAL (CPU; cuda / resident if built here);
   `tests/test_bots_v1.py::test_bots_v1_play_the_pre_g8p_pinned_match`,
   `tests/test_zombie_control.py::test_the_controller_plays_the_pinned_ship_demo_match`
   and `::test_the_controller_plays_the_pinned_match_on_a_small_box` green
   UNCHANGED.
3. `pytest tests -q`, whole, green. Expected deliberate test edits, each
   named in the report: `tests/test_plant_charge.py:360` (`FORMAT_VERSION ==
   7` becomes `>= 7`: its property is the plant's, not the version's);
   `tests/test_match_replay.py::test_every_intent_kind_replays_live`
   (`:217-237`) -- its box gets a room graph through
   `level_lib.write_room_graph` (one room over the floor, no openings) and
   `_ThrowerControl` issues `order_intent` at a few ticks, so "every kind"
   still holds. `tests/test_rl_env.py`'s G-a tests stay green unedited: on
   `ship_demo` both sides (the farm and `_zero_policy_reference`) are built by
   the same `create_match_controllers` and so both carry the commander; the
   field digests equal the idle match (an order is unit state only); the
   boxes have no graph and get no commander.
4. `tools/twin_manifest_gen.py` regenerated with a `[not_twinned]` row for
   `src/simulation/orders.py` ("Class T: the order (#195 1f), applied at slot
   1b', expired at slot 10; information only until rung 4; a device column at
   the residency arc's P6"); `tools/twin_debt.py` and
   `tests/test_class_p_level.py` decide whether `tools/cuda_green.py` is
   needed. Expected not: `simulation.py`, `damage.py`, `ai_zombie.py`,
   `unit_schema.py` are `[not_twinned]` rows (`tests/twin_manifest.toml:81,
   87, 113, 123`) and `unit.py` (a Class-P file, `:476`) is not edited.
5. Report, not gated: `tools/rl_smoke.py` steps per second on `ship_demo`
   before and after (the commander's cost, §9.1 protocol).

**Recommended commit order:** A the reference dump; B schema rows +
`orders.py` + the verb + the slots + the clears, with the property tests (the
gate-1 dump already equal: nothing writes an order yet); C the format bump,
`EXCLUDED_KINDS`, the replay test's box; D the commander, `place_at`,
`DOOR_CONTROLLERS`; E `CLAUDE.md` and the twin manifest. Break one property
on purpose once (expire at `<` instead of `<=`; drop the death clear) and
report that the named test fails.

## Property tests (`tests/test_orders.py`)

On a box with a room graph written through `level_lib.write_room_graph` (two
rooms and one door opening between them), unless named. Each docstring states
the property and what breaks it.

| property | breaks if |
|---|---|
| a fresh unit, and a unit whose order was cleared, add no `order_*` key to `digest_record`; a held order adds `order_what` (and `order_until` / `order_target` only when not their sentinel) | a column hashes unconditionally or compares to the wrong default |
| issued at tick T, the order appears in the unit record after T's step (not before: `handle_frame` + the verb change no unit record and no field digest) | the verb writes the unit directly (the `set_flashlight` shape M4 forbids) |
| `ticks = d`: held after the steps of T .. T+d-1, gone after T+d-1's step; `ticks = 0` still held after 500 ticks | an off-by-one, or expiry at 1b (a zero-time order seen between ticks) |
| a unit killed with an order holds none after that step; a marine killed by a zombie and converted holds none as a zombie | the death or the conversion clear is missing |
| `what = 0` clears; a second `order_intent` for the same unit in one tick replaces the first | a request merges, or a clear needs a place |
| the door refuses by name: a level without a graph, an unknown / dead / zombie unit, `what` 5, a room id past the rooms, an opening past the openings, `ticks` -1 or `True`, a teammate / dead / unknown target, a field set with `what = 0`; a refused call reaches no tap and latches nothing | a door check is missing, or the tap records before the door |
| `place_at`: every floor tile of a room names it; every tile of an opening (grid -1) names the opening; a wall tile names nothing | the commander reads a second map, or the opening lookup is missing |
| ORDERS ARE INFORMATION ONLY: a `ship_demo` bots match with a test controller issuing random legal orders (fixed seed) to every marine every 10 ticks reaches the same result and the same `field_digest` at every checkpoint as the match without, and every unit record bar the `order_*` keys is equal tick for tick | a rule of the game reads an order |
| no module under `src/` but `orders.py`, `damage.py`, `ai_zombie.py`, `unit_schema.py`, `simulation.py` (the two calls) and `control_bots.py` (the commander's read of the held order) names `order_what` / `order_where` / `order_until` / `order_target` (a source scan, the `test_footprint_samples_builder` precedent) | a reader appears before rung 4 without a design line |
| `order_intent` has no action-row column and is in `EXCLUDED_KINDS`; `random_action_rows` never yields one | a soldier's policy could order itself |
| a recorded match with orders replays to every checkpoint; a v7 header is refused by name | the kind is fed back wrong, or the bump is missing |
| on `ship_demo` with team 0 on `policy` (zero rows) and team 1 on `bots`: team 0's living marines hold the commander's order from the first team update; its `what` is `PLAY_ORDER[brain.play]`, its `where` is `place_at` of the objective's centre tile, its target the brain's target id; team 1 holds none; the recording holds an `order_intent` only on a change | the translation, the door-driven test, or issue-on-change is wrong |
| a door-driven team on a level without a graph gets no commander and the match runs; no match of `bots` / `bots_v1` / `idle` / `zombies` teams ever calls `order_intent` (a tap over `ship_demo`, both bot generations, with and without the horde) | the commander rule leaks to a team not on the door |
| `PLAY_ORDER` has a row for every `PLAYS` row (a monkeypatched fourth play is refused at import by name) | a new play silently writes no order |
| gate 1 (the tournament dump) | |

## `CLAUDE.md` (in the patch)

- **New row "Order"** (Gameplay, after "Verb gates"), from the roadmap's
  draft ("An order is game state set through one verb by any seat ... and
  recorded like every verb"): `src/simulation/orders.py` (`ORDER_WHAT`,
  `check`, `apply_requests` at slot 1b', `expire` after the tick advance,
  `clear`) + `Simulation.order_intent` + the four `order_*` schema columns +
  `room_graph.place_at` + `control_bots.BotCommander` (`"commander"`,
  `PLAY_ORDER`) + `control_source.DOOR_CONTROLLERS` -- an order names a place
  on the room graph (a room id, or `ORDER_OPENING_BASE + opening id`), never a
  tile: what (none / go to / hold / clear / fall back to), where, until (a
  tick, or until told otherwise), an optional priority target (an enemy's
  id); set ONLY through `order_intent` (a commander's verb: refused by name at
  the door, out of every policy row, recorded, latched on the sim and applied
  at slot 1b'); cleared at death, at conversion and when its tick comes;
  hashed only while held; INFORMATION ONLY until rung 4 (nothing in the game
  reads it for a decision -- a reader is a design line); the format spec is
  "rooms" (design 5.5). Written by the bots' team layer as the commander ONLY
  for a team whose spec row names a `DOOR_CONTROLLERS` controller on a level
  with a room graph, so a match of bots alone plays and hashes as before. A
  new kind of order is a code of `ORDER_WHAT`; a new commander is a caller of
  the verb, never a writer of the columns.
- **Amend** "Match recording / replay" (the fourteen entry points incl.
  `order_intent`; v8 and its history clause); "MatchSession + the policy
  controller" (`order_intent` is excluded from the columns, as
  `drive_zombies`; the builder adds the commander after the team rows for a
  door-driven team); "Bot layers" (the team layer doubles as the commander,
  `BotCommander`, `PLAY_ORDER`); "Unit" (its order is four columns written
  only by `orders.py`, the Order row); "Room graph" (orders read it through
  `place_at`; the "bots still read the spawn rooms" clause stays); "Match
  controller registry" (`commander`, not nameable). The header line.

## CHOICES in one place

where-code range base 1<<16; integer codes, no enum; `*_intent` name; latch on
the sim (drive_zombies precedent), last call per unit per tick wins; no
possession; `ticks` relative, 0 = no limit; `what = 0` clears and allows no
other field; every `what` takes a room or an opening; target `int` -1;
expiry after the tick advance; the target not cleared at its death; the
door-driven test = spec row's controller in `DOOR_CONTROLLERS`, graph
required; the commander = `BotTeam` subclass running only the team layer
(`arrived` True); objective read at the body's centre tile; no place -> clear;
one team order, `ticks` 0, issued on change; `test_plant_charge.py:360` to
`>= 7`.

## STOP

None. The design (5.7, 9), the plan (§5 "1f") and the critique (M4) agree with
the code as it stands; every seam above exists at the cited line.
