# Rung 1, patch 1j: the bots through the door (the note)

2026-11-13. Written for the orchestrator of rung 1 (#195) from the plan
(`docs/rl_agent_rung1_plan_2026-11-11.md` §1, §2 row 1j, §5 "1j", §6), the
design (`docs/rl_agent_design_2026-11-11.md` 1, 2.1-2.4, 2.7, 8.6, 10), the
roadmap (rung 1's gate, rung 2's imitation experiment), the 1f note ("Note
for 1j"), the 1h note (§8, the inverse helpers) and the 1i note (as built),
checked against the code at main `e6a92a92` (1a-1i and #29 P3b merged).
Same shape as the 1f, 1h and 1i notes. **CHOICE:** marks where the design is
silent and this note decides. **STOP:** would mark a design/code
contradiction; there is none. The design is AGREED: this note chooses seams,
encodings and the measurement's plumbing, never what the game is or what a
soldier may know.

## Open question (for the lead; Erik may move the mark)

1. **The friendly-fire half of the pass mark.** The plan's words: "with no
   more friendly-fire hits per match than the free bots plus a fifth". This
   note reads "plus a fifth" as RELATIVE: `ff_door <= 1.2 * ff_free`, both in
   hits per match. The 1d tournament measured the layered bots' own rate in
   bots-vs-bots duels at **0.0 to 0.2 hits per match**
   (`docs/rl_agent_rung1_1d_pins/*.json`). On so small a base the relative
   reading fails on a handful of stray hits, and fails on ONE hit if the free
   bots score none. The other reading, `ff_door <= ff_free + 0.2` (a fifth of
   a hit per match), is robust at that base. **What I would choose:** judge
   on the relative reading as written, pooled over every held-out duel (both
   ships, horde in and out, so the base is as large as it gets), and print
   the absolute reading beside it. If the two disagree, that is the finding
   the checkpoint reports, and the lead decides before a design session is
   called.

Nothing in this note touches what the game is or what a soldier may know;
there is no question for Erik beyond that mark.

## What the code does today (the facts the build rests on)

- **The layered bots call the sim's verbs directly** (`src/control_bots.py`).
  `BotTeam.handle_frame` (`:907`) runs, once per tick, the team layer
  (`TeamBrain.observe` / `update` every `team_period_ticks` = 6), then per
  living marine `_due` (`:933`: staggered `(tick + 3 id) % unit_period_ticks`
  with `unit_period_ticks` = 8, at once on a new sighting or contact, never
  inside a wind-up), `_decide` (`:948`, the actions' `start` latches) and
  `_execute` (`:1067`, every tick). The verbs, all on the `sim` handed in:
  `set_move_dir` / `clear_move_dir` in `_follow` (`:1361`-`1401`; throttle
  `travel_throttle` out of contact, else 0 -- shipped `travel_throttle = 0.0`,
  so the bots only ever WALK), `set_aim` (`aim` `:1109`, the look in
  `_execute` `:1089`: along the move, or at the nearest known enemy's record
  centre for `retreat` / `stand`; NO call on a tick where the bot stands with
  no threat and no target), `set_trigger` (`shoot` `:1091`: held only when
  `_on_target`, the facing after this tick's turn within the weapon's spread;
  released when a team-mate is on the line), `throw_grenade_intent`
  (`_start_throw` `:725`, once, at a unit decision; frag and flash only; the
  default fuse `fuse_ticks_from_seconds(fuse_default_seconds)`; the reach the
  lob's centre distance), `shove_intent` / `melee_intent` (`try_shove` /
  `try_melee` `:1120`/`:1128`, every tick of a `shove` / `melee` action while
  ready and onto an enemy), `use_intent` (`_open_door` `:1433`, once per
  `door_retry_ticks` = 24 per door). The bots never call `reload_intent`
  (auto-reload is the game's), `plant_charge_intent`, `set_flashlight`,
  `order_intent` or `drive_zombies`.
- **The bots aim with the control seam's one quantizer**: `_direction_q`
  (`control_bots_v1.py:157`, `quantize_stick_direction` of a centre-tile
  offset) -- exactly `action_spec_v1.aim_vector(me, centre)` (`:226`). A shot
  aims at the LIVE centre of a seen unit (`_dir_to`), which is the team
  picture's record centre for that unit at the start of the tick (slot 11
  writes it after the previous step's movement).
- **During a wind-up** the bot holds `st.throw_aim` and releases the trigger
  (`_tick_throw` `:736`) while `u.throw_release_tick >= 0`; `_due` is False.
  A throw the game never started (`throw_release_tick < 0`) leaves the bot
  shooting as usual, its `last_throw_tick` spent (cool-down
  `grenade_cooldown_s` = 4 s).
- **The commander** (1f): `BotCommander(BotTeam)` (`:1537`) runs the team
  layer alone and writes `order_for` (`:1570`) through `_issue` (`:1589`) only
  on change. `create_match_controllers` (`control_source.py:322`) builds it
  for a team whose row names a `DOOR_CONTROLLERS` controller (`{"policy"}`,
  `:284`, whose comment reserves 1j's decision) on a level with a graph.
- **The decoder** (1h, `src/control_policy.py`): `AgentPolicy.configure("v1",
  tables)` (`:344`), `set_decisions(rows5, list_uid)` (`:360`: a pointer
  becomes a game id once; the no-sim fallbacks), `handle_frame` ->
  `_decode_frame` (`:455`: dead or turned rows issue nothing) -> `_decode`
  (`:492`: one raw row per living unit per tick, through `issue_row` (`:258`),
  so `VERB_RULES` (`:84`) decides hold vs latch; the throw's aim held while
  `throw_release_tick >= tick`), `aim_view()` (`:410`). A second
  `set_decisions` sets `_first` again: the latches go out on the next tick.
- **The spec's helpers** (`src/simulation/action_spec_v1.py`): `feet_index`
  (`:385`), `aim_index_for_vector` (`:395`), `aim_index_for_unit` (`:403`),
  `throw_choice` (`:412`), `plant_choice` (`:423`), all over `_nearest_dir`
  (`:369`, 16 bins, `atan2_q16`) and the `ActionTables`; `build_masks`
  (`:267`), `arg_mask`, `masked_choices` (`:351`), `check_actions`,
  `onto_reach_q`, `facing_vector`, `COMPASS_Q16`.
- **The env** (`src/rl_env.py`): `EnvRunner.observe` (`:986`) is, under
  `"v1"`, the accessor's `observe("v1", units, out, sim=, context=, aims=)`
  then `build_masks(..., self.run.decimation, out)`; `aims` is gathered from
  `self.policies` (`isinstance(c, AgentPolicy)`, `:785`). The reward is
  `_count` (`:911`, the OLD counters, `team_kills` among them -- the
  `ff_kill` term's) plus `RewardBooks` `begin` / `end` around each
  `session.tick` (`:858`), grouped by `reward_v1.group_weights`. `_team_slots`
  (`:664`) gives a level's S.
- **The tournament** (`tools/bot_tournament.py`): `play(job)` builds the
  spec with `replace(...)` (controllers, time limit, zombies, sight) and runs
  `run_match(level, spec, seed=, on_tick=watch)`; a record holds outcome,
  team, ticks, wall, sight, the two digests and per team the `TeamLog`
  counts; `summarise` gives `a_rate` (A's wins over decided duels) and
  `ff_per_match` per side. `apply_sets` is the per-process precedent for a
  job-scoped setting. Duels carry the level's horde (match.toml `zombies`);
  `--zombies 0` removes it; `--horde` adds each side against `idle`.
- **The sim does not hold the match spec.** `spawn_match` sets `sim.sight`,
  `sim.rng`, the team picture, but no spec; `ObsContext.build` and
  `RewardBooks` need it (the per-team `n`, `zombies`, `time_limit_s`, which
  the tournament overrides).
- **Levels with a room graph:** in the repo only `levels/ship_demo`
  (`rooms.npy`; S = 6, n = 4 a side, 4 zombies). The second ship is the one
  1d measured on, generated outside the repo:
  `tools/gen_ship.py --seed 28 --size medium --recipe freighter --profile
  standard` (`ship_freighter_standard_s28_medium`; S = 6, n = 4, 4 zombies;
  `gen_ship` writes the room graph on both paths since 1b; the 1d copy in
  `C:/tmp/breach_1d/levels/` carries `rooms.npy`).

## Build

### 1. The controller's shape (`src/control_bots_door.py`, NEW)

`BotsThroughDoor(team, settings)`: ONE match controller that holds an inner
`BotTeam(team)` (the layered bots, unchanged) and an inner
`AgentPolicy(team)` configured `"v1"` (the 1h decoder, unchanged). The bots
decide; the controller captures their verbs instead of letting them reach
the sim; at each decision tick it builds the observation and masks with the
env's own builders, translates the bots' verbs into five integers per slot
with the 1h inverse helpers, and hands them to the decoder, which is then
the ONLY thing that issues soldier verbs to the sim. Not a subclass of
`AgentPolicy` (CHOICE: `EnvRunner` feeds every `AgentPolicy` it finds; this
team must feed itself).

**The interception: `IntentCapture(sim, passthrough=False)`.** A proxy handed
to `BotTeam.handle_frame` in place of the sim. It defines the eleven policy
verbs (`control_policy.POLICY_KINDS`) as capture methods that append `(tick,
kind, args)` -- the args normalised as the sim's tap normalises them -- to a
per-tick list and do nothing else; `order_intent`, `drive_zombies` and
`apply_action` RAISE by name (a bot never calls them). Every other attribute
READ is forwarded to the real sim (`__getattr__`), and every attribute WRITE
too (`__setattr__`: `vision.state_for` caches `_vision_cache` on the sim, so
the tick's one sight cache stays one). `passthrough=True` also re-issues
each captured call to the real sim at once, in order -- the form the
lossless test runs and the form the controller wraps around the decoder to
see what it issued. CHOICE: a proxy rather than an edit of `BotTeam`: the
free bots' code is not touched, and "the bots decide exactly as they do" is
then a test (the pass-through proxy plays bit for bit), not a promise.

**Every tick `T`** (`handle_frame(sim)`, once per tick):

1. If recording rewards: `books.end(sim)` for the step just run (its
   `tick_events` are still on the sim until the next step clears them) and
   the old counters through the lifted `rl_env.count_events` (section 4).
2. **Capture:** `self.bots.handle_frame(capture, None)`. The tick's captures
   per unit: the last `set_move_dir` / `clear_move_dir`, the last `set_aim`
   (or none), the last `set_trigger` (the sim's own last-call-wins for holds);
   every latch is appended to the unit's WINDOW (the latches captured since
   the last decision tick, exclusive, through `T`).
3. **Orders** (section 2): when `bots.brain.updated_at == T`, the team's
   order through the lifted `control_bots.team_order` /
   `issue_team_order`, on the real sim.
4. **On a decision tick** (`T % D == 0`, `D = settings.decision_ticks`,
   default 4 -- the env's own decision ticks, every episode starting at tick
   0, so a record lines up with the farm's decisions): build the observation
   and masks for the team's S rows (`rl_env.observe_v1`, section 4; `aims =
   self.policy.aim_view()`), translate (section 3), check the five
   integers (`check_actions`, and `masked_choices == 0` -- section 6's
   gate, raised here as `DoorRefused` naming tick, slot and head), record the
   decision if a sink records (section 5), `policy.set_decisions(rows5,
   list_uid)`, clear the windows. Under `trigger_per_tick` (a variant) a
   non-decision tick calls `set_decisions` again with the decision's feet
   and aim, THIS tick's captured trigger and hand "nothing", on the stored
   `list_uid` -- the decoder's own path, no latch.
5. **Issue:** `self.policy.handle_frame(issued)`, `issued` an
   `IntentCapture(sim, passthrough=True)`: the decoder's verbs reach the sim
   through the one tap, and the controller sees what was issued.
6. **Diagnostics** (section 3's table): per living unit, whether this tick's
   captured hold (move vector and throttle, aim vector, trigger) differs from
   the issued one.
7. If recording rewards: `books.begin(sim)` for the coming step.

**Between decision ticks** the bots keep running (their paths, staggered
decisions, cool-downs and wind-up holds advance on the game's real state);
their HOLDS are captured and discarded (the decoder holds the decision's,
which is the coarseness 1j measures); their LATCHES wait in the window for
the next decision tick (section 3, the hand). Nothing a bot captures between
decisions ever reaches the sim.

**Binding to the match.** A new no-op hook `ControlSource.bind_match(sim,
spec)`, called once by `create_match_controllers` on every team-row
controller after it is built (`control_source.py`; a no-op for every
existing controller, so nothing else moves). `BotsThroughDoor.bind_match`
refuses, by name, a level whose room graph is missing or not at the map's
shape (the `load_levels` rule: spec v1 reads the graph), then builds once
per match: `S = rl_env._team_slots(level)`'s max (the layout's rule for a
one-level run), the accessor, `ObsContext.build(sim, spec, S)`,
`action_tables(sim, S)`, `policy.configure("v1", tables)`, the uid -> row map.
`handle_frame` on an unbound controller raises, naming `bind_match`.

**The settings and the variants: `DoorSettings`** (frozen): `dir_stride` (1
= 16 directions, 2 = 8), `decision_ticks` (4, 6, 8), `trigger_per_tick`
(False), `throws` (`"nearest"`: onto an entry or by direction, whichever
lands nearer; `"direction"`: by direction and reach only). Registered as
match controllers, NAMEABLE (CHOICE: a match.toml or a run file may name
them, so `main.py --control match` can show Erik the door bots and a farm
can put them on a side; `bind_match` refuses a level without a graph):

| name | settings | design 2.7 |
|---|---|---|
| `bots_door` | 16 directions, every 4 ticks, trigger per decision, throws nearest | the design |
| `bots_door_dirs8` | `dir_stride = 2` | 8 against 16 directions (feet and aim) |
| `bots_door_every6` | `decision_ticks = 6` | a decision every 6 ticks |
| `bots_door_every8` | `decision_ticks = 8` | a decision every 8 ticks |
| `bots_door_trigger_tick` | `trigger_per_tick = True` | the trigger per tick |
| `bots_door_throw_dir` | `throws = "direction"` | throws by direction and reach |

`DOOR_VARIANTS` is that table (name -> settings); the names land in every
tournament record's `controllers`, so a record says which variant played.
No variant has a table of its own: `dir_stride` is a keyword of the 1h
helpers (section 3), the rate is `D`, the trigger variant re-issues through
`set_decisions`, the throw variant restricts the candidate set.

### 2. The commander: the door bots write their own brain's orders

**Decision: the door-driven team is NOT in `DOOR_CONTROLLERS`; its
controller writes ITS OWN brain's orders.** `DOOR_CONTROLLERS` stays
`{"policy"}` (its comment amended: "a controller with no brain of its own;
the door bots carry theirs and write their own orders").

- `BotCommander.order_for` and `_issue` are lifted, verbatim, into two
  module functions of `control_bots`: `team_order(brain, sim, mine, team) ->
  (what, where, until, target)` and `issue_team_order(sim, mine, want)`;
  `BotCommander`'s two methods become one-line calls of them (the commander
  plays and hashes as before: `tests/test_orders.py` green unchanged).
- `BotsThroughDoor` calls them, on the REAL sim, on every tick its inner
  brain updated (`bots.brain.updated_at == T`), with `mine` = the living
  non-zombie units of the team -- the commander's own cadence and rule
  (issue only on change).

**Why.** (a) ONE brain: the order a soldier's observation shows (spec v1's
order columns) is the order of the brain that actually drives the bots, so
an imitation record never pairs a teacher's choice with another brain's
order. (b) The Bound swap's `arrived`: the commander's second brain has no
unit layer, so `arrived` is True for every unit and its halves swap on
`unit_period_ticks` / `bound_max_s` alone, while the bots' own brain swaps
when the moving half has arrived. Today that difference does not reach the
order -- the order is the play, the objective and the target, and none of
them reads the halves or `bound_since` (`TeamBrain.update` `:365`-`425`,
`_objective` `:427`) -- so both choices issue the same orders today (a test
pins it, below). It would reach the order the day the objective or the play
reads the halves, and with (a) the order then stays the driving brain's.
(c) Half the team-layer work (one `update` and one objective flood per team
period instead of two). (d) `DOOR_CONTROLLERS` keeps one meaning: a team
whose driver has no brain gets the bots' brain as its commander.

### 3. The translation (`translate`, a pure function)

Input per living slot: the tick's captures, the unit's window, the slot's
observation rows (`list_uid`, the list's type column), the four masks, the
`ActionTables`, the settings, the unit (its centre tile, `facing_q`). Output:
the five integers and a loss record. **Every number comes from the 1h
tables through the 1h helpers; the translation has no table of its own.**
"Nearest" is always "nearest LEGAL": a choice the masks forbid is never
sent (section 6's gate).

**`stride` on the helpers** (the only edit of `action_spec_v1`, CHOICE):
`_nearest_dir(dx_q, dy_q, stride=1)`, `feet_index(..., stride=1)`,
`aim_index_for_vector(..., stride=1)`: the nearest bin among every
`stride`-th, `k = (a * 8 + stride * (PI_Q16 >> 1)) // (stride * PI_Q16)`,
bin `(k * stride) % 16` -- at `stride = 1` the 1h formula unchanged, bit for
bit. 8 directions are the even compass tokens: the same table, a coarser
chooser.

| head | the bots' verb at `T` | sent |
|---|---|---|
| **feet** | `set_move_dir(dx, dy, thr)` | `feet_index(dx, dy, thr, tables, stride)`: nearest bin by angle, nearest of walk / half / full (the bots walk: throttle 0 = walk, exact) |
| | `clear_move_dir`, or no move captured | 0, stand |
| **aim** | (the masks offer only "keep": a wind-up holds the aim through the decision) | 17, keep (the bot holds its `throw_aim` too) |
| | a vector equal to `aim_vector(me, record.centre)` of an aimable list entry (type 1 or 2, `mask_aim` 1) | that entry, `18 + i` (the first in list order; a second entry on the same bearing is counted ambiguous). This is every shot, every point-blank face of a shove / melee target, and every look at the nearest known enemy: the decoder then tracks the record each tick, as the bot tracks the unit |
| | a vector equal to the captured move vector (the bot looks along its walk) | 16, "where I walk" (the game's own rule: the travel becomes the facing target, tick by tick along the path) |
| | no `set_aim` captured (standing, no threat) | 16 -- a zero aim holds the last target, as the absent call does |
| | any other vector | `aim_index_for_vector(dx, dy, stride)`, the nearest compass token |
| **trigger** | the captured `set_trigger` (none: 0) | it, except 0 on a decision whose hand is a throw (CHOICE: the bot itself releases the trigger from the next tick through the whole wind-up, `_tick_throw`; holding the decision's trigger would fire through its own wind-up) |
| **hand** | the unit's window, newest first | the newest latch whose hand option is legal in `mask_hand` at `T`: `use_intent` -> 4, `melee_intent` -> 2, `shove_intent` -> 3, `reload_intent` -> 1, `plant_charge_intent` -> `plant_choice`, `throw_grenade_intent(kind)` -> `7 + kind`; none legal -> 0, nothing |
| **arg** | a throw (below); a plant -> `plant_choice`'s; anything else | 0 |

**A throw** (the hand chose a captured `throw_grenade_intent`): its POINT is
fixed at capture, `P = centre(capture tick) + (dx, dy) * reach_q` in Q16
tiles (the bots' lob: the point they meant; a latch carried from an earlier
tick keeps its point, re-aimed from where the soldier stands at `T`). The
decision's AIM and ARG are then the legal pair whose landing lies nearest
`P` (squared Q16 distance, Python ints; ties to the first candidate in this
order):

- family A, onto an entry: every legal aimable entry `e` with arg 4, landing
  at `e`'s record centre (the decoder's `onto_reach_q` along the turned
  body) -- skipped under `throws = "direction"`;
- family B, by direction: every legal compass token `d` (every `stride`-th)
  with arg `j` in 0..3, landing at `centre + COMPASS_Q16[d] *
  tables.reaches_q[j]`.

"Where I walk" and "keep" are not throw aims. CHOICE: two clean families
(design 2.7's "throws onto list entries against throws by direction and
reach"), the design run taking whichever lands nearer, the variant only B;
an entry with a fractional reach is a bearing B already covers within half a
bin. The aim of the throw's decision is the THROW's, never the fire
target's: the decoder holds it through the wind-up (1h), so the grenade
leaves the way the bot meant.

**When the nearest choice is masked**, per head: feet and trigger have no
masked option on a living row; the aim's entries are legal exactly when they
are in the team picture, so a bot target missing from the aimable list
(over the cap, or not in the picture) falls to the nearest compass token,
and a wind-up's "keep only" overrides the bot's vector; a hand verb masked
at `T` gives way to the next newest legal latch in the window, else
"nothing"; a throw's argument is chosen inside `arg_mask` by construction.
Each fallback is counted (below).

**Hold against latch.** The bots' holds (move, aim, trigger) become the
three hold heads at the decision tick and are re-issued by the decoder on
every tick of the decision (`VERB_RULES` "hold"); their latches (throw,
use, reload, shove, melee, plant) become the hand head and are issued once,
on the decision's first tick (`VERB_RULES` "latch"). A latch issued by a bot
between two decision ticks waits for the next one (at most `D - 1` ticks);
it is sent there if still legal, once, and the window is emptied either
way. The flashlight is reserved in v1 and the bots never touch it.

**What the bots do that v1 cannot express** -- one line each, each counted
per decision and slot in the record's `loss` bitfield (and summed per match
into the tournament record, section 5):

| # | the bots | v1 | counted as |
|---|---|---|---|
| 1 | walk along any bearing (a path waypoint's centre) | the nearest of 16 (8) bins | `feet_dir`: bin rounded; plus the summed absolute angle error (Q16 radians) |
| 2 | any throttle | walk / half / full | `throttle`: rounded (expected 0: the shipped bots only walk) |
| 3 | change move, aim and trigger on any tick (a path turn, a waypoint reached, on-target trigger discipline, a new target) | held for `D` ticks | `hold_diff_ticks` per head: living-unit ticks whose captured hold differs from the issued one (the decoder's own row, seen through the pass-through capture) |
| 4 | aim at a unit the aimable list does not hold | a compass token, not tracked | `aim_missing` |
| 5 | look at a point that is no entry and not the walk (a non-entry threat point) | a compass token | `aim_rounded` |
| 6 | two entries on one quantized bearing | the first in list order | `aim_ambiguous` |
| 7 | a latch on a tick that is not a decision tick | sent at the next decision tick | `carried`; plus the summed delay in ticks |
| 8 | two latches in one window | the newest legal one | `superseded` |
| 9 | a latch legal when issued, masked at the decision tick | nothing | `hand_masked` |
| 10 | throw onto any point | onto an entry, or a bin at a quarter-step reach | `throw_rounded`; plus the summed landing error (Q16 tiles); and which family won |
| 11 | keep the trigger as it is on the throw's own tick | released for the throw's decision | `trigger_throw` |
| 12 | the default fuse | the spec's one fuse, the same row | none: asserted equal at bind |
| 13 | orders | written by the controller through the commander's verb, outside the action spec | none |

Not a translation loss, and not counted (CHOICE): the bots read more than a
soldier's observation (`adjacent_enemy` scans every unit, the unit layer
reads `sim.units`); a record may therefore hold a choice its observation
does not fully explain. That is the teacher's knowledge, never the agent's;
rung 2's imitation experiment measures it as unexplained choices.

### 4. Two lifts in `src/rl_env.py` (code motion, bit for bit)

So that the records come from the env's builders and never from a copy:

- **`observe_v1(accessor, sim, ctx, units, out, aims, tables,
  decimation)`**: the `"v1"` branch of `EnvRunner.observe` (the accessor's
  `observe("v1", ...)`, then `build_masks`) as a module function;
  `EnvRunner.observe` calls it. The row flags stay the caller's.
- **`count_events(events, team_before, id_row, ci, counters)`**: the body
  of `EnvRunner._count` / `_row` as a module function; `EnvRunner._count`
  calls it.
- **The decoder's view from any controller**: `EnvRunner.observe` gathers
  `aims` from every controller of the session with an `aim_view` (the
  `AgentPolicy` teams as today, and a `BotsThroughDoor`, which returns its
  inner policy's), so a farm with a door team serves that team's cols 28,
  66, 71..76 and `sees_target` as the controller sees them. `self.policies`
  (the teams fed actions) is unchanged.

The 1g / 1h / 1i env gates stay green unedited (`test_rl_env.py`,
`test_obs_spec_v1.py`, `test_action_spec_v1.py`, `test_reward_v1.py`).

### 5. The records (`RecordSink`)

**Where.** `tools/bot_tournament.py --records DIR`, refused inside the
checkout (the `[run] out_dir` rule). Each worker installs, per job, a
process-local `control_bots_door.RecordSink` before `run_match` (the
`apply_sets` precedent: one match at a time per worker); every
`BotsThroughDoor` built in that match registers with it in `bind_match`;
after `run_match` returns, `play()` calls `sink.close(sim, result,
digests)`, which closes each door team's books (`end` for the last step,
`finish(result)`) and writes its file. The tournament ALWAYS installs a sink
for a match with a door team, in stats-only mode without `--records`, so
every tournament record gains `rec["door"]` (section 3's counts per team,
plus decisions, throws sent and `ThrowEvent`s started, and `masked_sent`,
which must be 0). Matches without a door team gain nothing: their records
stay byte-for-byte what they were. The rewards and the books run only when
a sink records (CHOICE: a watched or farm door team pays nothing for them).

**One file per (match, door team):**
`DIR/<level_id>/<c0>-<c1>__z<zombies>__s<seed>__t<team>.npz`
(`np.savez_compressed`), and once per match, beside it, the match's replay
recording `<c0>-<c1>__z<zombies>__s<seed>.replay.gz` (`play()` runs
`run_match(..., record=True)` under `--records` and writes it with
`match_replay.write_recording`; recording changes no sim behaviour).
`zombies` is in the name because the horde-in and horde-out duels share
level, controllers and seed.

**Arrays**, leading axes `(n_dec, S)` (the farm's `team_rows` block: `S`
rows in roster-slot order, empty and dead rows as the env serves them), in
the served dtypes:

| key | shape (S = 6, E = 39, K = 6) | dtype |
|---|---|---|
| `tick` | `(n_dec,)` | int64 |
| `vector`, `list`, `list_mask`, `list_uid`, `near_f`, `near_u`, `far_f`, `far_u` | `obs_spec_v1.shapes(1, S, S)` without the env axis | as served |
| `valid`, `in_match` | `(n_dec, S)` | uint8 |
| `mask_feet`, `mask_aim`, `mask_trigger`, `mask_hand` | `(n_dec, S, 49 / 57 / 2 / 13)` | uint8 |
| `arg_mask` | `(n_dec, S, 5)`: `arg_mask(hand, aim)` realised on the choice | uint8 |
| `actions` | `(n_dec, S, 5)`: the five integers sent | int16 |
| `loss` | `(n_dec, S)`: section 3's bits | uint16 |
| `reward` | `(n_dec, S, 5)`: the five groups over the ticks from this decision to the next (the env's step reward for the same step) | float32 |
| `rows_ended` | `(n_dec, S)` | bool |
| `meta` | 0-d: one JSON string | str |

The full view is not recorded (CHOICE: rung 2's experiment is the soldier's
imitation; a value pre-training would add `full_list` / `full_match` /
`full_watched` through the same door later).

**`meta` -- the tie to the match:** `record_format` (1; the reader refuses
another), `level_id` (`match_replay.level_id`), the level's `[levelgen]`
record when it has one (`recipe_hash`, `level_seed`, `generator_version`),
`seed`, `spec` (`match_replay.spec_to_dict`), both teams' controllers,
`team`, the settings, `S`, `E`, `K`, `hand_names`, the decision ticks `D`,
the observation spec `"v1"`, the action spec `"v1"`, the reward profile
`"fireteam"` and its terms, `build` (`match_replay.build_id()`, the
checkout's `git describe --dirty`) and `bp_build_id` (`breach_physics.
build_id()`, the compiled module's SHA), `rl_env.environment_versions()`,
the result (outcome, team, ticks), the final `field_digest` / `tick_digest`,
the replay file's name and sha256, and the per-match loss totals. The
replay file proves the match (its checkpoints re-simulate it); `tick` ties
each decision to it. A reader `read_record(path)` returns the arrays and the
parsed meta.

**Size.** Per row and decision ~40 KB raw (the list 10.5 KB, the four
window arrays 28.8 KB, the vector, masks and the rest ~0.7 KB). A duel runs
~2 300-4 200 ticks (1d's pins), so ~600-1 000 decisions of ~3 living rows on
average: ~70-120 MB raw per match and door team; empty and dead rows are
zeros, and the list is mostly padding, so expect roughly 10-20 MB
compressed. The design runs below write 600 such files: ~6-12 GB (C: has
~200 GB free). The implementer measures the first file and reports the
projection; above 25 GB, only the duels are recorded (CHOICE).

### 6. Cost

Per door team, per tick: the inner bots as today; two proxies (a list
append per verb); the diagnostics (a few integer compares per unit); with
records, the books (1i: ~0.2 ms a tick). Per decision: the observation of
the team's ~4 living rows (~4 ms: 1g's 7.7 ms per 8 rows), the masks
(~0.1 ms a row), the translation (integer compares; a throw decision ~100
candidates). About 1-1.5 ms per tick against a duel's ~25 ms per tick
(1d's walls: ~60 s over ~2 400 ticks), i.e. ~5 %, plus a few seconds of
compression per recorded match.

## The measurement

All through `tools/bot_tournament.py` (never a second loop), `PY` =
`C:/Users/steen/anaconda3/envs/data/python.exe`, `L` =
`C:/tmp/breach_1j/levels`, `R` = `C:/tmp/breach_1j/records`, `J` =
`C:/tmp/breach_1j` (all outside the repo, never committed).

**Levels:** both ships with a room graph, ship_demo (copied into `L`) and
the medium freighter regenerated at the patch's head:

```
$PY tools/gen_ship.py --seed 28 --size medium --recipe freighter --profile standard --out $L/ship_freighter_standard_s28_medium
cp -r levels/ship_demo $L/ship_demo
```

(check `rooms.npy` beside it; `diff -r` against `C:/tmp/breach_1d/levels/
ship_freighter_standard_s28_medium` should show no difference, or the
report names it).

**Sight:** `all_round` (CHOICE: the rule every later rung trains and is
measured under, 1a; the door bots exist to measure the agent's door, and
the agent plays all-round). Cone is not run.

**Seeds:** tuning `1-20`, held-out `101-130` (G8p's and 1d's sets).

**Sides:** `--a bots_door --b bots`: every duel is played mirrored (A as
team 0 and as team 1), so the pairing's win rate is side-fair.

**Horde:** in (the levels' match.toml, 4 zombies) and out (`--zombies 0`);
`--horde` adds each side against an `idle` team with the horde (survivors
against the horde, a secondary reading).

**The design runs (both seed sets, with records):**

```
for S in 1-20 101-130:
  $PY tools/bot_tournament.py --levels ship_demo,ship_freighter_standard_s28_medium --levels-dir $L \
      --a bots_door --b bots --horde --seeds $S --sight all_round --workers 12 \
      --records $R --json $J/t_door_horde_$S.json
  $PY tools/bot_tournament.py --levels ship_demo,ship_freighter_standard_s28_medium --levels-dir $L \
      --a bots_door --b bots --zombies 0 --seeds $S --sight all_round --workers 12 \
      --records $R --json $J/t_door_nohorde_$S.json
```

Per level and seed: 2 duels with the horde, 4 horde pairings, 2 duels
without: 8 matches; 50 seeds x 2 levels = **800 matches** (600 door-team
record files).

**The variants (tuning seeds, duels with the horde, no records)** -- run
whatever the pass mark says (design 2.7: each number is a measurement), and
again on the held-out seeds if the mark fails:

```
for V in bots_door_dirs8 bots_door_every6 bots_door_every8 bots_door_trigger_tick bots_door_throw_dir:
  $PY tools/bot_tournament.py --levels ship_demo,ship_freighter_standard_s28_medium --levels-dir $L \
      --a $V --b bots --seeds 1-20 --sight all_round --workers 12 --json $J/t_$V.json
```

5 x 20 x 2 levels x 2 sides = **400 matches**.

**Wall time.** 1d's per-match walls on these ships: bots-vs-bots duels
56-78 s, a side against the horde 41-61 s; plus ~5 % for the door. 1 200
matches x ~60 s ~ **20 CPU-hours, about 2 h wall at 12 workers** (16
threads here), the held-out design runs first so the verdict is known
early. Plus the bit-for-bit pins (gates below), ~1 CPU-hour each way.

**The pass mark** (the plan, verbatim: "the door-driven bots win at least
40 % of the decided matches against the free ones, with no more
friendly-fire hits per match than the free bots plus a fifth"). Judged on
the HELD-OUT seeds:

- **win:** `a_rate` >= 0.40 (bots_door's wins over the decided duels) in
  EACH of the four cells (two ships x horde in / out) -- CHOICE: per cell,
  so one ship cannot hide the other;
- **friendly fire:** `ff_per_match` of `bots_door` <= 1.2 x that of `bots`,
  pooled over every held-out duel (the open question at the top; the
  absolute reading printed beside);
- and the gate of section "Gates" 1 (nothing masked sent) on every match.

The checkpoint comment gives per cell and seed set: the win rate of the
decided duels, draws, survivors, rounds per kill, FF hits per match, throws
per match (and started / sent), ticks, wall; the loss table's totals per
decision; and the variants' win rates against the design's. **Below the
mark:** the variants say which coarseness costs (each variant's win rate
and FF against the design run's on the same seeds); the finding goes to a
design session before rung 2. A failure is never a patch to fix: no
`[bots]` weight, helper or rule is tuned to pass.

## Gates

1. **The door issues nothing the game refuses.** Every decision of every
   match: `check_actions` passes and `masked_choices == 0` (raised in the
   controller as `DoorRefused`, so a refusal stops the run naming the tick,
   the slot and the head); no `order_intent` refused at its door;
   `rec["door"]["masked_sent"] == 0` in every tournament record.
2. **The free bots are untouched.** Commit A, before any source change,
   takes the reference on main's code to `C:/tmp/breach_1j/before_<sight>.json`
   (never committed): `tools/bot_tournament.py --levels
   ship_demo,ship_freighter_standard_s28_medium --levels-dir $L --a bots
   --b bots_v1 --horde --seeds 1-4 --sight <cone|all_round> --workers 12
   --json ...`. After the patch the same commands give, for every (level,
   controllers, seed), the same `outcome, team, ticks, field_digest,
   tick_digest`. And `tools/match_pin_check.py` EQUAL;
   `tests/test_bots_v1.py::test_bots_v1_play_the_pre_g8p_pinned_match`,
   `tests/test_zombie_control.py::test_the_controller_plays_the_pinned_ship_demo_match`
   and `tests/test_orders.py` green UNCHANGED.
3. **A door-driven match replays.** `run_match(level, spec(bots_door vs
   bots), seed, record=True)` then `run_match(level, replay=rec)`: every
   checkpoint matched (ship_demo and the freighter, horde in).
4. **`pytest tests -q`** whole and green; `tools/twin_debt.py` and
   `tests/test_class_p_level.py` decide whether `tools/cuda_green.py` is
   needed (expected not: `control_bots.py`, `control_source.py`,
   `control_policy.py`, `rl_env.py` and `action_spec_v1.py` are host-side
   and no Class-P file is touched; `tests/twin_manifest.toml` regenerated only
   if `test_twin_debt` asks).
5. **Can fail:** break three things on purpose, each reverted, and report
   the test that failed: (i) the capture forwards one verb to the sim (the
   tap test fails); (ii) the translator sends the captured latch without
   reading the mask (gate 1 fails on the first masked latch, e.g. a carried
   throw during a wind-up); (iii) the throw's decision keeps the fire
   target's aim (the throw-landing test fails).

## Property tests (`tests/test_bots_door.py`, NEW)

On ship_demo unless named; "the door box" is a two-room level written
through `level_lib` with a room graph and a closed door entity between the
rooms (the 1f box plus a door), where a door bot must open the door to reach
an `idle` team. Each docstring states the property and the change that must
break it. No test pins a count, an order or a set of something designed to
grow.

| property | breaks if |
|---|---|
| the capture is lossless: a `BotTeam` driven through `IntentCapture(passthrough=True)` plays a ship_demo bots-vs-bots match segment (600 ticks) with the same `(field_digest, tick_digest)` on every tick as a plain `BotTeam` | the proxy drops, reorders or alters a verb, or a forwarded read or write diverges (a second sight cache) |
| capture only: with the intent tap attached, no call carries an inner bot's capture -- every soldier verb in the tap equals what the decoder issued (the pass-through capture's list), tick for tick | a verb leaks past the proxy |
| nothing masked is sent: over two seeds of ship_demo and the door box, every decision's five integers are in range and legal (`masked_choices == 0`) | the translator sends a masked option or ignores `arg_mask` |
| feet: the sent direction is the bin nearest the bot's move by `atan2_q16` among every `stride`-th, the throttle the nearest of the three, a cleared move is stand (random vectors and the captured moves of a match) | a second rounding or a second table |
| `stride`: at 1 the helpers return 1h's results bit for bit (1h's tests unchanged); at 2 every result is even and within half a coarse bin of the vector | the stride changes the 16-bin chooser |
| aim at a seen enemy: when the bot shoots at unit X, the sent aim is the entry whose `list_uid` is X, and on every tick of the decision the issued `set_aim` equals `aim_vector` to X's record centre on that tick | unit aims become compass tokens, or a list position is kept |
| aim along the walk: a bot that looks along its move gets "where I walk"; a bot with no `set_aim` gets "where I walk" | the move is rounded to a token instead |
| the throw: on a decision whose hand is a throw, (aim, arg) is the legal pair of the two families whose landing is nearest the bot's point (recomputed exhaustively), the trigger is 0, the kind is the bot's; under `bots_door_throw_dir` only compass aims are sent | the throw's aim is the fire target's, or the argument is not the nearest |
| the wind-up: while a throw winds up past the decision's last tick, the sent aim is "keep" and the grenade leaves along the aim of the decision that threw it | a later decision's aim steers the grenade |
| the latch window: a latch captured between decisions reaches the game on the next decision tick and on no other (the tap), once; one masked at that tick is not sent, its fallback is the next newest legal latch or nothing, and it is counted (`carried`, `superseded`, `hand_masked`) | a carried latch is lost, sent twice, or sent off a decision tick |
| the door box: the door bot opens the door (its `use` latch carried to a decision tick) and walks through | `use` is not translated |
| the cadence: `set_decisions` is called exactly on the ticks `T % D == 0` (D = 4, 6, 8, the three variants); under `bots_door_trigger_tick` the issued trigger equals the bot's captured trigger on every tick and no latch leaves off a decision tick | the rate or the trigger variant is wired wrong |
| the orders: a `bots_door` team gets no `commander` from `create_match_controllers`; its `order_intent` calls equal, tick for tick, those of a shadow `BotCommander` run beside it on a capture proxy (section 2: the two choices agree today) | `bots_door` joins `DOOR_CONTROLLERS`, or the lifted functions drift from the commander's |
| the records are the env's: an in-process `EnvRunner` on a run with `policy = ["bots_door", "idle"]`, `[obs] spec = "v1"`, `[action] spec = "v1"`, `[reward] profile = "fireteam"`, decimation 4: at every step the controller's recorded observation, masks and (one step later) five rewards equal the env's team-0 block, array for array | the controller observes at another moment, assembles its own observation, or computes its own reward |
| the records drive the game: a match whose door team is replaced by a plain `policy` team fed the recorded `actions` and `list_uid` at each recorded tick plays with the same digests on every tick (the policy team gets the commander, so this also holds section 2's agreement) | a record misses something that drove the game |
| determinism: the same match twice writes equal arrays and the same digests; records on or off leave the digests unmoved | hidden state, or the sink writes to the sim |
| the record's tie: `meta` holds the level id, seed, spec, controllers, team, build ids, result, digests and the replay file's sha256; the replay re-simulates to the meta's final digests; `read_record` refuses another `record_format` | a record cannot be tied to its match |
| the refusals: `bots_door` on a level without a room graph is refused by name at `bind_match`; an unbound door controller refuses to run; `--records` inside the checkout is refused | the door runs where spec v1 cannot be built |

Plus, unchanged and green: `tests/test_rl_env.py`, `test_obs_spec_v1.py`,
`test_action_spec_v1.py`, `test_reward_v1.py`, `test_bots.py`,
`test_bots_v1.py`, `test_orders.py`, `test_zombie_control.py`,
`test_match_replay.py`, `tests/_1h_forced.py`.

## The patch cut

**One patch, one Opus implementer** (this worktree, branch
`192-1j-bots-door`), then the tournament runs by the same implementer (or
the orchestrator, if the implementer's budget is spent). Commits, each
green:

| commit | content |
|---|---|
| A | the reference dump (gate 2), on main's code; no source change |
| B | the lifts: `control_bots.team_order` / `issue_team_order` (BotCommander calls them), `rl_env.observe_v1` / `count_events`, the aims from any `aim_view`, `ControlSource.bind_match` + its call in `create_match_controllers`, the `stride` keyword; gate 2 re-run EQUAL here |
| C | `src/control_bots_door.py`: `IntentCapture`, `DoorSettings` / `DOOR_VARIANTS` and their registration, `translate`, `BotsThroughDoor`, `RecordSink` / `read_record`; `tests/test_bots_door.py` |
| D | `tools/bot_tournament.py`: `--records` (with the replay files), the sink per job, `rec["door"]`; the deliberate breaks of gate 5 reported |
| E | `CLAUDE.md`, the as-built section of this note, the twin manifest if asked |
| F | after the tournament: the numbers in this note's as-built section (the JSON stays outside the repo) |

**Size:** ~700 lines of new module, ~600 of tests, ~150 of edits elsewhere.
**Tier:** Opus (the plan's table); the work is integration across five
modules with exact-equality gates, not new arithmetic. **Cost:** the
implementation, ~1 implementer session; the tournament, ~20 CPU-hours
(~2 h wall at 12 workers) plus ~2 CPU-hours for the pins.

**Files:** NEW `src/control_bots_door.py`, `tests/test_bots_door.py`;
`src/control_bots.py` (the two lifts), `src/control_source.py`
(`bind_match`, the registry rows through `register_controller`, the
`DOOR_CONTROLLERS` comment), `src/rl_env.py` (the lifts, the aims),
`src/simulation/action_spec_v1.py` (`stride`), `tools/bot_tournament.py`,
`CLAUDE.md`.

## Systems

**Used, never copied:** the Bot layers (the inner `BotTeam`, unchanged);
the Action spec (the tables, the inverse helpers, `build_masks`,
`arg_mask`, `masked_choices`, `check_actions`, the decoder inside
`AgentPolicy`); the Observation spec (through the sensor accessor's door,
via `rl_env.observe_v1`); the Reward profile (`RewardBooks`,
`group_weights`, the fireteam terms; the env's old counters via
`count_events`); the Team picture (the aim entries' record centres); the
Order (the commander's verb, the lifted commander functions); the Match
controller registry (`register_controller`, `create_match_controllers`);
the Match recording (`run_match(record=True)`, `write_recording`,
`level_id`, `build_id`, `spec_to_dict`); the Bot tournament and the
Headless match runner (the one loop).

**New, with their rules (written into `CLAUDE.md` by the patch):**

- **Bots through the door** (Gameplay, after "Reward profile"):
  `src/control_bots_door.py` (`BotsThroughDoor`, `IntentCapture`,
  `translate`, `DoorSettings` / `DOOR_VARIANTS`, `RecordSink`,
  `read_record`). The layered bots driven through action spec v1: an
  inner `BotTeam` decides on a capture proxy (its verbs never reach the
  sim), an inner `AgentPolicy`'s decoder is the team's one issuer; at the
  env's decision ticks the controller observes and masks through
  `rl_env.observe_v1` and translates each bot decision to the nearest LEGAL
  choice through the 1h helpers -- never a second table; latches between
  decisions wait for the next one; the team writes its own brain's orders
  (not in `DOOR_CONTROLLERS`). A variant is a `DOOR_VARIANTS` row; the
  records are one `.npz` per (match, door team) beside the match's replay,
  in the env's served dtypes, tied by `meta`. Gated by "nothing masked is
  sent", the records equal to the env's, and the records replaying the match
  through a plain `policy` team.
- **`ControlSource.bind_match(sim, spec)`**: the one way a match controller
  learns its match's spec; called by `create_match_controllers`, a no-op by
  default.

**Amended rows:** "Bot layers" (the commander's two functions lifted; the
door bots' brain writes its team's orders), "Order" (the second commander:
the door bots; `DOOR_CONTROLLERS` means "a driver with no brain"), "Match
controller registry" (`bots_door` and its variants, nameable; `bind_match`),
"Action spec" (the inverse helpers' `stride`), "Batched environment"
(`observe_v1`, `count_events`, the decoder's view from any controller),
"Bot tournament" (`--records`, `rec["door"]`), and the header line.
