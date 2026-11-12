# Observation spec v1 and the full view (rung 1, patch 1g)

2026-11-12. The contract for the three implementation patches of 1g (#195),
written from the plan (`docs/rl_agent_rung1_plan_2026-11-11.md` section 5,
"1g") and the design (`docs/rl_agent_design_2026-11-11.md`, "the design",
sections 3, 4, 4a, 5.2, 7, 9, 10), against the code at main `ee25fe98`
(1a to 1f merged). The design is agreed and critiqued and is not re-derived
here; the critique's information lens
(`docs/rl_agent_design_2026-11-11_critique/lens_2_information.md`) is taken as
done. **CHOICE:** marks where the design is silent and the simplest honest
thing was chosen. **STOP:** would mark a design / code contradiction that
cannot be resolved without changing one of them (there is none; section 9).

**v2 (2026-11-12)**, folding in the information critique
(`docs/rl_agent_obs_spec_v1_2026-11-12_critique_information.md`, 0 blockers,
6 major, 12 minor). Everything else of v1 stands. One line per finding:

| finding | resolution |
|---|---|
| MAJOR 1 object `list_uid` is a list index | the team's own object id, `2^20 + k` from a per-team counter at the team's first sighting, kept while the object lives (the object tracker, 1.6; 2.1); honesty test on it (7) |
| MAJOR 2 object `side` read off the live owner | `side` -1 on objects; `whose` by `owner_id` against my id and my team's roster ids, no live read (2.2 cols 8, 56; 0.5 `roster_ids`); pair case (e) (7) |
| MAJOR 3 the fields carry every body | **ACCEPTED GAP** (Erik, 2026-11-12: "on the wind question, its OK! We accept it!"; proposed in v2 as one -- a body really obstructs air and heat), stated in 3.2 with patch (b)'s measurement and in the CLAUDE.md row; the pair tests keep the physics planes out of (a)-(e) (7); register (9) |
| MAJOR 4 the wall is a two-name grep | `TeamKnowledge` (0.5) is the builders' only way in, `ObsWorld` a facade without `units` & co. (a run-time wall); trackers answer in scope; caches keyed by inputs, the full view's separate; the scan widened to every callee module (5.7, 7) |
| MAJOR 5 every team's rows and the counters | a policy of team t reads only team t's rows and nothing of `info` / `reward` / `counters`; what the trainer may read; `BatchedEnv.team_rows` (4.1, 5.7) |
| MAJOR 6 the pair test perturbs nothing read | the counterfactual family (a)-(f), same match.toml and seed, only the observing team's rows compared byte for byte (7); cut across (a), (b), (c) (8) |
| minor 1 damage window's saw rule order | against the records before this tick's sightings; third-side hit in the tracker test (1.6, 7) |
| minor 2 `bars` at first sighting | at the first update the unit exists (spawn) (1.6, 7) |
| minor 3 `aim_of` returns a category | returns `(category, uid)`, tie by squared distance then uid (1.7, 2.2 cols 31, 32, 34, 41, 7) |
| minor 4 object gate's `physical` | `physical=False`, sight (2.1) |
| minor 5 `full_watched` "beside" `watched` | its own key in `info["full_view"]` (5.5) |
| minor 6 no terminal full view | `info["final_full_view"]` (5.2, 7 farm) |
| minor 7 unit `list_uid` carries spawn order / conversion | a uid is an equality key only (2.1) |
| minor 8 the reconstruction test cannot see leaks | said: it is a correctness gate, the family is the honesty gate (7) |
| minor 9 what actually burns it | vector col 96 `felt_temp`, the heat row's `t_felt` over `temperature_max`, through a function extracted from the row (1.3; V = 97; patch (b)) |
| minor 10 a record's facing | `Record.facing_q`, list cols 4, 5 on type 2 (1.6, 2.2) |
| minor 11 records beyond the eighth | the far `mates` / `enemies` / `records` planes drawn from the whole picture (3.7, 7) |
| minor 12 bots know adjacent enemies through walls | a parity gap recorded for the orchestrator (9), not fixed in 1g |

What is fixed here: every feature as a row (position, name, the seam it is
read from, the table value that divides it, the squash, the stored type); the
list's caps, order, padding and mask; the planes of both windows; the layout
in the env's buffers and the two row flags; the value network's full view;
the cost budget; the tests; the cut into three patches. Everything a trained
network reads is a ratio of a table value, an angle's cosine and sine, a flag,
or a kind (a small integer the network embeds); never a game number.

## 0. Conventions

### 0.1 Frame, clock, reference values

- **Frame:** north up, centred on the soldier, the map's own axes (design 1).
  An offset is `(dx, dy)` in tiles from the soldier's centre tile
  (`Unit.center_tile_x/y`, `src/simulation/unit.py:601-607`), x east, y
  south. A facing is `(cos, sin)` of the sim's math-style angle
  (`Unit.facing_q`, Q16.16 radians, 0 = east, counter-clockwise; `unit.py:521`),
  taken through the kit (`unit_fixed.cos_q16 / sin_q16`, divided by 65536):
  the y-down conversion is the reader's, as in `team_picture.fire_lanes`
  (`src/simulation/team_picture.py:251-252`).
- **Decision:** the farm's step (`[run] decimation` ticks; the design's 4). The
  observation is built once per step per soldier at the step's end, from the
  sim as the next tick's controllers will see it (`EnvRunner.observe`,
  `src/rl_env.py:629`, after `MatchSession.tick`).
- **The reference length `L_ref`** (CHOICE): the `[marine] loadout` row's
  weapon `range_tiles` -- the default kit's reach on this map
  (`[marine] loadout` `config.toml:4824` -> `[loadouts.rifleman] weapon`
  `:4909` -> `[weapons.k5_carbine] range_m` `:3297`, derived once per map by
  `WeaponTable` as `range_tiles`, `src/simulation/weapons.py:122-140`; 270
  tiles on the 0.333 m tile). Every offset, distance, route length and blast
  radius is divided by it. One value per run, read once at `Layout` build.
- **The reference speed `v_ref`** (CHOICE): the marine's full speed on open
  floor in Q16 tiles per tick, `(ONE * ONE) // SpeedTable.full_cost_q`
  (`src/simulation/speeds.py:82`, from `[speeds] marine_full_mps`
  `config.toml:109`). Every velocity is divided by it.
- **The reference kit**: the `[marine] loadout` row's weapon: its
  `rof_interval_ticks` (`weapons.py:143`) and `spread_deg` (`:120`) divide
  every unit's own.
- **The compressed distance** `d_c` (design 3.1 "given again on a compressed
  scale"; CHOICE): `ln(1 + d) / ln(1 + L_ref)`, `d` in tiles: 0.12 at one tile,
  0.43 at ten, 0.70 at fifty, 1.0 at `L_ref`; capped by the squash above.
- **Units of the match**: the roster size `n_t` of a marine team is its
  match row's `n` (`MatchTeam.n`, `src/simulation/match.py:92`); the horde's
  `spec.zombies` (`:103`); the time limit `limit_ticks =
  MatchRules.limit_ticks(sim)` (`src/simulation/ruleset.py:331`).

### 0.2 The squash (design 3.1, the critique's one fixed curve)

Every ratio passes `squash` before it is stored; nothing else is applied, no
running statistic exists.

```
squash(x) = x                              for -1 <= x <= 1
          = sign(x) * (1 + ln|x|)          for |x| > 1, capped at +-4
```

Continuous and monotone; the identity on `[-1, 1]`; 2 at `e` times the
table value, 4 at about twenty times, and capped there (three burning tiles
read at most 4). Angles, flags and kinds are not squashed. CHOICE: the
natural log and the cap 4.

### 0.3 Stored types

| what | type | why |
|---|---|---|
| a flag, a plane of flags, a count plane, a code plane | `uint8` | design 3.4 "yes / no planes as single bytes" |
| a ratio plane | `float16` | design 3.4; the squash bounds it in `[-4, 4]`, where float16 keeps three decimals |
| the soldier's vector, a list entry | `float32` | small; a kind column holds its integer exactly |
| an entry's game id, a row flag | `int32`, `uint8` | the decoder's (1h) and the mask's; never a network input |

A kind (class, weapon, grenade kind, entry type, cause, side, order kind)
is stored as its row index in the named table, `-1` = none; the network
embeds it (design 3.1). The tables are named per row below.

### 0.4 Where the observation is built

- `src/simulation/obs_spec_v1.py` (NEW, patch (a)): the spec as code -- the
  feature tables of sections 1 to 4 as data (name, position, reader,
  denominator, squash, type), the builders, and `SPEC_NAME = "v1"`. Pure host
  Python over the sim, the team picture and the trackers; imports no
  controller and nothing from `renderer/`.
- Served through the sensor accessor, the one observation door (plan 5):
  `EntityFieldAccessor.observe(spec, rows, out)` (NEW method beside `crop`,
  `src/simulation/sensor_accessor.py:271`) dispatches by spec name; `crop`
  stays as the placeholder spec `"crop"`. The env calls only the accessor.
- The run file selects it: `[obs] spec = "v1"` (NEW key of `_OBS_KEYS`,
  `src/rl_env.py:235`; default `"crop"`, today's rows unchanged). Under
  `"v1"` the three `"crop"` keys (`channels`, `crop_radius`, `vector`) are
  refused by name, and `load_levels` (`rl_env.py:454`) refuses a level whose
  `LevelData.rooms` is `None` or not at the map's shape
  (`unit_orders.graph_of`, `src/simulation/unit_orders.py:57`): "a run whose
  observation spec reads the room graph refuses a level without one".
- **The trackers** (design 9: memory updated every tick, not unit state,
  not synced, rebuilt by re-simulation) are members of
  `src/simulation/team_picture.py` on the same driver, slot 11
  (`Simulation._step_post_physics`, `src/simulation/simulation.py:2426`; the
  1d note's "Not in 1d"): see 1.6.

### 0.5 The team knowledge view: the observation's ONLY way into the sim (v2; critique MAJOR 4)

The observation builders of sections 1-3 never receive the `Simulation`.
They receive ONE object per team per decision, `TeamKnowledge`, built by ONE
function, `TeamKnowledge.build(sim, picture, team)` in `obs_spec_v1.py` --
the only observation code that touches `sim`, reviewed as the wall and
tested by itself. It exposes exactly:

| member | what | why it is the team's |
|---|---|---|
| `team`, `tick` | the team id, `sim.tick` | public |
| `mates` | my team's LIVING non-zombie units, the live `Unit` objects, by id (me among them) | the radio is perfect (design 4.1) |
| `roster_ids` | the ids of `Simulation.roster(team)`, fixed at spawn (dead and converted slots included) | the team's own muster; the objects' `whose` (2.2 col 56) reads it, never a live owner |
| `records` | `picture.records(team)`, plain `Record` values | the team's memory (1d) |
| `seen` | `{uid: live Unit}` for `uid in visible_by_team[team]` (all of them are in `records` at this tick) -- the only live enemies a builder reaches, for the type-1 columns | seen by a team-mate this tick |
| `seen_by(mate_id, uid)` | `mate_id in vision.state_for(sim).observers_of(uid)`, answered for a mate id only (refused otherwise) | a mate's own sight |
| `objects` | the grenades and charges the team sees now, each `(obj, team_uid)` (2.1, the object tracker of 1.6) | gated by a mate's line of sight |
| `doors` | every living `DoorRuntime` (`sim._doors`) | doors are unfogged (design 3.1) |
| `lanes` | `team_picture.fire_lanes(sim, team, ...)`'s grid, computed here once (it reads only my team's units, `team_picture.py:241`) | my team's state |
| `trackers` | the 1.6 trackers, read through methods that refuse a uid outside their scope: `motion(uid)` for a mate only; `bar(uid)` and `damage_last_s(uid)` for a mate or a `seen` uid only; `last_hit()` for the observing soldier only | each tracker holds every unit; the view hands out only what the team may know |
| `world` | the `ObsWorld` facade (below) | the public world |

`ObsWorld` is a small object with an explicit attribute list and NO
`__getattr__` fallback: `gmap`, `tick`, `weapons_tables`, `speed_table`,
`impulse_table`, the room graph (`unit_orders.graph_of`), `level.wires`, the
config dials the spec names, and the unit-free sim methods the spec calls
with a MATE argument (`unit_turn_rate_q16`, `unit_turn_accel_q16`,
`facing_after_turn_q16`). It is what the builders pass as `sim` to the
sight routines (`vision.ray_clear` / `rays_clear` / `window_clear` read
only `sim.gmap`, `vision.py:195-230`), `line_of_fire`, `route`. It has no
`units`, `roster`, `get_unit`, `projectiles`, `planted_charges`,
`team_picture`, `tick_events`, `vision` cache or `_units_by_id`: a read of
one raises `AttributeError` at run time -- the wall is enforced, not only
grepped.

**Caches** (section 6's order of attack plans several): every cache an
observation builder keeps is keyed by the INPUTS it was computed from (the
team, the source tile, the candidate set as a frozenset of `(uid, centre)`,
the tick), never by a uid alone; the full view (section 5) owns SEPARATE
caches and never reads or fills the observation's. A cache shared between
the two is the bug the pair test (f) of section 7 catches.

## 1. The soldier's vector (design 3.2, 4.4, 5.2; 97 float32)

One row per feature. "Position" is the column index. "Divided by" names the
table value (or says `flag` / `kind` / `angle`). "Squash" is yes unless the
value is a flag, kind, angle or already in `[-1, 1]`. Every unit attribute
named is a column of the unit schema (`src/simulation/unit_schema.py:469-574`)
unless a function is cited.

### 1.1 Body (design 3.2)

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 0 | `hp` | `Unit.current_hp` (`unit_schema.py:503`) | the unit's BAR (1.6: `bars[uid]`, its health at spawn or at turning; `stats.effective_vitality` for a marine, `src/simulation/stats.py:55`) | yes |
| 1, 2 | `facing_cos`, `facing_sin` | `cos_q16 / sin_q16(Unit.facing_q)` / 65536 (`unit.py:521`) | angle | no |
| 3 | `omega` | `Unit.omega_q` (`:497`) | the turning law's top speed `Simulation.unit_turn_rate_q16(u)` (`simulation.py:1053`, `[marine] turn_rate_deg_per_s` `config.toml:4810`) | yes (an outside spin may exceed it) |
| 4 | `turn_left` | `turning.wrap_q16(Unit.facing_target_q - facing_q)` (`src/simulation/turning.py:76`, `unit_schema.py:495`) | `turning.PI_Q16` (`:62`): signed, counter-clockwise positive | no |
| 5, 6 | `vel_x`, `vel_y` | the motion tracker's one-tick displacement `motion[uid]` (1.6), Q16 tiles per tick | `v_ref` | yes |
| 7, 8 | `pushed_x`, `pushed_y` | `Unit.pvx`, `pvy` (`unit_schema.py:511-513`) | the fall limit `ImpulseTable.v_fall_q[0]` (`src/simulation/unit_impulse.py:201`, `[impulse] v_fall_fwd_mps` `config.toml:4862`) | yes |
| 9 | `effort` | `Unit.effort` | `unit_fixed.FP_ONE` (the hold; `EFFORT_MAX` is 2, `unit_impulse.py:71`) | yes |
| 10 | `react` | `Unit.react_ticks` (catching itself: `unit_impulse.walk_allowed`, `:766`) | `ImpulseTable.react_ticks` (`:202`, `[impulse] t_react_seconds` `:4865`) | no |
| 11 | `prone` | `status.composed_flags(u).is_prone` (`src/simulation/status.py:214`) | flag | no |
| 12, 13, 14 | `can_move`, `can_act`, `can_aim` | `composed_flags(u)` | flag | no |
| 15..23 | `status_left[k]`, k = 0..8 in `STATUS_REGISTRY` order (knocked_down, immobilized, stunned, paralyzed, burning, poisoned, suffocating, regen, blinded; `status.py:80-88`) | the largest `remaining_ticks` among the unit's statuses of kind k (`StatusEffect`, `:169`), 0 if none | `STATUS_SCALE_TICKS[k]` (1.5) | yes |

### 1.2 Weapon and kit (design 3.2)

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 24 | `rounds` | `combat.mag_status(u, weapon, tick)[0]` (`src/simulation/combat.py:836`); 1.0 when the weapon's `mag_size` is 0 | `WeaponDef.mag_size` (`weapons.py:144`) | no |
| 25 | `mags` | `mag_status(...)[1]`; 1.0 when `None` (untracked) | `LoadoutDef.magazines` (`src/simulation/loadouts.py:103`; the loadout's own count, 1 when 0) | yes |
| 26 | `ammo_untracked` | `mag_size == 0` or spares `None` | flag | no |
| 27 | `reloading` | `mag_status(...)[2]` | flag | no |
| 28 | `reload_left` | `Unit.reload_done_tick - tick` (`unit_schema.py:518`), 0 when not reloading | `WeaponDef.reload_ticks` (`:146`) | no |
| 29..40 | `grenade[k]`, `grenade_kind_ok[k]`, k over the hand-grenade rows (`WeaponsTables.hand_grenade_rows`, config order, `weapons.py:1038`; 6 rows today) | `Unit.grenade_counts[k]` (`:520`); the flag = the loadout names that row (`LoadoutDef.grenades`, `loadouts.py:99`) | the loadout's count of that kind (a kind the loadout lacks: value 0, flag 0 -- the mask, never 0/0) | no |
| 41 | `throw_windup` | `Unit.throw_release_tick >= 0` (`:523`) | flag | no |
| 42 | `throw_left` | `throw_release_tick - tick` | `[weapons.hand_grenade] throw_windup_ticks` (`weapons.py:174`, `config.toml:3320`) | no |
| 43 | `throw_kind` | `Unit.throw_kind` (`:524`) while winding up, else -1 | kind (hand-grenade row) | no |
| 44 | `charges` | `Unit.has_explosive` (`:522`) | `LoadoutDef.explosives` (`:100`; 0 -> value 0) | no |
| 45 | `has_charge_kind` | the loadout's `charge` names a row (`:113`) | flag | no |
| 46 | `planting` | `Unit.plant_done_tick >= 0` (`:527`) | flag | no |
| 47 | `plant_left` | `plant_done_tick - tick` | `[weapons.c4_satchel] plant_ticks` (`weapons.py:176`, `config.toml:3373`; the `PLANT_WEAPON` row, `weapons.py:77`) | no |
| 48 | `plant_timer` | `Unit.plant_fuse_ticks` (`:528`) while planting | `fuse_ticks_from_seconds(c4.fuse_max_seconds)` (`weapons.py:80`, `config.toml:3375`) | no |
| 49 | `melee_cool` | `max(0, last_blow_tick + melee.rof_interval_ticks - tick)` (`:531`; 0 without a melee row) | the melee row's `rof_interval_ticks` (`LoadoutDef.melee` -> `WeaponDef`) | no |
| 50 | `has_melee` | `Unit.melee_id != ""` (`:482`) | flag | no |
| 51 | `shove_cool` | `max(0, last_shove_tick + shove_cooldown_ticks - tick)` (`:532`; `unit_impulse.shove_cooled`, `:889`) | `ImpulseTable.shove_cooldown_ticks` (`:212`, `[impulse.shove] cooldown_seconds` `:4894`) | no |
| 52 | `class` | `Unit.loadout_name` as its `[loadouts.*]` row index (`unit_schema.table_row("loadouts", ...)`, `:178`) | kind | no |
| 53 | `weapon` | `Unit.weapon_id` as its `[weapons.*]` row index | kind | no |
| 54 | `range` | the weapon's `range_tiles` | `L_ref` | yes |
| 55 | `rof` | the weapon's `rof_interval_ticks` | the reference kit's | yes |
| 56 | `spread` | the weapon's `spread_deg` | the reference kit's | yes |
| 57 | `trigger` | `Unit.live_trigger` (`:565`) | flag | no |

### 1.3 The air where it stands (design 3.2, 3.4)

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 58..65 | `air[p]`, p over the eight physics planes of 3.2 in their order (temperature, fire, poison, teargas, smoke, pressure, wind_x, wind_y) | the near window's plane value at the soldier's centre tile (built by the same plane builders, 3.2; the unit's own profile for the per-unit denominators) | as 3.2 | as 3.2 |
| 96 | `felt_temp` (v2; critique minor 9; APPENDED so no column moves) | the heat row's own felt temperature for this body, `t_felt` (`exchange.py:358-397`: the max over the footprint of the `heat` buffer and the radiation sensor `rad_flux_sweep`, through the row's absorption) -- the one law extracted as a read-only function `exchange.felt_temperature(gmap, u, ...)` that the heat row calls, never a copy (bit-identical: `match_pin_check.py` EQUAL); the row's `ambient_ref` when no flux reaches the body | `profile.temperature_max` (the row's own edge: 1.0 = it starts to burn) | yes |

`air[0]` is the TILE's gas temperature; a soldier cooked by a fire's
radiation across a room reads a cool tile there and its burning body in
`felt_temp`. Both are its own body's reads: no honesty question (but see the
fields' row in 3.2).

### 1.4 What just happened to it (design 3.2; the critique's minor 6)

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 66 | `hit_recent` | the last-hit tracker (1.6) holds a hit within the memory time | flag | no |
| 67 | `hit_age` | `tick - last_hit[uid].tick`; 1.0 when none | `team_picture.memory_ticks` (`team_picture.py:80`, `[team_picture] memory_s` `config.toml:5001`) | no (capped at 1) |
| 68, 69 | `hit_from_cos`, `hit_from_sin` | minus the hit's travel vector `(dir_x_q, dir_y_q)` of its `UnitHitEvent` (`src/simulation/events.py:304-305`), normalised, y negated into the facing convention; `(0, 0)` when the event carried none (heat, gas, status) | angle | no |
| 70 | `hit_cause` | the event's `source` as its `events.CAUSE_NAMES` index (`:89`); -1 none | kind | no |

### 1.5 Previous choices, the clock, the team, the order (design 3.2, 4.4, 5.2)

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 71..76 | `prev[h]`, h = feet direction, feet speed, aim, trigger, hand, hand argument | the previous decision's chosen option index per head; -1 until 1h's decoder writes them (1g reserves the six columns and writes -1; the previous aim entry is marked on the list, 2.2 col 66, never a position) | kind | no |
| 77 | `time_left` | `limit_ticks - tick` | `limit_ticks` | no |
| 78 | `mates_alive` | living non-zombie units of my team (me included) | `n_t` of my team | no |
| 79 | `team_ammo` | the team's rounds: for each living mate `rounds + spares * mag_size` (`TeamBrain.update`'s count, `src/control_bots.py:381-387`; untracked spares count as `magazines` full ones) | `n_t * LoadoutDef.magazines * mag_size` of the team's loadout | yes |
| 80 | `enemies_known` | marine records of my team's picture (`TeamPicture.known(team, zombies=False)`, `team_picture.py:184`) | the other marine teams' `n_t` summed | no |
| 81 | `enemies_seen` | those records whose uid is in `vision.state_for(sim).visible_by_team[team]` (`src/simulation/vision.py:387`, `:261`) | the same | no |
| 82 | `zombies_known` | zombie records (`known(team, zombies=True)`) | `spec.zombies` (0 -> value 0) | yes (converted marines may exceed it) |
| 83 | `zombies_near` | zombie records within `[bots] zombie_near_tiles` of a living mate (the bots' count, `control_bots.py:377-380`, `config.toml:4962`) | `spec.zombies` | yes |
| 84 | `order_what` | `Unit.order_what` (`unit_schema.py:552`; `unit_orders.ORDER_WHAT`, `unit_orders.py:34`) | kind | no |
| 85 | `order_has_limit` | `Unit.order_until != NO_LIMIT` (`:556`, `unit_orders.py:47`) | flag | no |
| 86 | `order_left` | `order_until - tick` (0 without a limit) | `limit_ticks` (CHOICE: an order's own length is not stored, only its end; the match's clock is the one table duration) | no |
| 87 | `inside` | `room_graph.place_at(graph, cy, cx) == order_where` (`src/simulation/room_graph.py:170`) | flag | no |
| 88 | `place_kind` | 0 none, 1 a room (`order_where < ORDER_OPENING_BASE`, `unit_orders.py:44`), 2 an opening | kind | no |
| 89, 90 | `place_dx`, `place_dy` | the place's centre tile minus mine (a room: `RoomRecord` rectangle centre `(x + w // 2, y + h // 2)`, `level_loader.py:1618`; an opening: `OpeningRecord`'s, `:1633`) | `L_ref` | yes |
| 91 | `place_dist_c` | the same offset's length | `d_c` | no |
| 92, 93 | `compass_dx`, `compass_dy` | the first step of the shortest walkable route (the compass, 1.7): a unit vector of the four directions, `(0, 0)` with no route or no order | angle | no |
| 94 | `route_len` | the route's steps from my anchor (1.7); -1 when unreached or no order | `L_ref` | yes |
| 95 | `has_target` | `Unit.order_target != NO_TARGET` (`:558`) | flag | no |

### 1.6 The trackers (NEW, patch (a); members of `team_picture.TeamPicture`, updated in `update` at slot 11)

Design 9 puts "the direction and age of a unit's last hit; damage over the
last second" in the Team picture; the 1d note left them to 1g. CHOICE: the
motion and the bars join them, on the same driver, so no reader ever
computes memory on demand. All are host-side, never synced or hashed.

| tracker | holds | written | read by |
|---|---|---|---|
| `motion[uid]` | `(pos_q_prev, pos_q_now)`, Q16 tiles (`unit_fixed.quantize_scalar(u.x)`, as `Record.pos_q`); the velocity is their difference, `(0, 0)` on the first update | every living unit, every update | 1.1 `vel`, the mate entries' `vel` -- through `TeamKnowledge.trackers.motion(uid)`, which refuses a non-mate (0.5); an enemy entry's velocity is ALWAYS its record's `vel_q` (`team_picture.py:103`), never this tracker |
| `last_hit[uid]` | `(tick, dir_x_q, dir_y_q, cause)` of the last `UnitHitEvent` with `damage > 0` on `uid` in the step's `sim.tick_events` | at the update, events in emission order (the last wins) | 1.4, the observing soldier's own only |
| `damage_window[team][uid]` | a deque of `(tick, damage)` of hits on `uid` the team DEALT (the event's `attacker_id` on `team`) or SAW (the victim's record has `tick == now - 1` as the records stood BEFORE this tick's sightings -- the seen-death rule's own moment, `team_picture.py:157-162`; v2, critique minor 1: the window is updated before step (3) rewrites a still-seen record to `tick == now`); entries older than `WINDOW_TICKS = ticks_per_second` dropped (CHOICE: the design's "last second", quantized once) | at the update, before the sightings | the enemy entries' `dmg_last_s` (2.2), for a `seen` uid only |
| `bars[uid]` | the unit's `current_hp` at the first update at which the unit EXISTS in `sim.units` (its spawn; v2, critique minor 2: not its first sighting, so an enemy first seen wounded reads its true ratio) and again at the update that first sees `is_zombie` true for it (the conversion ran in `on_tick_end` before slot 11, `simulation.py:2419-2426`; design 6.1 "health at spawn or at turning") | at the update | every `hp` ratio and `dmg_last_s` (a mate or a `seen` uid only, 0.5); patch 1i's bar tracker reads this one (CHOICE: 1g builds it because the hp ratio needs it first; 1i's note takes it over) |
| `objects[team]` (v2; critique MAJOR 1) | per team, `{obj: team_uid}` for every live grenade and charge the team has seen (2.1's gate, now evaluated HERE every tick, one `rays_clear` per team), the `team_uid` taken from a per-team counter at the team's FIRST sighting of that object (`2^20 + k` for the team's k-th seen object of the episode) and kept while the object lives; an object that detonated or was spent is dropped; plus `seen_now[team]`, the objects the gate passes this tick | at the update, objects in `sim.projectiles` then `sim.planted_charges` order (the counter's order among objects first seen on one tick) | 2.1's type-3 entries and their `list_uid`, through `TeamKnowledge.objects` |

The `Record` (1d) gains ONE field, `facing_q` (`Unit.facing_q` at the
sighting; v2, critique minor 10), written by `_sighting`
(`team_picture.py:107`): the team saw which way the enemy faced.

### 1.7 New read-only functions (design 3.7 names them; none exists today)

| function | where | what |
|---|---|---|
| `turning.ticks_to_within(unit, target_q, a, w, dw, tol_q, cap)` | NEW in `src/simulation/turning.py` beside `predict` (`:199`) | the ticks until `abs(wrap(target - facing)) <= tol_q`, by iterating `law_q16` (`:140`) from the unit's facing and `omega_q + dw` -- never a copy of the law; `cap` ticks at most (returns `cap` when not reached). The trigger aid of 2.2. |
| `GameMap.wind_load_plane()` | NEW in `src/simulation/gamemap.py` beside `footprint_wind` (`:2405`) | the per-tile term of that read, `W_i = N_raw * ((wind_x^2 + wind_y^2) >> 16)` over gas tiles (0 on solid), with `(wind_x, wind_y)` -- as float64 arrays (a read-side product like the crop, `sensor_accessor.py:123`; the int product overflows int64). The wind planes of 2.2. |
| `line_of_fire.solid_in_way(sim, a, b)` and `mate_in_way(a, b, mates, clear_tiles, fp)` | NEW module `src/simulation/line_of_fire.py` | as a round flies (the critique's minor 7): `solid_in_way` = `not vision.rays_clear(sim, [(ay, ax, by, bx)], physical=True)[0]` (`vision.py:195`) or a live cover entity's `blocks_segment` (every live cover, as the march stops on any, `combat.py:381-387`); `mate_in_way` = the bots' geometry verbatim (`control_bots.BotTeam._teammate_on_line`, `:1199-1223`, with `[bots] line_clear_tiles` `config.toml:4959`), a module function; held equal to the bots' method by a test; the bots are NOT moved onto it in 1g (a one-line finding). |
| `obs_spec_v1.aim_of(know, u)` | NEW in `obs_spec_v1.py` | the 4.1 rule judged by MY team's knowledge (`know`, the 0.5 view): among the units my team knows (`know.mates` at their live centres; `know.records` at their recorded `centre`), the nearest one, other than `u`, whose bearing from `u` is within `u`'s cone of `u`'s facing ray and to which `vision.ray_clear` from `u`'s centre tile holds -- nearest by squared centre distance, ties by the lower uid; answers `(category, uid)` (v2, critique minor 3): category 0 nobody, 1 me, 2 a team-mate, 3 a known unit of another side, and the uid of that mate or record (-1 for nobody) -- never the uid of a unit the team does not know. The facing ray is `facing_q` (for a team-mate `facing_target_q` instead, design 4.1); the cone is the unit's weapon `spread_deg` (its `melee_arc_half_q`, `weapons.py:213`, when it has no ranged weapon; CHOICE for a zombie: the reference kit's melee row `[weapons.rifle_butt] melee_arc_half_deg` `config.toml:3494`, the one table cone a weaponless body gets). An unknown unit never answers. |
| `obs_spec_v1.route(sim, graph, where, fp)` | NEW | the compass's field (1.5, 2.3): `horde.flood(gmap, tile_ok, fp, seeds)` (`src/simulation/horde.py:260`) with `tile_ok = horde.walkable_tiles(gmap)` (`:274`) OR the tiles of every opening whose `OpeningState.openable` (`room_graph.opening_states`, `:81`) -- a door the soldier could open is on the route -- and `seeds` = every anchor `a` with `horde.block_passable(tile_ok, fp)[a]` (`:219`) whose block's centre tile has `place_at == where`. Cached per `(where, fp)` on `gmap._tile_serial` (`gamemap.py:829`) and the doors' states (the openings' cache key). The first step from the soldier's anchor is the first entry of `horde.candidates(field, ay, ax)` (`:431`, `DESCENT_ORDER` ties, `:87`); the length is `field[ay, ax]` (`UNREACHED` -> -1). |
| `verb_gates.door_in_reach(u, door)` | NEW in `src/simulation/verb_gates.py`; `use_door` (`:85`) becomes the first live door for which it holds | the use verb's own geometry per door (the span touches the footprint or its one-tile ring), for the door entries' `in_reach`. |
| `STATUS_SCALE_TICKS[k]` | NEW in `obs_spec_v1.py`, built once from the tables | per status kind the longest shipped applier's duration in ticks, at least 1: knocked_down `[exchange] knockdown_getup_ticks` (`config.toml:5174`; `ImpulseTable.getup_ticks`); blinded `max(teargas_blind_ticks, max over payload rows of blind_ticks)` (`:5190`, `:4453`; `PayloadDef.blind_ticks` `weapons.py:337`); stunned / immobilized / paralyzed the largest `WeaponDef.status_ticks` among rows whose `status_kind` is that kind (`:205`); burning / poisoned / suffocating / regen have no shipped applier: 1 (a value that is always 0 divided by 1). A new applier joins the max; no game number enters the interface. |

## 2. The list (design 3.3, 4.2, 4.3, 4a, 5.2)

### 2.1 Entry types, caps, order, padding, mask

| block | type code | cap | who is in it | honesty rule |
|---|---|---|---|---|
| team-mates | 0 | `S - 1` (`S = Layout.n_slots`, the largest roster of the run, `rl_env.py:398`; the design's "all") | every living non-zombie unit of my team but me | the radio is perfect (design 4.1) |
| enemies seen now | 1 | 16 | the picture's records whose uid is in `visible_by_team[team]` this tick (marines and zombies) | seen by at least one team-mate this tick |
| last-seen records | 2 | 8 | the picture's other records (`TeamPicture.records(team)`, `team_picture.py:173`) | the team's own memory; a record outlives an unseen death (1d step ii) |
| grenades and charges | 3 | 6 | every live `Projectile` of `sim.projectiles` not `detonated` (`combat.py:94-124`, `simulation.py:381`) and every live `PlantedCharge` of `sim.planted_charges` (`src/simulation/charges.py:35-52`, `simulation.py:508`) whose tile some living unit of my team (me included) has `vision.rays_clear(..., physical=False)` to -- SIGHT, the units' own rule (`sight_screen` and `blocks_los` cover stop it, `vision.py:304-316`; v2, critique minor 4: never `physical=True`, which is 1.7's line of fire and would show a grenade through a grille no unit is seen through) -- one call per team per TICK over all (mate centre, object tile) pairs, in the object tracker (1.6); the builder reads `TeamKnowledge.objects` | a team-mate has a line of sight to its tile (design 3.3; an enemy's timer is shown, Erik's ruling) |
| doors | 4 | 4 | every living `DoorRuntime` of `sim._doors` (`src/simulation/door_system.py:123`; `alive`, `state`, `span`, `ordinal` as `room_graph.py:91-102` reads them) | doors are shown unfogged (design 3.1) |
| reserved | 5 downed team-mate, 6 bag, 7 button, 8 noise heard | 0 | nothing in v1 (design 3.3) | |

- **Order:** the blocks in the table's order, fixed widths `[S-1, 16, 8, 6,
  4]`, total `E = S + 33` entries (36 at S = 4). Within a block NEAREST
  FIRST by the squared centre distance (a door: its span's centre tile),
  ties by the lower id (a door: its ordinal); re-sorted at every decision
  (design 3.2). A block with more candidates than its cap keeps the nearest.
- **Padding and mask:** an empty slot is all zeros, its `mask` 0 and its
  `uid` -1; the network never reads an unmasked slot as a thing (design
  3.1). `uid` (int32) is the game's unit id for types 0-2, the TEAM's own
  object id for type 3 (`2^20 + k`, the object tracker's counter of 1.6: a
  function of the objects this team has seen, stable for as long as the
  object lives, so 1h may hold it across decisions; v2, critique MAJOR 1 --
  v1's index into `sim.projectiles` / `sim.planted_charges` counted the
  unseen objects ahead of it and moved when one of them detonated), the
  door's ordinal plus `2^22` for type 4 (doors are public) -- for the
  decoder and the aim mask (1h) only; never a feature.
- **What a `uid` may be used for** (v2, critique minor 7): an EQUALITY key
  only -- 1h's decoder matches a held uid against the list's uids and the
  masks key on it; nothing reads a uid's VALUE (a unit id carries spawn
  order, and a conversion keeps it, so a seen zombie's uid equals the
  marine it was). A network input built from a uid's value is the bug.
- **The previous aim mark** (col 66) is written by 1h's decoder (the held
  uid); 0 in 1g.

### 2.2 The entry (67 float32 columns; a column not applicable to the type is 0)

Common columns (every type):

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 0 | `type` | the block | kind (0..8) | no |
| 1, 2 | `dx`, `dy` | the entry's centre tile minus mine (a record: `Record.centre`, `team_picture.py:101`; a projectile: `int(fx), int(fy)`; a charge: `(x, y)`; a door: its span's centre) | `L_ref` | yes |
| 3 | `dist_c` | that offset's length | `d_c` | no |
| 4, 5 | `cos`, `sin` | a unit's `facing_q` (a seen unit: live; a record: `Record.facing_q`, its facing when last seen -- v2, critique minor 10, the field added in 1.6; objects: 0, 0) | angle | no |
| 6, 7 | `vel_x`, `vel_y` | a mate: `motion` (1.6); a seen enemy / a record: `Record.vel_q` (`:103`); a grenade in flight: `(target - start) / dist * travel_speed / tps` tiles per tick (`Projectile.update_position`'s own law, `combat.py:139-157`), `(0, 0)` once landed; a charge, a door: 0 | `v_ref` | yes |
| 8 | `side` | the unit's team as seen (`Record.team`, a mate's `Unit.team`; the horde reads `factions.ZOMBIE_TEAM`); objects and doors: -1 (v2, critique MAJOR 2: an object's owner is told only by `whose`, col 56, never by a live team read -- v1's owner side flipped on an unseen conversion and named which team threw an unseen thrower's grenade) | kind (team id; 255 the horde) | no |
| 9 | `class` | `Record.loadout` / `Unit.loadout_name` as its `[loadouts.*]` row index; a zombie: `len(loadouts)` (one code past the rows: the zombie class); objects, doors: -1 | kind | no |

Unit columns (types 0, 1; a record (2) shows only what it holds: cols 0-9, 54). A
type-1 column reads the live unit only through `TeamKnowledge.seen` (0.5);
a type-0 column through `TeamKnowledge.mates`:

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 10 | `hp` | `Unit.current_hp` (a seen enemy: the live unit, "what the bots read", `focus_key` `control_bots.py:488`) | `bars[uid]` | yes |
| 11 | `exposure` | `Unit.exposure[KINETIC]` (`unit_schema.py:508`, `damage.KINETIC` `src/simulation/damage.py:53`; the level the bots read) | `ExposureSettings.x_max_q` (`src/simulation/exposure.py:141`, `[damage.exposure] x_max` `config.toml:4709`) | no |
| 12 | `firing` | `Unit.live_trigger` (design 4.1: the held trigger, never the shot events) | flag | no |
| 13, 14 | `reloading`, `reload_left` | as 1.2 cols 27, 28 | `reload_ticks` | no |
| 15, 16, 17 | `winding`, `throw_left`, `throw_kind` | as 1.2 cols 41-43 | `throw_windup_ticks`; kind | no |
| 18, 19, 20 | `planting`, `plant_left`, `plant_timer` | as 1.2 cols 46-48 (design 4.2: a plant is unit state) | `plant_ticks`; the c4 `fuse_max` | no |
| 21 | `stunned` | `not composed_flags(u).can_act` | flag | no |
| 22 | `blinded` | `not composed_flags(u).can_aim` | flag | no |
| 23 | `down` | `composed_flags(u).is_prone` | flag | no |

"For me" (type 1; design 3.3):

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 24 | `seen_by_me` | `TeamKnowledge.seen_by(my id, uid)` (`vision.state_for(sim).observers_of(uid)`, `vision.py:266`, answered for a mate only, 0.5) | flag | no |
| 25 | `lof_solid` | `line_of_fire.solid_in_way(sim, my centre, its centre)` (1.7) | flag (1 = something solid in the way) | no |
| 26 | `lof_mate` | `line_of_fire.mate_in_way(my centre, its centre, my living mates, line_clear_tiles, fp)` | flag | no |
| 27 | `ticks_to_on` | `turning.ticks_to_within(me, atan2_q16(-dy_q, dx_q) of the offset to its centre, a, w, unit_impulse.pending_dw(table, me), spread_q, cap)` (1.7; `a`, `w` = `Simulation.unit_turn_accel_q16 / unit_turn_rate_q16`, `simulation.py:1053-1065`; `spread_q = quantize(radians(weapon.spread_deg))`, the bots' `_weapon_spread_q` `control_bots.py:1188`; `cap = 4 * HALF_TURN_TICKS`) | `HALF_TURN_TICKS = ceil(PI_Q16 / w)` (the half turn at top speed; 12 ticks at the shipped dials) | yes (reads 4 at the cap) |
| 28 | `on_next_tick` | for the entry that is my current aim target only (1h's held uid; 0 in 1g): `abs(wrap(want - Simulation.facing_after_turn_q16(me, dx_q, dy_q))) <= spread_q` (`simulation.py:1067`; the bots' `_on_target`, `control_bots.py:1180-1186`) | flag | no |
| 29 | `in_range` | squared centre distance `<= range_tiles^2` of my weapon | flag | no |
| 30 | `in_throw_range` | `<= max_throw_range^2` (`[weapons.hand_grenade] max_throw_range` `config.toml:3312`) | flag | no |
| 31 | `facing_on_me` | `aim_of(know, it)`'s category `== 1` | flag | no |
| 32 | `aims_at` | `aim_of(know, it)`'s category (1.7; design 4.1 / 4a) | kind (0 nobody, 1 me, 2 a team-mate, 3 a known unit of another side) | no |
| 33 | `zombie_state` | a seen zombie: 1 dormant, 2 hunting (`Unit.zombie_activated`, `unit_schema.py:541`), 3 walking to a noise (`zombie_noise_ticks > 0`, `:545`; `ai_zombie.py:346-369`); a marine or a record: 0 (design 4a; the critique's minor 3 is an accepted gap the design kept) | kind | no |

The team display on an enemy (type 1; design 4.3):

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 34 | `mates_aiming` | living mates whose `aim_of(know, mate)` answer's UID is this unit's (the 4.1 rule evaluated per mate, with its facing target; me excluded) | `n_t` | no |
| 35 | `mates_firing` | those with `live_trigger` | `n_t` | no |
| 36 | `dmg_last_s` | `damage_window[team][uid]`'s sum (1.6) | `bars[uid]` (bars) | yes |
| 37 | `priority` | `uid == my Unit.order_target` (`unit_schema.py:558`; design 5.2) | flag | no |
| 38 | `frag_catch` | living mates (me included) whose centre is within the frag row's blast radius of this unit's centre, walls ignored (design 3.3 accepted gap): `WeaponsTables.payload_for_ammo(FRAG_ROW).radius` (`weapons.py:1123`, `:71`; `[payloads.frag_standard] radius` `config.toml:4061`) | `n_t` | no |

The team display on a team-mate (type 0; design 4.2):

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 39, 40 | `aim_cos`, `aim_sin` | `cos / sin(Unit.facing_target_q)` -- the point it is turning to (`unit_schema.py:495`; the live aim becomes the target at slot 1a, `turning.py:213-215`) | angle | no |
| 41 | `aim_is_unit` | `aim_of(know, it)`'s category != 0 (with its facing target in place of its facing) | flag | no |
| 42, 43, 44 | `feet_dx`, `feet_dy`, `throttle` | `Unit.live_move_dir` (`:561`; `MoveDirIntent.dx_q / dy_q / throttle_q`, `src/simulation/intents.py:60-62`) / `FP_ONE`; `(0, 0, 0)` when `None` | angle; `THROTTLE_ONE` (`movement.py:174`) | no |
| 45, 46 | `land_dx`, `land_dy` | where its wound-up throw lands, from ME: its centre + `throw_reach_q` (`:525`; `THROW_NO_REACH` -> the verb's fixed 6 tiles, `simulation.py:1509`) along `(cos, -sin)(facing_target_q)` (design 3.7: the facing target and the reach), minus my centre; `(0, 0)` when not winding up | `L_ref` | yes |
| 47 | `order_what` | its `Unit.order_what` | kind | no |
| 48, 49 | `order_place_dx`, `order_place_dy` | its order's place centre minus MY centre (as 1.5 cols 89, 90); 0 without an order | `L_ref` | yes |
| 50 | `order_same_place` | its `order_where == mine` (both held) | flag | no |
| 51 | `order_left` | as 1.5 col 86 for its order | `limit_ticks` | no |
| 52 | `me_in_its_lane` | the lanes grid (2.5) at my centre tile holds its uid or -2 (several owners, one of them it: recomputed from the segments per pair -- CHOICE: the per-pair answer is the rasteriser's own segment test, `draw_lanes`'s cells for that one owner) | flag | no |
| 53 | `it_in_my_lane` | the same with the roles swapped | flag | no |

A record (type 2):

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 54 | `age` | `tick - Record.tick` (`team_picture.py:104`) | `memory_ticks` | no |

A grenade or a charge (type 3):

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 55 | `obj_kind` | the `[ammo.*]` row index of `Projectile.ammo_name` / `PlantedCharge.ammo_name` among the hand-grenade rows followed by the charge rows (`WeaponsTables.hand_grenade_rows + charge_rows`, `weapons.py:1038`; `loadouts.py:289`) | kind | no |
| 56 | `whose` | `owner_id` (`combat.py:111`, `charges.py:45`), decided WITHOUT a live read (v2, critique MAJOR 2): 0 `owner_id` is my id, 1 it is in `TeamKnowledge.roster_ids` (my team's muster, fixed at spawn -- a mate who threw it and has since died or turned still reads 1), 2 any other `owner_id >= 0`, 3 `owner_id == -1` | kind | no |
| 57 | `time_left` | `get_detonate_tick() - tick` (`combat.py:132`) / `det_tick - tick` (`charges.py:48`); 0 when remote | `fuse_ticks_from_seconds(fuse_max_seconds)` of its family's weapon row (`hand_grenade` `config.toml:3314`; `c4_satchel` `:3375`) | yes (a charge may be timed past a grenade's max) |
| 58 | `remote` | a charge with `det_tick is None` | flag | no |
| 59 | `blast_r` | its payload row's `radius` (`PayloadDef.radius`, `weapons.py:361`; `payload_for_ammo`) -- 0 for a row without one (smoke, tear, flash) | `L_ref` | yes |
| 60 | `in_blast` | my centre within that radius of its tile (walls ignored, design 3.3) | flag | no |
| 61 | `on_floor` | a charge, or a projectile at its target (`fx == target_fx and fy == target_fy`) | flag | no |

A door (type 4):

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 62 | `open` | `door.state == DOOR_OPEN` (`src/simulation/entities/door.py:44`) | flag | no |
| 63 | `in_reach` | `verb_gates.door_in_reach(me, door)` (1.7) | flag | no |
| 64 | `hull` | `horde.is_hull_door(gmap, door.span)` (`horde.py:297`) | flag | no |
| 65 | `wired` | `door.ordinal in room_graph.wired_door_ordinals(sim.level.wires)` (`room_graph.py:74`) | flag | no |

The mark:

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 66 | `prev_aim` | this entry was the previous decision's aim target (1h's held uid; 0 in 1g) | flag | no |

## 3. The two windows (design 3.4, 3.5, 4.5, 5.2)

### 3.1 Frames

- **Near:** `WIN = 25` tiles square at full resolution, centred on the
  soldier's centre tile `(cy, cx)`: rows `cy - 12 .. cy + 12`, columns
  likewise; plane cell `[i, j]` is tile `(cy - 12 + i, cx - 12 + j)`.
- **Far:** `WIN = 25` blocks square of `BLOCK = 4` tiles, centred on the
  soldier's block `(cy // 4, cx // 4)`; cell `[i, j]` is block
  `(cy // 4 - 12 + i, cx // 4 - 12 + j)`, i.e. tiles `[4 by .. 4 by + 4) x
  [4 bx .. 4 bx + 4)`. The block grid is anchored at tile `(0, 0)`; the map
  is padded to a multiple of 4 with the outside value.
- **Outside the map** (CHOICE): `wall` reads 1 (nothing enters there), every
  other plane 0, `route` -1. (`has_los` lets a line run outside the map
  without blocking, `gamemap.py:2457`; the sight planes follow it.)
- Both windows never rotate (design 1).

### 3.2 The physics planes (both windows; design 3.4's table, the critique's denominators)

Each is read ONCE per tick for the whole map (a float64 plane), reduced to
blocks for the far window (3.3), then cropped per soldier. `u` is the
observing soldier; "profile" is `u.environment` (`unit.py:213`,
`EnvironmentProfile`, `src/simulation/environment.py:23`).

| order | name | read from | divided by (the unit's own terms) | squash | type |
|---|---|---|---|---|---|
| 0 | `temperature` | `gas_fixed.dequantize_q16(gmap.temperature)` (the crop's read, `sensor_accessor.py:137`): game degrees above ambient | `profile.temperature_max` (`environment.py:40`, `[env_profiles.*] temperature_max` `config.toml:5044`; the plane shows where the hot tiles are, the bots avoid them at `[bots] hot_temp`, `control_bots.py:599-610`) | yes | float16 |
| 1 | `fire` | `fire_fixed.dequantize_f32(gmap.fire)` (`:139`) | the field's full scale, 1.0 | yes | float16 |
| 2 | `poison` | `gas_fixed.dequantize_trace_f32(gmap.gas[POISON])` (`src/simulation/gas_fixed.py:127`) | `[exchange] poison_min_density` (`:5193`; the row's threshold, `exchange.py:634`), the quotient TIMES the unit's `mitigation_for(u).resist_mult[POISON]` (`damage.py:298`; a zombie's 0 reads 0 -- harmless to it, `exchange.py:644`) | yes | float16 |
| 3 | `teargas` | the teargas slice likewise | `[exchange] teargas_blind_density` (`:5185`, `exchange.py:589`) | yes | float16 |
| 4 | `smoke` | the smoke slice likewise (`:133`) | `[bots] smoke_dense` (`:4956`; acts on nothing yet, the bots avoid it) | yes | float16 |
| 5 | `pressure` | `atmosphere_fixed.dequantize_f32(gmap.atmosphere)` (`:130`), atm | the profile's band: `(p - pressure_min) / (pressure_max - pressure_min)` (`environment.py:31-32`; CHOICE: 0 at the low limit, 1 at the high, a vacuum negative; no coupling yet, design 3.4) | yes | float16 |
| 6, 7 | `wind_x`, `wind_y` | `GameMap.wind_load_plane()` (1.7): per tile the wind's one-tick impulse `unit_impulse.wind_impulse_q(table, W_i)` (`unit_impulse.py:515`; standing area) over `u.mass_g` (`unit_schema.py:487`) as a velocity change, along `(wind_x, wind_y)` of that tile (the wind's own x east, y SOUTH: the planes are world axes, `gamemap.py:2406-2412`) | `ImpulseTable.cap_q` (`:197`; one tick of hold, `[impulse] k_hold` `:4856`: at 1.0 the wind pushes the standing body, the #140 fall edge) | yes | float16 |

Reserved and not built (design 3.4): oxygen, water depth, fuel vapour.

**ACCEPTED GAP: the fields carry every unit's body** (v2; critique MAJOR 3;
RULED by Erik, 2026-11-12: "on the wind question, its OK! We accept it!").
The eight planes are not a function of public state alone:
`GameMap.stamp_units` (`gamemap.py:2071-2090`) writes every living body into
`dyn_permeability` (0.5 on a marine's 3x3), `dyn_wave_absorb` and
`dyn_heat_atten_q`, which the gas solver and the radiation sweep read
(`physics_runner.py:995, 1215, 1293`). The critic measured it: two
`ship_demo` matches, idle, seed 7, identical but for one marine dead from
tick 0, differ after 300 ticks in ~4 700 cells out to 96 tiles (wind up to
0.44 m/s, mean 0.08 within 3 tiles of the body against a field noise of
~0.1; temperature mean 5.7 game degrees within 3 tiles). The physics is
real -- a body does obstruct air and heat -- so the handling is an ACCEPTED
GAP, stated here and in the CLAUDE.md row (section 10): "the fields carry
the bodies' own aerodynamic and radiative imprint (`stamp_units`)". A
hidden body can therefore be read off the physics planes where the air
moves; the candidate remedies if that ever matters (a v2 spec, Erik's
call) are a coarser quantisation of the wind planes, the far window's
physics planes from the block MEAN, or no far wind.

**The measurement** (patch (b)'s instrument `tools/obs_body_imprint.py`,
re-run at (c) on the merged (b) code, 2026-11-12): the `test_wind_doorway_e2e`
scene (ship_demo, the zombie room over-pressured by +0.44 atm venting through
its two west doors -- a half-second JET, ticks 1..24, then a lasting DRAFT),
played twice in lock-step with ONE hidden body (standing in the draft in
front of the upper door; walking out through the lower door) and with none;
per tick and plane the largest absolute difference of the SQUASHED float16
plane values within the 25 x 25 window around the body (near) against the
same statistic beyond 50 tiles from it (far), the max over the phase's
ticks, and their ratio:

| case, phase | window | temperature near / far (ratio) | wind_x | wind_y | pressure |
|---|---|---|---|---|---|
| standing, jet | near | 6.10 / 0.15 (40) | 1.50 / 0.034 (44) | 1.93 / 0.029 (68) | 0.011 / 0.0024 (4.5) |
| standing, jet | far | 2.01 / 0.11 (18) | 4.29 / 0.037 (118) | 4.13 / 0.029 (145) | 0.0066 / 0.0024 (2.7) |
| standing, draft | near | 2.22 / 0.27 (8.4) | 0.12 / 0.012 (10) | 0.42 / 0.017 (25) | 0.0012 / 0.0015 (0.8) |
| standing, draft | far | 1.03 / 0.17 (6.2) | 0.26 / 0.016 (16) | 0.42 / 0.017 (25) | 0.0015 / 0.0015 (1.0) |
| walking, jet | near | 5.66 / 0.15 (37) | 1.96 / 0.008 (255) | 2.48 / 0.009 (284) | 0.017 / 0.0032 (5.5) |
| walking, jet | far | 2.07 / 0.10 (21) | 2.53 / 0.008 (330) | 3.24 / 0.009 (372) | 0.0098 / 0.0032 (3.1) |
| walking, draft | near | 1.70 / 0.35 (4.8) | 0.047 / 0.009 (5.1) | 0.090 / 0.007 (13) | 0.0010 / 0.0007 (1.3) |
| walking, draft | far | 0.78 / 0.19 (4.0) | 0.17 / 0.017 (10) | 0.19 / 0.023 (8.0) | 0.0012 / 0.0007 (1.7) |

Fire, poison, teargas and smoke: 0 everywhere (none in the scene). In the
jet a body's wake reaches the squash's cap in the wind planes (4.1-4.3 of
4, far window) and ~6 in temperature (two capped values apart), tens to
hundreds of times the far-field divergence; in the draft it is 5-25 times
it in wind and 4-8 in temperature; pressure never leaves the noise.
Consequence for the tests: the pair tests of section 7 compare the physics
planes (and the `air` / `felt_temp` columns that read them) only in the
pair whose bodies are identical, (f).

### 3.3 How the far window is made (design 3.7's third thing)

Once per tick per env, not per soldier: each physics plane `P` (h, w) is
padded to `(4 ceil(h / 4), 4 ceil(w / 4))` with 0 and reduced
`P.reshape(H, 4, W, 4).max(axis=(1, 3))` -- the MAX over the block's tiles,
the coupling rows' own reduction (the hottest, densest tile on a body is
the exposure, `exchange.py:594, 641`). The wind planes reduce by the block's
largest `W_i` and carry that tile's direction (as `footprint_wind`,
`gamemap.py:2425-2433`). The structure planes reduce as 3.4 says. The
soldier's far window is then a crop of the reduced planes. The per-unit
denominators (temperature, poison, wind) are applied AFTER the reduce, per
soldier, on the cropped 25 x 25 (a divide of 625 cells).

### 3.4 The structure planes (both windows)

| name | near | far | type |
|---|---|---|---|
| `wall` | `gmap.solid` (what stops a round and a line of sight; CHOICE: a grille is not solid and reads as floor here -- an accepted gap, the route plane carries walkability) | the share of solid tiles in the block, `count / 16` (outside tiles count as solid) | near uint8; far float16 (16 levels) |
| `door` | 0 none, 1 a tile of a living door's span with `state == DOOR_OPEN`, 2 with `DOOR_CLOSED` (`door.py:43-44`; the spans from `sim._doors`) | the largest code in the block | uint8 |

### 3.5 The sight planes (near window; design 3.4, the many-tiles routine of 1c)

| name | what | read from |
|---|---|---|
| `watched` | the tiles at least one KNOWN enemy has a line of sight to | for each record of my team's picture (seen or remembered, at its recorded `centre`): `vision.window_clear(sim, ry, rx, cy - 12, cx - 12, 25, 25, to_window=False)` (`vision.py:217`: FROM the enemy's tile to each window tile), OR-ed. A known enemy's RECORDED place, never its live one (honest: the team's knowledge). |
| `sees_target` | the tiles from which my aim target can be seen | `window_clear(sim, ty, tx, ..., to_window=True)` (FROM each window tile to the target's tile) for the entry 1h's decoder holds as my aim (its live centre if seen, its record's if remembered); all 0 with no aim entry (1g: always 0 -- CHOICE: the plane exists, with its builder, so 1h adds one line) |

Both uint8. Cost: one window call (625 lines) per known enemy per soldier
per decision, plus one for the target; the routine is C++ (`sight_lines`,
the 1c row). CHOICE: per soldier, not shared across the team in (b); if the
budget of section 6 fails, the measured next step is one window per known
enemy over the union rectangle of the team's windows, cropped per soldier.
When cones return (rung 8) the enemy's facing enters here (design 3.4).

### 3.6 The fire-lanes plane (near window; design 4.5)

`TeamKnowledge.lanes` (0.5): `fire_lanes(sim, team, lane_cell,
lane_halfwidth)` once per team per decision, inside the view's build
(`team_picture.py:227`, from STATE: every mate holding its trigger;
the dials `[bots] lane_cell` / `lane_halfwidth` `config.toml:4953-4954`,
CHOICE: the same lanes the bots read, one source); the plane cell for tile
`(y, x)` is 1 when `grid[y // cell, x // cell]` is an owner other than me or
-2 (several owners) -- "the tiles that lie in a team-mate's line of fire".
uint8.

### 3.7 The far window's unit planes (design 3.4 "in the far window also"; the critique's major 4)

Drawn from the TEAM KNOWLEDGE view (0.5), never from the crop's `units_*`
planes (`sensor_accessor.py:84-92`, which see every unit and are not part of
this spec). v2 (critique minor 11): drawn from the WHOLE picture, not from
the capped list -- a ninth remembered enemy 30 m away was in no array, while
the bots' team layer reads every record:

| name | what | type |
|---|---|---|
| `mates` | the number of `TeamKnowledge.mates` (me excluded) whose centre tile lies in the block | uint8 count |
| `enemies` | the number of `TeamKnowledge.seen` units (every one, past the list's cap of 16) whose RECORDED centre (= its live one this tick) lies in the block | uint8 count |
| `records` | the number of the other `TeamKnowledge.records` (every one, past the cap of 8) whose recorded centre lies in the block (CHOICE: the remembered places drawn too, as a third plane) | uint8 count |

### 3.8 The order's planes (far window; design 3.5, 5.2)

| name | what | type |
|---|---|---|
| `route` | `route(sim, graph, order_where, fp)` (1.7): per block the MIN of the field over the block's 16 anchor cells, divided by `L_ref`, squashed; -1 where every cell is `UNREACHED` or when I hold no order | float16 |
| `place` | 1 on a block that holds a tile of my order's place (a room's rectangle, `graph.grid == room`; an opening's rectangle); 0 otherwise or with no order | uint8 |

### 3.9 The planes in order

Near (13 planes, `(13, 25, 25)`): `wall` u8, `door` u8, the eight physics
planes f16 in 3.2's order, `watched` u8, `sees_target` u8, `lanes` u8. Stored
as two arrays, `near_f` `(8, 25, 25)` float16 and `near_u` `(5, 25, 25)`
uint8, in the orders `[temperature, fire, poison, teargas, smoke, pressure,
wind_x, wind_y]` and `[wall, door, watched, sees_target, lanes]`.

Far (15 planes): `far_f` `(10, 25, 25)` float16 = `[wall_share, the eight
physics planes, route]`; `far_u` `(5, 25, 25)` uint8 = `[door, mates,
enemies, records, place]`.

Per soldier per decision: `(8 + 10) * 625 * 2 + (5 + 5) * 625 = 28.75 KB`.

## 4. The layout in the env's buffers, and the two row flags

### 4.1 Shapes (`Layout.shapes`, `rl_env.py:413`, under `[obs] spec = "v1"`)

`N` envs, `U = T * S` rows (row `t * S + s` = team `t`'s roster slot `s`,
`rl_env.py:57-61`), `E = S + 33` entries, `V = 97`, `F = 67`.

| key | shape | dtype | what |
|---|---|---|---|
| `vector` | `(N, U, V)` | float32 | section 1 |
| `list` | `(N, U, E, F)` | float32 | section 2 |
| `list_mask` | `(N, U, E)` | uint8 | 1 = a real entry |
| `list_uid` | `(N, U, E)` | int32 | the entry's game id (2.1), -1 empty; the decoder's and the mask's, never the network's |
| `near_f` | `(N, U, 8, 25, 25)` | float16 | 3.9 |
| `near_u` | `(N, U, 5, 25, 25)` | uint8 | 3.9 |
| `far_f` | `(N, U, 10, 25, 25)` | float16 | 3.9 |
| `far_u` | `(N, U, 5, 25, 25)` | uint8 | 3.9 |
| `valid` | `(N, U)` | uint8 | the `acts` flag (4.2; the existing key, its meaning unchanged) |
| `in_match` | `(N, U)` | uint8 | the second row flag (4.2) |
| `final_*` | the same eight observation arrays | | the terminal observation of an env that ended inside the step (`BatchedEnv._final_obs`, `rl_env.py:1222`) |

The compass and the order are columns of `vector` (1.5); there is no
separate key. `actions`, `reward`, `counters`, `terminated`, `truncated`
and the info arrays are unchanged (`rl_env.py:418-433`). Every observation
array is ENV-OWNED, overwritten by the next step (`BatchedEnv._buf`,
`rl_env.py:1111-1116`; the module doc's rule): the new keys join `_buf` and
`_obs` / `_final_obs` copy into them, never allocate. The `"crop"` spec's
keys (`crop`, `vector` of `VECTOR_FIELDS`) are what they are today; a run
file names one spec and gets that spec's keys.

**Who may read which rows** (v2; critique MAJOR 5). `obs` holds every
marine team's rows, because one env serves every team; a row is the
knowledge of ITS team only. A policy acting for team `t` -- its network,
any batching, shared attention or pooling over rows, rung 4's commander --
reads ONLY rows `t*S .. t*S + S - 1` of every `obs` key (its own soldier's
row and its team-mates' rows: the radio is perfect, design 4.1, so a
team-mate's row is team knowledge), including `valid` / `in_match` of
those rows only (another team's `valid` is a death counter of the enemy).
It reads NOTHING of `info`, `reward` or `counters` (they credit kills and
hits the team never saw, `EnvRunner._count`, `rl_env.py:585-612`): no
previous reward fed to a recurrent policy, no counter as an input. Another
team's rows are full-view material (section 5's side of the wall). The
TRAINER (not the policy) may read `reward`, `counters`, `terminated`,
`truncated` and `info` (the full view for the value network, section 5) to
form its losses. The one path a trainer takes to a team's observation is
the env's helper `BatchedEnv.team_rows(obs, t)` (NEW, patch (a)): a dict of
the same keys, each the view `arr[:, t*S : (t+1)*S]` (a view, never a
copy, the env-owned rule), `final_*` likewise; 1i's and rung 2's trainer
tests assert that the policy's forward pass is fed only through it.

Per soldier per decision: vector 0.4 KB + list 36 x 67 x 4 = 9.6 KB + windows
28.75 KB = about 39 KB; a batch of 64 envs x 8 rows = 20 MB, within the
module doc's measured budget for the env-owned copies (`rl_env.py:87-93`).

### 4.2 The two row flags (design 6.3; the critique's L4 M2)

| row | `in_match` | `valid` (= acts) | set from |
|---|---|---|---|
| a living soldier | 1 | 1 | the roster unit is `alive` and `team == t` (`EnvRunner.row_units`, `rl_env.py:618-627`) |
| dead or turned | 1 | 0 | the roster holds a unit for the slot (`Simulation.roster(t)`, `simulation.py:584`) that is dead or whose `team != t` |
| an empty roster place | 0 | 0 | no unit in the slot (`slot >= n_t` of the level's match row) |

Written by `EnvRunner.observe` beside the observation, both every step and
in `final_*`. A row with `valid` 0 has a zero vector, an empty list (mask
0) and zero windows: a dead soldier's observation is not built (its value
estimate is the full view's business, 5.3). The trainer reads `in_match`
to pay reward and value to a dead row and `valid` to let it choose (1i's
contract, design 6.8).

## 5. The value network's full view (design 7)

### 5.1 What it is

One named spec, `FULL_VIEW_NAME = "full_v1"`, served beside the
observation by the same door (`EntityFieldAccessor.observe(spec, ...)`)
into the farm's own buffers, under `info["full_view"]` of
`BatchedEnv.step` / `reset` -- NEVER a key of `obs`. Centred on the same
soldier as its observation (design 7.1). The soldier's own vector and
windows are read a SECOND TIME from the observation arrays by the trainer
(design 7.1: "the same stored arrays, not a copy"); the full view holds
only what the soldier does not get.

### 5.2 Its arrays

| key | shape | dtype | what |
|---|---|---|---|
| `full_list` | `(N, U, 64, F + 18)` | float32 | 5.3 |
| `full_mask` | `(N, U, 64)` | uint8 | |
| `full_uid` | `(N, U, 64)` | int32 | |
| `full_match` | `(N, U, M)` | float32 | 5.4; `M = 2 T + 3` |
| `full_watched` | `(N, U, 25, 25)` | uint8 | 5.5 |

All five are keys of `info["full_view"]`. The terminal full view of an env
that ended inside the step (a truncated episode's value bootstrap needs it)
is `info["final_full_view"]`, the same five keys (v2, critique minor 6) --
never a `final_full_*` key inside `info["final_obs"]`, the policy's terminal
dict.

The full view's builders take `sim` itself (they are the other side of the
wall, 0.5) and keep their OWN caches. Its type-3 entries carry `side` -1 and
`whose` by the roster as 2.2 does; their `full_uid` is `2^20` plus the
object's position in the full list's object block (privileged, never an
`obs` value).

### 5.3 The full list (replaces the soldier's list, design 7.1 / 7.2)

Up to 64 entries: EVERY living unit of every side where it truly is (the
live `Unit`), the marines first in `(team, slot)` order, then the zombies
nearest first (dormant ones included), then every live grenade and charge
(types 3, as 2.2, all of them, with owner and time left), then every
living door (type 4). More than 64: the farthest zombies, then the
farthest objects, are dropped (never a marine). The first 67 columns are
2.2's entry, built with the same builders against the observing soldier
(so `dx`, `dist_c`, `in_range`, `lof_*`, `aims_at` mean the same), with
these differences:

- a unit entry's `type` is 0 for a team-mate and 1 for every other unit
  (there are no records: the unit itself is shown);
- `hp`, `exposure`, `firing`, `reloading`, `winding`, `planting`, the
  statuses, `zombie_state` are the LIVE unit's for every entry;
- `aims_at` is evaluated with EVERY living unit as a candidate (the full
  view knows them all), still from the unit's facing (or a mate's target).

The 18 extra columns (`F .. F + 17`; design 7.1 "per unit"):

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 67 | `knowledge` | 0 my team sees it (uid in `visible_by_team[team]`), 1 my team has a record of it (in `records(team)`), 2 my team does not know of it; my own team-mates: 0 | kind | no |
| 68, 69, 70 | `rounds`, `mags`, `ammo_untracked` | as 1.2 cols 24-26 for this unit | its weapon's `mag_size`, its loadout's `magazines` | as 1.2 |
| 71..76 | `grenade[k]` | as 1.2 cols 29.. (the count ratios only; 6 rows today) | its loadout's counts | no |
| 77 | `charges` | as 1.2 col 44 | | |
| 78, 79 | `melee_cool`, `shove_cool` | as 1.2 cols 49, 51 | | |
| 80..83 | `status_left` for knocked_down, stunned, burning, poisoned (CHOICE: the four kinds a shipped path applies; the soldier's own vector carries all nine) | as 1.1 | `STATUS_SCALE_TICKS` | yes |
| 84 | `order_what` | its `Unit.order_what` | kind | no |

Its order's place and the air where it stands are NOT columns (CHOICE): the
place is the far window's `place` plane for the soldier, not per unit; the
air is readable from the soldier's windows for every unit inside them, and
a unit outside them is beyond the design's purpose. A reader that needs
them adds a version of this spec (the Full view row, design 10).

### 5.4 The match (design 7.1 "the match")

`full_match`, `M = 2 T + 3`: per marine team `t` of the run's `T`
(zero for a team the level lacks) its living count / `n_t` and its
BALANCE -- the sum over its living units of `current_hp / bars[uid]` over
`n_t` (CHOICE: the reward's balance of design 6 is 1i's; until 1i lands
the balance here is this health sum, the same quantity 1i's team part
differences; 1i replaces the builder, not the column) -- then the zombies
awake / `spec.zombies`, dormant / `spec.zombies` (`zombie_activated`),
and `time_left` (as 1.5 col 77). The soldier's own team counts (1.5 cols
78-83) stay in its vector.

### 5.5 The watched-by-everyone plane (design 7.1 "the map")

`full_watched`: 3.5's `watched` built from EVERY living enemy of the
soldier's team at its LIVE centre (zombies included, dormant or not), a
`window_clear` per enemy per soldier. Its own key of 5.2,
`info["full_view"]["full_watched"]` -- never a channel of an `obs` array
(v2, critique minor 5); the soldier's own `watched` (a channel of `near_u`)
stays as it is.

### 5.6 A dead soldier's view (design 7.2)

For a row with `in_match` 1 and `valid` 0 the full view IS built, centred
on the unit's last centre tile (a corpse keeps its position; a converted
unit its live one), with `full_match` and `full_list` as for a living row,
and `full_list`'s first entry replaced by the dead soldier itself with
`type` 0, `hp` 0 and `age` (col 54) = `tick - death_tick` over
`limit_ticks` (the death tick from the trackers: the first update that saw
it dead or turned). The soldier's own vector and windows for that row are
zero (4.2). The "blanked" variant of 7.4's experiment is the trainer's
(it zeroes the arrays); the env always builds the view.

### 5.7 The proof obligation: the policy never sees it

- `obs` (the dict `BatchedEnv._obs` returns) holds exactly the keys of
  4.1 and no key of 5.2: asserted by `tests/test_obs_spec_v1.py` on the
  live env.
- **The chokepoint** (v2; critique MAJOR 4, replacing v1's two-name grep):
  the observation builders of sections 1-3 take `(know, u)` -- the
  `TeamKnowledge` view of 0.5 and the observing soldier -- and never the
  `Simulation`; `TeamKnowledge.build(sim, picture, team)` is the only
  observation function that touches `sim`. Every routine they call
  receives `know.world` (the `ObsWorld` facade) as its `sim`, so a read of
  `units`, `roster`, `get_unit`, `projectiles`, `planted_charges`,
  `team_picture`, `tick_events` raises at run time; the trackers answer
  only in their scope (0.5). The full-view builders take `sim` and own
  separate caches.
- **The scan test**, widened to every module the builders call: an AST scan
  of `obs_spec_v1.py` (bar `TeamKnowledge.build` and section 5's
  functions, listed by name), `line_of_fire.py`, `turning.ticks_to_within`,
  `verb_gates.door_in_reach`, `obs_spec_v1.route`'s callees in `horde.py`
  (`flood`, `walkable_tiles`, `block_passable`, `candidates`),
  `gamemap.wind_load_plane`, `exchange.felt_temperature` and the vision
  routines `ray_clear` / `rays_clear` / `window_clear` finds none of the
  banned names (`.units`, `.roster`, `.get_unit`, `.projectiles`,
  `.planted_charges`, `.team_picture`, `.tick_events`, `_units_by_id`,
  `state_for`) outside the listed exceptions; a new callee joins the list
  in the patch that adds it. Every observation cache is keyed by its inputs
  (0.5).
- **The rows** (4.1): a policy of team `t` reads only team `t`'s rows,
  through `BatchedEnv.team_rows(obs, t)`, and nothing of `info`, `reward`
  or `counters`; the trainer reads those for its losses only.
- **The behavioural tests** (section 7, "shows nothing the team cannot
  know"): the counterfactual family (a)-(f), each comparing only the
  observing team's rows.
- The network-side wall (separate weights, scrambling the full view's
  inputs changes no choice) is rung 2's test (design 8.6), not 1g's.

## 6. The cost budget per decision (design 3.7)

The design gives no number; it names three things that must be built for
the cost to hold and says the farm's speed is measured against today's.
Two of the three exist (1c's C++ routine; 1a's all-round sight skips
zombie observers, `vision.py:341`); the third is 3.3. The sight cache is
built every tick already (slot 11's update calls `vision.state_for`,
`team_picture.py:149`), so the observation never builds it.

**The budget** (CHOICE, the lead's; Erik may move it): with `[obs] spec =
"v1"`, the full view on and `decimation = 4`, the farm's env-ticks/s is at
least **70 %** of the same run under `"crop"` -- the whole v1 observation
may cost at most three tenths of the match. Measured by the §9.1 protocol
of `docs/residency_benchmark_2026-11-07.md` with `tools/rl_smoke.py`
(`drive`, `:41`): the SAME fixed number of steps per env from fresh
episodes (3 000 steps at decimation 1 was the lock-step reading's,
`:1340`; 750 at decimation 4 covers the same ticks), 16 workers, 4 envs per
worker (`:1359`), `ship_demo` 4 v 4 + 4 zombies, random rows, before and
after, both run files' text in the patch's checkpoint comment.

**What must be measured**, each a line of the report, in ms per soldier
per decision on the Home Desktop (`tools/frame_profile.py`'s timers around
the builders, or `time.perf_counter` in a one-off script; the estimates are
the lead's and are to be replaced):

| part | where | estimate | measured by |
|---|---|---|---|
| the trackers' update | slot 11, per tick per env | 0.05 per tick | (a) |
| the vector and the list (40 entries; `aim_of` over ~30 candidates with a `ray_clear` each; `ticks_to_within` per enemy, at most 48 law steps) | per soldier | 0.6 | (a) |
| the objects' line-of-sight gate (`rays_clear`, one call per team, in the object tracker; v2: every tick, not every decision) | per team per tick | 0.05 | (a) |
| `TeamKnowledge.build` (the seen map, the roster ids, the lanes) | per team per decision | 0.05 | (a) |
| the physics planes read + the far reduce | per tick per env (3.3) | 0.3 per tick | (b) |
| the two crops and the per-unit denominators | per soldier | 0.1 | (b) |
| the sight planes: one `window_clear` (625 lines) per known enemy per soldier, plus the target | per soldier | 0.4 at 12 known enemies (the union-rectangle sharing of 3.5 is the fallback) | (b) |
| the route field (a flood per `(place, fp)` per topology change, cached) | per change | 1 per flood (`horde.flood`, C++) | (b) |
| the lanes grid | per team per decision | 0.05 | (b) |
| the full view: the 64-entry list, `full_watched` (a window per living enemy per soldier) | per soldier | 0.8 | (c) |

At 8 soldiers and a decision every 4 ticks the estimates sum to about 2.5
ms per soldier-decision = 20 ms per 4 ticks against ~36 ms of ticks
(`:900`: 9 ms a tick), i.e. a 35 % loss -- OVER the budget on the
estimates. The order of attack if the measurement agrees: (1) the sight
planes' union sharing (3.5); (2) `aim_of` and the lanes per TEAM per
decision, cached and read per soldier; (3) the full view's `full_watched`
from one window per enemy over the team's union. Every such cache is keyed
by its inputs and the full view's are separate from the observation's
(0.5): sharing a window between the two sides is not an optimisation this
spec allows. The budget is held at
(c)'s merge, not at (a)'s or (b)'s: each of those reports its parts and
the farm's number as information.

**As built, (c), 2026-11-12** (the protocol above, one session on the Home
Desktop, 64 envs / 16 workers, decimation 4, 750 measured steps after 20
of warmup; the run files `run_crop.toml` / `run_v1.toml` /
`run_v1_nofull.toml` = `run_v1.toml` + `[obs] full_view = false`):
`crop` 417.3 env-ticks/s (p50 597 ms a step), `v1` without the full view
385.3 (92 %, p50 665), `v1` with it 358.2 (**86 %**, p50 722). The budget
holds; nothing of the order of attack was applied. Per part on ship_demo
(in-process, 8 rows, `bots` vs `bots`): the observation 7.7 ms per
env-decision (the windows about half of it: 16 crops and squashes per
soldier), the full view 6.6 (first built at 9.6; the two things that
cut it: the per-unit columns computed once per unit instead of once per
(row, unit), and the rows' lines of fire in ONE `rays_clear` call --
`line_of_fire.solid_in_way_many` -- instead of one per pair; what is
left is the `window_clear` per (soldier, enemy) for `full_watched`, ~1
ms, and `ticks_to_within` per pair, ~1 ms).

## 7. The tests (properties; `tests/test_obs_spec_v1.py` unless named)

| property | breaks if |
|---|---|
| **Equals an independent reconstruction.** Over a `ship_demo` match of `bots` vs `bots` with the farm's rows observed every 4 ticks for 600 ticks, every vector column, every list entry and every plane equals a reference built in `tests/_obs_reference.py` from the sim's state alone -- a second, plainer implementation written from THIS document (loops over units and tiles, `has_los` per tile instead of the window routine, the bots' own `_teammate_on_line`, the horde's `flood_reference`), never importing `obs_spec_v1` (v2, critique minor 8: this is a CORRECTNESS gate -- the reference is written from this document, so a leak written into the document passes both; the honesty gate is the next row) | a builder reads the wrong seam, denominator, order or frame |
| **Shows nothing the team cannot know: the direct checks.** For every decision of that match: every `list_uid` of types 0-2 is my living team-mate's id or a key of `records(team)`; every type-3 `list_uid` is a value of the team's object tracker for an object in `seen_now[team]` (a function of the objects the team has seen), and its tile is `rays_clear(physical=False)` from a living team-mate's centre; the far `mates` / `enemies` / `records` planes equal the counts of `TeamKnowledge.mates` / `.seen` / the other records inside each block (the whole picture, not the capped list); `watched` equals the OR of `has_los` from the RECORDED centres (never a live position) | an entry, a plane or a flag leaks a unit the team has no record of |
| **Shows nothing the team cannot know: the counterfactual family** (v2; critique MAJOR 6, replacing v1's sealed-room zombie pair, which perturbed nothing a builder reads and changed `spec.zombies`, the bodies in the air and the unit ids). Each case is TWO sims built from the SAME match.toml, level and seed (so every denominator, roster, `n_t`, `spec.zombies` and unit id is equal), stepped identically to a tick `T`, then differing ONLY in hidden state of a unit outside `visible_by_team[team]` since its last sighting, introduced at `T` through the sim's own doors (a `DamagePacket`, a status row, a conversion call, a throw intent) or, for (b), a test-only position write named as such; the OBSERVING team's rows of `vector`, `list`, `list_mask`, `list_uid`, `near_u`, `far_u`, `valid`, `in_match` are compared BYTE FOR BYTE at every decision for `memory_ticks` after `T` (only that team's rows: the other team legitimately differs). Because a body is stamped into the air (3.2, an ACCEPTED GAP), the comparison of (a)-(e) excludes the physics planes (`near_f`, `far_f` bar `wall_share` and `route`) and the vector's `air` and `felt_temp` columns (58-65, 96) -- the fields are judged by (b)'s imprint measurement, not here. The cases: (a) a remembered enemy killed unseen by a status vs. left alive (every array, not only its type-2 entry: counts, `watched`, `aims_at`, the far planes); (b) a remembered enemy moved unseen; (c) a remembered enemy converted unseen; (d) an extra grenade thrown by an unseen unit before a grenade the team sees (the seen grenade's `list_uid` unchanged); (e) an unseen conversion of the owner of a charge the team sees (`side`, `whose` unchanged); (f) the same sim with the full view ON vs OFF -- here EVERY observation array is compared, physics planes included, since nothing physical differs (a cache shared with the full view) | a builder reads liveness, place, class, team or count of a unit or object the team cannot know, or a cache crosses the wall |
| **A record outlives its unit's unseen death.** A seen enemy walks out of sight and is killed by a status out of sight: its type-2 entry stays for `memory_ticks` with its `age` rising and its `dx`, `class`, `vel` frozen at the sighting; a seen death drops it on the next decision; its `cos`, `sin` are `Record.facing_q`'s (frozen too). Every OTHER array's invariance under the unseen death is case (a) of the family above | the list reads liveness or the live unit |
| **A level without a room graph is refused.** `load_levels` on a run with `spec = "v1"` and `levels = ["playground"]` (no `rooms.npy`) raises naming the level and the spec; the same run under `"crop"` loads | the refusal is missing or on the wrong path |
| **The two row flags' truth table.** A soldier killed mid-match: `in_match` 1, `valid` 0, zero vector, mask all 0; a converted one likewise; a slot beyond the team's `n`: 0, 0; a living one: 1, 1 -- in `obs` and in `final_*` | a dead row reads as an empty place, or is observed |
| **The squash.** Identity on `[-1, 1]`, monotone, continuous at +-1, capped at +-4; `squash(e) == 2`; every float16 plane lies in `[-4, 4]` | the curve changes or a builder skips it |
| **No raw number.** Every vector column and entry column is declared in `obs_spec_v1`'s tables with a denominator, `flag`, `kind` or `angle`; a column with none fails the test (a scan of the table, the roadmap's rule) | a builder stores a game number |
| **The far reduce.** On a random map the reduced planes equal a per-tile reference (the max over each block; the wall share; the wind's direction from the block's largest load) | the reshape or the padding is off by one |
| **The sight planes equal the one sight test.** `watched` cell for cell equals `gmap.has_los(ry, rx, y, x)` OR-ed over the records; `sees_target` equals `has_los(y, x, ty, tx)` | a window argument (`to_window`) is swapped |
| **The lanes plane.** A mate holding its trigger facing east marks the near cells along its facing to its range, widened by the dials; released, none; my own lane marks nothing | the plane reads the wrong owner |
| **The new functions.** `ticks_to_within` equals stepping `law_q16` by hand on three cases (already within; a quarter turn; the cap); `mate_in_way` equals `BotTeam._teammate_on_line` on 1 000 random triples; `door_in_reach` makes `use_door` return what it returned before (the verb-gate tests stay green); `wind_load_plane`'s footprint max equals `footprint_wind` on random footprints; `aim_of` on a hand-built scene: me / a mate / another side / nobody with the right uid, two candidates at equal distance answer the lower uid, an unknown unit never answered (category nor uid), a wall between answers nobody; `exchange.felt_temperature` equals the heat row's `t_felt` on random bodies near a fire and the heat row's damage is unchanged (`match_pin_check.py` EQUAL); `route`'s field at a room's seeds is 0 and its first step descends | a function is a copy of a law instead of a call of it |
| **The trackers.** `motion` equals the position delta of the last tick; `last_hit` holds the last damaging hit's direction and cause and clears nothing (age grows); `damage_window` sums hits my team dealt plus hits on a unit my team saw at the previous tick, nothing else, over the last `ticks_per_second` ticks -- including a hit by a THIRD side on an enemy my team sees at both ticks (the saw rule against the records before this tick's sightings, critique minor 1); `bars` = the unit's `current_hp` at spawn (`effective_vitality` for a marine) even for an enemy first seen wounded, and `[zombie] hp` for a converted one; the object tracker gives a seen object one id for its life, the next id to the team's next first-seen object, and nothing to an object never seen; the view refuses `motion` of a non-mate and `bar` of an unseen uid | a tracker reads on demand, counts an unseen hit, or hands out out-of-scope memory |
| **The full view.** Every living unit of the sim is in `full_list` exactly once (`full_uid`), the marines first in `(team, slot)` order; `knowledge` agrees with the picture; `full_watched` equals the OR of `has_los` from every living enemy's LIVE centre; `full_match` reproduces the counts; a dead row's view is built and its first entry is itself | the full view misses a unit, or uses the picture where it must not |
| **The policy never sees it.** `obs` has no key of 5.2 and `info["final_obs"]` none of 5.2's; the chokepoint of 5.7: a builder handed `know.world` and asked for each banned name raises `AttributeError`, and the widened AST scan of 5.7 passes; `BatchedEnv.team_rows(obs, t)` returns exactly team `t`'s rows of every key, as views | a full-view array or another team's row reaches the policy, or a builder reaches the sim |
| **The farm.** `tools/rl_smoke.py` on a `"v1"` run file with the full view, 2 envs, 2 workers, 10 steps: no exception, every array of 4.1 and 5.2 of its shape and dtype, `final_*` and `info["final_full_view"]` filled for an env that ended; the §9.1 numbers in (c)'s report | a shape, a dtype or the env-owned copy is wrong |
| **Replay unmoved.** A recorded `policy` match (zero rows) with `"v1"` observation on replays with every checkpoint equal; `tools/match_pin_check.py` EQUAL; `pytest tests -q` green | the observation writes sim state |

## 8. The cut into three implementation patches

Each its own branch, worktree, brief and `--no-ff` merge (plan 1.2, 1.5);
each small enough for one Fable agent to finish without a resume (plan 5:
a killed agent is rebuilt from its commits, never resumed). The order is
fixed: (b) builds on (a)'s module and layout, (c) on both. Every patch:
this machine's breach Python, `pytest tests -q`, explicit paths staged,
a commit at every coherent step, tests as properties from section 7, a
failing old test a finding, the house rules of plan 1.4. None touches
`src/control_bots_v1.py`. None has a HUMAN-TEST.

### (a) `192-1g-a-vector-list`: the soldier's vector, the list, the row flags

| | |
|---|---|
| files | NEW `src/simulation/obs_spec_v1.py` (sections 0.2, 0.4, 0.5 -- `TeamKnowledge`, `ObsWorld` --, 1, 2, 4.2 as data and builders; the window and full-view builders ABSENT), NEW `src/simulation/line_of_fire.py`, `src/simulation/team_picture.py` (the five trackers of 1.6 -- the object tracker included --, `Record.facing_q`, and the death tick of 5.6), `src/simulation/turning.py` (`ticks_to_within`), `src/simulation/verb_gates.py` (`door_in_reach`), `src/simulation/sensor_accessor.py` (`observe`), `src/rl_env.py` (`[obs] spec`, the refusals, `Layout` keys `vector` / `list` / `list_mask` / `list_uid` / `in_match` and their `final_*`, `EnvRunner.observe` through the accessor, `_buf`, `BatchedEnv.team_rows`), NEW `tests/test_obs_spec_v1.py` + `tests/_obs_reference.py` (the vector, list and flag parts), `tests/test_team_picture.py` (the trackers), `tests/test_turning_law.py` (`ticks_to_within`), `tests/test_verb_gates.py` (`use_door` unchanged), `CLAUDE.md` (section 10), `tests/twin_manifest.toml` (`[not_twinned]` rows for the two new modules, regenerated) |
| builds | `TeamKnowledge` and `ObsWorld` (0.5) with the chokepoint, the whole vector (the field-dependent `air` block, cols 58-65, and `felt_temp`, col 96, written from the builders of (b): in (a) those nine read 0 and the test skips them -- CHOICE), the five list blocks and the 67-column entry (cols 28, 52-53, 66 read 0 until 1h / (b): `on_next_tick`, the lane pair, `prev_aim`), `list_uid`, the two row flags, the run-file door, the graph refusal, the env-owned copies |
| gates | section 7's reconstruction (vector, list), honesty: the direct checks (list) and the counterfactual family's cases (a)-(e) over `vector` / `list` / `list_mask` / `list_uid` / the row flags, the policy-never-sees-it row's chokepoint, scan (its (a) module list) and `team_rows`, record-outlives, graph refusal, row flags, squash, no raw number, the new functions, the trackers, replay unmoved; `tests/test_team_picture.py` green with the trackers; the 1d pins and `match_pin_check.py` EQUAL (the trackers write no sim state); `tools/rl_smoke.py` on a `"v1"` run, 2 envs; the farm's number as information (section 6) |
| report | the measured ms of its parts; every CHOICE it had to make beyond this document, one line each |

### (b) `192-1g-b-windows`: the two windows, the sight planes, the lanes

| | |
|---|---|
| files | `src/simulation/obs_spec_v1.py` (section 3: the plane readers, the far reduce, the crops, the sight planes, the lanes plane, the unit planes from the whole picture, `route` and the order planes; the `air` block and `felt_temp` of the vector), `src/simulation/gamemap.py` (`wind_load_plane`), `src/simulation/exchange.py` (`felt_temperature` extracted from the heat row, the row calling it), a measurement script for the body imprint (in `tools/` or the scratchpad, named in the report), `src/rl_env.py` (`Layout` keys `near_f` / `near_u` / `far_f` / `far_u` and their `final_*`), `tests/test_obs_spec_v1.py` + `tests/_obs_reference.py` (the plane parts), `tests/test_sight_lines.py` or a new `tests/test_wind_load_plane.py` (the plane equals the footprint read), `CLAUDE.md` (the row's amendment) |
| builds | 3.1-3.9 and 1.3 (with `felt_temp`); the 3.2 imprint measurement |
| gates | the reconstruction (every plane), honesty (the far unit planes, `watched` from recorded centres; the counterfactual family's (a)-(e) extended to `near_u` / `far_u` and the non-physics far planes; the scan's module list extended to its new callees), `felt_temperature` (the heat row's damage unchanged, `match_pin_check.py` EQUAL); `gamemap.py` is in Class-P twins' `files`, so the merge needs `.cuda_green` == HEAD (the Class P row; noted in v2, v1 did not say it); the far reduce, the sight planes equal `has_los`, the lanes plane, `route`'s field, `wind_load_plane`; `pytest tests -q`; the farm's number as information, with the sight planes' part measured at 12 known enemies |
| report | the ms per part of section 6's table; whether 3.5's union sharing is needed; the body-imprint measurement of 3.2 (per plane, the near/far ratio of the largest squashed float16 difference, a body standing in a vent draft and one walking), for the orchestrator to put to Erik |

### (c) `192-1g-c-full-view`: the full view

| | |
|---|---|
| files | `src/simulation/obs_spec_v1.py` (section 5's builders, `FULL_VIEW_NAME`, their own caches), `src/rl_env.py` (the `full_*` keys in the layout and the shared memory, `info["full_view"]` and `info["final_full_view"]` in `step` / `reset`, env-owned; the dead row of 5.6), `tests/test_obs_spec_v1.py` (the full-view, policy-never-sees-it and farm properties), `CLAUDE.md` (the Full view row), `tests/twin_manifest.toml` if a file moved |
| builds | 5.1-5.7 |
| gates | section 7's full view, policy-never-sees-it, the counterfactual family's case (f) (full view on vs off, every observation array), farm; `pytest tests -q`; **the budget of section 6**: the §9.1 before / after reading, in the checkpoint comment; below 70 %, the order of attack of section 6 inside this patch, and if it still fails: commit, report, stop (plan 1.7) |
| report | the final farm numbers under `"crop"` and `"v1"`; the per-part table filled |

1h then writes the columns this document reserves for it: `prev[h]` (1.5),
`on_next_tick` (2.2 col 28), `prev_aim` (col 66), `sees_target` (3.5), and
reads `list_uid` (as an equality key only, 2.1) and the gates of 1e for
its masks. 1i reads `in_match`, `valid` and `bars`, and its trainer test
asserts the policy is fed only through `team_rows` (4.1).

## 9. The CHOICE and STOP register

Every CHOICE above, in one line each, so the critic reads them in one
place; then (v2) the one question left to Erik and the one finding for the
orchestrator. No STOP was found (none in v2 either: every finding of the
information critique was resolved inside the design): the design and the code at `ee25fe98` agree once
1a-1f's facts are known (the sight cache every tick, the picture's seen
rule, the order's absolute `until`, the wind read per footprint).

| where | CHOICE |
|---|---|
| 0.1 | `L_ref` = the `[marine] loadout` weapon's `range_tiles`; `v_ref` = the marine full speed; the reference kit's rof and spread divide every unit's |
| 0.1 | `d_c = ln(1 + d) / ln(1 + L_ref)` |
| 0.2 | the squash: identity on [-1, 1], `sign(x)(1 + ln|x|)` beyond, capped at 4 |
| 0.3 | float16 planes, float32 vector and list, int32 ids, uint8 flags and counts |
| 0.4 | the spec lives in `obs_spec_v1.py`; the accessor's `observe(spec, ...)` is the door; `[obs] spec` selects it, `"crop"` the default; the graph refusal in `load_levels` |
| 1.1 | `omega` over the top speed; `pushed` over the forward fall limit; `effort` over ONE; all nine status kinds in the vector |
| 1.2 | `mags` over the loadout's total; untracked = 1.0 + a flag; kinds stored as row indices |
| 1.5 | `order_left` over the match's `limit_ticks` (an order's own length is not stored); the previous-choice block is six reserved columns 1h fills; team counts over the match rows' `n` and `zombies` |
| 0.5 | (v2) the observation reaches the sim only through `TeamKnowledge` + the `ObsWorld` facade (a run-time wall, not only a scan); caches keyed by their inputs, the full view's separate |
| 1.3 | (v2) `felt_temp` APPENDED as col 96 (no column moves), over `temperature_max`, read through a function extracted from the heat row |
| 1.6 | the motion, last-hit, damage-window and bar trackers are members of `TeamPicture` on slot 11; the damage window is `ticks_per_second` ticks; 1g builds the bar tracker 1i was to build |
| 1.6 | (v2) the object tracker: a per-team counter at the team's first sighting, kept while the object lives (re-seeing it after a gap gives the same id: the team's memory, as a record); the gate runs every tick there |
| 1.6 | (v2) `Record` gains `facing_q` |
| 1.7 | `line_of_fire.py` holds the bots' line geometry verbatim; the bots are not moved onto it in 1g |
| 1.7 | `aim_of`'s cone for a weaponless body (a zombie) is the reference kit's melee arc |
| 1.7 | `route`'s `tile_ok` adds the openable openings' tiles; seeds = passable anchors whose block centre is in the place; the first step by `horde.candidates` |
| 1.7 | `STATUS_SCALE_TICKS`: per kind the longest shipped applier's duration, 1 when none |
| 2.1 | the mate block's cap is `S - 1`; nearest first in every block, ties by id; `list_uid` codes objects by the team's own counter (v2) and doors by ordinal plus a base; a uid is an equality key only |
| 2.2 | (v2) a record's facing is `Record.facing_q`; an object's `side` is -1, its `whose` by the roster; the lane pair per mate from the rasteriser's own per-owner cells |
| 3.1 | outside the map: wall 1, planes 0, route -1 |
| 3.2 | pressure as the profile's band, 0 at the low limit, 1 at the high; the poison quotient times the unit's `resist_mult` |
| 3.4 | `wall` = `solid` (a grille reads as floor; the route plane carries walkability) |
| 3.5 | `sees_target` always 0 in 1g (1h's aim fills it); the sight windows per soldier, the team's union rectangle the fallback |
| 3.6 | the lanes plane uses the bots' `lane_cell` / `lane_halfwidth` |
| 3.7 | a third far unit plane, `records`; (v2) all three drawn from the whole picture, past the list's caps |
| 4.1 | `valid` keeps its name and meaning (= acts); `in_match` is the new flag; (v2) `team_rows(obs, t)` returns views, the one path to a team's rows |
| 5.3 | the full view's four status columns; its order's place and its air are not per-unit columns |
| 5.4 | the balance column is the health sum until 1i replaces the builder |
| 5.2 | (v2) the terminal full view is `info["final_full_view"]`; the full view's object uids are block positions plus `2^20` |
| 5.6 | a dead row's full view is built, centred on the corpse, its first entry itself |
| 6 | the budget: 70 % of today's env-ticks/s at decimation 4, held at (c) |
| 8 (a) | the vector's `air` block and `felt_temp` read 0 until (b) |

**RULED BY ERIK** (v2 put it to him; not a CHOICE -- the lead did not decide it):

| where | question | ruling |
|---|---|---|
| 3.2, 10 | **The fields carry every unit's body** (`stamp_units`): accept the bodies' aerodynamic and radiative imprint in the eight physics planes as an ACCEPTED GAP (proposed: the physics is real), or remove it from the observation (a later spec version). Patch (b) measured it in a vent draft (the table in 3.2) before the question was put. | **ACCEPTED GAP** -- Erik, 2026-11-12: "on the wind question, its OK! We accept it!" |

**For the orchestrator** (findings outside 1g's scope, v2):

| where | finding |
|---|---|
| bots | **The bots know of adjacent enemies through walls** (critique minor 12; the design critique's lens 2 minor 8, still open at `ee25fe98`): `BotTeam.adjacent_enemy` (`control_bots.py:1323-1337`) scans `sim.units` with no line of sight; the soldier has no such knowledge. A PARITY GAP between the opponent and the agent, recorded here, not fixed in 1g (a bot change is measured by the Bot tournament; `ray_clear` in the bots' scan is the one-line remedy). |

## 10. The `CLAUDE.md` rows (written by the patches, from design 10's draft rules)

- **New row "Observation spec"** (Gameplay, after "Batched environment";
  patch (a), amended by (b)): an observation layout is a named, versioned
  spec served by the sensor accessor's `observe` -- `obs_spec_v1.py`'s
  tables are the spec, the run file's `[obs] spec` names it, `"crop"` is
  the placeholder; a new idea is a new spec, never an edit of one a
  trained network uses; every feature is a ratio of a table value, an
  angle's cosine and sine, a flag or a kind, squashed by the one curve;
  units, grenades and charges reach a soldier only through its team's
  picture and a team-mate's line of sight, the map, doors and fields
  unfogged (the fields carry the bodies' own aerodynamic and radiative
  imprint, `stamp_units` -- an ACCEPTED GAP, Erik 2026-11-12, 3.2); the
  builders reach the sim ONLY through the
  `TeamKnowledge` view and its `ObsWorld` facade, never `sim.units` /
  `roster` / `get_unit` / `projectiles` / `planted_charges`, every cache
  keyed by its inputs; an object's id is its team's own counter, its owner
  told by the roster, never a live read; a policy of team t reads only its
  team's rows (`BatchedEnv.team_rows`) and nothing of `info`, `reward`,
  `counters`; the windows never rotate; the sight planes call the many-tiles
  routine, the lanes the picture's `fire_lanes`, the route the horde's
  flood -- nothing walks its own lines or floods its own field; a run on
  this spec refuses a level without a room graph; the two row flags
  `in_match` / `valid`.
- **New row "Full view"** (patch (c), design 10's draft rule): what the
  training-only value network is shown is one named spec (`full_v1`)
  served beside the observation by the same door, under `info`, never a
  key of `obs` (its terminal twin `info["final_full_view"]`); a policy
  never reads it; its builders own their caches, never shared with the
  observation's; a new privileged input is a new version of that spec.
- **Amend "Team picture"**: the five trackers (motion, last hit, damage
  window, bars, the per-team object ids) and the death tick are members of
  the picture on the same slot-11 driver; the observation reads them
  through `TeamKnowledge`, in scope, never memory on demand; `Record`
  carries `facing_q`.
- **Amend "Turning law"** (`ticks_to_within`, the trigger aid: a call of
  the law, never a copy), **"Verb gates"** (`door_in_reach`), **"Wind load
  on a body"** (`wind_load_plane`, the per-tile term of the one read, a
  read-side product), **"Coupling table"** (`exchange.felt_temperature`,
  the heat row's felt temperature as one function the row and the
  observation both call, patch (b)), **"Sight for many tiles"** (the sight planes as its
  user), **"Batched environment"** (`[obs] spec`; the v1 arrays; the two
  row flags; the full view in `info`), **"Vision / engagements / cover"**
  (`line_of_fire.py`: the line-of-fire flags judge a round's path --
  `rays_clear(physical=True)` and live cover -- never sight).
- The header line, each patch.

