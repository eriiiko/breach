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

## 1. The soldier's vector (design 3.2, 4.4, 5.2; 96 float32)

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
| 58..65 | `air[p]`, p over the eight physics planes of 2.2 in their order (temperature, fire, poison, teargas, smoke, pressure, wind_x, wind_y) | the near window's plane value at the soldier's centre tile (built by the same plane builders, 2.2; the unit's own profile for the per-unit denominators) | as 2.2 | as 2.2 |

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
| `motion[uid]` | `(pos_q_prev, pos_q_now)`, Q16 tiles (`unit_fixed.quantize_scalar(u.x)`, as `Record.pos_q`); the velocity is their difference, `(0, 0)` on the first update | every living unit, every update | 1.1 `vel`, the mate entries' `vel` (an enemy entry's velocity is its record's `vel_q`, `team_picture.py:103`) |
| `last_hit[uid]` | `(tick, dir_x_q, dir_y_q, cause)` of the last `UnitHitEvent` with `damage > 0` on `uid` in the step's `sim.tick_events` | at the update, events in emission order (the last wins) | 1.4 |
| `damage_window[team][uid]` | a deque of `(tick, damage)` of hits on `uid` the team DEALT (the event's `attacker_id` on `team`) or SAW (the victim's record has `tick == now - 1`, the picture's own seen rule, `team_picture.py:157-162`); entries older than `WINDOW_TICKS = ticks_per_second` dropped (CHOICE: the design's "last second", quantized once) | at the update | the enemy entries' `dmg_last_s` (2.2) |
| `bars[uid]` | the unit's `current_hp` at the first update that sees it (spawn) and again at the update that first sees `is_zombie` true for it (the conversion ran in `on_tick_end` before slot 11, `simulation.py:2419-2426`; design 6.1 "health at spawn or at turning") | at the update | every `hp` ratio and `dmg_last_s`; patch 1i's bar tracker reads this one (CHOICE: 1g builds it because the hp ratio needs it first; 1i's note takes it over) |

### 1.7 New read-only functions (design 3.7 names them; none exists today)

| function | where | what |
|---|---|---|
| `turning.ticks_to_within(unit, target_q, a, w, dw, tol_q, cap)` | NEW in `src/simulation/turning.py` beside `predict` (`:199`) | the ticks until `abs(wrap(target - facing)) <= tol_q`, by iterating `law_q16` (`:140`) from the unit's facing and `omega_q + dw` -- never a copy of the law; `cap` ticks at most (returns `cap` when not reached). The trigger aid of 2.2. |
| `GameMap.wind_load_plane()` | NEW in `src/simulation/gamemap.py` beside `footprint_wind` (`:2405`) | the per-tile term of that read, `W_i = N_raw * ((wind_x^2 + wind_y^2) >> 16)` over gas tiles (0 on solid), with `(wind_x, wind_y)` -- as float64 arrays (a read-side product like the crop, `sensor_accessor.py:123`; the int product overflows int64). The wind planes of 2.2. |
| `line_of_fire.solid_in_way(sim, a, b)` and `mate_in_way(a, b, mates, clear_tiles, fp)` | NEW module `src/simulation/line_of_fire.py` | as a round flies (the critique's minor 7): `solid_in_way` = `not vision.rays_clear(sim, [(ay, ax, by, bx)], physical=True)[0]` (`vision.py:195`) or a live cover entity's `blocks_segment` (every live cover, as the march stops on any, `combat.py:381-387`); `mate_in_way` = the bots' geometry verbatim (`control_bots.BotTeam._teammate_on_line`, `:1199-1223`, with `[bots] line_clear_tiles` `config.toml:4959`), a module function; held equal to the bots' method by a test; the bots are NOT moved onto it in 1g (a one-line finding). |
| `obs_spec_v1.aim_of(sim, picture, team, u, mates, known)` | NEW in `obs_spec_v1.py` | the 4.1 rule judged by MY team's knowledge: among the units my team knows (my living team-mates at their live centres; the picture's records at their recorded `centre`), the nearest one, other than `u`, whose bearing from `u` is within `u`'s cone of `u`'s facing ray and to which `vision.ray_clear` from `u`'s centre tile holds; answers 0 nobody, 1 me, 2 a team-mate, 3 a known unit of another side. The facing ray is `facing_q` (for a team-mate `facing_target_q` instead, design 4.1); the cone is the unit's weapon `spread_deg` (its `melee_arc_half_q`, `weapons.py:213`, when it has no ranged weapon; CHOICE for a zombie: the reference kit's melee row `[weapons.rifle_butt] melee_arc_half_deg` `config.toml:3494`, the one table cone a weaponless body gets). An unknown unit never answers. |
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
| grenades and charges | 3 | 6 | every live `Projectile` of `sim.projectiles` not `detonated` (`combat.py:94-124`, `simulation.py:381`) and every live `PlantedCharge` of `sim.planted_charges` (`src/simulation/charges.py:35-52`, `simulation.py:508`) whose tile some living unit of my team (me included) has `vision.rays_clear` to (one call per team per decision over all (mate centre, object tile) pairs) | a team-mate has a line of sight to its tile (design 3.3; an enemy's timer is shown, Erik's ruling) |
| doors | 4 | 4 | every living `DoorRuntime` of `sim._doors` (`src/simulation/door_system.py:123`; `alive`, `state`, `span`, `ordinal` as `room_graph.py:91-102` reads them) | doors are shown unfogged (design 3.1) |
| reserved | 5 downed team-mate, 6 bag, 7 button, 8 noise heard | 0 | nothing in v1 (design 3.3) | |

- **Order:** the blocks in the table's order, fixed widths `[S-1, 16, 8, 6,
  4]`, total `E = S + 33` entries (36 at S = 4). Within a block NEAREST
  FIRST by the squared centre distance (a door: its span's centre tile),
  ties by the lower id (a door: its ordinal); re-sorted at every decision
  (design 3.2). A block with more candidates than its cap keeps the nearest.
- **Padding and mask:** an empty slot is all zeros, its `mask` 0 and its
  `uid` -1; the network never reads an unmasked slot as a thing (design
  3.1). `uid` (int32) is the game's unit id for types 0-2, the projectile's
  index in `sim.projectiles` plus `2^20` or the charge's index plus `2^21`
  for type 3, the door's ordinal plus `2^22` for type 4 -- for the decoder
  and the aim mask (1h) only; never a feature.
- **The previous aim mark** (col 66) is written by 1h's decoder (the held
  uid); 0 in 1g.

### 2.2 The entry (67 float32 columns; a column not applicable to the type is 0)

Common columns (every type):

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 0 | `type` | the block | kind (0..8) | no |
| 1, 2 | `dx`, `dy` | the entry's centre tile minus mine (a record: `Record.centre`, `team_picture.py:101`; a projectile: `int(fx), int(fy)`; a charge: `(x, y)`; a door: its span's centre) | `L_ref` | yes |
| 3 | `dist_c` | that offset's length | `d_c` | no |
| 4, 5 | `cos`, `sin` | a unit's `facing_q` (a seen unit: live; a record: 0, 0 -- the record stores no facing, CHOICE; objects: 0, 0) | angle | no |
| 6, 7 | `vel_x`, `vel_y` | a mate: `motion` (1.6); a seen enemy / a record: `Record.vel_q` (`:103`); a grenade in flight: `(target - start) / dist * travel_speed / tps` tiles per tick (`Projectile.update_position`'s own law, `combat.py:139-157`), `(0, 0)` once landed; a charge, a door: 0 | `v_ref` | yes |
| 8 | `side` | the unit's team as seen (`Record.team`, a mate's `Unit.team`; the horde reads `factions.ZOMBIE_TEAM`); objects: the owner's side (col 56) ; a door: -1 | kind (team id; 255 the horde) | no |
| 9 | `class` | `Record.loadout` / `Unit.loadout_name` as its `[loadouts.*]` row index; a zombie: `len(loadouts)` (one code past the rows: the zombie class); objects, doors: -1 | kind | no |

Unit columns (types 0, 1; a record (2) shows only what it holds: cols 0-3, 6-9, 54):

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
| 24 | `seen_by_me` | my id in `vision.state_for(sim).observers_of(uid)` (`vision.py:266`) | flag | no |
| 25 | `lof_solid` | `line_of_fire.solid_in_way(sim, my centre, its centre)` (1.7) | flag (1 = something solid in the way) | no |
| 26 | `lof_mate` | `line_of_fire.mate_in_way(my centre, its centre, my living mates, line_clear_tiles, fp)` | flag | no |
| 27 | `ticks_to_on` | `turning.ticks_to_within(me, atan2_q16(-dy_q, dx_q) of the offset to its centre, a, w, unit_impulse.pending_dw(table, me), spread_q, cap)` (1.7; `a`, `w` = `Simulation.unit_turn_accel_q16 / unit_turn_rate_q16`, `simulation.py:1053-1065`; `spread_q = quantize(radians(weapon.spread_deg))`, the bots' `_weapon_spread_q` `control_bots.py:1188`; `cap = 4 * HALF_TURN_TICKS`) | `HALF_TURN_TICKS = ceil(PI_Q16 / w)` (the half turn at top speed; 12 ticks at the shipped dials) | yes (reads 4 at the cap) |
| 28 | `on_next_tick` | for the entry that is my current aim target only (1h's held uid; 0 in 1g): `abs(wrap(want - Simulation.facing_after_turn_q16(me, dx_q, dy_q))) <= spread_q` (`simulation.py:1067`; the bots' `_on_target`, `control_bots.py:1180-1186`) | flag | no |
| 29 | `in_range` | squared centre distance `<= range_tiles^2` of my weapon | flag | no |
| 30 | `in_throw_range` | `<= max_throw_range^2` (`[weapons.hand_grenade] max_throw_range` `config.toml:3312`) | flag | no |
| 31 | `facing_on_me` | `aim_of(...) == 1` for this unit | flag | no |
| 32 | `aims_at` | `aim_of(sim, picture, team, it, ...)` (1.7; design 4.1 / 4a) | kind (0 nobody, 1 me, 2 a team-mate, 3 a known unit of another side) | no |
| 33 | `zombie_state` | a seen zombie: 1 dormant, 2 hunting (`Unit.zombie_activated`, `unit_schema.py:541`), 3 walking to a noise (`zombie_noise_ticks > 0`, `:545`; `ai_zombie.py:346-369`); a marine or a record: 0 (design 4a; the critique's minor 3 is an accepted gap the design kept) | kind | no |

The team display on an enemy (type 1; design 4.3):

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 34 | `mates_aiming` | living mates whose `aim_of` answer names this unit (the 4.1 rule evaluated per mate; me excluded) | `n_t` | no |
| 35 | `mates_firing` | those with `live_trigger` | `n_t` | no |
| 36 | `dmg_last_s` | `damage_window[team][uid]`'s sum (1.6) | `bars[uid]` (bars) | yes |
| 37 | `priority` | `uid == my Unit.order_target` (`unit_schema.py:558`; design 5.2) | flag | no |
| 38 | `frag_catch` | living mates (me included) whose centre is within the frag row's blast radius of this unit's centre, walls ignored (design 3.3 accepted gap): `WeaponsTables.payload_for_ammo(FRAG_ROW).radius` (`weapons.py:1123`, `:71`; `[payloads.frag_standard] radius` `config.toml:4061`) | `n_t` | no |

The team display on a team-mate (type 0; design 4.2):

| pos | name | read from | divided by | squash |
|---|---|---|---|---|
| 39, 40 | `aim_cos`, `aim_sin` | `cos / sin(Unit.facing_target_q)` -- the point it is turning to (`unit_schema.py:495`; the live aim becomes the target at slot 1a, `turning.py:213-215`) | angle | no |
| 41 | `aim_is_unit` | `aim_of(it, ...)` != 0 (with its facing target in place of its facing) | flag | no |
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
| 56 | `whose` | `owner_id` (`combat.py:111`, `charges.py:45`): 0 me, 1 a team-mate (on my team), 2 another side, 3 nobody (-1) | kind | no |
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

<!-- continued -->
