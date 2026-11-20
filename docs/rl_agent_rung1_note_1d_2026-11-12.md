# Rung 1, patch 1d: the team picture, in two steps (the note)

2026-11-12. Written for the orchestrator of rung 1 (#195) from the plan
(`docs/rl_agent_rung1_plan_2026-11-11.md` §5 "1d"), the design
(`docs/rl_agent_design_2026-11-11.md` 3.3, 3.6, 3.7, 4.1, 4.5, 9, 10), the
roadmap's "Team picture" row and the critique (lens 2 M1, M2; lens 3 m1, m8),
checked against the code at main `51023a02`. Same shape as the plan's 4.1-4.3.
**CHOICE:** marks where the design is silent; **STOP:** would mark a
design/code contradiction (there is none; see the end).

**Design:** 3.6 (records outlive an unseen death), 4.5 (lanes from state),
9 (the picture is not unit state). **Why now:** 1g's list and lanes plane read
it, 1h's aim mask reads "in the team picture"; the bots must read the same
picture the agents will (roadmap rule: never a second picture).

## What the code does today (the facts the build rests on)

- The picture is controller memory: `TeamBrain.picture` (`src/control_bots.py:308`),
  `Sighting` holds the LIVE `Unit` (`:249-254`). `TeamBrain.observe`
  (`:325-337`) runs in `BotTeam.handle_frame` (`:919-932`), i.e. at tick T
  BEFORE `sim.step()` (`src/simulation/match_session.py:97-108`), only while the
  team has a living marine. It writes every id in
  `vis.visible_by_team[team]` (tile, centre, tick), then deletes a record whose
  unit is dead, on the bot's own team, or older than `memory_ticks`
  (`max(1, round(memory_s * tps))`, `:305`; `[bots] memory_s` 6.0,
  `config.toml:4943`, door `_SCALARS` `:103`).
- Live reads through `sg.unit`: the class in `known(zombies=)` (`:339-346`),
  the id everywhere, the shared target's candidates (`:389`, seen units only).
  Other readers: `position_context` (`:1032-1042`), `threat_of` (`:1066-1079`),
  `frag_point` (`:1273`, seen only), `flash_point` (`:1312-1317`, unseen only).
- **A zombie's kill never empties a record:** `convert_marines_to_zombies`
  runs in `ContinuousRealtime.on_tick_end` (`src/simulation/ruleset.py:275-280`,
  inherited by `MatchRules`), inside the same `step()` as the death, so no
  observe ever sees that unit dead: its record survives and its class flips
  to zombie through the live `is_zombie` (lens 2 M1's "and its conversion").
  Every other death drops the record at once, seen or not.
- Lanes: `TeamBrain._lanes` (`:445-472`) rasterises shooter-centre to
  LIVE-target-centre segments from `TeamBrain.firing`, which `handle_frame`
  fills with whatever each bot's hands action returned last tick (`:936-944`;
  `shoot` returns its target even with the trigger released, `:1107-1119`).
  Drawn at each team update (every `team_period_ticks`).
- The sight cache `vision.state_for` (`src/simulation/vision.py:387-396`) is
  keyed by `sim.tick` only, and nothing inside `step()` calls it (only
  controllers and the renderer's fog: `simulation.py:1715-1727`). So computing
  it after the tick counter advances gives exactly what the next tick's
  controllers would have computed.
- **The plan's "bots has no frozen reference match" is half true:** ship_demo
  seed 1 IS a `bots` pin (`tests/test_zombie_control.py:188`,
  `_PRE_122_SHIP_DEMO`, re-pinned at G8p; `tools/match_pin_check.py` plays it).
  One seed, one ship, cone only: not enough, so step (i) takes more.

## Build

### 1. Where it lives, who drives it

- **Module** `src/simulation/team_picture.py` (host Python, sim side so the
  conductor can drive it; reads `vision`, never imports a controller):
  `Record`, `TeamPicture`, the door `picture_settings(cfg)` for a new
  `[team_picture] memory_s = 6.0` (moved from `[bots]`, same value, same
  quantization `max(1, int(round(memory_s * tps)))`; `memory_s` leaves
  `_SCALARS` and `config.toml [bots]`; `flash_memory_s` stays the bots').
  CHOICE: its own table, because the picture is no longer the bots'.
- **Switched on only for a match:** `Simulation.team_picture = None` beside
  `_vision_cache` (`simulation.py:473-479`); `Simulation.start_team_picture(teams)`
  builds it and takes its first update at once. `simulation.match.spawn_match`
  calls it as its last line before `return spawned` (`src/simulation/match.py:368`)
  with `range(len(spec.teams))`: one picture per marine team of the spec; the
  horde gets none (nothing reads what a zombie sees). Every match path spawns
  there (`MatchSession`, `control_match.py:72`, the replay, the farm). Hand-built
  test sims call `start_team_picture` themselves.
- **Driver, one call line:** the conductor, a new slot 11 at the very end of
  `_step_post_physics`, after `self.ruleset.on_tick_end(self)`
  (`simulation.py:2355`): `if self.team_picture is not None:
  self.team_picture.update(self)`. After `on_tick_end` so a conversion has
  happened (as today's observe sees it); after the tick advance so the
  records carry the tick the next controllers run at. Covers `step()` and the
  `ResidentBatch` (it calls `_step_post_physics`). A paused step does nothing,
  so nothing is missed.
- **Strict, never on demand:** `update` is a no-op when already current and
  raises when `sim.tick` is not its last tick + 1 (a missed or doubled tick is
  the bug). Readers call `records(team)`, which raises unless the picture is at
  `sim.tick`. Not synced, not hashed, not in any digest or recording; it is
  rebuilt by re-simulation only (design 9).
- **Readers:** the bots (below); later 1g through the sensor accessor, 1h's
  aim mask. `records(team)` returns a read-only mapping uid -> `Record`; a
  `known(team, zombies=None)` helper returns them in uid order.

### 2. What a record holds (frozen, slotted dataclass; no live object)

| field | type | value when written (the team sees the unit this tick) |
|---|---|---|
| `uid` | int | the unit's id |
| `team` | int | its side then (`u.team`) |
| `zombie` | bool | its class then (`u.is_zombie`) |
| `loadout` | str | `u.loadout_name or ""` (CHOICE: the class a 1g kind vector needs) |
| `tile` | (int, int) | top-left `(u.tile_x, u.tile_y)` (today's `Sighting.tile`) |
| `centre` | (int, int) | `(u.center_tile_x(), u.center_tile_y())` (today's) |
| `pos_q` | (int, int) | `unit_fixed.quantize_scalar(u.x)`, `(u.y)`: Q16.16 tiles, exact (G5p-a2's grid) |
| `vel_q` | (int, int) | motion at the sighting, Q16 tiles per tick: `pos_q` minus the previous record's `pos_q` when that record's `tick == now - 1`, else `(0, 0)` |
| `tick` | int | `sim.tick` at the update |

CHOICE (motion): a one-tick displacement between two consecutive sightings,
from the team's own two observations, never a later tick; a first sighting
moves at 0. No extra store: the previous record is the memory.

### 3. Step (i): the bots read the picture, today's exact meaning

- `TeamPicture.update` in step (i) mode, per team in team order: write a
  record for every id in `vis.visible_by_team[team]` (uid order); then for
  each record (uid order) look the unit up (one `{u.id: u}` map per update,
  not `sim.get_unit`'s linear scan), **refresh `zombie` and `team` from the
  live unit** (today's live class read; this is what reproduces the
  conversion flip) and delete it if the unit is dead, on this team, or
  `tick - rec.tick > memory_ticks`. This refresh and the alive check are step
  (i) only and are deleted in step (ii); no switch survives the patch.
- Bots: `TeamBrain.observe` becomes one line, `self.picture =
  sim.team_picture.records(self.team)`; `Sighting` is deleted; `sg.unit.id`
  -> `sg.uid`, `sg.unit.is_zombie` -> `sg.zombie`; the target's candidates
  (`:389`) are the seen units looked up by id (seen = alive, so honest). A
  `BotTeam` on a sim with no picture raises, naming `start_team_picture`.
- Lift the rasteriser out of `_lanes` (`:445-472`) unchanged as
  `team_picture.draw_lanes(h, w, segments, cell, halfwidth)`, `segments` =
  `(owner_id, (sx, sy), (ex, ey))` in owner-id order, the grid
  `(h // cell + 1, w // cell + 1)`; `_lanes` feeds it today's segments from
  `firing`. This is "the one drawing routine" of design 4.5 / lens 3 m1.
- `tests/test_bots.py` `_know` / the lane test (`:136-140`, `:254-257`) move
  to the picture's own writer (`TeamPicture._write`) and to `draw_lanes`.

**Gate (i), bit for bit.** Commit A, before any code: `tools/bot_tournament.py`
gains `--sight cone|all_round` (`replace(spec, sight=...)` in `play`,
`:136-166`) and records `field_digest` beside `tick_digest`; measurement only.
Then, on commit A (main's game, unchanged), with `PY` =
`C:/Users/steen/anaconda3/envs/data/python.exe` and `L` = `C:/tmp/breach_1d/levels`
(outside the repo; never committed):

```
$PY tools/gen_ship.py --seed 28 --size medium --recipe freighter --profile standard --out $L/ship_freighter_standard_s28_medium
cp -r levels/ship_demo $L/ship_demo
$PY tools/bot_tournament.py --levels ship_demo,ship_freighter_standard_s28_medium --levels-dir $L \
    --a bots --b bots --horde --seeds 1-5 --sight cone --workers 10 --json docs/rl_agent_rung1_1d_pins/cone.json
(the same with --sight all_round ... all_round.json)
```

That is 30 matches per rule (bots vs bots, and bots vs idle on both sides,
the horde in; the freighter is G8p's tournament ship). Commit the two JSON
files on the branch before the code (they prove the order). After step (i):
the same runs to a scratch path; for every (level, controllers, seed) the
five keys `outcome, team, ticks, field_digest, tick_digest` equal (a 10-line
comparison; `wall` is ignored). Plus: `pytest tests -q` whole,
`tests/test_zombie_control.py::test_the_controller_plays_the_pinned_ship_demo_match`
and `tests/test_bots_v1.py::test_bots_v1_play_the_pre_g8p_pinned_match`
unchanged, `tools/match_pin_check.py` EQUAL.

### 4. Step (ii): honest records, lanes from state; the bots on both

- **Records outlive an unseen death.** `update` order per team: (1) drop
  the record of every victim of this step's `UnitKilledEvent`s
  (`sim.tick_events`) whose record has `tick == sim.tick - 1` (the team saw
  it at the observation just before the step it died in: a seen death);
  (2) drop records older than `memory_ticks`; (3) write this tick's
  sightings (a converted unit seen again gets a fresh zombie record). No
  live lookup remains: class, side and place stay as seen. CHOICE: "the
  team saw the death" = seen at the last observation before it; a kill the
  team scored on an unseen unit (a frag) is not a seen death.
- **Lanes from state:** `team_picture.fire_lanes(sim, team, cell, halfwidth)`,
  read-only, no memory: for each living non-zombie unit of `team` (id order)
  whose `live_trigger` is held (`simulation.py:1057`) and whose weapon row has
  `range_tiles > 0`: start S = its centre tile; end E = S + `range_tiles`
  along `facing_q` (`unit_fixed.cos_q16` / `sin_q16`, `shr_trunc(.., 16)`, y
  negated: the grid is y-down); clipped to the last tile before the first
  `gmap.solid` (or off-map) tile on a per-tile walk S -> E (`n = max(|dx|,
  |dy|)`, samples `S + d*i // n`); then `draw_lanes`. CHOICE: the clip walk
  is a drawing step on a widened coarse grid, not a line-of-fire test (1g's
  line-of-fire flags use 1c's routine with `physical=True`).
- **Bots:** `TeamBrain.update` draws `fire_lanes(sim, self.team, lane_cell,
  lane_halfwidth)` at the team period (CHOICE: today's cadence); `firing`,
  `brain.firing` and `_lanes` are deleted. No `[bots]` weight is re-tuned in
  1d (CHOICE: a tuning pass would be its own bot change).
- `bots_v1` is not touched in either step: `git diff` of
  `src/control_bots_v1.py` empty, its pin green.
- **Re-pin** `_PRE_122_SHIP_DEMO` once (`tests/test_zombie_control.py`), with
  the house's rationale lines (sole cause: the honest picture and state
  lanes; step (i)'s last commit reproduces the old pin);
  `match_pin_check.py` follows it (and run it on cuda / resident if built).

**Gate (ii), the tournament** (Erik blessed it: it merges on the numbers).
"Before" = step (i)'s last commit, "after" = step (ii)'s, same commands, for
`S` in `cone`, `all_round` and seed sets `1-20` (tuning) and `101-130`
(held-out, G8p's):

```
$PY tools/bot_tournament.py --levels ship_demo,ship_freighter_standard_s28_medium --levels-dir $L \
    --a bots --b bots_v1 --horde --seeds <set> --sight <S> --workers 12 --json C:/tmp/breach_1d/t_<before|after>_<S>_<set>.json
```

Eight runs, about 30 CPU-hours (G8p's per-match wall clocks); start the
"before" runs as soon as step (i) is green. The checkpoint comment gives,
per rule, seed set, level and pairing, before -> after: `bots` win rate of
the decided duels, survivors against the horde, rounds per kill, FF hits per
match, ticks, wall per match (the wall also shows the cost of the
conductor's every-tick sight cache in `bots_v1` / `idle` matches). CHOICE,
the floor: if on the held-out seeds under cone `bots` falls below G8p's own
gate on either ship (>= 65 % of decided duels against `bots_v1`, fewer FF
hits per match than `bots_v1`, no fewer survivors against the horde), the
implementer stops and reports instead of merging.

## Property tests (`tests/test_team_picture.py` unless named)

| property | breaks if |
|---|---|
| after every step of a match `team_picture` is at `sim.tick`, updated exactly once per step (a counting wrapper) | the slot is missing, doubled, or placed before the tick advance |
| a sim that never spawned a match has `team_picture is None` and the conductor touches nothing; `bots_v1` and idle-vs-idle matches hash as before (the existing pins) | the picture runs on a game path, or its update writes sim state |
| `update` raises on a skipped tick; `records` raises when stale | a reader builds it on demand |
| every `Record` field is an int, bool, str or tuple of ints | a live `Unit` (or any object) is stored |
| a record equals the unit's state at its sighting tick; after the unit leaves sight and walks, the record does not move | a reader or the update reads the live position |
| `vel_q` = the displacement between two consecutive sightings; `(0, 0)` on a first sighting and after a gap | motion is taken across an unseen tick |
| a record is kept at `tick - rec.tick == memory_ticks` and gone at `+ 1` | an off-by-one in expiry |
| every record ever held names a unit that was in `visible_by_team[team]` at its `tick` (a whole match) | the picture shows a unit the team never saw |
| (ii) an unseen death keeps the record its full memory time; a seen death drops it on that update | death is detected by liveness |
| (ii) an unseen conversion keeps the marine record; seen again, the record is the zombie's | the class is read live |
| (ii) a unit with the trigger held facing east owns the lane cells along its facing to its range; a wall tile ends the lane; trigger released: no lane | lanes read anything but state |
| (ii) an `idle` team whose unit's trigger is set by `set_trigger` gets the same lanes as a bot in the same state | lanes read controller memory |
| `draw_lanes` is independent of segment order; two owners' shared cells are -2 | the routine's -1 / -2 / uid law changes |
| the gate (i) pins and the gate (ii) runs above | |

## `CLAUDE.md` (in the patch)

- **New row "Team picture"** (Gameplay, after "Bot layers"), from the
  roadmap's draft: what a team knows of the enemy lives in ONE place,
  `simulation/team_picture.py`, switched on by `spawn_match`
  (`start_team_picture`), updated by the conductor once per tick (slot 11),
  never on demand, not synced, not hashed; a record is the sighting's
  place, class, side, motion and tick, never a live unit, kept
  `[team_picture] memory_s` unless the team saw the death; the fire lanes are
  `fire_lanes` from state through the one `draw_lanes`; the bots, the
  agents' observation and the screen read it, never a second picture.
- **Amend "Bot layers":** the TEAM layer READS the shared picture (no longer
  owns it; `memory_s` moved) and draws the lanes from state.
- **Amend "Tick conductor":** slot 11, the team picture (match only).
- **Amend "Bot tournament":** `--sight`, `field_digest` in the records.
- Header line. `tests/twin_manifest.toml`: a `[not_twinned]` row for
  `src/simulation/team_picture.py` ("host-side match memory; the device's
  own arrays at the residency arc's later stage, design 9"), regenerated by
  `tools/twin_manifest_gen.py`; `tests/test_class_p_level.py` /
  `tools/twin_debt.py` decide whether `cuda_green` is needed (expected not:
  no device file, no build path).

## Not in 1d

The two other trackers design 9 assigns to the Team picture system (the
direction and age of a unit's last hit; the damage it took in the last
second, counting only hits the team dealt or saw) are added by 1g as members
of this module, on the same driver (CHOICE: nothing reads them before 1g and
1d's gate cannot test them). The shared target, the play, the halves and the
objective stay in `TeamBrain` (1f moves target and play into the order).

## One branch, two merges (recommended)

One branch `192-1d-team-picture`, one implementer, two `--no-ff` merges:
step (i) merges when its bit-for-bit gate is green; step (ii) continues on
the same branch (main merged in) and merges on the tournament. Reasons: (ii)'s
"before" is (i)'s last commit and both use the same scratch levels and pins;
a refactor proven bit for bit is a known-good point on main that 1g can build
on even if (ii) stops at its floor.

## STOP

None found. The design and the code agree once the conversion fact above is
known; the one correction (ship_demo seed 1 already pins `bots`) changes the
gate's size, not the design.
