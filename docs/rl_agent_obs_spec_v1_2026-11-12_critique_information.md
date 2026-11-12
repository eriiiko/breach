# Critique of the observation spec v1, lens: information (what a soldier may know)

Critic: Opus, 2026-11-12, independent; read against the code at `ee25fe98` (the worktree's `src/` is
unchanged from it). Read-only on code; one headless experiment run (MAJOR 3). The design critique's lens 2
(`docs/rl_agent_design_2026-11-11_critique/lens_2_information.md`) is taken as settled: its Majors 1-4 are
fixed in the code (1d's records, `fire_lanes` from state) or in the spec (3.2's denominators, the
unfogged-fields statement in section 10), and are not repeated here.

## Findings

### MAJOR 1. `list_uid` for grenades and charges is a list INDEX: it counts the unseen objects ahead of it
- Spec: 2.1 "Padding and mask": a type-3 uid is "the projectile's index in `sim.projectiles` plus `2^20` or
  the charge's index plus `2^21`"; 4.1 puts `list_uid` in `obs`.
- Evidence: `sim.projectiles` holds EVERY live grenade of every side in throw order
  (`simulation.py:1527` appends, `:2678` drops the detonated); `sim.planted_charges` every live charge
  in plant order (`charges.py:74` appends, `:134` drops the spent). The gate of 2.1 filters which objects
  become entries, not the index they carry.
- Leak: a seen grenade with uid `2^20 + 3` tells the holder of `obs` that three other grenades (any side,
  seen or not) were thrown before it and are still alive; the same for charges (an enemy's hidden C4 two
  rooms away shifts the uid of mine). The id is also unstable (it changes when an earlier object
  detonates), so it cannot serve 1h as an identity either.
- "Never a feature" is a promise about the network, but `list_uid` sits in the policy's dict, and 7's
  honesty test checks a type-3 uid's TILE, never its value.
- Smallest fix on paper: give objects an id that carries nothing: the entry's position in its block
  (`2^20 + j`, j < 6), or a per-team counter assigned at the team's FIRST sighting of the object (a
  tracker of 1.6, keyed by `id(obj)`), if 1h needs identity across decisions. Add to 7's honesty test:
  "every type-3 / type-4 `list_uid` is a function of the entries the team sees" (the pair test of MAJOR 6
  with an unseen extra grenade catches it).

### MAJOR 2. The object `side` column reads the owner's LIVE team: an unseen conversion flips it, and it names which side threw
- Spec: 2.2 col 8 `side`: "objects: the owner's side (col 56)", a kind = the team id; col 56 `whose`.
- Evidence: `Projectile.owner_id` / `PlantedCharge.owner_id` are unit ids only (`combat.py:111`,
  `charges.py:45`); a team must be looked up on the live owner, and a conversion moves the owner to
  `ZOMBIE_TEAM` while keeping the same object and id (`ai_zombie.py:416-417`: `u.team = ZOMBIE_TEAM` on the corpse itself).
- Leak: an enemy plants a 60 s charge, walks out of sight, is bitten and turns: the charge, still in view,
  changes its `side` from that team's id to 255 -- the team learns of an unseen conversion (the
  exact signal 1d's records were built to withhold). With three or more marine teams, `side` also names
  WHICH team threw a grenade whose thrower nobody saw, which `whose` = "2 another side" deliberately does not.
- Smallest fix on paper: `side` = -1 on objects (as on doors); `whose` decided without a live read:
  0 = `owner_id` is my id, 1 = `owner_id` is in my team's ROSTER ids (fixed at spawn, `Simulation.roster`
  -- the team's own knowledge, read by the row-flag code, not the builder), 2 = any other `owner_id >= 0`,
  3 = -1. Test: the counterfactual pair of MAJOR 6 with an unseen conversion of a charge's owner.

### MAJOR 3. The physics planes are not a function of public state: every body is stamped into the air, and removing one changes the fields across the map
- Spec: 3.2 (the eight physics planes, both windows, unfogged), section 10's row ("the map, doors and
  fields unfogged"). The lens-2 accepted gap was about EVENTS (a door opened, a fire lit) showing through
  walls; nothing in the spec says the fields also encode where the units ARE.
- Evidence: `GameMap.stamp_units` (`gamemap.py:2071-2090`) writes every living body into
  `dyn_permeability` (0.5 on a marine's 3x3, measured), `dyn_wave_absorb` and `dyn_heat_atten_q`, which
  the gas solver and the radiation sweep read (`physics_runner.py:995, 1215, 1293`). Experiment
  (scratchpad `imprint.py`): two `ship_demo` matches, idle teams, no zombies, seed 7, identical except that
  in B one team-1 marine is dead from tick 0; after 300 ticks the planes differ in ~4 700 cells, out to 96
  tiles from the body: wind up to 0.44 m/s (mean 0.08 m/s within 3 tiles of the body, 0.05 beyond 50),
  temperature mean 5.7 game degrees within 3 tiles (max 27), 1.6 beyond 50 (max 50); pressure up to
  0.0015 atm. In this still air the body leaves no clean silhouette in the wind (the field's own noise is
  ~0.1 m/s), so the immediate signal is weak and mostly chaotic divergence; in a vent draft, a breach or a
  blast front a half-permeable 3x3 block casts a wake (UNVERIFIED: not measured in a draft).
- Leak: a hidden unit's presence and movement perturb every field plane of both windows (the far one
  spans 100 tiles); whether a network can localise it is an empirical question nobody has asked.
- Smallest fix on paper: (a) say it: section 10's row and 3.2 add "the fields carry the bodies' own
  aerodynamic and radiative imprint (`stamp_units`): an ACCEPTED GAP, measured"; (b) a measurement in
  (b)'s report: the same A/B with one hidden body standing in a vent draft and one walking, the largest
  squashed float16 difference per plane inside a 25 x 25 window around the body against the same
  statistic far from it; if the near/far ratio is large, the lead decides (coarser quantisation of the
  wind planes, or the far window's physics planes from the block MEAN, or drop the far wind). (c) The pair
  test of section 7 cannot be byte-identical unless the extra unit's room is gas- and heat-isolated from
  every observed window (see MAJOR 6).

### MAJOR 4. The proof obligation (5.7) does not close the wall: its grep names two attributes, and the live-unit reads the spec itself needs go through others
- Spec: 5.7 bullet 2: "A grep test holds that `obs_spec_v1`'s observation functions never read `sim.units`
  or `sim.roster` directly"; builders take `(sim, picture, trackers, team, u)`.
- Evidence and holes:
  1. The spec's own type-1 columns read the LIVE enemy (2.2 cols 10-33: `current_hp`, `exposure`,
     statuses, `live_trigger`, `zombie_state`), and an object's owner (MAJOR 2). With `sim.units` banned,
     the builder reaches them through `sim.get_unit(uid)` (`unit_orders.py:112` shows the idiom) -- which
     the grep does not name, so a builder that calls `sim.get_unit(rec.uid)` on a TYPE-2 record (a
     remembered, possibly dead, possibly converted unit) passes the grep and leaks its liveness, place and
     class.
  2. The grep covers `obs_spec_v1` only. Every helper it calls in another module is unchecked:
     `team_picture.fire_lanes` iterates `sim.units` (`team_picture.py:241`; correct today, filtered to the
     team), `vision.state_for` all units (correct), and the NEW `line_of_fire`, `aim_of`'s helpers and the
     trackers are free to.
  3. The trackers (1.6) are keyed by uid for EVERY unit (`motion` "every living unit", `bars`, `last_hit`,
     the death tick of 5.6) and are read by both builders. A builder that takes a record's velocity from
     `motion[rec.uid]` instead of `Record.vel_q` (2.2 cols 6-7 name both readers in one cell) reads the
     live motion of an unseen unit; the grep cannot see it.
  4. Section 6's order of attack caches `aim_of`, the lanes and the sight windows per TEAM per decision and
     shares `full_watched`'s windows "over the team's union". The full view evaluates `aim_of` "with EVERY
     living unit as a candidate" and `watched` from LIVE centres with "the same builders": a cache keyed by
     `(team, uid)` or by enemy uid, filled by whichever builder runs first, hands the full view's answer to
     the observation. Nothing in 5.7 names caches.
- Smallest fix on paper: replace the grep with a CHOKEPOINT: the observation builders receive a
  `TeamKnowledge` view built once per team per decision -- my living mates (live units), the records, the
  seen-now set as `{uid: live unit}` restricted to `visible_by_team[team]`, the gated objects -- and
  `sim` only through a narrow facade exposing the map, the doors, `tick`, the tables and the sight
  routines (no `units`, `roster`, `get_unit`, `projectiles`, `planted_charges`, `team_picture`). The
  trackers are read through that view too (`motion` only for mates, `bars` / `damage_window` only for
  seen uids). Any cache is keyed by the INPUTS it was computed from (source tile, candidate set's
  identity), never by uid alone, and the full view owns separate caches. The grep test then bans the
  facade's absent names in `obs_spec_v1`, `line_of_fire` and the observation half of `team_picture`.

### MAJOR 5. `obs` hands the trainer every team's rows, and `info` the counters: 5.7 never says what ONE policy may read
- Spec: 4.1: every key is `(N, U, ...)` with `U = T * S`, all marine teams' rows; `valid` / `in_match`
  for all rows; 5.7 proves only "no key of 5.2 is in `obs`".
- Evidence: `rl_env.py:57-61` (row `t * S + s`), `_obs` (`rl_env.py:1213-1220`) returns all rows; the
  step's `reward` / `counters` (`EnvRunner._count`, `rl_env.py:585-612`) credit kills and hits on units the team never saw
  (a frag through a wall).
- Leak: a soldier of team 0 that reads (through a trainer's batching, a shared attention over rows, a
  team-level pooling, or rung 4's commander) any row of team 1 sees team 1's own vector -- its place,
  health, ammo -- and team 1's records of team 0; team 1's `valid` column is a death counter of the
  enemy. A recurrent agent fed its previous reward (a common pattern, and 1.5's "previous choices" block
  is its sibling) learns of every unseen kill from `kills`.
- Smallest fix on paper: add to 5.7: "a policy of team t reads only rows `t*S .. t*S+S-1` of `obs` (its
  own row, and its team-mates' rows only as the radio already shows them) and nothing of `info`, `reward`
  or `counters`; another team's rows are full-view material." Give the env a helper
  `team_rows(obs, t)` and make 1i / rung 2's trainer test assert it is the only path.

### MAJOR 6. The tests would not catch Majors 1-5: the pair test perturbs the one thing no builder reads
- Spec: 7 "Shows nothing the team cannot know": "two sims equal but for one dormant zombie in a sealed
  room no team-mate sees give byte-identical `vector`, `list`, `near_*`, `far_*`".
- Problems:
  1. A dormant zombie in a sealed room exercises none of the leak paths: it is in no record, throws
     nothing, owns nothing, is in no `aim_of` candidate set, and is never `get_unit`-ed. The leaks
     above are all about DYNAMIC hidden state: an unseen death or conversion of a REMEMBERED unit
     (MAJOR 4.1, 4.3), an unseen object before a seen one (MAJOR 1), an unseen conversion of an owner
     (MAJOR 2), and a cache shared with the full view (MAJOR 4.4).
  2. As written it is ill-posed. Added through `spec.zombies` it changes the vector's denominators
     (1.5 cols 82-83 divide by `spec.zombies`) and fails for the wrong reason; added by hand, its body
     still stamps the air (MAJOR 3), so the planes of any window that shares gas or heat with that
     room differ. And spawned before the marines it shifts unit ids.
  3. The record-outlives test checks the type-2 entry's own columns, not that every OTHER array is
     unchanged by the unseen death (counts, `watched`, `aims_at`, the far planes).
- Smallest fix on paper: replace the pair by a COUNTERFACTUAL family, each two sims identical up to a
  tick T and then differing only in hidden state of a unit not in `visible_by_team[team]` since its last
  sighting, the observing team's rows compared byte for byte over the following decisions (only that
  team's rows; the other team legitimately differs): (a) a remembered enemy killed unseen by a status
  vs. left alive; (b) a remembered enemy teleported unseen; (c) a remembered enemy converted unseen;
  (d) an extra grenade thrown by an unseen unit before a grenade the team sees (MAJOR 1); (e) an unseen
  conversion of a visible charge's owner (MAJOR 2); (f) with the full view ON vs OFF (a shared cache,
  MAJOR 4.4). To keep the fields out of it, run the family on a level variant with the physics planes
  zeroed in the comparison (or `[obs]` test hook that masks them) and test the fields separately by
  MAJOR 3's measurement. The 5.7 grep becomes the chokepoint test of MAJOR 4.

## MINOR

1. **The damage window's "saw" rule depends on the order inside the update (1.6).** "The victim's record
   has `tick == now - 1`" is true only if the window is updated BEFORE step (3)'s sightings
   (`team_picture.py:167-169` rewrites a still-seen record to `tick == now`). After them, every hit on a
   continuously seen enemy fails the test and only the team's own hits count. Say "evaluated against the
   records as they stood before this tick's sightings (the seen-death rule's own moment)". Test 7's
   tracker row should include a hit by a third side on an enemy seen at both ticks.
2. **"The first update that sees it" (1.6 `bars`) reads as the first SIGHTING.** Say "the first update
   at which the unit exists in `sim.units`" -- spawn -- so an enemy first seen wounded reads its true ratio.
3. **`aim_of` returns a category but 2.2 col 34 needs the unit.** `mates_aiming` counts mates "whose
   `aim_of` answer names this unit", which a 0..3 answer cannot. Return `(category, uid)` with the uid of a
   record or a mate, never of an unknown unit; state the tie rule for "the nearest" (squared distance, then
   uid).
4. **2.1's object gate does not pin `physical`.** It must be `rays_clear(..., physical=False)` (sight:
   `sight_screen` and `blocks_los` cover, `vision.py:304-316`), the units' own rule; 1.7's
   `line_of_fire` uses `physical=True`, and an implementer copying it would show a grenade through a
   grille the team cannot see units through.
5. **`full_watched` "stored beside the observation's `watched`" (5.5)** can be read as an extra channel of
   `near_u`. Say "its own key of 5.2, in `info['full_view']`, never a channel of an `obs` array".
6. **No terminal full view.** 4.1 gives `final_*` for the observation; 5.2 none for the full view, which a
   truncated episode's value bootstrap needs. Name it now as `info["final_full_view"]`, so nobody adds
   `final_full_*` into `info["final_obs"]` (the policy's terminal dict).
7. **`list_uid` of units links a seen zombie to the marine it was** (a conversion keeps the id) and, by
   value, carries spawn order. Harmless while no network reads it; say in 2.1 that 1h's decoder uses it only
   as an equality key against the list, never as a value.
8. **The 600-tick reconstruction test cannot see leaks** (it compares the builder to a reference written
   from the same document, so a leak written into the spec passes both). It is a correctness gate; the
   honesty gate is MAJOR 6's family. Say so in 7.

## Things the soldier should know and does not (secondary)

9. **What actually burns it.** The heat row burns on the radiation sensor and the `heat` buffer
   (`exchange.py:358-397`); the vector's `air[0]` is the TILE temperature. A soldier cooked by a fire's
   radiation across a room reads a cool tile and only `hit_cause = heat` afterwards. Add one column:
   its felt temperature over `temperature_max` (its own body: no honesty question).
10. **A record's facing.** 2.2 writes (0, 0) because `Record` stores none; the team saw which way the
    enemy faced. One field in `_sighting` (1d's dataclass) gives the soldier "it was looking down this
    corridor when last seen"; cheap, honest.
11. **Records beyond the eighth.** The list keeps the 8 nearest records and the far `records` plane is
    drawn from the list (3.7), so a ninth remembered enemy 30 m away is in no array, while the bots' team
    layer reads every record (the objective's "nearest known", `control_bots.py` `_objective`). Draw the
    far `records` and `enemies` planes from the whole picture, not the capped list (still honest).
12. **The bots know of adjacent enemies through walls.** `BotTeam.adjacent_enemy` (`control_bots.py:1323-1337`)
    scans `sim.units` with no line of sight (lens 2 minor 8, still open at `ee25fe98`); the soldier has no
    such knowledge. Record it as a parity gap in 9, or put `ray_clear` into the bots' scan.

## Checked and found sound

- Type 1 is exactly `visible_by_team[team]` at the picture's tick; the picture and the vision cache are at
  `sim.tick` when `EnvRunner.observe` runs (slot 11 updates after the tick advance,
  `team_picture.py:136-170`; `vision.state_for` keyed by tick, `vision.py:390-398`).
- Records hold plain values (`Record`, `team_picture.py:88-117`), outlive an unseen death and an unseen
  conversion, and are dropped on a seen death only; the spec reads class, side, place and velocity from them.
- `watched` from recorded centres, FROM the enemy's tile (`to_window=False`, the observer->target walk
  `can_see` uses, `vision.py:281-284`); `sees_target` the reverse. Correct direction.
- The fire lanes come from mates' state only (`team_picture.fire_lanes` filters to `team`).
- The route and place planes read the material grid and the door states (`horde.walkable_tiles` is
  material-only, `horde.py:274-279`; `opening_states` keys on doors and wires, `room_graph.py:81-111`): no
  unit enters them.
- The commander's orders are honest: the shared target is chosen among SEEN units at the same tick
  (`control_bots.py` `TeamBrain.update`, `cands = ... if sg.uid in seen_ids`), the objective from records,
  the own spawn room or the enemy's spawn room (public); an order's target is not cleared at the target's
  death (`unit_orders.py` has no such path), so `has_target` / `priority` do not flip on an unseen death.
- The hand-verb gates (1e, `verb_gates.py`) have no target term, so the 1h masks built on them reveal no
  adjacent hidden unit.
- `hit_from` is the hit's own travel vector (`events.py` `UnitHitEvent.dir_x_q`, #185), never the
  attacker's position.
- `whose` (as opposed to `side`) collapses every other side into one code, so an owner's unseen death or
  conversion does not change it (but see MAJOR 2 for the live read it needs).
- No light plane is in the spec (light refused until #81), so the flashlight cones -- the largest
  potential leak in the engine -- stay out.

## Summary

The spec is honest in its feature definitions for UNITS: every enemy reaches the soldier through the team
picture or the seen-now set, records are plain values that outlive an unseen death, the sight planes use
recorded places, and the commander's orders are drawn from what the team saw. The leaks are at the edges
of that design: object ids that are list indices (MAJOR 1) and an object `side` read off the live owner
(MAJOR 2); the fields, which the spec treats as public but which carry every body's imprint (MAJOR 3,
measured map-wide, magnitude in a draft unmeasured); and a wall whose proof is a two-name grep that the
spec's own live-unit reads, the all-unit trackers and section 6's planned caches walk around (MAJOR 4),
with the other teams' rows and the counters left outside the obligation altogether (MAJOR 5). The tests
as written would catch none of these, because the one counterfactual they run perturbs a unit no builder
ever reads (MAJOR 6). Each fix is a definition, a chokepoint or a test on paper; none changes the design.

**Count: BLOCKER 0, MAJOR 6, MINOR 12** (8 minors on the text and tests, 4 on what the soldier lacks).

## Resolution (2026-11-12, spec v2)

Folded into `docs/rl_agent_obs_spec_v1_2026-11-12.md` (same file name, a v2 changelog at its top),
commits `e9aadccf` and `1b0df6e8`.

| finding | spec section changed | how |
|---|---|---|
| MAJOR 1 | 1.6 (object tracker), 2.1, 7 | the team's own object id `2^20 + k`, counted at the team's first sighting, kept while the object lives |
| MAJOR 2 | 0.5 (`roster_ids`), 2.2 cols 8, 56, 7 (e) | `side` -1 on objects; `whose` by my id / my roster ids, no live read |
| MAJOR 3 | 3.2, 8 (b), 9 (PENDING ERIK), 10, 7 | PENDING ERIK: proposed an ACCEPTED GAP; (b) measures the imprint in a vent draft; the pair tests exclude the physics planes except (f) |
| MAJOR 4 | 0.5 (new), 1.7 `aim_of`, 3.6, 5.7, 6, 7 | `TeamKnowledge` + the `ObsWorld` facade as the only way in; trackers in scope; caches keyed by inputs, the full view's separate; the scan widened to every callee |
| MAJOR 5 | 4.1, 5.7, 7, 8 | a team-t policy reads only team t's rows, nothing of `info` / `reward` / `counters`; `BatchedEnv.team_rows` |
| MAJOR 6 | 7, 8 | the counterfactual family (a)-(f), same match.toml and seed, the observing team's rows only |
| minor 1 | 1.6, 7 | the saw rule against the records before this tick's sightings; third-side hit tested |
| minor 2 | 1.6, 7 | `bars` at spawn |
| minor 3 | 1.7, 2.2, 7 | `aim_of` returns `(category, uid)`, ties by distance then uid |
| minor 4 | 2.1 | `physical=False` |
| minor 5 | 5.5 | its own key of `info["full_view"]` |
| minor 6 | 5.2, 7 | `info["final_full_view"]` |
| minor 7 | 2.1 | a uid is an equality key only |
| minor 8 | 7 | the reconstruction named a correctness gate |
| minor 9 | 1.3, 4.1, 8 (b), 10 | vector col 96 `felt_temp` via an extracted `exchange.felt_temperature`; V = 97 |
| minor 10 | 1.6, 2.2 | `Record.facing_q` |
| minor 11 | 3.7, 7 | far unit planes from the whole picture |
| minor 12 | 9 | recorded as a parity gap for the orchestrator, not fixed in 1g |
