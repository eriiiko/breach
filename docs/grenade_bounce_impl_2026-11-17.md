# A thrown grenade bounces off walls (#215) -- as built

2026-11-17. Erik: "grenades should not pass through walls"; he approved the
bounce rule ("The grenade solution sounds good"). Branch `215-grenade-bounce`
off main `5671af04`. The 2c finding it answers:
`docs/rl_agent_rung2_report_2c_2026-11-17.md` section 7, point 1.

## What was built

- **The flight is a chain of legs** on `combat.Projectile`. `start_*_q` /
  `target_*_q` are the CURRENT leg (the throw's until the first bounce),
  `leg_tick` its first tick, `bounces` the legs a wall has started.
  `update_position` is the old arithmetic on the current leg (elapsed counted
  from `leg_tick`), so the first leg is the throw bit for bit.
- **The wall walk** `combat.flight_walk(gmap, ax, ay, bx, by)`: the Q16
  segment walked tile by tile (Amanatides and Woo's grid traversal, cited in
  the code), the next boundary chosen by comparing the two crossing
  parameters as cross products, equal parameters an exact corner (both axes
  at once). Only tiles the segment ENTERS are asked; the start tile never is.
  The stop is the last point in the free tile (`edge - 1` moving up, `edge`
  moving down; the other coordinate on the segment, rounded half away,
  clamped to the tile) and the flips (vertical edge: x; horizontal: y;
  corner: both).
- **The blocking read** `combat.flight_blocked`: the rounds' march's own stop
  read in `BulletInFlight.advance` -- off the grid, or `gmap.solid`. Not
  `has_los` (that is sight, and the sight rule keeps sight and flight apart);
  not the leg's target alone. A closed door (`door_closed`), glass, the
  legacy painted door (`door`, `MAT_DOOR`) and every wall are solid; a grille
  is not (it lets a round through too).
- **The bounce** `Projectile.bounce`: the remaining distance
  `|target - p|` (isqrt) times `bounce_restitution_q`, rounded once
  (`mul_q16_round`); the new target `p + delta * length / |leg|` per
  coordinate (one rational, half away), the delta mirrored per the flips.
  It rests the grenade at `p` instead when it already bounced `bounce_max`
  times, or the new leg is shorter than `combat.BOUNCE_MIN_Q` (a quarter
  tile). The new leg starts on the bounce tick (`leg_tick = tick`): it is
  flown from the next tick.
- **Slot 2** `Simulation._update_grenades` in two passes: every grenade
  `fly`s (the leg position, the piece flown since the last position walked
  against the map as the slot found it, a bounce), a fuse-out also `settle`s
  (the rest of the flight resolved at once by the same walk, bounces
  included, bounded by `bounce_max + 2` walks) and its detonation tile is
  decided (`combat.detonation_tile`); THEN the fuse-outs detonate in list
  order, each at once through the payload executor (each sees the previous
  one's blast, as before). Before #215 the flight asked no wall, so for every
  wall-free throw the two passes give the one loop's results.
- **The detonation tile** `combat.detonation_tile`: the resting point's tile
  (truncated, `target_tile`'s rule); if a wall stands there now (a door
  closed on a resting grenade; a grenade that never left the legacy painted
  door its thrower stood in) the first free 4-neighbour in the order N, S, W,
  E; walled in on all four, the tile itself.
- **The dials** `[weapons.hand_grenade] bounce_restitution = 0.4`,
  `bounce_max = 4` (config rows with their reason), on `WeaponDef`, refused
  outside 0..1 / below 0 (or not an integer) by name at the table's load;
  the restitution quantized once to `bounce_restitution_q`.
- **The device twin** (`cpp/src/cuda_unit_objects.cu`, Class T): the grenade
  table gains `leg_tick` and `bounces` (layouts regenerated); three device
  table rows `grenade_bounce_restitution_q`, `grenade_bounce_max`,
  `grenade_bounce_min_q`; `k_grenade_flight` flies, walks, bounces and
  settles per row over the env's `solid` plane (a slot input:
  `unit_kernels.GRENADE_FIELDS`, the plane as the slot found it);
  `k_grenade_fuse` records each PAYLOAD at `detonation_tile` over the same
  plane, approach the current leg's start; `k_table_appends` writes
  `leg_tick = thrown tick`, `bounces = 0`. A bounced leg whose target leaves
  the proved +-2^25 faults by name (only past 512 tiles).
- The WEGO order path's lob is the same `Projectile` in the same slot: it
  gets the same rule (no WEGO-specific test).

## approach

`approach` is read by `execute_payload` only when the detonation tile is
still solid after its wall damage (`blast_expansion.walk_seeds`). It is now
`Projectile.approach_tile()`: the CURRENT leg's start -- the thrower's tile
for an unbounced throw (the pre-#215 value, bit for bit), the last bounce's
hit point after a bounce (the conservative choice the brief named; for a
grenade resting after a bounce that is its own resting point, which is never
a blocked tile). Since the detonation tile is never blocked (rule 6) the
approach matters only in the walled-in corner case.

## Decisions taken here (computer science, not gameplay)

1. Slot 2 flies every grenade before any detonates, all against the map as
   the slot found it -- the device's shape (a wide flight, a serial fuse
   walk) and the attack slots' "decide against the start state" pattern.
   Rule 4 ("the map as it is that tick") is read as "as slot 2 found it".
2. `start_*` / `target_*` are the current leg instead of new columns beside
   the throw's: the observation (`obs_spec_v1._object_entry`'s velocity and
   "landed" flag) then reads the true flight with no change there, `approach`
   becomes the leg's start by itself, and the table grows by two columns,
   not six. The throw's own target is not kept after a bounce (nothing reads
   it).
3. A bounced leg starts on the bounce tick and is flown from the next: the
   grenade loses the rest of that tick at the wall (at most 1.25 tiles of
   flight at 30 tiles/s and 24 ticks/s).
4. The grenade gates' batches hold maps of different sizes; the lockstep's
   new `flight_solid` input pads the plane to the batch's largest grid with
   SOLID (off its own grid a tile is blocked to a flight), so the padded
   plane is the same law for every env.

## STOP: the two pinned ship_demo matches move (NOT re-baselined)

Both pinned matches (`tests/_pinned_matches.py`) have throws that meet
walls, so both move. Measured with `bot_census.py` (a read-only hook on
`_release_grenade` and `Projectile.bounce` around `run_match`, this branch):

| match | pin (main) | this branch |
|---|---|---|
| `bots` (`test_the_controller_plays_the_pinned_ship_demo_match`) | win team 0 at tick 2570 | **win team 1 at tick 2554** |
| `bots_v1` (`test_bots_v1_play_the_pre_g8p_pinned_match`) | draw at 7200, field `ab8d586e...` | draw at 7200, field `e3f499b8...` |

The first wall met decides the divergence: `bots` from tick 433, `bots_v1`
from tick 227. Every wall met (tick of the bounce; thrower = the owner id;
the leg from -> to as (y, x) tiles; the point hit; the blocked tile entered
and its material):

`bots` (12 throws, 9 bounces):

| tick | thrower | leg from -> to | hit at | wall tile | material | flips |
|---|---|---|---|---|---|---|
| 433 | 5 | (23, 69) -> (21, 44) | (23, 69) | (22, 68) | door (legacy painted) | x, y |
| 649 | 3 | (29, 11) -> (24, 30) | (28, 13) | (28, 14) | door | x |
| 650 | 1 | (22, 7) -> (23, 31) | (22, 13) | (22, 14) | steel | x |
| 1274 | 6 | (21, 91) -> (33, 72) | (24, 86) | (24, 85) | steel | x |
| 1275 | 5 | (24, 64) -> (34, 58) | (26, 62) | (27, 62) | door | y |
| 2410 | 6 | (28, 56) -> (23, 40) | (23, 41) | (23, 40) | door | x |
| 2410 | 7 | (23, 56) -> (23, 40) | (23, 41) | (23, 40) | door | x |
| 2426 | 3 | (32, 27) -> (47, 52) | (33, 30) | (34, 30) | steel | y |
| 2427 | 3 | (33, 30) -> (28, 39) (its 2nd leg) | (33, 30) | (33, 31) | steel | x |

`bots_v1` (3 throws, 1 bounce):

| tick | thrower | leg from -> to | hit at | wall tile | material | flips |
|---|---|---|---|---|---|---|
| 227 | 7 | (22, 58) -> (11, 45) | (18, 53) | (17, 53) | steel | y |

The ticks above are this branch's; up to tick 433 (`bots`) / 227
(`bots_v1`) they are the pinned match's own, after it the match is this
branch's. The lead takes the re-pin to Erik.

The full suite (one run at `fda55b96`, 1 h 34 min: 6458 passed, 7 failed)
names every pin that moves -- the two above and four more plays of the same
ship_demo match with one law switched off, each with the same throws:
`test_exposure.py::test_with_the_law_dormant_the_pinned_match_is_the_pre_g4p_match`,
`test_grenade_kinds.py::test_with_no_wind_up_the_pinned_match_is_the_pre_g3p_match`,
`test_horde_behaviours.py::test_with_every_behaviour_off_the_horde_plays_the_stage_1_law`,
`test_unit_impulse.py::test_g1_with_every_input_off_nothing_is_pushed_and_the_pins_are_re_pin_1s`.
No GOLDEN_AGGREGATE, field-digest golden or replica gate moved. The seventh
failure was this patch's rule section without an index row
(`test_claude_rules_paths`), fixed after the run by folding it into the
"Grenade kinds" section.

## Findings in the reference (reported, not changed)

1. **The throw starts at the CORNER of the thrower's centre tile.**
   `Simulation._release_grenade` passes `cx, cy` (integers) as the start and
   `centre + half + reach x dir` as the target, so the flight's start is the
   top-left corner of the tile while its target is offset by half a tile.
   With walls asked, the asymmetry shows: a throw up and to the left crosses
   the tile corner at once (both crossing parameters 0) into the diagonal
   tile; the first bounce of the `bots` match (tick 433, unit 5, standing
   with that diagonal tile a painted door) is exactly this. A start at the
   tile's centre would be the symmetric choice; it moves every throw by half
   a tile (Erik's / the lead's call).
2. **ship_demo's doorways are legacy painted doors (`MAT_DOOR`), solid to a
   flight.** Units walk through them (mobility 1000) but rounds already stop
   on them (`solid`), and now grenades bounce off them: five of the nine
   `bots` bounces are painted doors. Whether a painted door should let a
   grenade through (as it lets a body through) is a gameplay call; the A7
   migration to entity doors would settle it the same way for both.
3. **A grille lets a grenade through** (it lets a round through: `solid` is
   false on a grille). Physically a steel grating would stop a grenade.
4. **`has_los` (Bresenham between tile centres) and the flight disagree.**
   Two `bots` throws at tick 2397 had a physical line (`has_los(physical=
   True)` from the start tile to the target tile) yet their segment, from
   the tile corner, entered a door tile; they bounced at 2410. The bots
   judge a frag by the TEAM's sight of the group, not by a flight check.

## Who else aims a throw (nothing changed there)

- **The bots** (`src/control_bots.py`): the frag goes to the centre of the
  largest group of enemies the TEAM sees (`frag_point`: the team's sight,
  not the thrower's, no flight check); the flashbang to a known enemy the
  team does NOT see now, while it bounds onto it (`flash_point`: behind a
  wall or a corner by design). So yes, the bots throw at points behind
  walls. Counted on MAIN's law (the same census with `flight_walk` made a
  no-op and the detonation at the target: it reproduces both main pins bit
  for bit, outcome and field digest):
  - `bots`: 9 throws; 5 at a target out of the thrower's physical line
    (`has_los(physical=True)` start tile -> target tile), 5 whose straight
    flight enters a blocked tile (so each passed a wall on main);
  - `bots_v1`: 3 throws; 1 out of line, 1 entering a blocked tile.
  On this branch the first of them is where each match leaves its pin.
- **The gate and the mask** (`verb_gates.can_throw`, `action_spec_v1`): ask
  only whether a throw would START (wind-up, planting, count, can act); the
  target is never a term (the verb-gates rule), so nothing there reads the
  flight and nothing changes.
- **The observation** (`obs_spec_v1._object_entry`, `team_picture.
  object_tile`): reads the grenade's tile from its live position (`fx`,
  `fy`, the integer position's views), its velocity from `target - start`
  of the CURRENT leg times the travel speed, "lies on the floor" as
  `position == target`. Because `start_*` / `target_*` are the current leg,
  a bounce does not break it: after a bounce the entry shows the new leg's
  direction at the same speed, and a resting grenade (start == target ==
  position) reads as on the floor. Before a bounce nothing differs.

## Tests (properties; each names what it protects and the change that breaks it)

`tests/test_grenade_bounce.py` (17, about 4 s, no GPU):

- a throw that meets no wall flies (every tick's position), lands and
  detonates (tile, tick, approach) as the pre-#215 law -- 72 directions x 6
  reaches x 3 fuses (rule 1);
- a wall in the way on either axis: the right component flips, the stop is
  the free tile's edge, the new leg = left x restitution rounded once (its
  vector within one Q16 unit), the detonation on the thrower's side on a free
  tile -- a sweep of angles and reaches (rules 2, 5, 6);
- an exact corner flips both; restitution 0 drops at the wall; restitution 1
  mirrors in full and `bounce_max` ends it (a corridor two tiles wide); a leg
  under a quarter tile rests (rules 2, 3);
- a door closing mid-flight stops it; a wall destroyed mid-flight lets it
  through (rule 4);
- a fuse-out mid-leg never detonates beyond the wall (a sweep of fuses and
  reaches), and in the sim's slot 2 detonates where the rest of its flight
  ends, not where it was on the fuse tick (rule 5);
- a grenade resting where a wall now stands detonates on a free neighbour
  (rule 6); the walk enters every tile the segment crosses, in order (300
  random segments against an exact enumeration with fractions); the dial's
  door refuses a restitution outside 0..1 and a bad `bounce_max` by name;
- in the game: throws at a wall detonate on the thrower's side and the one
  that cannot reach it lands on its target tile as before; same seed, same
  flight (positions, bounces, detonations every tick); the 2c finding:
  from (14, 13) in `range_corner` at the reach that landed at (34, 35) (and
  the transposed pair) no longer detonates east of the first wall.

`tests/cuda_unit_objects_check.py` (wrapper `tests/test_cuda_unit_objects.py`,
Class T, in `tools/cuda_green.py`): PART FLIGHT's synthetic full tables now
start on the grid and mostly aim off it (most rows meet the hull; exact
45-degree corners constructed), at the shipped restitution, 0 and 1, with a
required census of the bounce branches (all ten taken: about 900 mirrored
legs, 470 exact corners, 665 bounce_max rests, 395 short-leg rests, 640
mid-leg fuse-outs per run); new PART BOUNCE: the bouncing scene (one wall, a
gap sealed while a grenade flies at it, a wall tile destroyed while one
flies at it, a mid-leg fuse-out, a horizontal wall, a pillar, a corridor) at
the three restitutions shadowed every tick (v0 against an unhooked twin),
its law facts on the host, a batch of N = 4 (three bounce variants and a
grenade scene: env i == env i alone), the chained 1b + 1b' + 1c + 2 on it;
every other part (scene, batch, padding, overflow, refuse, chained, ship)
with slot 2 reading its plane. `tests/cuda_unit_payloads_check.py` hands slot
2 its plane before the DETONATE stage writes it.

## The breaks (each new gate broken once; every break went red)

Host (`tests/test_grenade_bounce.py`), one mutation at a time:

| break | red in |
|---|---|
| a corner tie broken to x | the exact corner |
| the bounce length from the whole leg | the wall sweep (x, y), restitution 1, the short leg |
| the restitution ignored (a mirror always) | the wall sweep (x, y), restitution 0, the short leg |
| `bounce_max` off by one | restitution 1 / bounce_max (first GREEN: the corridor run ended on a short leg; the corridor was narrowed and the test now requires the last rest to have four tiles left) |
| no settle at the fuse-out in slot 2 | the in-game mid-leg fuse-out (first GREEN: the law tests settle by themselves; the in-game test was added) |
| the walk skipped in flight | 10 tests |
| the walk asks only the last tile | 8 tests (the walk order among them) |
| the detonation tile unchecked | the rest tile under a new wall |
| the first leg's tick moved by one | rule 1 |
| the quarter tile moved to an eighth | the short leg |
| no flip on a wall | the wall sweep (x), the corner |
| the stop point inside the wall | 6 tests |
| the restitution door accepting 1.5 | the door |

One mutation stayed green and is EQUIVALENT, not a gap: detonating at
`target_tile()` instead of `detonation_tile(pos)` -- after the settle the
current leg's target IS the resting point (an unobstructed leg arrived, a
bounced one rests with start == target == position), so the two differ only
in the blocked-tile fallback, which `detonation_tile`'s own test holds.

Device (`cpp/src/cuda_unit_objects.cu`, rebuilt each time, PARTS flight,
scene, bounce; every one RED): the corner tie broken to x; the restitution
ignored; the walk skipped in flight; the detonation tile without the
fallback (the PAYLOAD records differ); no settle at the fuse-out; an appended
row's `leg_tick` off by one (the 1b shadow names the column); `bounce_max`
off by one. An incremental CUDA build of the TU took about 30 s.

## Left out

- No re-pin (the STOP rule): `test_the_controller_plays_the_pinned_ship_demo_match`
  and `test_bots_v1_play_the_pre_g8p_pinned_match` fail on this branch by
  design until Erik rules.
- The start-at-the-tile-corner quirk, the painted-door and grille materials:
  findings above, unchanged.
- No index line in CLAUDE.md: the walk is the grenade's own (the rounds keep
  their march); its constraints are in `.claude/rules/hand-verbs.md`.

## Follow-up, 2026-11-18 (Erik's rulings): the hand, the grille, no re-pin here

### 1. The throw leaves the hand

Erik: "it can start in Q16 coordinates offset in any tile"; "it should take
into account the unit's position, as well as facing, then offset it a
little"; "we should try to get it perfect, since it directly shows
graphically". This replaces finding 1 above (the start at the centre tile's
corner).

- `combat.release_point(gmap, x_q, y_q, footprint, dx_q, dy_q, forward_q,
  side_q)`: the thrower's Q16 centre (`x_q` / `y_q` + half the footprint,
  formed in integers, the point the 3D model is drawn at) plus `forward` along
  the throw's Q16 vector and `side` to its right `(-dy, dx)`, ONE rational per
  coordinate rounded half away. The piece centre -> hand is walked by
  `flight_walk`: a hand beyond a wall starts the grenade at the last free
  point (no bounce counted). The throw's vector is the facing at the release
  (`cos_q16` / `-sin_q16`, the wind-up path) or the intent's own vector for a
  0-tick wind-up -- the same vector the flight takes, one code path.
- The dials `[weapons.hand_grenade] release_forward_m = 0.45`,
  `release_side_m = -0.29` (metres, negative = the unit's LEFT), converted
  once per map to `release_forward_q` / `release_side_q` (Q16 tiles) at the
  weapon table's build; refused beyond +-2 m by name.
- **The measurement** (`measure_hand.py`, the scratch tool: the REAL pose
  layer from `config.toml`, no window): the SOMA marine
  (`soma_space_marine`), state idle, role player; `PoseLayer.throw` with the
  wind-up the sim gives (0.3 s -> 7 ticks at 24 tps = 0.2917 s); driven
  0.2917 s at 1/240 s; the timed clip is then at 1.0999 s of
  `Throw_Overhand` (`[render.pose.throw] release_s = 1.10`, the release
  frame). The `LeftHand` bone's composed global (model space) is (0.2713,
  1.3850, 0.4277); through the renderer's own `body_matrix` (yaw = facing +
  90 degrees, the one scale: the space marine reference is 1.8847 model
  units tall = 6 drawn tiles), it sits **1.3616 drawn tiles forward and
  0.8637 drawn tiles to the unit's left** of the unit's centre, the same at
  facings 0, 37 and 90 degrees (the frame check). At the 0.333 m tile that is
  0.4534 m forward and -0.2876 m: **0.45 m / -0.29 m** to the centimetre.
- The left arm throws (the clip is mirrored), so the side is negative.
  Render-only left-handed units (10 %) are drawn throwing with the right hand
  but release from the left: handedness must not affect play (Erik's #187
  ruling), so the sim cannot follow the mirror.
- Exactness: the model is drawn 6 TILES tall whatever the tile size, while
  the dials are metres, so the hand is exact on screen at the 0.333 m tile
  (every game level) and not on the 1 m test worlds.
- The target is unchanged (the centre tile's point at the reach), so a bot's
  aimed point is still where a free throw lands.
- The request record carries the Q16 start (`start_x_q`, `start_y_q`; the
  throw's `tile_x` / `tile_y` are 0); the device verbs kernel forms it over
  the flight plane (`unit_lib::release_point`), and the walk moved into the
  device library (`cpp/src/cuda_unit_lib.cuh`) since two kernels use it; two
  device table rows `hand_release_forward_q`, `hand_release_side_q`.
- A thrower on a legacy painted door: the centre's own tile is never asked,
  so the start (the hand) is walked FROM the door tile; if the hand is in a
  free tile the grenade starts there normally; if the grenade never leaves
  the door tile, `detonation_tile`'s N, S, W, E neighbour fallback keeps the
  blast off the door.
- The WEGO order path (`spawn_projectiles_from_grenade_orders`,
  `timeline`'s lob) keeps its legacy centre start: it builds its own
  `Projectile` in its own code, not through `_release_grenade`.
- **Rule 1, re-read**: a wall-free throw no longer flies main's positions
  (its start moved to the hand) but detonates on main's tile at main's tick:
  `test_in_the_game_a_wall_free_throw_detonates_where_and_when_main_did`
  (12 facings x reaches through the sim: every detonation tile = main's
  target tile, every tick = thrown + fuse, every start moved). The leg law
  test keeps holding the first leg's arithmetic from a given start.

### 2. A grille stops a grenade

Erik: "grille should probably not let through a grenade" (bullets through a
grille stay as they are, undecided). `[materials.*] blocks_thrown` (bool,
optional; derived = "sealed to gas", exactly where the tile is `solid`;
`true` on `[materials.grille]`). `combat.flight_blocked` = off the grid, or
`solid`, or the tile's material's `blocks_thrown` -- read through the
material table each time, so it follows every topology change with no plane
of its own to keep in step (the material plane and `solid` are already the
topology seam's). `combat.flight_plane(gmap)` is that read over the grid,
the plane both device slots (1b's start, 2's flight) are handed. The rounds'
march is untouched (it reads `solid`).
`test_a_grille_stops_a_grenade_and_still_passes_a_round`.

### 3. No re-pin here

Erik (2026-11-18) retires the legacy painted doors in the same re-baseline,
which another patch builds on this branch. The six pin tests stay RED here,
by name; their outcomes at this branch's final tree are below (the "before"
of that patch).
