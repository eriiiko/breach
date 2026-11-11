# Bots, zombies and speeds — design (draft 1, 2026-11-10)

Part of the gameplay pack (#183). Three patches beside the pack's eight:
**G1p-c** speeds in m/s and direction-dependent walking, **G8p** the bot
pass, **G9p** the zombie law v2. Vocabulary (Erik, 2026-11-10): **bots** are
hand-written AI, **agents** are RL-trained policies.

Status: DRAFT. **RULED** = Erik's words of 2026-11-10; the rest is Claude's
proposal.

## 1. Speeds in m/s (G1p-c)

**RULED.** Speeds are given in m/s; the tiles-per-second dials are leftovers
and every number may change. Walking backwards is slower: forward 1.0,
sideways 0.8, backward 0.6. A sprinting zombie reaches 70 to 90 % of a
marine's walk and catches a marine walking backwards. The zombie's sprint is
unlimited.

**Today** (a tile is 0.333 m): marine walk 18 ticks per tile = 0.44 m/s,
full speed 8 ticks per tile = 1.0 m/s, zombie 14 ticks per tile = 0.57 m/s.
A real walk is about 1.4 m/s and a run with kit 4 to 5 m/s, so today's
marine sprints slower than a person strolls. The unit impulse law works in
real units (a shove gives about 1 m/s), so the speeds must be real for it to
read right.

**PROPOSED table** (all PLACEHOLDER for the play test, one `[speeds]` table
in m/s, quantized once at load):

| who | mode | m/s | note |
|---|---|---|---|
| marine | walk (throttle 0) | 1.4 | a normal walk, weapon ready |
| marine | full (throttle ONE) | 4.5 | a run with 20 kg of kit |
| zombie | shamble | 0.6 | |
| zombie | sprint | 1.15 | 82 % of the marine's walk |

**Direction rule (RULED 2026-11-10: sprinting is forward only).** With θ the angle between the move direction
and the facing:

```
speed = f(θ) · v_walk + throttle · (v_full − v_walk) · max(0, cos θ)
f(θ): 1.0 forward, 0.8 sideways, 0.6 backward (blended by cos θ through the kit)
```

So the walk is slower sideways and backward by Erik's factors, and the
sprint's extra speed exists only in the forward direction: nobody sprints
backwards. A marine backing away moves at 0.84 m/s whatever the throttle,
and a sprinting zombie (1.15) catches it; a marine that turns and runs
(4.5) escapes but cannot shoot behind itself. That trade is the point.

A zombie obeys the same rule with its own two speeds.

**Consequences to accept:** every movement-dependent pin re-baselines; the
map plays about three times faster on foot (a 20 m ship is crossed in under
five seconds at a run), so match lengths, the bots' thresholds and the
grenade timings want a look at the play test. The drawn walk-cycle rates
(`[render.pose]`, re-derived from the drawn speed by its own test) follow.

**Gate:** throttle 0 facing forward = `v_walk` exactly; speed monotone in
the throttle and in cos θ; the zombie sprint is greater than the marine's
backward speed and within 70 to 90 % of its walk (a config-door check, so a
later retune cannot break Erik's rule silently). **HUMAN-TEST.**

**As built (G1p-c, 2026-11-10).** `[speeds]` in `config.toml`, read by the
refusing door `simulation/speeds.py::speed_settings` (each key by name; the
sprint band and the backward-walk rule; and `walk <= full`,
`0 < backward_frac <= sideways_frac <= 1`, which the law's monotonicity
needs) and quantized once per map by `speed_table` into `Simulation.
speed_table`. The marine's two speeds are Q16 TICKS PER TILE, not Q16 tiles
per tick: the old walk, 1/18 tile a tick, has no Q16 tiles-per-tick value,
and in the cost currency `movement.default_speed` (terrain, wind) applies
unchanged, so the proof of cause below is exact. The law is
`movement.move_speed_tiles_per_tick` (one exact rational, one division, as
the G1p throttle was); theta is `atan2_q16(move) - facing` through
`turning.wrap_q16` and `cos_q16` (`movement.facing_cos_q16`, exact at the
cardinal angles), and a unit holding no aim has theta 0. The zombie's
shamble is its integer cadence (`ticks_from_seconds(tile / mps, tps)`):
`Unit.speed_ticks_per_tile` defaults to None = the table, so a converted
marine shambles too (before, it kept the marine's 18-tick attack dial). The
`[zombie] seconds_per_tile` dial is deleted; `[movement]` stays, WEGO's
alone. Realised on the 0.333 m tile at 24 tps: walk 1.400, full 4.500,
sideways 1.120, backward 0.840, zombie 0.615 m/s (13 ticks per tile), the
sprint 1.150 (unused until G9p). `ReloadEvent(unit_id, ticks)`, render-only,
from `combat.reload_begin`. The pinned ship_demo match: win 1 unchanged,
700 -> 240 ticks; with the table at the old cadences (18 / 8 / 14 ticks per
tile, factors 1.0) it is the G2p pin bit for bit. The SOMA walk rates were
re-derived (Walk 0.998, Rifle_Aim_Walk 1.044, Zombie_Hunch_Walk 0.572); the
walk clip plays at the walk rate at every throttle (a run clip is #184's).

## 2. The bot pass (G8p)

**RULED.** The three-layer shape. Bots stay Python on the host. Erik expects
Claude to choose the techniques: good and cheap.

### 2.1 Layers

1. **Team layer** (one object per team, a few times per second): the shared
   picture (known enemies with last-seen tick, who targets what), the shared
   target, and the current PLAY.
2. **Unit layer** (per bot, a few times per second, staggered by unit id so
   the cost is spread): every possible action gets a SCORE and the best one
   wins: shoot, hold fire, move to a position, throw (by kind), shove,
   melee. A new behaviour is a new scored action.
3. **Position layer**: where to stand. Candidate tiles near the bot are
   scored by FEATURES; the bot paths (A*, as today) to the best.

### 2.2 The position features (each a yes / no question about a tile)

- sees the shared target from here;
- is outside every teammate's fire lane (the line from a shooting teammate
  to its target, marked on a small grid);
- is at least the spacing distance from every teammate;
- is beside a corner or behind cover relative to the target;
- is not visible to an enemy that is not the target;
- is closer to the play's objective than the current tile;
- is not in fire, dense smoke or vacuum;
- is reachable within n steps.

v1 scores a tile by a hand-set weight per feature. The features are kept as
explicit booleans on purpose (see §2.5).

### 2.3 Plays (v1: three)

- **Bound**: the team is two halves; one half moves to its best positions
  while the other holds and watches; they swap when the movers arrive.
- **Hold**: everyone takes the best position covering a chokepoint or the
  last-seen enemy; nobody advances.
- **Fall back**: against a horde or when outnumbered, retreat toward home
  facing the enemy and shooting (the backward walk of §1), shove a zombie
  that reaches a bot; the half that is farther covers the nearer.

The team layer picks the play from counts it already has (enemies seen,
teammates alive, ammunition left, zombies near). Room breaching (flashbang,
two doors) is a fourth play for a second round.

### 2.4 Also in the pass

Focus fire through the shared target (most exposed by G4p's debuff, then
weakest, then nearest); flashbang or frag by the scored throw action; #142
(stacking: the spacing feature; doors: a bot opens an adjacent closed door
on its path) and #138 (judder: a bot commits to a position until the next
decision).

### 2.5 The gate is a tournament

`tools/bot_tournament.py` over the headless match runner: new bots against
the old bots (kept registered as `bots_v1`), and both against the horde,
over many seeds and both ships. Report win rate, survivors, rounds fired per
kill, friendly-fire hits. The pass merges when the new bots clearly beat the
old ones and hit each other less. The old bots stay as an opponent for
agents: a ladder of opponents is worth more than one.

### 2.6 Later: learned scoring (Erik's idea, v2)

The position features are yes / no questions, which is exactly a Tsetlin
machine's input. A v2 side project trains a small Tsetlin machine to score
tiles (and the same layer could later be a small neural network), from the
tournament's outcomes, as one more kind of opponent in a varied cast. The
feature list above is its interface, so nothing in v1 needs to change for
it. The octopus arm brains (#108) plan the same technique.

## 3. The zombie law v2 (G9p)

**RULED.** Zombies do not stand idle: they wander. They hear (by distance
alone, walls ignored, no sound system needed). They fear fire and would
rather avoid smoke. They bash doors. They are slow, with an unlimited
sprint (§1). A zombie is a STATE of a unit and in v1 a copy of the unit it
infected.

**Today.** Dormant until a living non-zombie is within 48 tiles in line of
sight; a woken zombie wakes others within 15 tiles; each walks by its own A*
(re-planned every 5 steps) to the Euclidean-nearest prey and bites when
adjacent.

**The residency design already replaces the pathing** (its P3d): the horde
walks ONE integer distance field per map, flooded from every living marine
over the passable tiles, refreshed every few ticks, each zombie stepping to
its lowest neighbour. It is far cheaper than an A* per zombie, it gives
path-nearest instead of straight-line-nearest prey, and it makes a horde
flow around obstacles and arrive from every open side.

**RULED 2026-11-10: the distance field moves forward now. G9p is P3d pulled
forward, with the new behaviours designed onto the field**, so the horde is rewritten once:

| behaviour | law |
|---|---|
| wander | a dormant zombie takes a slow step every few seconds in a direction picked by a hash of (zombie id, tick), staying within a few tiles of where it stood; no random stream is drawn |
| sight | as today (radius + line of sight), wakes it and its neighbours |
| hearing | every shot, detonation and door bash is a NOISE at a tile with a radius from its loudness (`[weapons.*] loudness` exists, unread; payloads get one); a zombie within the radius, walls ignored, wakes and walks toward the noise; reaching it without seeing prey, it returns to wandering there |
| the chase | awake with prey known: descend the prey field at the shamble; SPRINT when it has line of sight to prey within `sprint_radius` |
| fear of fire | burning and very hot tiles are removed from the field's passable mask: a zombie never paths through fire, so a fire line closes a corridor; one with no path waits at the edge |
| wary of smoke | PROPOSED later: the field is an unweighted flood, and "expensive" tiles need a weighted one; v1 treats dense smoke like any air |
| door bashing | a closed door counts as passable in the field; a zombie that reaches one strikes it with its melee through the door's existing per-tile hit points until it breaks; each strike is a noise |
| bite | unchanged |

**Open with Erik (small):** the sprint radius; how long a zombie keeps
chasing a noise; whether a door should hold for seconds or for a minute
(a number for the play test).

**Gate:** the field's distances equal a reference flood on every map; a
zombie never steps onto a burning tile; a noise wakes exactly the zombies
within its radius; with wander, hearing, fire and doors switched off the
horde's behaviour is the P3d field law alone (its own written spec and
re-pin). **HUMAN-TEST.** Tier: Opus.

**As built (G9p, 2026-11-11).** Two stages, each re-pinned on its own.

*Stage 1, the field law (residency P3d).* `src/simulation/horde.py` +
`cpp/src/horde_field.cpp` (`PhysicsEngine.horde_field`, a host call in the CPU
and the CUDA module, on the `/fp:strict` list; `horde.flood_reference` is the
same law for a map without an engine). A field is a 4-connected integer BFS
over the ANCHOR grid of a footprint (an anchor is passable when its whole
block is in bounds and on `mobility > 0`, `is_passable_block`'s own
predicate); seeds are labelled 0 passable or not, the frontier enters only
passable cells, an unreached cell is `0x7FFFFFFF`. The prey field is seeded
at every living non-zombie's centre in the zombie's anchor space
(`centre - fp // 2`: A*'s old goal when the footprints agree). Each awake
zombie steps to its lowest 4-neighbour STRICTLY below its own cell that it
can enter now, else waits (keeping its stride clock, so it steps the tick a
way opens). The cache (`HordeState`, never synced, never digested) drops
every field at a tick `% field_refresh_ticks == 0` and whenever
`GameMap._tile_serial` moved, and floods a field the first tick a step needs
it. Activation and the bite are unchanged, the `sqrt` gates integer squared
compares (identical results: a correctly rounded sqrt is monotone). No A*
call remains in the law; `Unit.zombie_path` / `zombie_path_idx` are gone.
DEVIATION: the descent's tie order is per cell parity (vertical first on an
even `x + y`, horizontal first on an odd one) instead of one fixed order --
on a 4-connected field a diagonal goal has two equal lower neighbours on
every cell, and one fixed order walks an L (all north, then all east); the
parity order walks the staircase. Still one fixed rule per cell.

*Stage 2, the behaviours on the field* (each a `[zombie]` switch through the
refusing door `horde.horde_settings`; all five off is stage 1 bit for bit,
the permanent gate `tests/test_horde_behaviours.py::test_with_every_
behaviour_off_the_horde_plays_the_stage_1_law`):

- WANDER: a dormant zombie steps on ticks with `(tick + 37 id) % period == 0`,
  in the direction `NEIGHBOURS[philox32_below(w, 4)]`, `w` the first word of
  the vendored Philox block with key 0 and counter `(tick, 0, "WAND", id)` --
  a seedless hash, never a stream (the sim's generator is untouched, gated).
  A step that would leave the leash is skipped; a zombie pushed outside it
  steps back toward its anchor.
- HEARING (`simulation/noise.py`, `GameMap.noise_log`): a noise is
  `(tile, radius in tiles)`; shots (each trigger pull of a ranged marcher and
  each spray burst, at the shooter's centre), detonations (`execute_payload`)
  and door bashes emit; the log is drained at slot 5 every tick. A non-awake
  zombie within a radius (squared, inclusive, walls ignored) takes the
  nearest noise as its target with a fresh timeout and walks the noise's
  field (seeded at the noise tile in anchor space) at the shamble; within
  `noise_arrive_m`, at the timeout, or with no path it SETTLES -- dormant
  there, its wander anchor moved there. At most `noise_fields_max` distinct
  targets live (the newest; a dropped one's zombies settle). `loudness` is
  METRES of hearing radius on weapon rows (required on projectile / hitscan /
  spray rows; the old 0..1 identities x 30 m) and payload rows (optional).
- FEAR OF FIRE: burning tiles and tiles with `temperature >= fear_temp` (Q16
  game degrees above ambient) are out of the mask at each refresh; every step
  (descent and wander) refuses a block holding one on the LIVE map. A zombie
  the feared field does not reach walks the same prey goals flooded without
  the fear, so it comes up to the fire's edge and waits there (the design's
  "waits at the edge"; without it a cut-off zombie would wait where it
  stood).
- DOOR BASHING: the tiles of a living closed door entity that is not a hull
  door (no span tile borders vacuum, the ambient ring or the map's edge,
  4-way) are in the mask; a descent blocked only by such tiles strikes the
  first one (row-major) through `combat.chew_wall` (the door tile's own
  `wall_hp`; `destroy_wall` at 0, and the slot-9e door sweep then destroys the
  whole door) at the bite's cadence (`last_melee_tick`), only while it can
  act; each strike is a noise. No schema hp was needed: a closed door's hp is
  its tiles' `wall_hp`.
- SPRINT: an awake zombie with `has_los` to a living prey within
  `sprint_radius_m` walks at `SpeedTable.zombie_sprint_ticks` (the `[speeds]`
  sprint as an integer cadence) through `default_speed` (terrain, effort).
  A zombie with its own int cadence keeps it.
- State: `Unit.zombie_noise_y/x/ticks`, `zombie_anchor_y/x` (fixed-width,
  synced), hashed in the unit digest as `noise` / `anchor` only while set,
  cleared at death, conversion and waking (`horde.clear`).

PLACEHOLDERS chosen (Erik tunes at the play test): `field_refresh_ticks = 2`
(a marine at full speed outruns the field by at most ~1.1 tile);
`wander_period_seconds = 3.0`, `wander_leash_m = 1.5` (5 tiles);
`noise_timeout_seconds = 20.0`, `noise_arrive_m = 0.7` (2 tiles),
`noise_fields_max = 4`; `fear_temp = 60.0`; `door_bash_damage = 2.0` (a
40 hp closed door takes 20 strikes at the 1 s bite cadence: 20 s for one
zombie), `bash_loudness_m = 12.0`; `sprint_radius_m = 8.0` (24 tiles); weapon
loudness carbine 24 m, whisper 4.5 m, lances 3 m, LR-50 / Jackhammer 30 m;
payload loudness frag 40 m, C4 / cook-off / engine death 60 m, smoke, teargas
and poison 8 m.

Re-pins: stage 1 -- ship_demo seed 1 win 0 at 266 -> win 1 at 264 (the
spawned horde's activation ticks unchanged, the first different step on tick
195, zombie 10 never bites); the box's outcome and kills unchanged. Stage 2 --
ship_demo win 1 at 264 -> win 0 at 264 (the sprinting horde bites and converts
three bots, 204-214); the box 130 -> 92 ticks (the sprint alone). Rationale and
values beside the pins in `tests/test_zombie_control.py`. Slot 5 on ship_demo
seed 1 (CPU, `bench_bot_matches --profile`): before 0.152 ms mean / 0.200 p95
/ 2.04 max (the A* spikes), stage 1 0.139 / 0.192 / 0.32, stage 2 0.162 /
0.248 / 0.81.

Not done: smoke wariness (later, by design); the device kernel (P7d). A noise
log entry made at slot 9e is heard the next tick and is not synced (accepted
gap for a save / load between).

## 4. Order

G1p (done) → G2p (building) → G1p-c speeds → G3p → G4p → G5p-a / G5p-b →
G6p → G7p → G9p zombies → G8p bots (last, so they use everything and fight
the new horde) → the `player_up` camera spike.

## 5. Systems

(a) Uses: the Match controller registry (`bots`, `zombies`), the intent
facade, A* (bots), Vision (`has_los`), Timed/blast seams for noise sources
(`blast_log`, `ShotFiredEvent`), the Footprint samples builder and
`default_speed` (speeds compose there), Fixed-point kits (`cos_q16`), the
Headless match runner, Config.

(b) Creates:

| system | draft rule |
|---|---|
| Speed table (`[speeds]`, m/s) | Every unit speed is a row in m/s, quantized once; never a ticks-per-tile literal; the direction rule is the one place facing changes a speed |
| Bot layers (`control_bots`: team / unit / position) | A new bot behaviour is a scored action, a play, or a position feature; never a fixed-order rule or a second controller |
| Position features | A tile is judged through named yes / no features; a learned scorer reads the same list |
| Bot tournament (`tools/bot_tournament.py`) | THE measure of a bot change: win rate and friendly fire over seeds, never an opinion |
| Horde field (the residency design's P3d) | The horde paths ONLY by the distance field; a new zombie behaviour is a mask, a goal set or a rule on the descent, never a per-zombie search |
| Noise (`loudness` on weapon and payload rows) | What can be heard is a noise event with a radius; hearing reads noises, never the sources |
