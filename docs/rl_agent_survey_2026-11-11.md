# RL agent survey: what the environment gives today, and how other groups built theirs

2026-11-11. The first product of the RL agent design session (state space,
action space, trainer). This is a survey, not a design: nothing in it is
decided, and nothing is built from it before the design is agreed.

**How it was made.** Part A was read from the code at `ce4c0be4`
(`src/rl_env.py`, `src/control_policy.py`, `src/simulation/sensor_accessor.py`,
`src/control_bots.py`, `src/simulation/vision.py`, `src/simulation/charges.py`)
and checked by running the environment's own schema functions. Part B was
assembled from seven research digests written for this session
(`docs/research/rl_agent_survey_2026-11-11_raw/01` to `07`). Each digest ends
with a ledger that says, per source, whether it was read in full, read in
part, known from its abstract only, or recalled from memory. July's landscape
survey (`docs/research/rl_litsearch_2026-07-20.md`) is not repeated here; this
one goes a level down, to the interfaces.

**Words used throughout.**

- *Policy*: the function that maps what a soldier observes to what it does. Here a neural network.
- *Observation*: the numbers the policy is shown at one decision. *Action*: the numbers it answers with.
- *Reward*: a number handed back after each action that says how good the outcome was. Learning means changing the policy so that the sum of rewards over a match grows.
- *Episode*: one match from spawn to end.
- *Trainer*: the program that plays many episodes with the current policy, collects (observation, action, reward) records and improves the policy from them. PPO is one such recipe (Part B.7).
- *Environment* ("env"): the game wrapped so that a trainer can step it: actions in, observations and rewards out.
- *Value function* (or *critic*): a second network, used only during training, that estimates how well a situation will end. It tells the trainer whether an action did better or worse than expected.
- *Decimation*: the number of game ticks one action lasts. Breach runs 24 ticks per second; at decimation 4 a soldier decides 6 times a second.

---

## Part A. What the environment gives an agent today

`BatchedEnv` runs N matches at once. One call, `step(actions)`, advances every
match by `decimation` ticks and returns, for every soldier of every match, an
observation, a reward, and whether its match ended. A trainer touches nothing
else. The env's own documentation calls the observation and the reward
"placeholder rows (agent era)": they were built to prove the plumbing, and
nobody has yet chosen them for learning. Read everything below in that light.

### A.1 Which soldier is which row

Every array has one row per soldier, `U = T x S` rows per match: T is the
largest number of marine teams any level in the run has, S the largest
`max_per_team` of any spawn room. Row `t * S + s` is team t's roster slot s.

- A slot is given at spawn and never reused. When its soldier dies, or is bitten and turns zombie, the row stays and its `valid` flag goes false; its action row is then ignored. Team size is a mask, never a change of shape.
- Zombies have no rows. They are the horde's, driven by their own controller.
- Every marine team has observation rows, whether a policy or a scripted bot drives it. So a bot team's observations can be read (useful for learning from the bots), but the env does not return what the bots did.
- On `ship_demo` (two teams of 4, rooms sized for 6): S = 6, U = 12 rows, 8 valid at the start.

### A.2 The observation

Three arrays per soldier.

**1. The map crop**, `(C, 2r+1, 2r+1)` numbers: a square window of the tile
map centred on the soldier's centre tile, north up (it does not turn with the
soldier), zero outside the map. The run file picks the radius r and which of
these eleven planes to include:

| plane | holds |
|---|---|
| `pressure` | air pressure, in atmospheres |
| `smoke` | smoke density |
| `water_depth` | water depth |
| `temperature` | game degrees above ambient |
| `fire` | fire intensity |
| `o2` | oxygen density |
| `solid` | 1 on a wall tile |
| `material` | the tile's material, as its table id |
| `units_own` | 1 on every tile a living unit of the soldier's own team stands on (itself included) |
| `units_enemy` | the same for other marine teams |
| `units_zombie` | the same for the horde |

Light is refused until light becomes game state (#81).

Facts about the crop that matter for the design:

- **It ignores line of sight.** A unit inside the window is shown whether or not a wall stands between. There is no fog of war in the env. (The game does have a sight model, and the bots use it: A.7.)
- **It says "someone is here", nothing more.** Not which soldier, not its health, its facing, or what it is doing.
- **The numbers are raw.** Pressure in atmospheres, temperature in game degrees, the material as an id number. Nothing is scaled or encoded for a network.
- **The example window is tiny against the fight.** The documented example is r = 8: 17 x 17 tiles, which at 0.333 m per tile is 5.7 m across. A marine's body is 3 tiles wide. The carbine reaches 90 tiles (30 m), a thrown grenade 30 tiles (10 m), and `ship_demo` is 100 x 70 tiles. A window that covered carbine range would be 181 x 181 tiles per soldier per plane.

**2. The vector**: up to six numbers per soldier, picked by the run file: `hp`,
`team` (the team's number), `alive`, `facing` (radians), `x`, `y` (its
position in tiles, absolute on the map).

**3. The `valid` flag** of A.1.

What the observation does **not** contain today:

- its own state beyond those six: rounds in the magazine, magazines left, grenades by kind, whether it is reloading or winding up a throw, whether it is stunned, blinded or knocked down, how fast it is moving, its weapon, whether its flashlight is on;
- any list of team-mates or enemies. A team-mate four metres away is outside the example window and so does not exist for the soldier;
- anything that happened: who hit it and from where, a shot or a blast it would have heard (the horde hears noises; the soldier does not);
- a grenade in the air or lying on the floor with its fuse running;
- the map beyond the window, the time left in the match, any objective, its own previous action.

One ruling already touches this: the residency design's 12.5 (Erik,
2026-11-07) says the observation door will offer both shapes, the per-soldier
crop and a reveal-all full map.

### A.3 The actions

One row of 17 integers per soldier per step. The columns are generated from
the game's own control verbs, the same calls the human's controls and the
bots make, so anything a policy does is recorded and replays like any other
match.

| columns | what it does | lasts |
|---|---|---|
| `set_move_dir.dx_q`, `.dy_q`, `.throttle_q` | Walk in this direction (a unit vector; 65536 means 1.0). Throttle 0 is the walk (1.4 m/s), 65536 the full run (4.5 m/s). The extra speed applies forward only: sideways a soldier moves at 0.8 of the walk, backward at 0.6. | held every tick |
| `clear_move_dir` | Stop holding a move. (A zero vector is different: it holds a motionless move.) | held |
| `set_aim.dx_q`, `.dy_q` | The direction the soldier wants to face. The body turns toward it with inertia: at most 360 degrees a second, and it takes 11 ticks to turn 90 degrees, 17 to turn 180. A zero vector keeps the last wanted direction. | held |
| `set_trigger.held` | While 1 the weapon fires along the direction the body faces now, at its own rate. The carbine: 12 rounds a second, 30 rounds a magazine, 3 magazines, 3 degrees of spread, 2 s to reload, reloads by itself when empty. | held |
| `throw_grenade_intent.dx_q`, `.dy_q`, `.fuse_ticks`, `.reach_q`, `.kind` | Throw a grenade. `kind` indexes the hand-grenade table (0 frag, 1 smoke, 2 tear gas, 3 poison, 4 flash, 5 incendiary; the rifleman carries 2 frags, 1 smoke, 1 flash). The verb is issued when the direction is not zero. The throw has a wind-up of 0.3 s, and the grenade leaves along the direction the body faces at that moment, to the given reach (at most 30 tiles), with the fuse counted from the release. | once, on the step's first tick |
| `use_intent` | Open or close the door the soldier touches. | once |
| `reload_intent` | Swap magazines now. The rounds left in the old one are lost. | once |
| `shove_intent` | Push the adjacent unit nearest the facing. | once |
| `melee_intent` | Strike with the rifle butt (20 damage, a short stun, a push). | once |
| `set_flashlight.off` | 1 turns the flashlight off. | held |

Facts about the actions that matter for the design:

- **The row is already "factored".** Moving, aiming, the trigger and one of the hand verbs can all be given in the same row. Nothing forces one action at a time.
- **The columns are raw control values, not choices.** A direction is two 17-bit integers, the fuse any whole number of ticks. A network does not output such numbers directly. A small translation layer between the network's choices ("aim at enemy 3", "throw a frag there") and this row is needed, and is where most of the action-space design will live. No such layer exists.
- **Aim is a direction, never a target.** To aim at an enemy, someone must compute the direction to it. The bots do that every tick and pull the trigger only when a predictor says the body will be on target within the weapon's spread. A policy that decides every 4 ticks holds one direction for 4 ticks while both soldiers move, and has no such predictor.
- **There are no masks.** The env does not say which verbs are possible now. A throw with no grenade left, a shove with nobody adjacent, anything while stunned: the game ignores it without a word.
- **Nothing uses the two explosives every rifleman carries** (A.8).

### A.4 Decimation, the end of a match, the reset

- One `step` is `decimation` ticks (a run-file number; the farm was measured at 1 and at 4). The held verbs are issued on every tick of the step, the once-verbs on its first tick.
- `terminated`: fewer than two marine teams have a living soldier. Zombies are not a team. `truncated`: the time limit ran out (300 s on `ship_demo`), which is a draw.
- When a match ends inside a step, the step stops there, the env starts that slot's next match at once, and the returned observation is the new match's first. The ended match's last observation and its result come back in `info`.
- Each match's level and seed are a pure function of (run seed, env number, episode number), so a run is reproducible and any episode can be replayed in the game window.

### A.5 The reward

Per soldier per step, the env counts seventeen things from the game's credited
hit and kill events, and the run file gives each a weight; the reward is the
weighted sum.

| counter | counts |
|---|---|
| `hits_dealt`, `damage_dealt` | damaging hits the soldier is credited with on another team's unit, marine or zombie |
| `team_hits` | its hits on its own team |
| `kills`, `team_kills` | the same split for kills |
| `hits_taken`, `damage_taken` | hits on itself, from anything |
| `deaths_<cause>` | its own death, by cause: bullet, laser, melee, explosion, heat, poison gas, burning, poisoned, suffocating, regen |

A grenade's kill is credited to its thrower. What the reward does not hold
today: winning or losing the match (the result is in `info` only), anything
about the team (a team-mate's kill gives a soldier nothing), time, ammunition
spent, progress toward an objective.

### A.6 Opponents and speed

The run file names who drives each marine team: `policy`, or a scripted
controller (`bots`, the layered bots; `bots_v1`, the first simple bots;
`idle`). Policy against bots, and policy against policy (both teams' rows are
in the same action array, and the trainer decides which network fills which),
both work today. The horde always runs its own controller.

Speed on the Home Desktop with 16 worker processes: about 1 400 match-ticks a
second on the small ship; the medium ship was last measured at 450 to 490,
before the latest farm improvement. At decimation 4 and 8 learning soldiers,
1 400 match-ticks is roughly 2 800 soldier decisions a second.

### A.7 What the layered bots' team picture holds

Erik wants the agent to see something like this, so here it is in full. Each
bot team has one shared `TeamBrain`.

**The picture (kept every tick).** For every enemy, marine or zombie, that any
team-mate sees: who it is, the tile where it was last seen, and the tick. An
entry stays for 6 seconds after the last sighting, then it is forgotten. An
enemy that dies is removed at once, seen or not.

"Sees" is the game's one sight model (`vision.py`), which exists and which
only the bots use: a soldier sees inside a **cone of 110 degrees** in front of
it (55 to each side) at **unlimited range**, and inside a **full circle of 4
tiles** (1.3 m) around it; walls block both; a team's vision is the union of
its soldiers'. Smoke does not block sight today.

**Four times a second** the brain recomputes:

| item | what it is |
|---|---|
| counts | team-mates alive; enemy marines known, and seen right now; zombies known, and within 24 tiles; rounds of ammunition left in the whole team |
| the play | one of three: *fall back* (zombies near, or outnumbered), *hold* (an enemy marine in sight), *bound* (otherwise: advance in two halves) |
| the shared target | among the enemies seen right now: the one most weakened by focused fire, then the one with least health, then the nearest. Everyone who can see it shoots it. |
| the objective | a tile. Fall back: the own spawn room. Hold: where the team stands. Bound: the shared target, else the nearest known enemy's last-seen tile, else the enemy spawn room. With it a distance map: steps to the objective from every tile. |
| the halves | the team split in two by roster order. In a bound, one half moves while the other stands and watches; they swap when the movers arrive. |
| the fire lanes | a coarse grid (cells of 3 tiles) marking the strip between each team-mate who fired last tick and its target. A bot avoids standing in another's lane. |
| who fires at whom | each team-mate's target of the last tick |

**Each bot** also keeps its own commitments: what its hands do (shoot, hold
fire, frag, flashbang, shove, melee), what its feet do (move, retreat, stand),
the tile it is walking to, and when it last threw. Team-mates read each
other's goal tiles to keep 5 tiles apart.

**To choose where to stand**, a bot asks eight yes / no questions of each
candidate tile: does it see the target; is it outside the team-mates' fire
lanes; is it spaced from team-mates; is it in cover from the threat; is it
hidden from the other known enemies; is it closer to the objective; is the
ground safe (no fire, dense smoke, vacuum, heat); can it be reached.

Two rules sit in the bots' code and not in the picture: a bot never fires
when a team-mate's body is near the line to its target, and it never throws a
frag within 8 tiles of a team-mate.

None of this reaches a policy today. The policy path and the bots share only
the game's control verbs.

### A.8 The explosives a soldier carries, and the missing plant verb

Asked for the C4 question ("tell me what it would take; do not build it").

What exists:

- Every rifleman already carries `explosives = 2`.
- The data rows exist: `[weapons.breach_charge]` (the door charge), `[weapons.c4_satchel]`, and `[payloads.demolition_c4]`: a blast of radius 8 tiles, 150 damage to a unit (a marine has 100 hit points), 800 to a wall, heard 60 m away. All placeholder values.
- A planted charge with a settable timer **was built**, for the turn-based mode: `simulation/charges.py` has a `PlantedCharge` (its tile, its owner, an absolute detonation tick or "remote"), planting as an action that completes only if the planter is not interrupted, and a detonation through the standard blast code, credited to the planter.
- It is reachable only through a turn-based *order*, and its clock is advanced only by the turn-based timeline. On the continuous path, which is what matches, bots, the human's direct control and the RL env all use, there is no verb, the match recorder refuses orders, and nothing counts a planted charge down.

So "no plant verb has ever existed" is not quite right: one exists and never
reached the continuous path.

What a continuous-path C4 would take, as a list and not a plan:

1. A new control verb, "plant a charge with this fuse", through the one tap all verbs pass. The action column and the replay record then follow by themselves; the recording format gets a new version.
2. Design answers that are Erik's: how long planting takes and what interrupts it; where the charge sits (the soldier's own tile, or the wall or door it faces); whether it can be seen, shot or disarmed by the other side; breach charge and C4 as two kinds or one.
3. The charge's clock on the continuous tick. The project already has one canonical "timed charge" system for level-authored charges; the planted charge should become a client of it and not a second timer.
4. Its state in the synced game state and in the determinism digests (the planted list is not hashed today, as far as I found), and in the unit schema: the count carried and the planting in progress. This is why it should be decided before the residency port's P3a, which freezes that schema.
5. Everyone must be able to see it: a model or marker in the picture, a key for the human, an entry in the agent's observation (where, whose, seconds left), a rule for the bots to plant one and to run from one.
6. A human test, since it changes how a match plays.

In size this is close to the grenade-kinds patch (G3p): one implementation
patch and one play test.

---

## Part B. How other groups built observation and action spaces

*(being assembled from the research digests)*
