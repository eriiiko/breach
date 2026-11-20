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

Six systems, then what changed since, then what it adds up to for each open
question. Every claim below comes from a digest, which names its source and
how much of it was read; B.9 says what was checked and what is recalled.

More words, used from here on:

- *Entity list*: the observation as a list of things (each soldier, each enemy, each object), one row of numbers per thing, instead of a picture.
- *Embedding*: the vector a network computes for one item of such a list.
- *Attention*: a network layer that, for each item, looks at all the others and weighs the ones that matter. It does not care about the order or the number of items. A *transformer* is a stack of such layers.
- *Pointer*: choosing an item from the list as the answer. The network forms a "query" vector from its state, scores every item's embedding against it, and picks by those scores. This is how "attack that unit" is expressed.
- *Head*: one output of the network, giving the choice for one part of the action.
- *Mask*: a list of which choices are legal right now; illegal ones get probability zero.
- *LSTM* / *GRU*: a memory. A vector the network carries from one decision to the next.
- *Ablation*: rerunning without one part to measure what that part bought.
- *Self-play*: training against copies of oneself, present or past.
- *Imitation* (behaviour cloning): training a network to copy recorded play, with no reward involved.
- *Elo*: a rating scale; 100 points more means winning about 64 % of games.

### B.0 At a glance

| | who decides | units seen as | map seen as | target / aim | choices per step | team-mates share | policy sees / critic sees |
|---|---|---|---|---|---|---|---|
| OpenAI Five | one net per hero, same weights | list of 189 units, pooled | 8 x 8 terrain patch | point at a unit, or a cell of a 9 x 9 offset grid | one verb + its arguments | the whole team's sight, by game rule | team sight / the same |
| Honor of Kings | one net per hero, same weights | unit vectors, pooled | 6 planes of 17 x 17 | point at a unit ("target attention") | one verb + who + how | team sight; in one model, allies' destinations | team sight / everything |
| AlphaStar | one net for the whole army | one list of up to 512 entities, transformer | 128 x 128 planes, units written into them | point at a unit, or a cell of a 256 x 256 grid | one verb + chained arguments | (one mind) | fog of war / the opponent's view too |
| Hide-and-seek | one net per agent, same weights | entity list with attention, plus 30 range rays | none (the rays) | turn by a torque, 11 levels | move, turn, grab, lock at once | nothing | 135 degree cone and walls / everything |
| Capture the flag | one net per agent, a population | none: 84 x 84 pixels | pixels | turn by 10 or 60 degrees | six choices at once | nothing | the camera / the same |
| MAPPO on SMAC | one net per soldier, same weights | fixed blocks in a flat vector | none | "attack enemy k"; the game aims | one choice of all | allies within sight | a sight circle / the whole state |

### B.1 OpenAI Five (Dota 2)

Berner et al. 2019, arXiv:1912.06680. Five heroes a side. One network, copied
five times with the same weights, each copy with its own memory.

- **Units.** A list of up to 189 units per step, grouped (allied heroes, enemy heroes, other allied units, other enemy units, neutrals). Each unit is a row: type, health over the last frames, position, facing as cosine and sine, "is it attacking me", "am I attacking it". Each row goes through the same small network and the results are pooled. The only map is an 8 x 8 patch of terrain near each hero. About 16 000 numbers a step.
- **Team-mates.** Dota shares sight across a team by its rules, so the five copies get nearly the same input. On top, a quarter of each copy's internal vector is pooled across the five. There is no message channel. A hero sees whether an ally is attacking, but not whom.
- **Hidden things.** A unit in fog is not in the list. A hidden enemy hero is kept as it was last seen, with a "time since last seen" number: a last-seen record, built by hand, shared by the team.
- **Actions.** One verb per step (move, attack, each ability, each item), chosen among the ones available now, plus arguments: a target unit, an offset on a 9 x 9 grid around the caster or the target, a delay. The target is chosen by pointing. Arguments the chosen verb does not use are left out of the learning signal. It acts every 4th frame (7.5 times a second), with a reaction time of 167 to 267 ms.
- **Reward.** Heavily shaped (win 5, death -1, plus last hits, health, towers and more). "Team spirit": each hero's reward is a blend of its own and the team's average, moved from 0.3 team to 0.8 team during training.
- **Many hero types.** One network, the hero's identity as an input, a pool of 17. With 80 heroes early training ran about 20 % slower.
- **What it cost.** Ten months on up to 1 536 GPUs. The network was altered more than twenty times mid-run by "surgery" as the game and the inputs changed. A clean rerun on the final design took two months and a fifth of the compute, and beat the original in 98 % of games. Their conclusion: when the environment is understood from the start, design the interface first.

### B.2 Honor of Kings (Tencent)

Ye et al., AAAI 2020 (arXiv:1912.09729) and NeurIPS 2020 (arXiv:2011.12692);
Wu et al. 2018 (arXiv:1812.07887); Gao et al. 2021 and 2023 (arXiv:2110.14221,
2304.11632). The same kind of game as Dota. It is here because it is the one
line of work that measured what each part bought.

- **Units and map.** A feature vector per unit, units of one type pooled; small map planes beside them (6 planes of 17 x 17 in the 5 v 5 agent).
- **Target.** The same pointer, which they call target attention. In the 1 v 1 ablation, each part was added alone to a base agent and played against it: the pointer won 75 %, memory 73 %, masks 50.5 % (no stronger, but trained in 65 hours instead of 80), all three together 90 %.
- **Actions.** What (the button), who (the target), how (a direction or an offset). The heads are chosen independently of each other, with one exception: the direction depends on the skill.
- **Critic.** The value network sees what is hidden from the policy, in training only. They report it raised strength, without a number.
- **Team-mates' intentions.** In the 2018 model each hero is told which of 144 map regions each ally is heading for. With that it beat the same model without it 62.5 % to 37.5 %. That model was trained by imitation of human games, not by reward.
- **Orders.** Two later papers give the hero an order as an input: a place on the map, what to do there, and a time limit. Obeying is paid by a reward for closing the distance, with a weight. At the weight where the agent completes 82 % of orders (humans complete 80 %) it still wins about half its games against the strongest earlier agents. Pushed toward 100 %, "this will significantly reduce the win rate", because the order is sometimes wrong.
- **Many hero types.** Forty heroes in one network failed with plain training: 480 hours and still 400 Elo short. What worked: train teachers on fixed line-ups, copy them into one student network, then keep training on random line-ups: 336 hours. The final network has 17 million weights against OpenAI Five's 159 million.

### B.3 AlphaStar (StarCraft II)

Vinyals et al., Nature 575, 2019. One network plays a whole army, so it is a
commander and not a soldier. It is read here for its parts.

- **Units.** One list of up to 512 entities, own, enemy and neutral together, with a field that says whose each is. Per entity: type (one of 256), health in buckets, position written in binary digits, weapon cooldown, its current order, "was targeted by my last action". A transformer of three layers reads the list.
- **Map.** Planes of 128 x 128 (height, visibility, walkable and so on) through a convolution network. "Scatter connections" write each unit's embedding into the map at the unit's position, so the map branch knows who stands where.
- **Actions.** A verb with arguments, chosen in a chain where each choice is fed to the next: the action type (several hundred; every ability is its own type), a delay, which of my units, then a target unit or a target place, never both. The target unit is a pointer whose query includes the chosen verb. A place is a choice among the cells of a 256 x 256 grid, not two coordinate numbers.
- **No aiming.** "Attack unit k" is carried out by the game's own unit logic.
- **Hidden things.** The policy sees what a human player would. During training the value network also sees the opponent's whole view.
- **A strategy as an input.** The policy takes a summary of a strategy, called z (a build order and what was built), taken from a human game. It is blanked 10 % of the time during imitation training, and in self-play following it earns a side reward that is switched on in a quarter of the games. With z fixed by hand the same network plays 17 chosen strategies on demand. This is the precedent for an order input.
- **What the parts bought** (one map, one matchup). Built up in this order, the imitation-trained agent's win rate against the built-in elite bot: baseline 0 %, with action delays 7 %, with the pointer 36 %, with the entity transformer 71 %, with scatter connections 87 %. The value network with the opponent's view: 82 % against 22 % without, the largest single training effect in the paper. Without human games: Elo 149 against 1 540.
- **What it cost.** 971 000 human replays; a league of 12 agents, each on 32 accelerators for 44 days. A later replication at a fraction of the compute (TStarBot-X, arXiv:2011.13729) found that without the transformer the agent "barely control[s] one or a few units", and that PPO did about as well as AlphaStar's own training rule.

### B.4 Hide-and-seek (OpenAI)

Baker et al. 2019, arXiv:1909.07528. Two to six agents with bodies in a
physics room. The closest in spirit to Breach.

- **Units.** An entity list (the other agents, boxes, ramps: position, velocity, size) read with attention, and beside it 30 range rays all round the body, like a lidar, for the walls. A memory. 1.6 million weights.
- **Sight.** An entity outside a 135 degree cone in front of the agent, or behind a wall, is masked out of the list. Team-mates too: an agent does not always know where its own team is. The rays are not masked.
- **Actions.** Three choices at once each step: a push along x and y and a turning torque (11 levels each), grab, lock. Turning is relative, on a body with inertia, as in Breach.
- **Team-mates.** The same weights, a team reward, no sharing of any kind.
- **Critic.** The training-time value network sees the whole room. Rerun with a value network limited to the agent's own view, training "never progressed past stage 3" of six.
- **Attention.** It learned from fewer samples than a network that only pooled the entities.
- **What it cost.** 31.7 billion frames to reach the fourth of its six strategies. At the farm's speed today that is about 260 days.

### B.5 Capture the flag, "For The Win" (DeepMind)

Jaderberg et al., Science 2019, arXiv:1807.01281.

- **Observation.** First-person pixels, 84 x 84. No entity list, no map, nothing about team-mates beyond what the camera shows.
- **Actions.** Six choices at once: turn left or right by 10 or 60 degrees, look up or down 5 degrees, strafe, forward or back, fire, jump. 15 decisions a second.
- **Team-mates.** No communication. Each agent learns on its own, in a population of 30.
- **Results.** It hit with 80 % of its shots against humans' 48 %, and reacted in 258 ms.
- **What it cost.** Two billion steps per agent. The paper has no ablation of the interface.

So turn-left / turn-right from pixels does work, at a price far beyond ours,
and this system has nothing to say about team information.

### B.6 MAPPO on SMAC

Samvelyan et al. 2019 (arXiv:1902.04043); Yu et al. 2021 (arXiv:2103.01955);
SMACv2, Ellis et al. 2022 (arXiv:2212.07489). SMAC is a benchmark: each
soldier of a small StarCraft squad is its own agent, against the game's
scripted army. MAPPO is PPO with the same weights for every soldier and a
value network that sees more.

- **Units.** A flat vector. For every ally and every enemy inside a sight circle there is a fixed block: distance, relative x and y, health, shield, type. An enemy's block starts with "can I attack it now". Everything is relative to the soldier itself. By default there is no map at all.
- **Actions.** One choice per step out of: nothing, stop, move north / south / east / west, attack enemy 1 ... n. A mask says which are possible; attack k only when k is alive and in range.
- **Aiming.** None. After "attack k" the game walks into range, turns and fires.
- **Why "enemy k" can be learned there.** Slot k is the same enemy for the whole episode, the mask removes impossible targets, and (in the networks that work best, such as UPDeT, arXiv:2101.08001) the score of "attack k" is computed from enemy k's own embedding. When UPDeT's authors broke that tie on purpose, the network fell below the simple baseline.
- **Team-mates.** Allies in sight appear as blocks. Showing the allies' last actions is an option, and the benchmark's own runs left it off. No channel.
- **MAPPO's findings.** The best value-network input is the whole state plus the soldier's own view, with duplicated features removed (removing them alone moved one hard map from 63.3 % to 84.4 %). For a dead soldier: zeros plus its id. Shared weights against a separate network per soldier, on a map with three unit types: 90.6 % against 13.0 %. Reuse each batch of data 5 to 15 times.
- **SMACv2's two findings.** On the original maps, with their fixed start positions, a policy that sees only the clock does respectably on many: agents memorise a script. So starts and team make-up must be random. And when only the soldier who spots an enemy sees it, the standard methods "struggle to attain even modest performance"; with sight shared across the team they "come close to solving the task". Team-mates did not learn to make up for unshared sight.
- **An unused option.** SMACv2's code has a switch, off by default and absent from the published results: twelve extra actions that each set the soldier's facing to one of twelve fixed directions, with sight and fire limited to a 30 degree cone.

### B.7 Since then

- **Lists and attention became the norm.** A transformer over entities replaced pooling. Nothing newer has replaced the pointer for choosing a target.
- **Industry shooter bots mostly keep aiming out of the learned part.** A 2024 model of professional Counter-Strike movement (arXiv:2408.13934) learns where to walk, with all ten players as a list, and leaves aiming to rules. Tencent's Arena Breakout (arXiv:2410.04936) writes that "pure RL alone cannot effectively control firearms" and sets the firing point by rule; that hybrid wins about 30 % more than pure learning. EA's Battlefield 6 work (arXiv:2606.20210) learns locomotion only. Riot's bots (arXiv:2501.00078) do learn aim, by imitation, as one choice on a 15 x 11 grid of directions.
- **Two good targets break a single-number aim.** Three groups ran into it: an output that is one angle lands between two enemies. Each fixed it by making the aim one choice among discrete options.
- **Commanders are now often language models.** In Honor of Kings (2025, arXiv:2508.21365) one picks among about 40 team objectives; in an EA shooter prototype (2026, arXiv:2606.20014) one picks among four skills per soldier twice a second. Unit control underneath is unchanged. In the EA work, learning with no commander layer lost clearly to both the rule bots and the hybrid, and the skills took no target as a parameter, which the authors name as the next step.
- **Squad methods.** The strongest current cooperative method (Sable, arXiv:2410.01706) is best on 34 of 45 tasks, but it is one network running the whole squad with every soldier's view, and on the SMAC-like tasks it leads plain independent PPO by one point.
- **Trainers.** The digest found no 2025 or 2026 result that beats PPO in competitive team games. A large 2026 study (arXiv:2502.08938, abstract only) finds plain PPO matching or beating the game-theory methods in two-player hidden-information games. The two ready-made trainers July's survey named no longer fit: PufferLib 5 takes only environments written in C, and Sample Factory has had no release since 2023 and does not run on Windows.
- **Tsetlin machines.** Their published reinforcement-learning results are grid worlds up to 10 x 10 and CartPole. They need yes / no inputs. The bots' eight position questions (A.7) are exactly that shape.

### B.8 What it adds up to, question by question

**1. How to show units, team-mates and the map.** Every system since 2019
that handles many units shows them as a list read by attention, plus a small
map. The measured gains are large (AlphaStar: 36 % to 71 % from the
transformer; hide-and-seek: fewer samples; the industry movement model got
worse without it). The gap: none of these games has Breach's walls and physics
fields at Breach's weight. The benchmarks closest to our soldier (SMAC and its
successors) have no map at all. AlphaStar's scatter connection is the one
published way to join the list to the map.

**2. Aiming.** Every many-unit system picks the target by pointing at it.
Every system that turns left and right is a first-person game played from
pixels, and paid billions of steps. No paper compares the two on one task.
Erik's leaning, "aim at enemy k", is the many-unit systems' choice, with one
difference to keep in view: there the game does the aiming once the unit is
picked. In Breach the turning law can play that part, because it already turns
a body toward a set point; when to pull the trigger stays a decision. For a
direction with no enemy in sight the precedents are a small set of fixed
directions (12 in SMACv2's option, 16 in the Counter-Strike movement model) or
a cell on a grid (OpenAI Five, AlphaStar). A pointer can only point at things
in the list, so doors, corners and last-seen positions must be in the list, or
be reached by direction. A single continuous angle is the one form with
evidence against it.

**3. Factoring the actions.** Two families. Commander-like agents choose one
verb per step with arguments. Body-like agents (hide-and-seek, capture the
flag, the shooters) make several choices at once: move, turn, fire. A Breach
soldier is the second kind, and today's action row already is. Each grenade
kind as its own action matches how every ability is its own verb in Dota and
StarCraft; it should be one choice among "nothing, frag, smoke, flash, ...", so
that a soldier cannot throw two at once. Heads chosen independently worked in
both MOBA systems; the one dependency they kept is that the target or
direction depends on the verb. Masks bought training time in Tencent's
ablation, and a later Honor of Kings study calls them essential.

**4. What team-mates share.** Shared sight, by rule, is in every strong team
system and has the clearest measurement (SMACv2). Allies' state is always
there as list entries. An ally's intention was shared in one system only
(Honor of Kings, the destination), was worth 62.5 to 37.5, and was fixed and
readable. Learned message channels have weak or negative results in combat
tasks. "Whom is my team-mate aiming at" appears in no system the digests read.
Friendly fire appears in no landmark system either. On both, Breach would be
on new ground, with the bots' picture as the only worked example.

**5. An order input for a later commander.** Every system that ended with an
obedient lower level had the order as an input while that level was learning.
The orders came from a generator, from a scripted layer, or from labelling
what the unit actually did, never from the final commander. Adding inputs
later is mechanically easy and cost OpenAI Five strength. The orders that
worked were small and concrete (a place among 144, a target among 3, fight or
flee) and were held for seconds. Obedience is a dial; near 80 % kept strength.

**6. Fog of war and field of view.** The pattern with the largest measured
effects in the whole survey: the policy sees only what is legal, the
training-time value network sees everything (AlphaStar 82 % against 22 %;
hide-and-seek's stall without it). A cone exists in hide-and-seek only and was
never ablated; that it produces arc-covering or corner-checking is a hope, not
a result. Starting with full information and removing it gradually has one
good precedent (Suphx, arXiv:2003.13590). Copying a teacher that sees through
walls into a student that does not is a known trap.

**7. Many unit classes.** One network with the unit's type as an input and
masks for the verbs it lacks works inside a family that shares its verbs (17,
40 and 102 heroes; three unit types in SMAC). Where bodies and verbs differ
wholesale, practice splits the networks (AlphaStar: one per race). No study
compares the two when verb sets truly differ. With many classes the proven
route is specialists first, then one student.

**8. The trainer.** PPO, in the MAPPO arrangement: same weights for every
soldier, a value network that sees everything, several passes over each batch
because our simulator is slow. Opponents: mostly the current self, a share of
past selves, a share of the scripted bots. Progress: a rating against a fixed
pool that includes the bots, through the existing bot tournament. Written by
ourselves against `BatchedEnv`, in about two thousand lines. The first
learning tasks must be SMAC-sized (about ten million steps, a figure recalled
and not re-read; under a day on the farm), not hide-and-seek-sized.

### B.9 What was checked and what is recalled

- **Checked by me.** All of Part A, from the code, with the schema functions run. Two claims of Part B, re-read at the source after the digests came in: MAPPO's shared-against-separate table, and the facing-direction option in SMACv2's code.
- **Read by the digests, in full or in the relevant parts.** OpenAI Five (the appendices and OpenAI's 2018 architecture diagram); the Tencent papers named in B.2; the AlphaStar paper with its Methods and its supplementary architecture text (the latter from an unofficial mirror); hide-and-seek (the paper and its released code); the capture-the-flag arXiv text; SMAC, MAPPO and SMACv2 (papers and code); UPDeT and three related papers; the shooter papers of B.7; the PPO references behind B.8 item 8.
- **Weaker, and marked so in the digests.** AlphaStar's ablation numbers were read from the figure's text layer, and which label goes with which bar is the reader's pairing (two cross-checks agree). The SMAC and SMAX code facts came through a summarising reader, not verbatim. The 2025 and 2026 language-model commander papers, Sable's details, the 2026 PPO study and most Tsetlin sources are known from abstracts. Capture the flag's network internals and AlphaStar's league details are from memory.
- **Looked for and not found.** A test of turning against pointing on one task. A test of a cone against all-round sight. A test of a hand-designed team display against learned messages. A shooter soldier trained to take orders. A comparison of one network against one per class when the classes' verbs differ. Any representation shown to make avoiding friendly fire learnable.

That last list is where Breach's design cannot lean on anyone, and where our
own experiments have to be able to fail.

---

## Existing systems the design will have to use

Not a design section; a reminder of what already exists, so nothing parallel
gets built. All are rows in the project's canonical-systems table.

- **Batched environment** and **MatchSession + the `policy` controller**: the one env and the one way a policy reaches the game. A new verb is a row in `INTENT_KINDS` and in `VERB_RULES`.
- **The sensor accessor** (`EntityFieldAccessor.crop`): the one door for observations. An entity list is a new product of that door.
- **The sight model** (`vision.py`, `GameMap.has_los`): the one answer to "who sees whom".
- **The turning law** (`turning.set_target`, `Simulation.facing_after_turn_q16`): already takes a set point and already predicts where a shot will go.
- **Bot layers** (`TeamBrain`) and **Position features**: the only worked team picture, a ready source of orders, and the yes / no list a Tsetlin machine could read.
- **Credited cause** and the reward counters; **Match recording / replay** (how a trained agent is watched); the **Bot tournament** (how it is measured).
- **Loadouts**, **Grenade kinds**, **Timed charge** and `charges.py` (the C4 question).
