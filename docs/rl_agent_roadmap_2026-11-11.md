# RL agent roadmap: the ladder

2026-11-11. **v0, agreed in outline by Erik the same day** ("everything
sounds very very good"); open to his edits. A plan in rungs for Breach's
learned agents. Each rung ends at a gate that can fail, and each is a place to
stop and still own something that works.

It replaces Track 2 items 2.2 to 2.4, Track 3 and milestones M2 to M4 of
`docs/roadmap_2026-07-30_rl_push.md`. Everything else there stands. Erik,
2026-11-11: any earlier decision may be overturned where we see worth in it,
and other ideas may be pursued along the way. This is a plan for what to do
next, not a fence.

The reasons behind the rules are in `docs/rl_agent_survey_2026-11-11.md`,
cited below as "survey A.x / B.x".

## 1. Where we stand

- **Built:** the batched environment (about 1 400 match-ticks a second on the small ship), the control verbs through one tap, two scripted bot teams, the horde, generated ships, replay in the game window, credited hit and kill events, the bot tournament.
- **Placeholder:** the observation and the reward (survey A.2, A.5).
- **Missing:** the layer from a network's choices to the control verbs, the masks, a trainer, a rating.
- **Decided by Erik, 2026-11-11:** soldiers first, the commander later. The game is real time. A soldier aims by pointing at enemy k or at a direction, and pulls its own trigger, shown the bots' prediction of whether the shot will be on target. One action per grenade kind. A team display. C4 comes to the continuous path in a separate patch of Erik's; this plan assumes it is there when needed. Different network designs and action spaces will be tried side by side.

## 2. What we keep from the survey

The rules of the build. Each is a result of the survey, and each can be
overturned by a measurement of our own.

1. **Units are a list, read by attention; the map is a small window for walls and physics fields** (survey B.8.1).
2. **The soldier points at its target, the game turns the body, the soldier pulls the trigger** (B.8.2; the trigger is Erik's ruling). No single continuous aim angle.
3. **Several choices at once: feet, aim, trigger, one hand action.** Masks on every choice, computed from the game's own rules (B.8.3).
4. **A team's sight is shared by rule, with last-seen records. Intentions are shown in a fixed, readable form, not as learned messages** (B.8.4).
5. **The order input exists in the first trained network.** Orders are small, concrete and held for seconds. Obedience is a dial, not a switch (B.8.5).
6. **The policy sees only what is legal; a training-only value network sees everything** (B.8.6).
7. **A unit's class is an input, and a mask removes the verbs it lacks.** One network per family of bodies that share their verbs (B.8.7).
8. **PPO in the MAPPO arrangement, written by us against `BatchedEnv`.** Strength is measured against a fixed pool that includes the bots (B.8.8).
9. **Interfaces are named and versioned.** An observation layout, an action layout and a reward profile are each a named spec chosen by the run file. A new idea is a new spec, and the old one keeps running. Agents trained on different specs can still fight each other, because they meet only through the game's verbs.
10. **No game number is baked into an interface.** Features are ratios of table values (health over full health, rounds over magazine size), so re-tuning hit points, armour or weapons never changes a spec. A trained network belongs to the snapshot it was trained on.
11. **Small tasks first.** The first learning tasks take about ten million decisions, not billions. Ships and spawns are random from the first serious run (B.6, SMACv2).
12. **Network numbers never enter the synced game.** Choices go in as integer verbs; every match replays.
13. **Every rung opens with a short lesson document** for the method it introduces (the standing rule of the July roadmap).
14. **Every claim we rely on becomes an experiment that can fail**, above all on the ground where the survey found nobody to lean on: friendly fire, "whom is my team-mate aiming at", a cone of sight, orders to a shooter soldier, classes with different verbs (B.9).

## 3. The ladder

| rung | name | ends when |
|---|---|---|
| 0 | The design | the specs are written, critiqued and agreed |
| 1 | The door | a network could be plugged in: every input and output exists and is tested, with no network yet |
| 2 | First light | one soldier learns to shoot, then to win a duel, and Erik watches the replay |
| 3 | A fireteam that fights | four soldiers beat the layered bots on ships they have never seen |
| 4 | Orders | the fireteam does what it is told about four times in five, and is no weaker for it |
| 5 | Scale | the same training runs on the GPU-resident backend and on Berzelius |
| 6 | The commander | a learned commander beats the scripted team layer over the same soldiers |
| 7 | Unit classes | other kits and other bodies on the same door |
| 8 | Senses | a cone, hearing, smoke that blocks sight, light |
| 9 | Other learners | Tsetlin machines, imitation of human play, populations |

Rungs 0 to 4 are in order. Rung 5 runs beside them (section 4). Rungs 6 to 9
are in no fixed order: take them by motivation.

### Rung 0. The design

- **Goal.** Write down exactly what a soldier sees and what it can choose.
- **Product.** A design document on this branch: observation spec v1 (every list, every feature and its encoding), action spec v1 (every choice, its mask, and how a choice becomes verbs tick by tick), the order format v1, reward profile v1, what the value network sees, and the list of unit state the design adds. Then the critique passes, one critic at a time.
- **How.** The discussion, one question at a time. Settled so far: the skeleton and the trigger. Next: what is in the list; the aim entries; the feet; the hand actions; the team display; the order; the reward; sight.
- **Gate.** Erik agrees; the critique is closed.
- **Lesson document.** How a policy learns from reward, in plain words (PPO): `docs/rl_lesson_0_learning_from_reward_2026-11-11.md`.

### Rung 1. The door

- **Goal.** Everything a network needs from the game, proven without a network.
- **Patches**, each with property tests, one at a time. Re-cut on 2026-11-11 after the design's critique (design document section 12), which found three things the first cut of five patches had no place for (the room graph, the sight setting, the verbs' gates) and one patch in the wrong place (the observation used the order before the order existed):

  | patch | what | gate | changes play? |
  |---|---|---|---|
  | **1a** | **Sight as a setting of the match:** a key in the match file, cone or all-round, default the cone. | every pinned match bit for bit | no |
  | **1b** | **The room graph saved with the level:** rooms, openings and a tile-to-room map, from the generator through the level writer; which openings are passable now is derived. | the saved graph equals the generator's; a level without one is refused for training | no |
  | **1c** | **Sight for many tiles, in C++,** and the per-tick sight test no longer run for zombies as observers. | equal to the existing sight test tile for tile; the farm's speed measured; touches the engine's build, so it needs the CUDA build green as well | no |
  | **1d** | **The team picture as a shared system,** in two steps. (i) What a team knows of the enemy moves out of the bots' brain into one place, unchanged. (ii) Records become honest (they outlive an unseen death), the fire lanes are read from state, and the bots are moved onto both. | (i) the layered bots bit for bit against a reference match taken before the patch; (ii) the bot tournament | (i) no; (ii) yes, the bots. Erik blessed it in advance (2026-11-11: "i dont need to look at 1d, i bless it all ready now"), so the tournament's numbers are the gate and it merges on them |
  | **1e** | **The verbs' gates lifted** into read-only functions that the verb and the mask both call. | the game's digests in lockstep before and after | no |
  | **1f** | **The order:** its four fields as unit state, the commander's verb (recorded; a new recording format version), the bots' team layer writing it for a team driven through the door. | a match of bots alone plays and hashes as before | no |
  | **1g** | **Observation spec v1 and the value network's full view** through the sensor accessor, with the two row flags, in the farm's buffers. | equals an independent reconstruction; never shows what the team cannot know; the farm's speed measured | no |
  | **1h** | **Action spec v1, its decoder and the masks,** inside the `policy` controller. | a masked choice issued by force does nothing in the game | no |
  | **1i** | **Reward profile v1:** the counters, the five numbers, the clock as an ending. | the books close (design 6.8) | no |
  | **1j** | **The bots through the door,** and their play saved as records for imitation. | see below | no |

- **The gate that matters: the bots through the door.** Drive the layered bots through action spec v1 at the agent's decision rate, each of their decisions translated to the nearest choice, and play them against the unconstrained bots in the tournament. If they lose badly, the action space or the decision rate is too coarse, and we learn it before any training. The same run yields the bots' play as (observation, choice) records, ready for imitation.
- **Outcome (2026-11-13 / 14).** All ten patches merged. The bots through the door won 0.27 to 0.37 of the decided duels on held-out seeds against the lead's mark of 0.40; the cost is reaction between decisions, not the choices. Erik, 2026-11-14: the test counts as passed, the rate stays at 4 ticks, the free bots stay as they are (design document 13.1). Rung 1 is closed.
- **Erik decides.** Nothing by eye. Step (ii) of 1d changes what the layered bots know (no more word of deaths nobody saw) and where they see fire lanes, so their play may shift; Erik blessed that in advance (2026-11-11), and the tournament gives the numbers. Everything else changes no behaviour of the game.

### Rung 2. First light

- **Goal.** A learning curve that rises, and a learned soldier Erik can watch.
- **Built.** The trainer (rule 8), network v1 (section 5), the trainer's own tests (identical choice probabilities at collection and at training; a masked choice never sampled; a known toy task solved), a checkpoint store, the rating against a fixed pool, the export of a training episode as a replay.
- **Tasks, in order** (re-ordered 2026-11-14, Erik: "I lean on copying first too"; design document 13.3).
  - **Copying the bots:** network v1 learns to predict the door bots' recorded choice from the recorded observation (rung 1's records; no game runs while it trains). The first test of the network, its masks and its pointer heads, and the first network-driven fireteam to watch.
  - **Shooting range:** one soldier, standing targets, one room. Minutes of training. The trainer's first proof, from scratch, without the memory.
  - **Duel:** one against one first-generation bot, random rooms. About an hour.
  - **Experiment:** the duel from the copied network against the duel from scratch: does copying shorten the path, or leave habits to unlearn? Erik's ruling in the residency design (12.3) wants the bots to bootstrap early training.
- **Waits for** the residency arc's P6 (the unit code, #29) before any training run (Erik, 2026-11-14); the trainer's design document does not.
- **Erik's note for the trainer's design (2026-11-11).** The trainer keeps collecting and learning behind a boundary, with a weights version on every batch, so that one run can later be spread over Erik's three PCs on Tailscale. Until then the three PCs run three experiments at once and share saved networks as opponents.
- **Gate.** On rooms it has never seen, the soldier wins more than half its duels against the first-generation bot, and the replay shows it aiming, firing and moving on purpose.
- **Lesson document.** The trainer, line by line.

### Rung 3. A fireteam that fights

- **Goal.** Four soldiers with one network play a whole match as a team.
- **Built.** The team reward blend. Opponents: mostly the current self, a share of past selves, a share of both bot generations, the horde in every match. Random ships from a level bank, with held-out ships for measuring. Grenades by kind. C4 when its verb lands.
- **Experiments, each able to fail.**
  - The team display: sight shared and nothing more, against sight plus readable intentions (each team-mate's target, "am I in a team-mate's line of fire").
  - Friendly fire: credited team hits per match, against the bots' own number.
  - The decision rate: 4, 6 and 8 ticks.
- **Gate.** On held-out ships the fireteam beats the layered bots in more than half of the decided matches, with no more friendly-fire hits per match than the bots. The result against the bots through the door (`bots_door`, the equal-reaction opponent) is printed beside it (2026-11-14, design document 13.1): beating those and not the free bots means a reflex gap, and the decision rate is then re-opened.
- **Erik decides.** How it plays, by eye, in replays: this is the first rung with feel in it.

### Rung 4. Orders

- **Goal.** The fireteam takes orders from any seat.
- **Built.** Orders in training from three sources: the bots' team layer, a generator of random legal orders, and labels of what a soldier did in fact do. A reward for following, with a weight that is swept. A way for a human to give the same orders in the game window.
- **Gate.** About four orders in five completed, at no loss against rung 3's fireteam in the tournament. Then Erik leads a learned fireteam in the window.
- **Erik decides.** The order vocabulary beyond the first few, and how obedient is right, by play.

### Rung 5. Scale

- **Goal.** The same training on the GPU-resident backend (the residency arc, #29).
- **Gate.** The arc's own gates, then: a fireteam trained on the device backend beats the farm-trained one.
- **Then.** Longer runs, larger networks (section 5), Berzelius.

### Rung 6. The commander

- **Goal.** A second network that gives the orders.
- **Built.** A commander that sees the team's whole picture and writes one order per soldier every second or two, over soldiers that are frozen at first.
- **Experiment: the tournament of architectures** the July documents wanted. The same soldiers under no commander, the scripted team layer, a learned commander, and a language model as commander.
- **Gate.** The learned commander's team beats the scripted team layer's team.

### Rung 7. Unit classes

- **Goal.** More than one kind of soldier, then more than one kind of body.
- **Order.** Kits first (a flamethrower, armour, a different weapon: same verbs, different numbers). Then bodies with other verbs (a learned zombie, the octopus, robots).
- **Experiment.** One network with the class as an input, against one network per class, where the verbs differ. Nobody has published this comparison.
- **Method when classes multiply.** Train specialists, then copy them into one student.

### Rung 8. Senses

- **Goal.** What a soldier may know becomes part of the game.
- **Experiments.** A cone of sight against all-round sight, brought in gradually. Noises as list entries. Smoke that blocks sight. Light as game state (#81), so that a flashlight is a decision.

### Rung 9. Other learners on the same door

- A Tsetlin machine over yes / no features: first as the scorer of the bots' position questions (Erik's idea), then as a chooser of orders.
- Imitation of human play: every hour played is already recorded as verbs.
- Populations and leagues, if the rating shows strategies going in circles.

## 4. Around the ladder

- **The residency arc (#29).** It was set to wait for this design because its patch P3a fixes the definition of a unit's state. That definition is built to be extended (a new column changes no digest until a unit uses it), so P3a needs only rung 0's list of unit state, not the rungs after it. That list is the design document's section 9, which the critique widened: it now names the unit state the observation *reads* as well as what the design adds, since all of it must be a device column. **Erik, 2026-11-11: P3a restarts when rung 0 is agreed**, and runs beside rungs 1 and 2.
- **C4.** Erik's separate patch, landed 2026-11-11 as #193 (`plant_charge_intent`; the design document's sections 2, 3, 4 and 9 are written against it). The agent side needed three things from it, all now there: the verb in the verb table (its action column then follows by itself); a planted charge listed where the observation can read it (where, whose, ticks left); the count carried and the planting in progress as unit state.
- **Light (#81), sight through smoke, noise the soldiers hear:** rung 8 needs them; nothing before does.
- **GitHub.** The design session is issue #192; rung 1 is issue #195, run from `docs/rl_agent_rung1_plan_2026-11-11.md`. When the ladder is agreed in detail, each rung becomes a milestone or an issue and the pinned roadmap issue gets the order.
- **Gameplay the design leans on,** Erik's to order, listed in the design document's section 11: a reason to walk (footsteps and spread by speed), stances, revive, drag, pick up / put down, a general use.

## 5. Network sizes, and what a run costs

What comparable agents used, from the survey's digests:

| agent | weights | what it did |
|---|---|---|
| MAPPO on SMAC | under 0.1 million (my estimate from its 64-unit layers, not a quoted figure) | squad micro-management on fixed small maps |
| hide-and-seek | 1.6 million | six emergent strategies with tools |
| Counter-Strike movement model | 5.4 million | team movement of ten players |
| Riot's shooter bots | 15 million | human-like 2 v 2 play, 10 ms a decision on a CPU |
| Honor of Kings, 40 heroes | 17 million | a full 5 v 5 game at professional level |
| OpenAI Five | 159 million | Dota 2 at world-champion level |

The plan:

| rung | policy weights | why |
|---|---|---|
| 2, first light | about 1 million | enough for aiming and a duel by a wide margin |
| 3 and 4, fireteam and orders | 1 to 3 million | hide-and-seek's size; the skeleton of rung 0 comes to about this by itself |
| 5 and later, at scale | 5 to 20 million | where the professional-level agents sit |

Three facts behind it:

- **Our limit is samples, not weights.** The farm gives about 3 000 soldier decisions a second on the small ship and about 1 000 on the medium one. A network of 2 million weights decides and trains far faster than that on the desktop's GPU. A bigger network would wait for data.
- **Not too small either.** Our own Civulator ladder (2026-09-02) saw four changes in a row measure as "no effect" until the network grew to about a million weights; the small network had been hiding them. A design experiment on rung 3 should not be judged on a network smaller than that.
- **Size is a sweep, not a belief.** Half and double the size are among the first experiments once anything learns.

Rough run costs on the Home Desktop farm, small ship, at decimation 4:

| task | soldier decisions | time |
|---|---|---|
| shooting range | about a million | minutes |
| duel | about ten million | an hour |
| fireteam on the small ship | 100 million to a billion | half a day to four days |

One Berzelius node runs the farm about ten times faster (residency design 2.4).
These are estimates from other people's tasks; our first runs replace them.

## 6. Open questions, parked at the rung that needs them

- Rung 0: everything in its "next" list.
- Rung 2: does imitation of the bots first help or hurt?
- Rung 3: the three experiments named there; whether an own-reward-to-team-reward schedule suits a game with friendly fire and no respawn.
- Rung 4: the order vocabulary beyond move, hold, attack, fall back; who may refuse a suicidal order.
- Rung 6: whether the commander and the soldiers ever train together.
- Rung 7: one network or several when verbs differ.
- Rung 8: whether a cone changes behaviour at all.

## 7. Systems

**Existing systems this plan must use** (rows of the project's
canonical-systems table): Batched environment; MatchSession + the `policy`
controller (`INTENT_KINDS`, `VERB_RULES`); the sensor accessor; the sight model
(`vision.py`, `GameMap.has_los`); the turning law (`turning.set_target`,
`facing_after_turn_q16`); Bot layers and Position features; Credited cause;
Match recording / replay; the Bot tournament; Training snapshot; the level
bank (LevelSpec); Loadouts, Grenade kinds, Planted charge; Match config;
the Level data layer and the ship generator; the horde's flood; Wind load on
a body; the Speed table and Throttle. The design document's section 10 adds
three new systems to the table below: Room graph, Sight for many tiles, Verb
gates.

**New systems it will create**, each with its draft rule. The rules are
written into the project instructions only when the code exists.

| system | rung | draft rule |
|---|---|---|
| Team picture | 1d | What a team knows of the enemy lives in one place; the bots, the agents and the screen read it; never a second picture. |
| Observation spec | 1g | An observation layout is a named, versioned spec served by the sensor accessor. A new idea is a new spec, never an edit of one a trained network uses. Every feature is a ratio of a table value. |
| Action spec and decoder | 1h | The one layer from a network's choices to control verbs. Masks come from the game's own gates, never a copy of them. |
| Order | 1f | An order is game state set through one verb by any seat (a scripted team layer, a network, a human) and recorded like every verb. |
| Reward profile | 1i | A reward is a named table of weights over counters, in the run file; a new term is a new counter. |
| Trainer | 2 | Outside the game's source tree; touches the game only through `BatchedEnv`. |
| Rating | 2 | Strength is a rating against a fixed reference pool through the bot tournament, on held-out seeds and ships. |
