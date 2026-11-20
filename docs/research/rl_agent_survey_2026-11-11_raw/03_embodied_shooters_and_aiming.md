# 03 — Embodied team agents that move and aim, and how AIMING became an action

*Raw digest, 2026-11-11, for the RL soldier-agent survey (one of seven). Goes one level below
`rl_litsearch_2026-07-20_raw/03_predator_prey_animals.md`, which has hide-and-seek at survey level.*

Tags: **[READ]** = read in a primary source this session (quoted or paraphrased close to the
text; the ledger at the end says how much of each source); **[RECALL]** = from memory, not
checked this session; **[OPINION]** = mine. A few RL terms, once: a *policy* is the network that
picks actions; a *value function* or *critic* predicts future reward and is used only in
training; *behaviour cloning (BC)* is supervised imitation of recorded play; *PPO* is the standard
on-policy RL algorithm; an *action head* is one output distribution of the policy; an *action
mask* zeroes the probability of actions that are illegal right now.

---

## 1. Systems, one by one

### 1.1 OpenAI hide-and-seek (Baker et al. 2019, arXiv:1909.07528)

**(a) Observation.** Entity-centric: a list of entities (other agents, boxes, ramps), each with
"position, velocity, and size", plus the agent's own team, the other agents' teams and the time
left in the preparation phase [READ]. Beside it, "30 range sensors arrayed evenly around them,
similar to a lidar", encoded by "a circular 1D-convolution" [READ]. Architecture: per-entity
embedding (128 wide), masked self-attention over the entities (4 heads x 32), pooling, a dense
layer, an LSTM (256), action heads; 1.6 M parameters [READ]. The attention is "permutation
invariant", so the net takes any number of boxes and agents [READ].

**(b) Actions.** Three action types chosen at once each step: movement = "a discretized force
along their x and y axis and torque around their z-axis", grab (binary, binds to the nearest
object while held), lock (binary) [READ]. The bin count is not in the paper text; the released
code discretises with `DiscretizeActionWrapper(env, 'action_movement')`, default `nbuckets=11`,
evenly spaced `np.linspace(low, high, nbuckets)` [READ, code]. So turning was a RELATIVE command
(one of 11 torque levels) on a physical body with inertia — the closest analogue in this digest
to Breach's turning law. Episodes are 240 steps [READ]; the code's default `n_substeps=15` physics
substeps per step [READ, code].

**(c) Teammates and hiding.** Teammates are just entities in the list, tagged by team. Any entity
"not in line-of-sight of the agent or not in a 135 degree cone in front of the agent" is masked out
of the policy's attention [READ] — teammates included, so an agent does not always know where its
own team is. The lidar is 360 degrees and is not cone-masked [READ: "arrayed evenly around them"].
No communication channel. Training used "a centralized omniscient value function" that sees the
full unmasked state, while "each agent acts given only its own observations and memory" [READ].

**(d) What it cost / bought [READ].** Batch 64 000, 60 SGD substeps per optimisation step.
Stage 4 (ramp defence) after 132.3 M episodes (31.7 billion frames), 34 hours; ~380 M episodes for
the last stage. A MASKED value function "never progressed past stage 3" at the same compute — the
strongest single ablation here for Breach's critic design. Self-attention beat a pooling-only net
of similar size in sample efficiency (slightly slower wall clock). Batch 16 000 and 8 000 "never
converged". Separate per-agent parameters reached all six stages but less sample-efficiently;
shared parameters were the default. Team reward: hiders +1 if all hidden, -1 if any seen.
No ablation of the lidar or the cone angle.

### 1.2 DeepMind capture-the-flag, "For The Win" (Jaderberg et al., Science 2019, arXiv:1807.01281)

**(a) Observation.** First-person RGB pixels at 84x84 [READ]. No entity list, no map, no
teammate vector.

**(b) Actions.** Six action dimensions composed into "540 individual actions" [READ]: yaw change,
pitch change, strafe (ternary), forward/back, tag (fire), jump. Look was RELATIVE and very coarse:
"we expose only two possible values for yaw rotations (10 and 60) and just one for pitch (5)"
[READ] (degrees, each with a sign and a no-op, as I read the count). "15 times (steps) per second",
each action "repeated four times" [READ]. Two-timescale recurrent core: a fast LSTM every step and
a slow LSTM every tau steps, whose prior regularises the fast one; tau tuned by population-based
training (PBT: a population of agents whose hyperparameters are copied and mutated from the
winners) [READ].

**(c) Teammates.** Nothing explicit and no communication: agents "act and learn independently,
resulting in decentralised control within a team" [READ]. All coordination came from pixels.

**(d) Cost and results [READ].** Population P = 30, 1920 parallel arenas, two billion steps per
agent (~450 K games). Internal reward = a learned look-up over the 13 game-point events, tuned by
PBT for win probability. Tagging accuracy 80 % (FTW) vs 48 % (humans); reaction to a newly
appeared opponent 258 ms vs 559 ms. Delaying responses up to 375 ms did not change win
probability; at 448 ms the win rate fell to 85 %. No aiming-interface ablation. (I recall a
267 ms-delayed agent experiment in the Science version [RECALL]; the arXiv text read did not show
it.) Lesson: coarse relative look bins plus a 15 Hz decision rate were ENOUGH for superhuman tagging
— but at two billion steps per agent.

### 1.3 Counter-Strike deathmatch by behaviour cloning (Pearce & Zhu 2021, arXiv:2104.04258)

**(a) Observation.** Pixels: 1024x768 game, centre crop 824x498, down-sampled to 280x150; radar,
score, clock, health and ammo excluded — adding some as vectors was "not critical" [READ].

**(b) Actions.** Relative mouse deltas as classification: "19 options for mouse x"
{-300, -200, …, -10, -4, -2, 0, 2, 4, 10, …, 200, 300} and "13 options for mouse y"
{-50, …, -10, -4, -2, 0, 2, 4, 10, …, 50}, an "unevenly discretised grid, finer in the middle, and
coarser at the edges" [READ]. Keys and clicks are independent binary heads. 16 actions per second
[READ].

**(d) Why discrete [READ].** With a mean-squared-error continuous output, "given a choice of two
pathways, the agent would output a point midway between the two"; "Discretising the mouse space
and framing it as a classification task was more successful." Fine central bins because fine
adjustment matters for aim. Known weakness: "poor at aiming vertically". Result: matches the
medium built-in bot (3.72 vs 2.41 kills per minute against medium), gap to strong humans remains;
8 GPUs to train.

### 1.4 Diffusion behaviour cloning on the same CS:GO task (Pearce et al. 2023, arXiv:2301.10677)

Mouse x, y real-valued plus a binary click; the task requires dealing "with multimodality in target
selection and aiming" [READ]. Discretised baseline (20 bins per dimension) vs diffusion policies:
game score 7.8 (discrete) vs 12.9 (diffusion BC) vs 17.3 (Diffusion-X) with a small CNN; 14.7 vs
24.0 with ResNet18; humans 36.5 [READ; column meaning inferred from the human row]. Reading: the
multimodality problem (two enemies = two good aims) is real; an expressive distribution over
the joint action beats both independent bins and a unimodal continuous output.

### 1.5 OpenAI VPT, Minecraft (Baker et al. 2022, arXiv:2206.11795)

Camera = relative mouse, "foveated binning" with 11 bins per axis, finer for small moves; bin
centres are executed [READ]. Independent per-key heads failed to match human behaviour, so they
went joint: the full joint (20 buttons x 11 x 11) would be ~1.2e8 combinations, so a binary
"camera moved or not" action plus a 121-class camera head gives 8 461 key combinations + 121 [READ].
20 Hz [READ]. Lesson: factorise, but keep the dependencies that matter in the joint.

### 1.6 Riot / modl.ai tactical-shooter bots (Justesen et al. 2025, arXiv:2501.00078)

**(a) Observation — rays, not pixels.** A 15x15 grid of 225 rays from the forehead, denser near
the crosshair; each ray returns the hit distance in the layer of the object type it hit (100 m
default) [READ]. One extra ray per important object (players, grenades, fire, bomb, site) within a
90-degree field of view [READ]. Eight directional audio sensors per sound type (5 types) [READ].
Vectors: distance and direction to teammates, bomb site and bomb, health, ammo, abilities, round
time; for enemies only a minimum distance [READ]. Streams: a 3x3 conv on the ray grid, dense layers
elsewhere, an LSTM, multiple heads. The motive is compute: rendering eats the game's budget, so no
pixels; best model 14.9 M parameters, 9.6 ms per decision on a consumer CPU [READ].

**(b) Actions.** Aim is discrete and JOINT: yaw over 15 angles matching the ray columns, pitch
over 11, combined into "a 2D action space with 11×15=165 options"; the agent rotates the character
body directly, not a mouse [READ]. Whether the angles are deltas or absolute is not stated; they
are tied to the ray grid, so in effect "look along ray column j" — a direction inside the current
view. Keys (WASD, jump, plant/defuse, reload, two abilities, fire) discrete. 16 decisions per
second [READ].

**(d) Reported [READ].** Sampling yaw and pitch independently caused "undesired behaviors when
multiple attractive targets are in vision", hence the joint head — the same multimodality as 1.4.
Models move faster than humans; some get stuck. Imitation only (supervised on human
trajectories), 2v2. Human judges: 54 % of bot clips correctly called bots, 30 % judged human. No
ray-count or pixel ablation.

### 1.7 Learning to move like pro CS players (Durst et al. 2024, arXiv:2408.13934)

**(a) Observation.** One token per player, 10 tokens: position, alive, team, bomb, time, plus
derived features approximating visibility; dead players masked out of attention; teammate vs enemy
learned through self-attention from a team flag [READ]. Four single-head attention layers at 256,
5.4 M parameters, ~8 ms per query on one CPU core [READ].

**(b) Actions.** Movement only: 97 classes = 16 ABSOLUTE directions x 3 speeds x jump/no-jump +
stand still, every 125 ms [READ]. **Aiming and firing are not learned: a rule-based module
emulates professional aiming** [READ].

**(d) Reported [READ].** Trained in ~1.5 h on one RTX 4090 from 123 h of pro play. Human-likeness
TrueSkill above the game bot and a rule mover; teamwork error 27 %/13 % (offense/defense) vs 55/42
(rules) and 58/87 (game bot). Removing attention worsened map occupancy and kill locations.

### 1.8 Tencent Arena Breakout (Zhang et al. 2024, arXiv:2410.04936)

Nine heads: Fire, Gun_yaw (13 values), Gun_pitch (8), Move_type, Path_type (atomic move vs a
navmesh macro that runs until arrival, contact or timeout), Motion (17), posture, lean, special
[READ]. Observation: scalar groups (self, opponent, environment), a depth map, ray casts "towards
entities" [READ]. Firing is partly RULE-BASED: the firing point is set from "the visibility
priority of the opponent's body parts", shots drawn from a truncated confidence region, because
"pure RL alone cannot effectively control firearms and may exhibit perplexing behaviors" [READ].
Hybrid win rate "consistently around 30% higher" than pure RL at equal steps. 8 GPUs, 3200 CPU
cores, PPO; deployed in the live game since early 2024 [READ].

### 1.9 EA SEED, Battlefield 6 locomotion (Sestini et al. 2026, arXiv:2606.20210)

RL replaces ONLY the locomotion leaf of a behaviour tree; aiming and shooting stay scripted, and
the RL node must "interact with other systems, such as the aiming and shooting behavior" [READ].
Observation: a 50x50 egocentric occupancy map (agent, terrain, obstacle, waypoint) from the navmesh
vs a baseline of 24 rays in a 360-degree fan: comparable training performance, the map ~2x cheaper
(14 vs 27 microseconds) [READ]. PPO, up to 240 agents on one headless server, ~22 h; 11 of 20 wins
1v1 against a hand-coded opponent [READ]. Limits named: the map misses layers, dynamic obstacles and
destruction [READ].

### 1.10 EA SEED, LLM commander over RL skills (Hösch et al. 2026, arXiv:2606.20014)

2v2 King-of-the-Hill shooter in Unity, top-down figure. Each RL skill sees a 20-number vector
relative to its own position and facing (distances and angles to goal, enemies, ally, pickups;
health; time; seven forward rays); acts with continuous forward, TURN, strafe in [-1, 1] plus jump
and fire; aiming is not otherwise described, and "an angle-to-enemy reward" was needed to make it
face enemies [READ]. An LLM picks one of four unparameterised skills (Navigate, Combat, Secure,
Retreat) per agent at ~2 Hz; skills run at 12.5 Hz; "which enemy to engage" as a skill parameter is
future work [READ]. Results: LLM+RL vs behaviour tree 46.4 / 51.5 % (n.s.); flat RL lost to both
(BT 72.2 %, LLM+RL 67.0 % against it); against human teams 25.6 % (LLM+RL), 13.3 % (BT), 1.1 % (flat
RL) [READ].

### 1.11 ViZDoom deathmatch with Sample Factory (Petrenko et al. 2020, arXiv:2006.11751)

Seven independent heads: move 3, strafe 3, attack 2, sprint 2, interact 2, weapon 8, horizontal aim
21 — "between −12.5° and 12.5° in 1.25° steps" plus no-op; ~1.2e4 joint actions; frameskip 2;
128x72 pixels [READ]. Discretising the "technically continuous" aim "empirically leads to faster
learning" — no data shown [READ]. Final agents beat the hardest built-in bots in 100 % of matches
(deathmatch 80.5 vs 12.6 frags) [READ].

### 1.12 Rocket League bots on RLGym (Lucy-SKG, Moschopoulos et al. 2023, arXiv:2305.15801)

Perceiver-style cross-attention: a query for the own car, keys/values for teammates, opponents and
ball, a key-padding mask so player count can vary [READ]. Actions: Nexto's 90-entry discrete look-up
table over the 8 controls; Necto/Lucy a 5-output keyboard-style parser [READ]. A community write-up
explains the 90 entries as pruning (yaw = steer, no reversing while boosting, …) from 1 944
combinations [READ, secondary]. 15 decisions per second (frame skip 8 at 120 Hz) [READ]. Lucy beat
Nexto 300-4 and Necto 300-54 at 1 B steps; dropping stacked previous actions and simplifying the
observation lost 4-300 [READ]. Lesson: a small hand-pruned joint table is a good continuous-control
discretisation.

### 1.13 Target-pointer systems (MOBA / RTS / MMO) — the "attack enemy k" lineage

- **OpenAI Five** (arXiv:1912.06680): an action's unit parameter is "One of the 189 visible units
  in the observation"; a location is an offset X, Y each in -4..+4 (81 cells); a delay 0-3 frames;
  available actions filtered each step, unused parameters masked out of the loss [READ]. Fog of war
  is SHARED per team, so the five heroes' observations are "nearly identical" [READ]. Acts every
  4th frame at 30 fps [READ]. (The target is chosen by attention between the LSTM output and unit
  embeddings [RECALL].)
- **Tencent Honor of Kings** (Ye et al. 2020, arXiv:1912.09729): an action = button, Move_X,
  Move_Y, Offset_X, Offset_Y, and a target unit; **target attention**: query = FC(h_LSTM), keys =
  the stack of unit encodings, p(target) = softmax(query · keys) over the units present [READ].
  Ablation vs their base: + action mask 50.5 % win rate (same strength, 65 h vs 80 h to converge);
  + target attention 75 %; + LSTM 73 %; full 90 % [READ]. Heads treated independently ("control
  dependency decoupling") [READ]. Observation mostly vectors (1600 features) plus 2 image channels
  [READ].
- **Neural MMO 2** (docs): Attack = Style (melee, range, mage) + Target `Discrete(101)` = up to 100
  observed entities + a no-op; observation carries per-head action masks ("ActionTargets"); vision a
  15x15 square (7 tiles Chebyshev), 360 degrees, no line of sight; attack range 3 tiles [READ].
- **AlphaStar** pointer network over observed units; **SMAC** attack[enemy id] with masks [RECALL;
  likely in another digest].

### 1.14 Action-space shaping, the general evidence (Kanervisto et al. 2020, arXiv:2004.00980)

In ViZDoom "continuous actions prevents learning in all but the simplest environment";
discretising "improves performance notably"; multi-discrete heads scale, a flattened single
discrete does not; removing actions can hurt (Doom) or be what makes a task learnable (SC2) [READ].
Advice: "Start by removing all but the necessary actions and discretizing all continuous actions",
"limit the number of choices per discrete action" [READ].

### 1.15 Not read in enough depth to count

EA SEED concurrent-action imitation (Harmer et al. 2018, arXiv:1803.05402): multi-action heads,
"4x improvement in training time and 2.5x improvement in performance over single action selection"
[READ, abstract only]. A 2024 MDPI team-FPS study with an aim curriculum ("aim at enemies placed in
front of them") [search summary only; page 403]. Ubisoft La Forge Hyper Scape bots (pixels to a
small embedding to RL) [secondary only]. Ackermann et al. 2025 2D shooter (arXiv:2509.15042):
BC-then-RL hybrid >70 % vs rule bots, pure DQN unstable; action set not stated in the abstract.
Gigastep (NeurIPS 2023): only abstract-level; no aiming details found.

---

## 2. Aiming interfaces compared

| System | Aim interface | Bins / form | Field of view and hiding | Teammates in obs | Aim learned? cost / note |
|---|---|---|---|---|---|
| Hide-and-seek 2019 | (i) relative torque on an inertial body | 11 even levels (code) | 135° cone + LOS mask on entities; lidar 360° unmasked | entities, cone-masked | yes; 31.7 B frames to stage 4 (task is not aiming) |
| FTW CTF 2019 | (i) relative yaw/pitch deltas | yaw ±10/±60°, pitch ±5° | first-person pixels | none explicit | yes; 2 B steps/agent; 80 % tag accuracy |
| ViZDoom SF 2020 | (i) relative yaw | 21 bins, ±12.5° | pixels | — | yes; beats hardest bots 100 % |
| CS BC 2021 | (i) relative mouse delta | 19 x 13 non-uniform | pixels, HUD removed | none | BC; "poor at aiming vertically" |
| CS diffusion BC 2023 | (ii-ish) continuous delta, diffusion | continuous | pixels | none | diffusion 17.3 vs 20-bin 7.8 score |
| VPT 2022 | (i) relative, foveated | 11 x 11 joint (121) + moved flag | pixels | n/a | BC + RL fine-tune |
| Riot/modl 2025 | (ii)/(iii)-like: look along ray column | 15 x 11 joint = 165, tied to ray grid | ray grid + 90° object rays; audio 8 dirs | direction + distance vectors, always | BC; joint needed for multi-target |
| Arena Breakout 2024 | (i)/(ii) yaw 13, pitch 8 heads + RULE firing point | discrete + rules | rays to entities, depth map | scalars | partly; "pure RL alone cannot effectively control firearms" |
| CS pro movement 2024 | aim NOT learned (rules) | — | derived visibility features | all 10 players as tokens | movement only |
| BF6 EA 2026 | aim NOT learned (behaviour tree) | — | 50x50 occupancy or 24 rays 360° | — | locomotion only |
| EA KotH 2026 | (i) continuous turn rate | [-1, 1] | 7 forward rays + angles to entities | ally angle/dist | needed angle-to-enemy reward |
| OpenAI Five 2019 | (iii) pick unit k + (iv) offset grid | 189 units; 9x9 offsets | team-shared fog | shared vision | yes; pointer is the attack |
| Honor of Kings 2020 | (iii) target attention + (iv) offsets | softmax over present units | vectors + 2 image planes | vectors | target attention +25 pts win rate vs base |
| Neural MMO 2 | (iii) target index | Discrete(101) + mask | 15x15 square, 360°, no LOS | entities | — |

No source found compares (i) relative turning against (iii) a target pointer on the same task.
The closest evidence is indirect: (1) every MOBA / RTS / MMO system with many units names the
target (iii); (2) the shooter systems that learn relative aim are the ones where the INPUT DEVICE is
a mouse and humanlike or pixel-level aim is the point (FTW, Doom, CS, VPT); (3) three of the four
industry shooter bots read here (Durst, Arena Breakout, Battlefield 6) take aiming OUT of the
learned policy and hand it to rules, and the fourth (Riot) ties its aim bins to its own sensor
directions; (4) where a continuous turn is learned without help it needed a shaping reward
(EA KotH). [The pattern is READ; its reading is OPINION.]

---

## 3. For Breach

**Q2, aiming as an action. [OPINION, supported by 1.4, 1.6, 1.8-1.10, 1.13.]**
- Take (iii) as the main aim: a pointer head over the observed enemy entities (and over
  remembered last-seen enemies), scored as in Honor of Kings: query from the soldier's core state,
  keys = the entity embeddings already computed for the observation, softmax over the present
  entities, masked where absent. The body then turns toward the chosen set-point through Breach's
  own turning law; this is exactly what the industry bots do with a rule module, except the
  inertia and the spread are the physics, not a script.
- Every source that had a multimodal aim problem (two good targets) fixed it by making the choice
  one categorical: Pearce's MSE averaging between two paths, Riot's independent yaw/pitch, the
  diffusion paper's 20-bin baseline. A pointer is categorical by construction; a continuous angle
  (Gaussian) would average between two enemies. Avoid a continuous angle output.
- For "no enemy in sight": add direction ENTRIES to the same pointer — e.g. 16 absolute directions
  (Durst used 16 absolute directions for movement) and possibly "along my travel", "hold current
  facing", "toward the last sound" — so one softmax chooses between "enemy k" and "direction d".
  Absolute (world-frame or map-frame) bins are a set-point; relative turn bins (i) would make the
  policy re-learn the turning law that Breach already integrates.
- What a pointer loses: deliberate leading of a moving target, aiming at a body part, suppressive
  fire at a corner. A small offset head (an aim lead / offset bin relative to the chosen entity, as
  OpenAI Five's 9x9 location offset) covers the first; a "last-seen position" entry covers the
  corner. Open: whether a lead offset is needed at Breach's bullet speeds.
- What Breach must not do: let the pointer be a perfect aimbot. The turning law (≈90° in half a
  second) and the weapon spread already make a target choice cost time; keep them, and the
  pointer's value is in WHICH target, not in the reflex.

**Q3, factoring. [OPINION, 1.5, 1.6, 1.11, 1.12, 1.14, HoK.]** Multi-discrete heads with masks are
the consensus (Kanervisto; Sample Factory's 7 heads; HoK's decoupled heads); keep few choices per
head. Keep JOINT only where two parts must agree (VPT's camera flag, Riot's yaw x pitch): for Breach,
aim target and trigger are a candidate pair (fire only makes sense once the turn has arrived; an
"auto-fire when on target" flag would be a game-side helper). One grenade action per kind is fine as
a categorical "throw: none / frag / smoke / flash" head with masks for kinds not carried; the throw
TARGET can reuse the pointer (an entity or a direction) plus a range bin. Action masks (HoK: same
strength in 65 h instead of 80 h) are cheap and should be used for reload, throw, melee, use.

**Q1, representation. [READ + OPINION.]** Entity lists with masked attention are the default in
every team system read (hide-and-seek, Durst, Lucy, OpenAI Five, HoK, Neural MMO); hide-and-seek's
ablation shows attention beating pooling. Breach already plans a local map crop: HoK kept vectors as
the main input and only 2 image planes; Battlefield 6 found a 50x50 occupancy map as good as 24 rays
and 2x cheaper. Ray sensors were chosen elsewhere to avoid PIXELS (Riot) or to see walls next to an
entity list (hide-and-seek's lidar); with a field-plane crop, rays add little except long sight
lines beyond the crop. Open: a few long LOS rays for corridors longer than the crop.

**Q4, team display. [READ.]** Systems split: FTW gave nothing and learned coordination from pixels
at enormous cost; hide-and-seek masked teammates by the cone like enemies; Riot gave teammate
direction and distance always; OpenAI Five shared the team's fog entirely; Durst gave all players as
tokens. [OPINION] Giving teammates as always-visible entity tokens (health, ammo state, their
chosen target id, their play) and the team picture's remembered enemies as tokens with an age
feature is well within precedent and is what the bots already compute; it also makes the pointer
able to choose "my teammate's target" (focus fire) for free.

**Q5, orders. [READ: 1.10.]** The only commander-over-RL shooter read used UNPARAMETERISED skills and
named target parameters as future work; flat RL without a commander lost badly to both. [OPINION]
Give the soldier an objective / order slot now (a goal position or entity pointer and an order type,
zero = none), trained with randomly drawn or bot-issued orders, so the commander layer later plugs
into an input the soldier already understands.

**Q6, fog and field of view. [READ.]** Cone + line of sight on entities with an all-seeing critic
(hide-and-seek, 135°) vs 360° without LOS (Neural MMO) vs a 90° object FOV with 360° audio (Riot).
The one hard result is about the critic: a masked value function stalled at stage 3. [OPINION]
Whatever the cone, train with a centralised critic that sees everything; audio-like 360° cues
(Breach already has noise events) are the natural complement to a cone.

**Q7, heterogeneous units. [READ + OPINION.]** Attention over typed entity tokens is the only
architecture here that already took variable kinds and counts (boxes of random shapes,
variable player counts with a padding mask). A unit class is a type embedding plus its own action
masks; a pointer head works for any class that has a "target" verb.

**Q8, state of the art. [READ.]** The 2024-2026 industry shooter bots are imitation-first or
hybrid (Riot BC; Durst BC; Arena Breakout RL + rules; Battlefield 6 RL locomotion in a behaviour
tree; EA LLM + RL skills). Pure end-to-end RL shooters with relative aim (FTW, Doom) remain the
compute-heavy research landmarks of 2018-2020. Diffusion policies beat discretised BC on CS aim
(2023). I found no 2025-2026 paper that ablates aiming interfaces.

**Contradictions to the leanings.** None strong. Two cautions: (1) "aim at enemy k" leaves no learned
path to aim at things that are not entities (a door to shoot open, a corner, a fuel tank, a wall to
breach) — the direction entries or a map-point head (iv) are needed for that; (2) "one separate
action per grenade kind" as independent binary heads would let the policy throw two at once —
make it one categorical head.

**Could not verify.** The FTW 267 ms experiment; OpenAI Five's target mechanism (attention) beyond
the "one of 189 visible units" sentence; Nexto's table contents; the MDPI aim-curriculum paper;
Gigastep's actions. No direct relative-vs-pointer ablation was found.

---

## 4. Source ledger

| # | Citation | URL | Status | Supports |
|---|---|---|---|---|
| 1 | Baker et al. 2019, Emergent Tool Use From Multi-Agent Autocurricula, arXiv:1909.07528 | https://arxiv.org/html/1909.07528 | PART (first 100 K chars incl. appendix) | 1.1: obs, cone 135°, lidar, actions, compute, masked-critic and attention ablations |
| 2 | openai/multi-agent-emergence-environments, `hide_and_seek.py`, `wrappers/util.py` | https://github.com/openai/multi-agent-emergence-environments | PART (two files) | 11 action buckets, n_substeps 15 |
| 3 | Jaderberg et al. 2019, Human-level performance in first-person multiplayer games…, arXiv:1807.01281 | https://ar5iv.labs.arxiv.org/html/1807.01281 | PART | 1.2: 84x84, 540 actions, yaw 10/60, 15 Hz, P=30, 80 % vs 48 %, 258 vs 559 ms |
| 4 | Pearce & Zhu 2021, Counter-Strike Deathmatch with Large-Scale Behavioural Cloning, arXiv:2104.04258 | https://arxiv.org/html/2104.04258 | PART | 1.3: mouse bins, MSE averaging, vertical aim |
| 5 | Pearce et al. 2023, Imitating Human Behaviour with Diffusion Models, arXiv:2301.10677 | https://arxiv.org/html/2301.10677 | PART | 1.4: continuous vs 20-bin aim scores |
| 6 | Baker et al. 2022, Video PreTraining (VPT), arXiv:2206.11795 | https://arxiv.org/html/2206.11795 | PART | 1.5: foveated 11 bins, joint camera head |
| 7 | Justesen et al. 2025, Human-like Bots for Tactical Shooters Using Compute-Efficient Sensors, arXiv:2501.00078 | https://arxiv.org/html/2501.00078 | PART | 1.6: rays, 165 joint aim, multi-target failure |
| 8 | Durst et al. 2024, Learning to Move Like Professional Counter-Strike Players, arXiv:2408.13934 | https://arxiv.org/html/2408.13934 | PART | 1.7: player tokens, 97 move classes, aim by rules |
| 9 | Zhang et al. 2024, Training Interactive Agent in Large FPS Game Map with Rule-enhanced RL, arXiv:2410.04936 | https://arxiv.org/html/2410.04936v1 | PART | 1.8: heads, rule firing, +30 % win rate |
| 10 | Sestini et al. 2026, Augmenting Game AI with Deep RL (EA), arXiv:2606.20210 | https://arxiv.org/html/2606.20210v1 | PART | 1.9: occupancy map vs 24 rays, aim stays scripted |
| 11 | Hösch et al. 2026, Hierarchical Control in Multi-Agent Games: LLM-based Planning and RL Execution, arXiv:2606.20014 | https://arxiv.org/html/2606.20014v3 | PART | 1.10: skills unparameterised, flat RL loses, angle reward |
| 12 | Petrenko et al. 2020, Sample Factory, arXiv:2006.11751 | https://arxiv.org/html/2006.11751 | PART | 1.11: 7 heads, 21 aim bins |
| 13 | Moschopoulos et al. 2023, Lucy-SKG, arXiv:2305.15801 | https://arxiv.org/html/2305.15801 | PART | 1.12: Perceiver obs, 90-entry table, ablations |
| 14 | Ripple / RLGym write-ups on the lookup table | https://wandb.ai/rolv-arild/ripple/reports/Ripple--VmlldzozNDE5NzE4 | ABS (search summary) | 1 944 -> 90 pruning |
| 15 | OpenAI et al. 2019, Dota 2 with Large Scale Deep RL, arXiv:1912.06680 | https://arxiv.org/html/1912.06680 | PART | 1.13: 189-unit target, offsets, shared fog |
| 16 | Ye et al. 2020, Mastering Complex Control in MOBA Games with Deep RL, arXiv:1912.09729 | https://arxiv.org/html/1912.09729v3 | PART | 1.13: target attention formula and ablation |
| 17 | Neural MMO game wiki | https://neuralmmo.github.io/_build/html/rst/game_wiki.html | PART | 1.13: Discrete(101) target, masks, 15x15 vision |
| 18 | Suarez et al. 2023, Neural MMO 2.0, arXiv:2311.03736 | https://arxiv.org/html/2311.03736 | ABS (page had no action details) | — |
| 19 | Kanervisto, Scheller & Hautamäki 2020, Action Space Shaping in Deep RL, arXiv:2004.00980 | https://arxiv.org/html/2004.00980 | PART | 1.14 |
| 20 | Harmer et al. 2018, Imitation Learning with Concurrent Actions in 3D Games, arXiv:1803.05402 | https://arxiv.org/abs/1803.05402 | ABS | 1.15: multi-action heads 4x / 2.5x |
| 21 | Ackermann, Spang & Gardi 2025, RL Agent for a 2D Shooter Game, arXiv:2509.15042 | https://arxiv.org/abs/2509.15042 | ABS | 1.15 |
| 22 | MDPI Technologies 12(3):34 (2024), RL to train multiplayer FPS agents | https://www.mdpi.com/2227-7080/12/3/34 | ABS (search summary; page 403) | aim curriculum |
| 23 | Ubisoft La Forge, Romoff CEDEC talk / Hyper Scape bots | https://www.ubisoft.com/en-us/studio/laforge/publications | ABS (secondary) | 1.15 |
| 24 | Lechner et al. 2023, Gigastep, NeurIPS D&B | https://proceedings.neurips.cc/paper_files/paper/2023/hash/00ba06ba5c324efdfb068865ca44cf0b-Abstract.html | ABS | none on aiming |
| 25 | Vinyals et al. 2019, AlphaStar (Nature); Samvelyan et al. 2019, SMAC | — | RECALL | pointer / attack[k] lineage |
| 26 | Direct comparison of relative turning vs target pointer for aiming | — | NF | — |
