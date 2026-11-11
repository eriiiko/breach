# MOBA agents: OpenAI Five and Tencent's Honor of Kings line

Digest 01 of the RL soldier-agent survey, 2026-11-11. Topic: five-hero team games
(Dota 2, Honor of Kings), read for how they represent units, choose targets, factor
actions, share information across a team, take orders, and handle many hero types.

Conventions. **READ** = I read it in the named source during this session (quotes are
verbatim). **RECALL** = from memory, not checked this session. Source tags [S1]... refer
to the ledger at the end. The July 2026 litsearch already covers OpenAI Five at the
level "shared LSTM per hero, PPO, team spirit"; this digest goes one level down.

Terms used once and then freely: a **policy** is the network that maps an observation
to a probability over actions; a **value function** (the critic) estimates the future
reward from a state and is only used during training; **PPO** (proximal policy
optimization) is the standard on-policy actor-critic algorithm; a **head** is one output
layer of the network, giving the distribution of one part of the action; an **action
mask** sets the probability of illegal choices to zero before sampling; **self-play** is
training against copies of oneself; **distillation** trains one student network to copy
the outputs of one or more teacher networks.

---

## 1. OpenAI Five (Berner et al. 2019, arXiv:1912.06680) [S1, S2]

Dota 2, 5 v 5, beat the world champions (OG) in 2019. Read: the arXiv HTML appendices
E (observation), F (action), G (reward), H (architecture), and OpenAI's own
architecture diagram (06/06/2018 version, PDF) [S2], which I rendered and read label by
label.

### 1a. Observation
- Per step "~16,000 total values" (READ, S1 §2). The bulk is per-unit: "per-unit
  observations, observed for each of 189 units" (READ, S1 Table 3).
- **Unit list, not a map.** Units are grouped by category -- in the 2018 diagram:
  Enemy Heroes, Allied Heroes, Enemy Non-Heroes, Allied Non-Heroes, Neutrals (READ, S2).
  Each unit row: unit type (embedding), "attacking/attacked by hero for all allied and
  enemy heroes", stats, "health over last 12 frames", "distance from allied heroes,
  enemy heroes", orientation as cos / sin, absolute position, animation (READ, S2).
  Heroes additionally carry pooled sets of their modifiers, abilities and items, each
  set encoded by a shared FC network and **max-pooled** (READ, S2). Unordered sets go
  through a "Process Set" module (READ, S1).
- **Which units, when there are too many:** "we observe only the units closest to
  allied heroes"; fewer than the allotted slots are zero-padded (READ, S1). No sorting
  by id.
- **Spatial input is tiny.** The only map is a "nearby terrain 8x8 grid of height,
  traversability, creep occupancy for each hero in team" (READ, S2). Everything else is
  entity rows plus global scalars (time of day, glyph cooldown, courier contents...).
- **Fog of war:** "Units in fog of war are not observed." A hidden enemy hero keeps
  "the observation from the last time step when the unit was visible" (READ, S1) -- a
  built-in last-seen memory, close to Breach's bot team picture.
- **Team-level observation.** The game itself shares vision among allies (RECALL: Dota
  rule; S1 does not use the words "shared vision"). The five replicas get almost the
  same input: "Almost all of the inputs are the same for each of the five replica
  LSTMs (the only differences are the nearby map, previous action, and a very small
  fraction of the observations for each unit)" (READ, S1 App. H). On top of that a
  **cross-hero pool**: "we maxpool the first 25% of the vector across the five replica
  networks", "to ensure that the non-identical observations can be used by other
  members of the team if needed" (READ, S1). So each hero effectively sees the
  team's union of information.
- **No message channel.** No learned communication between heroes (READ secondary:
  press coverage of the 2018 blog [S3]; the paper describes none -- READ, S1).

### 1b. Network
- One network, five replicas with shared weights, each with its own LSTM hidden state
  (READ, S1 Fig. 1). The controlled hero is marked by appending its own unit embedding
  to the state vector (READ, S1). Final model "158,502,815 parameters", single-layer
  4096-unit LSTM, "The LSTM composes 84% of the model's total parameter count" (READ,
  S1); the 2018 diagram shows a 1024-unit LSTM (READ, S2) -- the network grew by
  "surgery" (weight-preserving architecture changes during a 10-month run, READ S1).

### 1c. Actions and target choice
- Factoring: one **primary action** plus "3 parameter outputs, Delay (4 dim), unit
  selection (189 dim), and offset (81 dim)" -- the offset a 9 x 9 grid of -4..+4 around
  the caster or the target (READ, S1 App. F). Teleport gets its own target head.
  8,000 to 80,000 actions available per step depending on hero (READ, S1 §2).
- **How the target is chosen (pointer attention).** READ from the diagram [S2]: each
  unit's embedding is sliced ("slice 0:64") to form **"Unit Attention Keys"**; the LSTM
  output goes through an FC layer to a query; **dot** with every key; **Softmax**;
  "Sample/Argmax" -> "Target Unit". The primary action is chosen the same way: the
  **"Available Actions"** are embedded, dotted with an FC projection of the LSTM, then
  softmax -> "Selected Action". So both the verb and the target are *pointers* into a
  list of currently valid items, scored by a dot product -- not fixed-size logit
  vectors over a fixed index. The paper text only says the unit embeddings are
  "carried along beside the LSTM, and used by the action heads to choose a unit to
  target" (READ, S1).
- **Conditioning between heads.** In the 2018 diagram all heads branch in parallel from
  the LSTM (READ, S2): the target is not computed from the chosen verb's embedding. In
  the final version "Unit Selection and Offset are actually implemented within the
  model as several different, mutually exclusive parameters depending on the primary
  action" (READ, S1) -- i.e. conditioning by *switching head*, and the heads the chosen
  verb ignores are masked out of the loss "since their gradients would be pure noise"
  (READ, S1). A secondary blog claims the chosen action's embedding multiplies the unit
  keys [S4, ABS]; I could not confirm that in the primary text.
- **Masks:** scripted filters restrict available actions every step (READ, S1).
- **Rate and delay:** acts every 4th frame, "7.5 frames per second"; reaction time
  "randomly distributed between 5 and 8 frames (167ms to 267ms)" (READ, S1). The
  Delay head (which frame within the 4 to act on) went unused: "the model did not learn
  to do this" (READ, S1).

### 1d. Reward and team spirit
- r_i = (1 - tau) rho_i + tau * mean_team(rho) (READ, S1 App. G). Schedule (Table 2):
  OpenAI Five 0.3 -> 0.8, Rerun 0.3 -> 1.0, "a smooth monotonic transition (usually a
  linear change over one to three days)" (READ, S1). Reason: "lower team spirit reduces
  gradient variance in early training" (READ, S1).
- Zero-sum: subtract the enemies' average reward; non-win rewards decay by 0.6 per 10
  minutes of game time; win = 5, own hero death = -1, plus shaped terms (last hits,
  health change, towers...) (READ, S1 Table 5). Heavily shaped.

### 1e. Heterogeneous heroes
- One net for all heroes; team sampled at random from the pool each game (READ, S1).
  Final pool 17 heroes (READ, as stated in S6's comparison). "Training with 80 heroes
  has a speedup factor of approximately 0.8" -- i.e. slower, not failed (READ, S1). The
  authors list "its inability to play all the heroes" as a primary limitation (READ).

### 1f. Cost
- 770 +/- 50 PFlops/s-days; up to 1536 optimizer GPUs; batches of 1-3 M timesteps;
  10 months. Rerun from scratch (no surgery): 150 PFlops/s-days, 2 months, 512 GPUs and
  51,200 CPUs (READ, S1). Ablations useful to us: sample reuse above ~1 hurts badly;
  entropy 0.01 best (READ, S1). No ablation of the target attention itself.

---

## 2. Tencent, "Mastering Complex Control in MOBA Games" (Ye et al., AAAI 2020, arXiv:1912.09729) [S5]

Honor of Kings **1 v 1** (one hero per side). READ (HTML full text, via narrow quotes).
- **Observation:** "1600 vector features containing observable unit attributions and
  game information" plus "2 channels of image features ... (the obstacle channel and the
  hero position channel)"; "the same type units are mapped to a feature vector of fixed
  length by max-pooling" (READ).
- **Target attention:** "The target unit t of action a is predicted by a target
  attention mechanism over every unit"; keys are "the stack of all unit encodings"; the
  query is an FC of the LSTM state; p(t|a) = Softmax(FC(h_LSTM) h_keys^T) (READ). Same
  pointer mechanism as OpenAI Five.
- **Heads:** button, Move_X, Move_Y, Offset_X, Offset_Y, target (READ).
- **Control dependency decoupling:** "we treat each label in an action independently to
  decouple their inter-correlations" -- each head its own PPO term; justified by "it
  simplifies the policy structure" and "it increases the diversity of actions" (READ).
  Not autoregressive. No ablation of this choice is reported (READ).
- **Action mask:** forbidden map areas, skill/attack availability, enemy control
  effects, item restrictions (READ).
- **Dual-clip PPO:** for negative advantages add a lower bound c * A (c = 3, eps = 0.2)
  so a huge probability ratio on an off-policy sample cannot blow up the loss (READ).
- **Rate:** "every 133 ms which is about the response time of top-amateur players"
  (READ).
- **Heroes:** one model per hero; "playing a different MOBA hero is like playing a
  different game"; eight heroes shown (READ).
- **Ablation (Table 5, win rate of variant vs Base, training hours to converge):** Base
  80 h; +action mask 50.5 %, 65 h ("using action mask can largely reduce the training
  time"); +target attention 75 %, 90 h; +LSTM 73 %, 100 h; full 90 %, 80 h (READ).
  So target attention bought the largest single strength gain; the mask bought speed.
- **Cost:** 600,000 CPU cores and 1,064 GPUs in the framework; per hero 48 P40 GPUs and
  18,000 CPU cores (READ). Results: 99.81 % of 2,100 games vs top players; 5 pros beaten
  in BO5 (READ).

## 3. Tencent, "Towards Playing Full MOBA Games" (Ye et al., NeurIPS 2020, arXiv:2011.12692) [S6]

Honor of Kings **5 v 5**, 40-hero pool. READ (HTML, narrow quotes).
- **Observation:** 9,227 scalar features + 6 spatial channels of 17 x 17 local view
  (READ). Hidden info: "we use full information about the game state, including
  observations hidden from the policy, as input to the value function", "only during
  training" -- the **asymmetric critic** (privileged value, honest policy) (READ). One
  sentence in §3.1 lists "invisible opponent information" among scalar features; §8.6
  says it "is only applied to the value network" (READ: the paper is internally
  inconsistent; §8.6 is the specific one).
- **Teammates:** the paper describes no communication or concatenation of teammates'
  states; "one model with shared weights to control all heroes" (READ). How much of
  each teammate the per-hero input carries: not stated in what I read.
- **Actions:** "what action to take", "who to target", "how to act" (READ). "We assume
  independence between action heads", except "the direction of a skill ('How') is
  conditioned on the skill type ('What')" (READ). Target attention and masks as in S5.
  Response time 193 ms including a 133 ms observation delay (READ).
- **Value:** five value heads (farming, KDA, damage, pushing, win/lose), summed with
  weights (READ). Multi-head value and the invisible-info critic "can both increase the
  Elo score effectively" (READ; no numbers in what I read).
- **Heterogeneity -- curriculum self-play (CSPL):** phase 1 train teachers on fixed
  lineups, phase 2 **multi-teacher distillation** into one student, phase 3 continue RL
  on random lineups (READ). 144 h + 48 h + 144 h = 336 h; the baseline trained directly
  on random lineups ran >480 h without converging and ended ~400 Elo below (READ).
  Teachers ~9 M parameters, final model ~17 M (READ) -- an order of magnitude smaller
  than OpenAI Five.
- **Cost:** a "resource unit" of 320 GPUs and 35,000 CPUs (READ).
- **Results:** 40-2 vs professional teams; 97.7 % of 642,047 games vs top players (READ).

## 4. Tencent, Hierarchical Macro Strategy (Wu et al., arXiv:1812.07887; AAAI 2019 per RECALL) [S7]

The closest precedent to a commander telling soldiers where to go. READ (HTML).
- A supervised **attention layer** predicts, per hero, which of "144 regions of the
  map" (12 x 12) the hero will next attack in, labelled from human replays; a **phase
  layer** predicts which of 14 major resources (turrets, dragon...) is the current
  focus (READ).
- **Imitated cross-agent communication:** "During training phase, we put attention
  labels of allies as features"; "During testing phase, we put attention prediction of
  allies as features" -- each hero sees where its teammates intend to go (READ).
- Macro output "directly conveys macro strategy embedding to micro level models"
  (READ; the exact wiring is not stated).
- Ablation (40 games each, vs full model): without macro strategy the full model wins
  75 %; without communication 62.5 %; without phase layer 65 %; vs top-1 % human teams
  48.3 % (READ). The teammates'-intent feature is worth a clear margin.

## 5. Macro-goals and meta-commands (Tencent, 2021-2023) [S8, S9]

- **MGG, "Learning Diverse Policies in MOBA Games via Macro-Goals"** (Gao et al.,
  NeurIPS 2021, arXiv:2110.14221; READ full PDF text). A goal = region intent ("Where to
  go") + resource intent ("What to do"), a 64-dim vector extracted from top players'
  replays 30 s ahead (READ). A supervised Meta-Controller predicts the goal; the RL
  policy is pi(a | s, g) with the goal as extra input, plus an **intrinsic reward** =
  the decrease in distance between the current macro-state and the goal, on its own
  value head (READ). One model for all 102 heroes; 60,000 CPUs + 830 V100s, 420 h
  (READ). Win rate vs humans 97.81 % vs 97.77 % for the goal-free baseline -- strength
  kept, diversity gained; +312 and +560 Elo over baseline in two special lineup systems
  (READ).
- **MCC, meta-commands for human-agent teams** (Gao et al., ICLR 2023, arXiv:2304.11632;
  READ HTML, first 100k chars). A meta-command = (Location L on a 144-cell grid, Event
  E to do there, time limit T) (READ). The agent is a command-conditioned network with
  intrinsic reward for closing distance to L, plus extrinsic team/individual reward
  (READ). A **Meta-Command Selector** (a value estimator trained by self-play) lets each
  agent pick the most valuable of the commands it received -- the agent is not obliged
  to obey a bad one (READ). With humans: e.g. high-level 1 human + 4 agents 54 % vs 42 %
  for the baseline; humans' response rate to MCC agents' commands 73-79 % (READ).

## 6. Environment and newest work (2022-2026)

- **Honor of Kings Arena** (Wei et al., arXiv:2209.08483, 1 v 1 only) [S10, READ]: obs
  491 dims; action = (button, 16 x 16 direction, target among {none, self, enemy hero, 4
  nearest soldiers, nearest tower}) -- here the target is an index into a short
  **fixed, ordered** slot list, not attention; mask essential (without it the agent
  "quickly converges to a local optimum"); 133 ms; 20 heroes = 400 tasks, a model for
  one matchup "drops dramatically" on another, multi-task training and distillation
  help; PPO beats the scripted bot in 6.16 h on 128 cores, 0.89 h on 2048 (READ).
- **Think in Games** (Liao et al., Tencent, arXiv:2508.21365, Aug 2025) [S11, READ]: an
  LLM (Qwen-3-14B, SFT + GRPO) picks 1-2 of ~40 **macro team objectives** from a JSON
  of "all visible information from the primary player's perspective"; 90.91 % match to
  human replay choices; micro control is not covered. This is the 2025 direction for
  the macro layer: language-model commanders over RL micro.
- **Featherweight distillation** (Yang et al., arXiv:2602.07521, Feb 2026) [S12, ABS]:
  the HoK agent distilled for phones, 12.4x faster, < 0.5 ms per frame, the student
  keeps 40.32 % win rate vs the teacher.
- I found no 2024-2026 MOBA work that replaces the pointer-attention target head or the
  entity-set encoder; the newer Tencent papers build on S5/S6 unchanged (READ in S6,
  S8: "similar to" / "exactly the same as the RL-baseline"). RECALL: the general field
  moved from max-pool set encoders to transformers over entities (AlphaStar 2019
  onward); the MOBA line kept max-pool + LSTM, cheaply.

---

## 7. For Breach

(Opinions marked **[op]**; everything else is a reading of the sources above.)

**Q1 Representation.** Both lines encode the world as **entity lists by category**
(allied heroes, enemy heroes, other units...) with shared per-unit encoders and
max-pooling, plus a small spatial grid (OpenAI 8 x 8 per hero; Tencent 6 x 17 x 17).
**[op]** For Breach, where fields (fire, smoke, pressure) matter far more than in a
MOBA, the map crop deserves a bigger share than in either, but enemies and teammates
should be entity rows, not only pixels: the target head needs one embedding per
targetable thing. Replace max-pool with a small transformer over entities (the 2019+
default) -- cheap at Breach's entity counts.

**Q2 Aiming.** Both systems choose a target by **pointer attention**: query from the
core state, dot product with each candidate's key, softmax over the *currently present*
candidates, with absent/illegal ones masked. This is exactly the author's "aim at enemy
k" and it was the largest single gain in Tencent's ablation (+target attention 75 %
vs base). Note what it is not: k is not a fixed slot number; it is an index into
whatever entities are in the list this step, so it scales to any number of enemies.
For "no enemy in sight" both games have a positional head: OpenAI an 81-cell offset
grid (and Move X / Move Y heads), Tencent discretized move/skill directions (16 per
axis in HoK Arena). **[op]** Breach: target head over entity keys (enemies, and later
doors, grenades in flight, last-seen ghost positions as pseudo-entities) + a direction
head (e.g. 16-32 bins) used when the verb is "aim at direction"; the turning law then
turns toward the chosen target. MOBA agents act at 7.5 Hz with a 167-267 ms reaction
delay (OpenAI) or every 133 ms with 193 ms total response (Tencent) -- a useful
precedent for Breach's decimation (at 24 tick/s, every 3-4 ticks).

**Q3 Factoring.** Both use one categorical verb head + parameter heads, *mostly
independent* ("control dependency decoupling"), with the minimal conditioning that
matters: parameters depend on the chosen verb by switching head / masking the loss of
ignored heads (OpenAI), and "How" conditioned on "What" (Tencent 5 v 5). Both also make
the verb list itself a masked list: OpenAI literally points into "Available Actions".
This supports the author's leaning: a grenade of each kind as its own verb is natural
(like each Dota ability is its own primary action), and the mask removes kinds not
carried. **Mild contradiction to note:** none of these agents issues several verbs at
once; Dota/HoK pick ONE primary action per step. Breach's marine moves, aims, and fires
simultaneously, so Breach needs several *concurrent* heads (move, aim, trigger, one
discrete "special" verb), not one verb head -- closer to a twin-stick game than a MOBA.

**Q4 Team display.** OpenAI Five gives every hero almost the team's whole observation
plus a cross-hero max-pool over a quarter of each replica's features -- in effect a team
display, at the cost of never knowing what a teammate privately sees beyond that. Wu
et al. 2018 show that adding teammates' *intentions* (where they are going) as input is
worth a clear margin (62.5 % win for the full model over the no-communication one).
**[op]** Breach's team display (teammates' health, positions, current target, last-seen
enemies) is well supported; including each teammate's current target/intent is the part
with direct evidence behind it.

**Q5 Orders.** The macro-goal line is the precedent: a goal (map region + what to do +
time limit) as an extra policy input, an intrinsic reward for closing distance to it on
its own value head, and goals sampled during training from a model of plausible goals
(MGG: strength unchanged, 97.81 vs 97.77 %). MCC adds that the agent may choose among
received commands by value. **[op]** Give the soldier an order input from day one,
filled during training by randomly sampled or bot-generated orders (the team play
Hold / Bound / Fall back, a target region), with a small intrinsic reward -- retrofitting
an input later costs a retrain. An empty order must be a legal value.

**Q6 Fog of war / FOV.** Both games give the policy only legally visible units (team
vision) and the critic everything (Tencent's asymmetric critic, worth Elo). OpenAI keeps
a hidden hero's last-seen observation. **[op]** Copy both: honest policy, privileged
critic; last-seen ghost entities with an age feature. MOBA vision is team-shared and
radius-based; a per-unit view cone has no MOBA precedent -- open.

**Q7 Heterogeneous units.** One shared network for all hero types worked in both lines
(17, 40, 102 heroes), with the unit's own identity as an input embedding. Training
directly on all types was slow (OpenAI 0.8x speed at 80 heroes; Tencent's baseline did
not converge in 480 h) and the fix was a curriculum: specialist teachers, then
distillation into one student, then mixed training. Pointer heads over "available
actions" let different hero kits share one head design. **[op]** Breach can use the
same: per-class verb embeddings in an available-action list, so a robot or an alien
brings its own verbs without a new head.

**Q8 State of the art.** The micro-control architecture (entity encoder + LSTM + pointer
target + independent masked heads + PPO variant) has not been superseded inside the
MOBA line through 2026; what changed is (a) transformers instead of max-pool generally
(RECALL), (b) LLMs as the macro/commander layer (Think in Games, 2025), (c) compute
efficiency -- Tencent's 17 M-parameter model vs OpenAI's 159 M, and phone-sized students
in 2026. Avoid OpenAI's 10-month surgery path; Rerun from scratch needed 1/5 the
compute.

**Avoid.** Heavy shaped rewards that dominate the win signal (OpenAI's Table 5) unless
a bot ladder shows they help; the Delay sub-action (unused); fixed per-slot target
indices (HoK Arena) when the number of enemies varies.

**Open.** Whether Breach's simultaneous verbs need autoregressive conditioning (e.g.
trigger conditioned on target) -- MOBA evidence says independence is fine for one
verb per step, but says nothing about concurrent verbs; team spirit with deaths
mid-match (MOBAs respawn, Breach does not); the view cone.

---

## Source ledger

| Tag | Citation | URL | Status | Supports |
|---|---|---|---|---|
| S1 | Berner et al. 2019, Dota 2 with Large Scale Deep RL, arXiv:1912.06680 | https://arxiv.org/html/1912.06680 (+ ar5iv mirror) | PART (App. E-H, Tables 2, 3, 5 read; main text partly) | obs, cross-hero pool, fog, heads, masks, rate, team spirit, cost |
| S2 | OpenAI, "OpenAI Five Model Architecture (06/06/2018)" diagram | https://cdn.openai.com/research-covers/openai-five/network-architecture.pdf | FULL (rendered, every label read) | unit categories, attention keys + dot product target head, available-action pointer, parallel heads, 1024 LSTM, 8x8 terrain grid |
| S3 | Press coverage of the 2018 OpenAI Five blog (The Register, Synced, Inquirer, Kollortz slides) | via search, e.g. https://www.theregister.co.uk/2018/06/25/openai_dota_2/ | ABS (search snippets) | no explicit communication channel |
| S4 | xlnwel blog on OpenAI Five | https://xlnwel.github.io/blog/reinforcement%20learning/OpenAIFive | ABS (search snippet) | claim that chosen action conditions target -- unconfirmed |
| S5 | Ye et al. AAAI 2020, Mastering Complex Control in MOBA Games, arXiv:1912.09729 | https://arxiv.org/html/1912.09729 | PART | target attention formula, decoupling, mask, dual-clip, 133 ms, ablation table, compute |
| S6 | Ye et al. NeurIPS 2020, Towards Playing Full MOBA Games, arXiv:2011.12692 | https://arxiv.org/html/2011.12692 | PART | 5v5 obs, asymmetric critic, heads, CSPL distillation, 40 heroes, compute, results |
| S7 | Wu et al., Hierarchical Macro Strategy Model for MOBA Game AI, arXiv:1812.07887 (AAAI 2019: RECALL) | https://arxiv.org/html/1812.07887v1 | PART | 144-region attention, allies' intent as input, ablation |
| S8 | Gao et al. NeurIPS 2021, Learning Diverse Policies in MOBA Games via Macro-Goals, arXiv:2110.14221 | https://arxiv.org/pdf/2110.14221 | FULL (text extracted) | goal-conditioned policy, intrinsic reward, 102 heroes one model, win rates |
| S9 | Gao et al. ICLR 2023, Towards Effective and Interpretable Human-Agent Collaboration in MOBA Games, arXiv:2304.11632 | https://arxiv.org/html/2304.11632v1 | PART (first 100k chars) | meta-command (L, E, T), selector, human-team results |
| S10 | Wei et al. 2022, Honor of Kings Arena, arXiv:2209.08483 | https://arxiv.org/html/2209.08483 | PART | 1v1 env, fixed target slots, mask necessity, cross-hero generalization, cost |
| S11 | Liao et al. 2025, Think in Games, arXiv:2508.21365 | https://arxiv.org/html/2508.21365v1 | PART | LLM macro objectives, JSON obs, GRPO, accuracy |
| S12 | Yang et al. 2026, Pareto-guided Pipeline for Distilling Featherweight AI Agents in Mobile MOBA Games, arXiv:2602.07521 | https://arxiv.org/abs/2602.07521 | ABS | 2026 distillation numbers |
| S13 | Gao et al. 2024, RLHG (human gain), arXiv:2401.16444, ICLR 2024 | https://arxiv.org/html/2401.16444v1 | ABS (search snippet) | human-centred collaboration exists; not used above |
| -- | Dota 2 allied shared vision as a game rule | -- | RECALL | Q6 |
| -- | Transformers over entities as the post-2019 default | -- | RECALL | Q1, Q8 |
