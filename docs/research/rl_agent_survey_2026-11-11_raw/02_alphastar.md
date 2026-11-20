# 02 — AlphaStar and its descendants (StarCraft II)

Research digest for the Breach RL soldier-agent design, 2026-11-11. Status tags:
**[READ]** = read in a primary source during this session (quoted or closely paraphrased);
**[RECALL]** = from memory, not checked this session. Source keys [S1]... refer to the ledger at the end.

Prior repo work (rl_litsearch_2026-07-20) already named AlphaStar "a near-blueprint" (entity
transformer + scatter connections + LSTM + autoregressive masked head + league). This digest goes
one level down: exact mechanisms, sizes, masks, and the ablation numbers.

## 1. AlphaStar (Vinyals et al. 2019, Nature 575) [S1 main text + Methods, S2 supplementary architecture]

S1 was read from DeepMind's open "unformatted" manuscript (same text as Nature, incl. Extended Data).
S2 is the supplementary "detailed-architecture.txt" (3 700 words), read in full from a GitHub mirror
of the Nature supplementary files (chengyu2/learning_alpha_star). The mirror is unofficial; its
content matches every figure S1 states (139 M weights aside, which S2 does not state).

### 1a. Observation — what was fed and how it was encoded [READ S1 ED Table 1, S2]

**Interface.** The *raw* interface: a list of units with attributes, not pixels. Four input groups
(S1 ED Table 1): *Entities: up to 512*; *Map: 128x128 grid*; *Player data*; *Game statistics*.
Entities past 512 are dropped; empty slots get a bias of −1e9 (i.e. are masked out of attention) (S2).

**One entity list for everything.** Own units, enemy units, neutral objects (minerals) and even
force-fields are rows of the SAME list; the row's `alliance` field (one-hot of 5) says whose it is.
There is no separate "teammates" or "enemies" tensor (S2). The entity encoder is permutation-invariant
(a transformer has no notion of row order), so the count of units of each kind can vary freely.

**Per-entity feature list and encodings (S2, verbatim field names).** Everything is turned into
one-hots, booleans, a few floats, or binary codes, then concatenated into one vector per entity:
- `unit_type`: one-hot, max 256 (incl. "unknown"). *This single one-hot is how all three races and
  all unit classes are told apart* — there is no per-race or per-class network.
- `unit_attributes`: 13 booleans (e.g. biological, armored) — shared semantic attributes across types.
- `alliance`: one-hot, max 5.
- `current_health`: one-hot of `floor(sqrt(min(hp, 1500)))`, i.e. ~39 buckets; shields capped at 1000,
  energy at 200, same sqrt-bucket scheme. Plus `current_health_ratio` etc. as floats in [0, 1].
  (The text literally says `sqrt(min(current_health, 1000))` for shields — an evident copy-paste typo.)
- `x_position`, `y_position`: **binary encoding** of the coordinate in game units (bits of the integer).
- `weapon_cooldown`: one-hot, max 32 game steps. `display_type` (visible / snapshot / hidden), cloak,
  powered, hallucination, active, on-screen, in-cargo: small one-hots.
- `order_1` (current order) one-hot over all order ids; `order_2..4` building queue; order progress as
  float AND one-hot of tenths; `buffs` (first two only); weapon/armor/shield upgrade levels one-hot max 4.
- `was_selected`, `was_targeted`: whether this unit was selected / targeted by the agent's LAST action
  (a cheap memory of its own previous action, per entity).

The pattern: **bucketed one-hots for magnitudes (sqrt-spaced for health and resources so low values
get fine resolution), a ratio float beside them, binary bits for positions, one-hots for categories.**
The paper gives no ablation of this encoding choice.

**Entity encoder (S2).** Transformer, **3 layers, 2 heads**, keys/queries/values of size 128 per
head, each head's output widened to 256 by a 1x1 Conv1D, heads summed, then an MLP 1024 → 256.
Outputs: `entity_embeddings` (one 256-vector per entity, kept for the pointer heads and the scatter) and
`embedded_entity` (masked mean over entities → linear 256, the pooled summary for the core).
(Secondary blogs say "2 layers"; S2 says 3.)

**Spatial encoder + scatter connections (S2).** The 128x128 map planes (S1 ED Table 1: height,
visibility, creep, entity owners, alerts = "units under attack", pathable, buildable) plus two extra
planes: a camera mask, and **scattered entities**: each entity embedding is projected to 32 channels
(1x1 conv + ReLU) and *written into the map at the entity's position*, so the 32-vector at a cell is
the unit(s) standing there. Planes concatenated → 1x1 conv to 32 ch → three stride-2 4x4 convs
(64, 128, 128 ch) down to 16x16 → 4 ResBlocks (128 ch) → linear 256 = `embedded_spatial`. The
intermediate ResBlock outputs are kept as `map_skip` for the location head (a U-Net-like skip).
How overlapping units in one cell are combined is not stated in S2 (DI-star's re-implementation
offers "add" or "cover"; see §5).

*What scatter connections are for:* they let the convolutional (spatial) pathway know *which*
learned unit sits where, rather than only a coarse "owner" plane; and they give the location head a
spatial map that already contains unit identity. S1: "To integrate spatial and non-spatial
information, we introduce scatter connections."

**Scalar encoder (S2).** Agent statistics (log(x+1) → linear 64), races, upgrades (own and enemy's
*known* ones), time (transformer positional code, 64), `available_actions` (a boolean vector of which
action types are possible given owned units/tech → 64), `unit_counts_bow` (sqrt of a bag-of-words count
of unit types), MMR (skill conditioning: imitated player's MMR in SL, fixed 6200 in RL),
`cumulative_statistics` and `beginning_build_order` (the **statistic z**, below), `last_delay`,
`last_action_type`, `last_repeat_queued`. A subset (race, available actions, z) is also concatenated
into `scalar_context`, which **gates** (GLU) the action-type logits and the autoregressive embedding.

**Core.** Concatenate `embedded_entity`, `embedded_spatial`, `embedded_scalar` → **LSTM, 3 layers of
384**, layer-norm on gates (S2). Memory is how partial observability is handled (S1).

**Size.** 139 M weights total, 55 M needed at inference (the rest are the training-only value
networks) (S1 Methods). Each evaluation agent ran on "a single high-end consumer GPU" (S1).

### 1b. Actions — factoring, target selection, masks, cadence [READ S1 ED Table 2, ED Fig. 3, S2]

**The factoring.** An action is a *function with typed arguments* (S1 "AlphaStar Generality"):
`action_type` (several hundred: move, attack, build X, train Y, move camera, no-op …), `delay`,
`queued`, `repeat`, `selected_units` (who executes), `target_unit` OR `target_location` (never both:
"no action has both a target unit and a target location", S2). ~10^26 joint choices per step (S1).
Arguments that a given action type does not use are ignored and their loss/entropy is masked (S2).

**Autoregressive order (chain rule).** action_type → delay → queued → selected_units → target_unit
or target_location. After each argument is sampled, an embedding of it is ADDED to a running
1024-wide `autoregressive_embedding` that conditions every later head (S2). This is how "the target
depends on what kind of action was chosen and who does it".

- **Action-type head:** LSTM output → linear 256 → **16 ResBlocks** (layer norm, 256) → GLU gated by
  `scalar_context` → logits; sampled at temperature 0.8. Illegal action types: the gate input includes
  `available_actions`, but S2 does not describe a hard mask on action type. *Not stated in S2.*
- **Delay head:** 128-way softmax over "how many game steps until I next observe" (no temperature).
  **The agent chooses its own observation cadence** — it is part of the action (S1 ED Table 2).
- **Queued head:** 2-way.
- **Selected-units head = recurrent pointer network** (a pointer network outputs a choice *among the
  input items* by scoring each item, instead of a fixed-size softmax). Mechanism verbatim in S2:
  1. `func_embed`: one-hot of *which unit types can accept the chosen action_type* → linear 256.
  2. A mask of selectable units, "initialised to allow selecting all entities that exist (including
     enemy units)" — further restriction by action type is implied but not spelled out.
  3. **Keys** = each entity's `entity_embedding` through a 1x1 Conv1D to **32** dims; plus one extra
     "end selection" key.
  4. **Query** = `autoregressive_embedding` → linear 256, + `func_embed`, ReLU, linear 32 → fed through
     an **LSTM of size 32** (zero initial state; one step per selected unit).
  5. Logits = key · query (dot product); masked; sampled at temperature 0.8. The chosen entity is
     then masked out; its key (one-hot × keys, mean-reduced) → linear 1024 is added to the
     autoregressive embedding. Repeat up to **64** units or until the end key is picked.
- **Target-unit head:** the SAME construction with ONE step: query = autoregressive embedding
  (linear 256) + `func_embed` → ReLU → linear 32; logits = query · the same 32-dim entity keys;
  temperature 0.8 (S2). Extended Data Fig. 3 labels it "Attention". **How illegal targets are masked is
  not described in S2 for this head** (only the "entities that exist" mask for selection is stated).
  Because keys come from the shared entity encoder, the score of "attack enemy k" already contains
  k's type, health, position and its attention-context with every other unit.
- **Location head:** the autoregressive embedding is reshaped to 16x16x4, concatenated with the last
  `map_skip`, then Gated ResBlocks with FiLM (feature-wise scale-and-shift conditioned on the
  autoregressive embedding) using the skips in reverse order, then transposed convs **upsampling to
  256x256** logits; invalid locations masked by action type (e.g. outside the camera for build
  actions); temperature 0.8 (S2). So a point is chosen as a softmax over a 256x256 grid — a
  *spatial pointer*, not an (x, y) regression.

**Rate limits and delays (S1 Methods, READ).** "Agents are limited to executing at most 22
non-duplicate actions per five second window" (≈264/min); one game step = 45 ms; real-time delay
≈110 ms observe→act (ED Fig. 2: 113 ms); self-chosen wait between observations averages 370 ms
(ED Fig. 2: 369 ms), "possibly multiple seconds".

### 1c. What was hidden, and what the value function saw [READ S1 ED Table 1, Methods; S2]

- "Information is hidden if it would be hidden to a human player": units in the fog of war **do not
  appear in the list**; enemy buildings last seen appear as `display_type = snapshot` (a remembered
  ghost); enemy units outside the camera carry only owner, display type and position; cloaked units
  appear only inside the camera (S1 ED Table 1 caption). The visibility plane marks currently-visible cells.
- **Asymmetric critic.** The value function (the network that predicts the expected outcome and is
  used only to reduce gradient variance during training) gets the **opponent's full observations**
  (entity list, map, scalars) and the game's hidden `cumulative_score` (S2 "Network inputs";
  S1 Methods: "only used during training, as value functions are unnecessary during evaluation").
  The baseline net: concat features → linear 256 → 16 ResBlocks → scalar, squashed by
  (2/π)·atan((π/2)·x) (S2).

### 1d. The statistic z — a strategy the policy is conditioned on and rewarded for [READ S1, S2]

- z = the first 20 constructed buildings/units (build order) + "cumulative statistics" (booleans of
  which units, buildings, effects, upgrades appeared in the game) (S1 Methods). Extracted from each
  human replay; the policy is conditioned on it in SL and RL; set to zero 10 % of the time in SL.
- In RL, main agents sample a z from human data at game start (same map and race pairing, S2) and get
  **pseudo-rewards** for following it: negative edit (Levenshtein) distance on the build order,
  Hamming distance on units / upgrades / effects, each pseudo-reward active with 25 % probability,
  each with **its own value function** (S1 Methods). Unit/upgrade/effect pseudo-rewards halve after
  8 and 16 min and stop at 24 min (S2). Policy loss weights: win-loss 1.0, build order 4.0, the three
  Hamming rewards 6.0 each (S2).
- Ablation (Fig. 3E): "+ Statistics" gives the highest test Elo of the human-data variants (see 1e).
- Use for a commander: z is the documented precedent for **"an order is an input vector, and a
  pseudo-reward for complying with it is a separate reward head with its own critic"**. The 17
  "validation agents" were trained with z FIXED to hand-curated strategies (e.g. cannon rush) (S1
  "Agent sets") — i.e. the same network becomes a different, steerable player by its input.

### 1e. Ablations and costs (S1 Fig. 3, READ; one map, PvP, simplified setup)

Fig. 3 numbers as extracted from the manuscript's text layer. Bar labels and values are laid out in
the PDF side by side; the pairings below are my reading of that layout (the order is unambiguous for
F and K; for E the label-to-number pairing is my best reading).
- **F, architecture** (supervised win-rate vs built-in Elite bot), cumulative additions from a
  baseline: Baseline 0 % → + action delays 7 % → + pointer network 36 % → + transformer 71 % →
  + scatter connections 87 %. *So: the pointer network for unit selection/targeting added 29 points,
  the entity transformer 35, scatter connections the last 16 (the order of addition is fixed, so
  these are marginal gains in that order, not independent effects).*
- **H, interface:** non-camera (whole-map raw) 96 %, camera interface 87 %. Limiting attention to a
  camera costs ~9 points.
- **K, value function:** with opponent info 82 %, without 22 % (average win-rate of a best response vs
  fixed opponents). **The privileged critic was the single largest RL effect reported.**
- **I, off-policy learning:** V-trace 49 % → + TD(λ) 73 % → + UPGO 82 % (my pairing of the stacked labels).
- **G, APM limits** (test Elo, my reading of the layout: seven labels, seven numbers in order):
  no limit 1392; 200 % 1411; **100 % (AlphaStar's) 1540**; 50 % 1536; 25 % 1419; 10 % 1145; 0 % 0.
  Cross-check: 1540 equals the "+ Statistics" bar of panel E, i.e. the default agent. Caption:
  "Reducing APM significantly reduces performance. Surprisingly, increasing APM also reduces
  performance, possibly because the agent spends more effort refining micro-tactics than learning
  diverse strategies." (Similarly F's 87 % equals H's camera-interface 87 %: the full agent.)
- **E, human data** (test Elo): + statistics 1540; + supervised KL ~1400; human init ~1020;
  supervised ~936; no human data 149. "We found our use of human data to be critical" (Methods).
- **C/D, multi-agent:** naive self-play has high Elo "but is more forgetful"; PFSP/FSP less so.
- **Supervised fine-tuning:** on 16 000 winning replays with MMR > 6200 raised PvP win-rate vs Elite
  bot from 87 % to 96 % (Methods).

**Compute (S1).** SL on 971 000 replays (MMR > 3500, top 22 %). League: 12 agents (per race: 1 main,
1 main exploiter, 2 league exploiters); each agent 32 TPUv3 learner-side ("32 third-generation TPUs
over 44 days"); per agent 16 000 concurrent games, 16 actor TPUs, ~150 CPUs × 28 cores; learner
~50 000 agent steps/s, batch 512, actors refresh weights every 10 s; ~900 distinct players created.
Final: Grandmaster, above 99.8 % of ranked EU players.

**League (S1, READ).** Main agents: 35 % self-play, 50 % PFSP (prioritised fictitious self-play:
opponents sampled in proportion to (1 − p_win)^p so hard opponents get most games) vs all past
players, 15 % vs forgotten players and exploiters. Main exploiters target the current main agents
(reset to SL weights each time they are frozen); league exploiters target the whole league. RL loss:
V-trace for the policy (split per action-type / delay / other arguments, because a joint importance
ratio over the autoregressive product is cut too early), TD(λ=0.8) for the values, UPGO (a
self-imitation update toward better-than-expected partial trajectories), KL distillation to the SL
policy (weight 2e-3), entropy 1e-4 (S1, S2).

## 2. SC2LE / PySC2 (Vinyals et al. 2017, arXiv:1708.04782) — the interface AlphaStar replaced [S3, PART]

- **Observation:** pixel-aligned *feature layers* of a "screen" (camera view) and a "minimap", 64x64 in
  the experiments, plus non-spatial vectors; unit type and hit points exist only as screen layers (S3).
  No entity list.
- **Actions:** "an action a is represented as a composition of a function identifier a0 and a
  sequence of arguments": ~300 function ids, 13 argument types; units are selected the human way,
  e.g. `select_rect(select_add, (x1,y1), (x2,y2))`, then commanded at a later step (S3).
- **Spatial argument:** two baselines — Atari-net models (discretised) x and y *independently*;
  FullyConv outputs a 1x1-conv map over the screen (a spatial policy). Baseline factorisation was fully
  independent per argument, π(a|s) = Π π(a^l|s); an autoregressive variant was tried for supervised
  policies and "the accuracy of predicting a point on the screen can be improved by conditioning on the
  base action". Unavailable function ids masked and renormalised (S3).
- **Results:** DefeatRoaches best mean score ~100 vs novice tester 41 vs GrandMaster 215; on the full
  game vs the easiest built-in AI "baseline agents do not learn to win even a single game" (S3).
- **Why AlphaStar moved away (my synthesis of S1 vs S3, opinion):** unit selection by rectangles and
  points cannot address one unit among hundreds reliably; the entity list + pointer selects units by
  identity; independent x/y heads cannot represent a multi-modal target distribution (two good
  targets give a "cross" of four points), which a 2D softmax and the autoregressive chain fix.
  AlphaStar keeps the 2D-softmax idea (256x256 location logits) and adds the pointer heads.
- [RECALL] The January 2019 exhibition AlphaStar saw the whole map (no camera); the Nature version
  added the camera interface. S1 confirms only that a camera was used and that non-camera scored
  higher (Fig. 3H); TStarBot-X (S5) also refers to "the demonstration version of AlphaStar" without camera.

## 3. AlphaStar Unplugged (Mathieu et al. 2023, arXiv:2308.03526) — offline RL on the same interface [S4, PART]

- Same raw interface: 8 world planes at 128x128 (incl. `virtual_camera`), up to 512 units with a
  **43-feature vector each**, global vectors; seven action arguments sampled "function, delay,
  queued, repeat, unit_tags, target_unit_tag and world"; "unit_tags is actually obtained by sampling the
  pointer network 64 times autoregressively"; target unit by pointer; world = "a 2d point on the
  feature planes", "sampled from the upsampled feature planes" (S4).
- Architecture "an improved version of the model used in" AlphaStar: MLP for vectors, transformer
  for units, ResNet for planes, units scattered into planes, unit embeddings averaged into vectors (S4).
  Sizes are in the released code (github.com/google-deepmind/alphastar), not in the paper.
- Data: ~1.4 M games (MMR > 3500, 10 maps), ~3.5 G training observations; ~10^10 frames per agent (S4).
- **Results:** best offline agents (MuZero-Supervised + MCTS at inference; offline actor-critic) reach
  ~90 % vs the all-race AlphaStar Supervised agent; MCTS *during* training failed: "the policy would
  collapse quickly to a few actions" (S4).
- **Ablations that bear on Breach (S4 §5.6–5.7, quoted; win rate vs the very_hard built-in bot):**
  memory — LSTM 70 %, **no memory 84 %**, transformer 85 % (89 % at 2·10^10 frames): "Surprisingly, we
  found that no memory performs better than LSTM for behavior cloning." Width halved 83 → 76 %; depth
  halved barely matters (82 %). Batch 8k → 32k: 70 → 84 %. Inference temperature 0.8 raised BC
  84 → 88 %. No ablation of the transformer or the scatter. Caveat: behaviour-cloning numbers, not RL.

## 4. TStarBot-X (Han et al. 2020, Tencent, arXiv:2011.13729) — AlphaStar at far smaller compute [S5, PART]

- **Compute:** 144 V100 GPUs (96 train, 48 inference), 13 440 CPU cores, 57 days; ~20 M parameters
  (vs 139 M); 309 571 Zerg-vs-Zerg replays; one race only (S5 Table 1). Beat two Master players 11:0
  and 13:0 and two Grandmasters 7:0 and 4:1 (S5).
- **Observation:** like AlphaStar's but **without the camera** ("we indeed remove the camera
  constraint like the demonstration version of AlphaStar to facilitate the learning"); masks "encoding
  the game logic" given as inputs; unit list padded to 600 (S5).
- **Kept:** central LSTM, transformer over units, sequential selection head, "pointer network for
  choosing a target unit" (S5).
- **Cut:** FiLM, GLU except on the ability head, LSTM to one layer, ≤ 3 stacked ResBlocks; the
  build-order transformer replaced by an LSTM ("slightly superior"). **Value net:** instead of
  AlphaStar's symmetric full opponent encoding, only a short vector of enemy statistics + the
  opponent's z — "around 30 times smaller", still centralised (S5).
- **What failed, with mechanism (S5 §4.3):**
  - *Multi-binary unit selection* (each unit independently in or out) "generally works well and is
    much more efficient compared to sequential predictions" but misses some intended units (the
    independence problem).
  - *Scatter + "gather" connections with NO transformer* (units interact only through convolutions,
    to avoid O(n²) attention): "the trained agent prefers to control a group of units that are
    spatially close to each other and barely controls one or a few units for specific
    micro-management." → **attention between entities is what made single-unit micro possible.**
  - *Selection inside the autoregressive chain* is superhuman (select + command in one step); folding
    selection into the ability head (human-like, two steps) dropped EPM/APM from 0.8–0.85 to ~0.5.
  - A faithful AlphaStar league at their scale failed (exploiter periods > 10 days, weak exploiters);
    they added exploiter roles (DLT), a rule-guided loss toward an expert teacher on "critical" states
    (RGPS), and DAPO. "distributed PPO + UPGO does not show much difference compared with V-trace +
    UPGO within 5 days of training" (S5).

## 5. DI-star (OpenDILab, 2021–) and DI-engine [S6, S7, PART]

- Open PyTorch re-implementation; released Zerg-vs-Zerg agents: SL model "equal to diamond players",
  RL model "equal to master or grandmaster"; SL on replays → RL league initialised from SL; README
  shows a 4-GPU example and a "one PC" baseline guide (S6). No statement of architecture changes.
- DI-engine's `ScatterConnection` makes S2's unstated detail explicit: two entities on one cell are
  combined by `scatter_type` "add" or "cover"; "'cover' mode will result in the loss of information",
  "'add' mode is used as a temporary substitute"; integer placement, no interpolation (S7).

## 6. ROA-Star (Huang et al., NeurIPS 2023, Tencent) — opponent modelling behind the fog [S8, PART]

- Compute 64 V100 + 4 600 CPU cores, 50 days of league (its Table 1 lists AlphaStar as 256 TPU cores
  + 4 100 preemptible CPU cores — an accounting different from S1's per-agent figures) (S8).
- **Opponent prediction under fog:** during SL, a β-VAE-style encoder over *visible* opponent
  information only (armies, buildings, tech; own entities filtered out) predicts the opponent's
  *hidden* state as classifications (count per entity type, tech in research, location of key
  buildings), focal loss; in league training the encoder is frozen and its latent is a policy input (S8).
- **Scouting reward:** every 30 s, reward = decrease of that prediction's cross-entropy (never
  penalised for increases). "the 'scouting' reward greatly encourages the effective 'scouting'
  behavior and improves the overall performance" (S8). Its motivation: AlphaStar "rarely conducts
  effective scouting" and was fragile to e.g. the cannon rush (S8 footnote 5).
- Goal-conditioned exploiters: exploiters trained on fixed z's on which the main agent wins most,
  rewarded by the z pseudo-reward — z reused as a *steering* input for adversaries (S8).

## 7. 2024–2026: language models above StarCraft control [S9–S13, ABS]

- **TextStarCraft II + Chain of Summarization** (Ma et al., NeurIPS 2024, arXiv:2312.11865): the game
  as text through python-sc2; an LLM makes macro decisions; most of 10 LLMs beat the level-5 built-in
  AI; fine-tuned LLMs about Gold level vs humans (S9, search summary only).
- **Hierarchical Expert Prompt** (arXiv:2502.11122, 2025): claims the first LLM win over the Elite AI
  in TextStarCraft II; the environment "liberates LLM from high-speed micro-operations by handing over
  these operations automatically" (S10, snippet).
- **LLM-PySC2** (arXiv:2411.05348): LLM sub-agents act through the PySC2 action space; abstract: LLMs
  "cannot constantly generate correct decisions, especially in the recovered pysc2 action space and MA
  settings", with "hallucinations and inefficient collaboration" without task-specific instructions (S11).
- **HIMA** (COLM 2025, arXiv:2508.06042): a Strategic Planner orchestrates imitation-learned specialist
  agents; macro only, micro left for future work (S12).
- **Reading:** 2024–2026 language-model work sits at the *strategic* layer and leaves unit control to
  scripts or learned policies. I found no 2024–2026 StarCraft work that supersedes AlphaStar's entity
  transformer + pointer design for unit-level control (search was not exhaustive). A June-2026 SC
  micro paper (arXiv:2606.30092) uses tabular hierarchical Q-learning with influence maps — not a
  successor architecture (S13).

## 8. For Breach

Two readings: **(A) as a model for a later COMMANDER** — AlphaStar *is* one policy commanding many
units; and **(B) as a parts bin for the per-soldier policy**. Opinions are marked (op.).

**Q1 Representation.**
- Take: **one entity list, one shared encoder** — self, teammates, enemies, zombies, neutral objects
  (doors, grenades in flight) as rows of one list, told apart by an alliance one-hot and a type
  one-hot, padded and masked with −1e9 (S2). Transformer over the list; masked mean pooled for the
  core; per-entity embeddings kept for the pointer heads (S2).
- Take: **scatter connections** into the local field-plane crop: project each entity embedding to a
  few channels and write it at its tile, so convolutions over fire/smoke/pressure planes see *who*
  stands in them (S2; +16 points in Fig. 3F, the last addition). Use "add" on overlaps (S7). The
  location head reads the same spatial trunk via skips (S2).
- Take: the encoding habits — sqrt-bucketed one-hot + ratio float for health; one-hot cooldowns;
  `was_selected` / `was_targeted` ("was this the target of my last action") (S2).
- Change (op.): AlphaStar uses *absolute* binary-coded map positions because it is one global player;
  a per-soldier policy should get positions *relative to itself* (and bearing relative to its facing).
  The binary coding is unablated in S2.
- Avoid: TStarBot-X's convolution-only interaction (scatter + gather, no attention) — it lost
  single-unit micro (S5).

**Q2 Aim as an action.** S2 gives the exact template for "aim at enemy k":
- **Keys** = per-entity embedding → 1x1 conv to 32; **query** = core output + the embedding of the
  chosen verb → MLP to 32 (an extra 32-wide LSTM only for multi-unit selection); logits = query·key;
  masked; softmax at temperature 0.8 (S2). The verb conditions the query through the autoregressive
  embedding, so "fire at k" and "throw at k" can prefer different k.
- The mask is under-specified in S2 (only "entities that exist", for selection). Breach must define it
  (op.: enemy rows seen or remembered; never self; teammates masked unless a verb needs them).
- AlphaStar never aims: "attack unit k" is executed by the game's own unit AI. Breach's turning law
  already turns toward a *target* (the Turning law row of the project CLAUDE.md), so "aim at k" = set
  the facing target to k: the same division of labour. **This supports the author's leaning** against
  turn-left / turn-right.
- **"A direction when no enemy is in sight":** AlphaStar's answer for "a place, not a unit" is a
  separate *location head*, a 2D softmax over a grid (256x256, upsampled from the spatial trunk, masked)
  (S2), never an (x, y) regression; SC2LE's independent-x/y baseline is the version it replaced (S3).
  For Breach (op.): a spatial softmax over the crop (aim at a cell) or a ring of bearing bins.
  Remembered-enemy rows let "aim at k" also mean "aim where k was last seen".
- "No action has both a target unit and a target location" (S2): unit-or-place is a choice inside the chain.

**Q3 Factoring.**
- Verbs with typed arguments and chain-rule sampling (action type → … → target) support **one action
  type per grenade kind**: every SC2 ability is its own action type, and `func_embed` tells the
  pointer which units can take it (S2).
- **Contradiction with Breach's needs (op., important):** AlphaStar emits ONE action type per step
  (≤ 22 actions per 5 s, S1). A soldier moves, aims and holds the trigger *at once* every tick.
  Copying the single action-type head would force "move OR shoot". Use the chain only *within* a
  channel whose arguments depend on each other (verb → target) and independent heads for concurrent
  channels (movement, aim target, trigger / verb). The split V-trace (action type, delay, rest updated
  separately, S1/S2) is the precedent for treating parts of an action as separate policy terms.
- **Delay head:** AlphaStar chooses *when it next observes* (128-way; mean 370 ms; S1, S2) — a learned
  action repeat. A Breach option (op.) if per-tick decisions are too costly; `VERB_RULES` decimation
  is the fixed version.
- S1: AlphaStar's beyond-human interface powers were "not exploited" *because of the human prior*.
  Breach has no human replays; recorded bot matches are the only analogue.

**Q4 Team display.** AlphaStar has no teammates, so no direct precedent. As a commander (A) it shows
shared knowledge inside one network: each own unit's current order is a feature of its row
(`order_1`, order progress) and the last action's target is flagged on the target's row (S2).
Op.: the soldier's team display can reuse the shape — teammate rows carry health, current verb and a
"targeting" link rendered as a count or flag on the targeted enemy's row (pointer-friendly), plus
remembered enemy rows from the shared picture with a last-seen age (precedent: `display_type =
snapshot` for fogged buildings, S1 ED Table 1).

**Q5 Orders.** The **statistic z** is the strongest precedent: an order is a conditioning vector in
the scalar context (it *gates* the action logits), sampled per episode, zeroed 10 % of the time in SL,
with a **pseudo-reward for compliance that has its own value head**, each active 25 % of the time,
plus an extra distillation on action types for the first four minutes when z is used (S1, S2).
Fixing z produced 17 distinct steerable strategies (S1); ROA-Star reused z to steer exploiters (S8).
Op.: give the soldier an objective input from day one (target tile / hold here / follow teammate k),
trained with a separate compliance reward and sometimes absent, so a commander can drive it later
without retraining.

**Q6 Fog and field of view.**
- Unseen enemies are *absent from the list*; remembered buildings appear as snapshots; detail is
  withheld outside the camera (S1). Camera vs whole map cost 96 → 87 % (Fig. 3H) — DeepMind paid it
  for fairness, not performance. Op.: a cone FOV is Breach's analogue of the camera; measure its cost
  the same way before deciding.
- **Privileged critic:** the value function saw the opponent's full observation — 82 % vs 22 % without
  (Fig. 3K, the largest single RL ablation in the paper). TStarBot-X kept a ~30x smaller privileged
  critic (S5). Take this: the critic sees the true full state, the policy only its view.
- ROA-Star's next step: an auxiliary head predicting hidden enemy state from visible information, and
  a reward for *reducing that prediction's error* to make scouting pay (S8).

**Q7 Heterogeneous units.** One encoder for ~256 unit types and three races: a type one-hot + 13
shared attribute booleans per row (S2); one policy per race in the Nature version (S1), all races in
one model in Unplugged (S4). Legality per type is an input (`func_embed`, `available_actions`), not a
separate network. Op.: lean on shared *attributes* (armour, speed, kit, can-melee) more than a type
id, so a new class is a new row of attributes; per-class verb masks as inputs.

**Q8 State of the art.** Questioned since 2019: the LSTM core (Unplugged: no memory, or a
transformer, beat the LSTM in BC, S4); V-trace vs PPO is a wash at TStarBot-X scale (S5); the huge
symmetric critic (30x smaller works, S5). Not superseded for unit control in this literature: entity
transformer + pointer targeting + spatial softmax. Language-model agents sit above control (S9–S12);
op.: an LLM belongs, if anywhere, at Breach's commander layer.

**Avoid.** The recipe's scale (12 agents x 32 TPUs x 44 days; ~1 M human replays) and its dependence
on human data ("critical": Fig. 3E, no human data 149 Elo vs 1540). Breach has no human corpus; a bot
corpus or the bots as teachers (TStarBot-X's RGPS-style loss) is the substitute (op.). The
superhuman select-and-command-in-one-step that TStarBot-X flags is a commander-layer issue only.

**Open.** Target masking rules; relative vs absolute position code; whether the soldier needs
recurrent memory if the team display carries the shared picture (Unplugged's no-memory result hints
it may not, op.); the cost of a cone FOV; whether to scatter remembered enemies into the crop.

## 9. Source ledger

| Key | Citation | URL | Status | Supports |
|---|---|---|---|---|
| S1 | Vinyals et al. 2019, "Grandmaster level in StarCraft II using multi-agent reinforcement learning", Nature 575, doi:10.1038/s41586-019-1724-z (DeepMind's open unformatted manuscript with Methods and Extended Data) | https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf | FULL (pdftotext; Fig. 3 values from the text layer, label pairings my reading, cross-checked) | interface, fog, APM 22 per 5 s, delays 110 / 370 ms, Fig. 3 ablations, z, privileged critic, league, compute, 139 M / 55 M weights |
| S2 | AlphaStar Supplementary Data, "detailed-architecture.txt" (unofficial GitHub mirror of the Nature supplement) | https://github.com/chengyu2/learning_alpha_star/blob/master/detailed-architecture.txt | FULL | per-entity features + encodings, layer sizes, pointer keys / queries / masks, location head, critic inputs, pseudo-rewards, loss weights |
| S3 | Vinyals et al. 2017, "StarCraft II: A New Challenge for Reinforcement Learning" (SC2LE), arXiv:1708.04782 | https://arxiv.org/html/1708.04782v1 | PART | function-id action space, independent x/y vs FullyConv spatial policy, results |
| S4 | Mathieu et al. 2023, "AlphaStar Unplugged: Large-Scale Offline Reinforcement Learning", arXiv:2308.03526 | https://arxiv.org/html/2308.03526 | PART (first ~100k chars) | 43 unit features, argument order, pointer x64, memory / size / batch ablations, offline results |
| S5 | Han et al. 2020, "TStarBot-X: An Open-Sourced and Comprehensive Study for Efficient League Training in StarCraft II Full Game", arXiv:2011.13729 | https://arxiv.org/abs/2011.13729 | PART (§1–4.6 + Table 1, pdftotext) | compute, slimmed net, no camera, small critic, failed alternatives, league changes, PPO ≈ V-trace |
| S6 | OpenDILab, DI-star README | https://github.com/opendilab/DI-star | PART | released levels, pipeline |
| S7 | DI-engine `ScatterConnection` source | https://di-engine-docs.readthedocs.io/en/latest/_modules/ding/torch_utils/network/scatter_connection.html | PART | add vs cover on overlap |
| S8 | Huang et al. 2023, "A Robust and Opponent-Aware League Training Method for StarCraft II" (ROA-Star), NeurIPS 2023 | https://papers.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf | PART (§1–3 + Table 1, pdftotext) | opponent prediction behind fog, scouting reward, goal-conditioned exploiters, compute |
| S9 | Ma et al. 2024, "Large Language Models Play StarCraft II: Benchmarks and A Chain of Summarization Approach", NeurIPS 2024, arXiv:2312.11865 | https://arxiv.org/abs/2312.11865 | ABS (search summary) | LLM macro layer, results vs built-in AI |
| S10 | "Hierarchical Expert Prompt for Large-Language-Model: An Approach Defeat Elite AI in TextStarCraft II for the First Time", arXiv:2502.11122 | https://arxiv.org/html/2502.11122 | ABS (search snippet) | LLM macro only, micro automated |
| S11 | LLM-PySC2, arXiv:2411.05348 (+ repo NKAI-Decision-Team/LLM-PySC2) | https://arxiv.org/abs/2411.05348 | ABS | LLM failures in the PySC2 action space and multi-agent settings |
| S12 | Ahn, Kim, Choi 2025, HIMA, COLM 2025, arXiv:2508.06042 | https://arxiv.org/abs/2508.06042 | ABS | planner over imitation agents, macro only |
| S13 | Bai et al. 2026, HRL-IM/CBS, arXiv:2606.30092 | https://arxiv.org/abs/2606.30092 | ABS | 2026 SC micro work, tabular, not a successor |
| — | mini-AlphaStar (liuruoze) and "Rethinking of AlphaStar", arXiv:2108.03452 | https://github.com/liuruoze/mini-AlphaStar | ABS | reproduction exists; its criticisms were not read |
| — | AlphaStar January 2019 exhibition without camera | — | RECALL (TStarBot-X's "demonstration version" phrase is consistent) | §2 last bullet |

