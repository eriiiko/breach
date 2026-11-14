# 06 — Orders for a later commander, and many unit classes in one architecture

*Raw research-agent digest, 2026-11-11, for the RL soldier-agent survey. Builds on
`../rl_litsearch_2026-07-20_raw/02_hierarchical_commander.md` (not repeated) and goes one
level down on mechanism: what the order input was, where training orders came from, how
obedience was paid for, and what it cost.*

Status tags: READ = read in a source this session (quotes are from the extracted text);
PART = relevant sections read; ABS = abstract or secondary account; RECALL = from memory,
not checked. "Opinion" marks the agent's own judgement.

**Short answer:** reserve an order input now, as a small typed tuple
`<kind, tile-or-entity, duration>` with `none` as a normal value; train it from day one
with generated, hindsight-labelled and scripted-team-layer orders; pay obedience with a
shaped reward whose weight is a swept dial and that is off in a share of episodes. The
best measured case (Tencent MCC) shows obedience near 100 % costs win rate sharply, and
~80 % keeps it. For classes: one network per verb-grammar family with a class embedding
and a union action space under per-class masks; distil specialists into one when needed.

## A. Low-level policies built to take orders before (or without) the final commander

### A1. AlphaStar's strategy statistic z (Vinyals et al., Nature 2019) — READ (DeepMind's unformatted preprint PDF)

- **What the "order" is.** z is a summary of one human player's game: the build order
  ("the first 20 constructed buildings and units") plus "cumulative statistics, defined as
  the units, buildings, effects, and upgrades that were present during a game". It is
  extracted from each human replay. The policy is pi(a | observations, z) — z is an extra
  input to the one network, present from the first (supervised) training stage.
- **How it was trained to be usable.** In supervised learning (imitating human replays) the
  network sees the replay's own z — i.e. the "order" is *relabelled from what the human
  actually did*, the same idea as HER (hindsight relabelling) — and "in supervised learning
  we set z to zero 10% of the time", so the network also learns an unconditioned mode. In RL
  the main agents get z sampled from human data and pseudo-rewards (extra reward terms
  added to win/loss) for following it: "edit distance between sampled and executed build
  orders, and the Hamming distance between sampled and executed cumulative statistics".
  "Each type of pseudo-reward is active (i.e. non-zero) with probability 25%, and separate
  value functions and losses are computed for each pseudo-reward." Agents are trained
  either conditioned on z ("receive a reward for following the strategy") or
  "unconditionally, in which case the agent is free to choose its own strategy".
- **What it bought (ablation, Fig. 3E, Protoss-v-Protoss, one map, test Elo).** Reading of
  the figure's bars, top to bottom (layout reconstructed from extracted PDF text; the
  label-to-number pairing is my reading): +Statistics 1540, +Supervised KL 1400, Human init
  1020, Supervised 936, No human data 149. So adding z + pseudo-rewards on top of
  everything else added roughly +140 Elo; the big jumps come from human data in general.
  The authors: "We found our use of human data to be critical."
- **What z is used for besides strength.** Steering: "Validation Agents ... fixing z to a
  hand-curated set of interesting strategies (e.g. a cannon rush or early flying units)" —
  the *same trained network* plays a chosen strategy on demand. That is exactly a
  low-level policy that takes a high-level order, built before any commander existed.
- **Caveat stated by the authors:** "the particular choice of statistics required domain
  knowledge ... Pseudo-rewards were based on appropriate distance metrics for these
  statistics, such as edit distance or Hamming distance."
- **For Breach (opinion):** the z recipe is the cleanest precedent for "reserve an order
  input from day one, fill it by hindsight from what was actually done, sometimes zero it,
  and pay a *probabilistic* (25 %-of-episodes) obedience bonus". Its order is coarse
  (whole-game strategy), not a per-second tactical order.

### A2. OpenAI Five: lane assignments, and surgery (Berner et al., arXiv:1912.06680) — READ (full PDF text)

- **Lane assignments = a hand-given order with an obedience penalty.** Appendix O: agents
  "developed a preference to stick together as a group of 5 on a single lane" (a local
  minimum); "we introduced lane assignments, which randomly assigned each hero to a subset
  of lanes, and penalized them with negative reward for leaving those lanes". The lane
  assignment was also an observation ("Lane Assignments" in the per-hero observation
  table). Result: "the ablation study in Figure 30 indicates that this may not have been
  necessary in the end" ("this randomization actually provided little benefit").
  Note: the assignments were *random*, i.e. orders sampled from a distribution during
  training, not from a commander.
- **Adding an input later: surgery.** "Most of our surgeries caused the observation space
  changes"; for an additive new observation they extend the first weight matrix with zero
  columns, W_new = [W 0], so the policy is unchanged and the new weights grow from zero by
  gradient. New actions were "annealed" in: "starting with 0% of rollout games played with
  the new environment or actions, and slowly ramping up to 100%". "Over twenty surgeries"
  in 10 months, about one per two weeks; frozen past opponents must be surgered too.
- **Cost of surgery (the key evidence for "reserve vs add later").** "the model ultimately
  plateaued at a weaker skill level than the from-scratch model"; Rerun (trained from
  scratch on the final environment) used ~20 % of the compute and "reached over 98%
  winrate against the final version of OpenAI Five". Removing inputs is not supported:
  "our model continued to observe some 'deprecated' observations, which were simply always
  set to constants". The authors: "in situations where the environment is pre-built and
  well-understood from the start, we see little need for surgery."
- **For Breach (opinion):** zero-column surgery makes "add the order input later"
  technically cheap for a feed-forward input, but OpenAI Five's own evidence is that a
  network grown by surgery ends weaker than one trained on the final interface. Adding an
  input that the policy must *obey* (not just exploit) also needs a new reward term and
  retraining of behaviour, which surgery does not give for free.

### A3. Goal-conditioned agents in rich worlds: XLand, SIMA, SIMA 2

- **UVFA / HER (RECALL, already in the July digest).** UVFA: one value network V(s, g)
  takes the goal g as an input next to the state. HER (hindsight experience replay):
  after an episode, store it again as if the goal had been whatever the agent actually
  achieved, so failed attempts become successful examples of *some* order. AlphaStar's
  supervised z (A1) and SIMA's post-hoc annotation (below) are the same trick at scale.
- **XLand (Open-Ended Learning Team, arXiv:2107.12808) — READ (full PDF text).** The order
  is a goal written as a Boolean formula over ~200 "atomic predicates" (near, on, seeing,
  holding, and negations, over objects, players, floors): a disjunction of up to 3
  "options", each a conjunction of up to 3 predicates. Reward is simply 1 on every step the
  goal formula is true — the order *is* the reward, there is no separate obedience bonus.
  Network: the goal is embedded and used as the *query of an attention module* (GOAT,
  goal attention) over the LSTM's hidden state; an auxiliary head predicts which
  predicates are true. Ablation: removing GOAT for "a simpler architecture" cost "another
  ~30% reduction in performance"; removing PBT ~20 %, removing dynamic task generation
  (the curriculum that picks goals neither too easy nor too hard) ~65 %. Orders during
  training come from a *procedural generator* filtered by the curriculum, not from a
  commander. Fine-tuning the general agent on a new task for 100 M steps (~30 min) reached
  77 % of an upper bound against 39 % zero-shot and ~9 % from scratch.
- **SIMA (arXiv:2404.10179) — PART (HTML, via WebFetch quotes).** Keyboard-and-mouse
  actions, language instructions, trained "via behavioral cloning" (imitation of human
  play) plus "an auxiliary objective of predicting goal completion". Instructions come from
  "allowing single players to freely play, and then annotating these trajectories with
  instructions post-hoc" (hindsight labelling by humans) and from "setter-solver"
  sessions where "one player instructs another". 1,485 tasks in 9 skill categories, 7
  games. The "No language" ablation "performs very poorly": on No Man's Sky 34 % vs 11 %
  success (humans 60 %). No RL in the main agent.
- **SIMA 2 (arXiv:2512.04797, Nov/Dec 2025) — ABS (search results and secondary accounts
  only).** Built on Gemini; trained on human gameplay; adds a self-improvement loop where
  Gemini proposes tasks and a learned reward model scores trajectories. Not read in full;
  numbers in secondary accounts not verified.
- **For Breach (opinion):** the strongest goal-following agents got their orders from a
  *task generator* (XLand) or *hindsight labels* of what was done (SIMA, AlphaStar SL),
  never from a commander; and the order entered as an embedding the network attends with,
  not as a flat bit appended late. A language interface is far beyond what Breach needs.

### A4. Hierarchies in team combat

- **FMH, Feudal Multi-Agent Hierarchies (Ahilan & Dayan, arXiv:1901.08492) — READ (full).**
  The manager's message *defines the worker's reward* ("each defining a worker reward
  function proportional to the negative distance from a corresponding object"), and the
  message is also part of the worker's observation. Workers are trained FIRST: "a
  bottom-up 'pretraining' procedure ... Although the manager is not trained during this
  period, it still acts ... and sets subgoals" — i.e. during pretraining the orders come
  from an *untrained, near-uniform* manager, so every order gets practised ("If the set of
  possible subgoals is sufficiently compact"). Second trick: a "communication-repeat
  heuristic" — orders are held for 8 steps of a 25-step episode — and this "is however
  more critical" than pretraining. Rewards are not fully hidden: workers keep their own
  collision costs. Vocabulary sizes: 3 to 12 landmark targets; toy particle worlds only.
- **RODE (Wang et al., arXiv:2010.01523) — PART.** The "order" is a role = a restricted
  action subset. Actions are clustered by learned *effect* (k-means on action
  representations predicting next observation and reward) into role action spaces;
  "a role selector assigns a role to each agent every c timesteps"; role policies see
  action representations, which allowed transfer "to tasks with different numbers of
  actions and agents, including tasks with three times as many agents". Best on 10 of 14
  SMAC maps. Relevant to Breach as "an order can be a mask over verbs".
- **Honor of Kings HMS (Wu et al., arXiv:1812.07887) — READ (full, short paper).** Macro
  model outputs an attention map over 144 map regions ("where to go") plus a phase head
  (which of 14 major objectives); its output "directly conveys macro strategy embedding to
  micro level models". *All supervised* from human replays, and the label for "where to
  go" is relabelled from behaviour: "we define ground-truth regions as the regions where
  players conduct their next attack". Allies' predicted macro targets are fed to each other
  ("Imitated Cross-agents Communication": allies' *labels* in training, their
  *predictions* at test). Ablations (40 games each): full HMS beat "AI Without Macro
  Strategy" 75 %–25 % (that agent "mainly focused on nearby targets"), without
  communication 62.5 %, without phase layer 65 %. Vs top-1 % human teams 48 %.
- **Selmonaj et al., air combat (arXiv:2309.11247, 2023; arXiv:2505.08995, 2025) — READ
  (full text of both).** Low level trained first, then *frozen*: "low-level policies are
  fixed ... and serve as options for the commander". The order is a *policy switch plus a
  target index*: "ac ∈ {0,1,2,3}, where 0 activates [escape] and [fight] otherwise ... which
  of the three observable opponents the agent should attack" (2025: two opponents). The
  2025 version describes the low level as trained "with a command flag that identifies the
  scenario target". Separate networks per aircraft type sharing one layer; one commander
  for all types. 2023 result: the commander helps small teams, larger teams drift to draws.
  **The obedience failure, stated by the authors (2025, Future Work):** "fleeing maneuvers
  occur even under the fight policy ... tactical behavior should exclusively be provided by
  the commander agent", attributed to too little training in "highly asymmetric
  scenarios"; they want richer commands ("aggressive attacking", "defending the position",
  choice of weapon). (I did not find the phrase "obedience vs local optimisation" in
  either paper; the July digest's wording is a paraphrase of this passage.)

### A5. The best-measured obedience trade-off: Tencent's Meta-Command Communication (MCC, Gao et al., ICLR 2023, arXiv:2304.11632) — READ (full PDF text)

Not on the lead's list, but the most direct answer to "how do you reward obedience
without killing initiative" in a team combat game with a later human commander.
- **The order format:** a meta-command is a tuple <L, E, T_mc>: L = where to go (one of
  144 map grid cells), E = the event to do there, T_mc = a time limit. "Come And Kill The
  Dragon" = (dragon location, attack dragon, until killed). They deliberately do *not*
  write a reward per event: "customizing corresponding rewards for each specific event ...
  is not conducive to generalization and is even impossible. Instead, we train a
  micro-action network to learn to do optimal event E at location L". T_mc is fixed at
  20 s ("20 seconds corresponds to an 80% completion rate" for humans).
- **Where the training orders came from:** a Command Encoder Network trained by supervised
  learning on human replays, labels made by *hindsight*: "extract the location L_t and
  the event E_t from the state s_{t+T_mc}" — i.e. "where this player actually was 20 s
  later". The executor is conditioned on "the meta-command sampled from the Top-k Softmax
  distribution of CEN" — orders that are plausible, not uniform-random.
- **How obedience was rewarded:** the executor (MCCAN) is the existing strong agent
  (WuKong) *fine-tuned* with the command as an extra input, by self-play RL with
  extrinsic (game) reward plus an intrinsic potential-style reward: the decrease in
  distance from the agent to L, weighted by alpha.
- **The measured trade-off (Appendix, Fig. 16):** "As alpha increases, the completion rate
  ... gradually increases, and the win rate ... first increases and then decreases
  rapidly. ... we can train an agent with a completion rate close to 100% by increasing
  alpha, but this will significantly reduce the win rate because the meta-command executed
  is not necessarily optimal and may result in the death of agents." Chosen alpha = 16:
  completion 82 % (humans 80 %), win rate near 50 % against the state-of-the-art agents.
  Note: obedience *helped* win rate at small alpha.
- **Who decides whether to obey:** a separate Meta-Command Selector, trained last by RL
  with the executor *frozen*, picks which received command (its own or a human's) to
  execute; baselines "MC-Rand" (random choice) and "MC-Rule" (nearest) were worse, and the
  authors call random obedience "nonintelligent and can even be a disaster".
- **Training order:** CEN (SL) -> executor (goal-conditioned RL) -> selector (RL), each
  stage frozen for the next — the "low level first" order the Breach author describes.

### A6. LLM or planner above trained low-level policies (2024-2026) — mostly ABS

- **L2M2 (IJCAI 2025, proceedings page read, PART).** An LLM (zero-shot, not trained)
  emits "a joint plan of discrete movement subtasks, one per RL agent, at a lower frequency
  than the RL agents act"; a translation module writes each subtask *into the RL agent's
  observation*; agent reward = environment reward + a subtask-related reward. Two modes:
  trained end-to-end under the LLM, or "direct integration with pre-trained MARL models".
  VMAS and MOSMAC (StarCraft siege-tank navigation). Pattern = goal-in-observation plus
  goal reward, i.e. the same interface as FMH and MCC, with a language model as the
  commander.
- **HIMA (arXiv:2508.06042), StarEvolve (arXiv:2508.10428), Adaptive Command
  (arXiv:2508.16580) — ABS (search snippets only).** LLM planners over imitation-learned
  executors, an LLM executor, and behaviour trees respectively. None found with trained RL
  soldiers in a shooter. I found no shooter-genre example (NF).
- **Pattern across A1-A6:** every system that worked gave the low level a *small, typed,
  concrete* order (a location among 144 cells; a target index among 2-3; fight / escape; a
  landmark among 3-12; a build order), held for a fixed duration or until a termination
  event, and visible to the low level as an input.

## B. Many unit classes in one architecture

### B1. OpenAI Five: one network, 17 heroes (arXiv:1912.06680) — READ

- One LSTM per hero slot, *same weights* for all five; identity enters as an embedding:
  "In order to tell each LSTM which of the team's heroes it controls, we append the
  controlled hero's Unit Embedding ... to the Game State vector", plus a "cross-hero
  pool" (max-pool over the first 25 % of each replica's vector) so replicas can read each
  other.
- Different abilities per hero handled by *action filters* (masks): "we wrote simple
  action filters, which determine whether the action is available ... At each timestep we
  restrict the set of available actions using these filters and present the final choices
  to the model"; the target head uses "a learned per-action mask".
- Pool size cost: "Training with 80 heroes has a speedup factor of approximately 0.8,
  meaning early training runs 20% slower than with the base 17 heroes" (early training
  only; "it could be that the speedup factor becomes worse later"). Random line-ups were
  kept partly as "additional exploration encouragement".
- Caution: Dota heroes share one verb grammar (move, attack, cast spell slot k, use item);
  they differ in statistics and spells, not in body plan.

### B2. AlphaStar: one network per race — READ

- "We train one policy for each race, with the same architecture" (the supervised
  policies; the league has main agents and exploiters per race). Unit types enter through
  an entity encoder (self-attention over a list of units, "Unit type" a per-entity
  feature in Extended Data Table 1). Races differ in their whole unit roster and economy;
  the split is by *who the player is* (the race), not by unit — every unit of a race is
  driven by that race's one network through pointer/attention heads over the unit list.
  No ablation of one-network-for-all-races found in the text.

### B3. Honor of Kings, 40 heroes in one network (Ye et al., arXiv:2011.12692, NeurIPS 2020) — PART

- "a unified actor-critic network architecture to capture the playing mechanics and
  actions of different heroes", with an action mask. The problem was *training*, not
  representation: naive self-play on random line-ups of 40 heroes "results in unacceptably
  slow training, which fails to converge, with an Elo score 400 lower than the upper limit
  even being trained for 480 hours". Their fix, curriculum self-play learning: Phase 1
  train *fixed line-ups* (four 10-hero groups) as separate teacher models; Phase 2
  "multi-teacher policy distillation ... merged into a single student model" (supervised);
  Phase 3 continued RL with random line-ups. Converged in ~336 h. "The student's
  converged Elo is marginally below the teacher."
- **For Breach (opinion):** this is the most transferable recipe for "many classes": train
  specialists on few classes, distil into one class-conditioned network, then continue.

### B4. Parameter-sharing evidence

- **Terry et al., "Revisiting Parameter Sharing" (arXiv:2005.13625) — PART.** Proves that
  one shared network can represent each agent's optimal policy if the agent's identity is
  in the observation ("agent indication") and if action spaces are *padded* to the largest
  one ("padding the vector with zeros for agents with smaller action spaces"); notes this
  was done before only for SMAC's medivac. Empirically "Prospector, an environment with
  heterogeneous action, observation and agents, was able to learn with parameter sharing
  using agent indication and the two padding methods". How identity is written in an
  image observation mattered and was environment-specific ("highly sensitive to agent
  indication method").
- **Christianos et al., Selective Parameter Sharing SePS (arXiv:2102.07475, ICML 2021) —
  PART.** Naive sharing's "effectiveness is highly dependent on the environment". In a
  task where agents must go to different colours they cannot see, full sharing learns "a
  local minimum" and even "FuPS+id ... still has issues"; on SMAC MMM2 (marines,
  marauders, medivacs) "Even fully shared parameters ... outperforms" no sharing. SePS
  clusters agents by learned embeddings of their *transitions and rewards* (pretraining +
  k-means) and shares one network per cluster; "the clustering ... matched the various
  types of agents" (unit types in SMAC); overestimating the cluster count "is of little
  significance", underestimating hurts.
- **HARL (Zhong et al., JMLR 2024, arXiv:2304.09870) — PART.** Argues parameter sharing
  "confines [methods] to only homogeneous-agent setting"; proposes sequential per-agent
  trust-region updates (HAPPO, HATRPO, ...). Its ablation: parameter sharing "causes
  HAPPO to converge to suboptimal policies. The suboptimality is more severe in the task
  with more diverse agents".
- **UPDeT (Hu et al., arXiv:2101.08001, ICLR 2021) — PART.** Transformer over observation
  *entities*; each action group is read off the matching entity's output token
  ("observation-entity -- action-group pairs"), so an attack-unit-k action is scored from
  unit k's token. "no need to introduce any new parameters for new tasks"; transfer
  between 3- and 7-marine maps; "roughly 10 times faster" transfer than RNN models.
  Relevant as the template for a variable-size target list ("aim at enemy k").
- **Kaleidoscope (Li, Pan, Zhang, NeurIPS 2024, arXiv:2410.08540) — ABS.** One shared
  parameter set plus learnable per-agent masks over it, with a term pushing masks apart;
  partial sharing between full and none. Benchmarks MPE, MAMuJoCo, SMACv2.
- **Air combat (B5, from A4) — READ.** Selmonaj: "a distinct policy for each aircraft type
  ... Agents of the same type use the same shared policies", the two type networks sharing
  one layer; one commander for all types.

### B6. Synthesis on "one network + class input" vs "one network per class"

- Evidence (READ) supports one shared network *within* a class family that shares a
  verb grammar (OpenAI Five, HoK 40 heroes, SMAC MMM2), with identity as an embedded
  input and per-class *masks* for verbs a unit lacks. Representation is not the
  bottleneck; *training* is (HoK needed specialist teachers + distillation).
- Evidence (READ) against *naive* full sharing when roles truly differ (SePS colour
  tasks, HARL's more-diverse tasks), and practice (AlphaStar per race, Selmonaj per
  aircraft type) splits networks where the body and verb set differ wholesale.
- I found no study comparing the two for agents whose *action sets* differ as much as a
  biting zombie, a swimming octopus and a marine (NF). Terry's padding proof says one
  network *can* represent it; nothing measured says it *learns* it well.

## C. Tsetlin machines as game-agent learners

- **What a Tsetlin machine (TM) is (READ, Rahimi Gorji & Granmo 2023 intro; RECALL for
  detail):** a learner made of many small "learning automata" that each decide whether to
  include one Boolean input (or its negation) in an AND-clause; clauses vote with + / -
  polarity and the sum is thresholded. "A TM takes a feature vector of propositional
  values as input". Output: rules a human can read ("x3 AND NOT x7 -> vote +1").
- **RL with a TM — the first framework (Rahimi Gorji & Granmo, "Off-policy and on-policy
  reinforcement learning with the Tsetlin machine", Applied Intelligence 53, 2023) —
  READ (abstract and method sections of the open-access PDF).** A *regression* TM is the
  state-value function inside value iteration / multi-step TD (SARSA). Problem size:
  gridworlds of 3x6, 5x7 and 10x10 with internal walls. State encoding: "we encode these
  coordinates to an (m + n)-bit vector using one-hot" (row one-hot + column one-hot). The
  regression TM needs a known output range [Ymin, Ymax] and discretizes the target
  ("discretization gaps of 0.5, 0.25 and 0.1"). Findings, in the abstract: accurate
  off-policy, but "learns significantly slower than neural networks on-policy"; with
  multi-step TD "we are able to close the performance gap" and "can outperform comparable
  neural network models" on those gridworlds. They also turn the learned clauses into a
  graph of the policy.
- **Other game uses (ABS, search snippets only):** Hex winner *prediction* on 6x6 boards
  (Giri & Granmo 2022, arXiv:2203.04378; reported ~92 % accuracy) — supervised, not play;
  a 2020 master's thesis pairing a TM with Monte-Carlo tree search for 6x6 Hex (TM alone
  could not play well); a 2023 Agder master's thesis on model-free TM RL that solved
  CartPole and scored "several points" in Pong without solving it; the Graph Tsetlin
  Machine (Granmo et al., arXiv:2507.14874, 2025) lists "contextual bandit learning" as a
  supported mode and reports up to +20.6 points over "other reinforcement learning
  methods" on an "action coreference" tracking task (not checked in the full text).
- **Not found (NF):** any TM agent in a real-time multi-agent combat game, any TM trained
  by self-play, any TM policy at the scale of a Breach soldier (hundreds of field-plane
  inputs, a dozen verbs). The largest RL problems shown are toy gridworlds and CartPole.
- **Input form a TM needs:** Boolean literals only. Continuous values must be
  *booleanized* — in TM practice usually by thresholds ("x > t_k" for several t_k, a
  "thermometer" code; RECALL), positions by one-hot (READ, above). A shared observation
  interface for neural nets and TMs is therefore realistic only if the *semantic* features
  are named yes / no or thresholded facts (enemy visible, in cover, ammo < 1 magazine,
  ordered target within 3 m). Breach's bot already judges tiles "ONLY through named yes /
  no features" with a note that "a learned scorer (Tsetlin, v2) reads the same list"
  (project CLAUDE.md, Position features row) — that list is exactly the right shape.
- **Opinion:** a TM is plausible as a *scorer* over a small Boolean feature vector (the
  bot's position features, a "which order to accept" selector, an imitation of the
  layered bot), not as a replacement for a pixel-crop RL policy on 2026 evidence.

## For Breach

### The author's question: reserve an order input now, or not? (opinion, argued from the evidence above)

**Yes — reserve it now, in a concrete typed form, train it from the start with orders
that come from a generator and from hindsight, and pay a small, probabilistic obedience
reward.** Reasons:
1. Every system that shipped an obedient low level had the order as an input *while the
   low level learned* (AlphaStar z from supervised learning on; HMS and MCC executors;
   FMH workers; Selmonaj "command flag"; XLand goals). None added obedience later to a
   finished soldier. MCC is the nearest to "later": it fine-tuned an existing strong agent
   with a new command input and still needed a full self-play RL stage to make it obey.
2. Adding the input later by surgery is cheap mechanically (OpenAI Five's zero-column
   W_new = [W 0]) but their own evidence is that the surgered model "plateaued at a weaker
   skill level" than one trained on the final interface, and inputs cannot be removed,
   only held constant. Reserving the slot costs little: an order field that is "none"
   most of the time behaves like AlphaStar's z set to zero.
3. An order the soldier has never been trained with is just noise to it; a commander
   trained over soldiers that ignore orders learns nothing (FMH's non-stationarity
   argument for pretraining workers under a near-uniform manager).

**In what form (opinion).** A small tuple, MCC-style: `<kind, argument, duration>`, where
kind is from a short vocabulary, argument is a *tile* (or a cell of a coarse grid) or an
*entity index* into the soldier's existing target list (UPDeT/AlphaStar pointer style),
and duration is a fixed hold (MCC 20 s; FMH 8 of 25 steps) or a termination event. A
first vocabulary that each reward can measure without a human:
`none`, `move_to(tile)`, `hold(tile, radius)`, `attack(enemy k)` / `focus(enemy k)`,
`fall_back(tile)`. The July list (stack-on, breach, suppress, throw, set-charge,
follow-me) can be grown into, kind by kind, each with its own hindsight labeller. Keep
`none` as a first-class order so the soldier stays a full agent without a commander.

**Trained how (opinion).** Three sources of training orders, mixed:
- *Random / generated* orders from a generator over the map (FMH's uniform manager,
  OpenAI Five's random lane assignments, XLand's procedural goals), held for their
  duration — so every kind gets practised.
- *Hindsight* orders: label a stretch of the soldier's own trajectory with the order it
  happened to fulfil (where it was 10-20 s later, whom it shot) — AlphaStar's z in SL,
  HMS's "region of the next attack", MCC's CEN labels, HER. This trains obedience without
  new rollouts.
- *The scripted team layer as commander:* Breach already has one — the layered bot's
  TEAM layer picks a play, a shared focus target and an objective every
  `team_period_ticks` (project CLAUDE.md, Bot layers row). Its play and target, mapped to
  the order tuple, are realistic orders from day one, and the RL soldier can then be
  dropped under the same team layer that drives the bots (opinion; not tried anywhere I
  read, but it is the FMH "pretrain under a manager that acts" setup with a better
  manager).
Reward: game reward + a shaped obedience term (MCC: decrease in distance to L; for
`attack(k)`: damage dealt to k), **active only in some episodes or with a tunable weight**
(AlphaStar: each pseudo-reward "active ... with probability 25%", with separate value
heads; MCC: weight alpha chosen where completion ~= human level and win rate holds).
MCC's curve is the warning: pushing obedience toward 100 % "will significantly reduce the
win rate ... may result in the death of agents". So the obedience weight is a dial to
sweep, and episodes with `none` keep initiative alive. For "obedience-high" feel, aim
for MCC's regime (~80 % completion) rather than 100 %, and let a selector — a commander
or the soldier itself — decide when an order is suicidal (MCC's Meta-Command Selector).

### Unit classes in the observation and the action mask (opinion)

- **Observation:** a per-unit *class id embedding* plus the class's numeric stats (mass,
  speed, armour, weapon row values) as features, for the unit itself and for every
  entity in its target list — OpenAI Five's hero embedding and AlphaStar's per-entity
  unit type. Stats as features let a new class with familiar numbers generalise; the id
  embedding lets the network learn what the numbers miss.
- **Action space:** one *union* action space over all verbs (Terry's padding), with a
  per-class, per-tick legality mask (OpenAI Five's action filters, HoK's action mask) —
  a zombie's bite and a marine's melee can be the same column if the game treats them
  alike; an octopus's swim is a column every other class has masked off. Breach's action
  rows are already GENERATED from `INTENT_KINDS` (`control_policy.action_schema`), so the
  union and the per-class mask can come from the same registry (opinion).
- **Network split:** one shared network per *body plan / verb grammar family* (marines
  of all armours and weapons in one; zombies-like melee animals perhaps in another),
  with class embedding inside the family; a separate network where the verb grammar
  differs wholesale (AlphaStar per race; Selmonaj per aircraft type, sharing a layer).
  When the family count grows, the Honor of Kings recipe — specialists first, then
  distil into one class-conditioned student, then continue RL — is the measured way to
  merge them. SePS-style clustering (by behaviour) is the automatic version.
- **Orders across classes:** keep the order tuple class-agnostic (a tile, an entity, a
  kind); a class that cannot do a kind gets it masked at the commander, not
  re-interpreted at the unit (opinion; Selmonaj's single commander over two aircraft
  types worked because both low levels accepted the same command set).

### What to avoid (opinion, with the evidence it rests on)

- A large or abstract order vocabulary before a small concrete one works (FMH: results
  hinge on "an adequate set of subgoals"; every working system used 2 to ~144 concrete
  targets).
- Orders that change every tick: FMH found the hold ("communication-repeat") "more
  critical" than pretraining.
- 100 % obedience by reward weight (MCC's win-rate collapse); and its opposite, an order
  with no reward at all (Selmonaj's fight policy still flees — the low level does what
  its own reward says when the order and the reward disagree).
- Learned latent orders (FuN-style) as the first rung — not interpretable for a human
  commander, and a human player is one of the planned commanders.
- Naive full parameter sharing across classes with genuinely different roles (SePS,
  HARL) — give identity as input at the least.
- Removing inputs later — surgery can only add; plan the slot list now.

### What remains open

- No source tested a shooter-genre soldier with orders; MOBA and air combat are the
  closest. The obedience weight, the order hold time and the vocabulary are Breach
  experiments.
- Whether one network can learn classes with truly different verb sets (bite, swim,
  shoot) well: represented in principle (Terry), not measured anywhere I found.
- Whether hindsight-labelled orders produce real obedience in a *combat* game where the
  hindsight label is often "where I died" — needs a filter (e.g. label only from stretches
  the unit survived; MCC labelled from human play, not from failing agents).
- Tsetlin machines: no evidence at Breach's scale; a Boolean-feature view is cheap to
  provide and is the only fair test.

## Source ledger

| Citation | URL | Status | Supports |
|---|---|---|---|
| Vinyals et al. 2019, AlphaStar, Nature 575 (DeepMind unformatted preprint) | https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf | FULL | z = build order + cumulative stats; z zeroed 10 % in SL; pseudo-rewards active 25 %; Fig. 3E Elo; z fixed for validation agents; one policy per race |
| Berner et al. 2019, OpenAI Five, arXiv:1912.06680 | https://arxiv.org/abs/1912.06680 | FULL | hero embedding + cross-hero pool; action filters; lane assignments + ablation; surgery (zero columns, >20 surgeries, plateau, Rerun 98 %); 80-hero pool 0.8x speed |
| Open-Ended Learning Team 2021, XLand, arXiv:2107.12808 | https://arxiv.org/abs/2107.12808 | PART | predicate goals; GOAT attention; ablation −20 % / −30 % / −65 %; fine-tuning 77 % vs 39 % vs 9 % |
| SIMA Team 2024, arXiv:2404.10179 | https://arxiv.org/abs/2404.10179 | PART (via WebFetch quotes) | keyboard+mouse, BC, post-hoc instruction labels, setter-solver, no-language 11 % vs 34 % |
| SIMA Team 2025, SIMA 2, arXiv:2512.04797 | https://arxiv.org/abs/2512.04797 | ABS | Gemini-based, self-improvement via task/reward proposals |
| Ahilan & Dayan 2019, FMH, arXiv:1901.08492 | https://arxiv.org/abs/1901.08492 | FULL | message defines worker reward; worker pretraining under untrained manager; comm-repeat more critical |
| Wang et al. 2021, RODE, arXiv:2010.01523 | https://arxiv.org/abs/2010.01523 | PART | roles = action subsets by effect clustering; role every c steps; transfer to 3x agents |
| Wu et al. 2018, HMS Honor of Kings, arXiv:1812.07887 | https://arxiv.org/abs/1812.07887 | FULL | 144-region "where to go" labelled from next attack; ablations 75 % / 62.5 % / 65 %; 48 % vs top humans |
| Selmonaj et al. 2023, arXiv:2309.11247 | https://arxiv.org/abs/2309.11247 | FULL | low level first then frozen; commander action = escape or attack opponent k; per-type networks |
| Selmonaj et al. 2025, arXiv:2505.08995 | https://arxiv.org/abs/2505.08995 | PART (method, conclusion, future work) | "command flag"; fight policy still flees (obedience failure); wish for richer commands |
| Gao et al. 2023, MCC, ICLR, arXiv:2304.11632 | https://arxiv.org/abs/2304.11632 | FULL | meta-command <L,E,T>; hindsight labels; intrinsic distance reward; alpha trade-off curve (82 % completion, ~50 % WR; ~100 % completion kills WR); selector |
| L2M2, IJCAI 2025 | https://www.ijcai.org/proceedings/2025/12 | PART (proceedings page) | LLM subtasks into RL observation + subtask reward; pre-trained or end-to-end |
| HIMA arXiv:2508.06042; StarEvolve arXiv:2508.10428; Adaptive Command arXiv:2508.16580 | arxiv.org/abs/... | ABS | LLM planners over IL executors / LLM executor / behaviour trees |
| Ye et al. 2020, Full MOBA with deep RL, arXiv:2011.12692 | https://arxiv.org/abs/2011.12692 | PART | one unified network for 40 heroes; baseline fails (−400 Elo at 480 h); fixed-lineup teachers + distillation converge ~336 h |
| Terry et al. 2020, Revisiting Parameter Sharing, arXiv:2005.13625 | https://arxiv.org/abs/2005.13625 | PART | agent indication + action/obs padding: representational proof; Prospector learns |
| Christianos et al. 2021, SePS, arXiv:2102.07475 | https://arxiv.org/abs/2102.07475 | PART | naive sharing env-dependent; FuPS+id fails colour task; FuPS beats NoPS on MMM2; clusters match unit types |
| Zhong et al. 2024, HARL, JMLR, arXiv:2304.09870 | https://arxiv.org/abs/2304.09870 | PART | sequential updates; parameter sharing suboptimal, worse with more diverse agents |
| Hu et al. 2021, UPDeT, arXiv:2101.08001 | https://arxiv.org/abs/2101.08001 | PART | entity tokens matched to action groups; no new params for new tasks; ~10x faster transfer |
| Li, Pan, Zhang 2024, Kaleidoscope, NeurIPS, arXiv:2410.08540 | https://arxiv.org/abs/2410.08540 | ABS | learnable masks over shared parameters |
| Rahimi Gorji & Granmo 2023, Off-/on-policy RL with the TM, Applied Intelligence 53:8596 | https://link.springer.com/content/pdf/10.1007/s10489-022-04297-3.pdf | PART | first TM RL framework; gridworlds 3x6, 5x7, 10x10; one-hot booleanization; slower on-policy, gap closed with multi-step TD |
| Giri & Granmo 2022, Hex winner prediction, arXiv:2203.04378 | https://arxiv.org/abs/2203.04378 | ABS | supervised TM on 6x6 Hex, ~92 % |
| Granmo et al. 2025, Graph Tsetlin Machine, arXiv:2507.14874 | https://arxiv.org/abs/2507.14874 | ABS | contextual-bandit mode; +20.6 points on action coreference |
| UiA master's thesis 2023, "Potential and Limitations of the TM in Model-Free RL" | https://uia.brage.unit.no/uia-xmlui/handle/11250/3077205 | ABS | CartPole solved, Pong not |
| Schaul et al. 2015 UVFA; Andrychowicz et al. 2017 HER arXiv:1707.01495 | (July digest) | RECALL | goal as input; hindsight relabelling |
| Shooter-genre RL soldier taking orders | — | NF | — |
| Study of one net vs per-class nets with different verb sets | — | NF | — |
