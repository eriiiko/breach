# Single-Agent Deep RL, Curriculum, Imitation Bootstrap, Single→Multi Transfer — Digest for Breach

*Raw research-agent output, 2026-07-20. Organized to answer: (1) which core algorithm fits a fast deterministic discrete-action sim; (2) how to represent "move-to-tile × order × target"; (3) reward shaping without corrupting the objective; (4) bootstrapping from Erik's scenario-painter + order-recorder demos; (5) level curriculum; (6) goal-conditioned "fulfill the order"; (7) one unit → shared-weight team.*

Deterministic-sim advantage runs through everything: cheap, parallelizable, forkable, bit-reproducible → shifts the calculus toward **on-policy PPO** and toward **planning/model-based (MuZero-family / true-model AlphaZero)** that most projects can't afford.

## 1. Core single-agent algorithms
**Value-based (discrete — Breach's native shape):**
- **DQN** — Mnih et al. 2015, Nature 518, doi:10.1038/nature14236 (precursor arXiv:1312.5602). CNN → per-action Q; replay buffer + target network. *Breach:* archetype Breach's stacked-plane observation already mirrors.
- **Double DQN** — van Hasselt et al. 2016, arXiv:1509.06461. Cuts overestimation (worse in large action spaces). 
- **Dueling** — Wang et al. 2016, arXiv:1511.06581. V(s) + advantage; shares state value across near-equivalent actions (200 destination tiles).
- **Prioritized Experience Replay** — Schaul et al. 2016, arXiv:1511.05952. Replays high-TD-error transitions; the scoring mechanism reused by PLR. *Breach:* natural bucket for injecting demos (DQfD).
- **Rainbow** — Hessel et al. 2018, arXiv:1710.02298. Six DQN extensions combined; batteries-included discrete baseline. See Beyond the Rainbow (2024, arXiv:2411.03820) for desktop-scale.
**Recurrent (partial observability):**
- **R2D2** — Kapturowski et al. 2019, ICLR (openreview r1lyTjAqYX). LSTM + stored hidden states + burn-in. *Breach:* if marines have fog-of-war/LoS; reference technique for recurrent agents from replay.
- **Agent57** — Badia et al. 2020, arXiv:2003.13350. Adaptive exploration family; overkill first, matters for hard sparse-reward objectives.
**Policy-gradient / actor-critic (workhorses):**
- **PPO** — Schulman et al. 2017, arXiv:1707.06347. Clipped surrogate; robust; native discrete/factored heads + masking. *Breach:* **very likely first algorithm** — on-policy, bottlenecked by env throughput, which Breach's GPU sim delivers.
- **SAC** — Haarnoja et al. 2018, arXiv:1801.01290 (+1812.05905); SAC-Discrete arXiv:1910.07207. Off-policy max-entropy; sample-efficient. *Breach:* sample efficiency matters less when sim is free — keep as fallback.
- **IMPALA + V-trace** — Espeholt et al. 2018, arXiv:1802.01561. Decoupled actors/learner; V-trace corrects lag; ~250k FPS. *Breach:* scaling pattern for thousands of parallel instances.
- **Sample Factory (APPO)** — Petrenko et al. 2020, arXiv:2006.11751; github.com/alex-petrenko/sample-factory. Single-machine async PPO >10^5 FPS. *Breach:* strong candidate harness matching Erik's dev setup.
**Model-based / planning (available BECAUSE Breach can fork state):**
- **MuZero** — Schrittwieser et al. 2020, Nature, arXiv:1911.08265. Learns latent dynamics + MCTS without the rules. *Breach:* Breach already HAS a perfect forkable deterministic simulator → can run AlphaZero-style search with the TRUE model, sidestepping MuZero's hardest part. Serious medium-term "commander" option. See EfficientZero-V2 (2024, arXiv:2403.00564).
- **Muesli** — Hessel et al. 2021, arXiv:2104.06159. MuZero-level with one-step look-ahead. *Breach:* lighter PPO↔MCTS bridge.
- **DreamerV3** — Hafner et al. 2023, arXiv:2301.04104. World-model agent, fixed hyperparameters across 150+ tasks. *Breach:* robustness attractive, but cheap real simulator undercuts the "imagined simulator" motivation — a benchmark, not the default.

## 2. Action spaces: "move-to-tile × order × target"
- **Invalid Action Masking in Policy Gradients** — Huang & Ontañón 2022, arXiv:2006.14171. Masking invalid actions (logits → −∞) keeps the policy gradient valid; essential as action spaces grow. *Breach:* `get_legal_actions()` IS an action mask — adopt from day one.
- **AlphaStar autoregressive action head** — Vinyals et al. 2019, doi:10.1038/s41586-019-1724-z. Decompose a combinatorial action into a sequence of conditional decisions (action-type → target → modifier), masked at each step; pointer network over units. *Breach:* **direct template** — build two autoregressive sub-heads (order-type, then parameter conditioned on order-type), masking illegal parameters.
- **P-DQN / MP-DQN** — Xiong et al. 2018 arXiv:1810.06394; Bester et al. 2019 arXiv:1905.04388. Hybrid discrete-continuous actions. *Breach:* only if orders carry continuous parameters (throw angle). If everything stays on the integer tile grid (likely, for determinism), the fully-discrete autoregressive head is better; P-DQN is the escape hatch.
- **Wolpertinger (large discrete)** — Dulac-Arnold et al. 2015, arXiv:1512.07679. Action embedding + nearest-neighbor. *Breach:* fallback if the flat set becomes enormous; masking+factorization likely make it unnecessary.
- **Action Space Shaping** — Kanervisto et al. 2020, arXiv:2004.00980. Discretizing/removing/combining actions dramatically affects whether an agent learns at all. *Breach:* practical guide for designing the action set.

## 3. Reward shaping
- **Policy Invariance / Potential-Based Shaping** — Ng, Harada, Russell 1999, ICML. F(s,s′)=γΦ(s′)−Φ(s) provably leaves the optimal policy unchanged; any other shaping can silently change the objective. *Breach:* **the single most important guardrail** — express shaping (closer-to-target, in-cover) as a potential difference so it can't teach degenerate behavior; protects the emergent-strategy goal from reward hacking.
- **Reward is Enough** — Silver et al. 2021, AIJ 299, doi:10.1016/j.artint.2021.103535. A single well-chosen scalar reward can drive rich emergent behavior — Breach's implicit bet.
- **Improving Potential-Based Shaping** 2025 arXiv:2502.01307; **Concrete Problems in AI Safety** (reward hacking) Amodei et al. 2016 arXiv:1606.06565.

## 4. Imitation / bootstrap (Erik's scenario-painter + order-recorder plan)
- **Behavioral Cloning (ALVINN)** — Pomerleau 1988/1991, doi:10.1162/neco.1991.3.1.88. Supervised π(a|s) from demo pairs, no env interaction. *Breach:* literally step one — recorded (state, action) pairs train an initial policy by classification.
- **DAgger** — Ross, Gordon, Bagnell 2011, arXiv:1011.0686. Fixes BC's compounding covariate shift: roll out current policy, expert labels visited states, aggregate, retrain. *Breach:* cheaply implementable — re-open any scenario, let the half-trained marine act, have Erik provide the correct order for states it drifted into.
- **DQfD** — Hester et al. 2018, arXiv:1704.03732. Pre-fill replay with demos + TD loss + large-margin supervised loss; continue with self-data. *Breach:* cleanest "demos → RL" bridge if value-based; seed the buffer, keep demos mixed in via prioritized replay. PPO-side equivalent: **POfD** (Kang et al. 2018, arXiv:1802.05313 *verify*).
- **GAIL** — Ho & Ermon 2016, arXiv:1606.03476. Occupancy matching; imitate without designing a reward. *Breach:* when Erik can demonstrate but not write the reward; adversarial instability caveat → second-phase tool.
- **Cal-QL (offline→online)** — Nakamoto et al. 2023, arXiv:2303.05479; github.com/nakamotoo/Cal-QL. Calibrated conservative value fn → online fine-tuning improves monotonically, no crash. *Breach:* principled version of the two-phase plan; the fix if BC→PPO shows the classic dip (also CQL arXiv:2006.04779).

## 5. Curriculum learning
- **Curriculum Learning for RL: Survey** — Narvekar et al. 2020, JMLR 21(181). Taxonomy of task generation/sequencing/transfer. *Breach:* map for scenario progression.
- **Teacher-Student Curriculum Learning** — Matiisen et al. 2017/2019, arXiv:1707.00183. Teacher picks the sub-task where the student's learning progress is fastest. *Breach:* automatic-curriculum controller over Erik's scenario set.
- **Prioritized Level Replay (PLR)** — Jiang et al. 2021, arXiv:2010.03934. Samples the next LEVEL by learning potential (TD error / value loss). *Breach:* generate/vary ship-interior levels; PLR decides which to revisit → better zero-shot generalization. Reuses PER scoring.
- **PAIRED / Unsupervised Environment Design** — Dennis et al. 2020, arXiv:2012.02096; robust UED/ACCEL arXiv:2308.10797. Adversary generates regret-maximizing "hard but solvable" levels. *Breach:* advanced end-state auto-level-designer; roadmap item.
- **Domain randomization** — Tobin et al. 2017, arXiv:1703.06907. Randomize layouts/enemy placement/cover. *Breach:* prevents overfitting to a handful of hand-painted maps.

## 6. Goal-conditioned single agent ("achieve the order it was given")
An order is a goal g; the marine is π(a|s,g). Kill-X, reach-Y, hold are all goals in one parametric policy.
- **UVFA** — Schaul et al. 2015, PMLR v37. V(s,g)/Q(s,a,g). *Breach:* formal backbone of "orders as goals" — feed the order (target ID/destination/objective type) as goal input alongside [C,H,W] planes.
- **HER** — Andrychowicz et al. 2017, arXiv:1707.01495. Relabel a failed trajectory as a success for the goal actually achieved. *Breach:* orders are sparse-reward by nature; HER densifies signal without violating Ng's theorem (relabeling changes the goal, not the reward function).
- **GCSL** — Ghosh et al. 2019/2021, arXiv:1912.06088; github.com/dibyaghosh/gcsl. Goal-conditioned supervised learning: relabel each trajectory by its achieved goal, imitate, iterate; optimizes a bound on the RL objective with only supervised updates. *Breach:* **unification of Erik's two ideas** (imitation bootstrap + goal-conditioning) — recorded (state, action, order) demos plug straight in; strong candidate for the very first loop (stable, simple).
- **Decision Transformer** — Chen et al. 2021, arXiv:2106.01345; github.com/kzl/decision-transformer. Return-conditioned sequence modeling; supervised on offline data. *Breach:* offline goal/return-conditioned alternative; naturally handles partial observability; heavier — offline-RL comparison arm.

*Design note:* UVFA (representation) + HER (relabeling) + GCSL/goal-conditioned BC (bootstrap loss) form a coherent stack — one goal-conditioned network, seeded by order recordings, densified by hindsight, fine-tuned by PPO. Technical heart of "train one unit to fulfill any order."

## 7. Single→multi transfer & scaling ("train one, deploy many")
Key fact: if marines are homogeneous, a single goal-conditioned network can be **shared across all of them** — train one, instantiate many, each fed its own local observation + order.
- **Parameter sharing (Cooperative Multi-Agent Control)** — Gupta, Egorov, Kochenderfer 2017, AAMAS. All homogeneous agents share one policy's weights, trained on pooled experience. *Breach:* literal mechanism for "one unit scales into a team."
- **Revisiting Parameter Sharing** — Terry et al. 2020, arXiv:2005.13625. Shared params + an agent ID/index still allow behavioral diversity. *Breach:* add a role/ID channel for pointman vs rear guard from one network.
- **Selective Parameter Sharing** — Christianos et al. 2021, PMLR v139. Learns WHICH agents should share vs specialize. *Breach:* for heterogeneous squads (mixed classes/abilities).
- **SMAC** — Samvelyan et al. 2019, arXiv:1902.04043; github.com/oxwhirl/smac. Standard decentralized squad-micromanagement benchmark; home of QMIX-family (CTDE). *Breach:* closest existing analogue to Breach's squad layer. **MAPPO** (Yu et al. 2022, arXiv:2103.01955) shows shared-parameter PPO is a shockingly strong MARL baseline — reinforces "one PPO brain, many marines."
- **Population Based Training (PBT)** — Jaderberg et al. 2017, arXiv:1711.09846. Population + online weight-copy + hyperparameter perturbation → discovers hyperparameter schedules. *Breach:* tuning method + self-play substrate (used in AlphaStar/OpenAI Five).
- **Emergent Tool Use (hide-and-seek)** — Baker et al. 2019, arXiv:1909.07528. Simple objective + competition → six rounds of emergent strategy, no reward for any of it. *Breach:* **proof-of-concept for Breach's entire thesis** — richness, not reward engineering, is the driver.
- Context: **OpenAI Five** — Berner et al. 2019, arXiv:1912.06680. Shared-parameter PPO + self-play → 5-unit team beating Dota 2 champions; largest datapoint that the recipe scales.

## 8. Sample-efficiency ↔ sim-throughput coupling
- **What Matters in On-Policy RL** — Andrychowicz et al. 2021, arXiv:2006.05990. 250k+ agents isolating which of 50+ PPO choices matter. *Breach:* checklist to get PPO right first time.
- **37 Implementation Details of PPO** — Huang et al. 2022, iclr-blog-track (CleanRL). *Breach:* read before writing the PPO loop; use CleanRL reference impls.
- **EnvPool** — Weng et al. 2022, arXiv:2206.10558. ~1M FPS batched env stepping. *Breach:* pattern for exposing the GPU sim as a batched vectorized env.
- **Isaac Gym** — Makoviychuk et al. 2021, arXiv:2108.10470. Thousands of sim instances ON the GPU; observations/actions resident on-device; revives on-policy PPO. *Breach:* architectural validation of Breach's premise — keep observations + policy forward-pass on the GPU with the physics.

*The coupling, stated plainly:* off-policy methods were invented because steps were expensive. Breach inverts that economy (steps nearly free + reproducible) → **on-policy PPO is throughput-optimal**, and **true-model AlphaZero planning is uniquely affordable**. Off-policy re-enters only if a specific cost dominates (expensive reward computation; heavy reuse of a fixed demo set → DQfD).

## Recommended starting points for Breach (ranked recipe for the FIRST single-unit experiment)
Goal of experiment #1: one marine, goal-conditioned, trained to fulfill a single order type ("reach tile Y" or "kill unit X") on a handful of hand-painted rooms. Prove the loop end-to-end before adding complexity.
1. **Algorithm: PPO** (from CleanRL / Sample Factory, following "37 details" + "What Matters"). On-policy suits the cheap deterministic GPU sim; robust; native discrete + masking.
2. **Action handling: masked, autoregressive, fully discrete head** (AlphaStar-style). Sub-head 1 = order-type; sub-head 2 = parameter conditioned on order-type; apply `get_legal_actions()` mask at every step. Stay on the integer tile grid (no continuous parameters → determinism preserved).
3. **Goal-conditioning: UVFA-style input** — concatenate the order as extra planes / a goal vector; one network learns π(a|s,g).
4. **Reward: sparse terminal for order-fulfillment + ONLY potential-based shaping** (Ng); densify the sparse signal with HER relabeling rather than hand-tuned dense rewards.
5. **Imitation bootstrap: goal-conditioned BC / GCSL first, then PPO fine-tune.** Phase 0 — Erik plays optimal orders through the scenario painter; record (state, order, action); pre-train the same goal-conditioned net by supervised classification. Phase 1 — init PPO from those weights, fine-tune online. If the handoff dips → Cal-QL. If it drifts → a DAgger round.
6. **Curriculum: hand-painted easy→hard, scheduled;** once >handful of levels exist, adopt PLR + domain randomization.
7. **Throughput: GPU-resident vectorized env** (EnvPool pattern; Isaac Gym lesson) — thousands of parallel deterministic instances feeding one PPO learner.

**Scaling to a team (experiment #2+):** keep the SAME goal-conditioned policy, instantiate K times with shared parameters (Gupta 2017), each fed its local observation + own order, with a role/ID channel (Terry 2020) if specialization wanted. Validate against MAPPO/SMAC. Only then consider self-play/autocurricula or true-model AlphaZero planning for a commander.

**Alternative first algorithm (only if a constraint bites):** if reward computation is expensive or Breach leans hard on the fixed demo set → start value-based with **DQfD** (demos in prioritized replay + large-margin loss). Same goal-conditioning + masking apply.

## Key open problems / cautions
- **Determinism vs RL stochasticity:** keep neural-net policy inference (floats, GPU nondeterminism, softmax sampling) OUTSIDE the synced sim path; the action enters the sim as an integer order.
- **Reward hacking / shaping corruption:** Ng-potential-based shaping only; prefer HER over dense rewards; watch loiter-in-cover / kite-forever.
- **BC covariate shift:** budget DAgger-style correction rounds.
- **Offline→online collapse:** Cal-QL/CQL; monitor the transition.
- **Partial observability:** frame-stacks or recurrence (R2D2 stored-state) — decide observation fullness early.
- **Action-space blow-up:** masking + autoregressive factorization; Wolpertinger fallback.
- **PPO's hidden sensitivity:** follow the "37 details" / "What Matters" checklists; use a vetted implementation.
- **Single→multi is not free:** shared parameters give K bodies, but coordinated team behavior may need CTDE (QMIX/MAPPO) or self-play pressure.
- **Sim-throughput is the real bottleneck:** instrument FPS early — a slow CPU↔GPU observation path silently starves the learner.

## Reference list
DQN https://doi.org/10.1038/nature14236 (precursor https://arxiv.org/abs/1312.5602) · Double DQN https://arxiv.org/abs/1509.06461 · Dueling https://arxiv.org/abs/1511.06581 · PER https://arxiv.org/abs/1511.05952 · Rainbow https://arxiv.org/abs/1710.02298 · Beyond the Rainbow https://arxiv.org/abs/2411.03820 · R2D2 https://openreview.net/forum?id=r1lyTjAqYX · Agent57 https://arxiv.org/abs/2003.13350 · PPO https://arxiv.org/abs/1707.06347 · SAC https://arxiv.org/abs/1801.01290 / https://arxiv.org/abs/1812.05905 · SAC-Discrete https://arxiv.org/abs/1910.07207 · IMPALA https://arxiv.org/abs/1802.01561 · Sample Factory https://arxiv.org/abs/2006.11751 · MuZero https://arxiv.org/abs/1911.08265 · EfficientZero-V2 https://arxiv.org/abs/2403.00564 · Muesli https://arxiv.org/abs/2104.06159 · DreamerV3 https://arxiv.org/abs/2301.04104 · Invalid Action Masking https://arxiv.org/abs/2006.14171 · AlphaStar https://doi.org/10.1038/s41586-019-1724-z · MP-DQN https://arxiv.org/abs/1905.04388 · P-DQN https://arxiv.org/abs/1810.06394 · Wolpertinger https://arxiv.org/abs/1512.07679 · Action Space Shaping https://arxiv.org/abs/2004.00980 · Potential-Based Shaping https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf · Reward is Enough https://doi.org/10.1016/j.artint.2021.103535 · Improving PBRS https://arxiv.org/abs/2502.01307 · AI Safety https://arxiv.org/abs/1606.06565 · ALVINN https://doi.org/10.1162/neco.1991.3.1.88 · DAgger https://arxiv.org/abs/1011.0686 · DQfD https://arxiv.org/abs/1704.03732 · POfD https://arxiv.org/abs/1802.05313 · GAIL https://arxiv.org/abs/1606.03476 · CQL https://arxiv.org/abs/2006.04779 · Cal-QL https://arxiv.org/abs/2303.05479 · Curriculum survey https://jmlr.org/papers/v21/20-212.html · TSCL https://arxiv.org/abs/1707.00183 · PLR https://arxiv.org/abs/2010.03934 · PAIRED https://arxiv.org/abs/2012.02096 / robust UED https://arxiv.org/abs/2308.10797 · Domain Randomization https://arxiv.org/abs/1703.06907 · UVFA https://proceedings.mlr.press/v37/schaul15.html · HER https://arxiv.org/abs/1707.01495 · GCSL https://arxiv.org/abs/1912.06088 · Decision Transformer https://arxiv.org/abs/2106.01345 · Parameter Sharing (Gupta) stanford.edu · Revisiting Param Sharing https://arxiv.org/abs/2005.13625 · Selective Param Sharing https://proceedings.mlr.press/v139/christianos21a.html · SMAC https://arxiv.org/abs/1902.04043 · MAPPO https://arxiv.org/abs/2103.01955 · PBT https://arxiv.org/abs/1711.09846 · hide-and-seek https://arxiv.org/abs/1909.07528 · OpenAI Five https://arxiv.org/abs/1912.06680 · What Matters On-Policy https://arxiv.org/abs/2006.05990 · 37 PPO details https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ · EnvPool https://arxiv.org/abs/2206.10558 · Isaac Gym https://arxiv.org/abs/2108.10470

*Confidence:* POfD ID (1802.05313) is the one to double-check first; R2D2 has no standalone arXiv (OpenReview only); AlphaStar/MuZero/DQN are Nature papers cited by DOI.
