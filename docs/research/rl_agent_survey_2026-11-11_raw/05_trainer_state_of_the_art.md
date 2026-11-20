# 05: Which trainer, built how (state of the art, late 2026)

*Raw digest by a research agent, 2026-11-11, for the RL agent survey. One of seven; the lead
merges them.*

**How to read the tags.** **[READ]** means I read it in the source during this session (ledger
status FULL or PART). **[ABS]** means I saw only the abstract or a secondary account. **[RECALL]**
means it comes from my memory and was not checked this session; treat it as a lead to verify.
**(opinion)** marks my judgement, not a source's.

**Prior work, not repeated here.** The July digests (`rl_litsearch_2026-07-20.md`, raw 04 and 05)
already recommend PPO / MAPPO first, an AlphaStar-style masked autoregressive action head, a league
for self-play, PQL (Parallel Q-Learning) as an off-policy upgrade, and Sample Factory or PufferLib as
the front end. They assumed the simulator is "nearly free" because it is resident on the GPU. **That
assumption is false for the first runs.** The CPU farm gives about 1 400 env-ticks/s on the small
ship and 500 on the medium one. So this digest asks the question again for a *slow* simulator, and
it checks the July front-end advice against the libraries as they stand in November 2026. Two of
those libraries have changed underneath it.

---

## 0. Words used below (one clause each)

- **Policy**: the network that maps an observation to a probability for each action.
  **Value function / critic**: a network that predicts the reward still to come from a state.
- **On-policy**: the learner uses only data collected by the current policy, then throws it away.
  **Off-policy**: it can also learn from older data (a *replay buffer*), which needs a correction
  or a value-based method.
- **Advantage**: how much better an action did than the critic expected. **GAE** (generalised
  advantage estimation) is the standard way to compute it, with a knob λ (lambda) that trades
  bias against noise.
- **PPO** (proximal policy optimisation): an on-policy method. It takes a batch of fresh data and
  does several gradient passes over it (**epochs**, each split into **minibatches**). A *clip*
  stops any single action's probability from moving too far from where it was when collected.
- **Policy lag / staleness**: how many updates old the policy that collected a sample is.
  **V-trace** (from IMPALA) and importance weights correct for it.
- **Self-play**: the policy trains against copies of itself. A **league** or **population** is a
  set of stored or co-trained opponents. **Strategy cycling** means A beats B, B beats C, and C beats
  A, so "the latest is best" stops being true.
- **CTDE** (centralised training, decentralised execution): each soldier acts on its own view, but
  during training the critic may see everything. **MAPPO** is PPO with this setup and one set of
  weights shared by all soldiers. **IPPO** is the same with a critic that sees only the soldier's
  own view.

---

## A. The algorithm

### A1. PPO "done right": the details that decide whether it works

**The 37 Implementation Details of PPO** (Huang et al., ICLR blog track 2022) [READ]. It lists
13 core details, among them: vectorised envs, orthogonal init, Adam epsilon, learning-rate
annealing, GAE, minibatch updates, advantage normalisation ("at the minibatch level instead of the
whole batch level"), the clipped surrogate, value-loss clipping, the entropy bonus, global gradient
clipping, debug variables, and shared vs separate actor/critic trunks. It lists five LSTM details:
init, zero initial state, "reset the LSTM states at end of episode", "the mini-batches fetch the
sequential training data from sub-environments" (minibatch by whole env sequences, never by shuffled
steps), and saving `initial_lstm_state` before the rollout so training can reconstruct it. For
MultiDiscrete actions it treats the components as "probabilistically independent". For invalid
actions it replaces the logits "with negative infinity before passing the logits to softmax". It
notes that Andrychowicz et al. found value clipping "even hurts performance".

**What Matters in On-Policy RL** (Andrychowicz et al. 2020, arXiv:2006.05990; 250 000 agents
trained, continuous control only) [READ, the boxed recommendations]:
- PPO loss, clip start 0.25.
- Last policy layer initialised 100x smaller.
- "Always use observation normalization and check if value function normalization improves
  performance".
- GAE λ = 0.9, "neither Huber loss nor PPO-style value loss clipping".
- "Go over experience multiple times", recomputing the advantages on each pass.
- γ (the discount) is "one of the most important hyperparameters" (start at 0.99).
- Adam with lr 3e-4.
- No special timeout handling is needed for long step limits.

**MAPPO** (Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games",
arXiv:2103.01955) [READ, the suggestions and numbers] gives five suggestions:
1. "Utilize value normalization to stabilize value learning."
2. The critic input holds both local, agent-specific features and global features.
3. "Use at most 10 training epochs on difficult environments and 15 training epochs on easy
   environments. Additionally, avoid splitting data into mini-batches." One minibatch was best on
   22 of 23 SMAC maps.
4. Clip ε under 0.2; 0.05 gave consistently high final performance but learned slower on some maps.
5. A large batch.

Death masking (appendix C.3): a dead agent's critic input becomes a zero vector plus its agent id.
On sample efficiency: "PPO often achieves competitive or superior results in both final returns and
sample efficiency" compared with the off-policy MARL baselines.

**The tension that matters for Breach: how much to reuse each sample.** OpenAI Five
(arXiv:1912.06680) [READ, sections] targeted "a staleness between 0 and 1" and a sample reuse of
about 1. It found that reusing data "only 2 to 3 times could halve training speed" and that
"reusing it 8 times may prevent the learning of a competent policy altogether". MAPPO says the
opposite: 10 to 15 epochs over one batch. The two are measuring different things. OpenAI Five was
compute-rich and data-rich, so it measured *wall-clock speed*, and its reuse is total passes per
sample under asynchrony. MAPPO measures *samples to a result*. (opinion) A slow simulator is in
MAPPO's regime. Data is the scarce resource, so several epochs are right, with a KL or clip-fraction
monitor and early stopping (detail "early stopping of policy optimizations" in the 37) to catch the
point where reuse starts to hurt. Neither paper settles the right number for Breach; it is a sweep.

### A2. Asynchronous PPO and V-trace (IMPALA, APPO / Sample Factory)

IMPALA (arXiv:1802.01561) decouples actors from the learner and corrects the lag with V-trace
[RECALL; July digest]. Sample Factory's APPO is asynchronous PPO on one machine with a lag
correction [RECALL; July digest]. PufferLib 3.0 folds the correction into PPO: "Puffer Advantage
is equal to GAE when you set both clip coefficients to infinity" and "equal to VTrace with
lambda=1, so our method is a strict generalization of both" [READ, puffer.ai/blog/ppo]. The post
reports that combining V-trace with filtering "was way too off-policy and didn't work at all"
[READ]. **For Breach:** the farm already steps in lock step, so synchronous PPO is the natural fit.
Asynchrony would buy only an overlap of learning with collection (policy lag 1). If that overlap is
added later, a V-trace-clipped GAE of the Puffer kind is the cheap correction. (opinion)

### A3. The 2024 to 2026 challengers

- **PufferLib's own PPO changes** (3.0, "Puffing Up PPO") [READ]:
  - Muon optimiser instead of Adam ("initially didn't seem to make much of a difference", then
    "an immediate step-change in capabilities" in full sweeps; Breakout solved "~30% faster").
  - Cosine learning-rate annealing.
  - Prioritised replay of trajectory *segments* by advantage (alpha = 0, beta0 = 1 gives uniform
    sampling, so it is sweepable; "a massive improvement if rewards are sparse", no numbers).
  - The Puffer Advantage above.
  - Hyperparameters found by their tuner Protein (CARBS-based).

  The evidence is sweeps on their own single- and multi-agent C environments ("Ocean"). The post
  makes no claim about competitive self-play. **PufferLib 4.0** [READ]: Torch replaced by
  "5,000 lines of deterministic CUDA-C" (Torch kept as a fallback), minGRU instead of LSTM,
  "highway connections" instead of normalisation layers, 15M steps/s on one RTX 5090, envs solved
  3-5x faster than in 3.0. (opinion) Muon and cosine are cheap ablations to adopt. Segment
  prioritisation is worth a sweep because Breach's kill rewards are sparse.
- **PQN** (Gallici et al., "Simplifying Deep Temporal Difference Learning", arXiv:2407.04811, v6
  Apr 2025; no venue on the abs page) [READ, PART]. LayerNorm makes Q-learning stable without a
  target network or a large replay buffer; many parallel envs replace the buffer. On Atari, "PQN
  outperforms PPO in terms of sample efficiency, final score, and training time". On Craftax with
  an RNN it reached 16 % against PPO-RNN's 15.3 %. "PQN-VDN runs successfully on Smax without a
  large buffer, outperforming MAPPO and QMix". Smax is a *cooperative* benchmark against a scripted
  enemy. **For Breach:** Q-learning picks the greedy action, which gives a deterministic policy.
  In a competitive game the best play is often a *mixture* (feint left or right at random), so
  greedy-Q self-play is a known poor fit (opinion; general game-theory knowledge, [RECALL]). PQN
  also needs a way to take an argmax over a pointer-and-factored action. It stays a later
  experiment for the cooperative zombie scenario, not the first trainer.
- **SAPG** (Singla, Agarwal, Pathak, ICML 2024 oral, arXiv:2407.20230) [ABS]. PPO "fail[s] to
  ingest the benefit of parallelized environments beyond a certain point". SAPG splits the envs
  into subsets and aggregates them with importance sampling. This matters only at 10^4+ GPU envs,
  so it belongs to the GPU-resident phase.
- **PQL** (arXiv:2307.12983): off-policy, built for massive GPU parallelism, continuous control
  [RECALL; July digest]. Same verdict as SAPG: a later lever.
- **Model-based methods.** DreamerV3 (arXiv:2301.04104, v2 Apr 2024) [ABS]: fixed hyperparameters
  across domains, held together by "robustness techniques based on normalization, balancing, and
  transformations" (symlog, two-hot returns). EfficientZero V2 (arXiv:2403.00564) [ABS]: better
  than DreamerV3 on 50 of 66 tasks under limited data (Atari 100k, control); single-agent. Muesli
  (arXiv:2104.06159): a model-based regulariser on a policy-gradient core [RECALL]. None of them
  shows results in multi-agent competitive self-play. A learned model of fire, smoke and pressure
  fields is a research project of its own. (opinion) Their *tricks* transfer (symlog / two-hot
  value targets, return normalisation); their *algorithms* are not a first trainer for Breach.
- **Bigger networks.** SimBa (ICLR 2025 spotlight, arXiv:2410.09754) [ABS]: observation
  normalisation + residual blocks + LayerNorm lets RL networks grow with better sample efficiency,
  "across off-policy, on-policy, and unsupervised" methods (the abstract does not name PPO). BRO
  (arXiv:2405.16158), SimBa-v2: [RECALL]. "Stop Regressing" (Farebrother et al.,
  arXiv:2403.03950; ICML 2024 per [RECALL]) [ABS]: training value functions with categorical
  cross-entropy (HL-Gauss / two-hot) instead of MSE "can yield substantial improvements". It was
  shown with value-based and large-network agents, not with PPO self-play.
- **Policy gradient in adversarial games, 2025 to 2026.** Rudolph et al., "Reevaluating Policy
  Gradient Methods for Imperfect-Information Games" (ICLR 2026, arXiv:2502.08938) [ABS]. It ran
  more than 7 000 runs with exact exploitability (how much a perfect opponent could beat you) on
  five large games and found that "FP-, DO-, and CFR-based approaches fail to outperform generic
  policy gradient methods" such as PPO. This is the strongest recent evidence that plain PPO
  self-play is not a naive choice for competitive play. Follow-ups regularise PPO self-play toward
  a "magnet" policy (uniform, or an exponential moving average of its own weights: EMAgnet,
  arXiv:2606.23995) [ABS, search snippet only]. All of these are two-player card or matrix games.
- **Newer MARL sequence models.** Sable (arXiv:2410.01706, retention-based) [ABS]: beats the
  state of the art on 34 of 45 *cooperative* tasks and scales to 1 000+ agents. MACA
  (arXiv:2508.06836) [ABS, snippet]: a credit-assignment add-on to MAPPO, also cooperative.
- **I found no 2025 or 2026 paper that claims to beat PPO in multi-agent *competitive team*
  games.** One extended and two standard searches turned up nothing. The nearest is LeRoy & Kolb
  (arXiv:2512.08877) [ABS]: plain IPPO self-play "generalizes to novel teammate algorithms despite
  not experiencing teammate diversity during training", matching a rotating-teammate scheme.

### A4. What a slow simulator changes

Compare budgets: hide-and-seek (arXiv:1909.07528) [READ] needed "132.3 million episodes (31.7
billion frames)" to reach ramp defence. The farm gives about 1.2 x 10^8 env-ticks per day (1 400/s
x 86 400 s). At that rate hide-and-seek's budget would take about 260 days (my arithmetic). So the
first Breach tasks must be **SMAC-sized, not hide-and-seek-sized**. MAPPO-scale problems reach
their results in about 10^7 env steps [RECALL], which is under a day here. The levers, ordered by
evidence:
1. **Reuse each batch several times** (MAPPO 5 to 15 epochs, with a KL / clip-fraction guard)
   [READ]. Strong evidence in cooperative MARL. OpenAI Five shows the danger when overdone [READ].
2. **Act every k ticks** (the July digest's frame-skip k ≈ 4). This cuts the decisions to learn
   per second of game time, not the ticks simulated, but it shortens the horizon and the credit
   chain [RECALL, July].
3. **Every soldier is a sample.** With shared weights, 2 teams x 6 policy soldiers give 12
   decisions per env-step, so about 3.6 x 10^8 soldier-decisions per day at k = 4 (arithmetic).
4. **Cheap value-learning upgrades** (value normalisation [READ, MAPPO]; classification-style
   value loss [ABS]).
5. **Off-policy replay or a learned model.** These give the largest sample-efficiency gains in
   single-agent benchmarks [ABS: EfficientZero V2, PQN], but there is no evidence in competitive
   team self-play, and they cost the most engineering. A recent study of PPO with windowed sample
   reuse (Montenegro et al., arXiv:2610.01399, Oct 2026) [ABS] did not state its result in the
   abstract.

**Verdict on A.** The evidence supports **PPO, as MAPPO, with a centralised critic**: strong for
cooperative MARL (MAPPO; every big game result) and moderately strong for competitive play (OpenAI
Five, hide-and-seek, TiZero, and the 2025 Rudolph result in two-player games). I found no
competitive-team evidence for any challenger. The slow simulator argues for tuning PPO toward data
reuse, not for leaving PPO.

---

## B. Self-play and opponents

- **Bansal et al., "Emergent Complexity via Multi-Agent Competition"** (arXiv:1710.03748) [READ]:
  - "training against the latest opponent leads to worst performance".
  - "training against random old versions of the opponent to work much better".
  - The best window was δ = 0.5 (Humanoid) and δ = 0 ("uniform sampling over the entire history",
    Ant).
  - A dense exploration reward was "gradually annealed to zero", used for "only about 10-15% of
    the training epochs".
- **OpenAI Five** [READ]:
  - "They play the latest policy against itself for 80% of games, and play against older policies
    for 20% of games".
  - Progress was measured "to a pool of fixed reference agents with known skill using the TrueSkill
    rating system". About 8.3 TrueSkill corresponds to an 80 % win rate.
  - Team spirit (how much of a teammate's reward each agent shares) was ramped from 0.3 to 1.0.
  - Batch of 1 to 3 million timesteps in LSTM windows of 16; a single-layer 4096-unit LSTM.
- **TiZero** (Google Research Football 11 v 11, arXiv:2302.07515) [READ, sections]:
  - Two stages. First, curriculum self-play over "ten difficulty levels". Second, "Challenge &
    Generalise" self-play.
  - Challenge step: "the current agents play against the most recently saved agents with
    probability of 80%", older ones 20 %.
  - Generalise step: opponents sampled from the whole pool with weight (1 − p)², where p is the
    estimated win probability against that opponent. This is prioritised fictitious self-play
    (PFSP), as in AlphaStar [RECALL for AlphaStar's exact form].
  - Algorithm: MAPPO extended to JRPO (one joint probability ratio for the team), a shared policy
    conditioned on player id, an LSTM.
  - Compute: about 45 days on 800 CPUs + 2 A100s, batch 2.15M, γ = 0.999.
  - Evaluation: TrueSkill over a fixed pool that includes the built-in AI and rule-based agents.
- **Hide-and-seek** [READ]: plain self-play, and "we require neither population-based training".
- **AlphaStar league** (Nature 2019): main agents + main exploiters + league exploiters, PFSP,
  V-trace + UPGO [RECALL]. **FTW / Capture the Flag** (Jaderberg et al., Science 2019): population
  of 30 with population-based training and Elo-matched opponents [RECALL]. **NeuPL**
  (arXiv:2202.07415) [ABS]: "representing a population of policies within a single conditional
  model", so skills are not relearned for each new population member. **PSRO** (policy-space
  response oracles: repeatedly train a best response to a mixture of the current population)
  [RECALL]. All four cost a multiple of one training run. At 10^8 ticks a day that multiple is
  felt directly.

**The smallest trustworthy scheme** (opinion, assembled from the READ items):
1. **Opponent mix per match:** about 80 % the latest policy or recent snapshots, about 20 % past
   snapshots sampled by PFSP weight (1 − p)², plus a fixed share of scripted `bots` / `bots_v1`
   teams. The bots anchor the opponent distribution so the policy cannot forget how to beat simple
   play. Snapshots are frozen every N updates; Breach's determinism makes each one exactly
   replayable.
2. **Progress measure:** a TrueSkill or Elo ladder over a **fixed reference pool** (`bots_v1`,
   `bots`, an idle team, and frozen milestone snapshots), played on a held-out seed set of levels,
   both sides mirrored. This is the existing Bot tournament instrument, reused. Win rate against
   `bots_v1` is the first plain-language curve. Rating against the pool is the curve that does not
   cycle.
3. **Cycling detector:** a win-rate matrix over the snapshot pool. Non-transitive triangles show
   up in it directly.
4. **Defer** exploiters, PBT, NeuPL and PSRO until the ladder shows a plateau or cycling.

Strength of evidence: the 80/20 + past-snapshot + fixed-reference recipe is well documented in
three READ sources at large scale. That it is *sufficient* to prevent cycling in Breach is not
shown anywhere; it is an assumption that the matrix in item 3 tests.

---

## C. Memory and network details the trainer must support

- **Recurrent memory.** POPGym (arXiv:2303.01859) [READ]: "the GRU is the best general-purpose
  memory model". Its linear-attention variants "were some of the worst contenders in RL". GTrXL
  (arXiv:1910.06764) [ABS] "surpasses LSTMs on challenging memory environments" (DMLab-30), with
  gating and reordered layer norms making transformers trainable under RL [RECALL for the exact
  changes]. PufferLib 4.0 swapped its LSTM for minGRU, which trains in parallel over time and is
  "equal to or better than an LSTM on every environment we've tested" [READ]. OpenAI Five, TiZero
  and hide-and-seek all used an LSTM [READ]. **Trainer requirement:** store the hidden state at the
  start of each rollout segment, reset it on episode end *and on a soldier's death*, and minibatch
  by whole segments (37 details) [READ]. (opinion) A GRU first; transformer memory over time is
  deferred until a memory task proves the GRU short.
- **Entity attention with variable counts.** Hide-and-seek: entities embedded "with shared weights
  across entity types", masked self-attention, pooling, then an LSTM [READ]. Entity Gym (Clemens
  Winter): observations with "dynamically-sized lists of entities"; its docs list "Action Masks"
  and "SelectEntityAction" (a pointer over entities) sections [READ, index only]. enn-trainer
  offers "PPO or behavioral cloning" [READ]. Neither repo shows a release or a recent commit, and I
  could not see a commit date, so treat them as reference code, not a dependency. **Trainer
  requirement:** padded entity tensors `(B, max_entities, F)` plus a validity mask. Attention is
  masked on padding, never on zeros.
- **Masked, factored and autoregressive actions.** Masking sets illegal logits to −∞ [READ, 37
  details; Huang & Ontañón arXiv:2006.14171, RECALL]. Factored heads treat components as
  independent [READ]. An autoregressive head (AlphaStar: action type, then target pointer
  conditioned on the type) [RECALL] needs the trainer to (a) store each sub-action and its mask,
  (b) form the joint log-probability as the sum over the sub-heads actually used (a "move" action
  contributes no aim term), and (c) recompute it at training time with the *stored* sub-choices
  fed as conditioning. The entropy bonus is summed over used heads under the masks. (opinion) This
  is the place a self-written trainer most often gets subtly wrong. It deserves a test that
  recomputes log-probs at collection and at training time and checks they agree exactly before any
  update.
- **Value scale.** Value normalisation is a MAPPO must [READ]. PopArt (van Hasselt et al. 2016;
  multi-task PopArt arXiv:1809.04474) rescales targets while preserving outputs [RECALL]. Symlog /
  two-hot (DreamerV3) [ABS] and HL-Gauss cross-entropy ("Stop Regressing") [ABS] are alternatives.
  (opinion) Start with running mean/std value normalisation; classification loss is an ablation.
- **Centralised critic.** Hide-and-seek: "a centralized omniscient value function ... access to the
  full environment state" while "each agent acts given only its own observations and memory state"
  [READ]. MAPPO: local + global features, death masking [READ]. TiZero: a centralised value [READ].
  **Trainer requirement:** a separate critic input (the full entity list, both teams, without fog)
  produced by the env beside the soldier observation. The critic is never used at play time.
- **Large batches.** MAPPO: a large batch, little or no minibatch splitting [READ]. OpenAI Five:
  1 to 3M steps per batch [READ]. TiZero: 2.15M per GPU [READ]. (opinion) At 1 400 env-ticks/s, k = 4 and
  12 soldiers, the farm produces about 4 200 soldier-decisions/s, so a 1M-decision batch is about
  4 minutes of farm time (arithmetic). Large batches are affordable, but each update is minutes
  apart, which is one more reason to squeeze each batch with several epochs.

---

## D. What to build on (PyTorch)

| Option | Multi-agent rows + mask, external batched env | Masked / factored / pointer actions | Recurrent | Self-play with frozen snapshots | Windows + Linux | Status Nov 2026 |
|---|---|---|---|---|---|---|
| **Self-written CleanRL-style PPO** | yes: consumes `BatchedEnv` directly | written to fit | written (37-details LSTM recipe) | written (a snapshot store + a sampler) | yes (plain PyTorch) | ours |
| **PufferLib 5.0** | **no**: "Ocean environments are written in C"; "Where did all the Python/third-party stuff go? ... 100x+ slower" [READ, docs] | n/a | minGRU [READ] | not documented [READ] | "no CPU training option"; install script only, no Windows statement [READ] | active, CUDA-C native backend |
| **PufferLib 3.0.x** (pufferlib-core 3.0.17 on PyPI [ABS]) | had Python / PettingZoo emulation [RECALL] | MultiDiscrete [RECALL] | LSTM [RECALL] | not built in [RECALL] | Linux; Windows via WSL [RECALL] | superseded by 4.0 / 5.0 |
| **Sample Factory 2** | multi-agent, self-play, multiple policies, PBT listed [READ] | discrete / tuple [RECALL] | yes [READ, implied] | yes [READ] | "There is no Windows support at this time" [READ] | last PyPI release 2.1.1, 19 Jun 2023 [READ] |
| **RLlib (Ray)** | multi-agent PPO on the new API stack [ABS, search]; its own EnvRunner abstraction | via custom RLModule [RECALL] | yes [RECALL] | self-play and league examples [ABS] | Windows unconfirmed [ABS] | active; heavy |
| **TorchRL** | multi-agent tutorials (VMAS) [RECALL]; TensorDict everywhere | yes [RECALL] | yes; GAE with validity mask, recurrent-safe constructor (v0.15) [READ] | not built in [RECALL] | yes [RECALL] | very active: v0.15.0 (10 Nov), v0.14 (Sep) [READ] |
| **JAX: Mava / JaxMARL** | Mava: IPPO, MAPPO, Sable, MAT and others; non-JAX envs only via "Sebulba" and listed as a roadmap item [READ] | yes [RECALL] | yes [READ] | cooperative focus [RECALL] | JAX on Windows is CPU-only or WSL [RECALL] | active (© 2026) |

**Recommendation (opinion): a self-written PPO in PyTorch, CleanRL-style, about 1 500 to 2 500
lines, written against `BatchedEnv` directly. Borrow components (TorchRL's GAE with a validity
mask, PufferLib's V-trace-clipped advantage, Muon), never a framework.** Reasons:
1. `BatchedEnv` already *is* the vectorised multi-agent env: rows `(team, slot)`, a valid mask,
   shared-memory buffers. Every framework wants to own the env loop (PufferLib emulation, Sample
   Factory workers, RLlib EnvRunners, TorchRL collectors). Wrapping Breach to fit one would add a
   copy and a translation layer on exactly the path that already costs time.
2. The two single-box front ends the July digest named are gone for Breach's case. PufferLib 5
   accepts only C environments and needs a GPU; Sample Factory has not been released since 2023
   and does not run on Windows [READ].
3. What Breach needs is custom anyway: pointer-over-enemies heads, autoregressive masks, death
   resets of the recurrent state, a privileged critic input, self-play against frozen snapshots
   *and* scripted controllers, bit-exact replay links. No library does all of these; the glue
   would be most of the code.
4. A trainer this size is readable, which matters for an author learning the field.

**Main risk:** a silent bug in the self-written trainer. Typical ones are a wrong joint log-prob
for autoregressive heads, a hidden state not reset at death, an advantage leaking across episode
boundaries, or a mask applied at collection but not at training. Any of these looks like "RL is
hard" instead of a bug. Mitigations (opinion):
- Port CleanRL's `ppo_atari_lstm` semantics line by line and check learning curves on one tiny
  known task first (a toy env or CartPole-with-masking) before Breach.
- Property tests: identical log-probs at collection and training time on the first epoch; ratio
  == 1 before the first update; masked actions never sampled; GAE equal to a hand-computed case.
- Log the 37 details' debug variables (approx-KL, clip fraction, explained variance, entropy per
  head).

---

## For Breach: the first trainer (a proposal, opinions marked)

**Algorithm.** Synchronous PPO in the MAPPO setup: one policy shared by all soldiers of the
learning side, a centralised critic on privileged state, GAE, decisions every k ticks (k swept,
July's ≈ 4). (opinion, grounded in A)

**The short list of details that must be right** (each from a READ source):
1. Masked logits = −∞ at collection *and* training; joint log-prob over used sub-heads.
2. Recurrent state: zero init, reset at episode end and at death, stored per segment start,
   sequence minibatches.
3. Advantage normalisation per minibatch; GAE λ around 0.9 to 0.95 and γ swept (γ = 0.99 at
   k = 4 gives a horizon of about 100 decisions ≈ 17 s of game time, my arithmetic; long fights
   may want 0.995).
4. Value normalisation; no value clipping (Andrychowicz) as the default, with a switch.
5. Observation normalisation (running statistics) on the vector features; small init of the last
   policy layer.
6. Several epochs (start around 4 to 10) on a large batch with few minibatches, with approx-KL
   early stopping. This is the main sample-efficiency lever for the slow simulator.
7. Truncation at the 300 s limit bootstrapped from the critic, not treated as a death.
8. Death masking of the critic input for dead soldiers (MAPPO App. C.3).

**Self-play scheme.** Latest and recent snapshots about 80 %, PFSP over the past-snapshot pool
about 20 %, a fixed share of `bots` / `bots_v1` opponents, and the horde in every match as the
match config sets it. Snapshots frozen every N updates and stored with the build id, so any match is
replayable. (opinion)

**Progress measure.** TrueSkill or Elo against a fixed reference pool (`bots_v1`, `bots`, frozen
milestones) on held-out seeds and levels through the Bot tournament instrument, plus win rate
against `bots_v1` as the first curve and a snapshot win-rate matrix as the cycling check.
(opinion, after OpenAI Five and TiZero)

**Built on.** Self-written PyTorch PPO against `BatchedEnv`; the learner on the desktop GPU,
inference batched for all envs in one forward pass per decision step. (opinion)

**Cheap ablations once it learns.** Muon vs Adam, cosine lr, segment prioritisation by advantage
(PufferLib 3), HL-Gauss value loss, GRU vs minGRU, epochs, k. (opinion)

**Defer.** Asynchronous collection (add V-trace-clipped GAE when it comes), PQN / PQL / SAPG
(GPU-resident scale or cooperative-only tasks), world models (DreamerV3, EfficientZero V2),
transformer memory over time, the AlphaStar league / NeuPL / PSRO / PBT, JRPO-style joint ratios.

**Open questions.**
- How many epochs the slow simulator tolerates before reuse hurts (OpenAI Five vs MAPPO): a sweep,
  not a citation.
- Whether 80/20 + PFSP + bots is enough to stop cycling in Breach: measured by the matrix.
- Whether a shared team reward, a per-soldier reward, or OpenAI Five's annealed "team spirit"
  fits friendly fire on and no respawn: digest 01 / reward digest territory.
- Whether greedy-Q methods (PQN) are worth trying in the cooperative-versus-horde scenario.
- PufferLib 3's Windows support and its Python emulation in its last release: RECALL, unverified.

---

## Source ledger

| Citation | URL | Status | Supports |
|---|---|---|---|
| Huang et al., The 37 Implementation Details of PPO, ICLR blog track 2022 | https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ | FULL (first 100k chars) | PPO detail list, LSTM recipe, masking, adv. norm, value clipping note |
| Andrychowicz et al., What Matters in On-Policy RL, arXiv:2006.05990 | https://ar5iv.labs.arxiv.org/html/2006.05990 | PART (recommendations) | clip, init, obs/value norm, GAE 0.9, no value clip, reuse, γ, Adam |
| Yu et al., MAPPO, arXiv:2103.01955 | https://ar5iv.labs.arxiv.org/html/2103.01955 | PART | 5 suggestions, epochs 10/15, 1 minibatch, clip, death masking, sample efficiency |
| Berner et al., OpenAI Five, arXiv:1912.06680 | https://ar5iv.labs.arxiv.org/html/1912.06680 | PART | 80/20 self-play, staleness, reuse harm, batch, LSTM, TrueSkill pool, team spirit |
| PufferLib, "Puffing Up PPO" (3.0) | https://puffer.ai/blog/ppo/ | FULL | Muon, cosine lr, segment prioritisation, Puffer Advantage = GAE/V-trace |
| PufferLib 4.0 post | https://puffer.ai/blog/4.0/ | FULL | CUDA-C trainer, minGRU, highway connections, 15M sps |
| PufferLib docs (5.0) | https://puffer.ai/docs.html | FULL | C-only envs, no CPU training, Python stuff removed, minGRU claim |
| PufferLib site index | https://puffer.ai/ | FULL | blog list |
| PufferLib GitHub | https://github.com/PufferAI/PufferLib | PART | 5.0 branch; README minimal |
| pufferlib-core 3.0.17 (PyPI) | https://pypi.org/project/pufferlib-core/3.0.17/ | ABS (search hit) | a 3.0 line exists |
| PufferLib RLC 2025 paper | https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_151.pdf | ABS (search snippet) | 3.0 algorithmic improvements + tuner |
| Gallici et al., PQN, arXiv:2407.04811 | https://arxiv.org/html/2407.04811 | PART | LayerNorm TD, beats PPO on Atari, PQN-VDN > MAPPO on Smax |
| Singla et al., SAPG, ICML 2024, arXiv:2407.20230 | https://arxiv.org/abs/2407.20230 | ABS | PPO plateaus with many envs |
| Li et al., PQL, arXiv:2307.12983 | https://arxiv.org/abs/2307.12983 | RECALL (July digest) | off-policy at GPU scale |
| Petrenko et al., Sample Factory, arXiv:2006.11751 | https://github.com/alex-petrenko/sample-factory | PART (README) | no Windows; multi-agent, self-play, PBT |
| sample-factory on PyPI | https://pypi.org/project/sample-factory/ | FULL | last release 2.1.1, 2023-06-19 |
| Espeholt et al., IMPALA / V-trace, arXiv:1802.01561 | https://arxiv.org/abs/1802.01561 | RECALL | off-policy correction |
| Hafner et al., DreamerV3, arXiv:2301.04104 | https://arxiv.org/abs/2301.04104 | ABS | robustness tricks; v2 Apr 2024; Nature venue not confirmed on page |
| Wang et al., EfficientZero V2, arXiv:2403.00564 | https://arxiv.org/abs/2403.00564 | ABS | > DreamerV3 on 50/66, limited data, single-agent |
| Hessel et al., Muesli, arXiv:2104.06159 | https://arxiv.org/abs/2104.06159 | RECALL | model-based regularised PG |
| Lee et al., SimBa, ICLR 2025, arXiv:2410.09754 | https://arxiv.org/abs/2410.09754 | ABS | scaling networks, on- and off-policy |
| Nauman et al., BRO, arXiv:2405.16158 | https://arxiv.org/abs/2405.16158 | RECALL | network scaling (off-policy) |
| Farebrother et al., Stop Regressing, arXiv:2403.03950 | https://arxiv.org/abs/2403.03950 | ABS | classification value loss |
| Rudolph et al., Reevaluating PG for Imperfect-Information Games, ICLR 2026, arXiv:2502.08938 | https://arxiv.org/abs/2502.08938 | ABS | generic PPO matches/beats FP/DO/CFR deep methods |
| EMAgnet, arXiv:2606.23995 | https://arxiv.org/abs/2606.23995 | ABS (search snippet) | EMA-magnet regularised PPO self-play |
| Mahjoub et al., Sable, arXiv:2410.01706 | https://arxiv.org/abs/2410.01706 | ABS | cooperative SOTA 34/45, 1000+ agents |
| MACA, arXiv:2508.06836 | https://arxiv.org/html/2508.06836v1 | ABS (search snippet) | MAPPO credit-assignment add-on, cooperative |
| LeRoy & Kolb, IPPO Learns the Game, arXiv:2512.08877 | https://arxiv.org/abs/2512.08877 | ABS | IPPO self-play generalises to new teammates |
| Montenegro et al., Reusing Past Samples in PPO, arXiv:2610.01399 | https://arxiv.org/abs/2610.01399 | ABS | windowed reuse in PPO (no result in abstract) |
| Bansal et al., Emergent Complexity via Multi-Agent Competition, arXiv:1710.03748 | https://ar5iv.labs.arxiv.org/html/1710.03748 | PART | latest-only worst; past sampling δ; dense reward annealing |
| Lin et al., TiZero, arXiv:2302.07515 | https://ar5iv.labs.arxiv.org/html/2302.07515 | PART | curriculum, 80/20 + (1−p)² PFSP, JRPO, compute, TrueSkill |
| Liu et al., NeuPL, arXiv:2202.07415 | https://arxiv.org/abs/2202.07415 | ABS | population in one conditional net |
| Vinyals et al., AlphaStar, Nature 2019 | https://doi.org/10.1038/s41586-019-1724-z | RECALL | league, PFSP, autoregressive head, V-trace + UPGO |
| Jaderberg et al., Capture the Flag (FTW), Science 2019 | https://doi.org/10.1126/science.aau6249 | RECALL | population + PBT in a team FPS |
| Lanctot et al., PSRO, arXiv:1711.00832 | https://arxiv.org/abs/1711.00832 | RECALL | population best-response scheme |
| Baker et al., Hide-and-seek, arXiv:1909.07528 | https://ar5iv.labs.arxiv.org/html/1909.07528 | PART | entity attention + LSTM, omniscient critic, PPO+GAE, 31.7B frames, no PBT |
| Parisotto et al., GTrXL, arXiv:1910.06764 | https://arxiv.org/abs/1910.06764 | ABS | transformers > LSTM on memory tasks |
| Morad et al., POPGym, arXiv:2303.01859 | https://arxiv.org/html/2303.01859 | PART | GRU best general memory; linear attention poor |
| Huang & Ontañón, Invalid Action Masking, arXiv:2006.14171 | https://arxiv.org/abs/2006.14171 | RECALL | masking keeps PG valid |
| van Hasselt et al., PopArt (arXiv:1602.07714); Hessel et al. multi-task PopArt, arXiv:1809.04474 | https://arxiv.org/abs/1809.04474 | RECALL | value target normalisation |
| Entity Gym (C. Winter) | https://github.com/entity-neural-network/entity-gym ; https://entity-gym.readthedocs.io/en/latest/index.html | PART (README + docs index) | variable entity lists; masks; SelectEntityAction |
| enn-trainer | https://github.com/entity-neural-network/enn-trainer | PART (README) | PPO or BC over entity obs; no visible release |
| RLlib docs (new API stack) | https://docs.ray.io/en/latest/rllib/rllib-new-api-stack.html | ABS (search summary) | multi-agent PPO on new stack, self-play/league examples |
| TorchRL releases | https://github.com/pytorch/rl/releases | PART | v0.15.0 (10 Nov), GAE validity mask, recurrent-safe GAE, PPOTrainer |
| Mava | https://github.com/instadeepai/Mava | PART (README) | IPPO/MAPPO/Sable/MAT; non-JAX envs via Sebulba/roadmap |
| GT Sophy (Wurman et al., Nature 2022) | https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf | ABS (search summary) | slow real-time sim -> off-policy QR-SAC on 1 000+ PS4s (single-agent racing; not used above, context only) |
