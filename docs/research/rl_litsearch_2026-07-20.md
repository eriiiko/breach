# Breach RL Literature Review & First-Steps Recommendation

*Dated capture, 2026-07-20. Append-only per doc culture — this is a research synthesis, not canon. Five parallel research agents swept the literature across Breach's RL threads; the full cited digests live in [`rl_litsearch_2026-07-20_raw/`](rl_litsearch_2026-07-20_raw/) (one file per theme) and are the durable reference. This doc distills them and ends with an opinionated "what to build first."*

**Feeds:** [ML & Training canon](architecture/ml/01_ml_and_training.md) · [Multi-agent bot training brainstorm](multiagent_training_architectures_brainstorm.md) · [Topic 6 backlog](notes_2026-07-05_topics_backlog.md). **Relates to** the three-model design (Model 1 monolithic commander / Model 2 per-unit goal-conditioned RL / Model 3 commander distributes tile-orders) and the animal-AI track.

---

## 0. The one idea that unifies Erik's four threads

The four things you asked about — team play, single-unit training, the commander↔individual mix, and animals — are **not** four separate research programs. Three of them are the *same architecture at different settings of one dial*: **how abstract is the order the commander sends, and how much is left to the unit to work out?**

```
Model 1 ───────────────────────────────────────────────► Model 3 + Model 2
commander emits CONCRETE orders          commander emits GOAL-LEVEL orders
(move/attack/cover THIS tile)            (take objective / kill X / protect Y)
units = dumb executors                   units = RL-trained to figure out HOW
  │                                                    │
  └── low split, strategy in commander    high split, tactics learned in unit ──┘
              "Battlefield commander gives an objective" lives near this end
```

The literature's clearest practical message (from the hierarchical-RL sweep) is: **make that dial an experimental variable, not an architectural commitment.** One codebase, one order channel, and you slide the abstraction boundary to compare — which is *exactly* the architecture-vs-architecture tournament the brainstorm doc already wants as a first-class goal.

The **animal track** is the one genuinely separate thread: it's the *per-tick reactive brain* (currently rule-based), and its literature is about **emergent behavior from pressure**, not about command hierarchies. It's the most exploratory and, happily, the one where Breach's determinism + parallelism pay off most directly.

---

## 1. What every theme agreed on (the convergent findings)

Five agents searched independently; these points showed up in three or more digests, which is a strong signal:

1. **PPO is almost certainly the first algorithm.** On-policy PPO is sample-*inefficient* but trivially parallel and stable — and sample-inefficiency stops mattering when the environment is nearly free. A deterministic GPU sim makes rollouts cheap, which *inverts the classic algorithm calculus* toward on-policy. Every large game result (AlphaStar, OpenAI Five, hide-and-seek, Isaac Gym) is PPO/actor-critic at scale. `MAPPO`/`IPPO` for the multi-agent case.

2. **The AlphaStar architecture is a near-blueprint for Breach's observation stack.** Transformer entity-encoder over per-unit features + **scatter connections** (place a unit's embedding at its tile in a map plane) + recurrent core + an **autoregressive action head** (order-type → target, masked at each step). Breach's stacked [C,H,W] planes *are* AlphaStar's spatial input; per-unit HP/footprints *are* its entity set. This is the single most reusable design across the whole search.

3. **Determinism is Breach's under-appreciated superpower — and it's structural, not incidental.** Most RL infra literature fights environment nondeterminism (non-associative float reductions, seed variance). Breach removes it: bit-identical trajectories mean distributed self-play never silently diverges, replay buffers/offline datasets/league checkpoints are bit-stable, evaluation needs fewer seeds, and *every emergent-behavior surprise is exactly replayable for debugging*. The CPU reference build becomes an **exact** oracle, not an approximate one.

4. **Don't act every tick.** At 12–24 Hz a tick is ~40–80 ms. Committing to actions for **~3–6 ticks (≈150–300 ms)** ≈ a human reaction budget, quarters the effective horizon (easing credit assignment), and quarters expensive forward-passes. Start with a fixed frame-skip (~4), sweep it early — it's the highest-leverage single knob — then consider a *learned* cadence (FiGAR/TempoRL). Frame it as options/semi-MDPs so value targets stay correct.

5. **Procedural randomization is a requirement, not polish (the SMACv2 trap).** With fixed ship layouts and spawns, agents memorize timestep-scripted openings instead of learning reactive tactics — SMACv2 showed a policy conditioned *only on the clock* beats many fixed SMAC maps. Randomize layouts, spawns, and squad composition or you won't get genuine emergent strategy.

6. **Imitation bootstrap has a clean formalization that unifies your two ideas.** `UVFA` (goal-conditioned network) + `HER` (hindsight relabeling) + `GCSL` (goal-conditioned supervised learning) = "one network, seeded by your scenario-painter recordings, densified by hindsight, fine-tuned by PPO." GCSL *is* goal-conditioned behavioral cloning with relabeling — your recorded `(state, order, action)` demos plug straight in.

7. **Self-play cycles unless you run a population/league.** Naive self-play produces rock-paper-scissors churn. The AlphaStar league (main agents + exploiters) / PSRO is the proven stabilizer — budget for a versioned opponent zoo from the start (cheap under determinism).

---

## 2. Thread-by-thread landscape (compressed)

### Thread A — Team play (StarCraft-style squad of 5–10 with abilities)
The field is **cooperative MARL under CTDE** (centralized training, decentralized execution). Two families:
- **Value decomposition** (`VDN`→`QMIX`→`Weighted-QMIX`/`QPLEX`) — natural for discrete tile-orders; `QMIX`'s hypernetwork is a clean place to inject globally-visible atmosphere/fire/heat planes.
- **Policy gradient** (`MAPPO`, `IPPO`, `HAPPO`/`HARL`, `MAT`) — `MAPPO` is the strong practical baseline; `IPPO` the mandatory cheap ablation.
- **Abilities = heterogeneity**, the hardest structural requirement (breaks naive parameter-sharing). Answers: `HARL` (principled, no parameter-sharing assumption), `ASN` (action-semantics: self-affecting vs other-affecting actions), `UPDeT` (one transformer across unit types), and role methods `RODE`/`ROMA`.
- **Credit assignment** (which marine's action won the round?) is genuinely open. Start with OpenAI Five's **"team-spirit" annealing** (anneal each marine's reward from selfish→shared) — proven and simpler than counterfactual baselines (`COMA`). *Marine deaths mid-round make this an "open-team" problem with no settled solution.*
- Closest benchmark: **SMAC/SMACv2**; closest to Breach's *thesis*: **SMAX/JaxMARL** (GPU-native SMAC without the SC2 engine).

→ Full digest: [`01_marl_squad_combat.md`](rl_litsearch_2026-07-20_raw/01_marl_squad_combat.md)

### Thread B — Commander ↔ individuals (the Battlefield-objective mix)
This is **hierarchical RL** (manager/worker). The lineage: `Feudal RL` → `Options`/`FeUdal Networks` → `h-DQN`/`HIRO` (single-agent) → `FMH` (Feudal Multi-Agent Hierarchies — *the single most on-point paper*, essentially Model 3+2 built and benchmarked) → `RODE` (commander assigns a role that *bounds* the unit's action set). The unit half is **goal-conditioned RL** (`UVFA`+`HER`). The order channel can be a fixed symbol (recommended first) or a learned message (`DIAL`/`TarMAC` — defer).
- The recurring lesson: **the order vocabulary dominates outcomes.** FMH wins only with an adequate subgoal set; too vague → stalls, too concrete → collapses into Model 1.
- The recurring failure mode (from military-HRL papers): **obedience vs local optimization** — a unit rewarded both for following orders *and* local success trades them off; that balance is a tuning dial.
- Non-stationarity: while the unit is still learning to execute "take tile," the *meaning* of that order shifts under the commander — `HIRO`'s off-policy correction (or a staged/frozen-worker curriculum) is the fix.

→ Full digest: [`02_hierarchical_commander.md`](rl_litsearch_2026-07-20_raw/02_hierarchical_commander.md)

### Thread C — Single-unit RL (train one at a time) → scale to a team
The clean recipe: **one goal-conditioned PPO policy** with a **masked autoregressive action head** (AlphaStar-style: order-type, then target conditioned on order-type; `get_legal_actions()` *is* the mask). Bootstrap with **GCSL / goal-conditioned BC** on your recordings, then PPO fine-tune (wrap with `Cal-QL` if the offline→online handoff dips; run a `DAgger` round if it drifts). Reward = sparse terminal + **only potential-based shaping** (Ng et al. — the one theorem that guarantees your shaping can't teach a degenerate loiter-in-cover policy), with **HER** doing the densification instead of hand-tuned dense rewards.
- **Scaling one → many is a known mechanism:** parameter sharing (Gupta et al.) — train one policy, instantiate K copies, each fed its own observation + order, with a role/ID channel for specialization. `MAPPO`/`SMAC` is the validation benchmark.
- **Planning is uniquely affordable for Breach:** because the sim is a perfect forkable model, you can run **AlphaZero-style search with the true model** (skipping MuZero's hardest part) — a serious medium-term option for a *commander*, not a first experiment.

→ Full digest: [`04_single_agent_curriculum_imitation.md`](rl_litsearch_2026-07-20_raw/04_single_agent_curriculum_imitation.md)

### Thread D — Animals (wild/tame, predator–prey, danger-response, taming)
The dominant lesson: **you design a pressure, not a behavior.** Concretely:
- **Predator–prey co-training:** competitive self-play produces an *autocurriculum* — OpenAI's hide-and-seek got six escalating strategy regimes from one objective. Start with "Aquarium/simple_tag on real Breach physics" (N slower predators, few faster prey, **species-shared weights**, PPO/MADDPG, opponent-sampling from a frozen zoo).
- **Danger-response comes for free:** prey rewarded *only to survive* a predator spontaneously flock (`SELFish`; "minimize your domain of danger"). Add `Intrinsic Fear` for durable lethal-cell avoidance.
- **The standout paper:** Kanagawa & Doya 2025 (arXiv:2507.09992) *evolves the reward function itself* (a fear term, a social term) on top of RL behavior — and finds fear evolves **only against actively-hunting predators, and only after a social reward exists.** Almost turnkey for "how would one train animals to react to dangers."
- **Taming = human/scripted-tamer shaping:** `Deep TAMER`/`COACH` (clicker-training analogue) + `MEDAL-ADR` cultural transmission (with expert-dropout so the behavior survives the tamer leaving).
- **Variety = quality-diversity:** `MAP-Elites`/`QDax` illuminates a *repertoire* of temperaments (speed, aggression, cover-use, risk) from one run — literally "wild AND tame, many varied animals."
- **A living ecology** (later): add reproduction/death for endogenous Lotka–Volterra population cycles, then a `POET`/MCC loop co-evolving hazards vs animals. Note the scale finding: some behaviors *only emerge above a population/world-size threshold* — Breach's compute advantage.

→ Full digest: [`03_predator_prey_animals.md`](rl_litsearch_2026-07-20_raw/03_predator_prey_animals.md)

### Thread E (cross-cutting) — Infrastructure & the deterministic-GPU superpower
The prior art for "what Breach is" is **Madrona/GPUDrive** (batch ECS GPU sim + C++/CUDA obs kernels + identical CPU debug backend), **Isaac Gym** (GPU-resident tensors, no CPU round-trip), and **WarpDrive** (multi-agent, single in-place GPU store). The governing principle: **never leave the device** — sim → CUDA encoder → inference → sampling → sim, all in GPU memory, tensors shared zero-copy via **DLPack / `__cuda_array_interface__`**. `SEED RL`'s centralized batched inference is something Breach gets "for free" by construction; the **Anakin** pattern (env + agent compiled into one on-device program) is the ideal end-state. Front-end: **Sample Factory (APPO)** or **PufferLib/PuffeRL** — single-box, high-FPS, PyTorch-native (no JAX rewrite needed). Upgrade lever if PPO's sample cost bites: **PQL** (parallel Q-learning, purpose-built for massively-parallel GPU sim, beats PPO on both wall-clock and sample efficiency).

→ Full digest: [`05_gpu_parallel_infra_cadence.md`](rl_litsearch_2026-07-20_raw/05_gpu_parallel_infra_cadence.md)

---

## 3. What I'd actually build first (concrete, staged)

The brainstorm doc already names the three prerequisites — **state encoder, action masking, reward** — all currently stubs. Nothing trains until those exist. Here is a phased ladder that (a) builds them once, (b) climbs the commander↔unit dial in the order the literature says is safest, and (c) keeps the animal track as a parallel, low-risk sandbox. Each rung is a checkpoint you can stop at.

**Phase 0 — The three prerequisites (gates everything, architecture-independent)**
1. **CUDA state-encoder + zero-copy hand-off.** A kernel that reads the resident Q16.16 state and writes `[num_envs, num_agents, obs_dim]` in place, handed to PyTorch via DLPack — *no device→host bounce*. This is milestone one; every training front-end plugs into it. (Prior art: GPUDrive/Madrona.)
2. **Real `get_legal_actions` masking** — reachable tiles (A* exists) × order type × visible/in-range targets. Masking is *foundational*, not optional, as the action space grows.
3. **A first reward** for one objective (start with a single order type, e.g. "reach tile Y" or "kill unit X"), sparse-terminal + **only potential-based shaping**.

**Phase 1 — One unit, goal-conditioned (Model 2 in isolation)** — *the de-risking rung the brainstorm's Fable note also favored.*
- **PPO** (CleanRL / Sample Factory, following the "37 details" checklist), **masked autoregressive action head**, **UVFA goal input** (the order as extra planes / a vector).
- **Bootstrap:** scenario-painter + order-recorder → **GCSL / goal-conditioned BC** pretrain → PPO fine-tune. **HER** for sparse-reward densification. (This also builds the scenario painter, which overlaps the map editor — Topic 5.)
- **Cadence:** fixed frame-skip ≈4, swept.
- **Success signal:** one marine reliably fulfills a given order across randomized rooms.

**Phase 2 — Many units from one brain (parameter-shared team)**
- Instantiate the Phase-1 policy K times with **shared weights** + a role/ID channel; each marine gets its own observation + order. Validate against **MAPPO/SMAC** conventions.
- Add **team-spirit reward annealing** (selfish→shared) for the CS-objective. Watch for the open-team credit problem when marines die.

**Phase 3 — Put a commander on top (climb the dial)**
- Start at the **RODE/HIRO band** (the research agent's explicit recommendation): a commander that every N ticks emits a **small discrete order per marine from a fixed, human-legible vocabulary** `{move-to-tile, attack-entity, cover-tile, hold, take-objective}` — concrete *argument*, but the *how* left to the Phase-1/2 goal-conditioned unit. This *is* the Battlefield-objective feel, and it gives ablation knobs to slide toward Model 1 (shrink argument granularity) or pure Model 3+2 (widen to abstract objectives + more unit autonomy).
- Keep a **flat no-commander squad** (IPPO/BiCNet-style) as the control that proves the commander earns its cost.
- Use `HIRO`'s off-policy correction (or frozen-worker staging) for the manager↔worker non-stationarity.

**Phase 4 — Arena & league**
- Wrap it in an **AlphaStar-style league / PSRO** so factions embodying different architectures (Model 1 vs 2 vs 3) compete — the tournament that is itself a stated goal. Determinism makes match-ups fair and reproducible.

**Parallel track — Animals (independent, low-risk, high-fun)**
- **A1:** predator–prey co-training ("simple_tag on Breach physics", species-shared PPO, frozen-zoo opponent sampling).
- **A2:** danger-response (survival-only reward → emergent flocking; + Intrinsic Fear). Optionally replicate Kanagawa & Doya's evolved fear/social reward.
- **A3:** taming (Deep TAMER scalar shaping + MEDAL-ADR cultural transmission).
- **A4:** variety via MAP-Elites/QDax.
This track needs only the Phase-0 encoder + a per-tick policy slot (the brainstorm already frames the animal brain as a sibling selected by `nn_intelligence_tier`), so it can run alongside the marine work without blocking it.

---

## 4. Reading queue (if you only read a handful)

Ranked for Breach's specific situation, mixing "read to decide architecture" and "read before writing code":

1. **AlphaStar** (Vinyals et al. 2019, Nature) — the architecture blueprint (entity-encoder, scatter connections, autoregressive head, league). Everything else references it.
2. **MAPPO** (arXiv:2103.01955) + **the 37 PPO implementation details** (iclr-blog-track 2022) — your first algorithm, done right.
3. **FMH — Feudal Multi-Agent Hierarchies** (arXiv:1901.08492) — the commander↔squad structure, built and benchmarked; the most on-point paper for Thread B.
4. **SMACv2** (arXiv:2212.07489) — read *before* designing the scenario generator; the procedural-randomization requirement.
5. **GCSL** (arXiv:1912.06088) + **HER** (arXiv:1707.01495) — the imitation-bootstrap + goal-conditioning stack that unifies your two single-unit ideas.
6. **Ng, Harada & Russell 1999** (potential-based shaping) — the one-page theorem that protects the emergent-strategy goal from reward hacking.
7. **Kanagawa & Doya 2025** (arXiv:2507.09992) — the animal-fear standout; near-turnkey for danger-response.
8. **GPUDrive/Madrona** (arXiv:2408.01584) + **SEED RL** (arXiv:1910.06591) — the infra pattern Breach already half-embodies; the encoder + zero-copy hand-off.
9. **Emergent Tool Use / hide-and-seek** (arXiv:1909.07528) — the proof-of-concept for Breach's entire thesis (emergent strategy from physics + a simple objective).

---

## 5. Cautions carried across the whole search

- **Reward hacking is the biggest threat to the "emergent strategy" goal.** Potential-based shaping only; prefer HER relabeling over dense rewards; watch degenerate loiter/kite policies.
- **Determinism is an asset you must actively guard.** Keep neural-net inference (floats, softmax sampling, GPU nondeterminism) *outside* the synced sim path; the action enters the sim as an integer order. Ensure intrinsic-reward density models / novelty archives / any CPU-side RNG don't leak nondeterminism (cf. the X-ARCH spawn-stat RNG finding — same failure class).
- **Sim-determinism ≠ training-determinism** (cuDNN atomics, async timing) — a deliberate choice, not automatic.
- **Heterogeneity (abilities) breaks parameter-sharing** — commit to HARL/ASN/UPDeT structuring, don't silently share weights across different-kit marines.
- **Self-play cycles; the commander/unit interface is non-stationary; credit assignment under a shared team reward is unsolved** — none are blockers, but budget for them rather than being surprised.
- **Citations are agent-surfaced.** arXiv IDs are reliable but a few venue/year details and 3–4 flagged IDs (Neural MMO orig, GAIL, POfD, Option-Critic, the Madrona arXiv ID, LEHCA DOI) should be spot-checked before formal use — flagged inline in the raw digests.

---

## 6. Ideas Erik raised while skimming (2026-07-20 — capture, act on later)

Not in scope until the engine is finished; recorded so they aren't lost. Each already has a home in the literature above.

- **Enemy swarms — xenomorph-like or Halo-Flood-like hordes.** The directly relevant technique is **large-scale MARL**: `MAgent`/`MAgent2` (scales to ~1e6 agents on one GPU; its "Battle" scenario already produces emergent flanking/encirclement) and `Gigastep` (team combat at ~1e9 steps/s) — the "ran battlefields" large-numbers stuff. Swarm *feel* (relentless, tide-like) also overlaps the animal track: emergent collective motion from simple survival/pursuit rewards (`SELFish`, selfish-herd), and per-tick reactive brains rather than a commander. A swarm likely wants **thousands of cheap parameter-shared reactive agents**, not a small squad of expensive ones — a different scaling regime than the marine work, and one Breach's determinism + parallelism is well-suited to.
- **Eliteness tiers — an elite marine team beating many lower-tier guardsmen through superior tactics *and* equipment.** Two orthogonal axes: (1) **equipment** = different action sets / stats → the **heterogeneous-agent** methods (`HARL`, `ASN`, `UPDeT`); (2) **tactical skill** = training level. A clean, cheap way to get a *tier ladder* is **training checkpoints**: elite = fully-trained policy, lower-tier = earlier checkpoints of the *same* policy (this is Civulator's "save winning weights for beginner opponents" idea, and it doubles as the difficulty knob and as league opponents). Asymmetric elite-vs-horde is then a natural **self-play / league** matchup, and a striking demonstration of *emergent superior tactics* — the project's whole thesis, made legible.
- **The common constraint Erik named: time (compute).** Swarms multiply agent count; tiers multiply training runs. Both are throughput-bound — which is precisely why Thread E (the deterministic GPU-resident, never-leave-the-device infra) is the enabler that makes these affordable rather than aspirational.

---

*Next session: this is capture, not canon. When the RL arc actually opens, fold the chosen first-experiment design into a proper design doc + the ML canon chapter, and archive the exploratory bits. The three-model comparison framing and the "abstraction boundary as an experimental variable" idea are the through-line to carry forward.*
