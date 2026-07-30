# Multi-Agent RL for Team / Squad Combat — Literature Digest for Breach

*Raw research-agent output, 2026-07-20. Scope: one coordinating intelligence over 5–10 heterogeneous units with distinct ability kits, trained in a deterministic GPU tick-sim as an RL state space, targeting emergent CS-style team-vs-team tactics. arXiv IDs verified against search results; uncertain venue/year details flagged.*

Breach sits at a specific intersection: **cooperative micro-control of a small heterogeneous squad** (like SMAC/AlphaStar micromanagement) wrapped in a **competitive team-vs-team objective** (like OpenAI Five / a CS match), running on a **determinism-first, massively-parallelizable engine** (like Gigastep/JaxMARL). The literature splits along those three axes.

---

## 1. Foundational cooperative MARL — the CTDE backbone

Dominant paradigm: **Centralized Training with Decentralized Execution (CTDE)** — agents learn with access to global state during training, act on local observation at execution. Maps to Breach: during self-play you have the full deterministic engine state; at inference each marine acts on its own feature-plane crop plus shared context.

### Value-decomposition family (value-based, discrete actions)
Learn a joint action-value Q_tot that factorizes into per-agent utilities, satisfying **IGM (Individual-Global-Max)**. Workhorse family for SMAC-style discrete micro; natural first fit for Breach's discrete tile-targeted orders.

- **VDN** — Sunehag et al. 2017, arXiv:1706.05296 (AAMAS 2018). Q_tot = Σ Q_i; solves "lazy agent" / spurious-reward. *Breach:* simplest credit-assignment baseline.
- **QMIX** — Rashid et al. 2018, arXiv:1803.11485 (ICML 2018; JMLR 21, 2020). Monotonic mixing network via hypernetwork on global state. *Breach:* default best-supported CTDE baseline; hypernetwork is a clean place to inject globally-visible atmosphere/fire/heat planes.
- **QTRAN** — Son et al. 2019, arXiv:1905.05408 (ICML 2019). Lifts structural constraints; finicky in practice. *Breach:* option if focus-fire/kiting needs action-ordering dependence QMIX can't represent.
- **Weighted QMIX (CW/OW)** — Rashid et al. 2020, arXiv:2006.10800 (NeurIPS 2020). Up-weights better joint actions. *Breach:* cheap upgrade if QMIX plateaus on coordination-heavy maps.
- **QPLEX** — Wang et al. 2020, arXiv:2008.01062 (ICLR 2021). Dueling/advantage IGM; complete IGM class, strong offline. *Breach:* most expressive value-decomposition; good for offline datasets (clean under determinism).

### Actor-critic / policy-gradient family
- **COMA** — Foerster et al. 2017, arXiv:1705.08926 (AAAI 2018). Counterfactual baseline marginalizing one agent's action. *Breach:* answers "did THIS marine's grenade matter?" under shared team reward.
- **MADDPG** — Lowe et al. 2017, arXiv:1706.02275 (NIPS 2017). Per-agent centralized critics; mixed coop/competitive; continuous actions. *Breach:* "mixed coop-competitive" = team-vs-team; per-agent critic supports heterogeneous kits.
- **MAPPO** — Yu et al. 2021, arXiv:2103.01955 (NeurIPS 2022 D&B). PPO + centralized value function; matches/beats value-decomposition on SMAC/MPE/Hanabi/GRF. *Breach:* **top practical pick** — on-policy PPO thrives on cheap massively-parallel rollouts.
- **IPPO** — de Witt et al. 2020, arXiv:2011.09533. Independent PPO, no centralized critic; startlingly competitive on SMAC. *Breach:* cheapest baseline + crucial ablation — if IPPO ties MAPPO, task doesn't yet need centralized coordination.
- **HATRPO/HAPPO → HARL** — Kuba et al. 2021, arXiv:2109.11251 (ICLR 2022); Zhong et al. 2024 (JMLR 25), arXiv:2304.09870. Multi-agent advantage decomposition + sequential updates → monotonic joint improvement WITHOUT parameter sharing. *Breach:* **most theoretically-grounded fit for heterogeneous kits.**
- **MAT (Multi-Agent Transformer)** — Wen et al. 2022, arXiv:2205.14953 (NeurIPS 2022). Encoder-decoder maps joint obs → actions autoregressively; HAPPO guarantee, linear in agents. *Breach:* autoregressive decoder emits coordinated orders for 5–10 marines in sequence.

---

## 2. Benchmarks that match Breach

- **SMAC** — Samvelyan et al. 2019, arXiv:1902.04043. 14 decentralized micromanagement scenarios. *Breach:* closest existing analog; study its reward shaping (damage, kills, win bonus). Became too easy → SMACv2.
- **SMACv2** — Ellis et al. 2022, arXiv:2212.07489 (NeurIPS 2023 D&B). An open-loop policy conditioned only on timestep wins many original-SMAC maps → SMAC lacked stochasticity to REQUIRE reactive policies. Fix: procedural generation (random team comps, positions, types). *Breach:* **read before designing the scenario generator.** Procedural map/spawn randomization is a REQUIREMENT for genuine closed-loop tactics.
- **JaxMARL + SMAX** — Rutherford et al. 2023, arXiv:2311.10090. GPU-native; up to ~12,500× wall-clock; SMAX = pure-GPU SMAC without SC2. *Breach:* closest philosophical match to Breach's thesis; align env API so JaxMARL algos can port.
- **Gigastep** — Lechner et al. 2023, openreview UgPAaEugH3 (NeurIPS 2023 D&B). Up to 1e9 steps/s; team combat, coop+adversarial. *Breach:* team-vs-team tractable at extreme throughput.
- **MAgent/MAgent2** — Zheng et al. 2017, arXiv:1712.00600 (AAAI 2018); magent2.farama.org. Up to ~1e6 agents; Battle scenario. *Breach:* less direct (5–10 units) but emergent flanking/encirclement reference; PettingZoo-API exemplar.
- **Google Research Football** — Kurach et al. 2019, arXiv:1907.11180 (AAAI 2020). Physics 3D football; sparse+shaped rewards; long-horizon credit. *Breach:* best analog for objective-based team play with sparse terminal reward (bomb-plant/round-win); its checkpoint-reward study is directly applicable.

**What micro teaches:** focus-fire emerges from kill/damage reward + shared target info; kiting needs genuine closed-loop policies (why SMACv2 stochasticity matters); positioning/cover emerges from partial observability + terrain in observation.

---

## 3. Full-game RTS / MOBA

- **AlphaStar** — Vinyals et al. 2019, Nature 575, doi:10.1038/s41586-019-1724-z. (1) Architecture: Transformer entity-encoder + **scatter connections** (entity embedding → placed at its tile in a map plane) + LSTM core + **autoregressive pointer-network** action head. (2) League training: main agents + main exploiters + league exploiters + prioritized fictitious self-play. *Breach:* near-blueprint — Breach's stacked feature planes = AlphaStar's spatial input; per-unit HP/footprints = entity set; scatter-connection directly reusable; league = arena reference.
- **OpenAI Five (Dota 2)** — Berner et al. 2019, arXiv:1912.06680. Five-hero team, ~20k-step horizons; distributed PPO + shared LSTM per hero + **"team spirit"** scalar interpolating selfish→team reward. *Breach:* most direct 5-unit precedent; PPO scales to team play given parallel experience; team-spirit annealing schedule is a concrete credit-assignment recipe (simpler than counterfactual baselines).

---

## 4. Heterogeneous agents & special-ability kits

Breach's marines have distinct action sets — hardest structural requirement (naive parameter-sharing assumes identical agents).
- **HARL/HAPPO/HATRPO** (§1) — principled answer, no parameter-sharing assumption. **Primary recommendation for true heterogeneity.**
- **ASN (Action Semantics Network)** — Wang et al. 2019, arXiv:1907.11461 (AAMAS 2020). Separates actions affecting self vs other entities. *Breach:* directly addresses grenade-at-tile vs move-to-tile vs fire-at-enemy semantics.
- **UPDeT** — Hu et al. 2021, arXiv:2101.08001 (ICLR 2021 spotlight). Transformer matching obs entities to action groups; transfers across obs/action configs; ~10× faster transfer on SMAC. *Breach:* one net serves all marine types + transfers across squad compositions.
- **ROMA** — Wang et al. 2020, arXiv:2003.08039 (ICML 2020). Emergent roles. *Breach:* roles ≈ emergent tactical assignments (breacher/anchor/support).
- **RODE** — Wang et al. 2020, arXiv:2010.01523 (ICLR 2021). Clusters actions by effect into role action-spaces; slow role-selector + fast role-policies. SOTA on SMAC at release. *Breach:* action-effect clustering fits Breach's order types; two-timescale hierarchy matches decoupled decision cadence.
- **PettingZoo** — Terry et al. 2020, arXiv:2009.14471 (NeurIPS 2021). Standard multi-agent API; supports heterogeneous obs/action spaces. *Breach:* Breach's facade is philosophically a PettingZoo parallel env — conform/mirror for ecosystem compatibility.

---

## 5. Credit assignment — the central hard problem

With one shared team reward: (1) value decomposition (implicit); (2) counterfactual baselines (COMA, explicit); (3) reward shaping / team-spirit annealing (OpenAI Five; GRF checkpoint rewards). Remains open/active (2024–26): multi-level advantage (arXiv:2508.06836), influence-scope (arXiv:2505.08630), game-theoretic/Shapley-core (arXiv:2506.04265), **open-agent-system credit** where team composition changes (arXiv:2510.27659 — relevant when marines die mid-round). *Breach:* start with team-spirit annealing (cheapest, proven); keep COMA as principled fallback; note marine deaths → open-team problem, unsolved.

---

## 6. Self-play, league & population for team-vs-team

- **NFSP** — Heinrich & Silver 2016, arXiv:1603.01121. Approximate Nash from self-play in imperfect-info games. *Breach:* grounding for why naive self-play cycles.
- **PSRO** — Lanctot et al. 2017, arXiv:1711.00832 (NeurIPS 2017). Best-response to a meta-distribution over the policy population; survey arXiv:2403.02227. *Breach:* principled recipe for the team-vs-team arena → robust non-exploitable tactics.
- **AlphaStar league** (§3) — production-scale realization. *Breach:* concrete blueprint; exploiters force robustness, avoid degenerate metagames.

---

## 7. Real-time / partial observability at fixed tick rate
- Decoupled cadence (act every N ticks) is standard action-repeat/frame-skip; also stabilizes credit assignment by lengthening effective horizon.
- Partial observability → recurrence (RNN/LSTM/Transformer memory); a single frame is non-Markovian. Budget for recurrent or history-stacked core.
- Determinism is a genuine asset: bit-stable replay buffers, offline datasets, league checkpoints; exact replay of emergent tactics for debugging.

---

## Recommended starting points for Breach (ranked)
1. **MAPPO (+ IPPO ablation)** — arXiv:2103.01955 / 2011.09533. Best match for a deterministic GPU sim; strongest documented coop baseline. Run IPPO alongside as the "do we even need centralized coordination yet?" test. *Start here.*
2. **QMIX** — arXiv:1803.11485. Canonical value-based CTDE baseline; hypernetwork injects global planes. Upgrade → QPLEX / Weighted-QMIX if it plateaus.
3. **HARL/HAPPO** — arXiv:2304.09870 / 2109.11251. Once heterogeneous kits are the bottleneck.
4. **AlphaStar-style architecture** — entity-encoder + scatter connections into [C,H,W] planes + recurrent core + autoregressive order head (cf. MAT arXiv:2205.14953). Orthogonal to the learning algorithm.
5. **AlphaStar-league / PSRO** — arXiv:1711.00832. For the arena once single-squad training works.
6. **Study SMAX/JaxMARL + SMACv2 before finalizing the env** — arXiv:2311.10090 + 2212.07489.

## Key open problems / cautions
- **Open-loop degeneracy (SMACv2 trap):** fixed layouts/spawns → memorized scripts. Procedural randomization is a requirement.
- **Credit assignment under sparse team reward is unsolved;** marine deaths → open-team problem.
- **Self-play cycles, not converges;** budget for population/league.
- **Heterogeneity breaks parameter-sharing;** commit to HARL or ASN/UPDeT structuring.
- **Partial observability demands memory.**
- **Value-decomposition expressiveness ceilings** (VDN/QMIX monotonicity).
- **Benchmark→Breach transfer is unproven;** re-tune everything.

## Reference list
VDN https://arxiv.org/abs/1706.05296 · QMIX https://arxiv.org/abs/1803.11485 · QTRAN https://arxiv.org/abs/1905.05408 · Weighted-QMIX https://arxiv.org/abs/2006.10800 · QPLEX https://arxiv.org/abs/2008.01062 · COMA https://arxiv.org/abs/1705.08926 · MADDPG https://arxiv.org/abs/1706.02275 · MAPPO https://arxiv.org/abs/2103.01955 · IPPO https://arxiv.org/abs/2011.09533 · HAPPO https://arxiv.org/abs/2109.11251 · HARL https://arxiv.org/abs/2304.09870 · MAT https://arxiv.org/abs/2205.14953 · SMAC https://arxiv.org/abs/1902.04043 · SMACv2 https://arxiv.org/abs/2212.07489 · JaxMARL/SMAX https://arxiv.org/abs/2311.10090 · Gigastep https://openreview.net/forum?id=UgPAaEugH3 · MAgent https://arxiv.org/abs/1712.00600 · GRF https://arxiv.org/abs/1907.11180 · AlphaStar https://www.nature.com/articles/s41586-019-1724-z · OpenAI Five https://arxiv.org/abs/1912.06680 · ASN https://arxiv.org/abs/1907.11461 · UPDeT https://arxiv.org/abs/2101.08001 · ROMA https://arxiv.org/abs/2003.08039 · RODE https://arxiv.org/abs/2010.01523 · PettingZoo https://arxiv.org/abs/2009.14471 · NFSP https://arxiv.org/abs/1603.01121 · PSRO https://arxiv.org/abs/1711.00832 · PSRO-survey https://arxiv.org/abs/2403.02227 · credit-assignment-recent https://arxiv.org/abs/2508.06836 https://arxiv.org/abs/2505.08630 https://arxiv.org/abs/2506.04265 https://arxiv.org/abs/2510.27659
