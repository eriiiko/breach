# Hierarchical RL & Commander↔Subordinate Architectures — Literature Digest for Breach

*Raw research-agent output, 2026-07-20. Design question this feeds: where should the abstraction boundary between a commander and its units sit? Model 1 = monolithic commander emits concrete low-level tile orders, units are dumb executors. Model 3+2 = commander emits goal-level orders (attack X / protect Y / take tile), each unit RL-trained (goal-conditioned) to achieve it — the "Battlefield commander gives an objective, unit figures out how" flavor.*

Recurring axis: **manager→worker channel abstraction** — how coarse/fine the commander's signal is, how much autonomy the unit keeps. That axis *is* the Model-1↔Model-3+2 spectrum.

## 1. Classic Hierarchical RL (single-agent foundations)
- **Feudal RL** — Dayan & Hinton 1993, NIPS. Manager issues commands to a slave; rewards it for satisfying commands. Founding principles: goals set top-down; goal-setting decoupled from goal-achievement (reward hiding + information hiding — manager says *what*, not *how*). *Breach:* conceptual charter for Model 3+2 and the Battlefield-objective idea. (Pre-arXiv; cite via Semantic Scholar.)
- **Options framework** — Sutton, Precup & Singh 1999, *Artificial Intelligence* 112:181–211. Option = (initiation set, intra-option policy, termination). Options over an MDP form a semi-MDP. *Breach:* formalism for "an order is a temporally-extended behavior with a termination condition"; directly models "act every N ticks" (commander decides at option boundaries, unit acts every tick).
- **MAXQ** — Dietterich 1999/2000, JAIR, arXiv:cs/9905014. Recursive value-function decomposition; safe state abstraction conditions. *Breach:* theory for why decomposed commander/unit value functions can be learned safely, and which state abstractions each level may ignore.
- **Option-Critic** — Bacon, Harb & Precup 2017, AAAI (arXiv:1609.05140 — *verify ID*). Learns intra-option policies AND termination end-to-end, no hand-specified subgoals. *Breach:* the "let the boundary emerge" extreme; caution: options often collapse without regularization (risk if Breach wants interpretable orders).
- **h-DQN** — Kulkarni et al. 2016, NIPS, arXiv:1604.06057. Meta-controller picks a subgoal (entity/location); controller acts to reach it via intrinsic reward; cracked Montezuma's Revenge. *Breach:* cleanest template for Model 3+2 with hand-specified goal semantics — commander picks a subgoal tile/entity, marine gets intrinsic reward for achieving it. Explicit, interpretable, matches "objective on screen."
- **FeUdal Networks (FuN)** — Vezhnevets et al. 2017, ICML, arXiv:1703.01161. Manager sets goals as directions in a learned latent space at low temporal resolution; Worker rewarded by cosine similarity to that direction. *Breach:* seminal "abstract (latent) order" design and the reference for the *bandwidth* question. Known limitation: latent directions can be too coarse for agility-demanding control.
- **HIRO** — Nachum et al. 2018, NeurIPS, arXiv:1805.08296. Higher level emits raw state-space goals (desired relative position); lower level intrinsically rewarded to reach them; off-policy (TD3) with an **off-policy correction** re-labeling stale high-level actions. *Breach:* most practical recipe for concrete human-legible subgoals (target position = "move to tile X"). The off-policy correction is the key trick for the non-stationarity you WILL hit.
- **Probabilistic Subgoal Representations** 2024, arXiv:2406.16707; **Multi-Resolution Skills** 2025, arXiv:2505.21410. *Breach:* multi-resolution skills let one commander mix coarse ("hold sector") and fine ("move to tile") orders → a way to SWEEP the Model-1↔Model-3 spectrum in one architecture.

## 2. Hierarchical MULTI-agent RL (commander + subordinates)
- **Feudal Multi-Agent Hierarchies (FMH)** — Ahilan & Dayan 2019, arXiv:1901.08492. One manager learns to communicate subgoals to multiple workers; workers rewarded for achieving them. Beats shared-reward baselines on sparse-reward tasks — *given an adequate subgoal set*. *Breach:* **single most on-point paper** — essentially Model 3+2 built and benchmarked. Lessons: (a) payoff largest in sparse-reward coordination (Breach's regime); (b) performance hinges on the ORDER VOCABULARY you expose.
- **RODE** — Wang et al. 2021, ICLR, arXiv:2010.01523. Role selector assigns each agent a role every c steps; role restricts action space (actions clustered by effect). Beats SOTA on 10/14 SMAC maps. *Breach:* "role = a restricted action space chosen by a higher level" is a concrete middle ground — commander hands each marine a role that bounds its tactics (assault/cover/flank). Very implementable given Breach's discrete abilities.
- **ROMA** — Wang et al. 2020, ICML, arXiv:2003.08039. Roles emerge from a stochastic role-embedding space. *Breach:* "let the order vocabulary emerge" at squad level; serves the emergent-strategy thesis but harder to interpret than RODE — later ablation.
- **HMASD / HSD** — NeurIPS 2023 (openreview xMgO04HDOS); earlier HSD arXiv:1912.03558. Transformer high-level policy assigns team + individual skills; low-level independent Q-learning. *Breach:* team-level skills (joint maneuvers) + individual skills from one hierarchy — for squad-wide orders ("breach room together").
- **TAG** — Paolo et al. 2025, arXiv:2502.15425. Hierarchies of arbitrary depth via a LevelEnv abstraction (each level is the environment for the level above); fully decentralized. *Breach:* cleanest recent engineering pattern for stacking commander-over-units; "each level is the env for the one above" maps onto Breach's Gymnasium facade.
- **Subgoal-based HRL for Multi-Agent Collaboration** 2024 arXiv:2408.11416; **Hierarchical Message-Passing Policies** 2025 arXiv:2507.23604; **LAGMA (Latent Goal-guided MARL)** 2024 arXiv:2405.19998. *Breach:* menu of 2024–25 refinements; LAGMA steers workers toward a latent goal (abstract-order end).
- **A Taxonomy of Hierarchical Multi-Agent Systems** 2025 survey, arXiv:2508.12683. Distinguishes top-down info flow (global planner disseminates instructions — Model 1) from other patterns. *Breach:* design-space map for the arc's design doc.

## 3. Goal-Conditioned RL (unit "achieves the order it's given")
- **UVFA** — Schaul et al. 2015, ICML, PMLR v37. V(s,g)/Q(s,a,g) — one network conditioned on state AND goal. *Breach:* formal basis for a single marine policy that accepts "the order" g and generalizes across orders.
- **HER (Hindsight Experience Replay)** — Andrychowicz et al. 2017, NeurIPS, arXiv:1707.01495. Relabels failed trajectories with the goal actually achieved. *Breach:* near-mandatory for sparse "achieve your order" rewards.
- **GCRL survey / awesome-gcrl** — github.com/GongXudong/awesome-gcrl. *Breach:* living index for the unit-policy learner.

## 4. Communication / order-passing
- **RIAL/DIAL** — Foerster et al. 2016, NeurIPS, arXiv:1605.06676. RIAL = comms as discrete actions; DIAL = differentiable channel. *Breach:* two regimes for the commander→unit link; DIAL if the order is a LEARNED signal (defer; use fixed symbol first).
- **CommNet** — Sukhbaatar et al. 2016, arXiv:1605.07736. Continuous broadcast-and-average comms. *Breach:* baseline for learned unit↔unit coordination.
- **TarMAC** — Das et al. 2019, ICML, arXiv:1810.11187. Targeted signature-based soft-attention messaging. *Breach:* if the commander should address specific marines ("you three, flank").
- **IC3Net** — Singh et al. 2019, ICLR, arXiv:1812.09755. Gated comms + individualized rewards; learns WHEN to communicate. *Breach:* gating + per-unit credit fixes "shared reward drowns individual signal."
- **BiCNet** — Peng et al. 2017, arXiv:1703.10069. Bi-directional RNN coordination backbone; human-level StarCraft combat. *Breach:* flat (no-commander) coordination baseline — the control condition proving a commander earns its keep.
- Substrate: **QMIX** arXiv:1803.11485 (CTDE), **MADDPG** arXiv:1706.02275 — standard low-level learners under a commander.

## 5. Instruction / command following
- **NL Instruction-Following w/ task-related language** — NeurIPS 2023. *Breach:* you almost certainly do NOT want natural language in v1 — a small symbolic order vocabulary is the tractable analog of "objective on screen."
- **Compositional Instruction Following w/ LMs+RL** 2025 arXiv:2501.12539; **Feudal RL by Reading Manuals** 2021 arXiv:2110.06477; **Semantically Aligned Task Decomposition in MARL** 2023 arXiv:2305.10865. *Breach:* "Reading Manuals" ≈ objective-on-screen; task-decomposition = splitting a squad objective into per-marine orders.

## 6. Battlefield / RTS command hierarchies in RL
- **Mastering the Digital Art of War (Wargaming via HRL)** — Scotty Black 2024, arXiv:2408.13333. HRL aligned with military decision-making structures. *Breach:* same genre/motivation; pull the PDF for its strategy/tactics split.
- **Enhancing Aerial Combat Tactics via Hierarchical MARL** 2025 arXiv:2505.08995; **Air Combat Maneuvering HRL** — Selmonaj et al. 2023 (IDSIA). Heterogeneous low-level per-unit policies + high-level commander macro-commands. *Breach:* concrete recent Model-3+2 instantiations; note failure mode — units struggle to balance obedience vs local tactical optimization.
- **Hierarchical control of a MARL team in RTS games** — 2022, Expert Systems with Applications.
- LLM-as-commander line (context): HIMA 2025 arXiv:2508.06042; LLM-PySC2 arXiv:2411.05348; LEHCA (Nature Sci Reports, *verify DOI*) — LLM Commander emits sub-goals + reward-shaping + action masks over QMIX low-level agents. *Breach:* shows commander shaping low-level RL via sub-goals + reward shaping + action masking — three grounding mechanisms Breach could borrow WITHOUT an LLM.

## Recommended starting points for Breach
1. **FMH** (arXiv:1901.08492) — read first; Model 3+2 built and benchmarked; payoff largest in sparse-reward coordination conditional on a good subgoal vocabulary.
2. **HIRO + HER** (arXiv:1805.08296 + 1707.01495) — machinery for a first working experiment; interpretable move-to-tile/take-objective orders; off-policy correction fixes non-stationarity; HER makes sparse per-unit reward learnable.
3. **RODE** (arXiv:2010.01523) — pragmatic middle: commander assigns a role that bounds the unit's action set; interpretable/debuggable.
4. **FeUdal Networks** (arXiv:1703.01161) — reference for the abstract-order extreme and bandwidth trade-off; likely an ablation.
5. **TAG** (arXiv:2502.15425) + **UVFA** — clean engineering pattern to stack depth later.
6. Domain reality-check: arXiv:2408.13333, 2505.08995, Selmonaj 2023.

**Where to put the split for the FIRST experiment (recommendation):** Start at the **RODE/HIRO band, not the extremes.** A commander that, every N ticks, emits a small discrete order per marine from a fixed human-legible vocabulary — {move-to-tile g, attack-entity e, cover-tile g, hold, take-objective g} — where the tile/entity ARGUMENT is concrete but the HOW (pathing, ability use, target selection) is left to a goal-conditioned unit policy trained with HER + an off-policy learner (HIRO-style correction). This is deliberately BETWEEN Model 1 and pure Model 3+2:
- Preserves the Battlefield-objective feel (unit works out execution itself — no scripted pathfinding).
- Fixed symbolic order channel avoids the separable hard problem of LEARNING a comms code (defer DIAL/TarMAC).
- Gives clean ablation knobs to sweep the whole spectrum in one codebase: shrink argument granularity → toward Model 1; widen to abstract objectives + more latent autonomy → toward pure Model 3+2. **The abstraction boundary becomes an experimental variable, not an architectural commitment.**
- Keep the flat BiCNet/QMIX squad (no commander) as the control.

## Key open problems / cautions
- **Non-stationarity of the manager→worker interface** (HIRO's motivating problem) — plan for off-policy correction or a staged/frozen-worker curriculum from day one.
- **Obedience vs local optimization** (military-HRL papers) — a tuning dial, not solved.
- **Subgoal-vocabulary design dominates outcomes** (FMH) — a first-class research variable.
- **Emergent-boundary methods trade interpretability for autonomy** (Option-Critic, ROMA) — interpretable hand-specified orders (h-DQN/HIRO/RODE) are the safer first rung.
- **Bandwidth of the channel is a real dial** (FuN's lossy latent direction).
- **Credit assignment across levels + units** — budget per-unit intrinsic reward shaping (IC3Net/FMH).
- **Determinism constraint (Breach):** prefer discrete symbolic orders; any order signal feeding back into sim-affecting logic must be quantized deterministically (neural policies are inference-layer / render-exempt-like).
- **Real-time cadence:** the "act every N ticks" parameter is the option-termination / manager-horizon knob.

## Reference list
Feudal RL (Dayan & Hinton 1993) semanticscholar · Options (Sutton/Precup/Singh 1999) https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf · MAXQ https://arxiv.org/abs/cs/9905014 · Option-Critic https://ojs.aaai.org/index.php/AAAI/article/view/10916 (arXiv:1609.05140 verify) · h-DQN https://arxiv.org/abs/1604.06057 · FeUdal Networks https://arxiv.org/abs/1703.01161 · HIRO https://arxiv.org/abs/1805.08296 · Probabilistic Subgoals https://arxiv.org/abs/2406.16707 · Multi-Resolution Skills https://arxiv.org/abs/2505.21410 · FMH https://arxiv.org/abs/1901.08492 · RODE https://arxiv.org/abs/2010.01523 · ROMA https://arxiv.org/abs/2003.08039 · HMASD openreview xMgO04HDOS / HSD https://arxiv.org/abs/1912.03558 · TAG https://arxiv.org/abs/2502.15425 · Subgoal-HRL https://arxiv.org/abs/2408.11416 · Hier Message-Passing https://arxiv.org/abs/2507.23604 · LAGMA https://arxiv.org/abs/2405.19998 · HMAS taxonomy https://arxiv.org/abs/2508.12683 · UVFA https://proceedings.mlr.press/v37/schaul15.pdf · HER https://arxiv.org/abs/1707.01495 · DIAL/RIAL https://arxiv.org/abs/1605.06676 · CommNet https://arxiv.org/abs/1605.07736 · TarMAC https://arxiv.org/abs/1810.11187 · IC3Net https://arxiv.org/abs/1812.09755 · BiCNet https://arxiv.org/abs/1703.10069 · QMIX https://arxiv.org/abs/1803.11485 · MADDPG https://arxiv.org/abs/1706.02275 · Compositional Instruction https://arxiv.org/abs/2501.12539 · Reading Manuals https://arxiv.org/abs/2110.06477 · Semantic Task Decomp https://arxiv.org/abs/2305.10865 · Wargaming HRL https://arxiv.org/abs/2408.13333 · Aerial Combat HRL https://arxiv.org/abs/2505.08995 · Air Combat Selmonaj https://ipg.idsia.ch/preprints/selmonaj2023a.pdf · HIMA https://arxiv.org/abs/2508.06042 · LLM-PySC2 https://arxiv.org/abs/2411.05348 · AI+Wargaming survey https://arxiv.org/abs/2009.08922

**Uncertainty flags:** Option-Critic arXiv:1609.05140 from memory — verify. LEHCA Nature Sci Reports DOI/date unconfirmed. Other arXiv IDs returned by search and cross-checked against titles/authors.
