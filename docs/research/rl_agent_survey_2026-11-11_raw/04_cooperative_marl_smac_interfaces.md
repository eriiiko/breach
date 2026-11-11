# 04 — Cooperative MARL on StarCraft micromanagement: the per-agent interface

Research digest for the Breach RL soldier design, 2026-11-11. Scope: SMAC and
its successors, read at the level of the OBSERVATION and ACTION interface (the
algorithm families are in `rl_litsearch_2026-07-20_raw/01_marl_squad_combat.md`
and are not repeated). Tags: [READ src] = read in that source this session;
[RECALL] = from memory, not checked this session.

Vocabulary used below:
- **Decentralised execution / centralised training (CTDE)**: each agent acts on
  its own local observation, but during training a value function (the
  "critic", which estimates how good a situation is) may see more — e.g. the
  whole map.
- **Action mask / availability mask**: a 0/1 vector the env hands the agent each
  step; actions marked 0 get probability zero (their logits are set to minus
  infinity before the softmax).
- **Parameter sharing**: all agents run the same network weights; they differ
  only by their inputs (often including an agent-id or unit-type one-hot).

---

## 1. SMAC (Samvelyan et al. 2019, arXiv:1902.04043; oxwhirl/smac code)

### (a) Observation
- Each agent sees units "within the sight range"; sight range is 9 for all
  agents, shooting range 6 [READ paper]. "Having a larger sight range than a
  shooting range forces agents to make use of the move commands" [READ paper].
- Per unit inside sight range: distance, relative x, relative y, health, shield,
  unit_type [READ paper]. In the code [READ smac `starcraft2.py`, via a summary
  of the source, not a verbatim quote]:
  - **move features**: 4 flags, can I move N/S/E/W; optionally 8 pathing-grid
    values and 9 terrain-height values (`obs_pathing_grid`, `obs_terrain_height`,
    both off by default — so by default the agent sees NO map).
  - **enemy features**, one fixed-size block per enemy, in a fixed slot order:
    `[attackable?, distance, rel_x, rel_y, health, shield (Protoss), unit-type
    one-hot]`; the block is all zeros if the enemy is dead or outside sight.
    Note the first entry: "can I attack this enemy now" is IN the observation.
  - **ally features**, one block per other ally: `[visible?, distance, rel_x,
    rel_y, health, shield, unit-type one-hot, (last action one-hot if
    obs_last_action)]`.
  - **own features**: health, shield, unit-type one-hot. (No own position: the
    whole observation is egocentric, relative to self.)
  - optional timestep number.
  - A dead agent's observation is all zeros.
- Distances and positions are normalised by sight range; health by max health
  [RECALL].
- Defaults: `obs_all_health=True`, `obs_own_health=True`,
  **`obs_last_action=False`**, `state_last_action=True`,
  `obs_instead_of_state=False` [READ code summary]. The paper says the
  experiments excluded the allies' last actions from the agent observation
  ("except for the last actions of the allied units") [READ paper].

### (b) Actions
- Discrete: `no-op`, `stop`, `move[N/S/E/W]`, `attack[enemy_id]` (Medivacs:
  `heal[agent_id]` instead) [READ paper]. In code: actions 0..5 are no-op, stop,
  4 moves; action 6+k = attack enemy k (heal ally k for healers)
  [READ code summary]. So the action vector length is `6 + n_enemies`.
- Mask: attack k is available only if enemy k is alive and within SHOOTING
  range (6); a move only if the cell is pathable; stop always for the living;
  a dead agent has only no-op [READ code summary + paper].
- `attack[k]` hands targeting to the game: the SC2 engine walks into range,
  turns and fires; the agent never aims. There is no facing, no projectile
  travel the agent controls, no friendly fire.
- Decision rate: agents act every 8 game frames (`step_mul=8`) [RECALL; SMAC
  README].
- Enemy slot order: fixed for the episode (enemies indexed once at episode
  start, sorted by the engine's unit tags) [RECALL]. Slot k is the same enemy
  for the whole episode, which is what makes "attack k" learnable: the
  observation block k and the action 6+k refer to the same unit.

### (c) Who knows what
- Agents see only within sight range; an ally outside sight range is zeroed
  with `visible=0`. No communication channel. The allies' last actions are
  optional and were off in the paper's experiments [READ paper].
- Centralised critic / mixer gets the **global state**: all allies
  `[health, cooldown or energy, rel_x, rel_y (to map centre), shield, unit
  type]`, all enemies `[health, rel_x, rel_y, shield, unit type]`, plus all
  agents' last actions (on by default) [READ paper + code summary]. Note what
  the state LACKS: it is not agent-specific and contains no per-agent view —
  MAPPO's paper (below) identifies this as a weakness.

### (d) Reported costs / gains
- Reward: shaped = hit-point damage dealt + kill bonus + win bonus; sparse
  option +1 / -1 [READ paper].
- (Ablations of the observation fields are in MAPPO, §2.)

---

## 2. MAPPO (Yu et al. 2021/2022, arXiv:2103.01955) and IPPO (de Witt et al. 2020, arXiv:2011.09533)

MAPPO = PPO (a policy-gradient method that clips how far each update may move
the policy) with a centralised critic and shared weights. It changes the
interface around the SMAC env, not the env itself.

### (a)+(c) Observation, and what the critic sees
- The policy keeps SMAC's local observation; a dead agent's local
  observation is "all zeros except for the agent's ID" [READ paper appendix] —
  i.e. the agent-id one-hot is part of the policy input.
- Four critic inputs compared [READ paper]:
  - **CL**, the concatenation of all local observations: grows with agent
    count, can miss what no agent sees; "ineffective" on SMAC, especially with
    many agents.
  - **EP**, the env-provided global state (SMAC's `get_state`): omits
    agent-specific information (agent id, available actions); worse on hard
    maps.
  - **AS**, agent-specific: EP concatenated with agent i's local observation.
  - **FP**, feature-pruned AS: AS with duplicated features removed. The critic
    state includes "agent-specific information, such as available actions and
    relative distances to other agents" [READ paper appendix].
  - Numbers: on 3s5z_vs_3s6z, MAPPO(FP) 84.4 % median win rate vs MAPPO(AS)
    63.3 % (Table 5) [READ paper]. The other CL/EP comparisons are figures only
    [READ paper — no table].
- **Death masking**: once agent a is dead its critic input becomes a zero
  vector with a's id appended, instead of the still-informative global state;
  the authors argue post-death states are rare and would shift the critic's
  input distribution. It beat both keeping the real global state and
  dropping post-death samples, with lower value loss on most maps; keeping the
  id matters on maps with roles (3s5z_vs_3s6z) [READ paper; figures only].

### The five recommendations [READ paper, verbatim fragments]
1. "Utilize value normalization to stabilize value learning" (running mean/std
   of returns; "never hurts training and often improves the final performance
   ... significantly").
2. Critic input: "include both local, agent-specific features and global
   features", without needlessly growing the dimension.
3. Data reuse: at most 10 epochs on difficult tasks, 15 on easy; avoid
   splitting the batch into mini-batches.
4. PPO clip ratio below 0.2, then tune.
5. "Utilize a large batch size".

### (d) Parameter sharing [READ paper, Table 4, median win %]
| Map | shared | independent weights |
|---|---|---|
| MMM2 | 90.6 | 13.0 |
| 6h_vs_8z | 88.3 | 11.4 |
| 5m_vs_6m | 89.1 | 44.4 |
| 3s5z | 96.9 | 80.4 |
| corridor | 100.0 | 82.2 |
Note MMM2 is heterogeneous (marines, marauders, medivacs) and sharing still
won by a wide margin — with the unit-type one-hot and agent id in the input.

### IPPO
- Independent PPO, each agent's critic sees only its own observation; on
  SMAC it was competitive with QMIX/MAPPO on the hard maps they tested
  [ABS + the earlier digest; RECALL for details]. MAPPO's own table shows IPPO
  84.4 % on 6h_vs_8z vs MAPPO(FP) 88.3 % [READ MAPPO paper].
- Reading for Breach: the centralised critic buys a few points on SMAC, not
  an order of magnitude; the policy input design matters more.

---

## 3. SMACv2 (Ellis et al. 2022/2023, arXiv:2212.07489; oxwhirl/smacv2 code)

### What changed [READ paper + code]
- **Procedural teams**: unit types drawn per episode from three types per race
  at fixed weights (`team_gen: weighted_teams`); **random start positions**,
  "reflect" (each side spawns on its half) or "surround" (allies in the
  centre, enemies around) [READ paper + README].
- **True unit ranges** (`use_unit_ranges=True`): sight and shoot ranges per
  unit type instead of SMAC's flat 9 / 6 — e.g. shoot: stalker 6, marine 5,
  medivac 4, melee floored to 2; sight: stalker 10, zergling 8, medivac 11
  [READ code].
- New own features: own position normalised by map size (`obs_own_pos`) and,
  with the cone on, own view direction [READ code].
- **Extended partial observability (EPO)**: when the first agent spots an
  enemy, each other agent gets that enemy in its observation only with
  probability `prob_obs_enemy` (paper recommends p = 0: only the spotter
  sees it); the available-actions mask is removed in this mode [READ paper +
  code]. This is SMACv2's nearest thing to fog of war with an information
  bottleneck between teammates.
- **Conic field of view**: present in the code as an option, `conic_fov`,
  DEFAULT OFF [READ code]. With it on: 12 extra discrete actions, each SETS
  the facing to one of 12 absolute directions (30° apart; not turn-left /
  turn-right); the cone is 30° wide (2π/12) and applies to both observing
  (sight range) and attacking (shoot range); units start facing east; own
  facing (`own_fov_x/y`) enters the observation. Action count becomes
  18 + n_enemies. The published paper text (v2, Oct 2023) as read does not
  mention a cone [READ paper]; I recall it was in the v1 (Dec 2022) feature
  list and dropped from the benchmark [RECALL, uncertain]. Either way, no
  published SMACv2 result uses it as far as I read.
- **Index order**: allies are sorted by (unit type, x, y) at episode start;
  enemies get indices in the order the engine lists them; indices are then
  fixed through the episode by unit tag, and a dead enemy keeps its slot with
  health 0 [READ code]. So "attack k" in SMACv2 means "the enemy that got
  slot k at spawn", and slot k's unit type varies between episodes.

### The open-loop finding [READ paper]
- Policies given ONLY the timestep and agent id were trained on SMAC: they
  were "significantly worse" overall than closed-loop ones, but on many maps
  not much worse, and only four maps (3s5z_vs_3s6z, corridor, 6h_vs_8z,
  5m_vs_6m) defeated the open-loop policy entirely.
- A second probe regressed learned Q-values from masked observations: on SMAC
  masking everything still gave low error; on SMACv2 errors were higher —
  "SMACv2 is significantly more stochastic than SMAC".
- Lesson: a benchmark with fixed spawns can be "solved" by a memorised script.
  Breach's generated ships and seeded spawns are on the right side of this.

---

## 4. Tying "attack enemy k" to enemy k

The precedent: in SMAC, action 6+k and observation block k name the same
enemy for the whole episode. A flat MLP/GRU must LEARN that block k's
features are relevant to logit 6+k, separately for every k (nothing in the
weights says so). Entity-wise networks build that tie into the architecture.

### UPDeT (Hu et al. 2021, arXiv:2101.08001) [READ ar5iv full text]
- Observation split into entity tokens: own unit, each other ally, each
  enemy (3m example: "main agent, two other ally agents and three enemies"),
  embedded by one shared layer, then self-attention (a transformer).
- **Policy decoupling**: the action set is split into "action groups", each
  matched to one entity; the attack-k logit is read from the transformer's
  OUTPUT for enemy token k (a shared linear layer maps a token to its
  group's action count); move/stop/no-op are read from the own-unit token;
  ally tokens' outputs are discarded.
- Size changes: k = the map's maximum enemy count, dead enemies masked; "no
  restriction on policy dimension", so the same weights run 3m, 5m, 7m.
- Ablations (figures only, no table): a vanilla transformer with a flat
  output head "fails to beat the enemies"; an aggregation (pooled) head is
  worse than the GRU baseline; **breaking the entity–action pairing leaves
  performance "heavily decreased ... even worse than the GRU model"**; ~80 %
  gain over GRU on 4m_vs_5m; in transfer, timesteps to converge "at least 10
  times less than the GRU-based model and 100 times less than training from
  scratch".
- This is the key mechanism for Breach's "aim at enemy k".

### ASN, action semantics network (Wang et al. 2019/ICLR 2020, arXiv:1907.11461) [READ ar5iv]
- Splits actions into those affecting only self (moves) and those aimed at
  another unit j (attack j). Moves are scored from the own embedding; attack j
  is scored by a "pairwise interaction function" of the own embedding and an
  embedding of the observation of j (inner product given as the example).
- Sharing: one sub-module per target type ("all influencing agents can share
  one sub-module"; mixed ASN one per unit type).
- Numbers: on 15m without the hand-coded invalid-action rule, valid-action
  rate 71.9 % vs 44.3 % (vanilla QMIX); win rate about 80 % vs about 20 %
  (read off the text's figure description).
- Limitation stated: pairwise only; continuous actions are future work.

### REFIL (Iqbal et al. 2021, arXiv:2006.04222) [READ ar5iv]
- State = a SET of entities; each agent gets a binary **observability mask**
  over them ("agents can always observe themselves"); entity-wise layers apply
  "an identical linear transformation to all input entities"; attention
  scores for entities an agent cannot see are set to minus infinity. This is
  the clean way to put fog of war into an attention network: the same
  network, the mask decides what each agent may attend to.
- Targeting: each enemy carries a TAG, there is one attack action per
  possible tag, and agents see the tags of the enemies they observe; for
  UPDeT as a baseline the outputs "are shuffled by tag to maintain the
  entity-to-action mapping". The paper gives no formula for its own attack
  Q-values beyond this [READ].
- Training trick: each episode the entities are split at random into two
  groups and an auxiliary "imagined" value is learned from in-group and
  out-group utilities (weight λ = 0.5); it shapes learning, never acts.
- Multi-task SMAC with varying team composition: 3-8sz (39 tasks), 3-8csz and
  3-8MMM (66 tasks each); REFIL "consistently outperforms"; numbers in
  figures only; "~22 % fewer time steps" on 3-8MMM is mentioned [READ].

### HPN (Hao et al., ICLR 2023, arXiv:2203.05285) [READ html, partly]
- Names the problem: a fixed-order concatenation of m entities has m!
  orderings carrying the same information; exploiting that shrinks the space
  by 1/m!.
- Input: a hypernetwork (a small network that OUTPUTS the weights of another)
  turns each entity's features into that entity's embedding weights; the
  embeddings are summed → order-invariant.
- Output: "permutation equivariant" — the weights producing the per-entity
  action outputs are themselves generated from that entity's features, so
  "the input order change will result in the same output order change".
  Claimed "100% win rates in almost all hard and super-hard scenarios"
  (abstract); adding the equivariant output "could significantly boost" a
  set-based QMIX (curves only).
- Transfer 5m→12m etc.: better initialisation, curves only.

### What breaks in a flat vector when sizes change [synthesis of the above]
- A flat MLP has one weight row per slot. A new team size changes the input
  and output length (UPDeT: the GRU baseline "needed its output layer
  reinitialized for each new scenario").
- Even at fixed size, slot k's meaning depends on the env's ordering rule
  (SMACv2: spawn order; allies sorted by type and position). If the ordering
  is arbitrary, the network must learn the same skill k times.
- The architectures above (UPDeT, ASN, HPN, REFIL) all make one weight set
  serve every slot, and read the attack-k score FROM enemy k's own
  embedding. UPDeT's unpaired ablation is the direct evidence that the tie is
  what makes it work.

---

## 5. Sequence-model MARL

### MAT, Multi-Agent Transformer (Wen et al. 2022, arXiv:2205.14953) [READ html, partly]
- Encoder: one token per agent = that agent's local observation, "in arbitrary
  order"; outputs one latent per agent and (in training) a value per agent.
- Decoder: actions generated autoregressively — agent m conditions on the
  encodings of ALL agents and on the actions already chosen by agents
  1..m-1; the order is a random permutation each iteration; in training all
  actions are computed in parallel from stored actions.
- Execution: the decoder conditions on everyone's observations, so MAT needs
  the whole team's observations at decision time — centralised execution
  (the paper does not say so in words; it follows from π(a^{i_m} | ô^{i_1:n},
  a^{i_1:m-1}) [READ]). MAT-Dec, a decentralised-actor variant, was added
  for fair comparison.
- SMAC (Table 1) MAT vs MAPPO: 5m_vs_6m 90.6 vs 88.2; 8m_vs_9m 100 vs 93.8;
  3s5z 100 vs 72.5; 27m_vs_30m 100 vs 93.1; 6h_vs_8z 98.8 vs 88.4 [READ].
  (MAPPO's own paper reports 96.9 on 3s5z, so baselines differ between papers.)

### Sable (Mahjoub et al. 2024/ICML 2025, arXiv:2410.01706) [READ html, partly]
- Retention (RetNet) instead of attention, over agents AND time: it "can
  process entire episode sequences as stateful memory" — a recurrent memory
  of the episode in the same model that orders agents.
- "a centralised algorithm" — the encoder sees all agents' observations each
  timestep. Scales to >1000 agents with linear memory.
- Best on 34 of 45 tasks over six environments (RWARE, LBF, MABrax, SMAX,
  Connector, MPE). On SMAX (aggregate): Sable ~0.94, IPPO ~0.93, MAT ~0.92,
  MAPPO ~0.86 — IPPO and MAT inside Sable's confidence interval [READ].
- SMAX observation as described there: "Each agent observes all allies and
  enemies within its field of view, including itself": position, health,
  unit type, weapon cooldown, previous action [READ].

### State of the art 2025–2026 (what I could establish)
- No single SMACv2 leaderboard exists [search, 2026-11-11]. The strongest
  published broad-benchmark claim I read is Sable's; its honest SMAX gap over
  IPPO is ~1 point. MACTAS (2025, attention-based communication) claims SOTA
  on some SMACv2 maps [ABS, search snippet only]. COMPASS (2025) uses LLMs
  with a skill library, 57 % on protoss 5v5 vs QMIX 27 % [ABS, search
  snippet; not a pure-MARL result].
- Superseded: MAT's per-step autoregressive decoding is generalised by Sable
  (memory over time). Value-decomposition methods (QMIX family) lose to PPO
  methods on SMAX [READ JaxMARL]. The flat-vector MLP/GRU interface is
  superseded for variable-size teams by entity-token / equivariant heads
  (UPDeT 2021, HPN 2023; REFIL's masking).
- Contradiction to note: JaxMARL reports MAPPO "significantly"
  outperforming IPPO on SMAX; Sable's runs (Mava) put IPPO above MAPPO
  [READ both]. Implementation details decide this; treat IPPO-vs-MAPPO as
  an experiment to run, not a settled fact.

---

## 6. SMAX (JaxMARL, Rutherford et al. 2023, arXiv:2311.10090; code)

- Units are discs in continuous 2D, no terrain, no obstacles, no line of
  sight: visibility is distance against sight range only [READ code].
  Heuristic enemy AI decentralised. All SMAC v1 scenarios + three
  SMACv2-like ones [READ paper].
- Discrete actions: 4 moves + stop + one attack per enemy (index 5+k);
  attack k available if both alive and the target in attack range; the mask
  does NOT check cooldown or sight [READ code].
- **Continuous action option** (`action_type="continuous"`, 4 values in
  [0,1]): `do_shoot` (probability of shooting rather than moving),
  `shoot_last_enemy` (probability of keeping the last target), and a polar
  point (r, θ). Moving goes along r·(cos θ, sin θ); shooting hits the
  opponent NEAREST to that point offset from the shooter [READ code]. A
  precedent for "aim at a point and let the env snap to the nearest
  target" — the soft alternative to an index.
- Observation (`unit_list`): per other unit 13 values — health, relative x/y
  (÷ sight range), last movement x/y, last target, weapon cooldown, 6-way
  unit-type one-hot; allies first, then enemies; self excluded; own block of
  10 values (health, position ÷ map size, cooldown, type). 127 floats for
  5v5. A zeroed slot = not visible or dead. The enemies' last actions are
  visible by default (`see_enemy_actions=True`) [READ code].
- A `conic` observation option bins units into 32 angular sectors around the
  agent, up to 2 units each (842 floats): an egocentric polar layout, not a
  limited view cone [READ code].
- Critic world state: every unit's features with absolute positions, team
  ids and unit types, no visibility limit [READ code].
- Speed: "40,000x faster" with many vectorised runs on one 2080; a single
  setup up to 31x [READ paper].

---

## 7. Heterogeneous agents: one network or several

- **Full sharing + agent id / unit type** (the SMAC/MAPPO default): the
  default that works. MAPPO Table 4 (above) shows sharing winning hugely even
  on heterogeneous MMM2 [READ].
- **HARL / HAPPO** (Zhong et al. 2023, arXiv:2304.09870; Kuba et al. 2021,
  arXiv:2109.11251): separate networks per agent, updated one at a time in a
  random order, each update accounting for the ones before it — which gives
  a monotonic-improvement guarantee. Argues full sharing "could lead to an
  exponentially-suboptimal outcome in the extreme case", that agent ids
  still leave interference "when they have different abilities and goals",
  and that padding to a common input size "cannot be generally applied".
  Claims comparable or better SMACv2 win rates "while not relying on the
  parameter-sharing trick" [READ intro; numbers not read].
- **SePS, selective parameter sharing** (Christianos et al. 2021,
  arXiv:2102.07475): partition agents "based on their abilities and goals"
  and share within each partition; sharing everything "can also have a
  detrimental effect" [READ abstract; mechanism details RECALL: an
  autoencoder of each agent's transitions, clustered before training].
- **Kaleidoscope** (Li, Pan, Zhang, NeurIPS 2024, arXiv:2410.08540): one
  common weight set, a learnable mask per agent selecting which weights it
  uses, a regulariser pushing the masks apart; on SMACv2 built on QMIX;
  baselines NoPS, FuPS, FuPS+ID, SePS, multi-head, SNP; "FuPS + ID generally
  outperforms NoPS"; SMACv2 numbers in figures only [READ html, partly].
- **HyperMARL** (Tessera, Rahman, Albrecht, NeurIPS 2025, arXiv:2412.04233):
  a hypernetwork conditioned on the agent generates its actor and critic
  weights; finds that sharing's loss of diversity comes from cross-agent
  gradient interference, "exacerbated by the common practice of coupling
  agent IDs with observations"; competitive with six baselines over 22
  scenarios up to 30 agents [ABS via search snippets].
- Reading: for many unit CLASSES whose observation and action sets differ
  (Breach's long term), the entity-token network handles different input
  sets naturally; the action heads must differ per class (a robot with no
  grenades has no grenade head). The 2024–2025 work (Kaleidoscope,
  HyperMARL) is about letting same-shaped agents specialise; it is not about
  different action sets.

---

## 8. Where the precedent stops (what SMAC/SMACv2/SMAX do not have)

- **No aiming.** `attack[k]` hands walking-into-range, turning and firing to
  the game engine. There is no facing with inertia, no spread, no projectile
  the agent steers, no miss from turning too slowly. The one facing in the
  family is SMACv2's optional cone, set in one step to one of 12 absolute
  directions (no inertia), default off.
- **No walls / line of sight.** SMAC has terrain only through optional
  pathing and height features (off by default); SMAX has none. Visibility is
  a radius. Cover, doors and corridors are absent.
- **No friendly fire, no area weapons, no grenades.** Every attack is a
  single-target index; no action affects a region or allies.
- **No physics fields** (smoke that blocks sight, fire, pressure).
- **Fixed episodes of one fight**, small teams (2–27), one enemy team; no
  zombie third party; no respawn (same as Breach).
- **Decision rate**: one decision every 8 frames [RECALL]; Breach's 24 Hz
  continuous control with a decimation step is closer to SMAX's step.

---

## 9. For Breach

Tied to the eight design questions. Opinions are marked (OPINION).

**Q1 Representation.** Take: entity tokens (own unit, each teammate, each
enemy/zombie known) with one shared embedding per entity class, self-attention,
and an observability mask on the attention (REFIL) rather than zeroed slots.
Keep a spatial channel for the map: SMAC's default observation has NO map, and
SMACv2/SMAX have no walls, so the benchmarks give no evidence on how to
encode walls and fields — Breach's local field-plane crop has no precedent
here (look to the AlphaStar/OpenAI Five digests). Egocentric relative
positions (SMAC normalises by sight range) are the shared convention.
(OPINION) A flat fixed-slot vector would work for one fixed squad size but
breaks the moment the roster or the unit mix changes.

**Q2 "Aim at enemy k".** It is a good precedent ONLY under three conditions,
all present in SMAC: (i) k names the same enemy for the whole episode (SMAC's
index-by-tag; Breach needs a stable id → slot map for enemies, including
zombies that appear, and a dead slot kept masked); (ii) an availability mask
(alive, in sight / remembered); (iii) the logit for k is computed from enemy
k's own embedding (UPDeT, ASN, HPN), not from a flat output row — UPDeT's
ablation that broke the pairing fell below a GRU baseline. Where SMAC stops:
its attack k also delegates turning, approach and firing. In Breach "aim at
k" must be a TARGET handed to the turning law (which already accepts targets:
`turning.set_target`), with the trigger separate, so the agent learns when the
turn has caught up (the bots already read `facing_after_turn_q16`). For the
no-enemy case, SMACv2's `conic_fov` gives the shape the author leans to:
a small set of absolute directions (12 there) as a discrete "face direction d"
action, not turn-left/right. SMAX's continuous option (aim at a point, snap
to the nearest opponent) is an alternative that unifies both cases.
(OPINION) A pointer head over tokens with extra "direction" tokens (fixed
direction slots, or a remembered last-seen tile as a token) unifies "enemy
k", "last-seen position of k" and "a direction" into one head.

**Q3 Factoring.** SMAC is ONE flat categorical (no-op, stop, 4 moves, attack
k) — move and attack are exclusive. Breach's verbs are concurrent (walk while
aiming while firing), so the precedent does NOT transfer: a factored action
(independent heads: move dir + throttle, aim target, trigger, verb flags,
grenade kind) is needed; masks per head as in SMAC. One action per grenade
kind fits the precedent's style (each discrete action = one effect), and
ASN's split "acts on self" vs "acts on an entity" is the natural grouping:
moves/reload/flashlight from the own token, aim/melee/shove target from the
entity tokens, a throw TARGET either an entity or a point. (OPINION)

**Q4 Team display.** Evidence is thin and mixed: SMAC's allies' last actions
are optional and were OFF in the SMAC paper's runs; SMAX shows last movement
and last target of visible allies by default; SMACv2's EPO (only the spotter
sees an enemy) deliberately REMOVES shared sight to make the task harder,
which says shared sight is a large aid. No ablation I read isolates "ally
last action" or "ally target" in the observation. (OPINION) Put the bots'
team picture in as tokens: teammates with health, their current target id
(the pairing makes "ally j targets enemy k" expressible as attention between
tokens), and remembered enemies with last-seen tile and age. That is
SMACv2's EPO turned around — shared memory instead of none.

**Q5 Order input.** Nothing in SMAC/MAPPO conditions a soldier on an order;
roles there are emergent or via agent id. (OPINION) Reserve an objective
token (a tile / area entity with a type) from v1, fed by the bots' plays at
first: it costs one token and keeps the commander layer an add-on. Not
evidenced by this literature.

**Q6 Fog / FOV.** Take: the training-time critic sees everything (MAPPO's
agent-specific global state: global + the agent's own view, and death
masking: a dead soldier's critic input zeroed except its id). The policy sees
its own sight plus whatever the team display shares. A cone: only SMACv2's
unused option (30°, set facing) exists as precedent; no result on it. With
Breach's facing inertia a cone makes facing a scouting decision too — an
experiment, not a precedent. (OPINION) start 360° within line of sight, add
the cone as a switch.

**Q7 Heterogeneous units.** The evidence favours ONE shared network with
type information as long as the observation and action sets coincide
(MAPPO Table 4: MMM2 90.6 % shared vs 13.0 % separate). For different action
sets, give each class its own heads over a shared entity trunk; per-class
specialisation methods (SePS, Kaleidoscope, HyperMARL) are the refinement, and
HyperMARL warns that the agent-id input itself causes gradient interference.

**Q8 State of the art.** Superseded: flat fixed-slot interfaces (by entity
tokens with equivariant action heads), QMIX-family value factorisation for
this kind of task (PPO methods beat it on SMAX). Current: Sable (2024/25) —
but it is centralised at execution and its margin over a tuned IPPO on SMAX is
~1 point. (OPINION) For Breach, where friendly units must act on their own
view and the "team display" is the designed information channel, a
decentralised entity-attention policy with a centralised agent-specific
critic (MAPPO-style, plus IPPO as the ablation) is the state of the art that
fits; MAT/Sable become relevant only if the commander layer issues joint
actions.

**What contradicts the author's leanings.** Nothing contradicts "aim at
enemy k"; but SMAC's version is easier than Breach's (the engine does the
aiming), so its success does not prove the Breach version; the concurrent
verbs break SMAC's one-categorical design. The cone has no evidence base.

**Open.** How to encode walls / fields next to entity tokens (not in this
literature); how a remembered (not seen) enemy should look to the target
head; whether a direction head and a target head should be one pointer or
two heads; the order input.

---

## 10. Source ledger

| Citation | URL | Status | Supports |
|---|---|---|---|
| Samvelyan et al. 2019, SMAC | https://arxiv.org/html/1902.04043 | PART | obs features, sight 9 / shoot 6, actions, state, last actions off, reward |
| oxwhirl/smac `starcraft2.py` | https://github.com/oxwhirl/smac | PART (via summary of source) | feature layout, defaults, mask, state |
| SMAC README (step_mul 8) | https://github.com/oxwhirl/smac | RECALL | decision every 8 frames |
| Yu et al. 2022, MAPPO | https://arxiv.org/html/2103.01955 | PART | 5 recommendations, CL/EP/AS/FP, death masking, sharing Table 4, FP vs AS |
| de Witt et al. 2020, IPPO | https://arxiv.org/abs/2011.09533 | ABS | IPPO ≥ joint methods on SMAC |
| Ellis et al. 2023, SMACv2 | https://arxiv.org/html/2212.07489 | PART | procedural teams/starts, unit ranges, EPO, open-loop finding |
| oxwhirl/smacv2 code + README | https://github.com/oxwhirl/smacv2 | PART | conic_fov (12 set-direction actions, 30°, default off), unit ranges, index order, prob_obs_enemy |
| SMACv2 v1 cone in paper | https://arxiv.org/abs/2212.07489v1 | RECALL (unverified) | cone in v1 feature list |
| Hu et al. 2021, UPDeT | https://ar5iv.labs.arxiv.org/html/2101.08001 | FULL (via summary) | entity tokens, policy decoupling, unpaired ablation, 10x transfer |
| Wang et al. 2020, ASN | https://ar5iv.labs.arxiv.org/html/1907.11461 | PART | self vs entity actions, pairwise scoring, 15m numbers |
| Iqbal et al. 2021, REFIL | https://ar5iv.labs.arxiv.org/html/2006.04222 | PART | entity sets, observability mask in attention, tags, multi-task SMAC |
| Hao et al. 2023, HPN | https://arxiv.org/html/2203.05285v2 | PART | permutation invariant input / equivariant output, m! argument |
| Wen et al. 2022, MAT | https://arxiv.org/html/2205.14953v3 | PART | encoder/decoder, random order, centralised execution, Table 1 |
| Mahjoub et al. 2024, Sable | https://arxiv.org/html/2410.01706 | PART | retention over agents+time, centralised, 34/45, SMAX numbers |
| Rutherford et al. 2023, JaxMARL/SMAX | https://arxiv.org/html/2311.10090 | PART | SMAX design, speed, PPO > QMIX, MAPPO > IPPO there |
| JaxMARL `smax_env.py` | https://github.com/FLAIROx/JaxMARL | PART (via summary of source) | obs layout, continuous point-aim action, conic sectors, mask, world state |
| Zhong et al. 2023, HARL | https://arxiv.org/html/2304.09870 | PART (intro/method) | sequential update, critique of sharing |
| Kuba et al. 2021, HAPPO/HATRPO | https://arxiv.org/abs/2109.11251 | RECALL | origin of HAPPO |
| Christianos et al. 2021, SePS | https://arxiv.org/abs/2102.07475 | ABS | selective sharing by abilities/goals |
| Li, Pan, Zhang 2024, Kaleidoscope | https://arxiv.org/html/2410.08540v1 | PART | learnable masks; FuPS+ID > NoPS; SMACv2 on QMIX |
| Tessera et al. 2025, HyperMARL | https://arxiv.org/abs/2412.04233 | ABS (search snippets) | agent-conditioned hypernetworks, id-coupling interference |
| MACTAS 2025; COMPASS 2025 | (search results only) | ABS | 2025 SMACv2 claims |
| ICML 2026 slide on cross-attention routing, SMACv2 transfer | https://icml.cc/media/icml-2026/Slides/78769.pdf | NF (PDF unreadable) | claim that MLP heads memorise slot patterns — unverified |

