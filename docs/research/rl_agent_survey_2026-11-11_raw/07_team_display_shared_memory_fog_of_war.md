# 07 — Team display, shared enemy memory, fog of war and field of view

*Raw research digest, 2026-11-11, for the Breach RL soldier design. Topic: what a
soldier should know about its team and about hidden enemies, and how that
knowledge is represented. Tags: [READ] = read in a primary source this session
(quote or close paraphrase); [RECALL] = from memory, not checked this session.*

Two terms used throughout. The POLICY (or actor) is the network that picks actions
and, in Breach, runs in every soldier at play time. The VALUE FUNCTION (or critic)
is a second network used only during training to estimate how good a situation is;
it is thrown away at play time, so it may see things the soldier cannot. "Centralised
training, decentralised execution" (CTDE) names exactly that split.

**Headline findings.**
1. Every landmark team system that had fog of war SHARED VISION BY RULE (Dota's game
   rule in OpenAI Five; the human-legible "allies' predicted destination" in Honor of
   Kings) rather than leaving each agent to learn to communicate it; and SMACv2's
   experiment shows what happens without sharing: with enemy sight not shared, MAPPO
   and QMIX "struggle to attain even modest performance", shared they come close to
   solving the task.
2. OpenAI Five already used Erik's "team picture" idea for enemies: fogged enemy
   heroes keep their last-seen observation plus a "time since last seen" feature.
3. A privileged, all-seeing value function is standard and cheap, and in
   hide-and-seek it was decisive (a masked one never got past stage 3). Feed it the
   global state PLUS the agent's own view (MAPPO's finding), not the concatenation of
   all views.
4. Learned message channels (CommNet, TarMAC ...) have no independent evidence of
   beating a strong no-communication baseline on combat tasks, and lose to QMIX on
   some SMAC maps in a 2024 author's table; they are also hard to interpret and
   unstable to train. No paper I found compares a hand-designed team display head to
   head with learned messages — that comparison is OPEN.
5. No source I found measures a frontal cone against 360° vision. Starting with more
   information and removing it later has direct evidence (Suphx's oracle guiding;
   "learning by cheating"), with a known trap (the imitation gap).
6. Friendly fire: none of the landmark systems had it (Dota, SMAC, GRF, hide-and-seek
   have no team damage). The two small precedents handle it with a reward penalty;
   neither shows a representation that made it learnable. Breach is near new ground.

## A. What team-mates knew of each other in the landmark systems

### Hide-and-seek (Baker et al. 2019, arXiv:1909.07528)
- [READ] "Agents observe the position, velocity, and size ... of objects and other
  agents." "If entities are not in line-of-sight of the agent or not in a 135 degree
  cone in front of the agent, then they are masked out in the policy." Plus 30 lidar
  range sensors around the agent. The text does not exempt team-mates from the mask.
- [READ] "At optimization time, we use a centralized omniscient value function for
  each agent, which has access to the full environment state." Ablation: "training
  with a masked value function never progressed past stage 3 (ramp usage)" at the
  same compute.
- [READ] Self-attention over entities "increases sample efficiency" against
  embed-and-pool only, slightly slower in wall clock; no number in the text.

### OpenAI Five (Berner et al. 2019, arXiv:1912.06680)
- [READ] Vision is shared by the game's rule: "Because visible information and fog
  of war ... are shared across a team in Dota 2, the observations are nearly
  identical for each hero" (a few derived features depend on which hero is
  controlled, plus a one-hot "which of the five heroes is being controlled").
- [READ] Five replicas of one policy with identical weights, each with its own
  LSTM hidden state. A small learned channel exists: "we maxpool the first 25% of
  the vector across the five replica networks" (an implicit, learned message).
- [READ] Per-unit features (189 units a tick, allies included) carry attack state:
  "currently attacking?", "time since last attack", "am I attacking this unit?",
  "is this unit attacking me?", "last 16 timesteps' hit points", "current
  animation", incoming projectile ETA. Per hero: "is currently alive?", "time since
  this hero last seen", "is using ability?", mana, cooldowns; "current gold (allies
  only)". So an ally's current activity (animation, attacking or not) is visible,
  but I found no feature naming WHICH unit an ally is targeting — only whether a
  unit attacks *me* or *I* attack it. Not stated in the source either way.
- [READ] The last-seen rule is hand-built: "Units in fog of war are not observed."
  For fogged enemy heroes "we reuse the observation from the last time step when the
  unit was visible", with a "time since this hero last seen" feature. This is
  exactly a last-known-entity-with-age display, shared by team vision.
- [READ] Value head shares the LSTM with the policy (no privileged critic). The paper
  notes AlphaStar's "value network observed full information about the game state
  (including observations hidden from the policy)" and calls that "a promising
  direction for future work" for Dota.
- [READ] "Team spirit" = how much of each hero's reward is shared with team-mates
  (1 = split equally); annealed 0.3 → 0.8 in training.

### AlphaStar (Vinyals et al. 2019, Nature)
- [READ via OpenAI Five's account] value network saw information hidden from the
  policy. [RECALL] One policy controls all units of the player, so "team-mates" are
  just units in its own entity list; the policy saw only what the player's units
  saw (fog of war) through a camera-limited view. No team display question arises.

### SMAC and MAPPO (Samvelyan et al. 2019, arXiv:1902.04043; Yu et al. 2021, arXiv:2103.01955)
- [READ] SMAC: "agents receive local observations drawn within their field of view"
  (a circle of sight range, no cone): per unit "distance, relative x, relative y,
  health, shield, and unit_type" for allies and enemies in range. "In addition,
  agents have access to the last actions of allied units that are in the field of
  view" — but the paper's own experiments EXCLUDED those last actions (appendix C.2).
  Shooting range 6 < sight range so "agents [must] make use of the move commands
  before starting to fire". "The global state ... is only available to agents during
  centralised training."
- [READ] Focus fire: SMAC names it "one important technique ... ordering units to
  jointly attack and kill enemy units"; the built-in heuristic AI does "a basic form
  of focus-firing". The SMAC paper does NOT say how learned agents acquire it. My
  reading (opinion): in SMAC focus fire is learned with no target display at all,
  because every ally sees the same enemies' health in a shared sight circle, an
  enemy's dropping health is a visible common signal, and the reward pays damage and
  kills. [RECALL] Several SMAC papers show replays of learned focus fire; I did not
  re-read one this session.
- [READ] MAPPO studied what the training-time value function should see:
  concatenated local observations (CL) "is ineffective" (too high-dimensional);
  the environment's global state alone (EP) misses agent-specific detail and was
  worse on harder maps; the best was an AGENT-SPECIFIC global state (global state
  plus this agent's own local observation, "AS"), optionally with duplicates pruned
  ("FP"): e.g. 3s5z_vs_3s6z FP 84.4 vs AS 63.3 median win %. Advice: "include both
  local, agent-specific features and global features in the value function input".
  Death masking: give a dead agent's value input as zeros plus its agent id —
  "significantly outperforms" the alternatives.

### Honor of Kings (Wu et al. 2018 arXiv:1812.07887; Ye et al. 2020 arXiv:2011.12692, 1912.09729)
- [READ] The hierarchical macro strategy model (1812.07887) predicts each hero's
  "attention" (where on the map it will go). "Imitated cross-agent communication":
  in training each hero's input holds its allies' attention LABELS; at test time the
  allies' PREDICTED attention. "We put attention prediction of allies as features and
  make decision correspondingly." Ablation: full model beat the no-communication
  version 62.5 % to 37.5 %; expert-rated reasonable openings 22 % → 83 %. Note this
  model was SUPERVISED on human games, not RL — the evidence is that a hand-chosen,
  legible intention (a destination) shared between team-mates helps, not that RL
  learns to use it.
- [RECALL] The later full-game RL agents (2011.12692) control each hero with its own
  model, team vision shared by the game's rule; I did not re-read their observation
  tables this session.

### Google Research Football / TiZero (Lin et al. 2023, arXiv:2302.07515)
- [READ] Actor observation in six parts including "teammate information" and
  "opponent information"; the critic is "a single centralized value network" on a
  five-part global state (ball, ball holder, teammates, opponents, match). No
  communication channel, no ablation of one, and no feature for team-mates' actions
  or intentions; coordination came from the shared policy with player id, joint
  policy optimisation and a passing reward. GRF itself is full information.

### Capture the flag, FTW (Jaderberg et al. 2019, arXiv:1807.01281)
- [RECALL] Pixels only (first-person view, 84x84 RGB), each agent with its own
  recurrent memory, no team display and no communication; a teammate is only what
  the camera shows. Behaviours like following a team-mate emerged. [READ abstract]
  "using only pixels and game points as input"; the abstract says nothing of
  communication.

### What part A adds up to
- Allies' STATE (position, health, alive, ability or animation in use) was always in
  the observation when visible, usually as entities in a list the network reads by
  attention or pooling. Allies' INTENTION was given explicitly only in Honor of Kings
  (predicted destination, supervised) and optionally in SMAC (last action, switched
  off in the paper's own runs). OpenAI Five showed an ally's current activity, not
  its target. Nothing I read gave "whom is my team-mate aiming at" as a feature.
- The training-time value function saw more in most systems (AlphaStar,
  hide-and-seek, SMAC/MAPPO, TiZero); OpenAI Five is the exception.

## B. Sharing intentions and targets

- **Last-action inputs.** SMAC offers allies' last actions within sight [READ], but
  its benchmark runs left them out [READ]. [Search summary of a 2018 review] DDRQN's
  success was attributed partly to last-action inputs (with weight sharing). For
  Breach, a last action at 6 Hz is a weak intention signal: it says "fired", not "at
  whom".
- **Sequential (autoregressive) team action selection.** The Multi-Agent Transformer
  (Wen et al. 2022, arXiv:2205.14953) [READ abstract] turns "the joint policy search
  problem into a sequential decision making process" and beats MAPPO and HAPPO on
  SMAC, MuJoCo and GRF. [RECALL] Its decoder picks agent 1's action, then agent 2's
  given agent 1's, and so on: team-mates see each other's chosen action before
  choosing. Cost: one central network at play time and an ordering of the squad. It
  would directly solve "two soldiers pick the same target or step into each other's
  lane in the same tick", but it breaks the one-policy-per-soldier plan unless the
  squad is run as one network.
- **Intention sharing** (Kim, Park, Sung, ICLR 2021) [ABS via search]: each agent
  rolls out an imagined future trajectory with learned models of the world and of the
  others, encodes it with attention, and sends it as its message; reported better than
  MADDPG, DIAL, TarMAC and ATOC on particle tasks (predator-prey, cooperative
  navigation, traffic junction). Small tasks only; it needs a learned world model.
- **Learned communication** (CommNet 2016, IC3Net 2018, ATOC 2018 [RECALL]; TarMAC,
  Das et al., ICML 2019, arXiv:1810.11187 [READ abstract]: learns "whom to address"
  from task reward alone, on grid shapes, a traffic junction, 3D indoor scenes).
  - Evidence of value in combat is weak. In CommFormer's own table (arXiv:2405.08550,
    [READ]), on three hard SMAC maps TarMAC scored 39.1 / 44.2 / 40.1 % and MAGIC
    22.3 / 0.0 / 5.8 %, against the NO-communication QMIX at 51.1 / 82.6 / 51.4 %
    (CommFormer itself 96.9 / 100 / 96.9 %, the graph method QGNN 93.8 / 92.2 /
    98.0). On the standard maps the no-communication MAPPO already reaches 82–100 %
    on most. Caveat: an author's table, baselines run by the proposer.
  - Pitfall: Lowe et al. 2019 (arXiv:1903.05168) [READ abstract] found "a scenario
    where agents appear to communicate" yet "the messages do not impact the
    environment or other agent in any way", and that intuitive metrics "can be
    misleading". A learned channel can be decorative.
  - The 2026 "Five Ws" survey (Chen et al., arXiv:2602.11583) [PART]: "This dynamic
    context makes it difficult to learn stable communication strategies"; "Unlike
    hand-crafted protocols with explicit semantics, learned messages lack
    transparency"; and with centralised learning "communication during execution is
    not needed to supply missing global context". No head-to-head experiment of a
    hand-designed display against learned messages in what I read.
- **Hand-designed against learned.** The best positive evidence for a FIXED, legible
  team signal is Honor of Kings' allies'-destination feature (62.5 vs 37.5 % win,
  openings 22 → 83 % reasonable) [READ], but in a supervised model. The best
  negative evidence for learned messages is above. No direct comparison found: OPEN.
  Opinion: a fixed display has practical advantages for Breach regardless — no extra
  thing to learn, inspectable in a replay, and the bots and a human HUD can share its
  definition.
- **Learned common picture.** arXiv:2411.05683 (2024) [ABS]: agents encode their
  perceptions into vectors decoded into a "common operational picture" of all
  friendly and enemy agents in StarCraft II, "less than 5% error", trained jointly
  with the policies. A middle road: a learned encoding supervised to rebuild a
  legible picture.
- **Focus fire.** [READ] SMAC defines it and its heuristic opponent does it; SMAC does
  not explain how learners get it. SMAC had no display of team targets: every agent
  sees the same enemies' health within its sight range, and the reward pays damage
  and kills. Evidence that sharing sight matters for target choice: SMACv2's EPO
  (part C) — "This requires agents to implicitly communicate enemy information to
  prioritise targets", and without sharing they failed. OpenAI Five [READ] gave per
  unit "am I attacking this unit?", "is this unit attacking me?" and 16 ticks of hit
  points, so a unit being focused is readable from its falling health.
- **Friendly fire.** No landmark system had team damage. Small precedents: (i) air
  combat (arXiv:2505.08995, [READ]): a −2 "Friendly Punishment" on both shooter and
  victim gave only "a slight reduction" of friendly events and "a notable increase in
  draws" (more caution); agents observed the nearest friendly aircraft's position,
  speed and angles. (ii) Multi-robot laser tag (Li & Zhao 2026, arXiv:2603.11980,
  [READ]): a −10 "cross fire avoidance reward ... if ally robots enter the shooting
  zone"; the state teacher saw team-mates' relative positions, the camera student only
  an "ally" heatmap channel. Neither ablates a fire-lane feature. OPEN, and Breach's
  most specific need.

## C. Shared team memory of enemies ("last seen")

- **Hand-built last-known entity with age, shared by team vision**: OpenAI Five [READ]
  ("we reuse the observation from the last time step when the unit was visible" +
  "time since this hero last seen"). No ablation of it in what I read.
- **Recurrent memory only**: FTW (pixels and a recurrent core per agent; architecture
  [RECALL]); hide-and-seek (masked entities, an LSTM [RECALL]); SMAC baselines (a GRU
  per agent [RECALL]). The memory must learn that a hidden enemy still exists.
- **Hand memory against LSTM, measured**: StarCraft build-order selection (Gehring et
  al. 2018, arXiv:1811.08568, [READ table]). With an LSTM in both, "visible" features
  (units seen now) against "memory" features ("hard-coded rules to keep track of enemy
  units that were seen before but are currently hidden"): win rate against the
  training bots 0.878 vs 0.886; against two strong bots 0.635 vs 0.587 and 0.706 vs
  0.693. Small and mixed: the hand-built memory did NOT clearly help an LSTM agent. An
  auxiliary loss predicting the TRUE hidden counts (a privileged label, training
  only) also moved little. Caveat: one strategic choice per minute, not a gunfight.
- **Belief / defogging**: StarCraft defogger (Synnaeve et al. 2018, arXiv:1812.00054,
  [READ abstract]): conv + recurrent encoder-decoders predict the hidden state; [ABS
  via a summary page] they beat "previous seen" and "perfect memory" heuristics on
  prediction; used by a rule bot, "improvements in win rates against several strong
  community bots" (no number in the abstract).
- **Auxiliary features speed learning**: Lample & Chaplot 2016 (arXiv:1609.05521,
  [READ abstract]): a Doom agent trained to also predict "the presence of enemies or
  items" showed "dramatically improve[d] the training speed and performance". A cheap
  variant for Breach: predict the team picture as an auxiliary task.
- **Shared or private memory?** In OpenAI Five sharing is by rule (team vision), so
  the last-seen record is team-wide. SMACv2's extended partial observability (EPO,
  Ellis et al., arXiv:2212.07489, [READ]): only the first agent to spot an enemy is
  sure to see it, each other agent with probability p; "We suggest p=0". Result: "p=0
  struggles to attain even modest performance, whereas p=1 comes close to solving the
  task in a number of instances"; p=0.5 is about as bad as p=0 (no win-rate numbers in
  the text). The closest experiment to "team vision shared by rule against each agent
  on its own": sharing won by a wide margin, and implicit sharing was not learned.

## D. Fog of war and field of view

- **Who limited the policy's view, and how**
  - Hide-and-seek: line of sight AND a 135° frontal cone for entities, plus a 30-ray
    360° lidar for geometry; value function omniscient [READ]; a masked value function
    "never progressed past stage 3" at the same compute [READ].
  - SMAC: a sight circle (360°), sight range > shooting range [READ]. SMACv2 set
    StarCraft's ranges; I found no cone in the text I read [READ, 100 k of 122 k
    characters — unverified for the rest].
  - OpenAI Five: team fog of war by the game's rule, 360° [READ].
  - FTW: a first-person camera, so a frontal view by construction [RECALL].
  - Laser tag student: a 110° camera with an LSTM over the images; the teacher saw all
    relative positions [READ].
  - TABX battle simulator (arXiv:2602.01665) [ABS via search]: field-of-view masking,
    unseen units zeroed.
- **Asymmetric actor-critic** (Pinto et al. 2017, arXiv:1710.06542, [READ abstract]):
  "the critic is trained on full states while the actor ... gets rendered images",
  which "significantly improves performance". Caveat (Baisero & Amato 2022,
  arXiv:2105.11674, [READ abstract]): the common state-only critic has "fundamental
  issues" that can bias the policy under partial observability; their unbiased fix
  [RECALL] gives the critic the agent's history as well as the state — consistent
  with MAPPO's empirical advice (global state + the agent's own observation).
- **Cone against 360°**: no measurement found (two searches). Limited-view Atari
  studies [ABS via search] report degraded performance and learned view-shifting.
  So "a cone makes agents cover arcs and check corners" is PLAUSIBLE but UNMEASURED.
  Hide-and-seek's cone coexisted with emergent tactics but was never ablated.
- **Full information first, restricted later**
  - Suphx (Mahjong, Li et al. 2020, arXiv:2003.13590, [READ]): "oracle guiding" —
    train with all features including the hidden ones, then drop the perfect features
    with a Bernoulli mask whose keep-probability decays "from 1 to 0"; then continue
    at a tenth of the learning rate, rejecting samples with large importance weights
    (without these, continual training "is not stable"). Reported: the normal agent
    "improves much faster than standard RL training"; no number in the text. This is
    the recipe for "360° now, cone later" or "omniscient now, fog later".
  - Learning by cheating (Chen et al. 2019, arXiv:1912.12294, [READ abstract]): a
    driving agent that sees the ground-truth layout teaches a camera-only student;
    100 % on the original CARLA benchmark, infractions down an order of magnitude.
  - Legged robots (Lee et al. 2020, arXiv:2010.11251): privileged teacher,
    proprioceptive-history student [RECALL; the abstract does not describe it].
  - The trap, the "imitation gap" (Weihs et al., arXiv:2007.12173, [READ abstract]):
    when the teacher acts on information the student cannot have, it is "marginalized
    during imitation", with "potentially, poor results"; their fix mixes imitation and
    RL adaptively. For Breach: a teacher who sees through walls teaches a student to
    walk toward enemies it cannot know about. A measured gap in a team-shooting task:
    laser-tag teacher 48.6 vs student 38.4 Hit@15, collision rate 10 % vs 36 % [READ].

## For Breach

Breach already computes a team picture for its bots (seen enemies with last-seen tile
and tick, forgotten after `memory_s`; a focus target; a play; fire lanes). The
question is how much of it the learned soldier sees, and in what form.

### Candidate 1 — team vision by rule, raw (OpenAI Five style)
- Entities: self; every living team-mate (relative position, facing, health, alive,
  reloading / throwing / blinded, magazine state, last action); every enemy seen now
  by ANY team-mate (relative position, facing, health); every enemy seen earlier and
  hidden now, as its last-seen record with an AGE feature, dropped after `memory_s`.
  Read by attention over the entity list (hide-and-seek / OpenAI Five).
- No targets, plays or lanes: focus fire and lane discipline are learned from
  positions and reward.
- For: closest to the strongest evidence (OpenAI Five's design; SMACv2's p=1);
  nothing hand-designed beyond what the team sees, which is a game rule, not a tactic.
  Against: no precedent of friendly-fire avoidance learned from positions alone, and
  the bots' own history (more than half of deaths team kills) says it is hard.

### Candidate 2 — team vision plus legible intentions (Honor of Kings style)
- Candidate 1 plus, per team-mate: its current target (an index into the enemy list,
  or none), firing now or not, and its fire lane as geometry relative to ME (am I
  inside that lane; my distance to its line of fire). Per enemy: how many team-mates
  target it (a focus count) and its recent damage taken. Optionally the team play as
  a one-hot, only if a team layer exists to set it.
- For: the only positive evidence for a shared intention is a fixed, legible one
  (HoK); "am I in a lane" turns friendly-fire avoidance from inferring geometry into
  reading one bit; inspectable in replays and the same definition the bots and a HUD
  use. Against: targets are known only after team-mates decide, so two soldiers
  deciding in the same tick still collide (what MAT's ordering solves); risk of
  copying a bad focus convention. Opinion: the candidate to start with — the extra
  features are cheap, and dropping them is the ablation that shows whether they were
  needed.

### Candidate 3 — learned channel, or the picture as an auxiliary task
- Candidate 1 plus a small learned message (OpenAI Five's max-pool across replicas,
  or TarMAC-style attention), or an AUXILIARY task: each soldier predicts the team
  picture or hidden enemy positions (Lample & Chaplot; arXiv:2411.05683).
- For: no hand choice of which intentions matter. Against: learned messages in combat
  have weak to negative evidence (CommFormer's table, Lowe's pitfalls) and are not
  inspectable; the auxiliary version keeps a legible target. Opinion: a later
  experiment, not the start.

### The training-time value function (any candidate)
- The global state (every unit of every team, health, true positions of hidden
  enemies) PLUS the soldier's own observation; a dead soldier's input as zeros plus
  its id (MAPPO's agent-specific state and death masking; hide-and-seek's ablation;
  Baisero & Amato's caveat). Not the concatenation of all soldiers' views.

### Fog of war and field of view
- Start (opinion): 360° vision with walls blocking sight, sight shared by rule within
  the team (the team picture), last-seen records with age. This matches Erik's
  leaning and every landmark system except hide-and-seek.
- Later: if a cone is wanted (so that facing, covering arcs and checking corners
  matter), bring it in Suphx-style: mask out-of-cone entities with a probability that
  rises from 0 to 1 over training, then continue at a lower learning rate. Keep the
  shared team picture: per-soldier cones with shared vision make "who watches which
  arc" the team's problem, which is the behaviour a cone is meant to produce.
  Unmeasured: make the experiment able to fail (cone against 360° on held-out seeds).
- Do NOT start with a wall-seeing teacher distilled into a fogged student: the
  imitation gap predicts a student that walks toward enemies it cannot know about. A
  privileged CRITIC is the safe form of privilege.
- Contradiction worth noting: Erik's "360° to start" is well supported; but the
  literature does not say a cone would buy corner-checking — that is a hope.

### What remains open
- No head-to-head test of a hand-designed team display against learned messages.
- No measurement of a cone against 360° on behaviour or learning speed.
- No representation shown to make friendly-fire avoidance learnable; the two
  precedents use a reward penalty with modest effects.
- Same-tick target collisions: a display of current targets lags one decision.
- Whether an LSTM makes last-seen records redundant: the one measurement (StarCraft
  build orders) says nearly so, but for a slow, strategic decision.

## Source ledger

Status: FULL = full text read this session; PART = the relevant sections read; ABS =
abstract or a secondary account only; RECALL = from memory, not checked; NF = not found.

| Citation | URL | Status | Supports |
|---|---|---|---|
| Baker et al. 2019, Emergent Tool Use (hide-and-seek), arXiv:1909.07528 | https://ar5iv.labs.arxiv.org/html/1909.07528 | PART | 135° cone + LOS mask; omniscient critic; masked-critic ablation; attention vs pooling |
| Berner et al. 2019, OpenAI Five, arXiv:1912.06680 | https://ar5iv.labs.arxiv.org/html/1912.06680 | PART | shared team vision; last-seen reuse + age; per-unit attack features; 25 % max-pool; team spirit |
| Vinyals et al. 2019, AlphaStar, Nature | — | RECALL (+ via 1912.06680) | value net saw hidden info; one policy for all units |
| Samvelyan et al. 2019, SMAC, arXiv:1902.04043 | https://ar5iv.labs.arxiv.org/html/1902.04043 | PART | sight circle; ally last actions optional and unused; focus fire defined |
| Yu et al. 2021, MAPPO, arXiv:2103.01955 | https://ar5iv.labs.arxiv.org/html/2103.01955 | PART | critic inputs CL / EP / AS / FP; death masking |
| Ellis et al. 2022, SMACv2, arXiv:2212.07489 | https://arxiv.org/html/2212.07489 | PART | EPO: unshared sight p=0 fails, p=1 near-solves; no cone found |
| Wu et al. 2018, HMS (Honor of Kings), arXiv:1812.07887 | https://ar5iv.labs.arxiv.org/html/1812.07887 | PART | allies' predicted attention as input; 62.5/37.5; 22→83 % |
| Ye et al. 2020, HoK, arXiv:2011.12692 and 1912.09729 | — | RECALL | not re-read |
| Lin et al. 2023, TiZero, arXiv:2302.07515 | https://ar5iv.labs.arxiv.org/html/2302.07515 | PART | central critic; no communication; no intention features |
| Jaderberg et al. 2019, FTW capture the flag, arXiv:1807.01281 | https://arxiv.org/abs/1807.01281 | ABS | pixels and game points only |
| Wen et al. 2022, Multi-Agent Transformer, arXiv:2205.14953 | https://arxiv.org/abs/2205.14953 | ABS (decoder RECALL) | sequential joint action |
| Kim, Park, Sung 2021, Intention Sharing, ICLR | https://iclr.cc/virtual/2021/poster/3363 | ABS (search summary) | imagined-trajectory messages; particle tasks |
| Das et al. 2019, TarMAC, arXiv:1810.11187 | https://arxiv.org/abs/1810.11187 | ABS | targeted learned messages |
| CommNet (Sukhbaatar 2016), IC3Net (Singh 2018), ATOC (Jiang & Lu 2018) | — | RECALL | learned-communication lineage |
| CommFormer 2024, arXiv:2405.08550 | https://arxiv.org/html/2405.08550 | PART | TarMAC / MAGIC below no-comm QMIX on hard SMAC maps |
| Lowe et al. 2019, Pitfalls of measuring emergent communication, arXiv:1903.05168 | https://arxiv.org/abs/1903.05168 | ABS | decorative messages; misleading metrics |
| Chen et al. 2026, Five Ws of multi-agent communication, arXiv:2602.11583 | https://arxiv.org/html/2602.11583 | PART | unstable learned protocols; interpretability; no head-to-head |
| Distributed common operational picture via MARL 2024, arXiv:2411.05683 | https://arxiv.org/abs/2411.05683 | ABS | learned encoding of a legible team picture, <5 % error |
| Gehring et al. 2018, strategy selection in SC:BW, arXiv:1811.08568 | https://ar5iv.labs.arxiv.org/html/1811.08568 | PART | visible vs hand memory under an LSTM: small, mixed |
| Synnaeve et al. 2018, StarCraft defogger, arXiv:1812.00054 | https://arxiv.org/abs/1812.00054 | ABS | learned belief over hidden enemies |
| Lample & Chaplot 2016, Doom, arXiv:1609.05521 | https://arxiv.org/abs/1609.05521 | ABS | auxiliary "enemy present" prediction speeds learning |
| Pinto et al. 2017, Asymmetric actor-critic, arXiv:1710.06542 | https://arxiv.org/abs/1710.06542 | ABS | full-state critic, image actor |
| Baisero & Amato 2022, Unbiased asymmetric actor-critic, arXiv:2105.11674 | https://arxiv.org/abs/2105.11674 | ABS (fix RECALL) | state-only critic can bias the policy |
| Chen et al. 2019, Learning by Cheating, arXiv:1912.12294 | https://arxiv.org/abs/1912.12294 | ABS | privileged teacher → restricted student |
| Lee et al. 2020, quadrupedal locomotion, arXiv:2010.11251 | https://arxiv.org/abs/2010.11251 | ABS (teacher-student RECALL) | teacher-student not in abstract |
| Weihs et al. 2020, imitation gap / ADVISOR, arXiv:2007.12173 | https://arxiv.org/abs/2007.12173 | ABS | privileged-teacher trap |
| Li et al. 2020, Suphx, arXiv:2003.13590 | https://ar5iv.labs.arxiv.org/html/2003.13590 | PART | oracle guiding: perfect features decayed 1→0 |
| Li & Zhao 2026, multi-robot laser tag, arXiv:2603.11980 | https://arxiv.org/html/2603.11980 | PART | cross-fire penalty; 110° student; teacher-student gap |
| Hierarchical MARL air combat 2025, arXiv:2505.08995 | https://arxiv.org/html/2505.08995 | PART | friendly-fire penalty: slight reduction, more draws |
| TABX 2026, arXiv:2602.01665 | https://arxiv.org/pdf/2602.01665 | ABS (search) | FOV masking, unseen units zeroed |
| Limited-view Atari (arXiv:2203.16777, 2108.05701) | https://arxiv.org/pdf/2203.16777 | ABS (search) | narrow view degrades performance; learned looking |
| A cone vs 360° ablation in RL | — | NF | — |
| A fixed team display vs learned messages, head to head | — | NF | — |
