# Multi-Agent Bot Training Architectures — Design Brainstorm

> Status: **brainstorm / capture only — nothing decided, nothing built.**
> Captured 2026-07-07 (Fable) as an exploratory discussion on how to bring
> Civulator's select-and-move RL approach into Breach. Deliberately kept vague;
> the point today was to *open* the topic and record it, not to converge.
> Revisit fresh in a dedicated session before any design locks or code lands.
>
> Home for Topic 6 in `docs/notes_2026-07-05_topics_backlog.md`. The animal-AI
> half of Topic 6 (predators attack closest, vegetarians graze/flee, all
> rule-based, *not* NN-trained) is treated here only as the "hand-written
> executor" layer; it is not the subject of this doc.

---

## 1. The end goal

Two intertwined goals.

### 1a — A Counter-Strike-like match on a ship

Two teams of trained agents playing a **CS-like match**: either a **bomb
objective** (one team plants / sets off a device, the other defuses / prevents
it) or a plain **deathmatch** (wipe the other team). Everything below is a path
toward *that*, built in stages, simplest first, because this is a long-running
project. This is the direct expression of Breach's core reason to exist: a GPU
physics engine used as a state space to train agents for emergent strategy
(see the project-goal memory).

### 1b — Comparing the architectures against each other is itself a goal

The three models below are **not** a pick-one-and-discard-the-rest decision. We
implement one first (§6), but the intended end state is to have **several —
ideally all — of them coexisting and competing**, so we can measure which
approach actually produces better play. The CS-like match is the natural
**arena** for that comparison: put a Model-1 team against a Model-2 team against
a Model-3 team and see who wins.

A concrete and appealing framing: **different factions embody different
approaches.** One faction is driven by a monolithic commander (Model 1), another
by per-unit RL agents (Model 2), another by an order-distributing commander over
trained units (Model 3). Architecture-vs-architecture then becomes a first-class
part of the game *and* the experiment at once. (Breach's faction model already
supports this — stance is a per-mission table and three-plus mutually hostile
teams on one map is an explicit design target;
[01_units_and_entities.md](architecture/mechanics/01_units_and_entities.md) §8.)

This inherits Civulator's core methodology directly — its CLAUDE.md states the
project exists to "compare architectures ... through controlled experiments,"
and `scripts/tournament.py` already pits differently-built agents against each
other. Breach should carry the same discipline: one variable at a time, keep the
scoreboard, never delete old results. The comparison is not a nice-to-have bolted
on at the end; it is a reason the multi-model design exists.

---

## 2. Reframe — Civulator is *inspiration*, not a port

Civulator's "select-and-move" network is the seed idea, but we are taking
inspiration from its *shape*, not porting the network. Two facts make the
mapping cleaner than it first looks, and one makes it different:

- **Breach is also "turn-based" — just at a fine tick granularity.** The sim
  advances in discrete ticks (Erik referenced ~24 turns/second; current config
  is **12 Hz**, `clock.ticks_per_second` in
  [04_turn_and_control.md](architecture/mechanics/04_turn_and_control.md) — the
  exact number is not load-bearing here, reconcile later). Each tick is a
  discrete decision point in principle.
- **We do not need a decision every tick.** The commander (or unit policy) can
  be activated **once every N ticks**. That cadence is a *free parameter*,
  fully decoupled from the sim tick rate. This is the bridge between Civ's
  "one decision per turn" and Breach's 12/24-per-second clock.
- **What differs:** Breach's action space is richer than Civ's (order types:
  move-attack / move-cover / sprint / fire / grenade / door-explosive, each
  targeting a tile), there is **no map wrap** (ship interior, so Civulator's
  cylindrical padding drops away), and execution is continuous real-time
  physics rather than instant resolution.

The clean insertion point already exists in the engine: the turn doc frames a
*game* as a **ruleset + control mode** layered on the deterministic
`Simulation`, and explicitly notes that **an RL agent is "just a control mode
that never pauses."** A trained commander or unit policy is a new control mode,
not an engine change.

### Aside: could we drop "select"?

Noted and parked. One could imagine dropping the *select* step entirely and
having the commander emit only moves/orders (e.g. one order per unit per
activation, or a tile-per-unit map). **We are keeping select for now** — it
keeps the Civulator analogy intact and the action space simple to reason about.
Flag to revisit.

---

## 3. Two control cadences (framing)

It helps to separate *when* a brain acts, because the models below live at
different cadences:

- **Per-tick reactive brain** — acts (up to) every tick during continuous
  execution, like the existing zombie brain
  ([02_ai_and_los.md](architecture/mechanics/02_ai_and_los.md)). This is the
  natural home for creatures / zombies / animals. Per Topic 6, these stay
  **rule-based** for now; a future NN brain would be a sibling selected by
  `nn_intelligence_tier`, not a rewrite.
- **Periodic commander** — acts every N ticks, issuing / updating orders across
  the squad. This is where the *trained team* intelligence for the CS-like
  match lives.

The three models are mostly about the periodic-commander layer and how much
intelligence sits above vs. inside each unit.

---

## 4. The three models (as discussed)

### Model 1 — The Commander (select-then-move, monolithic)

A single **commander** is activated each turn (every N ticks). On each
activation it makes one choice: **select one unit**, then **move that unit**
(move / attack / move-with-cover / etc.). Then it repeats.

Key difference from Civilization: there is **no "already moved this turn"**
bookkeeping. Instead, **units that already have orders present those orders to
the commander** as part of what it observes. So the commander sees the whole
squad's current order state and incrementally assigns or revises orders over
successive activations.

- Closest to Civulator's literal select-and-move controller.
- The commander issues **concrete low-level orders**; the unit just executes
  them mechanically via existing machinery (A* path preview, auto-fire
  targeting, the footprint→speed cadence).
- Open sub-question: does this commander run in a **planning pause** (batch the
  whole round, like the shipping game) or **periodically during continuous
  execution** (every N ticks)? Erik leaned toward the periodic-during-execution
  reading ("every N ticks"), which implies a *new* control mode rather than the
  current plan-then-execute round.

### Model 2 — Individual unit agents (RL to achieve an order)

Each **unit** is RL-trained to **achieve the order it has been given**. The
order is the goal / conditioning input; the policy learns *how*.

- Order = "kill unit X" → the unit may seek cover, flank, ambush, wait.
- This is the decentralized / per-unit layer. It maps naturally onto Breach's
  data model: stable unit ids, per-unit `get_reward(unit_id)`, brain-selection
  already a unit property.
- This is the layer where **emergent** tactical behaviour would actually appear
  (the project's stated aim).

### Model 3 — Commander distributing tile-orders

A **commander distributes orders to its units**, where an **order is a selected
tile** — the *semantics derive from what occupies the tile*:

- Enemy on the tile → **attack** that unit.
- Friendly on the tile → **protect** that unit.
- Mission objective on the tile → a **mission-specific** order (e.g. plant /
  defuse at that tile).
- Empty tile → **move to** the tile.

This collapses a rich order space into a single, uniform "pick a tile per unit"
action whose meaning is read off the board — elegant, and a small action space
for the commander to learn.

---

## 5. Stacking, and the interesting question of *where* to split

Stacking these layers was always the intent. The genuinely interesting design
question is **exactly where the split between commander and unit sits** — i.e.
how abstract the orders are and how much intelligence lives above vs. inside the
unit:

- **Model 1 alone** = commander emits *concrete* orders (move/attack/cover to a
  tile); units are dumb executors. Low split — the commander does the tactics.
- **Model 3 + Model 2 stacked** = commander emits *goal-level* orders (attack
  X / protect Y / take objective); units are **RL-trained** to achieve them.
  High split — tactics are learned inside the unit, strategy in the commander.
- Intermediate splits exist (commander sets a target and a stance; unit handles
  pathing, cover, timing).

Finding the right abstraction boundary is itself part of the research.

---

## 6. Where to start (staged, simplest-first)

Undecided. Three candidate starting points were named, and the honest answer is
"start with the simplest and grow":

1. **Individual unit only** (Model 2 in isolation) — one policy, egocentric,
   learns to fulfill a fixed/hand-set order. No commander yet.
2. **Simple commander over hand-written rule-following units** (Model 1 / 3 with
   dumb executors) — trains the strategic layer first; execution is scripted.
3. **A mix — commander commanding RL-trained units** (Model 3 + Model 2) — the
   full hybrid, hardest to bootstrap.

Starting with one does **not** narrow the field: per §1b every model we build
stays in the roster as a competitor, so v1 is simply the first entrant in an
eventual tournament, not the winner by default.

Erik is not sure which of these is the right first step. Leaning is unresolved;
capture and revisit. (Fable's earlier suggestion — validate the whole pipeline
with the smallest possible commander, e.g. a focus-fire "pick one target for
the whole squad" selector, then grow — is one concrete way to de-risk, recorded
as an option, not a decision.)

---

## 7. Imitation bootstrap — confirmed wanted

**Yes, we want this.** Civulator's imitation-learning path made a real
difference there, and the same approach should seed Breach and dodge the
sparse-reward cold start (especially since Breach's reward function is still a
stub — see §8).

The Civulator design to adapt is the **Scenario Painter + Order Recorder**
(`civulator/docs/combat_training_tool_design.md`):

- **Scenario Painter** — place units / objectives / terrain, save scenarios.
- **Order Recorder** — a human plays optimal orders through a scenario using the
  *same action interface the agent uses*; every (state, action) pair is recorded.
- Augment via symmetry (Civulator uses 6-fold hex rotation; Breach's square grid
  gives 4 rotations × 2 reflections = 8-fold), pretrain the policy on the
  demonstrations, then RL-fine-tune (or mix demos into the replay buffer, DQfD).

Bonus: a Breach scenario painter overlaps with **Topic 5 (map editor)** — both
need a tile-painting tool. Worth designing them together.

---

## 8. What transfers from Civulator — code vs. pattern

Grounded against Civulator's `civulator/agents/` and Breach's shipped sim.

**Transfers as reusable code / near-verbatim:**
- `ReplayMemory` (replay buffer).
- The self-play / shared-weight training loop (`scripts/train_shared.py`) — one
  network, one buffer, all agents update shared weights (fixed Civulator's OOM).
- The `StateEncoder` ABC pattern (`encode()` + `get_depth()`), swap channels.
- Action-masking infrastructure (valid-select / valid-move masks).
- The imitation tooling *design* (§7).
- The tournament / eval harness (`scripts/tournament.py`) — differently-
  architected agents competing on shared scenarios; directly serves the §1b
  comparison goal, and the one-variable-at-a-time methodology comes with it.

**Transfers only as a pattern (must be re-shaped for Breach):**
- The specific two-head branching DQN (`SelectAndMoveNetwork` /
  `SharedBackboneNetwork` / `FullyConvNetwork`). Breach's action space differs,
  so the *shape* (CNN backbone + select-then-parameterize heads) carries, the
  literal heads do not.
- Cylindrical wrap padding → **dropped** (no wrap in Breach).

---

## 9. Prerequisites — the gating work (architecture-independent)

Regardless of which model wins, Breach's ML layer is **design-only today**
([01_ml_and_training.md](architecture/ml/01_ml_and_training.md)). Three
game-specific pieces gate *all* training and must be built first:

1. **State encoder** — the `[C, H, W]` feature-plane tensor (material, walls,
   obstacles, wall_hp, atmosphere, smoke, fire, heat, per-team unit footprints,
   per-unit HP). None of it is coded; the channel table is a design. Civulator
   donates the ABC; Breach supplies the channels.
2. **`get_legal_actions` / action masking** — currently returns `[]`. Real
   enumeration = reachable tiles (A* exists) × order types × visible/in-range
   targets. Game-specific; Civulator donates the masking *pattern*.
3. **`get_reward`** — currently returns `0.0`. Fully game-specific; must be
   authored for the CS-like objective (kills, objective progress, survival).

The good news: **determinism is further along in Breach than Civulator** (the
whole cross-machine fixed-point / X-ARCH effort), so parallel self-play is on
firmer ground once the above exist.

---

## 10. Open questions (for the next session)

1. **Emergent vs. directed** — is the bot-team a lab for *emergent* tactics
   (favours per-unit Model 2) or deliberate top-down team play (favours a
   commander, Models 1/3)? These pull in different directions; unresolved.
2. **Where does the commander/unit split sit** (§5)? Concrete vs. goal-level
   orders.
3. **Commander cadence & control mode** — planning-pause batch (whole round) vs.
   periodic during continuous execution (every N ticks)? Model 1 implies the
   latter, i.e. a new control mode.
4. **Which of the three starting points** (§6) is v1?
5. **Keep or drop "select"** (§2 aside)?
6. **Objective design** — bomb plant/defuse vs. deathmatch first? The objective
   shapes both the reward and the mission-specific order type in Model 3.
7. **Reconcile the tick rate** — 12 Hz (config) vs. the ~24 Erik referenced; and
   pick N (commander activation interval).
8. **Reward shaping** — the single hardest game-specific piece; nothing exists.
9. **Comparison methodology (§1b)** — how do we run *fair* architecture-vs-
   architecture matches? Shared scenarios and seeds, symmetric spawns, a
   tournament format (cf. Civulator's `scripts/tournament.py`), and an agreed
   metric for "better" (win rate, objective completion, sample efficiency). The
   eval / tournament harness is part of the build, not an afterthought.

---

## 11. References

- **Breach:** [ML & Training](architecture/ml/01_ml_and_training.md) ·
  [Turn & Control](architecture/mechanics/04_turn_and_control.md) ·
  [AI & LoS](architecture/mechanics/02_ai_and_los.md) ·
  [Units & Entities](architecture/mechanics/01_units_and_entities.md) ·
  [Unit Class Design](breach_unit_class_design.md) ·
  [cross_project_overlap.md](../cross_project_overlap.md) §6 ·
  [Topic 6](notes_2026-07-05_topics_backlog.md)
- **Civulator** (`C:\Users\steen\projects\civulator`): `civulator/agents/networks.py`
  (the select-and-move / shared-backbone / fully-conv networks),
  `civulator/agents/state_encoders.py` (Basic / Enhanced encoders),
  `docs/combat_training_tool_design.md` (Scenario Painter + Order Recorder),
  `scripts/train_shared.py` (shared-weight self-play),
  `scripts/tournament.py` (architecture-vs-architecture tournament — the §1b
  precedent), `NEXT_STEPS.md` (goals A/B/C: combat first).
