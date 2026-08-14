# Brainstorm — 2026-08-08

Capture doc (append-only). Free-form session: Erik offloading ideas, Fable reacting.
Two threads: (1) enemies that change behaviour when damaged, (2) reward design for
RL agents — Breach and Civulator both. Nothing here is canon; harvest into design
docs when an arc picks it up.

---

## 1. Enemies that change when you damage them

**Spark:** a new Quake expansion (30 years after release!) has an enemy that *swaps
into a different enemy* on taking damage — e.g. a gunner who transforms into a dog
(fast melee) when shot. Very C64-era mechanic, and it's actually great gameplay.

**Erik's refinement — believability:** one monster morphing into another is too
game-y. The believable version is a **behaviour change on being attacked**:

- Friendly/neutral animals that turn aggressive when shot.
- Zombies that change aggressiveness or behaviour once damaged.

**Design goal:** enable tactics around *how damage is distributed*:

- **Focus fire** — kill one target completely before touching the next; or
- **Spread damage** — hurt as many as possible as quickly as possible.

**Fable's note — the trigger condition is the dial, and both tactics fall out of it:**

- Enrage on *any* damage → spreading fire enrages the whole pack; chip damage is
  punished; focus-kill is rewarded.
- Enrage only *below a health threshold* → wounding many is safe; execution order
  matters; a different tactical texture from the same mechanic.

Implementation is the cheap kind: aggression/mode as a state variable flipped by
damage events — no new enemy assets needed. RL tie-in for later: train docile and
enraged policies separately and switch on the trigger; a policy switch is far more
believable than a monster swap.

---

## 2. Reward design: "RL favours cheating a little bit" — and why that's fine here

**Spark (from an AI talk):** RL probably favours cheating a little. True for humans
too — people who do things *for rewards* don't build deep motivation; they perform
when someone is looking. Raising kids (presumably) means holding back on rewards so
they *want* to do the right thing.

**Erik's arc:** was always skeptical of dense rewards in RL — didn't want to
contaminate the behaviour. New position: **we are not chasing optimal play, and we
are not raising our agents as if they were our kids.** Therefore: don't be afraid
to use rewards heavily at the start — and once behaviour is trained in, the rewards
can change.

**Fable's reframe (pushed hardest):** in alignment research, shaped rewards biasing
behaviour is the *failure mode*; in our projects it's the *feature*. **The reward
vector is a personality.** Rewards stop being a contamination risk and become a
design instrument — and later, those same dials can become difficulty/personality
settings a player or mission designer touches.

### Named concepts (Erik's instincts → literature)

- **"Lots of rewards first, relax later" = curriculum learning / reward annealing.**
  Standard good practice, not a hack. Sparse win/lose reward is the honest signal
  but nearly unlearnable from scratch; dense shaping is training wheels. Known trap
  on the *removal* side: yank a shaping term suddenly and behaviour can collapse —
  anneal weights gradually, or keep them as small residuals dominated by the
  terminal win reward.
- **The old skepticism was also correct: specification gaming (Goodhart's law for
  RL).** Classic example: boat-racing agent rewarded for score learned to circle
  three turbo pickups forever, never finishing the race. Every dense reward will be
  probed for exploits (teammate-proximity weight too high → agents huddle in a
  corner ignoring the war). Stance: shaping weights small relative to the true
  objective, and treat exploit-discovery as expected iteration — in a game project,
  watching agents cheat is *content*.
- **Precedents for reward-vectors-as-personalities:**
  - *OpenAI Five* (Dota 2): a literal dial, "team spirit" τ ∈ [0,1], blending each
    hero's individual reward with the team average — annealed during training
    (exactly "change the rewards once behaviour is trained in").
  - *AlphaStar* (StarCraft II): a **league** of agents with deliberately different
    reward functions (beat-one-opponent specialists, unusual-unit-composition
    objectives) precisely to get strategic diversity instead of one converged
    style. This is the proven recipe for Civulator's stated "diverse play styles"
    research aim.
- **Completion of the parenting analogy: intrinsic motivation.** The RL analogue of
  "wanting to do the right thing without being watched" is curiosity bonuses — the
  agent rewards itself for encountering surprising states. Extrinsic shaping =
  performing for the audience; intrinsic curiosity = wanting to explore. Later
  experiment worth running: a mostly-intrinsic agent (the "raised right" child)
  vs. the heavily-bribed agents — compare play styles.

### Concrete idea: Civulator warrior-advance reward

Civ 1v1, each side starts with one warrior. **Reward each tile of approach toward
the enemy city:** two warriors advancing n and m tiles → n+m reward. To prevent
farming reward by walking back and forth, either (a) lose the reward again on
retreating a tile, or (b) a watermark: only reward new record-closeness.

**Fable:** option (a) is exactly **potential-based reward shaping** (Ng, Harada &
Russell 1999), with potential = −distance to enemy city. The theorem: any loop nets
zero reward (the farming exploit is impossible by construction), and the shaping
provably does not change the optimal policy — only how fast it's found. The
watermark (b) also blocks the exploit but makes reward history-dependent
(non-Markovian; the observation would ideally need to include the watermark),
which is messier for learning. **Recommendation: start with the symmetric (a).**

Erik's own caveat, kept: this is obviously not a strategy that holds up long — but
it's a useful *vector to explore the game space from*. Once agents learn to play a
little aggressive, relax the reward (see annealing above).

### Concrete idea: soft formation preference

For single-agent-per-unit foot soldiers: a small reward for staying in formation —
**"they don't have to stay in formation, but they kind of like it."**

**Fable:** that sentence *is* a small auxiliary reward weight. The agent trades
formation-keeping against combat utility; the tradeoff point is the dial — high for
disciplined line infantry, near zero for skirmishers. Soft preferences via small
rewards give believable, *breakable* behaviour in a way scripted formation logic
never does. (Spatial cousin of OpenAI Five's team-spirit dial.)

### Meta

Different agent types with different reward vectors is not a side note — it is the
mechanism for behavioural diversity, and it serves both Breach (unit personalities,
full-AI teams) and Civulator (diverse play styles aim).

---

## 3. Water pass 2? — air pockets, and units in water

*(Added later same session, while reviewing the plan going forward.)*

**Context:** the blessed roadmap (`roadmap_2026-07-30_rl_push.md`) already parks
"aquarium/water arc scope" as an open question, and Track 4 lists aquariums
(glass + water volume + squid). This section is seed material for that scoping.

### 3.1 Erik's experiment: air bubbles in a room full of water

So far water has run around rooms full of air. Invert it: what happens with
bubbles of air in a room full of water?

**Fable's physics-honest answer:** literal bubbles can't exist — the model is a
2D depth field (engine/07), no vertical axis to rise through. But **trapped air
pockets in a flooded ship** are *partially emergent already* from three shipped
couplings: (1) fully-flooded cells seal airflow (`dyn_permeability = 0`) → a
pocket ringed by flooded corridors is isolated; (2) volume displacement
compresses the trapped air as water rises (isothermal P·V); (3) the pressure
head pushes that pressure back against the water. Diving-bell mechanics, never
deliberately tested. Caveats: shipped `k_p = 0.5` is far below the physical
10.3 m-water/atm, so pockets yield too easily; and canon flags **near-flooded
cells as the head term's stability worst case, feel-tuning owed** — Erik's
inverted-regime experiment is exactly the stress test that owed pass needs.
What will never happen without a new (small) rule: pockets migrating to higher
ceilings, or air bubbling *through* flooded regions.

### 3.2 Unit ↔ water interactions (one pass needed)

Erik's wish list: float, sink (water usually shallower than a human, but the
sinking-boat level goes deep), breaking aquariums with octopus coming out,
units pushed by massive water, slowed when deep, speed attenuated by currents
("do we have currents? perhaps differentiate the depth").

**Mapping to canon (engine/07 §5.5 — designed, NOT built):**

- Depth slows units / impassable past a threshold — designed.
- Washed-along: strong flow pushes units along the flow velocity — designed;
  canon says land it together with ch.04 decompression suction (same pattern).
- **Currents already exist as state**: `flow_vx/flow_vy` is a real
  momentum-carrying velocity field in the solver — no depth-differentiation
  needed; the unit coupling just reads it.
- **New (not in canon): swim/float/drown states** for deep water — wading was
  the assumed regime. `can_breathe_water` already exists as a creature trait
  (hook for octopi + drowning marines).

### 3.3 Shape of the arc, if/when it opens

Three pillars: (a) inverted-regime experiment + the owed `k_p`/near-flooded
tuning; (b) §5.5 unit couplings + swim/float/drown; (c) aquarium/critter
content (glass via the blast-threshold material column, squid per beastiary).
Timing per roadmap: after fire re-tune, before/interleaved with Track 1 —
content widens the training distribution.

---

*Session: Erik + Fable, 2026-08-08. Related: docs/research/rl_litsearch_2026-07-20.md
(PPO + AlphaStar-arch through-line), roadmap_2026-07-30_rl_push.md (§4 aquariums,
open question "aquarium/water arc scope"), architecture/engine/07_fluid_and_water.md
(§5.5), memory: rl-bots-vision, erik-rl-learning.*
