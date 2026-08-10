# Breach × Tsetlin Machines — Design & Architecture Notes

*Working notes from Claude session, 2026-08-09 (the airplane session). Intended for the Breach CLAUDE.md system; Claude Code should walk the repo and annotate each item as existing / aspirational.*

---

## 1. Tsetlin Machine primer (what matters for Breach)

- A TM learns **propositional clauses**: ANDs over Boolean literals (each feature x contributes x and ¬x). Half the clauses vote for a class, half against; sum votes, threshold.
- **Inference is pure bitwise ops** (AND/OR + popcount-style voting) — no floats, no matmuls. Training uses small integer counters (~8 bit per clause-literal automaton), stochastic feedback.
- **Model size:** at inference a clause is two bitmasks (include-positive, include-negated) + sign. 64 features → 128 literals → 32 bytes/clause. 100-clause model ≈ 3 KB; 1,000 clauses ≈ 32 KB. Fits in CUDA constant memory easily.
- **Training-time size:** ~1 byte per literal per clause (128 B/clause at 64 features).
- **Clause counts in the wild:** tens for small tabular tasks, thousands per class for MNIST-scale CTMs, tens of thousands for big NLP. For game decisions: start 20–100 clauses, grow only if accuracy demands. Fewer clauses = readable doctrine.
- **Binarization is where the work lives:** thermometer/quantile encoding for continuous quantities. Feature quality > data volume.
- **Determinism bonus:** trained model is a flat integer array — serializes exactly, composes with the Q16.16 no-floats discipline.
- Libraries: `tmu` (Agder group, GPU-capable), `pyTsetlinMachine` (CPU). Granmo's book at tsetlinmachine.org now has chapters 1, 2, 4, 5, 6, 7 (regression ch. 3 still "coming soon").
- **LLM-featurization thesis:** the historical bottleneck of rule systems — hand-authoring features — is dissolved by LLM-assisted enumeration of candidate literals; TM feedback prunes the useless ones. Complementary weaknesses.

---

## 2. Core TM applications in Breach

### 2.1 Hazard prognosis / danger sense (build first)
- Snapshot local binary features around a tile/unit: `fire_within_2`, `pressure_below_threshold`, `pressure_falling`, `breach_in_room`, `door_open_to_vacuum`, `water_present`, `falling`, …
- **Self-labeling loop:** fast-forward the sim N seconds; label "did this location become lethal, y/n". Run headless overnight → millions of labeled samples for free. The simulation is the oracle.
- Trained TM = per-unit danger sense evaluated every tick; inference cost negligible.
- Labeling pipeline ≈ a weekend of Claude Code work on top of the existing sim.

### 2.2 Doctrine as learned clauses (interpretability as a game feature)
- Train per-decision TMs (seal door? retreat? engage? weld here?) from scripted teachers or player traces.
- **Surface learned clauses in-game as readable doctrine:** "IF breach_in_room AND NOT wearing_helmet THEN flee."
- Veterans have richer clause sets than rookies; characterful failure modes fall out of subtly wrong clauses. Directly serves the design principle that companions must feel valuable and legible.

### 2.3 Other classifiers
- Target selection (binary threat features).
- Propagation surrogates: "will this fire jump the corridor?" — cheap TM the AI queries instead of rolling the sim forward.
- Task assessments: "can I weld this door in time?"

---

## 3. Engine architecture: hot/cold split (ECS thinking)

**Principle: data lives with its highest-frequency consumer.** Units are entity IDs + component arrays, not objects that live "somewhere".

- **Hot (GPU, SoA arrays):** position, health, physics coefficients, combat state, feature word, danger score, cooldown timers, room_id, vote-bias, clause-count cap. Structure-of-arrays for coalesced access.
- **Cold (CPU, rich objects):** inventory, identity, faction, dialogue, doctrine, AI memory, squad state.

### 3.1 Derived-stats compilation (inventory → physics)
Inventory stays cold but **compiles down** to hot scalars: `pressure_resistance`, `thermal_resistance`, `blast_multiplier`, `carried_explosive_yield`. On inventory change (rare), CPU recomputes and queues an update. GPU never knows what a medkit is. Matches the existing base/effective stats split — effective stats are what get mirrored to the GPU.

### 3.2 The membrane: one batched transfer each way per frame
PCIe punishes many small transfers; barely notices one 50 KB blob. 1,000 units × 32 B status is nothing.

**Up (CPU→GPU), once/frame — command buffer:**
- spawn/despawn, target assignments, derived-stat updates, teleports/scripted moves, door state changes.
- One channel, many command types, applied by a kernel at a defined point in the tick. Gameplay never writes GPU state directly → determinism, replay, no mid-tick corruption.

**Down (GPU→CPU), once/frame — two channels:**
1. **Status mirror** (fixed size): per-unit position, health, combat state, danger scores, flag bitfield (`on_fire`, `in_vacuum`, `submerged`, `took_damage_this_tick`). Gameplay reads only the mirror; a frame of staleness is fine.
2. **Event queue** (variable size): preallocated array + atomic counter. Attacks landed `{attacker, target, damage, position, type}`, deaths, breaches, ignitions, structural destruction. CPU drains and dispatches.

**VFX / audio / animation are consumers of these same channels, not new channels.** Audio + particle bursts subscribe to events; animation derives from the status mirror (GPU combat state machine = animation state machine; CPU skins it). Continuous field VFX (fire, venting) can later render straight from GPU fields without the CPU — a rendering decision, not an architecture change.

**Ownership rule (top of the contract doc): the GPU owns what happened; the CPU owns what it means.** Kernel says "damage landed, unit died"; CPU says "that was the last squadmate, change the music."

### 3.3 Determinism / fixed-point
- Sim time = integer ticks. All sim math Q16.16 — no floats.
- Integer `atomicAdd` is associative → damage races are deterministic regardless of thread scheduling (floats are not: FMA contraction, order-dependent atomics). Enables lockstep multiplayer / replays.
- Q16.16 multiply needs 64-bit widening: `(int64_t)a * b >> 16`. Wrap in a single `__device__` math header — `qmul`, `qdiv`, `qsqrt` — so determinism is a library property, not per-kernel discipline. "Abstraction in functions, not data."
- Integer tick comparisons (`sim_time - last_attack_tick >= interval_ticks`) are exact; no epsilons.

---

## 4. The three-tier mind hierarchy

| Tier | Units | Decision | Locus | Cadence |
|---|---|---|---|---|
| 1 Deliberative | Marines (~10) | RL policy / behavior trees, squad coordination, doctrine | CPU (TM outputs as sensory inputs) | 5–10 Hz |
| 2 Reactive-consequential | Zombies, hordes (10–200) | CPU owns goals/targets; GPU owns steering + contact | Split | Goals slow, steering per-tick |
| 3 Pure reflex | Mice, rats, fish, wasps (1000s) | Boids steering + TM danger gain; no CPU representation | Fully GPU | Per-tick |

- **Reflexive GPU-side reactions:** stimulus→response closes inside the physics tick. Kernel reads local fields (pressure/temperature gradients, TM danger score) and writes velocity directly. No command buffer, no readback, no latency — the school ripples away from the blast *the tick it arrives*. Avoids the "wavefront outruns the panic" artifact of AI-tick-cadence reactions.
- **Steering behaviors / boids:** motion = weighted sum of vector terms (flee-danger, align, cohere, avoid-walls). **TM danger score is the gain on the flee vector.** A mouse's entire mind: one 64-bit feature word, one TM eval, one weighted vector sum.
- **Diegetic UI:** fauna telegraph the simulation — mice streaming out of a corridor before the pressure readout updates. The simulation becomes legible through the animals.
- The NN/TM split is really **decision cadence**: slow deliberate decisions afford big CPU models; fast reflexes need bitwise GPU models. Marines use TMs too — as sensory organs feeding a bigger brain. For a mouse, the sensory organ *is* the brain.
- **Open paper-design question per tier:** one-way (read fields only — mice, fish) vs two-way (write back — zombies with collision/combat). Decide now to avoid migration later.
- 5v5 consequential units + ambient swarms = expensive brains few, numerous brains cheap. Hordes of ~200 zombies affordable in tier 2 with hot steering + cold goals.

---

## 5. Tier-2 combat on the GPU

**CPU owns intent (target selection — gameplay-meaningful, maybe a TM). GPU owns timing and contact (melee feel demands tick-rate resolution).**

Hot per-unit combat state: `last_attack_tick`, `attack_target` (CPU-assigned entity index), derived `attack_range/damage`, `combat_state` enum + phase timer, stagger timer.

- **Attack state machine (per-tick kernel):** IDLE → WINDUP (cancellable on stagger) → STRIKE (damage lands, event fires) → RECOVERY → IDLE. Four states, two timers. Makes melee readable and dodgeable — "meaningful constraint" applies to zombie attacks too. Shockwave kernel writes stagger timer; combat kernel reads it — physics and combat compose through shared scalars.
- **Damage races:** many attackers on one target ⇒ `atomicAdd(&health[t], -damage)`. A plain `-=` silently loses hits, a bug that only appears in hordes.
- **Deaths:** kernel sets dead flag + emits event; **CPU adjudicates** (loot, cleanup, music). GPU detects, CPU decides meaning.

### CUDA idioms (summary of the primer)
- Kernel = body of the loop; parallelism is the loop. `int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;`
- **Abstraction in functions, not data:** small `__device__` helpers, structs bundling pointers, templates — yes. AoS layouts, virtual functions, pointer-chasing object graphs — no. SoA gives coalesced warp loads (order-of-magnitude difference).
- **Divergence:** warps of 32 execute in lockstep; both sides of a branch get paid for. Avoid mega-kernels with `switch(unit_type)`. One kernel per behavior tier — which the tier architecture already does naturally.
- **Prototyping path:** Numba `@cuda.jit` kernels in Python against the same SoA arrays; port stable hot kernels to C++/CUDA. The architecture (arrays, command buffer, event queue) transfers unchanged.

---

## 6. TM on the GPU

- **Featurization is a CUDA kernel:** each thread reads GPU-resident fields around one unit, writes a 64/128-bit feature word. Never stream raw fields to CPU just to binarize.
- **Inference kernel:** per clause, `AND(features & mask_pos, ~features & mask_neg)` + signed voting; embarrassingly parallel across units × clauses; microseconds for 1,000 units. Danger scores ride home in the status mirror.
- **Model format:** flat array of uint64 mask pairs + signs, in `__constant__` memory. Model updates = asset reload.
- **Training stays offline** (Python, `tmu`, logged data), at least initially. On-GPU online learning is a cool later project (drags RNG + stochastic feedback into the engine for no launch benefit). FPGA/online-TM literature exists if that itch ever needs scratching.

---

## 7. Behavioral dials (one trained brain, many personalities)

- **Intelligence = clause count (weighted TM + truncation).** Train one 100-clause smart-zombie brain; sort clauses by learned weight; cut anywhere: top-10 = shambler, top-40 = walker, 100 = the flanker. Graceful degradation (least important patterns go first). One training run → continuous intelligence spectrum.
  - **Brain damage as a mechanic: array truncation at runtime.** Headshot deletes top clauses; the zombie visibly gets dumber. Ties into monsters that change AI after being hit.
- **Temperament = vote bias.** Decision is `sum(votes) > T`; add a per-unit signed constant to the attack-class sum. Positive = aggressive on flimsy evidence, negative = demands overwhelming justification. Continuous, runtime, per-individual, zero retraining. Dynamic: bias climbs on horde damage (enrage), drops on queen's caution order.
- **Numbers-awareness as a learned feature:** thermometer-encode local count advantage (`advantage_ge_-2 / 0 / 2 / 4`, from the neighborhood/steering kernel). The TM *discovers* "attack IF advantage_ge_2 AND NOT marine_has_flamethrower" from outcome labels — learns when numbers matter, not just that they do. (The original `n_zombies − n_marines > k` idea is subsumed: k-like control comes from the vote bias; the count features come from thermometer literals.)
- Both dials are just integers in hot SoA arrays. Intelligence × temperament = a bestiary from one model.

---

## 8. The octopus (distributed arm minds)

- Real cephalopod neurobiology: ~2/3 of neurons are in the arms; central brain issues intent, arms solve local problems semi-autonomously.
- **Architecture: eight independent TM instances, one per arm.** Each reads local features (threat direction, distance to surface, `is_gripping`, `arm_damaged`) → action class (shield / strike / anchor / reach). Small central arbiter (could itself be a TM) resolves conflicts and sets shared goal (flee vs fight), entering each arm's feature word as context bits.
- Emergent behavior: the arm between body and fire independently shields while others pull away — nobody codes "shielding".
- **Pair with destructible arms** → genuinely original gameplay: cutting arms literally removes minds.
- Duals of one template: octopus = one body, many TMs; wasp swarm = many bodies, one TM each, pheromone-style shared context bits.

---

## 9. Hybrid architectures (TM × NN)

- **TM → NN is structurally trivial:** TM outputs are integers (vote sums, or raw clause-activation bitvectors) — fine NN inputs. Framing: TM layer converts raw state into *semantic assertions* (`welding_viable`, `retreat_indicated`, `flank_open`); NN deliberates over assertions. Classic neuro-symbolic: symbolic perception, connectionist policy. Bonus: deep RL over ~30 meaningful predicates is far more sample-efficient and map-transferable than RL over raw state.
- **NN commander → TM agents:** the practical one; the tier hierarchy extended to squad level (slow rich deliberation over fast cheap reflexes).
- **TM commander → NN agents:** the thematic one. **Legible strategy, illegible tactics** — a hive queen whose doctrine the player can capture and read from a data terminal ("IF hull_breach_deck_3 AND marines_split THEN converge_reactor"), commanding individually opaque creatures. Players learn to exploit the queen's rules → interpretability as gameplay, again.
- **Prior art check (searched 2026-08):** ingredients exist (HVC+TM hybrids, TM composites, TM contextual bandits, FPGA online TM), but hierarchical heterogeneous multi-agent control with TMs in the loop appears unpublished. Breach would be a quiet systems-research novelty; "interpretable reflex layers under learned deliberative control" is a plausible WASP-adjacent paper someday.

---

## 10. Commander decisions: verb/noun factorization

- Split orders into **verb** (attack / retreat / hold — small discrete choice, TM territory) and **noun** (where — spatial argument, huge cardinality, not TM-shaped directly).
- **Candidate generation + scoring:** the ship *is* the candidate generator — rooms, doorways, chokepoints; 20–50 tactical nodes per map. Nothing needs to invent positions, only rank a finite set — and ranking a node from Boolean features is TM-shaped.
- **Location-scoring TMs:** per candidate room, feature word: `sealable`, `single_entrance`, `on_fire`, `pressure_ok`, `contains_medkit`, `enemy_los`, `distance_band_*`, `between_enemy_and_objective`. One TM per question ("good retreat spot?", "good attack staging?"); raw vote sum = score; argmax wins.
  - Fully interpretable chain: order readable, location choice readable ("retreat to medbay BECAUSE sealable AND single_entrance AND NOT enemy_los"). Captured terminals reveal the queen's *taste in real estate*.
  - Cheap enough to re-score every AI tick → retreat orders retarget as fires spread. Looks uncannily intelligent.
- **Features come from existing infrastructure:** static map annotations (computed at load) + per-room pooled GPU-field aggregates (one reduction kernel). Influence maps, except the influence is real physics.
- **Where an NN still earns its keep (later, maybe never):** inter-candidate relational / long-horizon value ("safe now, trapped when fire reaches the corridor"); continuous fine positioning within a room (though steering handles this at tier-2 quality).
- **Ship the all-TM version first.** Implementable in a week, debuggable by reading rules; if insufficient, it becomes the interpretable baseline the NN is benchmarked against (paper shape).
- **Faction mix-and-match:** queen = TM verb + TM noun (fully exploitable). Rogue military AI = TM verb + NN noun — you can read *what* it intends but never predict *where*. Unsettling by design.

---

## 11. Training curricula (the bootstrap answer)

TMs are supervised, not RL — they need **labeled outcomes**, not a worthy opponent.

1. **Scripted marines** (patrol waypoints, shoot nearest, fall back at low HP — an afternoon of behavior-tree work). Need variety, not skill.
2. **Headless episodes overnight**; label decisions by fast-forwarding: attacked while outnumbered → died in 10 s → "attack was wrong here." The sim is the oracle (same pattern as hazard prognosis).
3. **Ecosystem ratchet:** once NN marines exist, retrain zombie TMs on logs of those fights; iterate.
- Predator-prey (zombies vs rats) is a zero-infrastructure alternative curriculum, but zombies-vs-scripted-marines exercises the actual game loop — start there.

---

## 12. The rat economy (tier 3 with a graph brain)

- Rule: **stay and eat if food ≥ rats in room; leave if overcrowded** — this is *ideal free distribution* from behavioral ecology, rediscovered.
- **Per-room, not per-vicinity, definitively:** one reduction kernel pools per-room aggregates (rat count, food, fire, pressure — already needed for the commander's location scorer, so rats ride existing infrastructure). Each rat's decision = a couple of array lookups via its `room_id`, vs. per-agent spatial queries. Room granularity also *reads* correctly at rat scale ("the colony in the galley").
- Rat mind: room-level TM (or bare rules) picks *which room to want* (stay / flee-to-neighbor: overcrowded? on fire? neighbor smells of food?); tier-3 steering gets it through the door. **Verb/noun again, at rodent scale.**
- **Emergent freebies:**
  - Rat migrations pour through corridors *ahead of* spreading danger → the diegetic-UI effect gains an economic engine; players learn migrations mean something changed upstream.
  - If `rat_flux_in_room` is a literal in the zombie vocabulary, zombies *learn* disturbed rats as a marine-presence cue from training data. Nobody codes it. Systemic storytelling.

---

## 13. The room graph (first-class engine citizen)

One data structure, at least four consumers — the sign the abstraction is right. Design on paper next to the CPU/GPU contract.

- **Structure:** rooms = nodes; doors/vents = edges; per-room annotations (type label, food sources, sealable, entrance count, healing apparatus, …).
- **Consumers:** commander location scoring · rat economy · level-generation validation · graphics generation (via labels) · per-room field aggregates.
- **Generation-time validation:** connectivity, chokepoint existence, rats-can-reach-food, retreat-worthy rooms exist for the scorer. The same annotations are the generator's fitness function.

## 14. Map generation (gating requirement — new ideas, to be developed)

- **Room vocabulary:** crew quarters, cockpit/bridge, med bays (healing apparatuses), cantinas + food warehouses (food density high — feeds the rat economy), corridors, engineering, cargo, airlocks.
- **Weldable doors** as a generation-aware element (interacts with welding TM assessments and chokepoint design).
- **Ships vs stations:** spaceships must include the mandatory set (crew quarters, cockpit, …); stations get freer layouts; some maps are *fragments of huge stations* and need not contain every room type.
- **LLM-generated recipes/grammars:** believable rules the generator runs on — occupancy logic, adjacency preferences (cantina near quarters, med bay near engineering?), faction/lore-flavored variants. Recipes can reflect the lore (SPACE COM ledger, Grays, etc.).
- **Room labeling** is required output — used later for graphics generation, and immediately by the location scorer + rat economy.
- Status: to be talked through in a dedicated session (see roadmap).

---

## 15. Open design questions

1. Target unit counts per tier (barely changes the architecture at 100 vs 10,000, but decides dead-unit compaction).
2. Per-tier one-way vs two-way physics coupling (mice/fish read-only? zombies write — already two-way; fish blocking a torpedo?).
3. Where exactly the reflexive line sits: any GPU-side decision-making for tier 1/2 (auto-stagger between AI ticks)?
4. Event queue taxonomy: enumerate event types + payloads.
5. Room graph schema: annotation set, static vs dynamic fields, ownership (CPU-authored, GPU-mirrored aggregates).
6. Feature vocabularies: per-unit hazard word, per-room word, zombie decision word, arm word — enumerate literals and thermometer thresholds.
7. Weighted-TM training config for truncation-friendly models (clause count, T, s per decision).

## 16. Roadmap (rough order)

1. **Communication contract doc** → Claude Code walks the repo, annotates existing vs aspirational. The gap list is the implementation plan.
2. Q16.16 device math header (`qmul`/`qdiv`/`qsqrt`).
3. Hazard-prognosis pipeline: featurization kernel + headless labeling loop + offline `tmu` training + GPU inference kernel + danger score in status mirror.
4. Event queue + combat kernel (state machine, atomics, cooldowns).
5. Zombie tier 2: scripted-marine curriculum, target-selection TM, vote-bias + clause-truncation dials.
6. Room graph + per-room aggregates; rat economy; commander verb/noun TMs.
7. Map generation (dedicated design session first).
8. Exotics: octopus arms, TM/NN commanders, online learning — after the core loop proves out.
