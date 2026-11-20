# Lens 4: the network and the trainer's needs

Critic: Opus, 2026-11-11, independent of lenses 1 to 3 (not read). Object:
`docs/rl_agent_design_2026-11-11.md` at `fb47210d`, sections 2.1 to 2.5, 6.3,
6.5, 7, 8. Read beside it: roadmap sections 2, 3, 5; digest 05 in full; the
architecture parts of digests 01, 02, 04; `src/rl_env.py` (step / reset /
`valid` / `final_obs` / counters), `src/control_policy.py` (`VERB_RULES`).

## Verdict

**Survives with fixes. 0 blockers, 5 major, 10 minor** (MINOR 1 sits under the weight table). The arithmetic of 8.1
holds (my recomputation lands within 1 % of 1.55 M, under assumptions the
table leaves unstated), the head order of 2.3 gives a well-defined joint
log-probability that a trainer can recompute from stored data, and keeping dead
rows "on the books" is a coherent arrangement for GAE, arguably the simplest
correct one. Nothing makes a sound PPO trainer impossible. What needs fixing
before rung 1's patches freeze the env contract: the value outputs do not
match the reward groups the env returns (7.2 against 6.8), so moving the team
share would NOT leave the estimate intact as 7.2 claims; the env has no
per-row flag that tells "dead but on the books" from "not in this match", so
a trainer using today's `valid` would cut every dead soldier's team share and
outcome; the per-decision storage of the two map windows (about 90 KB in
float32) makes the batch sizes the survey discusses impossible unless the
storage form is decided; "no input statistics because every input is a ratio"
leaves the heavy-tailed physics ratios (fire, blast, wind) unbounded; and the
memoryless value network is a known biased critic once memory matters, which
needs a named test, not new machinery.

## Weights, recomputed

Assumptions (the table does not state them; each figure depends on the one
named): 3x3 kernels; biases counted; FFN width 2x in attention; PyTorch GRU
(two bias vectors).

| part | document | mine | depends on |
|---|---|---|---|
| vector reader | 66 000 | 66 400 if 130 -> **256** -> 128; only 33 300 if 130 -> 128 -> 128 | the hidden width, unstated ("two layers, 128 out") |
| entry readers | 38 000 | 37 600 = 6 x (48 x 128 + 128) | ONE linear layer per type, 48 inputs (estimate) |
| list attention | 265 000 | 265 000 = 2 x (4 x 16 512 + 65 920 FFN + 512 LN) | FFN 256 wide; with the usual 4x FFN it is 397 000 |
| list summary | 66 000 | 66 000 (Q, K, V, O at 128) | one cross-attention, no FFN |
| near reader | 229 000 | 228 500: convs 97 300 + flatten 4 x 4 x 64 -> 128 = 131 200 | 17 planes (estimate), one stride-1 then three stride-2 layers (25 -> 13 -> 7 -> 4) |
| far reader | 229 000 | 228 800 | 18 planes (estimate) |
| join | 131 000 | 131 300 = 512 x 256 + 256 | |
| GRU 256 | 395 000 | 394 800 = 3 x (2 x 256^2 + 2 x 256) | input 256 |
| heads | 135 000 | 100 000 to 140 000 | unspecified: feet 12 600, aim query 256 -> 128 = 32 900 + 17 token vectors 2 200, verb query from 384 -> 128 = 49 300, trigger and argument ~3 000; 135 000 needs small hidden layers |
| kind embeddings, LayerNorms | not listed | 2 000 to 5 000 | the number of kinds; negligible |
| **total** | **about 1 550 000** | **1 510 000 to 1 560 000** | |

Value network: the same readers without the GRU sum to 1 024 000; "about
1.1 million" needs one 256 x 256 layer between the join and the outputs
(+66 000), which 8.4 does not name. The 64-entry list changes no weight count.
Nothing is double-counted. Missing from the table: the layer(s) after the join
in the value network; the hidden widths of the vector reader and the heads.
**Fix (MINOR 1):** state these three widths in 8.1 / 8.4, since "the number
of weights is printed and matches the spec" (8.6) needs a spec to match.

## Findings

### MAJOR 1. The value outputs do not match the reward groups, so the team-share schedule rebuilds the estimate

- **Claim:** 7.2 gives one value output per group "(outcome, balance, friendly fire, order)" and says a weight "can then be moved during a run ... and the estimate needs no rebuilding". 6.3 moves the team share 0.3 -> 0.8 during a run.
- **Evidence:** 6.8 has the env return FIVE groups: outcome, **own balance, team balance**, friendly fire, order. A single "balance" head must predict the blend `(1 - tau) own + tau team`, whose target moves every time tau moves: the head is non-stationary for the whole schedule, exactly the rebuilding 7.2 says is avoided. HoK's five heads (digest 01) are fixed-weight groups, not a blend.
- **Fix:** five value outputs, one per env group (own balance and team balance separate). The trainer forms `V = sum_g w_g V_g` and the advantages from the same weights; tau is then a trainer number like any other weight.

### MAJOR 2. The env has no flag for "dead but on the books"; `valid` means alive

- **Claim:** 6.3 "a dead soldier stays on the books", 7.2 the value network keeps estimating for it.
- **Evidence:** `rl_env.py` (module doc, `row_units`, `observe`): a row is `valid` "while its unit is alive and still on its team". A row of a team smaller than `S` on this level (padding), a dead soldier and a turned soldier are all `valid = false`. The natural trainer reading (`valid` false -> trajectory over, V = 0) deletes the post-death team share and outcome that 6.3 is built on, and GAE then never carries a win back to the soldier who died for it.
- **Fix (the env's, patch 1b / 1e):** two per-row flags: `in_match` (spawned in this episode; false for padding) and `acts` (alive, on its team; today's `valid`). Rule for the trainer, written in 6.3: a row with `in_match` and not `acts` enters GAE and the value loss, never the policy loss or the entropy; a row without `in_match` enters nothing. The reward groups are emitted for every `in_match` row (6.8 says "dead rows included"; padding rows should be stated as zero).

### MAJOR 3. Per-decision storage of the map windows decides the batch size, and nothing decides it

- **Evidence (arithmetic):** near 25 x 25 x 17 + far 25 x 25 x 18 = 21 875 floats = 87.5 KB per decision in float32, plus the list (~40 x 48) and the value view's extra list (64 entries) and plane: about 110 KB. The survey's "1M-decision batch is about 4 minutes of farm time" (digest 05 C) would be ~110 GB; 100 000 decisions ~11 GB, beyond a desktop GPU's memory for training with activations (UNVERIFIED which GPU; dev_setup names none). Per env step at 64 envs x 8 rows the farm's shared buffer grows from today's ~10 MB crop to ~56 MB (the 2026-11-11 page-fault incident was at 10 MB).
- **Fix on paper:** (a) the value network reads the soldier's two windows from the SAME stored arrays (7.1 says "the same windows"), with its one different plane ("watched" from every enemy) stored as a delta; (b) the observation spec fixes a storage form: the yes / no planes (wall, door, sight, lanes, units) as uint8 or bit-packed, the ratio planes as float16 after the clamp of MAJOR 4; (c) rung 2's trainer design states its batch in decisions and its memory budget. That brings ~110 KB to ~25 KB.

### MAJOR 4. "No running statistics because every input is a ratio" leaves the heavy tails unbounded

- **Claim (8.3):** inputs are ratios, flags, cosines or learned vectors, so no normalisation is needed.
- **Evidence:** a ratio is bounded only where its table value bounds it. Temperature over the bearable maximum is ~0.1 in a room and tens in a fire; wind push over the knockdown push spikes by orders of magnitude in a blast; pressure over the bearable range in a breach; "time left" in seconds-ratio terms. Andrychowicz et al. 2020 (digest 05 A1): "always use observation normalization"; digest 05's own short list item 5 asks for it. A conv over a plane that is 0.1 everywhere and 40 on three tiles is what makes early PPO steps blow up (OPINION, common experience).
- **Fix (no machinery):** the observation spec defines, per feature, a fixed squash at the door: `clip(x, 0, c)` or `symlog(x)` (DreamerV3, digest 05 A3), chosen once; plus LayerNorm after each reader (pre-LN in the two attention layers, the form GTrXL found trainable under RL, digest 05 C, [RECALL]). Keep the "no running statistics" rule: both fixes are fixed functions.

### MAJOR 5. The memoryless value network is a biased critic exactly where memory matters, and 7.2's reason does not hold

- **Claim (7.2):** "It sees the whole state, so it has nothing to remember."
- **Evidence:** the soldier's choices depend on its GRU state, and the bots' on their team picture, commitments and shared target. A value of the state alone averages over those hidden variables. Lyu, Xiao, Daley, Amato, "Contrasting Centralized and Decentralized Critics in MARL" (AAMAS 2021) and Lyu et al., "A Deeper Understanding of State-Based Critics in MARL" (AAAI 2022) show a state-only critic can bias the policy gradient under partial observability, and that a history-and-state critic removes it ([RECALL], titles and venues not re-checked this session). Today the bias is small (shared all-round sight, last-seen records in the observation); at rung 8 it is not.
- **Fix:** keep v1 memoryless (simplest, agreed). Add to 7.4 a rung-3 experiment and to rung 8's entry condition: value input + the soldier's GRU state, detached (256 numbers, no gradient into the soldier's network, so no leak into a choice; 8.6's scramble test still applies to the other extras). Change 7.2's reason to "memory adds little while sight is shared; revisited at rung 8".

### MINOR

2. **The argument's mask depends on both upstream samples.** "Onto my aim target" is legal only when the aim is a list entry; reach vs timer depends on the verb. The trainer must store the argument mask AS REALISED after the aim and verb were sampled (or recompute it from stored base masks + stored choices by one shared function). State it in 2.4 / 8.2. Per-verb argument heads (OpenAI Five's mutually exclusive parameters, digest 01 1c) avoid one head whose index 0 means "onto target" for a throw and "5 s" for a plant (OPINION; the shared head with the verb told is also sound).
3. **What a stored decision holds** (for 8.6's recompute test; nowhere listed): the observation as served (the list order IS the index space, so a pointer index needs no game id inside a sample); feet, aim, trigger, verb, argument indices; the four masks (aim over list + 17, verb over the class's verbs, argument as realised); the "used" flags per head; the old log-prob per head; the GRU state at segment start; the per-group rewards; `in_match` / `acts`.
4. **The pointer index must not cross a decision.** 3.2's "its own previous choices" and the decoder's held aim need the GAME ID of the entry (the worker keeps the id per served index until the next step; after an auto-reset the served list is the new match's, which is consistent). In the observation the previous aim must be a flag on the entry it names, never an index into a list that has reordered. State both.
5. **"No effect" must be read off the decision-time masks, never off what happened.** A stun that ends at tick 2 of 4 makes the held choices act after all; that is acceptable bias, but the exclusion must be a stored flag so collection and training agree. "Nothing" (hand) and the compass tokens (aim) are never masked, so no head is ever all-masked (a softmax of all -inf is NaN); a soldier that cannot act samples a fixed default.
6. **Entropy:** summed over used heads, each conditional entropy at the stored parents (standard; digest 05 C). Excluded heads get no entropy bonus, or they drift uniform and the argument head learns noise.
7. **The clock as a real terminal (6.5) is correct** (time left is in both networks' inputs; Pardo et al., "Time Limits in RL", ICML 2018 [RECALL]) but contradicts digest 05's item 7 ("bootstrapped from the critic"). Write in the env contract that, under these profiles, `truncated` is terminal: then `final_obs` is never needed for a bootstrap, a simplification worth naming. The trainer's own segment cuts still bootstrap from V of the next served observation.
8. **Value normalisation with five heads:** per-head running return statistics (MAPPO's value normalisation, or PopArt), each head's loss on its own normalised target, GAE on the de-normalised weighted sum, advantage normalisation per minibatch after the sum; a std floor, since the friendly-fire and order heads are zero nearly always.
9. **8.3's "never waits on the network" is believable for inference, not established for training.** Inference: ~30 M multiply-adds per soldier (the near reader's stride-1 first layer alone 3 M; both readers 17 M; attention ~11 M), ~100 GFLOP/s at 3 000 decisions/s: trivial. Training: forward + backward of both networks, ~0.4 GFLOP per decision per epoch; at 5 to 10 epochs that is 6 to 12 TFLOP/s sustained to keep pace with the farm, on small convolutions that run far below a GPU's peak (OPINION, estimate). Synchronous PPO may spend as long learning as collecting. Fix: rung 2 measures it; a stride-2 first convolution cuts the near reader ~3x; mixed precision.
10. **The value input duplicates the soldier's list** ("the soldier's own observation, all of it" + every unit). MAPPO's best input was the feature-PRUNED one (digest 04, 84.4 % vs 63.3 %). Smaller: the value network reads the full list, each entry carrying the soldier-relative "for me" features and the "my team sees / has a record / does not know" flag, plus the soldier's own vector and windows; not the soldier's list as well.

## The other questions of the lens, answered

- **Joint log-probability (2.3, 8.2):** `log pi = log p(feet) + log p(aim) + u_t log p(trigger | aim) + u_v log p(verb | aim) + u_a log p(arg | aim, verb)`, `u` the stored used-flags. Well defined, recomputable with teacher forcing on the stored aim and verb (the chosen entry's embedding regathered from the recomputed attention, a token's from its learned vector). Dropping a term from both old and new log-prob keeps PPO's ratio valid (OpenAI Five, AlphaStar did so). Sound.
- **Variable pointer sets batch cleanly:** pad the list to its cap (~40) plus 17 fixed tokens, verbs to the largest class's count, masks -inf on padding (digest 05 C). Sound.
- **Masks within the held 4 ticks:** latched verbs go out on tick 1 (`VERB_RULES`), so a stale verb mask cannot fire anything; an aim entry that dies or leaves sight is resolved by the decoder (last-seen position); the trigger's 4-tick latency is rung 1's measured variant. Sound.
- **Dead rows against GAE / MAPPO death masking:** MAPPO zeroes a dead agent's critic input to its id (digest 04) because its dead agents' rewards are the team's and post-death states are rare. Here the dead row's only job is to give the death decision its bootstrap and carry the team's later rewards back to it; dead steps in GAE and value loss, no policy term, is coherent. A cheaper equivalent exists in principle (the dead row's value is the team's team-balance + outcome value, the same for every dead soldier of a team), but it saves only cheap value forwards; keep the design and run its 7.4 experiment.
- **Memory and the env's resets:** reset the GRU of every row of an env with `info["ended"]` before the next forward (the returned obs is already episode k+1's); store h at each segment start; within a BPTT unroll multiply h by (1 - ended) per step (CleanRL `ppo_atari_lstm`, the 37 details); a dead row's h is simply unused. "Memory on and off in the duel" is enough as an experiment; OPINION: build the shooting range feed-forward first and switch the GRU on for the duel, so the recurrence bugs (digest 05's main risk list) meet a trainer already proven on a task.
- **Readers:** two attention layers at 128 for ~40 entries, one linear embed per type (adding a ReLU MLP is OPINION), two small position-keeping conv readers, no scatter: none is a known mistake at this scale (hide-and-seek 1 layer, AlphaStar 3 at 512 entities).
- **Imitation (rung 1 records, rung 2):** every bot decision maps onto these heads: the focus target -> the aim pointer (after patch 1a the target is in the same team picture, so in the list unless beyond the 16 cap); the predictor-gated trigger -> trigger (majority over the 4 ticks, rung 1's translation); A* travel -> nearest of 16 at "walk" (bots walk at throttle 0); a lob at a group -> a compass direction + nearest reach fraction; `use_intent` on a door -> "use". Bots never reload by verb, so reload is never a label. Behaviour cloning is the same joint log-prob as a loss. Sound.

## 8.6's tests: kept, and what is missing

Kept: masked choice has probability zero; recomputed probability equals the
collected one; list permutation equivariance; empty slot inert; scrambled value
extras change no choice; weight count. **Amend** "the same" to "within 1e-5,
and the PPO ratio is 1 on the first minibatch before any update": batched
GPU inference and minibatched training differ in float rounding, so "exactly"
would fail spuriously. **Add**, each a property a self-written trainer most
often breaks (digest 05 D's risk list):

- GAE equals a hand-computed case containing a death mid-segment (dead steps carry rewards and value, no policy term), an `ended` env auto-reset inside the segment (no leak across the boundary), and a clock end (terminal, no bootstrap);
- the GRU state of a row is zero on the first decision of every episode, and h recomputed from the stored segment start reproduces the collected h;
- an excluded head has zero gradient and zero entropy contribution;
- padding rows (`in_match` false) contribute nothing to any loss;
- the de-normalised weighted value sum is unchanged by an update of the normalisation statistics;
- one known toy task is solved (the roadmap's rung 2 lists it; 8.6 should too).

## Checked and sound (short)

Feet as one 49-way choice (no speed head to mask when standing); aim tokens
and entries scored in one softmax; the verb pointer over a per-class table;
the hand argument told the verb (HoK's "how" on "what", digest 01); the
reward-group outputs in principle (HoK); the clock as a terminal; the half-minute
look-ahead (gamma ~0.994 at 6 decisions a second) against five-minute matches,
with the outcome reached only through the value's bootstrap (6.5 says so).

## Not checked

The plane counts 17 / 18 and the 48 entry features (estimates; I counted
11 to 12 physics + 2 structure + 2 sight + 1 lanes in the near window,
consistent); whether the desktop GPU's memory holds a given batch (no spec
found in `docs/dev_setup.md`); whether MAPPO's code also drops dead agents
from the value loss (UNVERIFIED); the Lyu et al. and Pardo et al. citations
(RECALL).
