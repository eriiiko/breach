# Lens 2: would the trainer, built as written, learn correctly?

Critic: Opus 5.5, 2026-11-15. Object: `docs/rl_agent_rung2_design_2026-11-14.md`
at `67c00bee` (branch `192-rung2-cadence`), sections 2, 3, 4, 6 (as trainer
mechanics), 8. Read beside it: design `rl_agent_design_2026-11-11.md` sections
2, 6, 7, 8, 12, 13; lesson 0; digest 05 in full; the earlier lens 4
(`rl_agent_design_2026-11-11_critique/lens_4_network.md`). Read-only in the
main checkout: `src/rl_env.py` (step, reset, `_final`, `row_flags`, the module
doc), `src/simulation/action_spec_v1.py` (`build_masks`, `default_masks`,
`arg_mask`, `heads_used`, `masked_choices`), `src/simulation/reward_v1.py`
(groups, counter terms), `src/control_bots_door.py` (`LOSS_BITS`, the record
arrays, `read_record`), `src/control_policy.py` (the decoder's `prev`), and
the vector columns of `src/simulation/obs_spec_v1.py`. Nothing was run.

## Verdict

**Survives with fixes. 0 BLOCKER, 7 MAJOR, 14 MINOR.** The skeleton is sound
and leaves little room for the classic silent faults: the masks are stored
as served, the argument's mask and "which heads counted" are one function
each, the pointer is resolved against the stored list, five value outputs
match the env's five groups, the clock is a true ending, dead rows sit
correctly in grading, and the earlier lens 4's five majors are all
honoured. What does not hold is mostly AROUND that skeleton. One of the
three toy tasks cannot be passed by a correct trainer (a 20-decision memory
gap against 16-decision stretches) and reveals an unstated limit on what the
memory can learn. The copying step has three faults that its own measures
cannot see: the 2 % floor makes the copy plant its charge in about two duels
of three and throw its grenades at random; "agreement" can be won by
repeating the previous choice, which the observation hands to the network;
and the memory copy's training procedure is unsaid. Arm B's value-only
warm-up removes one cause of a wrecked copy and leaves three (advantage
normalisation, the entropy bonus, a fresh Adam), so section 6's answer could
be decided by trainer defaults. Which rows are learning rows is never said,
so a literal reading grades the opponent's rows, and the must-stay-0 masked
counter fires on rows nobody drives. And every test runs on toys; one live
check (recomputed log-probabilities equal the stored ones on the first
piece of every round) would guard the real run against the seam faults the
toys cannot see. Each fix is a few sentences or a few lines of test.

## Findings

### MAJOR 1. Toy task 3 cannot be passed by a correct trainer: the cue is 20 decisions back and learning sees at most 16

- **The design:** 4.5 "A stored batch is cut into stretches of 16 decisions
  ... Learning replays each stretch in order from the memory vector stored at
  its start". Section 8, toy 3: "a cue at the first decision, the answer
  asked for 20 decisions later". Section 8: "No run with the game starts
  before all of these are green."
- **What goes wrong.** Replaying from a stored memory vector carries the
  cue's *information* forward at acting time, but it carries no *gradient*
  back: the stored vector at a stretch start is a constant in the loss. The
  cue arrives at decision 0 and the reward for using it at decision 20; no
  stretch of 16 can hold both (the widest span inside one stretch is 15),
  wherever the stretch grid falls relative to the match start. So nothing
  ever teaches the GRU to WRITE the cue: the update that would make
  decision 0's memory keep the cue needs the loss at decision 20 to flow back
  20 steps, and truncation at 16 cuts it. A GRU with random weights forgets
  the cue's perturbation geometrically; there is no other pressure in the toy
  to keep it. A correct trainer therefore scores chance on this toy, and the
  gate of section 8 never turns green. The cheapest ways to "fix" a red toy
  under pressure are the wrong ones: lengthen the stretch for the toy alone,
  or let the value network or the stored vector leak the answer.
- **The design error under it.** The memory horizon a trainer of this kind
  can LEARN is the stretch length (16 decisions = 2.7 s), not the match. The
  document never says so, and toy 3 reads as if the author expected the
  stored vector to bridge stretches. That matters beyond rung 2: design 8.3
  keeps the GRU for rung 8's cones ("where did the enemy go"), a dependency
  that is often longer than 2.7 s. OpenAI Five's 16-step windows (digest 05 B)
  worked for their task; it is not evidence for ours.
- **How one would notice.** Loudly, at least: the toy stays red. The silent
  version is the fix someone applies to make it green.
- **Smallest fix.** In toy 3 put the answer at most 12 decisions after the
  cue (inside one stretch, with margin for the grid), and add a SECOND
  variant with the gap at 24 that is EXPECTED TO FAIL at stretch 16 and to
  pass at stretch 32: the pair proves that the stretch replay works and that
  its limit is where the theory says. State in 4.5: "the learnable memory
  horizon is the stretch length; the stretch is a swept number (16, 32, 64)
  before rung 8". Burn-in (R2D2, Kapturowski et al. 2019, [RECALL]) addresses
  stale states, not this limit; only a longer stretch does.

### MAJOR 2. The 2 % floor makes the copy plant its charge and throw its grenades at random, which contaminates "does it play" and arm B's start

- **The design:** 3.2 "The copy is therefore trained toward 'the bot's
  choice with 98 %, the other legal options sharing 2 %'", applied, by the
  wording, to every head that counted. Its stated purpose: sprint, reload and
  plant must not be driven to zero.
- **What goes wrong, worked.** The hand head of a healthy rifleman typically
  has 8 legal options (`build_masks`): nothing, reload (once the magazine is
  not full), melee, shove, plant (it carries one C4), and three throws (frag,
  smoke, flash, while carried); a ninth, use, near a door. The bot's label is
  "nothing" on about 99 % of decisions (3.3's own figure). Cross-entropy
  toward the smoothed target drives the network to put about 2 % / 7 =
  **0.29 % on each other legal verb, on every decision**. At 6 decisions a
  second:
  - plant: P(no plant in a 60 s duel, 360 decisions) = (1 − 0.0029)^360 ≈
    0.36, so in **about two duels of three the copy lays its C4 at its own
    feet**, with a timer from an argument head that never saw a plant label
    (the bots never plant), so whatever it generalises from the throws'
    reaches: anything from 5 to 60 s. The blast
    hurts the planter (design 2.1, #193);
  - throws: about 0.86 % per decision over the three kinds, so the copy's
    four grenades leave the hand along its facing at random moments over a
    duel; in a fireteam (4 soldiers, about 390 decisions each) that is
    several random frags and about four random C4s per team per match;
  - reload: about 0.29 % per decision, each one discarding the rounds left
    in the magazine (the Magazines row), with 3 magazines carried.
- **Why it matters for this lens.** (a) 3.3's "Does it play?" will read the
  floor's self-harm as the copy's quality. The design pre-explains a bad
  result as compounding error ("one small mistake puts the soldier somewhere
  its teacher never was") and names DAgger as the remedy, which cannot cure
  it: DAgger's labels are again "nothing", smoothed again. (b) Arm B starts
  from a policy with a built-in self-harm rate that arm A does not have, so
  the experiment of section 6 measures the floor as much as the copying; the
  first thing reward learning does in arm B is unlearn planting and random
  throws, which reads as "habits to unlearn" (the roadmap's question) when
  it is a choice of this document. (c) The floor is not a floor: label
  smoothing sets the probabilities only on the training states; off them
  (where reward learning goes) nothing holds them up.
- **How one would notice.** Only by watching replays closely or by counting
  plants and self-damage in the tournament; no listed measure counts them.
- **Smallest fix.** Separate the two purposes. Train the copy with no
  smoothing on the hand and argument heads (or a per-option floor of the
  order of 1e-5 there), keep a floor only where a random use is harmless
  (the feet's speeds); report "does it play" both as sampled and with the
  hand head taken greedily; count plants, throws and self-inflicted damage
  per match in the tournament line. If a floor is wanted during reward
  learning, put it INSIDE the choosing module as a mixture `(1 − e) p + e u`
  over the legal set, so acting and learning compute the same probability.

### MAJOR 3. The copying measures can be passed by a copycat: the previous choices are inputs

- **The design:** 3.3 "agreement with the bot's choice on held-back matches,
  per head and per option ... against 'always the most common option'"; the
  1 000-decision memorisation test; 3.2 "A memory can recover part of that
  from what the soldier did a moment ago."
- **What goes wrong.** The observation already hands the network what it
  did a moment ago: the vector's `prev` block (cols 71-76, the decoder's six
  decoded choices of the last decision, `control_policy` line 575), the held
  trigger (col 57, `live_trigger`), the facing target's offset (col 4), the
  body's velocity (cols 5-6), and on the list the `prev_aim` flag (col 66).
  The bots' choices are strongly autocorrelated at 4 ticks: the trigger is
  held for runs of decisions, the feet follow an A* path for seconds, the aim
  stays on the focus target. A network that learns "repeat the previous
  choice" scores high agreement on held-back matches, beats "always the most
  common option" by a wide margin on feet, aim and trigger, and memorises
  1 000 decisions trivially, while it never starts or stops firing at the
  right moment and never turns at a corner. This is the known failure of
  behaviour cloning with the past action in the input ("causal confusion",
  de Haan, Jayaraman, Levine 2019; "copycat" policies, Wen et al. 2020;
  both [RECALL], not in our survey). Its signature is exactly 3.3's last
  row: predicts well, plays badly; and the design's reserved remedy (DAgger)
  helps against it only slowly.
- **How one would notice.** With the listed measures, one would not: every
  number of 3.3 looks good. Only the tournament shows it, and the design
  pre-attributes a bad tournament to compounding error.
- **Smallest fix.** Two yardsticks and one split in 3.3: (a) add "repeat the
  previous decision's choice" as a baseline beside "the most common option",
  per head; (b) report agreement separately on CHANGE POINTS (decisions where
  the bot's choice differs from its previous one), where a copycat scores
  near zero; (c) a run with the `prev` block and `prev_aim` zeroed at the
  copy's input, as a control (the observation spec is unchanged; the
  network's input is masked). Whether the previous choice stays an input for
  the copy is then a measured decision, not an accident of the spec.

### MAJOR 4. Arm B's value-only warm-up does not protect the copy, and arm B differs from arm A in more than its start

- **The design:** 6, arm B: "the value network is new, so for the first
  rounds only the value network learns and the soldier's network is held
  still. Without this, the first grades are noise and they wreck the copy";
  arm C "only if B forgets the copy at once"; 12: "the value network learns
  alone first (6, arm B); arm C in reserve".
- **What the warm-up fixes.** Data collected by the frozen copy (it must be
  the copy that plays, which the text implies and should say) teaches the
  value network the copy's returns, so the first grades are no longer
  "return minus a random number". That part is sound and standard [RECALL:
  critic warm-up in fine-tuning practice].
- **What it does not fix: three wreckers the design leaves on.**
  1. **Advantage normalisation** (the 37 details, digest 05 A1; the design
     does not mention it, so the CleanRL default will be inherited):
     advantages are rescaled to unit spread per piece. When the copy is
     already near what it can do and the value is good, the true advantages
     are small and mostly noise; normalisation scales that noise to the same
     size the scratch arm sees, so the copy's first updates are as large as
     a random network's, in random directions. This IS "the first grades are
     noise", after the warm-up.
  2. **The entropy bonus** (0.01 per used head): a constant push toward the
     uniform. Arm A starts near the uniform and barely feels it; the copy
     starts far from it (98 % on one option) and is pushed off its choices
     by this term alone, whatever the grades. The two arms are not treated
     alike by the same coefficient.
  3. **A fresh Adam at full step size.** Adam's first steps move every
     weight by about the step size whatever the gradient's size (the bias
     correction makes the first moments sign-like): 4 passes x 4 pieces is
     16 such steps in the first round, up to about 5e-3 per weight against
     weights of order 0.05 [opinion, arithmetic]. The clip stops a sample's
     push only after its ratio has left the band; early stopping would stop
     the round, but its measure and threshold are not given (MINOR 3).
- **The arms' bookkeeping.** "Decisions needed to reach half the duels won"
  must count arm B's warm-up decisions, and the falling step size ("to zero
  over the run") must be defined for an arm whose policy starts learning
  late; otherwise the two arms run different schedules.
- **Arm C cannot be built or triggered from the text.** "Forgets the copy at
  once" has no measure. The pull is not defined: which direction of the
  divergence (toward the copy, `KL(copy || current)`, keeps the copy's
  support; the other direction lets the current policy drop options), over
  which heads (the used ones, at the stored parents), on which states (the
  batch's), inside or outside the clip, its weight and the fade. With
  memory, the copy's own GRU must be replayed along each stretch too, which
  doubles the learner's work. AlphaStar kept such a pull to its supervised
  policy for the whole of training [RECALL].
- **How one would notice.** One would see "B forgets the copy" and conclude
  that copying does not help (section 6: "either answer is a result"), when
  the trainer's defaults decided it.
- **Smallest fix.** In 6: arm B's warm-up stops on a stated condition (the
  value's explained variance on the copy's play above a number, or a fixed
  count of rounds, counted in B's budget); at unfreezing the policy's step
  size ramps up over the first rounds (or Adam's moments are warmed with the
  policy frozen); the entropy coefficient for B is 0 for the first rounds or
  is the run's swept number for both arms with the reason; "forgets" is a
  measured number every round (the mean `KL(copy || current)` over the
  batch's used heads, and the win rate against `bots_door`), and arm C's pull
  is written with the five choices above. Add the toy of MAJOR 6's list:
  "start from a policy that is already optimal on the toy; reward learning
  must not lower its return beyond noise over 20 rounds."

### MAJOR 5. Which rows are "learning rows" is never stated; read literally, the trainer stores the opponent's rows, and the must-stay-0 tripwire counts rows nobody drives

- **The design:** 4.1 "the network answers for every learning soldier of
  every match"; 4.2 names only two exclusions (a dead soldier's row keeps
  value and reward, an empty roster place is stored nowhere); design 6.3's
  flag table says "a living soldier ... everything". 7.1: "the count of
  masked choices sent (must stay 0)".
- **What the env does.** Every row of every team is observed, masked and
  rewarded: `row_units` and `row_flags` cover all teams, `build_masks` runs
  for every row, the reward is "for every row with `in_match` 1". In the
  duel the `bots_v1` row and in the range the `idle` targets' rows have
  `valid` = 1 and `in_match` = 1 exactly like the learner's. The actions of
  rows not on `policy` are ignored by the env.
- **What goes wrong.** A trainer that selects rows by the two flags, as the
  table says, runs the network on the opponent's rows too, draws "choices"
  for them, stores them with the opponent's real rewards, and grades them.
  Those choices had no effect, so their expected gradient is zero and their
  noise is not: in the duel half the batch is junk, in the range up to four
  fifths. The value network meanwhile learns the opponent's returns as a
  second task (harmless, but it shares the batch budget). At rung 3, with a
  frozen past self or a bot team in the pool, the same reading grades rows
  the current policy did not play, which is off-policy data inside an
  on-policy update. Nothing in section 8 has a row that is in the match but
  not learning.
- **The tripwire.** `masked_choices` counts every row of `actions`.
  `default_masks` leaves exactly (stand, "where I walk", release, nothing,
  arg 0) legal on a dead, turned or padding row, so a trainer that sends
  zeros for rows it does not drive (the natural thing) sends aim 0 (east),
  which is MASKED there: every padding row of the range's learner team
  counts on every step (S is the largest team of the run's levels, so with
  up to four targets the one learner's team carries up to three padding
  rows). And a
  `bots_v1` row winding up a throw has only "keep" legal. So the must-stay-0
  curve is non-zero from the first step for reasons that are not faults,
  and gets ignored, which is how a tripwire stops protecting.
- **Smallest fix.** One rule in 4.2: a stored decision is a row of a team
  the run file names as learning (`[run] policy` resolved per team), read
  through `BatchedEnv.team_rows`, with `in_match` 1; living rows get the
  policy loss, dead rows only the value loss; every other row is never
  stored. One rule for sending: every row the trainer does not drive gets
  the legal default `(0, AIM_WALK, 0, 0, 0)` (or "keep" where only "keep"
  is legal), and the 7.1 counter is read on driven living rows only. A toy
  with a second team whose rewards are the negative of the learner's catches
  a trainer that stores it.

### MAJOR 6. Every test of section 8 runs on toys; nothing guards the live run, and the toys cannot see the env's contract

- **The design:** 8 "three [toy tasks], each in a toy environment with the
  real layout's shapes ... solved in seconds without the game", the
  hand-worked grading case, the 1 000-decision copy test; "No run with the
  game starts before all of these are green." 2: "the probability of a
  choice can be recomputed later and compared (design 8.6)".
- **What the toys cannot catch.** The faults that live in the seam between
  the trainer and `BatchedEnv`, none of which a toy with "the real layout's
  shapes" reproduces unless it reproduces the API's behaviour too:
  - **env-owned buffers.** `obs`, `info["full_view"]`, `info["final_obs"]`
    and `info["final_full_view"]` are the same arrays on every call,
    overwritten by the next step, and `_final` WIPES last step's rows to
    zero (`rl_env.py` 145-153, 1644-1657). A collector that keeps a
    reference instead of copying (for the bootstrap's next state, or the
    batch's last full view) learns from the wrong arrays with no error;
  - **the in-step reset.** An env that ended returns its NEW match's tick-0
    observation and full view in the same call; the bootstrap and the
    memory reset must key on `info["ended"]` (MINOR 5);
  - **mask keys inside `obs`** under the `final_` twins, the `(N, U, 5)`
    reward in `REWARD_GROUPS` order, `rows_ended`, `team_rows`' offsets;
  - **number formats and GPU arithmetic** (float16 windows, autocast or TF32
    on the desktop GPU, batched inference against minibatched training):
    a toy on the CPU in float32 agrees with itself.
- **What also slips past the listed toys** (each a silent fault of the kind
  digest 05 D names): the group weights applied to the wrong axis or order
  (every toy's reward sits in one group); value rescaling with a group that
  is always 0 or rewards 100 times larger than 1 (every toy's rewards are
  order 1); a non-learning row in the match (MAJOR 5); the copied start
  being eroded by the update itself (MAJOR 4).
- **Smallest fix.** (a) One LIVE tripwire in every round of every run, a
  refusal and not a curve: on the first piece of the first pass, the
  recomputed log-probability of every stored choice equals the stored one
  (max difference under a stated tolerance, 1e-4 in float32) and the
  measured divergence is 0; plus the recomputed memory at each stretch's
  end equals the acting memory. This one check catches buffer aliasing,
  stale masks, a mismatched number format and any acting/learning
  divergence in the choosing module, on the real env, for free. (b) The toy
  environments implement `BatchedEnv`'s interface itself (owned buffers,
  in-step reset, `final_*` wiping, the reward groups, `rows_ended`), so the
  trainer under test is the one that will meet the farm. (c) Three more
  toys, each a few lines: rewards in the `own` group with an anti-correlated
  distractor in a weight-0 group; rewards scaled by 100 beside a group that
  is always 0; a learner row beside a non-learning row; plus the "already
  optimal start" toy of MAJOR 4.

### MAJOR 7. How the copy with memory is trained is not said, and the result "does the memory help?" depends on it

- **The design:** 3.2 "Without memory first, then with ... Both are trained;
  the difference is one of the results"; 3.1 "The records are unpacked once
  into one file per array on disk that is read in place, rows of dead
  soldiers and empty roster places left out"; 3.3 "Does the memory help?
  both of the above, memory on and off".
- **What goes wrong.** Supervised learning over a 45 GB on-disk set invites
  the usual loader: random decisions, or random windows of k decisions each
  started from a ZERO memory. The second is the common way to train a
  recurrent copy and it is wrong for this use: at play time (the tournament,
  and arm B's start) the GRU runs from the match's first decision to its
  last, about 390 decisions, and the network has only ever seen memories at
  most k decisions old and always started at zero mid-match. Either it
  drifts into states it never saw (and plays badly while predicting well),
  or it learns to ignore the memory because a zero start mid-match makes the
  memory unreliable (and memory "does not help"). Both read as findings
  about memory. A further trap: the unpacked arrays drop dead rows and are
  read in place, so whatever cuts the stream into sequences must still know
  where a soldier's match begins (memory to zero) and where it ends, and
  must not join two matches or two soldiers.
- **Also unsaid:** whether the copy's memory stretches match the trainer's
  (16 with a stored state, MAJOR 1), so that arm B's GRU continues in the
  regime it was trained in.
- **How one would notice.** Not from 3.3's numbers: they are computed the
  same way as the training, so they share the fault. Only the tournament
  would show it, and its reading is pre-explained (MAJOR 3).
- **Smallest fix.** In 3.2: the memory copy is trained on whole soldier-
  matches in order (about 390 decisions, small enough to unroll), or on
  stretches with the memory carried from the previous stretch of the same
  soldier-match exactly as the trainer of 4.5 does it (states stored once
  per epoch); the unpacked set carries a per-decision "match index, soldier,
  decision number" so sequences can be rebuilt; held-back agreement for the
  memory copy is measured by running each held-back match from its start.

### MINOR

1. **Advantage normalisation is not mentioned, so its set is undefined.**
   It must come AFTER the weighted sum of the group advantages (normalising
   each group first and then weighting makes the weights meaningless and,
   for a group that is always 0, divides 0 by its floor), over the decisions
   that enter the policy loss only (living learning rows; dead rows' steps
   have advantages but no choice and would skew the spread), per piece of
   whole stretches. One sentence in 4.3.

2. **Value rescaling is half specified.** (a) Running mean and spread
   (MAPPO's value normalisation) moves the network's de-normalised outputs
   whenever the statistics move; PopArt (van Hasselt et al. 2016, [RECALL])
   corrects the last layer so they do not. Lens 4's proposed test "the
   de-normalised weighted value sum is unchanged by an update of the
   statistics" holds only for PopArt and was not carried into section 8;
   say which one is built. (b) The order: grades computed with the
   collection-time statistics, statistics then updated with this batch's
   targets, the loss on targets normalised with the new ones; write it
   down, it is the usual place for an off-by-one-batch. (c) A spread floor:
   in rung 2 friendly fire and order are exactly 0 on every row, so their
   targets are 0/0 without one. (d) 7.1's "how much of the return each
   output explains" is undefined for a group that is always 0; print it
   only for groups with spread. (e) With per-group normalisation the
   weight-0 `team` output costs the shared value trunk as much loss as
   `own`; decide whether weight-0 outputs train at all in rung 2.

3. **The clip's object and the early stop's measure are open.** PPO's ratio
   can be the JOINT ratio (the product over the used heads, which is what
   the "sum of log-probabilities" of lesson 0 section 10 implies) or one
   clipped term per head. They differ a lot: 0.2 on a joint of up to five
   heads lets each head move about 4 %; per-head clipping lets the joint
   move up to 1.2^5 = 2.5 times. 4.4's "stopped early when the policy has
   moved too far" needs its measure (the approximate KL of the joint, from
   the stored and recomputed log-probabilities of the used heads) and a
   threshold. Both are a line in 4.4.

4. **Design 8.2's "every head on a soldier that could not act" is silently
   dropped.** `heads_used` counts a head as used when it has more than one
   legal option; a living soldier that is knocked down or stunned still has
   49 legal feet, two triggers and every aim legal, so those choices are
   graded and pushed by the entropy term though they act on nothing. The
   cost is variance, not bias. Section 14 rightly says "`heads_used` and
   `arg_mask` decide which heads counted, never a second rule"; then design
   8.2's sentence should be amended to match, so the next reader does not
   build the second rule.

5. **The bootstrap and the endings, said exactly.** 4.3 is right for two
   teams, but the trainer is "written for teams": (a) the value after the
   batch's last stored decision is `V(info["full_view"])` of the batch's
   last step, COPIED (env-owned), and 0 if that env ended on that step (its
   returned view is then the next match's); (b) with three or more marine
   teams, `rows_ended` is a row's end while its env goes on (rewards zero
   after it, design 6.6): the row's trajectory ends there and restarts at
   its env's next match; (c) the trainer refuses a run whose profile is
   `"counters"`, where `truncated` would need a bootstrap from
   `final_full_view`. The hand-worked case should hold: an ending on the
   batch's last step, a dead row of a two-soldier team receiving team and
   outcome rewards after its death, five groups two of them at weight 0, a
   partial last stretch.

6. **"Same number format" covers the inputs, not the arithmetic.** Acting
   runs batches of the farm's rows and learning runs pieces; autocast,
   TF32 and convolution algorithm choice can differ between the two shapes
   on the GPU. Write: one precision setting for acting and learning, the
   normalising layers are LayerNorm (design 8.3 says "a normalising layer"
   without naming it; a batch norm would differ between acting and
   learning), no dropout, and the tolerance of MAJOR 6's tripwire.

7. **One entropy coefficient over heads of very different sizes.** The most
   a head can contribute is ln 49 = 3.9 (feet), ln(18 + E) about 4.0 (aim),
   ln 13 = 2.6 (hand), ln 5 = 1.6, ln 2 = 0.69 (trigger). The trigger, the
   head that matters most in a duel, gets the weakest push and may lock
   early; the feet stay noisy longest. Either a coefficient per head or the
   entropy divided by ln(legal options); "entropy per head" (7.1) is the
   instrument. Also: a masked entropy must not compute 0 x (−inf) (a NaN,
   loud, but a common first bug).

8. **The discount contradicts design 6.5 and is the number most likely to
   matter for the duel.** 0.99 is about 17 s; design 6.5 says "at first a
   soldier looks about half a minute ahead" (about 0.9945; lens 4 wrote
   0.994). At 0.99 the clock's −3 in a 60 to 90 s duel is invisible until
   its last ~17 s and the +10 outcome nearly so, so the duel is learned
   almost wholly from the `own` group; digest 05 calls the discount "one of
   the most important" numbers and says long fights may want 0.995. Make
   0.99 / 0.995 the duel's first sweep rather than rung 3's.

9. **Two networks, one gradient clip?** "Gradient norm 0.5" with one
   optimiser over both networks (the habit of CleanRL's shared agent)
   lets a large value gradient (an outcome spike) shrink the policy's step
   through the global norm, silently. Separate optimisers and separate
   clipping; that is also what makes arm B's "policy held still" clean.

10. **Strength rounds (7.3) and the training farm.** The held-back rooms are
    other levels, so another run file: either a second `BatchedEnv` (memory
    on the 16 GB desktops) or a reset of the training farm. In the second
    case every row's memory must be zeroed and a new stretch begun, since a
    `reset` reports no `ended`; and the episode counters move, so the
    training sequence is no longer the one a resume would replay. Say which.

11. **Agreement of the conditional heads is overstated under teacher
    forcing.** Measured as in training, the hand and argument heads are
    scored given the BOT's aim. Report also the joint agreement (all used
    heads right) with the network's own aim and verb chosen first.

12. **Lossy labels and the stateful teacher.** The records carry loss bits
    (`carried`, `superseded`, `aim_missing`, `throw_rounded`, ...). The
    label is what was sent, which is right for copying `bots_door`, but a
    `carried` hand label was decided by the bot on a state up to three ticks
    after the observation it is paired with. Report agreement split by the
    loss bits, and run one control without the `carried` / `superseded`
    labels. For the reserved DAgger: the capture proxy's bot keeps its
    committed position and plan as if its decisions had been played; with
    the network driving they were not, so its labels refer to a plan the
    soldier is not following. Note it before building.

13. **Stale states, partial stretches, a match end inside a stretch.**
    After the first pass the stored start state comes from the acting
    weights (standard and acceptable; a burn-in switch is a few lines). The
    decisions per env per round must be a multiple of 16, or the last
    stretch is padded and masked. 4.5's "the memory is zero at a match's
    first decision" must say "also inside a stretch, during the replay".

14. **"One choosing module" proves less in the copying path than 2
    claims.** Copying is teacher-forced: it exercises the learning half
    (the masks, the used heads, conditioning on given choices), never the
    acting half (sequential drawing, the aim's embedding gathered from the
    drawn index). "A fault ... shows up in the cheapest of the three" holds
    for masks and used heads, not for acting against learning; only the
    tripwire of MAJOR 6 checks that.

## Holds

- **The joint probability is recomputable from what 4.2 stores.** The four
  masks are stored as served (they are `obs` keys, built once at the
  decision tick), the argument's mask is a pure function of the drawn hand
  and aim (`arg_mask`), and "which heads counted" is one function of those
  (`heads_used`); no head is ever all-masked (`default_masks`, "nothing" and
  the compass tokens never masked). The pointer is an index into the STORED
  list, turned into a game id only by the decoder at the decision
  (`set_decisions` against `list_uid`), so a later re-sort cannot change
  what a stored index meant. The previous aim reaches the next observation
  as a flag on its entry and as the collapsed token `AIM_TOKENS`, never as
  a stale list index (`control_policy` line 395; lens 4's minor 4 honoured).
  A masked in-range choice cannot be drawn by a module that draws under the
  masks, the env counts any that is sent, and the door bots' records carry
  none (`DoorRefused`), so no label is masked.
- **Grading per group, then weighting** equals grading the weighted reward
  against the weighted value: the λ-return is linear, so it holds for any
  weights (zeros included) provided every group uses the same discount,
  λ, endings and DE-normalised values. Five outputs fixed to the env's
  five groups keep a moving weight from making a moving target (lens 4's
  major 1 honoured).
- **The clock as a real ending** is right here: `time_left` is in the
  soldier's vector (col 77) and in the full view's match block, so the
  state is Markov in the time left; the env reports every end as
  `terminated` under a named profile (`rl_env.py` 876-878), so
  `final_full_view` is never needed for a bootstrap. Digest 05's item 7
  (bootstrap at the limit) is for an unseen limit and does not apply.
- **Dead rows on the books** (value and reward, no choice) are the simplest
  correct arrangement for GAE with team and outcome rewards after death,
  and the env provides exactly the two flags it needs (lens 4's major 2
  honoured). In rung 2's one-soldier teams a death is an elimination, so
  the machinery is exercised only by toy 3 and the hand case, which is
  where it should be tested first.
- **A separate value network** sharing no weights: nothing it sees can
  reach a choice, and the "scramble the extra inputs" test checks it.
- **Synchronous, with the learner refusing a batch from other weights**:
  the right shape for a lock-step farm, and the header is what a later lag
  correction needs.
- **The range without memory, the duel with it**: the recurrent faults meet
  a trainer already proven on a task (lens 4's advice taken).
- **Storage and number formats** (63 KB a decision, float16 and bytes as
  served, the value network re-reading the soldier's windows): lens 4's
  major 3 honoured; its major 4 (fixed squash, normalising layers) is in the
  spec; its major 5 (memoryless critic) is deferred with the right reason.
- **The list's size changes between levels** (E = S + 33: ships, duel rooms
  and the range differ): the pointer head and attention handle any E as
  long as design 8.6's shuffle and empty-slot tests pass, which rule out any
  per-position weight. So a copy made on ships can start arm B in the duel
  rooms; the spec-name check on loading is enough.
- **Not pressed:** step size 3e-4 with a fall to zero, grad norm 0.5,
  λ 0.95, clip 0.2, 4 passes in 4 pieces, batch 8 000: each is a common
  default with a sweep named, and I found no evidence in the digest that
  any is wrong for this task beyond the discount (MINOR 8) and the
  unspecified early stop (MINOR 3).

## Not checked

- Whether network v1 as code exists or what it looks like (nothing is built;
  every finding is against the text).
- The magnitude of GPU non-determinism between acting and learning shapes
  on the desktops (MINOR 6 and MAJOR 6's tolerance are by experience, not
  measured).
- The frequency of each hand verb in the records and the bots' actual
  throws of smoke and flash: MAJOR 2's arithmetic uses the masks' legal set
  and 3.3's "nothing 99 %", not counted records.
- How autocorrelated the bots' choices are at 4 ticks (MAJOR 3 rests on the
  bots' design: held triggers, A* paths, a committed position), not on a
  count.
- The recalled sources (de Haan et al. 2019, Wen et al. 2020, R2D2, PopArt,
  AlphaStar's pull to its supervised policy, critic warm-up practice): not
  re-read this session.
- Statistics, experiment design and cost: the other lenses'.
