# Lesson 2: the trainer, line by line

2026-11-18. The lesson document of rung 2 of the RL agent ladder. Lesson 0
(`docs/rl_lesson_0_learning_from_reward_2026-11-11.md`) gave the idea; this
one follows one training round through our own code, in the order it runs,
so that a run's report can be read and its numbers judged. Every claim about
what the code does points at the line on `main` (commit `d87497fc`) as
`file:line`; numbers carry their source.

Recall, in one paragraph. A soldier's network turns an observation into
chances for each of its five choices, and the game draws from those
chances. After a choice we ask whether more reward followed than the value
network expected; if so the choice is made a little more likely, if not a
little less. PPO adds a limit: within one batch of play, no choice may move
far from the chance it had when it was made. That is the whole method. What
follows is how our code does it.

## 1. What the trainer is made of

| file | what it is | its one job |
|---|---|---|
| `agents/trainer.py` | the round loop | refuses what it cannot train, then runs collect, learn, hand over |
| `agents/collector.py` | the collector | plays one round of decisions with fixed weights and fills the batch |
| `agents/batch.py` | the batch | the one layout of what is kept per decision |
| `agents/choosing.py` | choosing | turns raw network outputs and masks into chances, a draw and a probability |
| `agents/grading.py` | grading | pure arithmetic: from rewards and value estimates to a grade per decision |
| `agents/value_scale.py` | the value scale | keeps the value network's targets near size 1 |
| `agents/learner.py` | the learner | grades, moves the scale, runs the live check, steps both networks |
| `agents/saved.py` | saving | the one writer and reader of a run's state on disk |
| `agents/settings.py` | the settings | every number the trainer uses, in one record |
| `agents/network.py`, `spec.py`, `sizes.py` | the two networks | the soldier's network, the value network, and the sizes asked from the game |

Two words used throughout. A **row** is one roster place of one team in one
match: the batch is a rectangle of rows by time steps. A **decision** is one
row at one time step (a sixth of a second of play).

**The boundary to the game.** The trainer meets the game only through
`BatchedEnv` in `src/rl_env.py`: `reset`, `step`, and `team_rows`, which
hands back views of one team's rows (`src/rl_env.py:1683-1694`). No
Simulation is ever touched. The environment's arrays are its own buffers and
are overwritten at the next step, so everything the trainer keeps is copied
out: `LearningRows.gather` ends in `np.copyto` (`agents/batch.py:131`). In
the tests a stand-in environment (`agents/standin.py`) serves the same names.

**The two networks.** Both are built from the same code with their own
weights (`agents/network.py:281-334`, class `Body`): a reader of the
soldier's own vector, one reader per kind of list entry, two attention
layers over the list, two map readers of four convolutions each, and a join.
The soldier's network adds the memory (a GRU cell, section 5) and the five
heads; the value network reads the full view instead of the soldier's own
list and ends in one output per reward group (`agents/network.py:519-550`).
Sizes at full width: the soldier 1 770 480 weights, the value network
1 174 701 (plan, "As built (2a)"). The widths are `NetSpec`
(`agents/spec.py:32-49`); every size of the game's (list length, number of
map planes, number of hand verbs) is asked from the game's spec tables by
`agents/sizes.py`, never typed.

**Before anything runs**, `check_run` refuses by name what this trainer
cannot train (`agents/trainer.py:65-98`): a reward that is not a named
profile, an observation or action spec other than "v1", the full view
switched off, a network whose record does not fit the layout, no team on
the `policy` controller, and (with the memory on) a `decisions`, `piece` or
`lump` that is not a whole number of stretches.

**One round** is four lines (`agents/trainer.py:125-132`):

```python
batch = self.collector.collect(self.batch)
stats = self.learner.learn(batch)
self.collector.set_weights(self.learner.weights(),
                           self.learner.version)
```

The batch is allocated once and overwritten every round
(`agents/trainer.py:113`). The collector has its own copy of the soldier's
network with gradients switched off (`agents/collector.py:141-143`); the
learner owns the real one. One hand-over each way: weights in, a batch out.

## 2. Collecting

`Collector.collect` (`agents/collector.py:263-322`) plays `decisions` steps
(T, default 128: `agents/settings.py:36`) in every match at once. Per step:

1. **Copy the view in.** The learning rows of the served observation and
   full view are copied into the batch's slot for this step
   (`agents/collector.py:278`, through `_gather_step` at 190-196). The
   network then acts on *that slot*, not on the environment's buffer, so
   what was acted on is exactly what will be learnt from.
2. **Two flags are written before acting** (`agents/collector.py:279-280`):
   `first` (this decision opens a match for this row) and `value_row` (the
   row is in its match and its story has not ended; section 3).
3. **Choose.** The soldier's network reads the slot, and `choosing.draw`
   draws the five choices under the slot's masks
   (`agents/collector.py:198-213`).
4. **Fill in the rest and step.** Every row the trainer does not drive is
   sent the legal default: stand, aim where I walk, no trigger, hand nothing,
   argument 0 (`agents/collector.py:114-125`, used at 289-291). The step
   runs (`:293`).
5. **Check.** The environment counts any masked choice it received;
   on a driven living row anything but 0 stops the run with
   `MaskedChoiceSent` (`agents/collector.py:329-343`).
6. **Store the outcome.** The step's rewards and endings are copied into the
   batch, and the bookkeeping moves (`agents/collector.py:296-308`).

Between rounds nothing is reset: the next round continues the matches as
they stand (`agents/collector.py:24-25`).

**How a choice is drawn.** `choosing.py` is the one place chances are made.
For one head (`agents/choosing.py:90-112`), a forbidden option's raw score is
set to minus infinity before the softmax:

```python
logp = torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
```

so its chance is exactly 0, when acting and when learning. The five heads
are walked in the action spec's order, each later head told the earlier
choices (`agents/choosing.py:149-154`): feet, aim, then trigger and hand
given the aim, then the argument given aim and hand, whose mask is worked
out from the drawn hand and aim (`arg_mask`, `:64-79`). The probability of
the whole decision is the product of its heads' chances, so its logarithm
is the sum, but only over the heads that **counted**: a head with only one
legal option contributes nothing (`agents/choosing.py:155-157`). "Counted"
is the game's own rule, `action_spec_v1.heads_used`, batched here
(`agents/choosing.py:82-87`). `draw` and `evaluate` are the same walk; one
picks by drawing, the other takes given choices (`agents/choosing.py:164-180`).
That is why acting and learning cannot disagree about a probability, and the
live check (section 4) proves they do not.

**A near-even start.** The last layer of every head starts at 0.01 of
PyTorch's default size (`agents/network.py:72`, applied at 428-433), so a
fresh network gives every legal option about the same chance (2b note,
section 6). Learning starts from "undecided", not from an accident of the
random weights.

**What is stored per decision, and why exactly that** (`agents/batch.py:11-40`,
arrays at 174-201):

| kept | why |
|---|---|
| the observation as served (vector, list, list mask, the two map windows) | learning must recompute the network's answer to exactly what it saw |
| the four masks, and the argument's mask the draw worked out | the recomputed chances must be under the same masks, or the ratio of section 4 is meaningless |
| the value network's full view | the value network grades from it |
| the five choices, which heads counted | the choice being judged |
| the probability of each counted head and of the whole decision (`logp_heads`, `logp`) | the denominator of the ratio; PPO needs "the chance it had when it was made" |
| the five reward numbers paid by the step | grading |
| `valid`, `in_match`, `done`, `first`, `value_row` | which rows take part in what (section 3) |
| `memory` (only with the memory on) | the memory each decision started from (section 5) |

Once per batch, the view after the last step is also kept (`last_obs`,
`last_full`, copied at `agents/collector.py:324-327`): the value network's
estimate of it stands in for everything after the batch's end.

**Whose rows are stored.** Only the learning teams', the teams the run file
puts on `policy` (`agents/trainer.py:54-57`). The batch has R = N matches ×
L learning teams × S roster places rows, at fixed places
(`agents/batch.py:103-116`). The opponent's soldiers are observed and paid
by the environment too, but they are never stored, drawn for or graded: a
grade for a choice that had no effect would be pure noise (rule in
`.claude/rules/agents.md`, "Stored decision and batch").

**The weights' version.** When a round's collecting starts, the collector
writes the batch's header: the version of the weights it played with, the
run file's and the training file's sha256, the build id
(`agents/collector.py:272`; fields at `agents/batch.py:83`). The learner
refuses a batch whose header differs from its own, naming the field
(`agents/learner.py:296`, `agents/batch.py:213-224`). In rung 2 both sit in
one process, so this can only fire on a fault; it is what will let a later
form spread collecting over several PCs and still know which play belongs
to which weights. `learn` raises the version by one
(`agents/learner.py:422`).

## 3. Grading

**Which rows take part.** Two kinds of rows, set from the stored flags
(`agents/learner.py:310-311`):

- **value rows**: `value_row` is 1. The row is in its match and its story
  has not ended. A dead soldier is a value row: the value network learns
  what the death was worth.
- **policy rows**: value rows that are also `valid` (a living soldier). Only
  these have their choices graded and learnt from.

An empty roster place is neither. A row's "story" ends when its match ends
or when the environment says its records ended (`rows_ended`, possible with
three or more teams); the collector carries that across rounds
(`agents/collector.py:304-308`).

**The value network's estimates.** With the scale as it was while
collecting, the value network estimates every value row and the view after
the last step, in real reward units, without gradients
(`agents/learner.py:265-287`). There are five estimates per decision, one
per reward group: `outcome`, `own`, `team`, `friendly_fire`, `order`
(`src/simulation/reward_v1.py:77`).

**The method: generalised advantage estimation** (Schulman et al. 2016,
lesson 0 section 6). The code (`agents/grading.py:57-63`), per group and row,
walking backwards in time:

```python
nxt = last_value if t == T - 1 else v[t + 1]
keep = goes_on[t][:, None]
delta = reward[t] + gamma * keep * nxt - v[t]
a = delta + gamma * lam * keep * after
```

In symbols, with r(t) the reward paid by the step, v(t) the estimate, g the
discount (gamma), l the blend (lambda) and k(t) = 1 − done(t):

    δ(t) = r(t) + g·k(t)·v(t+1) − v(t)
    A(t) = δ(t) + g·l·k(t)·A(t+1),     A(T) = 0
    target(t) = A(t) + v(t)

δ(t) is the one-step surprise: what this step paid plus what the value
network now expects from the next view, minus what it expected from this
one. Unrolling the recursion inside one match:

    A(t) = δ(t) + (g·l)·δ(t+1) + (g·l)²·δ(t+2) + ...

The two ends of lesson 0's table fall out. With l = 0, A(t) = δ(t): the
short way (one real reward, then trust the value network). With l = 1 the
sum telescopes (the +g·v(t+1) in δ(t) cancels the −v(t+1) of δ(t+1), which
carries the factor g), and what is left is

    A(t) = r(t) + g·r(t+1) + g²·r(t+2) + ... − v(t)

the long way: the real return minus the expectation. Our l = 0.95 lies
between. At g = 0.99, l = 0.95 the surprises are weighted down by
g·l = 0.9405 per decision, so a grade is mostly made of the next 1 / (1 −
0.9405) ≈ 17 decisions' surprises (3 seconds), while the value estimates
inside them look 1 / (1 − 0.99) = 100 decisions ahead (`agents/settings.py:38-40`).

`k(t)` does the endings: after a match ends nothing is still to come,
whatever the view returned (it is already the next match's,
`agents/grading.py:16-18`). The only place the value network's guess stands
in for real reward that was never seen is the batch's end in the middle of
a match (`last_value`). A step that is not a value row has its estimate set
to 0 and its grade and target forced to 0 (`agents/grading.py:54, 62, 64`).
The target is what the value network learns toward; it is the blended
return (the "lambda-return").

**The reward groups.** The five grades are added with the group weights
(`agents/grading.py:68-76`). Rung 2's weights are outcome 1, own 1, the
other three 0 (`agents/settings.py:21-22`). All five value outputs still
learn, the weight-0 ones too (`agents/learner.py:25-29`): rung 3 needs them.

**The common size.** Before a piece is learnt from, its grades are brought
to mean 0 and spread 1 over that piece's policy rows
(`agents/grading.py:79-85`, called at `agents/learner.py:358-359`):

    grade' = (grade − mean) / (spread + 1e-8)

This makes the step's size independent of the reward's units, and it means
"better than expected" is judged against the piece's own average: within a
piece, the upward pushes and the downward pushes add up to zero.

**The value scale.** The value network's last layer gives a normalised
estimate; the real one is `spread × normalised + mean`, per group
(`agents/value_scale.py:59-66`). The reason: the outcome group pays ±10 at a
match's end, own pays a few tenths, and two groups pay nothing at all. A
network learns best when what it predicts is near size 1. Each round the
order is fixed (`agents/learner.py:306-324`): grade with the scale as it
was; move the scale toward this batch's targets
(`agents/value_scale.py:81-88`); learn on targets normalised with the new
scale. The move is a moving average with share b = 0.05 a round
(`agents/settings.py:61`), corrected for its zero start the way Adam
corrects its moments:

    m1 ← (1 − b)·m1 + b·mean(targets),   mean = m1 / (1 − (1 − b)^n)

after n moves; likewise for the mean square, and spread = √(m2 − m1²), never
under 0.01 (`agents/settings.py:62`) so that an always-0 group does not
divide by nothing. On the first move the correction makes the mean exactly
that batch's mean. The scale then remembers about 1/b = 20 rounds.

After every move but the first, the last layer is changed so that every
real-unit output is what it was before (the method is PopArt, van Hasselt
et al. 2016; `agents/value_scale.py:91-98`):

    W' = W·s_old / s_new,    b' = (s_old·b + m_old − m_new) / s_new

**Why the first move leaves a fresh network alone**
(`agents/value_scale.py:89-90`, and the reason at 22-34). On the first
move the scale jumps from (mean 0, spread 1) to the batch's own; in 2b's
"scale" task that was about 690 and 540. Keeping a fresh network's outputs
through that jump would divide its last layer by 540 and set its bias far
from 0. Adam moves each weight by about one step size per step whatever the
gradient's size, so climbing back took thousands of steps: the task sat at a
random player's score for 300 rounds (plan, "As built (2b)"). A fresh
network has learnt nothing worth keeping. Left alone, its output (near 0 in
normalised units) reads as "the batch's mean", which is the best first guess
there is. With the fix the task passes in 63 to 92 rounds (same source).
A scale is saved and loaded with its value network, so a resumed run does
not make a "first" move again (`agents/value_scale.py:32-34`).

## 4. The update

**Passes, pieces and lumps.** The learner makes `passes` = 4 passes over the
round's value rows (`agents/settings.py:44`). Each pass shuffles them afresh
(`agents/learner.py:347`) and cuts them into the fewest near-equal **pieces**
of at most 2 048 decisions (`agents/learner.py:162-171`; `piece`,
`agents/settings.py:49`). Each piece is one step of each network. Never a
short last piece: under Adam a step taken from three left-over decisions is
as long as one taken from two thousand, and its grades, brought to a common
size over three rows, are noise (`agents/learner.py:164-167`). A piece is
worked through in **lumps** of 512 (`agents/settings.py:51`) whose gradients
are added before the step: each lump's loss is divided by the piece's
number of policy rows, so the sum is the piece's mean
(`agents/learner.py:578`). The lumps exist because a piece of 2 048 does not
fit an 8 GB card at once (about 2.6 MB per decision while learning: plan,
"As built (2a)"); the step is the same either way.

In the 2c run (64 matches × 128 decisions = 8 192 decisions a round, one
soldier per match) that is 4 pieces × 4 passes = 16 steps a round.

**The soldier's loss, term by term** (`agents/learner.py:573-578`):

```python
log_ratio = d.logp - old
ratio = log_ratio.exp()
clipped = ratio.clamp(1.0 - s.clip, 1.0 + s.clip)
objective = torch.minimum(ratio * adv, clipped * adv)
variety = d.variety.sum(dim=-1)
loss = (-objective.sum() - s.variety * variety.sum()) / n_pol
```

In symbols, over the n policy rows of a piece:

    ρ = p_now(choice) / p_then(choice)
    loss = −(1/n) Σ [ min( ρ·A, clip(ρ, 1 − ε, 1 + ε)·A ) + c·Σ_heads V ]

- **ρ, the ratio**: the probability of the whole stored decision under the
  weights now, over the probability it had when it was drawn. `d.logp` comes
  from `choosing.evaluate` under the stored masks (`agents/learner.py:563`),
  `old` is the stored `logp`. It is 1 before any step. One ratio per
  decision, for all five heads together (the "joint" ratio).
- **A** is the decision's grade after the common size.
- **The clip**, ε = 0.2 (`agents/settings.py:42`). The minimum picks the
  smaller of the two terms. With A > 0 the term stops growing once ρ passes
  1.2, so the gradient vanishes there (lesson 0's worked case: a choice made
  at 0.20 stops being pushed at 0.24). With A < 0 it works the other way: at
  A = −1 and ρ = 0.7, ρ·A = −0.7 and the clipped term is −0.8; the minimum
  is the constant −0.8, so a choice already 30 % less likely is pushed no
  further down. The minimum never removes a correction: if a step has made a
  bad choice MORE likely (A < 0, ρ = 1.3), ρ·A = −1.3 is the smaller and its
  gradient stays.
- **V, the variety**: each counted head's entropy as a share of the most it
  could have, its entropy over the logarithm of its number of legal options
  (`agents/choosing.py:115-122`). So the trigger (2 options) and the feet (49)
  are measured on one scale, 0 for a sure head, 1 for an even one. The
  weight c = 0.01 per counted head (`agents/settings.py:56`). It rewards
  staying undecided a little, so that an option is not dropped before it has
  been tried.
- The minus sign: the optimiser lowers the loss, so it raises the objective.

**The value loss** (`agents/learner.py:598-606`): the squared error of the
normalised estimates against the normalised targets, summed over the five
outputs, averaged over the piece's value rows. The two networks share no
weights and each has its own Adam, so no weighting between the two losses
is needed.

**The distance, and the early stop.** For every piece the learner measures
how far the soldier's chances have already moved this round, from the same
forward pass the step uses (`agents/learner.py:588`, divided at 392):

    distance = mean over the piece's policy rows of ( log p_then − log p_now )

Since the stored choices were drawn from p_then, this is a sample of
Σ p_then·log(p_then / p_now), the Kullback–Leibler divergence of the new
chances from the old: 0 when nothing moved, growing as the policy moves.
The first piece whose distance exceeds `max_distance` = 0.02
(`agents/settings.py:47`) is not stepped, and the soldier takes no further
step this round (`agents/learner.py:408-411`). The value network finishes
its passes (`:417-420`). This is the second guard beside the clip: the clip
limits each decision, the distance the policy as a whole. In the 2c run the
early stop cut 51 of 360 rounds, at a mean of 15.3 of the 16 steps (2c
report, section 6).

**The step.** Each network's gradient is cut to a total length (norm) of 0.5
by itself (`agents/learner.py:413-414` and `:417-418`;
`agents/settings.py:58`), so a spike in one cannot shrink the other's step.
Both use Adam with step size 3e-4 (`agents/settings.py:53`), falling in a
straight line to zero over `planned_rounds` (`agents/settings.py:115-119`,
set each round at `agents/learner.py:299-302`). The plan is in planned
rounds, so `planned_rounds` must be the run's real length: a run cut short
by the wall stop never reaches its small steps (`agents/settings.py:83-84`).

**One arithmetic.** Acting and learning use 32-bit arithmetic with the
card's faster-but-different options switched off
(`agents/learner.py:153-159`), so the probability recomputed at learning is
the one computed at acting.

**The live check, every round** (`agents/learner.py:174-208`, called at
`:393-400`). On the first piece of the first pass, before any step:

1. the recomputed log-probability of every stored head choice equals the
   stored one to 1e-4 (`live_tolerance`, `agents/settings.py:67`);
2. which heads counted, and the argument's mask, are recomputed exactly
   equal;
3. the distance is 0 to 1e-4;
4. with the memory on, the memory recomputed at every stretch's end equals
   the stored one to 1e-4 (section 5).

Any failure stops the run with `LiveCheckFailed` naming the clause. It
catches a kept reference to an environment array (the batch would hold the
next step's view), masks that differ between acting and learning, a number
format that differs, and any difference between the two uses of
`choosing`. It cannot be switched off. In the 2c run its largest gap over
360 rounds was 4.8e-6 (2c report, section 6).

**The value-first rounds** (`value_first_rounds`, default 0:
`agents/settings.py:78`). In a run's first such rounds the value network and
its scale learn, the live check runs, and the soldier takes no step at all;
its optimiser is not touched, so its weights stay the same to the bit
(`agents/learner.py:404-407`). This is for a soldier that starts good (a
copy of the bots) beside a fresh value network: without it, the first
rounds would grade a good soldier with a value network that knows nothing.
In the "good start" task neither 0 nor 5 such rounds wore the start down
(2d note, section 8a).

## 5. The memory

**The cell.** With the memory on, the soldier's network carries a vector of
256 numbers (the `join` width, `agents/spec.py:42-49`) from decision to
decision: a GRU cell reads the joined view and the old memory and writes
the new one (`agents/network.py:465-475`). The memory is set to zero where
`first` says a match starts, inside the network, and nowhere else. Two
details of how it starts, both from findings of 2d.1b:

- **A spread of time scales** (Tallec and Ollivier 2018;
  `agents/network.py:78-96`). Unit i starts keeping T_i / (T_i + 1) of
  itself per decision, with T_i spread evenly in the logarithm from 1 to 32
  decisions (`agents/network.py:75`). PyTorch's own start keeps half per
  decision, so a mark is gone in a few decisions before any reward can teach
  the cell to keep it. Measured: 7 times more of a cue's mark is left after
  12 decisions, 27 times after 24 (2d note, section 8b).
- **The heads read the memory through a normalising layer**
  (`memory_norm`, `agents/network.py:405-413`). The cell's output starts
  about 8 times smaller than the join; read raw, a cue was learnt through
  the memory about twice as slowly as by a network without memory
  (`agents/network.py:407-412`; 2d note, section 8b).

**What the collector stores.** At every step the memory carried IN (before
the decision, before the network zeroes it at a `first`) is stored as
`memory[t]`, and the soldier acts from that stored copy
(`agents/collector.py:215-226`); after the last step, `memory[T]`
(`:318-319`). The memory is carried across steps and across rounds: a match
does not end with the round.

**The stretch.** Learning cannot replay every match from its start: that
would mean running hundreds of decisions in order for every piece. So it
replays **stretches** of at most `stretch` = 16 decisions (2.7 seconds;
`agents/settings.py:73`) of one row, in order, from the stored memory at the
stretch's start (`agents/learner.py:515-517`). That stored memory is a
constant: the gradient stops at the stretch's start. The consequence, which
design 4.5 states plainly: **what the memory can learn to hold is one
stretch long.** It may carry more when acting, but nothing teaches it to.
With the memory on, the stretch is the unit a pass shuffles and cuts (a
piece of 2 048 decisions is 128 stretches), which is why `decisions`,
`piece` and `lump` must be whole stretches (`agents/trainer.py:81-89`).

**Where the stretches are cut, and why at a match start**
(`agents/learner.py:449-468`). A row's batch time is cut at every decision a
match starts on, and every 16 decisions after such a start (or after the
batch's first decision). So a stretch never holds two matches, and a match
no longer than 16 decisions that starts inside the batch lies in one
stretch whole. With fixed cuts every 16 decisions, a cue at a match's first
decision and its answer 12 decisions later shared a stretch only when the
match started in the stretch's first 4 places (`agents/learner.py:47-51`),
so most cues could never teach anything. A stretch cut short is replayed
padded to 16 steps; the padding is not learnt from, and its end memory is
read at its true end (`forward_sequence(..., upto)`,
`agents/network.py:486-513`).

**The live check's memory clause.** The memory recomputed at each
stretch's end must equal the stored `memory[t0 + length]`
(`agents/learner.py:518-522`). It is never dropped as redundant: on a fresh
network a memory fault moves the log-probabilities by only 4e-4 to 8e-4
(the heads start near-even), while the memory itself is off by 0.2 to 0.65;
one fault (storing the memory after the decision) is seen only by this
clause (2d note, section 8a).

**What has been found: taught memory works, memory from reward alone does
not (yet).** The test task is `recall12`: at a match's first decision a cue
(one of four directions) is shown for that one decision; at decision 12 the
feet must walk the cued way, and only that decision is paid. A random
player scores 0.20 per answer, the pass line is 0.84 (2d note, section 8a).
The measurements, from the 2d note, sections 8a to 8c:

- 2d.1 learnt nothing; 2d.1b made the two changes to the cell above and
  2d.1c cut the stretches at match starts. By reward `recall12` still stayed
  flat, for 600 rounds.
- The gap ladder: answering at the cue itself (gap 0, no memory needed)
  passes in 93 rounds; gaps of 1, 2, 4 and 8 decisions stay at 0.24 to 0.26
  for 300 rounds. A cliff at a gap of one, not a slow decline with the gap.
- A "hold" form (every decision up to the answer paid for walking the cued
  way) stays flat at 0.25 for 3 000 rounds at gap 12.
- The reward's gradient points the right way on average (cosine +0.48 with
  the right direction on a fresh network, +0.83 at the plateau, over the
  mean of 20 batches) but a single batch is mostly noise (cosine +0.11 to
  +0.16). A round moves the policy 0.001 against the early stop's 0.02; the
  early stop never fires; half the distance goes to heads the task does not
  pay.
- **A memory taught by the right answers, then handed to the reward trainer
  with a fresh value network, is kept and improved**: seed 0 from 0.92 to
  0.97 in 200 rounds; seed 1, taught only to 0.73, held at 0.67 to 0.74.

The reading in the note: no fault and no stopped learner, but the noise
case. Reward keeps and sharpens a memory; from a start that ignores the cue,
the signal per batch is too weak and the steps too small to find one. The
three tests that need `recall12` to pass are skipped by name
(`MEMORY_FINDING` in `agents/tests/test_tasks.py`), not loosened. For the
ladder this means: a memory is expected to come from the copy of the bots,
and reward to keep it. The range of 2c was trained with the memory off.
