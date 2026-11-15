# RL agent, rung 2, patch 2b: the trainer's core (the note)

2026-11-15, the lead's. The design is `docs/rl_agent_rung2_design_2026-11-14.md`
("the design": sections 4.1 to 4.5, 4.8, 8, 14) over
`docs/rl_agent_design_2026-11-11.md` (6.5, 7, 8.6); the plan is
`docs/rl_agent_rung2_plan_2026-11-15.md`. This note settles what the plan's
section 5 asks of it. Where it and the design disagree, the design wins and
the note is corrected.

**In 2b:** which rows are stored; the stored decision and the batch; the
collector; grading; the rescaling of the value targets; the update; the live
check; a stand-in environment; the five made-up tasks that need no memory;
the grading case worked by hand; one small real run as a smoke.

**Not in 2b** (patch 2d): the memory path and its two tasks, the "already
good start" task, saving and resuming, the hand-over header and the refusal
of a foreign batch, the training manifest, the curves script. The training
file on disk is 2c's. So in 2b: **a network with the memory on is refused by
name**, and the settings are a plain record whose defaults are the design's
table (4.4).

All code under `agents/`; its tests under `agents/tests`. No game file is
edited. The game is met only through `rl_env.BatchedEnv` (and, in the tests,
through the stand-in).

## 1. The parts

1. **Settings:** one record holding the numbers of design 4.4 and the reward
   group weights (rung 2: outcome 1, own 1, the other three 0), nothing
   typed anywhere else.
2. **The batch:** one pre-allocated set of arrays, reused every round
   (section 2).
3. **The collector:** takes the weights, plays T decisions in each of the N
   matches, returns the batch (section 3).
4. **Grading:** a PURE function of arrays (section 4), so the hand-worked
   case tests it without a network.
5. **The value targets' scale** (PopArt, section 5).
6. **The learner:** takes the batch, returns the weights (section 6), with
   the live check (section 7).
7. **The round loop** that joins them, in one process.
8. **The stand-in environment and the made-up tasks** (sections 8, 9).

Collector and learner stay two parts with one hand-over each way (a batch
one way, weights the other), so that 2d only adds the header.

## 2. The stored decision

Rows: the roster places of every team the run file puts on `policy` ("the
learning teams"), R = N x S x (learning teams), each at a fixed place. Time
first: every array is `(T, R, ...)`. That order is what 2d's stretches cut.

| name | format | what |
|---|---|---|
| `vector`, `list`, `list_mask`, `near_f`, `near_u`, `far_f`, `far_u` | as served | the observation the decision answered (`list_uid` is not kept: it is no input) |
| `mask_feet`, `mask_aim`, `mask_trigger`, `mask_hand` | as served (bytes) | the masks as they were |
| `arg_mask` | bytes, 5 | the argument's mask the draw worked out |
| `full_list`, `full_mask`, `full_match`, `full_watched` | as served | the value network's view at the decision |
| `actions` | int16, 5 | the five choices |
| `used` | bytes, 5 | which heads counted |
| `logp_heads`, `logp` | float32, 5 and 1 | the probability of each counted choice and of the decision |
| `reward` | float32, 5 | the five group numbers paid by the step that carried the decision out |
| `valid`, `in_match` | bytes | the two row flags served with the observation |
| `done` | bytes | the row's story ended in that step (its match ended, or `rows_ended`) |
| `first` | bytes | the decision is the first of a match for this row (2d zeroes the memory there) |

Once per batch, for the value after the last stored decision: the
observation keys the value network reads and the full view as they stand
after the last step (`last_*`, `(R, ...)`), copied.

- Everything is COPIED from the environment's arrays at the step (they are
  overwritten by the next one).
- The layout is rectangular; a row takes part by its flags. The **policy
  rows** of a time step are those with `valid` 1. The **value rows** are
  those with `in_match` 1 whose story has not ended since their match began
  (a dead soldier is a value row; an empty roster place and a row after its
  `rows_ended` are in nothing).
- Sizes come from 2a's `Sizes` and the environment's layout; no literal.

## 3. The collector

Per step, for all learning rows in one pass: the served arrays through
`BatchedEnv.team_rows` (the full view from `info`, the same rows) to the
soldier's network without gradients, then `choosing.draw`; copy into the
batch; send; then store the step's rewards and endings.

- **Every row the trainer does not drive is sent the legal default**
  `(stand, aim where I walk, no trigger, hand nothing, 0)`. A learning row
  that is dead or empty carries one legal option per head, so its draw IS
  the default.
- `info["masked_choices"]` on the driven living rows must be 0 after every
  step; anything else stops the run, naming the env and the row.
- Between rounds nothing is reset: the next round goes on from the matches
  as they stand.
- **Refused by name at the start:** a run whose reward is not a named
  profile (the old "counters": the clock is not an ending there); an
  observation or action spec other than "v1"; the full view switched off; a
  network whose record does not fit the layout (`check_fits`); a network
  with the memory on.

## 4. Grading, exactly

Per reward group g and row, over the batch's T steps, with v the value
network's estimate in real units (computed once per round, without
gradients, with the scale as it was while collecting):

- `next[t]` = `v[t+1]`, and for the last step the estimate of the `last_*`
  view;
- `delta[t] = reward[t] + gamma * (1 - done[t]) * next[t] - v[t]`;
- `adv[t] = delta[t] + gamma * lambda * (1 - done[t]) * adv[t+1]`
  (`adv[T] = 0`);
- `target[t] = adv[t] + v[t]`.

So nothing is still to come after an ending, whatever the view returned (it
is the next match's); the one cut that leans on the value network is the
batch's end in the middle of a match. Only value rows count; a step that is
not a value row neither receives nor passes anything on.

The choice's grade is `sum over g of weight[g] * adv[g]`. It is brought to a
common size (minus the mean, over the spread plus a small floor) AFTER the
sum, over the policy rows of the piece being learnt from.

**The case worked by hand** (a test; the numbers written out in the test,
with gamma and lambda of one half so a person can follow): a match that
ends in the middle of the batch and the next one starting; an ending on the
batch's last step; a batch end in mid-match; a dead soldier of a two-soldier
team that goes on receiving team and outcome rewards; a row whose story ends
while its match goes on; an empty roster place; five groups, two of them at
weight 0.

## 5. The value targets' scale

Per output a mean and a spread. The network's last layer gives the
normalised estimate; the real one is `spread * normalised + mean`. Each
round, in this order: (1) grade with the scale as it was while collecting;
(2) move the mean and the mean square a share (0.05 a round, corrected for
the first rounds' bias) toward this batch's targets, the spread never under
a floor of 0.01; (3) change the last layer so that every real-unit output
is what it was before the move; (4) learn, on the targets normalised with
the new scale. All five outputs learn, the weight-0 ones too. The value loss
is the mean squared error of the normalised estimates over the value rows,
summed over the outputs.

## 6. The update

Design 4.4's numbers. Said exactly:

- **Passes** 4 over the round's batch; each pass cuts a fresh shuffle of the
  decisions into **pieces** of 2 048; a piece is worked through in **lumps**
  of 512 whose gradients are added so that the step is the step of the
  piece's mean loss (the piece's grade statistics are taken before its first
  lump: they need no network).
- **The soldier's loss** on the policy rows: the clipped objective on the
  JOINT ratio (the stored and the recomputed probability of the whole
  decision, `choosing.evaluate` under the stored masks), clip 0.2; minus
  0.01 times the sum of the counted heads' variety.
- **The early stop:** the distance moved is measured for every piece from
  the forward pass its step would use (the mean of stored minus recomputed
  log-probability over the piece's policy rows). The first piece over 0.02
  is not stepped and the soldier's network takes no further step this
  round. The value network finishes its passes.
- **Two optimisers** (Adam, 0.0003 falling in a straight line to zero over
  the run's planned rounds, epsilon 1e-5), each network's gradient cut to a
  norm of 0.5 by itself.
- **One arithmetic for acting and learning:** 32-bit, the networks in the
  same mode in both, the card's faster-but-different options (TF32, the
  convolution algorithm search) off.
- **A near-even first policy** (handed over by 2a): the last layer of every
  head (the feet, the trigger, the argument, the aim's and the hand's
  question) starts a hundred times smaller than PyTorch's default, so every
  legal option starts near the same chance. This amends `agents/network.py`;
  a test holds that a fresh network's variety is above 0.99 of its most on
  every head.

## 7. The live check

Every round, on the first piece of the first pass, before any step: on the
policy rows the recomputed probability of every stored choice equals the
stored one to 0.0001, "which heads counted" and the argument's mask
recomputed equal the stored ones exactly, and the distance moved is 0 to
0.0001. A failure stops the run with an error that says which of the three.
(The memory half is 2d's.)

**Broken on purpose in the tests, each must trip it:** the collector keeping
a reference to an environment array in place of a copy; the masks at
learning taken from another step.

## 8. The stand-in environment

It lives in `agents/` (the tasks of 2d use it too) and behaves as
`BatchedEnv` does at main's head. Before writing it, read `src/rl_env.py`'s
module text and `BatchedEnv.reset` / `step` / `team_rows`, list every
attribute and key the trainer reads, and give the stand-in the same names.
Its behaviours:

1. the observation arrays, the full view and the `final` arrays are the
   SAME arrays on every call, overwritten in place;
2. a match that ends inside a step is restarted inside the same call: the
   returned view is the new match's first, the ended match's last view is
   under `info["final_obs"]` / `info["final_full_view"]` (zeros on every
   other row, overwritten the step after), `info["ended"]` says so, and the
   step's rewards are the ended match's;
3. the reward is `(N, U, 5)` in the game's group order, zeros for an empty
   place and for a row whose records ended; `info["rows_ended"]`;
4. every team's rows are observed, masked and paid, the non-learning ones
   too; `team_rows` gives views of one team's rows;
5. a row with `valid` 0 is all zeros with the default masks;
6. actions are checked as the game checks them (shape, range), and a masked
   choice is counted in `info["masked_choices"]`;
7. the arrays have the game's names, number formats and shapes (a test
   compares them with the game's own shape tables) and values in the game's
   real ranges (kinds inside their tables, counts small).

## 9. The made-up tasks (no memory)

The first five rows of the design's table (8): the flagged list entry under
a shuffle; the argument's legal values depending on the hand and the aim,
with a head unused on most decisions; the reward in "own" with the opposite
in a weight-0 group; rewards a hundred times larger beside a group that is
always 0; a second team that does not learn and is paid the opposite.

- The real network code at reduced width, on the processor, **three fixed
  seeds** each.
- **Pass:** on every seed, the mean reward per decision over the last five
  rounds reaches 0.8 of the way from a random player's to the best
  possible, inside a number of rounds fixed in the test.
- **Each task is shown to FAIL with its own trap put in** (the table's
  "fails if" column), once, before the patch is called green.
- The tasks may take minutes; they carry a marker so the quick tests can be
  run alone. The gate runs everything.
- A task that will not pass is a FINDING: stop and report. The threshold,
  the task and the trainer's numbers are not bent to turn it green.

## 10. Gates

1. `agents/tests` green on the processor (the quick tests in seconds, the
   tasks in minutes), 2a's tests included and unchanged in what they assert.
2. The hand-worked case; the live check tripped by both faults; each task
   failed once by its trap.
3. **One small real run, as a script, not a test** (it needs the compiled
   module, so build the processor module once): two rounds on `ship_demo`
   with team 0 learning and team 1 on the bots, one worker, a few matches,
   short rounds, reduced width, on the processor. It must collect, grade and
   learn with the live check silent and no masked choice counted. Report
   what it printed.
4. **One measurement on the card, if the card is free** (under 2 GB in use
   by others): the full-width networks on one made-up task for a few rounds;
   report learning decisions a second for collecting and for learning
   apart, and the card's peak memory. Not a pass or a fail.
5. The game's suite is not run in full (no game file changes); the four
   tests that guard the boundary are:
   `tests/test_claude_rules_paths.py`, `tests/test_path_rules_hook.py`,
   `tests/test_game_never_imports_agents.py`, `tests/test_class_p_level.py`.
6. The lead reads the grading and the update line by line before the merge.

## 11. Systems

Uses: Batched environment (`team_rows`, the masks, the full view from
`info`, the five reward groups, `rows_ended`, the two flags); Observation
spec, Full view, Action spec, Reward profile as served; Network spec and
Choosing (2a). New, with the design's draft rules (14): **Stored decision
and batch**; **Trainer**. One index line each and their constraints in
`.claude/rules/agents.md`.
