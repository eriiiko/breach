# RL agent, rung 2, patch 2d: the trainer's rest (the note)

2026-11-17, the lead's. The design is `docs/rl_agent_rung2_design_2026-11-14.md`
("the design": 1, 4.5 to 4.8, 6 row B, 8, 10) over
`docs/rl_agent_design_2026-11-11.md`; the plan is
`docs/rl_agent_rung2_plan_2026-11-15.md` (section 2's row for 2d, section 5's
list, and what 2b and 2c handed on). Where this note and the design disagree,
the design wins and the note is corrected.

**2d is cut in two, landed one after the other, one implementer each:**

- **2d.1, the memory path:** the stretches, the stored memory, the collector
  and learner with the memory on, the live check's second clause, the two
  memory tasks, the "already good start" task with the value-first rounds.
  No game file is edited.
- **2d.2, saving and resuming:** the saved network, the resume, the batch's
  header and the refusal of a foreign batch, the training manifest, the run
  check extended to `agents/` and `tools/` (one edit of `src/rl_env.py`), the
  run script's `--resume`, the curves script's additions.

The reason for the cut and the order: the memory changes what a stored
decision and a learner's state ARE, and the saved file must hold both as
they finally are. Each half has its own gate and merges on green.

**Not in 2d:** the copy (2e); a driver that plays a saved network in a match
(2f); the seeded rooms (2g); "the best on the validation rooms" among the
kept files (there are no validation rooms before 2g; 2i adds it); any real
run.

All code under `agents/`; tests under `agents/tests`. The game is met only
through `rl_env.BatchedEnv` and the stand-in.

---

# Part 1 (2d.1): the memory path

## 1. What exists

`SoldierNet` already has the memory cell (2a): `forward(obs, state, first)`
for one decision and `forward_sequence(obs, state, first)` for a run of
decisions, both zeroing the memory where `first` says a match starts, held
equal by 2a's tests. The value network has no memory (it sees the whole
match). The trainer refuses a soldier with the memory on
(`trainer.check_run`); 2d.1 removes that refusal.

## 2. The stretch

- One new field of `Settings`: **`stretch`** (default 16, design 4.5), a
  key of the training file's `[train]` table the same day (the rule in
  `.claude/rules/agents.md`).
- With the memory on, `decisions` must be a whole number of stretches and a
  piece a whole number of stretches times rows that fits (`piece %
  stretch == 0`, `lump % stretch == 0`); anything else is refused by name
  when the trainer is built. With the memory off `stretch` is not read and
  nothing about the learner changes: **a run without memory gives the same
  weights, bit for bit on the processor, before and after this patch** (a
  test holds it; 2c's method).
- Stretches are cut on the batch's own time: stretch `k` of row `r` is the
  decisions `16k .. 16k + 15` of that row. A match that starts inside a
  stretch zeroes the memory there (the stored `first` flag), in acting and
  in learning alike.

## 3. What is stored

`agents/batch.py` (the ONE layout) gains one array, **`memory`**, `(T + 1,
R, core)` float32: the soldier's memory BEFORE its decision at each step,
and at index `T` the memory after the last step (what the next round starts
from). Storing every step and not only each stretch's start costs 128 x 64
x `core` x 4 bytes (a few MB) and buys the live check's second clause at
every stretch's end without a second bookkeeping. With the memory off the
array is not allocated.

The collector carries each row's memory across steps AND across rounds (a
match does not end with the round). A row whose soldier is dead or whose
roster place is empty is still stepped (its memory is whatever the network
makes of the served zeros) and is never learned from: the loss reads the
same `valid` rows as today.

## 4. Learning in stretches

- With the memory on, the unit the learner shuffles and cuts into pieces is
  the STRETCH (a row's 16 decisions), never a single decision: a piece of
  2 048 decisions is 128 stretches; a lump of 512 is 32.
- Each stretch is replayed in order by `forward_sequence` from the stored
  memory at its first step, with its `first` flags. The gradient flows
  through the stretch and stops at its start (the stored memory is a
  constant).
- Everything else is as it is: the probability of the stored choices
  through `choosing.evaluate`, the grades from the one grading function
  (computed on the whole batch before any cutting, as today), the clip, the
  early stop, the two networks' own steps. The value network learns from the
  same decisions in the same pieces (it has no memory; it only shares the
  cut).
- Grades are brought to a common size per piece over the valid decisions of
  the piece, as today.

## 5. The live check, second clause (design 4.8)

On the first piece of the first pass, before any step: for every stretch of
the piece, the memory `forward_sequence` returns after the stretch's last
decision equals the stored memory at the next step (`memory[t0 + stretch]`)
within `live_tolerance`, on rows that were valid throughout. A failure is
`LiveCheckFailed` naming the clause, the row (through `LearningRows.name`)
and the stretch. No switch turns it off. The first clause (the probability
of every stored choice) now runs through the sequence path, so it also
catches a stretch replayed from the wrong start.

Broken on purpose in the tests, each must go red: the collector stores the
memory AFTER the decision; the collector does not zero at a match's start;
the learner ignores `first` inside a stretch; the stretches are cut one
decision off.

## 6. The two memory tasks (design 8)

In `agents/tasks.py`, on the stand-in, reduced width, the processor, fixed
seeds, with the pass lines set as 2b set them (from a random player's score
and the task's best; the numbers written into the task's own text):

- **`recall12`:** at a match's first decision a cue (one of four) is shown
  in the soldier's vector for that one decision; for the next 11 decisions
  the vector shows nothing; at decision 12 the feet must walk the cued
  direction, and only that decision is paid. Soldiers die mid-match (a dead
  row stays in the match, paid nothing) and matches end and restart inside a
  round and across a round's edge, at lengths that are NOT multiples of the
  stretch. Passes with stretches of 16.
- **`recall24`:** the same with 24 decisions between cue and answer. It
  must **FAIL with stretches of 16 and pass with 32**. Both halves are
  tests: the fail is asserted (the score stays near the random player's),
  so a trainer that somehow learns it at 16 is a finding about the
  stretches, not a success.
- A third, cheap one for the traps of section 5 that the two above might
  forgive: with the memory on, `recall12` must FAIL when the cue is shown
  at a decision the match does not start on and the collector zeroes the
  memory at every stretch's start instead of at a match's (the classic
  fault: the memory silently reset at the cut).

A task that will not pass is a finding (commit, stop, report), never a
looser line or an easier task (the rule in `agents.md`).

## 7. The "already good start" task, and the value-first rounds

Design 6 row B: a copied soldier starts good with a NEW value network, so
"first only the value network learns while the frozen copy plays". That is
a trainer feature and it lands here with its test:

- One new field of `Settings`: **`value_first_rounds`** (default 0), a key
  of `[train]`. In those rounds the collector plays as always, the value
  network and its scale learn as always, and the soldier's network takes no
  step (its step size is 0; the step-size plan of the later rounds is
  unchanged: it falls over `planned_rounds` as before). The live check runs.
- **The task `good_start`:** take 2b's pointer task; teach a soldier on it
  until it passes (once, cached for the test session); then start a run
  from THAT soldier with a FRESH value network and a fresh scale, and learn
  20 rounds. Measured: the score at the start, the lowest score over the 20
  rounds, the score at the end.
  - With `value_first_rounds = 5` and the design's numbers: the lowest score
    stays at or above 0.9 of the start. This is the gate.
  - Measured beside it and REPORTED, not gated (nobody knows the answer
    yet): the same with `value_first_rounds = 0`, and with the variety push
    ten times the design's. If either wears the start down, the task's text
    says by how much; if neither does, that is written down as it is. The
    lead reads these before the copy (2e to 2h) is trained on.

## 8. Tests of part 1 (properties; each names what it protects)

- The stored memory along a stretch, recomputed, is what the soldier had
  (design 4.5's test): on the stand-in with matches ending inside stretches.
- Without memory: weights after 3 rounds are bit-identical to main's on the
  processor (the patch moved nothing it did not mean to).
- The refusals of section 2, each by name.
- The live check's second clause: silent on a correct run; red for each of
  the four faults of section 5.
- `recall12` passes; `recall24` fails at 16 and passes at 32; the reset-at-
  the-cut trap fails.
- `good_start` holds with the value-first rounds; in those rounds the
  soldier's weights do not change by one bit and the value network's do.
- The training file sets `stretch` and `value_first_rounds`
  (`test_training_file.py` already sets every field through the file).

**Gate of 2d.1:** all of `agents/tests` green on the processor; the game's
suite untouched (no game file changed: `git diff --stat main -- src tools
tests cpp` is empty) so it is NOT rerun; `tests/test_claude_rules_paths.py`
and `tests/test_game_never_imports_agents.py` green.

## 8a. As built (2d.1, 2026-11-17): where it differs, and the finding

Only what differs from sections 1 to 8, and why.

- **FINDING: `recall12` is not learnt with the design's numbers** (section
  6; the measurements are in `agents/tasks.py` `RecallTask`'s text). Seed
  0, stretches of 16: 220 rounds on the tasks' farm (8 matches x 32
  decisions) and 140 on a farm of 32 matches (2 048 decisions a round, one
  design piece) near the random player; with the answer 4 decisions after
  the cue, 0.11 of the way after 300 rounds; with the answer AT the cue (no
  memory needed, memory on) 0.8 at round 180. The live check was silent on
  every round. At the network's start the cue's mark on the memory is
  0.023 against a memory of size 0.36 and falls about 0.6 x per decision
  (1.4e-4 after 12): the cell starts at an even keep/replace and the cue
  enters through 4 vector columns. Following the rule, nothing was bent:
  no budget could be fixed, so the three tests that hang on `recall12`'s
  pass (its pass and `recall24`'s at 32, `recall24` failing at 16, the
  reset-at-the-cut trap failing) are written and SKIPPED BY NAME
  (`MEMORY_FINDING` in `agents/tests/test_tasks.py`); a fail of a task
  nobody passes proves nothing. Not measured, so not claimed: `recall24`
  at 16 or 32. What changes (the cell's start, e.g. an update gate biased
  to keep; the step size; the farm; more rounds) is the lead's call.
- **The memory tasks score per ANSWER**, not per decision (each task
  measures through `measure`, the old ones per decision as before): only
  one decision in 13 to 19 is paid, and per decision the best would depend
  on the match lengths and the deaths. Random 0.20, best 1, the line 0.84.
- **Matches last a drawn `gap + 1` to `gap + 7` decisions** (never a
  multiple of 16), two soldiers, slot 1 dying at a drawn decision in half
  the matches, before or after its answer.
- **The reset-at-the-cut trap** is built as the classic SILENT fault: the
  collector marks every stretch's first decision as a match's start
  (`first`), so acting and learning zero alike and the live check cannot
  see it. "The cue at a decision the match does not start on" is read as:
  the matches start anywhere inside a stretch, so most cues are cut from
  their answers.
- **The live check's second clause compares every stretch of the first
  piece**, not only rows valid throughout: a dead row's memory is as
  deterministic as a living one's (the batch keeps what was acted on), so
  the stricter check costs nothing and was silent on every correct run
  (gaps 0 to 3e-7). On a fresh network the first clause barely sees the
  memory faults (log-probability gaps 4e-4 to 8e-4, the heads start
  near-even); the second sees all four (0.2 to 0.65), and the "stored after
  the decision" fault ONLY through it.
- **The "cut one decision off" fault** is the learner replaying each
  stretch's decisions one step after the memory it starts from (a
  consistent shift of both would be a correct replay).
- **The units are the stretches holding a value row**; a stretch with none
  is never replayed. `HeadOutputs.take` (network) picks the policy rows out
  of a replayed stretch, the gradient flowing through.
- **The value-first rounds take no step at all** (not a step of size 0:
  Adam's moments would move); the soldier is replayed on the first piece
  only, for the live check.
- **Bit-identity** is held by a sha256 of both networks' weights after 3
  rounds, made with main's code before this patch touched it, keyed by
  PyTorch's version and the processor (`NO_MEMORY_REFERENCE` in
  `agents/tests/test_memory.py`); on another machine it skips naming how to
  add its key (the processor's float kernels decide the bits).
- **`good_start`** (measured table in its own text): planned rounds 150
  (the pointer's budget, so the step size is near full over the 20
  rounds); the start is the first round's score. The gate held on three
  seeds (lowest 1.00, 0.92, 1.00 of the start); neither no value-first
  rounds nor ten times the variety wore the start down. The one dip came
  while the soldier was still frozen: the 0.9 line is about three rounds'
  noise below a single-round start.

---

# Part 2 (2d.2): saving and resuming

## 9. The saved network (design 4.7)

One file, written by `torch.save` through ONE module, `agents/saved.py`
(`save`, `load`, `export_soldier`), atomically (a temporary name in the same
folder, then a rename). Its fields:

| field | what |
|---|---|
| `format` | the saved file's own version (1) |
| `soldier`, `value` | both networks' weights |
| `soldier_record`, `value_record` | what each network fits (`SpecRecord`), compared by `check_fits` on load |
| `learner` | both optimisers' state; the value scale (mean, mean square, spread, `rounds`: 2b's hand-on, the scale is part of its network's state) |
| `collector` | the generator's state; nothing of a match (matches start again on a resume) |
| `counters` | rounds done, learning decisions, batches, the weights' version |
| `settings` | the `Settings` record as a table, and the network spec's name |
| `specs` | the observation, action and reward-profile names |
| `hashes` | sha256 of the run file and of the training file; the game's build id (`bp.build_id()`); the commit of the checkout (`HEAD`) |
| `manifest` | the training manifest's hashes (section 11) |
| `started_from` | `"nothing"`, or `{"copy": <records' hash>}` (2e fills it), or `{"saved": <file's sha256>, "round": n}` |

`export_soldier` writes the soldier's weights, its record and the three spec
names alone (what an opponent or a player needs; about 6 MB).

**What a run keeps,** in its own folder outside the checkout: `last.pt`
(every `save_every` rounds and at the end) and `round_<n>.pt` every
`keep_every` rounds. Two new fields of `Settings` with keys under a `[save]`
table: `save_every` (default 10), `keep_every` (default 0 = none). "The best
on the validation rooms" is 2i's.

`agents/run.py` writes these in place of 2c's `*_NOT_A_SAVED_RUN.pt`.

## 10. The resume

`python -m agents.run --resume <the run's folder>` (the run and training
files are the folder's own copies; `--run` / `--train` are then refused).

- **What it compares, each refused by name on a difference:** the saved
  file's format; both records (`check_fits`); the three spec names against
  the environment's layout; the run file's and the training file's sha256
  against the folder's copies; the training manifest (section 11); the
  build id and the commit against the checkout's. The build id and commit
  may differ only with `--allow-code-change`, which is written into the
  curve's next line and into `started_from` of every later save (a bug fix
  mid-run is real life; a silent one is not).
- **What it restores:** both networks, both optimisers, the scale, the
  generator, the counters. The step size continues where the plan was
  (`step_size_at(rounds done)`). `curve.jsonl` is appended to; the first
  line after a resume carries `"resumed_from": <round>`.
- **What it does not restore:** the matches in flight. Every match starts
  again (the environment's first reset), and with the memory on every row's
  memory is zero. So a resumed run is not the uninterrupted run bit for bit,
  and on the card it would not be anyway (design 4.7).
- **The tests** (the stand-in, the processor):
  - save at round k, load into a fresh trainer: every weight, both
    optimisers' state, the scale, the generator's state and the counters
    are equal;
  - two fresh trainers loaded from the same file, each with a fresh
    stand-in of the same seed, produce bit-identical weights after two more
    rounds (the resume is deterministic);
  - the step size of the first resumed round is the plan's for that round;
  - each refusal, by name; `--allow-code-change` lets exactly the two code
    fields differ and is recorded;
  - a save interrupted before the rename leaves the old `last.pt` whole.

## 11. The training manifest (design 1)

`training_manifest.json`, written beside the curve at a run's start by
`agents/saved.py` (one writer), fields:

| field | what |
|---|---|
| `format` | 1 |
| `training_file_sha256` | of the folder's copy |
| `run_file_sha256` | of the folder's copy (the environment's manifest has it too; here so one file tells the run) |
| `levels_sha256` | one hash over the level folders the run file names: every file's relative path and bytes, in sorted order |
| `room_generator_commit` | the last commit that touched `tools/gen_training_rooms.py` (`null` when the run names no generated room) |
| `seed_lists_sha256` | `null` until 2g gives the lists a file |
| `commit`, `dirty` | the checkout's `HEAD`, and whether `git status` shows a tracked change or an untracked file under the run check's paths |

A resume refuses a change in any field but `commit` / `dirty` (those follow
section 10's rule). A run with `dirty` true prints one line saying its
network is not to be kept (design 4.7: an exploratory run may run at any
commit; a kept one runs from a tag, and there the environment's
`require_snapshot` refuses a dirty tree before the trainer is asked).

## 12. The batch's header and the refusal (design 4.6)

- `Batch` gains a `header`: the weights' version it was played with, the
  run file's and the training file's sha256, the build id. The collector
  writes it when a round's collecting starts; the learner's `learn` compares
  it with its own (its current weights' version, the same two hashes, the
  same build id) and refuses a batch that differs, naming the field
  (`ForeignBatch`).
- The weights' version is a counter the learner raises by one at every
  hand-over of weights, saved with the network.
- In rung 2 both sit in one process, so the refusal can only fire on a
  fault; the tests make it fire: a batch collected, the learner stepped
  once, the same batch handed again (version behind by one) is refused; a
  batch with another training file's hash is refused.

## 13. The run check extended (the one game file)

`src/rl_env.py`: `SNAPSHOT_PATHS` gains `"agents/"` and `"tools/"`, so that
`require_snapshot` refuses a training run whose checkout has an untracked
file there (a stray module would be imported, a stray tool would write the
rooms). Read the rule file `.claude/rules/rl-agent.md` (Training snapshot)
first and amend its line in the same patch.

- Its tests, in `tests/` beside the existing snapshot tests: an untracked
  file under `agents/` refuses by the check's name, one under `tools/`
  refuses, one under `docs/` does not; ignored files (`__pycache__`) do not.
- This is host Python outside the tick: `tools/twin_debt.py` and
  `tests/test_class_p_level.py` say whether anything is owed (expected:
  nothing).
- The main checkout carries untracked files under `tools/` on purpose (a
  downloaded upscaler). That is right: a kept training run starts from its
  own detached checkout at a tag, never from the main checkout.

## 14. The curves script

`agents/plot_curve.py` (THE curves script) gains: a vertical mark at every
resume; the memory's live-check number when the line has one; and a refusal
that names the line when `curve.jsonl` holds a `stopped` line (it draws up
to it and says so in the figure). No second script.

## 15. Tests of part 2, and its gate

The tests of sections 10, 12 and 13, plus: `export_soldier`'s file loads
into a `SoldierNet` built for the same layout and is refused by `check_fits`
for another; the manifest's `levels_sha256` changes when one byte of one
level file changes and not when the folder is copied elsewhere; a smoke of
`agents.run` on the real hand rooms (2 rounds, 4 matches, 2 workers, reduced
width, the processor), stopped, resumed for 2 more: the curve has 4 lines,
the third says `resumed_from`.

**Gate of 2d.2:** all of `agents/tests` green; because `src/rl_env.py`
changed, the game's suite once, uninterrupted, at the final tree
(`pytest tests -q`), with both modules built in the worktree;
`tools/twin_debt.py` none; `tests/test_claude_rules_paths.py` green.

## 15a. As built (2d.2, 2026-11-17): where it differs, and why

Only what differs from sections 9 to 15.

- **The weights' version is raised by `learn`**, not by the hand-over (it
  names the weights `learn` just moved; the trainer hands the new weights
  over with it). In the trainer's loop the two are the same; this form
  also refuses a batch learnt twice without a hand-over between.
- **The header's fields** are `version`, `run_sha256`, `training_sha256`,
  `build_id` (`batch.HEADER_FIELDS`); `Batch.set_header` writes it,
  `Batch.check_header` is the learner's reader; a batch with no header is
  refused too. The stand-in has no build id: `None` on both sides.
- **The saved file and the export also carry `net`** (the network spec's
  widths): the record names `"v1"` for both the full and the reduced
  width, so without it a reduced file loads into a full network only to
  fail on a tensor's shape. A difference is refused as `network widths`.
- **A resume compares the training manifest held IN the saved file** with
  one made now (the folder's `training_manifest.json` is written once, at
  the run's start, for people). `dirty` is `rl_env.check_clean_tree`'s
  verdict, the one check, now covering `agents/` and `tools/`.
- **The curve is trimmed at a resume.** A run killed after a round's line
  and before the next save leaves lines past the saved round; they (and a
  `stopped` line) move to `curve_discarded.jsonl`, so the curve goes on
  without a seam. The first resumed line carries `resumed_from`,
  `matches_restarted` (true: every match starts again and, with the memory
  on, every memory is zero), `code_change` and `curve_lines_discarded`.
  A stop (masked choice, live check) does not save: `last.pt` stays at the
  last save.
- **`started_from` of a resumed run** is `{"saved": <sha256 of last.pt>,
  "round": k, "from": <that file's own started_from>}` (the chain back to
  "nothing" or a copy), with `code_change` when `--allow-code-change` let
  the build id or commit differ. `--out` is refused with `--resume` as
  `--run` / `--train` are. The wall stop counts from the resume.
- **What a save must hold is a table** (`saved.SAVED` / `NOT_SAVED`, with
  why each is not saved); `test_saved.py` fails on an attribute of the
  trainer, learner, collector or scale in neither.
- **FINDING, fixed: the optimisers keep the tensors they are handed.**
  `Adam.load_state_dict` on the processor does not copy the moments, and
  steps them in place: two trainers restored from one loaded file shared
  them and parted after one round (the determinism test went red). The
  restore now copies the learner's tables first.
- **What is exact at a resume** (on the processor, held by the tests):
  every weight, both optimisers' state, the scale, both generators, the
  counters and the version are restored to the bit, and two restores of one
  file over fresh stand-ins end bit-identical after two more rounds. What
  is not: the matches (they start again), so a resumed run is not the
  unbroken one; no test claims it.
- **The curves script** draws a fifth figure, `curve_live.png` (the live
  check's two gaps against the tolerance); `read` stops at the first
  `stopped` line and returns it, the figures say so in their title, and
  the script prints that it refused the rest.
- **The real-rooms smoke** is the existing `test_a_smoke_on_the_real_rooms`
  extended: 2 rounds, then `--resume` for 2 more, 4 lines, the third
  `resumed_from` 2.
- **The levels hash** is length-prefixed (each file's relative path, then
  its bytes) so two folders cannot hash the same by moving a byte across a
  file boundary.

## 16. Systems

- **Uses:** the Batched environment and its Training snapshot row
  (`require_snapshot`, `write_run_manifest`: the environment's manifest is
  not extended, design 1); the Network spec row (`SpecRecord`,
  `check_fits`); Choosing; the Stored decision and batch row (ONE layout:
  `memory` and the header join it); the Trainer row (every number a field
  of `Settings`; the training file; the live check on every round; the
  made-up tasks each with a trap that fails).
- **Creates:**
  - *Saved network and training manifest* (`agents/saved.py`): THE one
    writer and reader of a saved network, a soldier's export and
    `training_manifest.json`; every saved network says what it was started
    from; a driver of a saved network (2f) loads through it, never its own
    `torch.load`. One index line in `CLAUDE.md`; its constraints in
    `.claude/rules/agents.md`.
  - The memory path adds no system: it is the Stored decision row's array
    and the Trainer row's learner; its constraints (stretches are the unit,
    the memory is zeroed at a match's start and nowhere else, the second
    clause of the live check) are lines under those rows in `agents.md`.
