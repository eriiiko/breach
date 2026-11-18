---
paths:
  - "agents/**"
---

# Agents: the trainer's side (networks, choosing, the trainer)

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say. The
`game-ai-lead` skill is the role's continuity; the design is
`docs/rl_agent_rung2_design_2026-11-14.md` (sections 2 and 4) over
`docs/rl_agent_design_2026-11-11.md` (section 8).

## Network spec
- The game never imports `agents` (`tests/test_game_never_imports_agents.py`) and `agents/tests` stays out of the game's suite (`testpaths`), so the game's suite never needs PyTorch. `agents` reads the game's Python, never edits it.
- Every size is asked from the game's spec tables through `agents/sizes.py`, never typed into a network. A kind column with no source there, or a code plane with no codes, is REFUSED: a new observation spec is a new network spec, never an edit of a spec a trained network uses.
- The networks take the arrays AS SERVED (the dtypes of `obs_spec_v1.shapes` / `full_shapes`) and refuse any other format, so acting and learning convert in one place. Layer normalisation only, no dropout, nothing read across rows of a batch; no list position is encoded (the pointer follows the entry, the list may be any length).
- With the memory on, the heads read the memory only through `memory_norm`, never the raw cell output (2d.1b: the raw output starts ~8 x smaller than the join and the cue was not learnt); what is carried, stored and checked is the cell's own state. A memory-on change draws no random number (the memory-off network and every other weight start as before).
- The soldier's network never reads the full view (refused by key) and shares no weight with the value network, whose outputs are `reward_v1.REWARD_GROUPS` in order.
- What a network fits is its `SpecRecord`; loading is refused by `check_fits` on any difference. A change to any field's source (a spec name, the grenade rows, a plane, a kind's table) changes the record in the same patch.
- The kind columns whose codes have no named table in the game (`aims_at`, `zombie_state`, `whose`, `place_kind`, and `side` with the zombies' team) read `CODE_ROWS` rows, a capacity: a new code past it fails loudly; it moves to a named table with a new network spec, never a silent clamp.

## Choosing
- `agents/choosing.py` is the ONE place the network's raw outputs and the masks become chances, a draw and a probability. The trainer, the copy and any driver of a saved network call `draw` / `evaluate`; never a second softmax under masks, sampler or log-probability beside it.
- Which heads counted is the GAME's rule (`action_spec_v1.heads_used` with `arg_mask`). The batched forms here exist only for a whole batch on the card and are held EQUAL to the game's by `agents/tests/test_choosing.py`; a change of the game's rule changes them in the same patch.
- A head that did not count adds nothing to the probability, the variety or the gradient; a given choice the masks forbid is refused, never graded.
- The random-legal chance (`explore`) lives only here, default 0; a decision is evaluated with the value it was drawn with.

## Stored decision and batch
- Only the rows of the teams on `policy` (`trainer.learning_teams`) are stored, read through `env.team_rows`; a row of any other team is never stored, drawn for or graded, even while it is valid and paid (critique lens 2, major 5: the opponent's rows would be graded choices that had no effect).
- Everything kept is COPIED out of the environment's arrays at its step (they are overwritten at the next), and the soldier acts on the batch's own slot, so what was acted on is what is learnt from. Never a reference to an environment buffer.
- ONE layout (`agents/batch.py`): a new kept array (2d's memory vector and header among them) is added there, with its reader in the learner, in the same patch; never a second buffer beside it. Shapes and formats come from the environment's layout, never typed.
- The batch's `header` (the weights' version it was played with, the run file's and the training file's sha256, the build id) is written by the collector when a round's collecting starts and read by the learner's `learn` through `Batch.check_header`, which refuses any difference by name (`ForeignBatch`). `learn` moves the weights, so it raises the version; the hand-over carries it. Never a learner that learns a batch without reading it.
- Which steps are value rows is the collector's bookkeeping, carried across rounds and stored as `value_row`; grading reads it and never re-derives it from the flags.
- The stored `memory` is the one BEFORE the decision (the state carried in, before the network zeroes it at a `first`), with `memory[T]` the one after the last step; the soldier acts from that stored copy. The memory is zeroed at a match's start (the `first` flag, inside the network) and nowhere else: never at a round's or a stretch's edge (the trap that makes `recall12` fail).

## Trainer
- It meets the game only through `BatchedEnv`, or in the tests through `agents/standin.py`, which serves the same names: a rename in `rl_env.BatchedEnv` changes the stand-in in the same patch (`agents/tests/test_standin.py` holds every name to the game's source).
- Grading is ONE pure function (`agents/grading.py`) and the value targets' scale ONE class (`agents/value_scale.py`); the learner computes no probability of its own (that is `choosing`'s) and no second grading. A round's order is fixed: grade with the scale as it was while collecting, move the scale, learn on the moved scale.
- Every trainer number is a field of `agents/settings.py`, its default the design's starting value; never a number typed elsewhere. A different number is a sweep (the training file, patch 2c), never an edit to turn a test or a task green.
- The live check runs on every round of every run, real or test: no switch turns it off and its tolerance is the design's. A failure is a finding about acting against learning, never a reason to loosen it.
- With the memory on, the STRETCH is the unit learning shuffles, cuts and replays (`forward_sequence` from the stored memory at its start, its `first` flags; the gradient stops at its start), never a single decision; `decisions`, `piece` and `lump` are whole stretches (refused by name). The stretches are cut ONLY by `Learner._stretches` (at every match start and every `stretch` decisions after one; never longer than `stretch`, never two matches); a short stretch is replayed padded after its end and its end memory read through `forward_sequence(..., upto)`, never a second cut or a replay of its own (so a cue at a match's start and its answer can share a stretch; the lead's ruling, 2026-11-18). The live check's second clause (the memory recomputed at every stretch's end equals the stored one) is part of the check, and the first clause runs through the same sequence path. On a fresh network the first clause barely sees a memory fault (2d.1: gaps of 4e-4 to 8e-4 against 0.2 to 0.65 for the memory), so the second clause is never dropped as redundant. With the memory off the learner's path is main's, held bit for bit (`agents/tests/test_memory.py`).
- The value-first rounds (`value_first_rounds`, design 6 arm B) leave the soldier's weights and its optimiser's state bit-identical (it takes no step, not a step of size 0); the live check and the value network run as always, and the step-size plan is not shifted by them.
- Rows the trainer does not drive are sent the legal default; `info["masked_choices"]` is read on the driven living rows and anything but 0 stops the run (a refusal, never a curve).
- The made-up tasks (`agents/tasks.py`) each carry a trap that must make them FAIL; a task that will not pass is a finding (commit, stop, report), never a looser line, an easier task or tuned numbers. What a finding changes is decided by the lead at the patch's boundary and written, with its measurement, into the task's own text (2b: a fault in the value scale; a free verb in the argument task's reward; eight directions on a 256-decision farm).
- The value scale's first move leaves a fresh network alone and every later move keeps its outputs; a scale is part of its value network's state, never built fresh for a taught one.
- THE TRAINING FILE (`agents/training_file.py`) is the only way a run's trainer numbers reach `Settings` from disk: one key per `Settings` field (`[train]`, `[weights]` naming every group, `[stop] wall_hours`) plus `[network]`; an unknown key, a missing `planned_rounds` or a wrong type is refused by name, never ignored. A new `Settings` field is a key the same day (`agents/tests/test_training_file.py` sets every field through the file).
- `planned_rounds` IS the plan the step size falls over: set it from a measured speed so the plan is the run; the wall stop only cuts a plan short, it never stretches the step size.
- `agents/run.py` joins the trainer, it never re-implements a round; every number of its per-round line is read from the batch, the collector's round log (`Collector.log` / `seconds`, from the env's `info`) or the learner's measurements, never a second bookkeeping. A number the line cannot read honestly is left out and named (2c: rounds fired, the geometric "aim on the target"). `agents/plot_curve.py` is THE curves script: 2d extends it.

## Saved network and training manifest
- `agents/saved.py` is the ONE way a run's state reaches disk and comes back: the saved file (`save` / `load` / `check_resume` / `restore`), a soldier's export (`export_soldier` / `load_soldier`) and `training_manifest.json`. Never a `torch.save` / `torch.load` of a network or a trainer state elsewhere; a driver of a saved soldier (2f) loads through `load_soldier`. Only `agents/run.py` writes a run's `last.pt` and `round_<n>.pt`, always through `save` (atomic: a temporary name, flushed, renamed).
- A new attribute of the trainer, learner, collector or value scale joins `SAVED` (and `state` / `restore`) or `NOT_SAVED` with why, in the patch that adds it; `agents/tests/test_saved.py` fails on an unclassified one.
- A resume refuses by name a changed format, record, widths, spec, run or training file, or manifest field; the build id and commit only under `--allow-code-change`, which is recorded in the curve and in every later save's `started_from`. A refusal is never loosened to resume a run: a run that cannot resume starts again.
- What a resume does NOT restore is the matches in flight (they start again, the memory with them at zero): resumed and unbroken runs are equal in what the learner owns, never claimed equal in what was played. The first resumed line says so (`resumed_from`, `matches_restarted`).
- `restore` gives the optimisers their own copies of the loaded state: they keep the tensors they are handed and step them in place (2d.2: two trainers restored from one loaded file shared their Adam moments and parted after one round).
