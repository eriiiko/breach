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
- Which steps are value rows is the collector's bookkeeping, carried across rounds and stored as `value_row`; grading reads it and never re-derives it from the flags.

## Trainer
- It meets the game only through `BatchedEnv`, or in the tests through `agents/standin.py`, which serves the same names: a rename in `rl_env.BatchedEnv` changes the stand-in in the same patch (`agents/tests/test_standin.py` holds every name to the game's source).
- Grading is ONE pure function (`agents/grading.py`) and the value targets' scale ONE class (`agents/value_scale.py`); the learner computes no probability of its own (that is `choosing`'s) and no second grading. A round's order is fixed: grade with the scale as it was while collecting, move the scale, learn on the moved scale.
- Every trainer number is a field of `agents/settings.py`, its default the design's starting value; never a number typed elsewhere. A different number is a sweep (the training file, patch 2c), never an edit to turn a test or a task green.
- The live check runs on every round of every run, real or test: no switch turns it off and its tolerance is the design's. A failure is a finding about acting against learning, never a reason to loosen it.
- Rows the trainer does not drive are sent the legal default; `info["masked_choices"]` is read on the driven living rows and anything but 0 stops the run (a refusal, never a curve).
- The made-up tasks (`agents/tasks.py`) each carry a trap that must make them FAIL; a task that will not pass is a finding (commit, stop, report), never a looser line, an easier task or tuned numbers. What a finding changes is decided by the lead at the patch's boundary and written, with its measurement, into the task's own text (2b: a fault in the value scale; a free verb in the argument task's reward; eight directions on a 256-decision farm).
- The value scale's first move leaves a fresh network alone and every later move keeps its outputs; a scale is part of its value network's state, never built fresh for a taught one.
