---
paths:
  - "agents/**"
---

# Agents: the trainer's side (networks, choosing)

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say. The
`game-ai-lead` skill is the role's continuity; the design is
`docs/rl_agent_rung2_design_2026-11-14.md` (section 2) over
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
