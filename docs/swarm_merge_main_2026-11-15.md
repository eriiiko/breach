# Arc #63: merging origin/main into 63-swarm-units (2026-11-15)

A mechanical sync: `origin/main` at `a20711f2` (115 commits since the last
merge's base `12a38642`: the residency arc's P6 through P6.4b at
`7eae5ecc`, #206's rules homes, #29's design 4.4c documents) merged
`--no-ff` into `63-swarm-units` at `39e2ad4d`. Nothing was redesigned. The
residency arc's handover (issue #63, comment of 2026-11-15) sections 1, 2
and 6 were the instructions.

- **Merge commit:** `5f0f189b` (parents `39e2ad4d`, `a20711f2`)
- **Follow-up commit:** this note, with the twin ledger's `level_at` moves
  (§3).

## 1. Conflicts and how each was resolved

Only four files were touched by both sides since `12a38642`: `CLAUDE.md`,
`config.toml`, `src/simulation/simulation.py` and
`tests/twin_manifest.toml`. Two conflicted.

| File | Kind | Resolution |
|---|---|---|
| `CLAUDE.md` | content | Main's #206 file taken whole; the branch's six swarm rows converted (§2). |
| `tests/twin_manifest.toml` | content | Main's lines plus the branch's `swarm.py` `[not_twinned]` line. Main had dropped the `status.py` `[not_twinned]` line (P6 put `status.py` into twins' traced files), and it stays dropped. The bracket and quoting were checked: `tools/twin_manifest_gen.py` parses the file and reports it current. |
| `config.toml` | auto | The branch's `[recorder] swarm_units_cap` and `[swarm.larva]` beside main's new rows; read back. |
| `src/simulation/simulation.py` | auto | The swarm import, `on_config_reload`, the facade in `_reset_internal`, the recorder's `swarm_units_cap` and `swarm=` carrier, beside P6's `_step_pre_physics` / `_step_physics` / `_step_post_physics` split; read back. |

`cpp/src/bindings.cpp` and `cpp/CMakeLists.txt` were NOT touched by the
branch since `12a38642`, so they are main's files byte for byte
(`git diff origin/main -- cpp` is empty). The handover's trap (a lost brace
or bracket) could not occur there; both modules were built right after the
merge commit anyway (§4).

## 2. CLAUDE.md under #206

Main's CLAUDE.md is one index line per system, with each system's
constraints in `.claude/rules/<region>.md`. The branch's six rows became:

- **Five index lines in Sim core**, after "Device topology seam": Swarm
  store, Swarm digest section, Swarm facade, Swarm species table, Swarm
  heading law. Each line says what the system is for and who must use it.
- **One index line after "Config"**: Config reload seam. Main has no
  equivalent (`git grep on_config_reload origin/main` finds nothing; main's
  Ctrl+R path calls only `CFG.reload()`).
- **A new rule file `.claude/rules/swarm.md`** (`paths:` `src/simulation/swarm.py`,
  `src/simulation/swarm_fixed.py`, `tests/*swarm*.py`), one section per
  swarm row, and a row in CLAUDE.md's region table. It holds only what the
  code cannot say: Erik's P6 ruling of 2026-11-14 (two stores, one batch
  axis, one source of draws, fields only) and what follows from it (no
  `Unit` or unit-store columns; the swarm meets units only through per-tile
  planes; no second N, salt enum, key derivation or clock; an
  order-dependent swarm law is an applier effect kind), what must never be
  built beside each system (no class per species, no compaction, no atomic
  spawns, no stray gmap array, no capacity clamp), and what must change
  together (a new column with a `SWARM_SECT` bump; a host write with its
  dirty flag). Shapes, values, function lists and history were left out.
- **The Config reload seam's constraint** is a section of
  `.claude/rules/sim-core.md`, whose `paths:` gained `src/debug_keys.py`
  (the region table's sim-core row names it).

`tests/test_claude_rules_paths.py`: 71 passed (every named path exists,
every rule file fires on tracked files, every section has an index row, the
region table matches the files).

## 3. Build, twin ledger and tests

All runs on the Home Desktop (DESKTOP-0E98HUV, RTX 3070), conda env `data`
(Python 3.12).

- **CPU module** (`cpp/build_cpu_home.bat`) and **CUDA module**
  (`cpp/build_cuda.bat`), built right after the merge commit:
  BUILD_EXIT=0 both, `build_id()` == `5f0f189b` both.
- **Twin manifest:** `tools/twin_manifest_gen.py` reports it current after
  the merge (no regeneration diff to commit).
- **Twin ledger:** after the merge `tools/twin_debt.py` listed nine twins in
  debt, all from this branch's own edits of `src/simulation/gamemap.py` and
  `src/simulation/simulation.py` (the swarm allocation, facade and reload
  seam; main itself carries no debt at `a20711f2`): Class P `eos`,
  `div_source`, `stamp`; Class T `integrate_turn`, `wind_impulse`,
  `movement`, `verbs`, `burn_through`, `tick_end_clears`. None of their laws
  changed. After the green `cuda_green.py` run at `5f0f189b` (which runs every
  one of their gates), `tools/twin_debt.py --mark-level <each> --at 5f0f189b`
  moved their `level_at`; the ledger then reads `twin debt: none`. The
  `level_at` moves are committed with this note. (Yesterday's merge left the
  same kind of debt unmarked; it is cleared now.)
- **Swarm and neighbours** (`pytest tests -q -k "swarm or philox or
  resident_batch or digest or recorder or twin or class_p or claude_rules or
  match_draws"`): 252 passed, 1 failed, the expected
  `test_class_p_level.py` stamp check (before `cuda_green`).
- **CUDA swarm batch gate + batch invariance**
  (`tests/test_cuda_swarm_batch.py`, `tests/test_cuda_batch_invariance.py`):
  2 passed.
- **Full suite** (`pytest tests -q`, at `5f0f189b`, 64 min):
  **6360 passed, 1 failed, 2 skipped, 5 xfailed.** The one failure is
  `test_class_p_level.py::test_a_class_p_change_merges_only_with_a_green_stamp_at_head`,
  expected until the stamp names HEAD (`gamemap.py` is Class P). After
  `cuda_green.py`, `test_class_p_level.py` + `test_twin_debt.py`: 15 passed.
  No other failure, so no throwaway `origin/main` reproduction was needed.
  Main's known flaky `test_fire_tuning_level::test_generator_is_byte_deterministic`
  passed.
- **`tools/cuda_green.py` at `5f0f189b`:** green, **50 passed, 6318
  deselected** (21 min); it wrote `.cuda_green = 5f0f189b…`. It is re-run on
  this note's commit before the push (§5), because the stamp must name
  HEAD.

## 4. Goldens

**No golden moved and nothing was re-baselined.** The GOLDEN_AGGREGATE and
pinned-digest tests pass in the full suite. The `__swarm__` section is
presence-gated and hashes nothing on a swarm-free run, and
`DIGEST_SPEC_VERSION` stays main's.

## 5. Push

`tools/cuda_green.py` is re-run on the commit that carries this note and the
`level_at` moves, and `63-swarm-units` is pushed only when it is green. The
result is in the session's final report. `.cuda_green` is untracked and
local.
