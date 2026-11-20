# Arc #63: merging origin/main into 63-swarm-units (2026-11-19)

A mechanical sync: `origin/main` at `461d150f` (since the arc's last merge
base `6c5e788c`: rung 2 trainer patches in `agents/`, and the #215 stack --
grenades that bounce, painted doors retired, body temperature and stress,
the old bots' door step, and the ONE approved re-baseline of eight
pinned-match tests) merged `--no-ff` into `63-swarm-units` at `55a134c8`
(P5a, the bite). Nothing was redesigned; no gameplay or physics changed.
Method: `docs/swarm_merge_main_2026-11-15.md`.

- **Merge commit:** `d7629517` (parents `55a134c8`, `461d150f`)
- **Ledger commit:** `d9e77e0f` (the twin ledger's `level_at` moves, §3)
- **This note:** the commit on top.

## 1. Conflicts and how each was resolved

23 files were touched by both sides since `6c5e788c`; eight conflicted. In
every case both laws were kept: main's version for what main owns, the
arc's additions re-applied on top.

| File | Resolution |
|---|---|
| `src/simulation/unit_schema.py` | Both default helpers (`_poison_cause_default`, main's `_body_temp_default`). Columns: after `poison_src`, main's `body_temp_q` and `stress_q` first, then the arc's `poison_cause`. All three hash only while non-default, so no unit record of a main pin changes (main never writes `poison_cause`). |
| `cpp/src/cuda_unit_damage.cuh` | `apply_packet`'s kill clear: main's `stress.clear` + `body_temp.clear`, then the arc's `poison_cause` back to `poison_gas`. Independent column writes; order is immaterial. |
| `cpp/src/cuda_unit_conversion.cu` | `k_convert_zombies`: main's stress and body-temperature clears, then the arc's `poison_cause` clear (the arc's `unit_events_layout.h` include kept). |
| `src/simulation/unit_kernels.py` | Main's `HEAT_FIELDS` (+ `atmosphere`, `temperature`, `solid` for the body temperature's air read); the arc's `GAS_FIELDS` (+ `swarm_bite_dose`). `gas_rows`' docstring names both the bite dose and the stress tick. |
| `src/simulation/simulation.py` | Only the conductor docstring: 9c3 teargas + poison dose + the larvae's bite dose; 9c3' the poison law; 9c3'' stress clears; 9d ignition. The code (`apply_bite_dose` then `poison.tick` then `stress.tick`) auto-merged in that order and was read back. |
| `CLAUDE.md` | The region table keeps the arc's `swarm.md` row and takes main's `units.md` row (+ body temperature) and `combat.md` row (+ clearing step, stress). |
| `tests/twin_manifest.toml` | Main's `level_at` values for the 13 conflicting rows (they were moved anyway, §3); `gas_rows`' generated file set takes main's `stress.py` beside the arc's `swarm.py`. `tools/twin_manifest_gen.py` then reports the manifest current. |
| `cpp/src/unit_store_layout.h` | GENERATED: main's copy taken, then `tools/gen_unit_layout.py --write` (W 183). Never hand-merged. |

Auto-merged and read back: `cpp/src/cuda_unit_exchange.cu` (the fused
`k_gas_rows`: gas dose, then the bite dose `reduce_sum` with cause
`larva_bite`, then the poison law carrying `poison_cause`, then main's
stress tick), `cpp/src/bindings.cpp` (`unit_gas_rows` keeps the `bite`
argument), `cpp/src/cuda_unit_kernels.h`, `src/simulation/exchange.py`,
`src/simulation/poison.py`, `src/simulation/ai_zombie.py`
(`convert_marines_to_zombies` clears poison, stress and body temperature),
`config.toml`, `level_loader.py`, the rule files, the renderer's inspector.

**Painted doors:** the arc's one level, `levels/larva_studio`, has none:
`tools/migrate_level_entities.py --dry-run --doors-only larva_studio`
reports "nothing to migrate". No level was migrated.

## 2. Regenerated

- `cpp/src/unit_store_layout.h` by `tools/gen_unit_layout.py --write`
  (W 183); every other unit header (`unit_tables_layout.h` W 2836,
  `unit_events_layout.h`, `unit_requests_layout.h`, `applier_layout.h`,
  `unit_draws_layout.h`, `object_tables_layout.h`) reported current.
- `cpp/src/swarm_tables_layout.h`: `tools/gen_swarm_layout.py` reports it
  current.
- The twin manifest's file sets: `tools/twin_manifest_gen.py` reports it
  current.

## 3. Builds, twin ledger and gates

All runs on the Home Desktop, conda env `data`.

- **CPU module** (`cpp/build_cpu_home.bat`) and **CUDA module**
  (`cpp/build_cuda.bat`), built one after the other right after the merge
  commit: BUILD_EXIT=0 both, `build_id()` == `d7629517`.
- **Swarm and neighbours first** (`pytest tests -q -k "swarm or bite or
  unit_schema or unit_store or body_temp or stress or poison"`):
  **219 passed**.
- **`tools/cuda_green.py` at `d7629517`:** green, **64 passed, 6606
  deselected** (51 min 35 s); it wrote `.cuda_green = d7629517`. Every CUDA
  gate the twins name ran and passed (the swarm gate PART 2c with the bite,
  the damage gate's gas scene, batch invariance, kernels, doors, horde, ...).
- **Twin ledger:** after the merge `tools/twin_debt.py` listed 13 twins in
  debt, all from the two sides meeting in shared reference files
  (`exchange.py`, `poison.py`, `swarm.py`, `gamemap.py`, `simulation.py`),
  no law changed: `integrate_turn`, `wind_impulse`, `movement`,
  `heat_damage`, `gas_rows`, `verbs`, `burn_through`, `tick_end_clears`,
  `table_appends`, `grenades_and_charges`, `rounds`, `doors`, `horde`. All
  their device gates were green in the run above, so `tools/twin_debt.py
  --mark-level <each> --at d7629517` moved their `level_at` (commit
  `d9e77e0f`); the ledger then reads `twin debt: none`.
- **Full suite** (`pytest tests -q`, at `d9e77e0f`, 1 h 23 min, the CUDA
  wrappers included): **6660 passed, 1 failed, 3 skipped, 6 xfailed.** The
  one failure is `test_class_p_level.py::test_a_class_p_change_merges_only_with_a_green_stamp_at_head`,
  expected until the stamp names HEAD (the ledger commit came after the
  stamp). No other failure.
- **`tests/test_claude_rules_paths.py` + `tests/test_game_never_imports_agents.py`:**
  **75 passed**.
- **Agents** (`python -m pytest agents/tests -q -k "not tasks"`; the long
  task tests are main's, green there, and were skipped here as instructed):
  **147 passed, 1 failed, 37 deselected.** The failure is the known Cause
  growth, §5.1.
- **`tools/cuda_green.py` is re-run on this note's commit** so the stamp
  names HEAD, then `tests/test_class_p_level.py`; the result is in the
  session's final report (`.cuda_green` is untracked and local).

## 4. Goldens and pins

**No golden or pin of main moved and nothing was re-baselined.** Main's
eight re-baselined pinned-match tests, the GOLDEN_AGGREGATE and every
pinned-digest test pass in the full suite. No swarm-specific golden or pin
moved. The new `poison_cause` column hashes only while non-default, and
main never writes it, so main's unit records are untouched by the arc;
main's `body_temp_q` / `stress_q` reach the swarm's tests only through main's
own laws (the swarm's host-vs-device gates compare both sides, which carry
the same columns).

## 5. Findings

1. **The Cause growth moves one agents pin (known, reported, not
   changed).** `agents/tests/test_memory.py::test_a_run_without_memory_gives_mains_weights_bit_for_bit`
   pins the sha256 of a stand-in run's weights. `agents/sizes.py:129` sizes
   the `hit_cause` input by `len(CAUSE_NAMES)`, which the arc's
   `larva_bite` grows from 10 to 11, so the soldier network's first layer is
   one input wider and every weight after it differs. Proven by a scratch
   probe (not committed): the same `_no_memory_run()` gives
   `a7b1def1...` with the arc's 11 causes and exactly main's pin
   `424deb3f...` with `larva_bite` removed from `CAUSE_NAMES`. P5a already
   reported this (a network saved at the old size is refused by the layout
   check when the arc reaches main). The agent's lead decides; `agents/` was
   not touched.
2. Nothing in the reference was found inconsistent by the merge. The three
   slow states and the poison cause meet in the same clears on host
   (`damage.apply_packet`'s death branch, `ai_zombie.convert_marines_to_zombies`)
   and device (`apply_packet`, `k_convert_zombies`), and the fused device
   9c3 runs gas dose, bite dose, poison law, stress tick in the host's order.
