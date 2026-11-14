# Arc #63: merging origin/main into 63-swarm-units (2026-11-14)

A mechanical sync: `origin/main` at `12a38642` (1403 commits since the
merge-base `fdc986a9`) merged `--no-ff` into `63-swarm-units` at `1eb3be0c`
(28 arc commits). Nothing was redesigned. Where main's systems and the swarm
store overlap in meaning, this note records it and leaves the call to Erik.

- **Merge commit:** `1951b154` (parents `1eb3be0c`, `12a38642`)
- **Follow-up commits:** `44170261` (swarm_fixed vs #88's gas_fixed rename,
  plus a stale digest-spec comment), `88b0206f` (classify `swarm.py` in the
  twin manifest), then this note.

## 1. Conflicts and how each was resolved

| File | Kind | Resolution |
|---|---|---|
| `.gitattributes` | add/add | Main's file. It is a superset: its LF lines for `philox32.py`, `philox32.h` and `tests/data/*.txt` cover the arc's three. |
| `cpp/src/philox32.h`, `src/simulation/philox32.py`, `tests/data/philox4x32_10_kat.txt` | none | Identical blobs on both sides (`0b1e4c61`, `38462dae`, `d25adf4b`); main vendored them from this branch (#29 P1a / P3b). The recorded-hash drift test passes. |
| `tests/test_philox32.py` | add/add | Main's version: the arc's test plus skips when the binding is absent and a drift hash that tolerates CRLF. |
| `cpp/src/bindings.cpp` | content + duplicate | Main's file, byte for byte. Git auto-merged the philox `m.def` block twice (main's copy and the arc's, with the same names and signatures), and the conflicting hunk was only the include comment. The arc's duplicate is dropped, so there is one binding and it uses main's names (which are the arc's names). |
| `stubs/breach_physics.pyi` | content | Main's stub plus the five `philox32_*` defs, `PHILOX32_TWO_PI_Q16` and their `__all__` entries. Main's stub lacked them even though main's module exports them. Hand-merged rather than regenerated: a CPU-build regeneration would flip `HAS_CUDA` and drop the CUDA names that main's committed stub carries. |
| `src/simulation/factions.py` | content | Both: main's `ZOMBIE_TEAM = 255`, `LEGACY_ZOMBIE_TEAM = 1` and `team_from_level`, and the arc's `FACTION_NONE = -1`. The three values do not alias. |
| `tests/field_ab_harness.py` | content | Main moved the per-tick snapshot into `capture_snapshot()` (public, used by `tools/run_match.py`). The arc's swarm carrier write and `require_swarm_carrier` moved into it, so every capture path writes the carrier, as the Swarm digest section row requires. |
| `config.toml` | content | Both: the arc's `[recorder] swarm_units_cap` and `[swarm.larva]`, then main's new `[sound]` block. |
| `src/simulation/gamemap.py` | content | `allocate_swarm_store(self)` stays the last statement of `__init__`, now after main's new `_seed_temp_init(...)` call. |
| `src/simulation/simulation.py` | content | Both imports (swarm, plus main's timed-charge and heat-source systems). In `_reset_internal`, main's `MatchDraws` construction comes first, then the arc's `SwarmUnits` facade. |
| `CLAUDE.md` | content | Main's Sim-core rows (Energy closure, Temperature solver, Radiation sweep … Batched resident engine) plus the arc's six swarm rows: Swarm store, Swarm digest section, Swarm facade, Swarm species table, Config reload seam, Swarm heading law. See §2 for the two rows dropped. |

Auto-merged without conflict, then checked by reading: `src/debug_keys.py`
(the `on_config_reload()` call; main has no competing reload seam),
`src/simulation/recorder.py` (main's #88 trace-plane dequantize beside the
arc's swarm ring family), `tests/field_digest.py` (the `__swarm__` fold in
`tick_digest` after main's v6/v7 bumps and its new `DIGEST_SECTIONS`) and
`tests/field_digest_spec.toml`.

### Follow-up fix (`44170261`)

Main's #88 split `gas_fixed` into trace, bulk and q16 names and **deleted**
`FP_SHIFT`, `FP_ONE`, `FP_ONE_F`, `quantize*` and `dequantize*`.
`tests/test_trace_currency.py`'s lint keeps them deleted. The arc's
`swarm_fixed` re-exported those names, so after the merge every
`import simulation.gamemap` failed. `swarm_fixed` now imports the q16 family
(`Q16_SHIFT`, `Q16_ONE`, `quantize_q16_scalar`, `dequantize_q16`) and keeps
its own public names (`FP_SHIFT`, `FP_ONE`, `quantize_scalar`, `dequantize`)
as aliases. No caller changed and no new rounding copy exists. The unused
`FP_ONE_F`, `quantize` and `dequantize_f32` re-exports are gone. The same
commit fixes the digest spec's swarm-section comment, which said "`version`
stays 5"; it has been 7 since #88.

## 2. Where main's systems and the swarm store overlap

1. **Batch axis vs `N_ENV`: left for Erik.** Main's batching (#29 P5
   `ResidentBatch`, `BatchedEnv`) keeps one `GameMap` per env and stacks
   them. `ResidentBatch` allocates `(n_env,) + field.shape` for every
   `GameMap._RESIDENT_FIELD_NAMES` entry and binds slice `e` to env `e`'s
   map. The swarm arrays are in `_RESIDENT_SYNCED` with shape
   `(N_ENV = 1, max_units)`, so in a batch they become `(n_env, 1, max_units)`
   device buffers. That works (shape-generic, env 0's slice is the map's own),
   but the swarm's `N_ENV` axis now duplicates main's batch axis. The Swarm
   store row ("N_ENV is the codebase's first batch-axis constant … never a
   second N") was written before main's batch existed and is now arguably
   stale. Erik's options: drop the leading `N_ENV` axis (per-map
   `(max_units,)`, with `ResidentBatch` supplying N), or keep it until P3.
   Unchanged here. The "RL-batch habits" row was deleted on main (absorbed
   into "Batched resident engine") and stays deleted.
2. **Philox: main's `MatchDraws` vs the swarm's planned use. Left for Erik,
   and it affects P3.** Main's "Match draws" row makes `src/simulation/draws.py`
   (`MatchDraws`, `sim.draws`) plus `rng_salts.Salt` ("THE salt enum") the
   only way the sim draws: key = the sim's seed, counter
   `(tick, draw_index, salt, unit_or_site_id)`. A new draw site is a `Salt`
   member. `sim.rng` is drawn from by nothing, and `spawn_match` replaces it
   with `RefusingRng`. The arc's "Deterministic RNG" row (Philox "beside
   `sim.rng`'s PCG64") contradicts that, so it was **dropped** in favour of
   main's row. Its counter layout is the same as main's. The P3 design
   (`docs/swarm_P3_impl.md` §6, D13) plans its own `philox_streams.StreamSalt`
   enum, a `match_key(sim.rng)` read of the SeedSequence entropy and a
   `Simulation.total_tick`. All three now overlap with main: a second salt
   enum where main has the one `Salt`; a second key derivation from the seed
   where main's `MatchDraws` already holds the key (and keys an unseeded sim
   differently, from one draw of its generator); and a second tick clock
   beside `sim.draws`'s clock and `tick_base`. P3's doc needs a revision pass against `draws.py`
   and `rng_salts.py` before implementation. The larva salts would most
   naturally become `Salt` members, and the device side would use the same
   key and counter. Not changed here.
3. **Unit schema (#29 P3a): no overlap.** Swarm units are not `Unit`s and
   never touch `unit_schema` or `Unit.__slots__`.
4. **Digest and recorder: no conflict.** Main's new `DIGEST_SECTIONS` (#31)
   holds field planes hashed while non-zero inside `field_digest`. The swarm
   carrier is a carrier-type section folded in `tick_digest` like
   `__entity__`, so it stays there. `DIGEST_SPEC_VERSION` stays 7 (main's)
   and the swarm section keeps its own `SWARM_SECT_V1` preamble. The
   recorder's swarm rings are additive and presence-gated.
5. **Reload seam: no overlap.** Main has no config-reload seam of its own.
   `Simulation.on_config_reload()` is still the only one. (Main's Config row
   still says "F5 reload" while the key is Ctrl+R; that predates this merge
   and is untouched.)
6. **Class P:** `src/simulation/gamemap.py` is a Class-P path in
   `tests/twin_manifest.toml`, and the arc changes it (the swarm import, the
   `_RESIDENT_SYNCED` extension, the allocation). So
   `tests/test_class_p_level.py` requires a green `tools/cuda_green.py`
   stamp at HEAD (§3).

## 3. Build and test results

All runs on the Home Desktop (DESKTOP-0E98HUV, RTX 3070), conda env `data`
(Python 3.12.14).

- **CPU module:** rebuilt with `cpp/build_cpu_home.bat` →
  `cpp/build/Release/breach_physics.cp312-win_amd64.pyd`, BUILD_EXIT=0. The
  tree's previous `.pyd` was cp311 from 2026-09-24 and could not load under
  `data`.
- **CUDA module:** rebuilt with `cpp/build_cuda.bat` (CUDA 12.4, Ninja) →
  `cpp/build_cuda/breach_physics.cp312-win_amd64.pyd`, BUILD_EXIT=0, then
  rebuilt again at `88b0206f` so its `build_id()` is HEAD.
- **Swarm + philox first** (`pytest tests -q -k "swarm or philox"`, at
  `44170261`): **47 passed**.
- **Full suite** (`pytest tests -q`, at `44170261`, 48 min 24 s):
  **6147 passed, 3 failed, 2 skipped, 5 xfailed**.
  1. `test_twin_debt.py::test_every_executed_simulation_module_is_twinned_or_classified`:
     `src/simulation/swarm.py` was unclassified, a merge consequence (main's
     #29 P1b twin-debt rule meets the arc's new module). Fixed in `88b0206f`
     by a `[not_twinned]` line, "setup only on the corpus". Re-run: 10/10
     passed.
  2. `test_light_curve.py::test_save_rewrites_only_its_three_lines`: my
     working copy of `config.toml` had CRLF line endings, left there when I
     resolved the conflict (the attribute says `eol=lf`; the index was always
     LF). Not a code defect and nothing to commit: re-checking out the file
     fixed it. Re-run: 31/31 passed.
  3. `test_class_p_level.py::test_a_class_p_change_merges_only_with_a_green_stamp_at_head`:
     expected, since `gamemap.py` is Class P. After
     `tools/cuda_green.py` (below): 5/5 passed.
  Main's known pre-existing failure,
  `test_fire_tuning_level::test_generator_is_byte_deterministic`, **passed**
  in this run. No failure touched swarm or philox code, so none needed the
  throwaway `origin/main` reproduction.
- **`tools/cuda_green.py` at `88b0206f`:** green, **43 passed, 6114
  deselected** (3 min 52 s); it wrote `.cuda_green = 88b0206f…`.

I did not re-run the full suite at `88b0206f`. The only tracked change after
`44170261` is the one-line manifest entry; both changed areas were re-run
(above).

## 4. Goldens

**No golden moved and nothing was re-baselined.** The GOLDEN_AGGREGATE /
pinned-digest tests (`test_b6_logic_golden`, `test_s3b_fire_determinism`,
`test_thermostat_books`, `test_light_field`, `test_fire_flicker`,
`test_radiation_sweep_shadow_wiring`, `test_sweep_heat_law_properties`,
`test_b1_signal_bus`, `test_b2_nodes`, `test_swarm_digest`) and
`test_match_draws` / `test_trace_currency` all pass. This is as designed: the
`__swarm__` section is presence-gated, absent on every swarm-free run, and
hashes nothing then. `DIGEST_SPEC_VERSION` stays at main's 7.

## 5. Push

Pushed `63-swarm-units` to origin with this note's commit on top.
`.cuda_green` is untracked and local; after this docs-only commit it no longer
names HEAD, and the next CUDA gate re-stamps.
