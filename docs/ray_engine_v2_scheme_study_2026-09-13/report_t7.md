# report_t7.md — T7: CLAUDE.md walked against the code, stub regenerated (issue #12)

> Branch `12-t7-claude-md-walkthrough` off `fire-12` at `b3e50cb`, worktree `breach-t7`. Commits
> `29c005d` (stub), `f2d95cb` (dev_setup), `4eb52be` + `0e9dca2` (CLAUDE.md). Written by the T7
> agent; committed by the orchestrator (the harness refused the agent's file write, as for M1/M2).
> Both modules built fresh in the worktree: `build_cpu_home.bat` (42 s), `build_cuda.bat` (61 s).

## 1. Rows changed (what was stale → what it says now)

- **Header**: "2026-08-22 rules-first restructure" → 2026-09-23, re-verified after #12.
- **Environment / Python**: "always the conda env `data`" (false here: the Home Desktop has no `data` env; also a machine spec in a machine-agnostic file) → this machine's breach env, named per machine in `docs/dev_setup.md`.
- **Environment / C++**: "build scripts `cpp/build_cuda*.bat`" (the CPU scripts `build_cpu_home/data.bat` exist too) → `cpp/build_*.bat`, which one where in dev_setup.
- **Material / gas tables**: one line added. Row values are placeholders (Erik, 2026-09-22), and the thin-rows material rulings (no massive object burns; the 6 mm lumped door) are PROVISIONAL.
- **Energy closure identity**: the solid books closed with `+ e_thermostat_sum`, the thermostat described as a live modelling boundary → the term is REMOVED (T5b step 7, R1: nothing relaxes to ambient). The P-G5 history is cut to a pointer to `report_t5b.md` §7.3. The old pointer, `gas_energy_thermostat_ledger_2026-08-30.md`, describes the thermostat as live.
- **Temperature solver**: clamp "dormant until P3, CUDA twin at P3" → LIVE on CPU and CUDA since T5b (`cuda_temperature.cu:243`, `C_RAD_CLAMP = 11`), fed by `rad_fluence` and `engine.emissive`. "SOLIDS' T derived here alone" gains "(bar combustion's booked object-site deposit)": `combustion.cpp:1153` writes `temperature[s]`.
- **Radiation sweep**: "(shadow, P1)" and "until P3 … the old cast feeds the fold's int32 `rad_net`" → LIVE. The fold reads `rad_net_sweep`, the clamp reads `rad_fluence`, units read `rad_flux_sweep`. The old `rad_net`/`rad_flux`/`rad_amb` (int64 since P3a-1, not int32) carry nothing, and `rad_net` is a dead `step_tail` argument. The sweep runs host-side on both backends until P4. The conservation identity now names the `*_sweep` planes. The Fleck example "g ≤ 1: 1068 game wood, 1424 a = 0.5" is dropped: it rested on emissivity 1.0, thermal_mass 8 and the fitted scale. The row now says f == 2²⁴ across the fire range, pinned by `test_m3_derived_h_bed.py`.
- **Emissive table**: "two owners until P3" → baked at `rad_scale_derived`, the only key since T6. The Raycaster copy is vestigial (only `test_emissive_table.py` reads it), and the identity holds at equal dials.
- **Extinction planes**: "`dyn_heat_atten_q` enters the digest at P3" → digested (spec v6, T5b). Its static twin deliberately is not.
- **Integer reference**: added that the default table is a RESOLVING scale, that shipped-game numbers are measured on `E_LIVE`, and that `config_dials_match()` guards `RAD_SCALE_LIVE` (M3).

## 2. Names verified

Every backticked path and symbol in CLAUDE.md (285 atoms) was grepped on this branch, and the code behind each row the arc touched was read. Nothing named is missing. The unresolved hits were fragments (`_blast` → `_blast_bench.py`, `_mint` → `gas_energy_mint`) or the retired `TODO.md`. The names that had to be fixed were `e_thermostat_sum` and `cool_shift` (deleted), plus the shadow / "until P3" status of five rows. Verified and left: rows cite bare filenames (`debug_keys.py` is `src/debug_keys.py`, `control_source.py` is `src/control_source.py`); `assets/models/props/` is untracked by design.

**Fleck onset at the live scale** (measured with the sweep's own `fleck_f_solid_q24` on the shipped rows): wood and foliage 4868 game, kindling 5988, furniture 7620. On hull, steel, glass and door, f = 1 up to the table top. Fires plateau at 285–300 game (M3), so f == 2²⁴ across the fire range.

## 3. The stub

- Generated from the **CUDA build** with pybind11-stubgen 2.5.5 (installed into anaconda base). That build is the CPU module plus the `BREACH_HAS_CUDA` block (42 more module functions); nothing in the CPU build is missing from it. Apart from the added names and `__all__`, the only line that differs from a CPU-build stub is `HAS_CUDA: bool = True`. The documented one-liner reproduces the committed file byte for byte.
- **pyright** (from the worktree root): **1768 → 1497 errors** (2 warnings both times). A fresh CPU-build stub gives 1641, so the CUDA-only gap was 144 and the old stub's staleness 127. "Not a known attribute of module `breach_physics`" errors: 215 → 0.
- Six errors are new, and are precise typing, not bugs: five from star-unpacking `*planes` into `RadiationSweep.run` / `TemperatureSolver.step` (pyright maps the list onto the trailing `fleck_enabled` / `clamp_enabled` bool), one a deliberate int32 TypeError probe (`test_radiation_sweep_reference.py:164`).
- dev_setup's "Known gap" is closed and rewritten.

## 4. Suite

`3416 passed, 0 failed, 4 skipped, 7 xfailed` in 255 s, identical to T6's gate. The worktree was clean after the run.

## 5. Findings (recorded, not fixed)

1. `tests/_sealedbox_bisect_bench.py:235` reads the deleted `e_thermostat_sum` and raises AttributeError on its first tick — T5b step 7's missed consumer. The other four §6 benches run and close: `_quiet_books` EXACT over 1440 ticks, `_fire` EXACT over 240, `_vent` EXACT over 120 EOS steps, `_blast` PASS.
2. Every gate that writes the gas identity (`test_e1_hot_rail`, `test_thermostat_books`, the benches) carries a fifth term, `−PhysicsEngine.e_water_evac_export_sum` (arc #54 §2.7). CLAUDE.md said "four groups — there is no fifth".
3. Stale code and config comments still call the sweep a shadow or the thermostat live: `radiation_sweep.h:38`; `physics_engine.cpp:223`, `:427`, and `:382` ("slot 13", now 11); `emissive_table.h:6-7`, `:95`, `:102`; `temperature_solver.h:67`, `:577`, `:695`; `combustion.h:542`; `gamemap.py:490–560`; `physics_runner.py:968`; `raycaster.h:496`; `config.toml:535` (`k_leak` "still in SHADOW"), `:615` ("Fleck NEVER engages … every shipped row", false for the thin rows above ~4 900 game) and `:625` ("nothing live reads this key yet").
4. The generated stub gives `WaterSolver.step`, `WaterSolver.step_ripple` and `cuda_water_step` a defaulted argument before required ones — the binding's argument order, invalid Python, already in the old stub; pyright does not check `stubs/`.
5. The `main.py` row lists `--cuda` and `--resident` among the `_parse_*` flags, but they are plain argv checks (`main.py:52`, `tools/run_on_cuda.py:143`). Not arc-related; left.

## 6. For Erik

1. The gas identity's "no fifth group" versus `e_water_evac_export_sum` (finding 2).
2. `_sealedbox_bisect_bench.py` is broken (finding 1).
3. dev_setup's Work Desktop interpreter (`anaconda3/envs/data`, 3.12) comes from environment.md, not verified on that machine.
4. The `main.py` row (finding 5).

## 7. Orchestrator dispositions (2026-09-23)

- (1) The row now names `e_water_evac_export_sum` as the existing boundary export; whether it belongs inside one of the four groups is left for a later design question.
- (2) and finding 3 go to T7b: the bench's dead term dropped, the stale comments brought in line — comments only, no behaviour.
- (3) To be confirmed on the Work Desktop. (4) The row stays as the rule for NEW flags.
