# Handover brief — Patch A: audit hygiene (2026-08-04)

**What this is.** A bounded, zero-decision work package extracted from
`docs/codebase_audit_2026-08-03.md`, safe to hand to an autonomous agent without
Erik in the loop. Everything here is **behaviour-preserving**, gated by the
existing suite, and requires **no physics or feel judgement**. Everything that
needs Erik is listed in §4 and explicitly excluded.

## ★ 0. Base branch — read this before creating a worktree

**Branch from `thermal-mass-axis`, NOT from `main`** (corrected 2026-08-04 after
an actual check; an earlier draft of this brief said main and was wrong).

The entire fire bench suite — `fire_timing_harness.py`, `fire_tune_loop.py`,
`fire_room_bench.py`, `fire_o2_supply_baseline.py`, `fire_supply_radius_sweep.py`,
`fire_smother_curve_sweep.py`, `fire_wind_level_probe.py`, `fire_tune_plot.py` —
**exists only on `thermal-mass-axis`.** `main` has none of them. Items **A1, A2,
A4 and A8 target files that do not exist on main**, so a worktree branched there
finds half the patch missing.

The four audit docs live on `main` (commit `daafdae`). Read them from there, or
merge `main` into the working branch first — the docs are additive, so the merge
is trivial.

Worktree layout is unusual, so check before assuming:
- `main` is checked out at `breach.worktrees/onephase-wego`, **not** at
  `projects/breach` (which sits on `o2-continuous-law` with a dirty tree).
- The fire branch is at `breach.worktrees/thermal-mass-axis` (HEAD `d7765bc`).

A fresh worktree needs its own CPU build: `cpp/build_cpu_data.bat` (the
checked-out `.pyd` predates P-F1a and will fail on `RADIATION_RANGE_MIN`).
A4 and A5 additionally need a CUDA build to verify; if no GPU is available,
**report those two as unverified rather than skipping them silently.**

**Execution:** `autonomous-patch-workflow` skill. Own branch, own worktree.
`pytest tests -q` green before merge. **No golden may be re-baselined in this
patch** — if a golden moves, that is a finding to report, not a thing to fix.

---

## 1. The work items

Ordered so the cheapest, highest-value ones land first. Each is independent;
land them as separate commits so any one can be reverted alone.

### A1 — Erik's tuning loop is a brick (1 minute, do this first)
`tools/fire_tune_loop.py:383` still carries `"k_fire_heat": 33.0` in the
executed `TUNE` dict. The key was retired from `[physics.fire]`; `_resolve_key`
raises `KeyError` and the tool exits before doing anything.
- Delete that dict entry (keep the number in the surrounding comment prose).
- Same dead key in the four `ALT_*` blocks (`:528-562`) — strip or delete them.
- Delete the `OPTIONAL_COLS` / `REQUIRES` banner machinery (`:94-120, 626,
  684-696`): it warns that the harness lacks `hot` / `Tfar_game` / `X_local` /
  the warm seed. All four are present. It is actively misleading.
- The scorecard's `predicted_T_star` (`:672-680`) and `I_crit` row (`:724`)
  compute from `k_fire_heat` and `fire_T_ext`, both retired/inert. Either delete
  the rows or re-derive them from the live `H_bed` chain. **Prefer deleting** —
  re-deriving is a physics judgement.

### A2 — `apply_overrides` is not atomic (15 min)
`tools/fire_timing_harness.py:144-152` mutates CFG key-by-key and only returns
the restore list on success. A `KeyError` on key N leaves keys 1..N−1 patched
with no handle to undo them — which is how A1 silently poisons every later run
in the same process (`fire_supply_radius_sweep.py:138`,
`fire_smother_curve_sweep.py:81` both call it outside their `try`).
Build the restore list first, apply inside a `try`, roll back on exception.

### A3 — the cross-toolchain determinism hole (~1 h)
`cpp/CMakeLists.txt:139-153` wraps the per-source `/fp:strict` override in
`if(MSVC)`; the non-MSVC branch applies **`-ffast-math` globally** (`:28-34`).
The file's own comment (`:112-120`) claims every sim TU is strict — true only on
Windows. Every `quantize((double)…)` boundary in the live EOS path is exposed on
a gcc/clang build.
- Hoist `set_source_files_properties` out of `if(MSVC)`.
- Give the else branch `-ffp-contract=off -fno-fast-math` (or `-ffp-model=strict`).
- Make `-Xcompiler=/fp:strict` (`:109`) host-conditional.

**MSVC output must be bit-identical** — verify with a digest run before/after.

### A4 — run the two gates that already exist (~30 min)
`tests/cuda_po2b_check.py` and `tests/cuda_sky_exchange_check.py` are complete,
correct parity gates with `PO2B_RESULT:` / `SKY_RESULT:` markers — and **no
pytest wrapper**, so `pytest tests -q` never executes them. This leaves the
shipped `draw_r = 2` combustion path with zero collected GPU parity coverage.
- Add `tests/test_cuda_po2b.py` and `tests/test_cuda_sky_exchange.py`, copying
  `tests/test_cuda_p69_combustion.py` verbatim (swap module name + marker).
- Add a meta-test asserting every `tests/cuda_*_check.py` is referenced by some
  `tests/test_*.py`, so the next one cannot be orphaned.

★ **These gates have never run.** If either goes red, **STOP and report** — do
not "fix" the engine to make a never-run gate pass. A red here is a finding.

### A5 — a confirmed CPU≠GPU (and GPU≠GPU) divergence (~30 min)
`cuda_fire.cu:286,483-486` returns the destroyed-wall list in **atomic arrival
order**; the CPU (`fire_simulation.cpp:324`) returns row-major. `destroy_wall`
is order-dependent — `gamemap.py:1749` writes `breach_mask[fy,fx]`, which the
next iteration's `exposes` test (`:1743-1748`) reads, as does
`_neighbor_mean(atmosphere)` (`:1756`). Two walls destroyed near each other in
one tick diverge, and diverge *run to run on the same GPU*.
- `std::sort(idx.begin(), idx.end())` before the `(y,x)` map at `cuda_fire.cu:483`.
- Change `tests/cuda_fire_check.py:170` from `set(...) != set(...)` to list
  equality, so the gate can see this class of bug at all.

CPU path bit-identical; no CPU golden moves. Requires a CUDA build to verify.

### A6 — restore the P6.4 gate's ambient coverage (~1 h)
`eos_kick_compression_reference` (`eos_solver.cpp:1345`) omits the B3c
`sponge_udamp` velocity-damping band that live `step()` applies
(`eos_solver.cpp:654-663`), and `bindings.cpp:2282` hard-codes
`is_ambient = nullptr`. The header claims it "Replays **EXACTLY**"
(`eos_solver.h:453`) — false since B3c. The CUDA twin
(`cuda_kick_compression.cu:168`) *does* have the band, so the CPU reference is
the one out of step and a lockstep failure on a planetside map would blame the
GPU.
- Add a `sponge_udamp` parameter (default `nullptr`) + the band block verbatim.
- Expose `is_ambient` / `sponge_udamp` through the binding.
Behaviour-preserving (nullptr default reproduces today).

### A7 — one reachable divide-by-zero (15 min)
`eos_solver.cpp:308` computes `(t_max_abs_raw << 16) / (int64_t)t_amb_q` with no
floor, and `T_AMB_K` is `def_readwrite`-exposed (`bindings.cpp:2134`). Every
other divide in that file is floored. Add `if (t_amb_q < 1) t_amb_q = 1;` after
`:278`. Mirror in `cuda_eos_step.cu` / `cuda_eos_resident.cu` where `c_local` is
recomputed. Never binds at 290 — behaviour-preserving.

### A8 — a test fixture that violates the invariant another test pins (15 min)
`tests/test_thermal_mass_axis.py:646` uses `int(0.21 * FP_ONE)` = **13762**
(truncation) where the suite's convention is `quantize_scalar(0.21)` = **13763**
(round-half-away). 13762 + 51773 = 65535, so the fixture breaks the exact
`N_amb == FP_ONE` invariant that `tests/test_eos_p1_calibration.py:55-62`
exists to pin.
Switch to `quantize_scalar`. **If this moves that test's expected values, report
the delta rather than silently re-pinning them.**

### A9 — delete the confirmed-dead surface (~1 h, all zero-risk)
Each verified by repo-wide grep as having no caller:
- `cpp/src/grid2d.h` — entire file, zero references anywhere (51 lines).
- The orphaned CUDA sink-hop twin — its **CPU counterpart was deleted**
  (`smoke_dynamics.h:34`): `cuda_smoke.cu:278-312`, `:430-503`,
  `cuda_smoke.h:67-88`, `bindings.cpp:746-769` (~150 lines). Fix the false claim
  at `cuda_smoke.h:17` while there.
- `physics_engine.h:55` `wave_p_f_`, `:70` `atm_f_` — declared, never used.
- `physics_engine.cpp:596-603` — orphaned comment describing a deleted function,
  now heading the live `step_water`; and `/fp:precise` → `/fp:strict` at
  `:614`, `:5`, `:83`.
- `combustion.cpp:82` — file-local `D4_OPP`, shadowed (`:654` uses `cd::D4_OPP`).
- `src/simulation/coords.py` — entire module, zero references (43 lines).
- `physics_runner.py:1434-1465` `_step_ripple`; `field_edit.py:405-420`
  `_combine_atmosphere`; `physics_runner.py:45-46` `residency_enabled`;
  `unit.py:95-96` `_COMPASS_LABELS` / `_SECTOR_HALF`.
- 7 unused imports: `combat.py:46,47,53`, `field_edit.py:51`,
  `simulation.py:70`, `recorder.py:226`.

**Do not delete** `tests/_xarch_perfield_digest.py` despite its "THROWAWAY
diagnostic" docstring — it owns the sanctioned `GOLDEN_AGGREGATE`.

---

## 2. Gates for the whole patch

1. `pytest tests -q` green (997+; the two new wrappers may add red — see A4).
2. **No golden re-baselined.** Report any movement.
3. A3: MSVC build digest-identical before/after.
4. A5/A6/A7: CPU↔CUDA lockstep at tol 0 where a CUDA build exists.
5. A1/A2: `fire_tune_loop.py` actually runs end to end and writes its PNG.

---

## 3. Two follow-on patches (each its own branch, also decision-free)

**Patch B — vectorize `find_burst_walls`** (`gamemap.py:1595`, called every tick
from `simulation.py:1156`). Pure-Python loop over every solid tile × 4
neighbours plus a full-grid float64 alloc: **10–38 ms/tick** measured at shipped
grid sizes, against a 27.1 ms full resident CUDA tick. It is a pure gather; the
only subtlety is keeping the tie-break stable (today: `list.sort` on the float
spread, stable → scan order). Behaviour-preserving, but gate it with a digest
A/B on `playground` and `unhcr_vessel`. Half a day.

**Patch C — retire the weapon-table global** (`weapons.py:671`). `combat.py`
reads the module global; `Simulation` reads its instance. Already caused one
cross-test poisoning bug, currently held back by an autouse fixture
(`conftest.py:1-24`). It becomes load-bearing the moment two `Simulation`s
coexist in one process — i.e. the RL training endgame. Thread
`sim.weapons_tables` through `combat.py`'s 8 call sites. 2–3 h.

---

## 4. Explicitly NOT in scope — these need Erik

Do not touch any of these; each is a decision, not a fix.

- **The two Kelvin maps / a `φ_exp` expansion dial** — a physics ruling.
- **Interior air damping** (`[materials.air] wave_absorb`) — feel-adjacent, and
  the overnight measurements found an instability window with fire in the loop.
- **P-F1b merge sequencing.**
- **The `cool_shift = 9` vs shipped `5` calibration anchor gap** (16× in
  `2^(cool_shift−his)`) — tuning.
- **`p*` precision staging** (`eos_solver.cpp:565`, the 0.44% quantization
  plateau) — digest-moving, needs a written re-baseline.
- **`rad_scale` / `burn_rate` / `RADIATION_RANGE` vs `tile_size_m`** — deriving
  them properly is a design decision.
- **`_BLAST_WALL_MATERIALS`** (kindling is currently blast-proof) — the ledger
  already has a designed fix (per-material `blast_pressure_threshold`) and it is
  feel-adjacent.
- **Anything in the fire/atmosphere oscillation doc.**
