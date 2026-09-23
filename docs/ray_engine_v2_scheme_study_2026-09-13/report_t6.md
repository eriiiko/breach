# report_t6.md — T6: delete the old heat law (issue #12)

> Branch `12-t6-old-law-deletion` off `fire-12` at `3b9fda3`. Worktree
> `C:\Users\steen\projects\breach-t6`. Skeleton committed before any code.

**Baseline**: `3420 passed, 4 skipped, 8 xfailed, 0 failed` in 263.73 s.
Timing probes before any edit: `default_scenario_sim` (radiatively inert)
0.83 ms/tick; playground + one hot burning wood tile (the old cast's real
worst case) 13.02 ms/tick.

## 1. Scope found by investigation

Bigger than the brief's list once traced end to end: three whole CUDA
test/check-script pairs called the deleted functions directly (§4);
`step_tail` still takes `rad_net` as a required argument nothing internally
reads (§3); five more files read a now-deleted config key, found by a
post-suite grep, not the brief (§7).

## 2. What was deleted

**Python** — `PhysicsRunner.cast_fire_heat` (both call sites) and everything
that fed only it: `_raycaster_on_cuda`, `fire_ray_count`/`fire_range_base`/
`fire_range_per_i`/`fire_intensity_base`/`fire_intensity_per_i`/`fire_color`,
`_fire_scratch_rgb`/`_dx`/`_dy`, `_fire_gas_f`, the `T_emit_gate`/
`RADIATION_RANGE` bindings + validation, `raycaster.rad_scale =` (fitted key).

**C++ + bindings** — `Raycaster::cast_from_fire_plane`, `build_fire_sources`,
`build_fire_ray_list`, `march_ray_radiation`, the `T_emit_gate`/
`RADIATION_RANGE_MIN`/`radiation_range` members, and their two pybind
bindings. `cast_source_directional` lost its `has_rad`/`RadRay`/
`march_ray_radiation`-call branch — dead even *before* this patch: neither
of its two callers ever passed non-null `rad`/`rs` (the pybind binding has
no such argument; the deleted cast's own light-only sub-cast always passed
null).

**Config** — `[physics.fire]` `rad_scale`/`T_emit_gate`/`RADIATION_RANGE`/
`fire_ray_count`/`range_base`/`range_per_intensity`/`intensity_base`/
`intensity_per_intensity`/`color`. Removed outright, not tombstoned like
`k_fire_heat`: nothing reads them, so there's no old-config case to protect.

**Tools** — `storm_ledger.py` (`fire_cast` pass dropped, not remapped: the
sweep runs *inside* `step_tail`, already wrapped as `tail`); `fire_tune_loop.py`
+`storm_probe.py` (`T_emit_gate` override rows dropped); `fire_timing_harness.py`
(tick-order docstring). Not in the brief, found by follow-up grep:
`bench_radiation_sweep.py` (rebased onto `rad_scale_derived`),
`fire_tuning_lab.py`'s reach bench (`"fitted"` key → literal — §7).

**Tests deleted** (property — why gone):
- `test_cuda_pr1_fire_plane.py`+check — CPU/CUDA bit-identity of
  `cast_from_fire_plane`/its twin; both gone.
- `test_s8c_fire_heat_bench.py`+check — batched-vs-per-source GPU payoff for
  `cast_fire_heat`'s dispatch; method + the dials its source-build read gone.
- `test_cuda_s2b_raycaster_live.py`+check — live GPU/CPU wiring of
  `cast_fire_heat` (called it directly).
- `...::test_the_flux_sensor_ceiling_did_not_move_with_the_width` — the old
  cast's `rad_flux` ceiling bounding unit damage; `rad_flux` is never written
  now (already strict-xfail, named by the brief).
- `test_emissive_table.py::test_live_runner_bakes_each_owner_at_its_own_configured_scale` —
  the two E° owners were DELIBERATELY bound to different keys during the
  P2b–P3c window; one key now, so "must differ" would itself be wrong.

**Tests edited** — `test_radiation_sweep_shadow_wiring.py`: two tests lost
their old-cast half (renamed to say so); `INT32_LIMIT`/`INT32_MAX` removed
(orphaned). `test_emissive_table.py::_dials()` rebased onto `rad_scale_derived`;
its consumer's reference bake made scale-explicit. `test_hover_readout.py`
and the reach bench needed the *old* number as a bare literal instead (§7).

## 3. What was kept, and why

- `gmap.rad_net`/`rad_amb`/`rad_flux` — **not deleted**. `rad_net` is a
  required argument to `step_tail` (every live tick) and to
  `TemperatureSolver.step`'s direct binding; reading `step_tail`'s body (CPU
  and CUDA) proves neither forwards this parameter's *contents* into the
  fold — only `rad_net_sweep` is — so the values are dead but the array is a
  required, tested argument of a live function. `rad_flux` is independently
  pinned by two unrelated E2E tests (`test_eos_p4_combustion.py`,
  `test_ps1_smoke_roundtrip.py`) that `.fill(0)` it themselves. `rad_amb` has
  no independent reason to survive but stays with its two siblings. Real
  dead-parameter debt on 3 bindings (`step_tail`, `TemperatureSolver.step`,
  `cuda_temperature_step`), each with its own dtype-refusal test — removing
  it is a temperature-solver signature change, outside "delete the cast."
- `RAD_LIM_SHIFT`/`rad_pair_budget`/`rad_pair_budget_s` — `cuda_raycaster.cu`
  (out of scope until P4) `#include`s `raycaster.h` for these `RC_HD` symbols
  in its own now-unreachable kernel; deleting them breaks a file this patch
  must not touch. Both builds verified green.
- `Raycaster::rad_scale`/`kelvin_ambient`/`k_temp_to_kelvin`/
  `bake_emissive_table`/`emissive_table` — Kelvin dials feed `engine.emissive`'s
  copy + `vacuum_ambient_K`'s default; bake stays the shared-implementation
  identity's other owner. `set_raycaster_backend`/`get_raycaster_backend` —
  write-only dead state (sole reader was `cast_fire_heat`), kept because ~6
  CUDA check scripts call them unconditionally, no `hasattr` guard.
- `cast_source_directional`/`march_ray_directional`/`build_ray_list` and their
  `RadCtx*`/`RadSource*`/`RadRay*` params — this **is** the kept render march,
  out of scope until P6; only the provably-unreachable branch was cut.
  `cuda_s2_check.py`+wrapper genuinely are still P4's (never called
  `cast_fire_heat`/`cast_from_fire_plane`, only hand-built `LightSource`s).

## 4. Deviation from the stale design v3 P4 assignment

Design v3's P4 row assigned the three CUDA test pairs above to "P4" — written
when P3/P3c were the binding-deletion step. The T-plan moved that deletion to
T6, leaving `cuda_raycaster.{cu,h}` themselves at the true P4; all three call
the deleted functions directly, so they break the moment T6 runs, regardless
of the older plan. Deleted now, confirmed by reading each file's body, not
by trusting the stale assignment. `report_t4.md` §1.4 independently flagged
the same three "disposed at P4" before the T-plan existed — superseded for
these three only; `cuda_s2_check.py` genuinely is P4's.

## 5. Gate

**3416 passed, 0 failed, 4 skipped, 7 xfailed** (was 3420/0/4/8) — passed
dropped by exactly the 4 deliberately-deleted tests, xfailed by exactly 1.
`test_w6_armory.py` (the `GOLDEN_AGGREGATE` owner) standalone: 21/21 green —
golden unmoved. CUDA gates (`-k cuda`): 23 passed, 0 failed. Both builds
green at every checkpoint, not just once at the end.

## 6. Timing before/after

No-fire scenario: 0.83 → 0.79 ms/tick. Hot-wood-tile scenario: 13.02 → 12.90
ms/tick. Smaller than "the old cast ran every tick" suggests: the radiation
sweep (live since T5b) does a whole-grid traversal per ordinate *regardless
of fire state*, already dominating a single-fire scene. The deleted cast's
cost scaled with emitter count, so a many-fires scene would show a bigger
win; not measured (no "before" baseline without reverting).

## 7. Findings (not fixed — out of scope)

1. Five files outside the brief read `[physics.fire] rad_scale` and broke:
   `test_hover_readout.py` (fixed — needed a literal; the *derived* scale
   never engages the Fleck damping it exercises, per config.toml's own
   documented consequence), `bench_radiation_sweep.py` (fixed — rebased onto
   `rad_scale_derived`), `fire_tuning_lab.py`'s reach bench (fixed — literal;
   reworking its "fitted vs derived" shape to drop an arm is bigger work this
   mechanical patch doesn't take on). All three verified green; the suite
   only caught the first.
2. `report_t4.md` §5.3's "one line for T6" (`MaterialTable`'s unused
   `comb_cfg` param) and §5.4's dangling `cool_shift`/`test_fire_heat_source`
   prose references (~8 files, one a 350-line historical trail) are both a
   different cleanup category from the radiation cast; left alone.
3. `step_tail`'s dead `rad_net` parameter (§3) is real, non-trivial debt
   (3 bindings, 2 tests) outside T6's mandate.

## For Erik
- Suite green, golden unmoved, both builds green (§5); one confirmed deviation from the stale design v3 P4 assignment (§4).
- `step_tail`'s dead `rad_net` argument (finding 3): real debt, candidate follow-up, maybe folded into T7.
- `fire_tuning_lab.py`'s reach bench (finding 1) compares a live key against a historical literal now — worth a ruling.
- Two out-of-scope T4 leftovers re-flagged, untouched (finding 2).
