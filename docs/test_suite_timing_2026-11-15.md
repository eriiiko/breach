# Where the test suite's time goes (#205) -- 2026-11-15

Step 1 of #205: the measurement table, what it says, and which of the issue's
candidates pay. Remade from any suite run by `tools/suite_timing.py`.

## The run measured

P6.2's final full-suite run on the Home Desktop, 2026-11-14: `pytest tests -q
--durations=200 --durations-min=1.0`, one pytest process, one BLAS thread, the
PC otherwise busy with other agents. 6196 tests (1 failed -- fixed since, `b891fbf5`),
**3151 s wall (52.5 min)**. The output is `C:\tmp\suite_durations_p62.txt`
(not committed). A test's time below is its setup + call + teardown.

With `--durations=200` only the slowest 200 tests are listed (every one over
1.3 s); together they take 2903 s, **92 % of the run**. The other ~6000 tests,
collection included, take 248 s.

## By kind

The kind is read from each test's source (its function, the module functions
it calls, the fixtures it names) by `tests/_suite_kinds.py`; first match wins.

| kind | seconds | share | tests listed |
|---|---:|---:|---:|
| plays a match (ship_demo bots / bots_v1 / a policy, through `run_match`, `MatchSession`, the farm) | 1967 | 62 % | 71 |
| CUDA subprocess check | 387 | 12 % | 27 |
| other (pure-Python sweeps, small scenes) | 224 | 7 % | 56 |
| window / GL subprocess (the E2E driver, GL checks) | 188 | 6 % | 12 |
| physics scene (a burn, a vent, a decay) | 102 | 3 % | 18 |
| level generation | 32 | 1 % | 11 |
| source scan (lints) | 3 | 0 % | 2 |
| the ~6000 tests not listed, and collection | 248 | 8 % | |

## By file (over 5 s)

| file | seconds | tests listed | last change |
|---|---:|---:|---|
| `test_unit_schema.py` | 563.4 | 5 | 2026-11-12 |
| `test_bots_door.py` | 227.7 | 10 | 2026-11-13 |
| `test_cuda_unit_kernels.py` | 211.3 | 1 | 2026-11-14 |
| `test_reward_v1.py` | 157.7 | 8 | 2026-11-13 |
| `test_unit_impulse.py` | 151.9 | 2 | 2026-11-13 |
| `test_obs_spec_v1.py` | 111.0 | 7 | 2026-11-12 |
| `test_bots_v1.py` | 97.2 | 1 | 2026-11-14 |
| `test_exposure.py` | 97.0 | 1 | 2026-11-13 |
| `test_rl_env.py` | 90.1 | 8 | 2026-11-12 |
| `test_explosion.py` | 84.6 | 1 | 2026-11-12 |
| `test_unit_positions_q16.py` | 84.2 | 1 | 2026-11-12 |
| `test_turning_law.py` | 54.5 | 2 | 2026-11-13 |
| `test_match_draws.py` | 54.4 | 2 | 2026-11-12 |
| `test_zombie_control.py` | 53.5 | 1 | 2026-11-13 |
| `test_follow_camera.py` | 49.3 | 2 | 2026-10-31 |
| `test_cuda_unit_store.py` | 47.9 | 1 | 2026-11-14 |
| `test_m3_derived_h_bed.py` | 45.2 | 4 | 2026-09-25 |
| `test_fire_vfx_gl.py` | 42.3 | 1 | 2026-10-23 |
| `test_cuda_s8a_residency.py` | 41.0 | 1 | 2026-10-04 |
| `test_fire_o2_invariant.py` | 38.5 | 2 | 2026-08-30 |
| `test_action_spec_v1.py` | 38.0 | 3 | 2026-11-13 |
| `test_orders.py` | 33.2 | 3 | 2026-11-12 |
| `test_control_direct_e2e.py` | 31.7 | 1 | 2026-11-10 |
| `test_magazines.py` | 28.8 | 3 | 2026-11-12 |
| `test_wind_doorway_e2e.py` | 25.4 | 4 | 2026-11-10 |
| `test_trace_currency.py` | 25.3 | 4 | 2026-10-20 |
| `test_cuda_batch_invariance.py` | 25.2 | 1 | 2026-11-08 |
| `test_unit_store.py` | 22.8 | 1 | 2026-11-14 |
| `test_sound_e2e.py` | 22.3 | 2 | 2026-11-12 |
| `test_pose_locomotion.py` | 19.3 | 12 | 2026-11-13 |
| `test_trace_resolution.py` | 19.1 | 2 | 2026-10-20 |
| `test_main_replay.py` | 18.9 | 4 | 2026-10-31 |
| `test_match_replay.py` | 18.9 | 4 | 2026-11-12 |
| `test_grenade_kinds.py` | 17.3 | 2 | 2026-11-13 |
| `test_fire_vfx_ref.py` | 13.6 | 3 | 2026-10-29 |
| `test_gen_ship_vents_air.py` | 13.3 | 3 | 2026-11-12 |
| `test_pose_layer.py` | 10.9 | 5 | 2026-11-10 |
| `test_pose_posture.py` | 9.3 | 4 | 2026-11-10 |
| `test_bake_level_art.py` | 8.9 | 3 | 2026-11-06 |
| `test_bots.py` | 8.3 | 3 | 2026-11-12 |
| `test_smoke_transport_v2.py` | 8.0 | 1 | 2026-11-14 |
| `test_cuda_kick_drag2_timing.py` | 8.0 | 1 | 2026-08-30 |
| `test_throttle.py` | 7.9 | 1 | 2026-11-10 |
| `test_twin_debt.py` | 7.7 | 1 | 2026-11-07 |
| `test_gen_ship_recipe.py` | 7.6 | 3 | 2026-11-06 |
| (21 more files between 5 and 7 s, then a long tail) | | | |

## What the tables say

1. **The suite is the matches.** 62 % of the time is ~70 tests that each
   play one or more whole matches, almost all on ship_demo (the bots match:
   2721 ticks). `test_unit_schema.py` alone plays eight (563 s).
   Many are A/B pins by design (a law dormant against the match before it,
   recorded vs not), so each plays a DIFFERENT match; but several only READ
   every tick of the shipped pinned match (`on_tick` checks: the schema's
   derived constants and capacities, every position on the Q16 grid, the
   turning predictor, the zombie controller) and replay the same 2721 ticks
   each time.
2. **The time is young.** The heavy files are 1 to 4 days old, and half of
   the match time is in files an unmerged branch is changing today
   (`183-no-fire-windup`: bots_door, unit_impulse, turning_law,
   zombie_control, action_spec_v1, orders, turn_rate; `29-p64b-topology`:
   the unit-kernel lockstep). Test files older than a week and on no active
   branch hold **462 s (15 %)** of the run.
3. **The device tests are 18 %** (CUDA 387 s, window / GL 188 s) and cannot
   overlap each other, but they need almost no CPU: they could run beside the
   CPU-bound match tests.
4. **The fast part is fast.** Without the 140 tests at 2 s or more, the suite
   is ~330 s.

## The candidates, judged on the table

| # | candidate | what it saves | verdict |
|---|---|---|---|
| 2 | two tiers | WIP runs: 52 min -> ~5.5 min (`-m "not slow"`, 140 tests at >= 2 s) | **built** (this branch); the merge gate is unchanged |
| 4 | parallel (pytest-xdist) | the merge run itself: ~52 min -> ~15-20 min at 4 workers (estimate: the 590 s device group on one worker, ~2560 s of CPU tests over three), bounded by RAM (16 GB) | **wired** (the device group, longest first); needs `pytest-xdist` in the `data` env and one measured run (peak RSS per worker sets the worker count) |
| 1 | share the repeated match | the read-only plays of the pinned ship_demo match (estimate 300-500 s): one recorded tape per session, read by every `on_tick` check | worth it; in young files on active branches -- after they merge |
| 5 | trim the longest | old non-active files: ~100 s at most (physics and GL gates whose length is their property: a full burn, an exhaustive 41^4 sweep); the young match tests are candidate 1's | small; one safe cut done (the trace-currency lints read the tree once, ~18 s) |
| 3 | run what a change can affect | the second run after main moved | later: needs `pytest-testmon` or coverage contexts, and its own proof of safety before it may replace the full run |
| 6 | one suite run per patch | the second 50 min | already in the briefs since 2026-11-14; with candidate 4 it costs a third |
| new | a faster match | every match test at once (62 % of the run), and the training farm with it | measure first: one profile of a ship_demo bots match; any fix is its own patch, held by `tools/match_pin_check.py` EQUAL |

## What this branch builds

- `tools/suite_timing.py` -- the tables above from any run with `--durations`,
  and `--write-slow` for the measured slow list.
- `tests/_suite_kinds.py` -- the one classifier (kind of test; `is_device`).
- `tests/slow_tests.txt` -- generated, 140 tests at >= 2 s.
- `tests/conftest.py` -- marks `slow` from the list; under `-n N` puts every
  device test in ONE xdist group and starts with the longest tests. A plain
  run keeps its order and reads no source.
- `tests/_trace_currency_lint.py` -- the tree read once per process.

How to run:

    pytest tests -q -m "not slow"                       # the fast tier (WIP)
    pytest tests -q                                     # the merge gate, as before
    pytest tests -q -n 4 --dist loadgroup               # the merge gate in parallel (needs pytest-xdist)

The slow list goes stale as tests are added: regenerate it from a full run with
`--durations=0 --durations-min=0.5` (`tools/suite_timing.py run.txt
--write-slow`). A test not on the list is in the fast tier until then -- the
safe side for the merge gate, which runs everything anyway.

## Before / after (patch finished -> merged)

Before: one suite run is 52.5 min serial on the Home Desktop; a patch runs it
once or twice (P6.1: 1 h 40 of its 4.5 h). After: to be measured with the first
parallel run.
