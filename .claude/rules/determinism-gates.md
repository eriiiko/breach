---
paths:
  - "tests/**"
  - "tools/bench_*.py"
  - "tools/e2e_drive.py"
  - "tools/frame_profile.py"
  - "tools/match_pin_check.py"
  - "tools/suite_timing.py"
  - "tools/affected_tests.py"
---

# Tests, gates, benches

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say. Run
the suite as `pytest tests -q` with this machine's breach env (`docs/dev_setup.md`).

## Field digest
- A membership or dtype change, or a change of a field's integer currency (`trace_fine_bits`), is a version bump + all goldens regenerated in the same commit.
- A plane that is zero nearly always is a `DIGEST_SECTIONS` row instead (hashed while non-zero behind its own preamble, absence-transparent), so no version bump and no golden moves.

## Ingress lint + float ratchet
- `SIM_HEADERS` (`tests/test_no_float_in_sim_tu.py`) holds the sim-law headers a TU scan cannot see at 0/0/0: a law compiled inline from a header joins it with its patch.

## Test conventions
- `test_*` is collected, `_*` is a harness. Tests assert properties, never snapshots, while systems land; feel regressions compare in real dequantized units.
- GL checks run as a `gl_*_check.py` script in a subprocess with a hidden window + a `test_*` wrapper (the `cuda_*` convention), never a GL context inside pytest.
- A fixture never writes a gas cell's `temperature` directly (Gas temperature is a mirror: it goes vacuous) and starts a fire by delivering heat (Starting a fire: a flame alone measures nothing).

## Suite timing
- `tests/slow_tests.txt` is GENERATED (`tools/suite_timing.py --write-slow` over a full run's `--durations=0` output); never hand-edit it and never hand-mark a test `slow`. A test missing from the list is fast until the next measurement.
- `-m "not slow"` is the tier for work in progress and NEVER the merge gate: the full suite, all determinism gates, stays it.
- Under pytest-xdist (`-n N --dist loadgroup`) every test that drives the GPU or opens a window joins the one `device` group through `tests/_suite_kinds.py::is_device`, so no two run at once. A new kind of device launch is a pattern there, never a hand mark on a test. A plain run keeps collection order and reads no source.

## Device check plays
- A device check's host references (`cuda_obs_check._PLAYED`, `cuda_conductor_batch_check._HOST_INTENTS`) are played ONCE per subprocess and shared by every part that only READS them (the reference, its taps, its per-decision arrays). A part that changes the match in any way (a perturbation, a break switch, a reset, another spec, seed or level) plays its own; a part never writes to a cached play (the obs check makes its arrays read-only).
- Every part is judged alone (`_check_parts.run_parts`): its own FAILS, the check's per-run tallies cleared before it (a judge counts only the gate runs of its own part, never another part's), a crash failing that part only. A new per-run tally joins the check's reset list in the same patch.
- The wrappers run ONE subprocess per pytest session with the selected tests' parts (`_check_parts.shared_run`, its `run_cuda_script` named in the wrapper so `_suite_kinds.is_device` sees it); a new test is a `PARTS_OF` row. Under xdist the device group keeps them on one worker. The deliberate breaks are never in that run; by hand `main([...])` runs as before.
- No check assumes a pinned match's length, ending tick or time limit: it asks for the property (an ending by elimination, a win, envs ending apart). *Incident: #217 moves the pinned bots match from 1349 ticks to about 741.*

## Affected tests
- A test file runs when its dependency closure (its transitive in-repo imports, the scripts it launches in a subprocess, the native module for `cpp/`) holds a changed file. Read from the source, never a hand mark.
- It may over-select, never under-select: anything the closure cannot see (conftest, pyproject, a deleted or renamed file, an unnamed data path) runs the FULL suite. Only `docs/` and scratch output are inert, never a whole file type: tests read textures, goldens, `slow_tests.txt`, CLAUDE.md and the rules.
- Combines with `-m "not slow"`; NEVER the merge gate.

## E2E driver
- Bug fixes start here (the real `main.py` in a hidden window with scripted keys); never a parallel harness. `--profile` is THE frame profiler (`tools/frame_profile.py`, imported only with the flag): never a second script that launches main.py.

## Benches
- Reuse the existing instrument before writing a new one.
