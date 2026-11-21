---
paths:
  - "tests/**"
  - "tools/bench_*.py"
  - "tools/e2e_drive.py"
  - "tools/frame_profile.py"
  - "tools/match_pin_check.py"
  - "tools/suite_timing.py"
  - "tools/affected_tests.py"
  - "tools/affected_cuda.py"
  - "tools/staging_gate.py"
  - "tools/staging_machines.toml"
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

## Affected tests
- A test file runs when its dependency closure (its transitive in-repo imports, the scripts it launches in a subprocess, the native module for `cpp/`) holds a changed file. Read from the source, never a hand mark.
- It may over-select, never under-select: anything the closure cannot see (conftest, pyproject, a deleted or renamed file, an unnamed data path) runs the FULL suite. Only `docs/` and scratch output are inert, never a whole file type: tests read textures, goldens, `slow_tests.txt`, CLAUDE.md and the rules.
- Combines with `-m "not slow"`; NEVER the merge gate.
- `reach(path, candidates)` is THE per-path rule: `select` and the staging gate both call it, never a copy. `rooted(root)` reads another checkout (a staging worktree) and restores the module after.

## Affected card checks
- A card check is a test file with a test cuda_green's own `-k` collects (read through `staging_gate.cuda_green_select`, never re-typed) that launches the harness; it runs with that same `-k`. Its reach is read from the source, never hand-marked: `affected_tests.closure` for Python, the bindings that closure names (attributes, names, identifier strings, f-string patterns), and the C++ definitions those bindings run, followed to a fixed point. Python and data paths go through `affected_tests.reach`, never a copy.
- The C++ grain is a DEFINITION (function, class with its inline members, namespace-scope variable, macro, enum), never a file: utility functions live in orchestrator TUs (`tstream` in `cuda_resident_tick.cu`), so file grain selects everything. A class member is never a name of its own (a local `h` must not reach every struct with a field `h`); an out-of-line member `C::m` is reached only while `C` is live in the reach. This rests on cpp/src having no virtual dispatch: the first `virtual` makes member gating unsound and must widen it.
- FULL card set for what touches every check: `bindings.cpp`, the build files, the kits (`fixed_point.h`, `cuda_fixedpoint_device.cuh`), a header more than half the TUs include, `config.toml`, affected_tests' unseen files, a deleted or unknown file, a C++ file whose braces do not balance.
- It may over-select, never under-select; NEVER the merge gate and never a substitute for `.cuda_green` == HEAD (Class P). Whether a sharded run could satisfy Class P is Erik's ruling (`docs/affected_cuda_evidence_2026-11-21.md`: what a static reach cannot see).
- Its evidence is execution, not agreement: a tripwire at a mutation site records which checks RUN it, and every such check must be selected. A new reach rule is proven the same way.

## Staging gate
- THE way ready branches share one full suite + one cuda_green (Erik, 2026-11-21: one full run covers parallel arcs). It merges the inputs `--no-ff`, in order, onto a throwaway `staging/<name>` in its OWN worktree; it never merges to main (it PRINTS the merge commands on green), never resolves a conflict (it stops, naming each file and the two sides), and pushes nothing but its staging branch.
- The halves are cuda_green's own `-k` selection (read from `tools/cuda_green.py`, never re-typed) + `tests/test_class_p_level.py` on the card machine, and the complement + `agents/tests` on the cpu machine. The gate asserts on every run that the two partition `tests/` (no gap, no overlap); a test whose name holds `cuda` runs on the card machine.
- Merge to main only when every half printed `exit 0` in its `STAGING_SUMMARY` at the SAME commit; a half run elsewhere uses `--at <staging branch>`, never a second merge.
- Per-machine reds live only in `tools/staging_machines.toml`: exact node ids, each with its why, reported as KNOWN; never a pattern, never a way to hide a new red. A gate stage that fails without naming a test is a red.
- A build is judged by its log, its `.pyd` (non-empty PE image, not zero-filled) and `build_id()` == the staging commit, never by the script's exit (the build scripts exit 0 on a compile failure). The proof is cuda_green's one `.cuda_green` stamp in the staging worktree, never a second stamp.
- It waits while two builds or full suites already run on this PC (the Work Desktop rule of 2026-11-21, 16 GB: a full suite counts as one build); `--force` overrides it and the busy-GPU refusal.
- Reds are sorted through `affected_tests.reach` (which change reaches the test) and the path-rules hook's own `paths:` parser (`.claude/hooks/path_rules.py`), never a second parser. `staging/<name>` branches are the gate's throwaway: `--remove` deletes one only while it is at the gate's own commit.

## E2E driver
- Bug fixes start here (the real `main.py` in a hidden window with scripted keys); never a parallel harness. `--profile` is THE frame profiler (`tools/frame_profile.py`, imported only with the flag): never a second script that launches main.py.

## Benches
- Reuse the existing instrument before writing a new one.
