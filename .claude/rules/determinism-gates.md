---
paths:
  - "tests/**"
  - "tools/bench_*.py"
  - "tools/e2e_drive.py"
  - "tools/frame_profile.py"
  - "tools/match_pin_check.py"
---

# Tests, gates, benches

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say. Run
the suite as `pytest tests -q` with this machine's breach env (`docs/dev_setup.md`).

## Field digest
- A membership or dtype change, or a change of a field's integer currency (`trace_fine_bits`), is a version bump + all goldens regenerated in the same commit.
- A plane that is zero nearly always is a `DIGEST_SECTIONS` row instead (hashed while non-zero behind its own preamble, absence-transparent), so no version bump and no golden moves.

## Test conventions
- `test_*` is collected, `_*` is a harness. Tests assert properties, never snapshots, while systems land; feel regressions compare in real dequantized units.
- GL checks run as a `gl_*_check.py` script in a subprocess with a hidden window + a `test_*` wrapper (the `cuda_*` convention), never a GL context inside pytest.
- A fixture never writes a gas cell's `temperature` directly (Gas temperature is a mirror: it goes vacuous) and starts a fire by delivering heat (Starting a fire: a flame alone measures nothing).

## E2E driver
- Bug fixes start here (the real `main.py` in a hidden window with scripted keys); never a parallel harness. `--profile` is THE frame profiler (`tools/frame_profile.py`, imported only with the flag): never a second script that launches main.py.

## Benches
- Reuse the existing instrument before writing a new one.
