"""META-GATE — every ``cuda_*_check.py`` must be referenced by some ``test_*.py``.

ADDED BY audit Patch A / A4 (2026-08-04), after the audit found TWO complete,
correct CUDA parity gates that had never once run:
``cuda_po2b_check.py`` (the shipped draw_r = 2 extended O2 draw) and
``cuda_sky_exchange_check.py`` (resident-vs-CPU gate e). Both were written,
committed, and then silently never collected, because by convention a
``cuda_*_check.py`` is not a pytest module — it is a script run in an isolated
subprocess by a thin ``test_*.py`` wrapper. Miss the wrapper and the gate
becomes decorative: it costs nothing, proves nothing, and reads in a directory
listing exactly like one that runs.

This test makes that failure mode loud instead of invisible. It is deliberately
NOT skipped without CUDA — it is pure filesystem/text analysis, and an orphan
must be catchable on a CPU-only checkout, which is precisely where nobody would
otherwise notice.

It checks WIRING, not passing: that some wrapper names each script. Whether the
gate then passes is that gate's own business.
"""
from __future__ import annotations

from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SELF = Path(__file__).resolve().name


def test_every_cuda_check_script_has_a_pytest_wrapper():
    check_scripts = sorted(p.stem for p in TESTS_DIR.glob("cuda_*_check.py"))
    assert check_scripts, (
        "found no cuda_*_check.py at all — this meta-gate has lost its subject "
        f"(looked in {TESTS_DIR}); it would pass vacuously forever"
    )

    # Exclude THIS file: it names the orphans it is reporting, and counting that
    # as a reference would let a script satisfy the gate by appearing in its own
    # failure message.
    wrappers = [p for p in TESTS_DIR.glob("test_*.py") if p.name != SELF]
    blob = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in wrappers)

    orphans = [s for s in check_scripts if s not in blob]

    assert not orphans, (
        "these CUDA parity gates exist but NOTHING references them, so "
        "`pytest tests -q` never collects them and they never run:\n"
        + "".join(f"  - tests/{s}.py\n" for s in orphans)
        + "Add a tests/test_<name>.py wrapper (see tests/test_cuda_po2b.py for "
          "the pattern: skipif not cuda_harness.cuda_available(), then assert "
          "the script's RESULT marker)."
    )
