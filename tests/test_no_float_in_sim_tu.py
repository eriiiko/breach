"""S0 scaffold — the "no float in a sim TU" ratchet (a BASELINE, not yet a gate).

PURPOSE / END STATE
-------------------
The fixed-point migration drives the simulation solver translation units to
INTEGER arithmetic, TU by TU, so that lockstep determinism becomes BIT-IDENTICAL
across machines (today it is only same-machine bit-identical under /fp:precise).
The end state is a HARD gate: each sim solver TU contains ZERO ``float`` /
``double`` / ``/fp:fast`` once it has gone integer.

We are not there yet — the solvers are still float. So this scaffold is a
RATCHET, not an enforced zero-gate:

  * It records the CURRENT per-TU baseline counts of ``float`` / ``double`` /
    ``fp:fast`` occurrences (below, in BASELINE).
  * It PASSES as long as no count EXCEEDS its baseline — tolerating the existing
    float while preventing NEW float from creeping in.
  * It FAILS if a count goes UP (new float added to a TU that is supposed to be
    shrinking toward integer).
  * As each solver is migrated to integer, drop its baseline numbers toward 0;
    when a TU reaches 0/0/0 this becomes a true "no float here" gate for that TU.

It is a GUARD MECHANISM, deliberately simple and robust — a whole-word line
scan, NOT a C++ parser. It counts LINES that mention the token (a line with two
``float``s counts once), which is all a drift-detecting ratchet needs. Comments
that say "float" count too; that is fine and conservative (the migration removes
the comments along with the code).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_no_float_in_sim_tu.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CPP_SRC = ROOT / "cpp" / "src"

# The simulation solver translation units the migration must drive to integer.
# (Render-only TUs like raycaster.cpp and the pure-glue bindings.cpp are NOT in
# scope — they are not part of the synced lockstep state.)
SIM_TUS = (
    "atmosphere_solver.cpp",
    "smoke_dynamics.cpp",
    "fire_simulation.cpp",
    "water_solver.cpp",
    "temperature_solver.cpp",
    "physics_engine.cpp",
)

# Whole-word ``float`` / ``double`` (so "floating" / "doubled" in prose don't
# match) and the literal ``/fp:fast`` build pragma anywhere on a line.
_FLOAT_RE = re.compile(r"\bfloat\b")
_DOUBLE_RE = re.compile(r"\bdouble\b")
_FP_FAST_RE = re.compile(r"fp:fast")

# Per-TU baseline: number of LINES containing each token, recorded 2026-06-24 on
# the s0-prereq-gate branch (the code is still float — these are the numbers the
# migration will drive to zero, TU by TU). The ratchet fails if any count rises
# above its baseline here; LOWER the numbers as each solver goes integer.
BASELINE = {
    "atmosphere_solver.cpp":  {"float": 68, "double": 0,  "fp:fast": 2},
    "smoke_dynamics.cpp":     {"float": 47, "double": 1,  "fp:fast": 1},
    "fire_simulation.cpp":    {"float": 29, "double": 0,  "fp:fast": 0},
    "water_solver.cpp":       {"float": 55, "double": 0,  "fp:fast": 1},
    "temperature_solver.cpp": {"float": 2,  "double": 1,  "fp:fast": 0},
    "physics_engine.cpp":     {"float": 64, "double": 29, "fp:fast": 1},
}


def _count_tokens(path: Path) -> dict:
    """Count LINES mentioning each tracked token. Missing file -> all zero (the
    water solver could be renamed; a missing TU should not crash the guard)."""
    counts = {"float": 0, "double": 0, "fp:fast": 0}
    if not path.exists():
        return counts
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if _FLOAT_RE.search(line):
            counts["float"] += 1
        if _DOUBLE_RE.search(line):
            counts["double"] += 1
        if _FP_FAST_RE.search(line):
            counts["fp:fast"] += 1
    return counts


def current_counts() -> dict:
    """Per-TU current token counts — usable as a small report from the CLI."""
    return {tu: _count_tokens(CPP_SRC / tu) for tu in SIM_TUS}


def test_sim_tus_exist():
    """The scan must actually be looking at the solver TUs (catch a rename/move
    that would silently make the ratchet vacuous)."""
    missing = [tu for tu in SIM_TUS if not (CPP_SRC / tu).exists()]
    assert not missing, (
        f"sim TU(s) not found under {CPP_SRC} — did a solver get renamed? "
        f"Update SIM_TUS/BASELINE: {missing}")


def test_no_new_float_in_sim_tus():
    """RATCHET: no tracked token count may EXCEED its recorded baseline.

    Tolerates the existing float (the migration is mid-flight); fails the instant
    NEW float/double/fp:fast is added to a sim solver TU. As each TU goes integer,
    lower its BASELINE — when it hits 0/0/0 this is a hard 'no float here' gate."""
    regressions = []
    for tu in SIM_TUS:
        cur = _count_tokens(CPP_SRC / tu)
        base = BASELINE[tu]
        for tok in ("float", "double", "fp:fast"):
            if cur[tok] > base[tok]:
                regressions.append(
                    f"{tu}: '{tok}' rose to {cur[tok]} (baseline {base[tok]}) — "
                    f"new float crept into a sim TU; either revert it or, if this "
                    f"TU is intentionally still float, justify and bump BASELINE")
    assert not regressions, (
        "no-float ratchet tripped (NEW float in a sim solver TU):\n  "
        + "\n  ".join(regressions))


def test_baseline_is_not_stale_low():
    """If a TU's real count drops BELOW its baseline (good — migration progress),
    the baseline is stale and should be tightened so the ratchet stays sharp.

    This is a SOFT nudge: it does not fail, it just reports. (Kept as a passing
    test with an informative message via ``pytest -rA`` / -s so progress is
    visible without blocking CI.)"""
    stale = []
    for tu in SIM_TUS:
        cur = _count_tokens(CPP_SRC / tu)
        base = BASELINE[tu]
        for tok in ("float", "double", "fp:fast"):
            if cur[tok] < base[tok]:
                stale.append(f"{tu}: '{tok}' now {cur[tok]} < baseline {base[tok]} "
                             f"(tighten BASELINE to lock in the win)")
    if stale:
        print("\n[no-float ratchet] baseline can be tightened:\n  "
              + "\n  ".join(stale))
    # Intentionally always passes — this is a nudge, not a gate.
    assert True


if __name__ == "__main__":
    import json
    print(json.dumps(current_counts(), indent=2))
