"""#29 P7.0a -- the falloffs' C twins (``fixed_point.h`` ``falloff_floor_q`` /
``falloff_round_q``, bound as ``bp.fp_falloff_*``) against ``unit_fixed``.

CPU build; the same functions run on the device in tests/cuda_unit_kit_check.py
part FALLOFF.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "cpp" / "build" / "Release", ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bp = pytest.importorskip("breach_physics")
if not hasattr(bp, "fp_falloff_floor"):
    pytest.skip("breach_physics predates the falloff twins (rebuild cpp/build)",
                allow_module_level=True)

from simulation import unit_fixed  # noqa: E402

V_MAX = (1 << 31) - 1


def _want(v, d2, r2):
    if v <= 0 or d2 >= r2:
        return 0, 0
    if d2 < 0 or v > V_MAX:
        return -1, -1
    return unit_fixed.falloff_floor(v, d2, r2), unit_fixed.falloff_round(v, d2, r2)


def _cases():
    rng = np.random.default_rng(70)
    out = []
    for r in range(1, 41):                          # the deposit weights' shape
        r2 = (r * r) << 32
        for d2 in (0, 1, r2 // 4, r2 - 1, r2, r2 + 1):
            out.append((65536, d2, r2))
        for dr in range(0, r + 1, max(1, r // 6)):
            for dc in range(0, r + 1, max(1, r // 6)):
                out.append((65536, (dr * dr + dc * dc) << 32, r2))
    for _ in range(4000):                           # blast damage, Q32 distances
        r2 = int(rng.integers(1, 1 << 44))
        out.append((int(rng.integers(0, 1 << 30)), int(rng.integers(0, r2 + 2)), r2))
    for v in (1, 2, 3, V_MAX - 1, V_MAX):           # the range ends
        for r2 in (1, 2, 3, 1 << 31, (1 << 62) - 1, 1 << 62):
            for d2 in (0, 1, r2 // 2, r2 - 1):
                out.append((v, d2, r2))
    for _ in range(4000):                           # anything in range
        r2 = int(rng.integers(1, 1 << 62))
        out.append((int(rng.integers(0, V_MAX + 1)), int(rng.integers(0, r2)), r2))
    return out


def test_the_falloff_twins_equal_unit_fixed_over_the_proved_range():
    """Property: ``falloff_floor_q`` / ``falloff_round_q`` equal
    ``unit_fixed.falloff_floor`` / ``falloff_round`` for every v < 2^31 and
    0 <= d2 < r2 < 2^63 (the deposit weights' and blast damage's shapes, the
    exact radius, the range ends), and are 0 past the radius. Breaks if the
    128-bit numerator, the 128 / 64 ceil division, the ceil root or the half-
    up rounding (m // 2) differs from the Python."""
    bad = [(c, (bp.fp_falloff_floor(*c), bp.fp_falloff_round(*c)), _want(*c))
           for c in _cases()
           if (bp.fp_falloff_floor(*c), bp.fp_falloff_round(*c)) != _want(*c)]
    assert bad == [], bad[:5]


def test_outside_the_proved_range_the_twins_refuse_and_trivial_zeros_stay_zero():
    """Property: a non-trivial operand outside the proved range (v >= 2^31, a
    negative d2 inside the radius) returns -1 (a device caller faults), while
    v <= 0 or d2 >= r2 is 0 for ANY operands, as in Python. Breaks if the
    range guard is dropped or moved ahead of the trivial zeros."""
    for c in ((V_MAX + 1, 0, 5), (1 << 40, 3, 1 << 20), (5, -1, 9), (7, -(1 << 40), 1)):
        assert bp.fp_falloff_floor(*c) == -1 and bp.fp_falloff_round(*c) == -1, c
    for c in ((0, -5, 7), (-3, 1, 9), (1 << 40, 9, 9), (V_MAX + 1, 1 << 40, 3),
              (5, 7, -2)):
        assert (bp.fp_falloff_floor(*c), bp.fp_falloff_round(*c)) == (0, 0) == \
            (unit_fixed.falloff_floor(*c), unit_fixed.falloff_round(*c)), c
