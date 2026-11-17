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


# ---------------------------------------------------------------------------
# #29 P6b.2: the vent sweep's floor division and floor product
# ---------------------------------------------------------------------------
def test_floordiv_twin_is_pythons_floor_division_on_both_signs():
    """Property: ``bp.fp_floordiv(n, d)`` (``fixed_point.h floordiv_q``, the
    vent sweep's ``e_plenum // n_bulk``) equals Python's ``n // d`` for
    negative, zero and positive operands of either sign, int64-wide. Breaks
    if the twin truncates toward zero (C's ``/``) on a negative remainder."""
    if not hasattr(bp, "fp_floordiv"):
        pytest.skip("breach_physics predates fp_floordiv (rebuild cpp/build)")
    rng = np.random.default_rng(6202)
    cases = [(n, d) for n in (-7, -6, -1, 0, 1, 6, 7, -(1 << 62), (1 << 62), -(1 << 63) + 1)
             for d in (1, 2, 3, 7, 64, -1, -3, (1 << 40) + 7)]
    for _ in range(20000):
        d = int(rng.integers(1, 1 << int(rng.integers(1, 63))))
        d = d if rng.random() < 0.8 else -d
        cases.append((int(rng.integers(-(1 << 62), 1 << 62)), d))
    neg_rem = 0
    for n, d in cases:
        assert bp.fp_floordiv(n, d) == n // d, (n, d)
        neg_rem += (n % d != 0) and ((n < 0) != (d < 0))
    assert neg_rem > 1000      # the truncating branch was really exercised


def test_muldiv_twin_is_the_exact_floor_product_and_refuses_outside():
    """Property: ``bp.fp_muldiv_floor(a, b, c)`` (``fixed_point.h
    muldiv_floor_q``: the vent sweep's ``total * w // sum_w``, the split's
    ``holding * want // n``, the deposit's ``share * trace // n_bulk``) is
    ``(a * b) // c`` exactly, products past 2^64 included, whenever the
    quotient is below 2^63, and -1 for a negative operand, ``c <= 0`` or a
    quotient past it. Breaks if the product is formed in 64 bits, the 128 / 64
    division rounds anywhere but down, or a refusal is missed."""
    if not hasattr(bp, "fp_muldiv_floor"):
        pytest.skip("breach_physics predates fp_muldiv_floor (rebuild cpp/build)")
    rng = np.random.default_rng(6203)
    wide = 0
    for _ in range(30000):
        c = int(rng.integers(1, 1 << int(rng.integers(1, 63))))
        b = int(rng.integers(0, c + 1))            # b <= c: the sweep's shape
        a = int(rng.integers(0, 1 << int(rng.integers(1, 63))))
        assert bp.fp_muldiv_floor(a, b, c) == (a * b) // c, (a, b, c)
        wide += a * b >= 1 << 64
    assert wide > 1000         # the 128-bit product was really exercised
    for a, b, c in ((1 << 62, 1 << 62, 1), ((1 << 63) - 1, 2, 1), (5, 7, 0), (-1, 3, 4),
                    (3, -1, 4), (3, 4, -2)):
        want = (a * b) // c if (a >= 0 and b >= 0 and c > 0 and (a * b) // c < (1 << 63)) else -1
        assert bp.fp_muldiv_floor(a, b, c) == want, (a, b, c)
    assert bp.fp_muldiv_floor((1 << 63) - 1, (1 << 63) - 1, (1 << 63) - 1) == (1 << 63) - 1
