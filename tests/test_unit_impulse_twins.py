"""G15 (#183 G5p-a2) -- the unit impulse law's integer helpers equal their
``fixed_point.h`` twins (critique B, M2 / G15).

PROPERTY PROTECTED: ``unit_fixed.isqrt`` equals the kit's fixed-iteration
``sqrt_q16``, and ``div_trunc`` / ``shr_trunc`` / ``effort_step`` equal C's
truncating ``/``, the truncating shift and the round-away effort step, on a
sample that includes negatives, zero, the law's extremes and random values --
so the device port of slot 1a / 3 computes the Python law's integers. Breaks
if a Python helper floors where C truncates (Python's ``//`` and ``>>`` on a
negative), or the root changes method.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from simulation import unit_impulse  # noqa: E402
from simulation.unit_fixed import (div_trunc, effort_step, isqrt,  # noqa: E402
                                   shr_trunc)

_RNG = np.random.default_rng(15)
#: Values the law meets: velocities to V_MAX, impulses (a shove is 8.2e8
#: Q16 gram-tiles per tick), squares of velocities to 1.4e11, the extremes.
_INTS = sorted({0, 1, -1, 7, -7, 8, -8, 9, -9, 65535, -65535, 65536, -65536,
                unit_impulse.V_MAX_Q, -unit_impulse.V_MAX_Q, 819_200_000,
                -819_200_000, 2 ** 40 + 3, -(2 ** 40) - 3, 2 ** 62 - 1,
                -(2 ** 62) + 1}
               | {int(v) for v in _RNG.integers(-2 ** 45, 2 ** 45, 400)})
_DIVISORS = [1, -1, 3, -3, 753, 27000, -27000, 80_000, 100_000, 2 ** 31 - 1]


def test_g15_div_trunc_is_cs_division():
    for n in _INTS:
        for d in _DIVISORS:
            assert div_trunc(n, d) == bp.fp_div_trunc(n, d), (n, d)


def test_g15_shr_trunc_is_the_kits_truncating_shift():
    for n in _INTS:
        for k in (0, 1, 3, 16, 30):
            assert shr_trunc(n, k) == bp.fp_shr_trunc(n, k), (n, k)


def test_g15_effort_step_is_the_kits_round_away_step():
    for d in _INTS:
        assert effort_step(d) == bp.fp_effort_step(d), d


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 15, 16, 17, 2 ** 32 - 1,
                               2 ** 32, 140_000_000_000, 2 ** 61, 2 ** 62 - 1])
def test_g15_isqrt_is_sqrt_q16_at_the_edges(n):
    assert isqrt(n) == bp.fp_sqrt_q16(n)


def test_g15_isqrt_is_sqrt_q16_over_the_laws_squares():
    for v in _RNG.integers(0, 2 ** 37, 2000):
        n = int(v)
        assert isqrt(n) == bp.fp_sqrt_q16(n), n
        m = n * n                                      # an exact square
        if m < 2 ** 62:
            assert isqrt(m) == bp.fp_sqrt_q16(m) == n
