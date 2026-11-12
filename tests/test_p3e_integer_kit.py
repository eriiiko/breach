"""#29 P3e -- the match path's integer kit and its draws.

``simulation.unit_fixed``'s P3e helpers (``div_round``, ``mul_q16_round``,
``falloff_floor``, ``falloff_round``) are EXACT: the real-number result
rounded once. ``simulation.draws`` maps ONE raw generator step to each
outcome in integers.

PROPERTIES PROTECTED
  1. ``falloff_floor(v, d2, r2) == floor(v * (1 - sqrt(d2 / r2)))`` and
     ``falloff_round == floor(v * (1 - sqrt(d2 / r2)) + 1/2)``, against an
     80-digit decimal reference, over every blast geometry the game makes
     (integer radii, Q16 radii, the walked ``r2``) and random large inputs.
     Breaks if the sqrt ratio is floored instead of ceiled, the ``m // 2``
     step is off by one, or the radius edge leaks a non-zero value.
  2. ``div_round`` / ``mul_q16_round`` round half AWAY from zero (the
     ``quantize_scalar`` rule), both signs. Breaks on a floor / truncation.
  3. Each draw helper takes exactly ONE raw step of the bit generator (so
     the draw count per shot / roll / noisy tile is the float path's), and
     its range is exact: the cone offset in ``[-c, c]`` with every value
     reachable, the roll True with probability ``p / 65536``, the noise in
     ``[lo, ONE]``. Breaks if a helper draws twice, uses ``uniform`` or
     maps outside its range.
"""
from __future__ import annotations

import random
import sys
from decimal import Decimal, getcontext, ROUND_FLOOR
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import draws  # noqa: E402
from simulation.unit_fixed import (  # noqa: E402
    FP_ONE, div_round, falloff_floor, falloff_round, mul_q16_round)

getcontext().prec = 80


def _ref(v, d2, r2):
    x = Decimal(v) * (Decimal(1) - (Decimal(d2) / Decimal(r2)).sqrt())
    return x


def _floor(x):
    return int(x.to_integral_value(rounding=ROUND_FLOOR))


def _cases():
    out = []
    # integer radii (apply_explosion, apply_blast_damage, the heat source)
    for r in range(1, 10):
        for d2 in range(0, r * r + 3):
            for v in (1, 5, 60, 150, 200, 800, 800 << 16, 13 << 16, 3):
                out.append((v, d2, r * r))
    # Q16 radii (the FieldEdit DISC: radius 0.8, 3.2, 2.5, 7.0)
    for rad in (0.8, 3.2, 2.5, 7.0, 0.333):
        r_q = int(round(rad * FP_ONE))
        for d2 in range(0, 60):
            for v in (FP_ONE, 4 * FP_ONE, 1):
                out.append((v, d2 << 32, r_q * r_q))
    # the walked r2 (generic explosion: heat_r2 = 4 r_s^2, odd squares)
    for r2 in (4, 13, 16, 25, 37, 64, 144):
        for d2 in range(0, r2 + 2):
            out.append((FP_ONE, d2, r2))
    rnd = random.Random(29)
    for _ in range(3000):
        r2 = rnd.randint(1, 1 << 40)
        out.append((rnd.randint(0, 1 << 30), rnd.randint(0, r2 + 5), r2))
    return out


def test_falloff_is_the_exact_real_result():
    for v, d2, r2 in _cases():
        x = _ref(v, d2, r2) if d2 < r2 else Decimal(0)
        want_f = max(0, _floor(x))
        want_r = max(0, _floor(x + Decimal("0.5")))
        assert falloff_floor(v, d2, r2) == want_f, (v, d2, r2)
        assert falloff_round(v, d2, r2) == want_r, (v, d2, r2)


def test_falloff_edges():
    assert falloff_floor(60, 0, 4) == 60
    assert falloff_round(60, 0, 4) == 60
    assert falloff_floor(60, 4, 4) == 0          # on the radius
    assert falloff_round(60, 5, 4) == 0          # past it
    assert falloff_floor(60, 1, 4) == 30         # half way, exact
    assert falloff_round(5, 1, 4) == 3           # 2.5 -> 3 (half up)
    assert falloff_floor(5, 1, 4) == 2


def test_div_round_half_away_both_signs():
    assert div_round(5, 2) == 3 and div_round(-5, 2) == -3
    assert div_round(7, 2) == 4 and div_round(-7, 2) == -4
    assert div_round(4, 3) == 1 and div_round(5, 3) == 2
    assert div_round(5, -2) == -3
    assert mul_q16_round(3, FP_ONE // 2) == 2       # 1.5 -> 2
    assert mul_q16_round(-3, FP_ONE // 2) == -2
    rnd = random.Random(3)
    for _ in range(2000):
        n = rnd.randint(-(1 << 50), 1 << 50)
        d = rnd.randint(1, 1 << 30)
        q = Decimal(n) / Decimal(d)
        want = _floor(abs(q) + Decimal("0.5"))
        assert div_round(n, d) == (want if n >= 0 else -want)


class _Counting:
    """A generator whose bit generator counts its raw steps."""

    def __init__(self, seed):
        self._g = np.random.default_rng(seed)
        self.steps = 0
        outer = self

        class _BG:
            def random_raw(self_inner):
                outer.steps += 1
                return outer._g.bit_generator.random_raw()
        self.bit_generator = _BG()

    def uniform(self, *a, **k):              # pragma: no cover - refused
        raise AssertionError("a P3e draw must not call uniform")


def test_each_draw_is_one_raw_step():
    g = _Counting(1)
    draws.cone_offset_q16(g, 3431)
    assert g.steps == 1
    draws.cone_offset_q16(g, 0)
    assert g.steps == 2
    draws.roll_q16(g, 36044)
    assert g.steps == 3
    draws.noise_mult_q16(g, FP_ONE - 55705)
    assert g.steps == 4


def test_draw_ranges():
    g = np.random.default_rng(7)
    seen = set()
    for _ in range(4000):
        o = draws.cone_offset_q16(g, 3)
        assert -3 <= o <= 3
        seen.add(o)
    assert seen == set(range(-3, 4))
    assert draws.cone_offset_q16(g, 0) == 0
    hits = sum(draws.roll_q16(g, FP_ONE // 4) for _ in range(20000))
    assert 4500 < hits < 5500
    assert not any(draws.roll_q16(g, 0) for _ in range(200))
    assert all(draws.roll_q16(g, FP_ONE) for _ in range(200))
    lo = FP_ONE - 55705
    vals = [draws.noise_mult_q16(g, lo) for _ in range(5000)]
    assert min(vals) >= lo and max(vals) <= FP_ONE
    assert draws.noise_mult_q16(g, FP_ONE) == FP_ONE
    with pytest.raises(ValueError):
        draws.cone_offset_q16(g, -1)
