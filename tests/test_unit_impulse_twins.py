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


# ---------------------------------------------------------------------------
# #29 P6.2: the device unit kernels' kit twins (fixed_point.h isqrt_q /
# isqrt_u64_q / isqrt_ceil_q / div_round_q / mul_q16_round_q).
# ---------------------------------------------------------------------------
from simulation.unit_fixed import div_round, isqrt_ceil, mul_q16_round  # noqa: E402

_WIDE = sorted({0, 1, 2, 3, 15, 16, 17, 2 ** 31 - 1, 2 ** 31, 2 ** 32 - 1,
                2 ** 32, 2 ** 32 + 1, 2 ** 52 + 7, 2 ** 61, 2 ** 62 - 1, 2 ** 62,
                2 ** 63 - 2, 2 ** 63 - 1}
               | {int(v) for v in _RNG.integers(0, 2 ** 63 - 1, 600,
                                                dtype=np.int64)})


def test_p62_isqrt_is_exact_over_the_whole_int64_range():
    """Property: the kit's ``isqrt_q`` is ``unit_fixed.isqrt`` (floor of the
    root) for every non-negative int64 -- the radicands the kernels form
    reach 2^63 (a squared 2-vector of int32 components) -- and 0 for a
    negative one; ``isqrt_ceil_q`` is ``isqrt_ceil``. Breaks if the root
    keeps ``sqrt_q16``'s int32 clamp or loses a bit at the top."""
    for n in _WIDE + [-1, -(2 ** 40)]:
        assert bp.fp_isqrt(n) == isqrt(n), n
        assert bp.fp_isqrt_ceil(n) == isqrt_ceil(n), n
        if n > 0:
            r = isqrt(n)
            if r * r <= 2 ** 63 - 1:
                assert bp.fp_isqrt(r * r) == r


def test_p62_isqrt_u64_is_exact_past_int64():
    """Property: ``isqrt_u64_q`` is the floor root of ANY uint64 (the wind's
    squared speed of two int32 components reaches 2^63). Breaks if the
    digit recurrence starts below the top bit pair."""
    for n in _WIDE + [2 ** 63, 2 ** 63 + 5, 2 ** 64 - 1,
                      (2 ** 32 - 1) ** 2, (2 ** 32 - 1) ** 2 - 1]:
        assert bp.fp_isqrt_u64(n) == isqrt(n), n


def test_p62_div_round_is_half_away_from_zero():
    """Property: ``div_round_q`` is ``unit_fixed.div_round`` -- n / d rounded
    half AWAY from zero -- for every sign, the exact halves and the int64
    extremes (wherever the result fits int64). Breaks if a half rounds toward
    zero or up, or the magnitude overflows."""
    nums = _INTS + [2 ** 63 - 1, -(2 ** 63) + 1, -(2 ** 63), 5, -5, 7, -7, 3, -3]
    for n in nums:
        for d in _DIVISORS + [2, -2, 65536, -65536, 2 ** 62, 2 ** 63 - 1]:
            want = div_round(n, d)
            if -(2 ** 63) <= want < 2 ** 63:     # the result fits int64
                assert bp.fp_div_round(n, d) == want, (n, d)


def test_p62_mul_q16_round_is_the_exact_product_rounded_once():
    """Property: ``mul_q16_round_q`` is ``unit_fixed.mul_q16_round`` -- the
    hook accessor's one rounding -- wherever the result fits int64, its
    product formed in 128 bits (a class value of 2^43 times a hook of
    2^19). Breaks if the product is formed in 64 bits or a half rounds the
    wrong way."""
    hooks = [1, 8192, 32767, 32768, 32769, 65535, 65536, 65537, 98304,
             524287, 524288, -32768, -65536]
    vals = _INTS + [2 ** 43, -(2 ** 43), 2 ** 46 + 1, 32768, -32768, 98304]
    for a in vals:
        for b in hooks:
            want = mul_q16_round(a, b)
            if abs(want) < 2 ** 62:
                assert bp.fp_mul_q16_round(a, b) == want, (a, b)


# ---------------------------------------------------------------------------
# #29 P6.2m: the slot-3 kernel's kit twin (fixed_point.h half_up_q).
# ---------------------------------------------------------------------------
from simulation.movement import half_up  # noqa: E402


def test_p62m_half_up_is_movements_floor_rounding():
    """Property: ``half_up_q`` is ``movement.half_up`` -- ``(num + den // 2)
    // den`` with Python's FLOOR divisions -- for every sign of ``num``, odd
    and even ``den > 0``, the exact halves and the magnitudes the slot-3
    kernel forms (a Q16 cost times 64 tiles times 1000 times ONE, < 2^58).
    Breaks if a division truncates toward zero where Python floors (a
    negative numerator), or the half is taken as ``(den + 1) // 2``."""
    nums = _INTS + [2 ** 58, -(2 ** 58), 5, -5, 6, -6, 7, -7, 1, -1, 0]
    dens = [1, 2, 3, 4, 5, 7, 1000, 65536, 65537, 3 * 2 ** 40 + 1, 2 ** 58]
    for n in nums:
        for d in dens:
            if abs(n) + d < 2 ** 62:
                assert bp.fp_half_up(n, d) == half_up(n, d), (n, d)
    for d in (2, 4, 1000):                  # the exact halves round UP
        for q in (-3, 0, 3):
            n = q * d + d // 2
            assert bp.fp_half_up(n, d) == half_up(n, d) == q + 1, (n, d)


# ---------------------------------------------------------------------------
# #29 P6.4a: unit_fixed.isqrt past uint64 (fixed_point.h isqrt_u128_q; the
# shove's lever term squares an impulse of up to 2^35).
# ---------------------------------------------------------------------------
def test_p64a_isqrt_u128_is_exact_over_128_bits():
    """Property: ``isqrt_u128_q(hi, lo)`` is ``unit_fixed.isqrt`` (the floor
    root) of ``hi * 2^64 + lo`` for ANY 128-bit radicand: perfect squares and
    their neighbours at every bit length (the off-by-one edges a root gets
    wrong), the shove's sums of two squared impulses up to 2^35, both ends.
    Breaks if a trial square is taken in 64 bits, the compare is strict, or
    the bit walk starts below bit 63."""
    rng = np.random.default_rng(6404)
    vals = [0, 1, 2, 3, 2 ** 64 - 1, 2 ** 64, 2 ** 64 + 1, 2 ** 128 - 1]
    for k in range(1, 64):
        r = 2 ** k
        vals += [r * r - 1, r * r, r * r + 1, (r - 1) * (r - 1)]
    for _ in range(400):
        jx = int(rng.integers(-(2 ** 35), 2 ** 35))
        jy = int(rng.integers(-(2 ** 35), 2 ** 35))
        vals.append(jx * jx + jy * jy)
        vals.append(int(rng.integers(0, 2 ** 62)) * int(rng.integers(0, 2 ** 62)))
    for n in vals:
        assert bp.fp_isqrt_u128(n >> 64, n & (2 ** 64 - 1)) == isqrt(n), n
