"""#106 P6b -- the kit's Python twins (src/simulation/fixedpoint_twins.py) are
bit-exact copies of cpp/src/fixed_point.h, value for value.

The arm chain's integer reference (docs/octopus_integer_reference/chain_ref_q.py)
is the executable specification the C++ solver is held to at tol 0; it can only
be that if every kit function it calls computes what the C++ computes. Each test
compares a twin to the C++ through the ``breach_physics`` bindings over a dense
seeded sample plus the adversarial edges (0, +-1, +-FP_ONE, the int32 / int64
extremes, the atan2 octant boundaries, denominators 1..3 and near 2^31).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_fixedpoint_twins.py -q
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bp = pytest.importorskip("breach_physics")
from simulation import fixedpoint_twins as T  # noqa: E402

I32 = (T.INT32_MIN, T.INT32_MAX)
ONE = T.FP_ONE
EDGES32 = sorted({0, 1, -1, 2, -2, 3, -3, ONE, -ONE, ONE - 1, -ONE + 1, ONE + 1,
                  T.INT32_MAX, T.INT32_MIN, T.INT32_MAX - 1, T.INT32_MIN + 1,
                  1 << 30, -(1 << 30), (1 << 30) - 1, 46340, 46341, -46341,
                  T.PI_Q16, -T.PI_Q16, T.PI_Q16 // 2, 102944, 51472, 411774})


def _rng(seed):
    return random.Random(seed)


def _i32_sample(n, seed):
    r = _rng(seed)
    out = list(EDGES32)
    for _ in range(n):
        k = r.randrange(32)
        v = r.randrange(1 << k) if k else 0
        out.append(-v if r.random() < 0.5 else v)
    return out


def test_mul_wide_is_the_exact_product():
    """Property: mul_wide twin == fixed_point.h mul_wide on every int32 pair.
    Breaks if the twin narrows or wraps an int32 x int32 product."""
    xs = _i32_sample(400, 1)
    for a in xs[::3]:
        for b in xs[::7]:
            assert T.mul_wide(a, b) == bp.fp_mul_wide(a, b), (a, b)


def test_mul128_shr_matches_including_the_int64_wrap():
    """Property: mul128_shr twin == the C++ (128-bit floor shift, int64 cast)
    for every shift 0..63, signed operands up to the int64 extremes, including
    results that wrap. Breaks if the twin floors differently (toward 0) or
    hides the cast's wrap."""
    r = _rng(2)
    vals = [0, 1, -1, T.INT64_MAX, T.INT64_MIN + 1, 1 << 62, -(1 << 62), 3, -3]
    for _ in range(300):
        k = r.randrange(64)
        v = r.randrange(1 << k) if k else 0
        vals.append(-v if r.random() < 0.5 else v)
    for i, a in enumerate(vals):
        for b in vals[i % 7::9]:
            for s in (0, 1, 15, 16, 24, 30, 32, 40, 48, 63):
                assert T.mul128_shr(a, b, s) == bp.fp_mul128_shr(a, b, s), (a, b, s)


def test_floordiv_q_floors_on_every_sign():
    """Property: floordiv_q twin == the C++ (toward -inf) for all sign pairs,
    denominators 1..3, near 2^31 and the int64 range. Breaks if the twin
    truncates toward zero on a negative quotient."""
    r = _rng(3)
    ds = [1, 2, 3, -1, -2, -3, (1 << 31) - 1, 1 << 31, (1 << 31) + 1, -(1 << 31),
          147456, 384, 1 << 40]
    ns = [0, 1, -1, 7, -7, T.INT64_MAX, T.INT64_MIN + 1, (1 << 62) + 5, -(1 << 62) - 5]
    for _ in range(300):
        k = r.randrange(63)
        v = r.randrange(1 << k) if k else 0
        ns.append(-v if r.random() < 0.5 else v)
        ds.append(r.choice((1, -1)) * (r.randrange(1, 1 << r.randrange(1, 40))))
    for n in ns:
        for d in ds[::3]:
            assert T.floordiv_q(n, d) == bp.fp_floordiv_q(n, d), (n, d)
            assert T.floordiv_q(n, d) == n // d


def test_reciprocal_q16_matches_and_sweep_ref_q_is_the_same_function():
    """Property: reciprocal_q16 twin == the C++ Newton reciprocal over a dense
    sweep of the positive int32 range and its guards (<= 0, 1, 2), and
    sweep_ref_q's own copy (the ray-engine reference) is the same function --
    one implementation's values. Breaks if either copy drifts from the kit."""
    sys.path.insert(0, str(ROOT / "docs" / "ray_engine_v2_scheme_study_2026-09-13"))
    import sweep_ref_q as R
    xs = [v for v in _i32_sample(3000, 4)] + list(range(-3, 70000, 97))
    for d in xs:
        got = T.reciprocal_q16(d)
        assert got == bp.fp_reciprocal_q16(d), d
        assert got == R.reciprocal_q16(d), d
    for n in (0, 5, -5, 1 << 40):
        for d in (1, 3, 147456):
            assert T.floordiv_q(n, d) == R.floordiv_q(n, d)
            assert T.mul128_shr(n, d, 16) == R.mul128_shr(n, d, 16)


def test_sqrt_q16_three_ways_agree():
    """Property: the C++ digit recurrence, its literal Python transcription and
    the math.isqrt form the reference calls return the same floor(sqrt) on
    every int64 radicand class: <= 0, perfect squares and their neighbours,
    the int64 top (the INT32_MAX clamp). Breaks if the fast form rounds instead
    of flooring or loses the clamp."""
    r = _rng(5)
    xs = [-5, -1, 0, 1, 2, 3, 4, 15, 16, 17, T.INT64_MAX, (1 << 62), (1 << 62) - 1,
          (1 << 62) + 1, ((1 << 31) - 1) ** 2, (1 << 31) ** 2 - 1, (1 << 31) ** 2]
    for _ in range(2000):
        k = r.randrange(1, 63)
        v = r.randrange(1 << k)
        xs += [v, v * v if v < (1 << 31) else v, (v * v - 1) if 0 < v < (1 << 31) else v]
    for x in xs:
        if x > T.INT64_MAX:
            continue
        c = bp.fp_sqrt_q16(x)
        assert T.sqrt_q16_digits(x) == c, x
        assert T.sqrt_q16(x) == c, x


def test_trig_twins_match_on_dense_and_edge_arguments():
    """Property: sin_q16 / cos_q16 twins == the C++ over every 7th Q16 count of
    [-4 pi, 4 pi] plus the int32 extremes and the quadrant edges; atan2_q16 ==
    the C++ over a dense grid, the axes, the diagonals (|y| == |x|, the octant
    boundary), off-by-one around them, and the int32 extremes. Breaks if a
    Horner stage, the quadrant split, the one rounded divide or the final
    round-half-away narrow is changed in either copy."""
    for a in list(range(-823550, 823551, 7)) + EDGES32:
        assert T.sin_q16(a) == bp.sin_q16(a), a
        assert T.cos_q16(a) == bp.cos_q16(a), a
    xs = _i32_sample(250, 6)
    for y in xs:
        for x in xs[::5]:
            assert T.atan2_q16(y, x) == bp.atan2_q16(y, x), (y, x)
    for m in (1, 2, 3, 1000, 46341, ONE, 1 << 29, T.INT32_MAX):
        for dy in (-1, 0, 1):
            for sy in (1, -1):
                for sx in (1, -1):
                    y, x = sy * m + dy, sx * m
                    if I32[0] <= y <= I32[1]:
                        assert T.atan2_q16(y, x) == bp.atan2_q16(y, x), (y, x)
                        assert T.atan2_q16(y, 0) == bp.atan2_q16(y, 0)
                        assert T.atan2_q16(0, x) == bp.atan2_q16(0, x)


def test_the_chain_rounding_kit_is_the_kits_rounding_generalised():
    """Property: the chain's round-half-away kit agrees with the kit functions
    it generalises -- shr_round_away(x, 16) == narrow_round_signed(x) and
    div_round_away(n, d) == mean_round(n, d) wherever the C++ result fits its
    q16 cast -- and mul_shr_away equals its stated C++ recipe through
    mul128_shr. All three are odd-symmetric. Breaks if a rounding becomes
    floor / toward-zero, or loses its symmetry (the "no slide" rule, design
    6.3)."""
    r = _rng(7)
    for _ in range(4000):
        k = r.randrange(48)
        x = r.randrange(1 << k) if k else 0
        x = -x if r.random() < 0.5 else x
        if abs(x) < (1 << 46):
            assert T.shr_round_away(x, 16) == bp.fp_narrow_round_signed(x), x
        for s in (0, 1, 10, 16, 24, 30):
            assert T.shr_round_away(-x, s) == -T.shr_round_away(x, s)
        d = r.randrange(1, 1 << r.randrange(1, 31))
        if abs(x) < (1 << 40):
            want = bp.fp_mean_round(x, d)
            got = T.div_round_away(x, d)
            if T.INT32_MIN <= got <= T.INT32_MAX:
                assert got == want, (x, d)
        assert T.div_round_away(-x, d) == -T.div_round_away(x, d)
        a = r.randrange(1 << 62) * r.choice((1, -1))
        b = r.randrange(1 << r.randrange(1, 62)) * r.choice((1, -1))
        s = r.randrange(1, 63)
        q = bp.fp_mul128_shr(abs(a), abs(b), s - 1)
        recipe = (q + 1) >> 1
        recipe = -recipe if (a < 0) != (b < 0) else recipe
        got = T.mul_shr_away(a, b, s)
        if T.INT64_MIN < got < T.INT64_MAX and abs(a * b) >> (s - 1) < T.INT64_MAX:
            assert got == recipe, (a, b, s)
        assert T.mul_shr_away(-a, b, s) == -got
    # the exact halves round AWAY from zero
    assert T.shr_round_away(3, 1) == 2 and T.shr_round_away(-3, 1) == -2
    assert T.div_round_away(5, 2) == 3 and T.div_round_away(-5, 2) == -3
    assert T.mul_shr_away(3, 1, 1) == 2 and T.mul_shr_away(-3, 1, 1) == -2


# ---------------------------------------------------------------------------
# #106 P6c: the chain's round-half-away kit is IN fixed_point.h now
# ---------------------------------------------------------------------------
#: Magnitudes the arm chain reaches (the reference's CHECK mode measures every
#: site <= 60 bits at the 3.39 m row) plus the int64 extremes each recipe's
#: own bound admits.
_CHAIN_EDGES = sorted({0, 1, -1, 2, -2, 3, -3, 127, -127, 128, -128, 129, -129, 255, 256, -256,
                       (1 << 30) - 1, 1 << 30, -(1 << 30), (1 << 31) - 1, -(1 << 31),
                       (1 << 40) + (1 << 7), -(1 << 40) - (1 << 7), (1 << 59) + 12345,
                       -(1 << 59) - 12345, (1 << 61), -(1 << 61), (1 << 62) - 1, -(1 << 62) + 1})


def _chain_sample(n, seed):
    r = _rng(seed)
    out = list(_CHAIN_EDGES)
    for _ in range(n):
        k = r.randrange(63)
        v = r.randrange(1 << k) if k else 0
        out.append(-v if r.random() < 0.5 else v)
    return out


def test_shr_round_away_is_the_cpp_kit():
    """Property: shr_round_away twin == fixed_point.h shr_round_away for every
    shift the chain uses (and 0..40), negative values, exact halves and the
    int64 range the recipe admits (|x| + 2^(s-1) < 2^63). Breaks if either side
    rounds a half toward +inf / -inf / zero, or loses odd symmetry."""
    for x in _chain_sample(3000, 21):
        for s in (0, 1, 8, 10, 14, 16, 24, 26, 30, 32, 40):
            if abs(x) + (1 << max(s - 1, 0)) >= T.INT64_MAX:
                continue
            assert T.shr_round_away(x, s) == bp.fp_shr_round_away(x, s), (x, s)
    for s in (1, 8, 16):
        h = 1 << (s - 1)
        for k in (-5, -1, 0, 1, 5):
            x = (k << s) + h            # an exact half
            assert bp.fp_shr_round_away(x, s) == T.shr_round_away(x, s)
            assert bp.fp_shr_round_away(-x, s) == -bp.fp_shr_round_away(x, s)


def test_div_round_away_is_the_cpp_kit():
    """Property: div_round_away twin == the C++ for negative and positive
    numerators, exact halves, divisors 1..3, the chain's divisors (rates
    ~2^17, Q30 units, W up to 2^51) and int64-range numerators; a divisor
    <= 0 is refused by both. Breaks if the C++ truncates toward zero on a
    negative numerator or rounds a half toward +inf."""
    r = _rng(22)
    ds = [1, 2, 3, 4, 7, 384, 147456, (1 << 30), (1 << 30) + 1, (1 << 51) + 3, T.INT64_MAX // 2]
    for _ in range(200):
        ds.append(r.randrange(1, 1 << r.randrange(1, 62)))
    for n in _chain_sample(400, 23):
        for d in ds[::3]:
            if abs(n) + d // 2 >= T.INT64_MAX:
                continue
            assert T.div_round_away(n, d) == bp.fp_div_round_away(n, d), (n, d)
    assert bp.fp_div_round_away(5, 2) == 3 and bp.fp_div_round_away(-5, 2) == -3
    assert bp.fp_div_round_away(-7, 2) == -T.div_round_away(7, 2)
    with pytest.raises(ValueError):
        bp.fp_div_round_away(1, 0)


def test_mul_shr_away_is_the_cpp_kit():
    """Property: mul_shr_away twin == the C++ (the 128-bit product through
    mul128_shr, no 128-bit add) at every shift the chain uses, both signs,
    products far past 2^63 (w Q32 x g^2, the drag's c |v|) whose rounded
    result fits int64. Breaks if the C++ floors the magnitude differently or
    forms the product in 64 bits."""
    xs = _chain_sample(600, 24)
    for i, a in enumerate(xs):
        for b in xs[i % 5::17]:
            for s in (1, 16, 24, 30, 32, 40, 48, 54):
                want = T.mul_shr_away(a, b, s)
                if not (T.INT64_MIN < want < T.INT64_MAX) or (abs(a * b) >> (s - 1)) >= T.INT64_MAX:
                    continue
                assert want == bp.fp_mul_shr_away(a, b, s), (a, b, s)
    assert bp.fp_mul_shr_away(3, 1, 1) == 2 and bp.fp_mul_shr_away(-3, 1, 1) == -2
    with pytest.raises(ValueError):
        bp.fp_mul_shr_away(1, 1, 0)


def test_the_chain_currency_is_one_number_on_both_sides():
    """Property: fixed_point.h's CHAIN_FINE_BITS / CHAIN_SHIFT equal
    creature_fixed's (the Python twin every chain conversion goes through).
    Breaks if either side changes the chain currency alone."""
    from simulation import creature_fixed as CF
    assert bp.CHAIN_FINE_BITS == CF.CHAIN_FINE_BITS == 8
    assert bp.CHAIN_SHIFT == CF.CHAIN_SHIFT == 24
