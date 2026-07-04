"""Q2-LIFT gate — the deterministic trig kit (fixed_point.h atan2_q16 / sin_q16
/ cos_q16, exposed via pybind in both builds).

WHAT IS PINNED HERE
-------------------
1. ACCURACY vs double libm, over a >= 1M-sample sweep per function family
   (full-density dense + edge cases + random):
     atan2_q16 : max error <= 9.0e-6 rad   (measured sup 7.70e-6; spec target 2e-5)
     sin_q16   : max error <= 9.0e-6       (measured sup 7.68e-6; spec target 1e-5)
     cos_q16   : max error <= 9.0e-6       (measured sup 7.68e-6; spec target 1e-5)
   All three sit at the unavoidable Q16.16 output-quantization floor
   (0.5/65536 ~= 7.63e-6): the internal Q.30 pipeline contributes < 1e-7.
   The bounds are pinned with a small margin per the spec ("tune the degree
   until met, then PIN the achieved bound").
2. EXACT symmetries (bit-equality, not tolerance): sin odd, cos even, atan2
   odd in y — these hold BY CONSTRUCTION (signs are stripped before the
   integer core and re-applied to the final q16).
3. EXACT edge cases: the documented axis/zero table (atan2(0,0) := 0, axes on
   quantize(pi)/quantize(pi/2), sin/cos at 0, q(pi/2), q(pi), q(2pi)).
4. Out-of-pinned-range behavior is DEFINED and frozen (the % wrap is total on
   any int32): two far-out inputs are pinned to their exact integer results.

PURE-INTEGER DETERMINISM (stated, satisfied by construction)
------------------------------------------------------------
The C++ bodies (cpp/src/fixed_point.h, "deterministic trig kit" section) use
ONLY int32/int64 add/sub/mul/shift/divide/modulo between the q16 input(s) and
the q16 output — no float, no double, no libm call, no compiler intrinsic, no
__int128. There is nothing a compiler flag, CRT version, or GPU architecture
can legally change, which is the entire point of the kit; this file therefore
gates ACCURACY and the OBSERVABLE integer contract (exact symmetries + pinned
edge values + frozen wrap pins), and determinism follows from the integer-only
body plus C++'s defined semantics for these ops (SAR on signed >> is
static_assert-pinned in the header).

Run:  C:/Users/steen/anaconda3/python.exe -m pytest tests/test_fixed_trig.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

import breach_physics as bp

FP_ONE = 65536
PI_Q = 205887          # round(pi   * 2^16) == the kit's quantized pi
PI_2_Q = 102944        # round(pi/2 * 2^16)
TWO_PI_Q = 411775      # round(2pi  * 2^16)
INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)

# The PINNED accuracy bounds (measured sup + ~17% margin; see module docstring).
ATAN2_BOUND_RAD = 9.0e-6
SIN_BOUND = 9.0e-6
COS_BOUND = 9.0e-6

# The pinned accuracy range for sin/cos: one wrap each side (|a| <= 2*(2pi)),
# swept at FULL density below (every representable count, +-830111 => 1.66M).
SINCOS_PIN_LIM = int((4 * math.pi + 0.1) * FP_ONE)   # 830111 counts (4pi + eps)


def test_constants_are_the_quantized_pi_family():
    """The Q16.16 pi family used by the edge-case table below is exactly the
    round-to-nearest quantization of double pi (the locked constant idiom)."""
    assert PI_Q == round(math.pi * FP_ONE)
    assert PI_2_Q == round(math.pi / 2 * FP_ONE)
    assert TWO_PI_Q == round(2 * math.pi * FP_ONE)


def test_sin_cos_accuracy_full_dense_sweep():
    """FULL-density sweep: EVERY representable Q16.16 input in the pinned range
    [-(2pi+0.1), +(2pi+0.1)] rad — 1,647,099+ samples per function — against
    double libm. Asserts the pinned max-error bound and the output range
    |result| <= FP_ONE."""
    sin_q, cos_q = bp.sin_q16, bp.cos_q16
    msin, mcos = math.sin, math.cos
    worst_s = worst_c = 0.0
    worst_s_at = worst_c_at = 0
    max_out = 0
    n = 0
    for a in range(-SINCOS_PIN_LIM, SINCOS_PIN_LIM + 1):
        ar = a / 65536.0
        qs = sin_q(a)
        qc = cos_q(a)
        m = max(abs(qs), abs(qc))
        if m > max_out:
            max_out = m
        ds = abs(qs / 65536.0 - msin(ar))
        dc = abs(qc / 65536.0 - mcos(ar))
        if ds > worst_s:
            worst_s, worst_s_at = ds, a
        if dc > worst_c:
            worst_c, worst_c_at = dc, a
        n += 1
    assert n >= 1_000_000, f"sweep too small: {n}"
    assert worst_s <= SIN_BOUND, \
        f"sin_q16 max error {worst_s:.3e} at a={worst_s_at} exceeds pinned {SIN_BOUND}"
    assert worst_c <= COS_BOUND, \
        f"cos_q16 max error {worst_c:.3e} at a={worst_c_at} exceeds pinned {COS_BOUND}"
    assert max_out <= FP_ONE, f"|sin/cos| output exceeded FP_ONE: {max_out}"


def test_sin_odd_cos_even_symmetry_exact():
    """EXACT (bit-equal) odd/even symmetry over the pinned range + beyond.
    Holds by construction: |a| is taken first, the sign re-applied last."""
    sin_q, cos_q = bp.sin_q16, bp.cos_q16
    for a in range(0, SINCOS_PIN_LIM + 1, 7):
        assert sin_q(-a) == -sin_q(a), f"sin odd symmetry broken at a={a}"
        assert cos_q(-a) == cos_q(a), f"cos even symmetry broken at a={a}"
    # beyond the pinned range too (the wrap path), incl. the int32 extremes
    for a in (SINCOS_PIN_LIM + 12345, 5_000_000, 100_000_000, INT32_MAX):
        assert sin_q(-a) == -sin_q(a)
        assert cos_q(-a) == cos_q(a)


def test_sin_cos_edge_cases_exact():
    """The pinned exact values at the quantized quadrant points."""
    assert bp.sin_q16(0) == 0
    assert bp.cos_q16(0) == FP_ONE
    assert bp.sin_q16(PI_2_Q) == FP_ONE
    assert bp.cos_q16(PI_2_Q) == 0
    assert bp.sin_q16(-PI_2_Q) == -FP_ONE
    assert bp.cos_q16(-PI_2_Q) == 0
    # sin(q(pi)) is 0: the true value at the QUANTIZED pi is sin(pi - 6.4e-6)
    # ~= +6.4e-6 ~= 0.42 counts -> rounds to 0. cos(q(pi)) == -FP_ONE.
    assert bp.sin_q16(PI_Q) == 0
    assert bp.cos_q16(PI_Q) == -FP_ONE
    assert bp.sin_q16(TWO_PI_Q) == 0
    assert bp.cos_q16(TWO_PI_Q) == FP_ONE


def test_sin_cos_out_of_range_is_defined_and_frozen():
    """Inputs beyond the pinned |a| <= 4pi range are still DEFINED (total `%`
    wrap) and deterministic. Freeze two far-out results as integer pins so any
    change to the wrap path fails loudly (these are our constants, not libm's:
    at 5215 wraps the checked-in TWO_PI_Q30's 0.26-count defect accumulates to
    ~1.3e-6 rad — still inside the pinned bound, but not contractually so)."""
    assert bp.sin_q16(INT32_MIN) == -60808
    assert bp.cos_q16(INT32_MIN) == 24441
    # and the symmetric partner of INT32_MIN+1 (|INT32_MIN| itself has no int32
    # positive twin; the int64 |.| in the kit makes INT32_MIN safe, tested above)
    assert bp.sin_q16(INT32_MAX) == -bp.sin_q16(-INT32_MAX)


def _atan2_err_rad(y: int, x: int) -> float:
    got = bp.atan2_q16(y, x) / 65536.0
    ref = math.atan2(y, x)
    d = abs(got - ref)
    # wrap-aware (defensive): -pi and +pi are the same direction
    return min(d, abs(d - 2 * math.pi))


def test_atan2_accuracy_sweep():
    """>= 1M (y, x) pairs: dense angle rings at several radii, the axis/diagonal
    specials, and 1M seeded random pairs across the full int32 magnitude range.
    Asserts the pinned max-error bound (radians) and the output range."""
    worst = 0.0
    worst_at = (0, 0)
    n = 0

    def check(y: int, x: int):
        nonlocal worst, worst_at, n
        d = _atan2_err_rad(y, x)
        n += 1
        if d > worst:
            worst, worst_at = d, (y, x)

    for radius in (3, 700, 65536, 9_999_991, INT32_MAX):
        for i in range(2000):
            th = -math.pi + (i + 0.31) * (2 * math.pi / 2000)
            y = int(round(radius * math.sin(th)))
            x = int(round(radius * math.cos(th)))
            if (y, x) != (0, 0):
                check(y, x)
    for v in (1, 2, 3, 65536, INT32_MAX - 1, INT32_MAX):
        for (y, x) in ((0, v), (0, -v), (v, 0), (-v, 0), (v, v), (v, -v),
                       (-v, v), (-v, -v), (1, v), (-1, -v), (v, 1), (-v, -1)):
            check(y, x)
    rng = np.random.default_rng(20260704)
    n_rand = 1_100_000   # headroom over 1M: the (0,0) shift-outs are skipped
    mag = rng.integers(1, 32, size=n_rand)
    ys = (rng.integers(INT32_MIN, INT32_MAX + 1, size=n_rand) >> mag).tolist()
    xs = (rng.integers(INT32_MIN, INT32_MAX + 1, size=n_rand) >> mag).tolist()
    at2 = bp.atan2_q16
    for y, x in zip(ys, xs):
        if y == 0 and x == 0:
            continue
        check(y, x)

    assert n >= 1_000_000, f"sweep too small: {n}"
    assert worst <= ATAN2_BOUND_RAD, \
        f"atan2_q16 max error {worst:.3e} rad at {worst_at} exceeds pinned {ATAN2_BOUND_RAD}"


def test_atan2_odd_in_y_symmetry_exact():
    """EXACT: atan2_q16(-y, x) == -atan2_q16(y, x) for y != 0 (by construction;
    y == 0 is excluded — the branch cut maps 0/x<0 to +PI_Q, as double atan2
    maps -0.0 vs +0.0, which int32 cannot distinguish)."""
    rng = np.random.default_rng(4)
    ys = (rng.integers(1, INT32_MAX, size=60_000)).tolist()
    xs = (rng.integers(INT32_MIN + 1, INT32_MAX, size=60_000)).tolist()
    at2 = bp.atan2_q16
    for y, x in zip(ys, xs):
        assert at2(-y, x) == -at2(y, x)


def test_atan2_edge_cases_exact():
    """The pinned axis/zero/diagonal table (documented in fixed_point.h)."""
    assert bp.atan2_q16(0, 0) == 0            # DEFINED as 0
    assert bp.atan2_q16(0, 5) == 0
    assert bp.atan2_q16(0, -5) == PI_Q        # the (-pi, pi] closed end
    assert bp.atan2_q16(7, 0) == PI_2_Q
    assert bp.atan2_q16(-7, 0) == -PI_2_Q
    assert bp.atan2_q16(1, INT32_MAX) == 0    # sub-half-count angle -> 0
    assert bp.atan2_q16(-1, -INT32_MAX) == -PI_Q
    # int32 extreme corner: |INT32_MIN| handled on int64 (no abs() UB)
    assert bp.atan2_q16(INT32_MIN, INT32_MIN) == -round(3 * math.pi / 4 * FP_ONE)
    # a real ratio lands on the correctly-rounded quantization of libm's answer
    assert bp.atan2_q16(3 * FP_ONE, 4 * FP_ONE) == round(math.atan2(3, 4) * FP_ONE)
    # quadrant signs
    assert bp.atan2_q16(5, 5) > 0 and bp.atan2_q16(5, -5) > 0
    assert bp.atan2_q16(-5, 5) < 0 and bp.atan2_q16(-5, -5) < 0
    # range: |result| <= PI_Q for a spread of inputs
    rng = np.random.default_rng(11)
    for y, x in zip(rng.integers(INT32_MIN, INT32_MAX + 1, size=20_000).tolist(),
                    rng.integers(INT32_MIN, INT32_MAX + 1, size=20_000).tolist()):
        assert abs(bp.atan2_q16(y, x)) <= PI_Q


def test_scale_invariance_of_the_ratio():
    """atan2 depends only on the ratio + signs: scaling both args by a power of
    two (exact in the integer domain) must not move the result by more than the
    re-rounding of the internal divide (<= 1 output count)."""
    rng = np.random.default_rng(9)
    for _ in range(20_000):
        y = int(rng.integers(-30_000, 30_000))
        x = int(rng.integers(-30_000, 30_000))
        if y == 0 and x == 0:
            continue
        a = bp.atan2_q16(y, x)
        b = bp.atan2_q16(y * 4096, x * 4096)
        assert abs(a - b) <= 1, (y, x, a, b)
