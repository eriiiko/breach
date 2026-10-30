"""Bit-exact Python twins of the fixed-point kit (``cpp/src/fixed_point.h``)
that the ARM CHAIN's integer reference uses (#106 P6b).

Each twin is a transcription of its C++ function ALGORITHM FOR ALGORITHM -- the
same polynomial, the same stages, the same rounding at each stage -- never a
re-derivation from the mathematics. ``tests/test_fixedpoint_twins.py`` holds
each one equal to the C++ (through the ``fp_*`` / trig bindings of
``breach_physics``) over a dense + adversarial sample.

Two kinds of function live here:

* **The kit's own functions** (``mul_wide``, ``mul128_shr``, ``floordiv_q``,
  ``reciprocal_q16``, ``sqrt_q16``, ``atan2_q16``, ``sin_q16``, ``cos_q16``,
  ``narrow_round_signed``, ``mean_round``): twins of code that exists in C++.
  ``sqrt_q16_digits`` is the literal 32-trip digit recurrence; ``sqrt_q16`` is
  the same function (the floor of a square root is unique) computed with
  ``math.isqrt`` for speed -- the test holds all three equal.
* **The chain's rounding kit** (``shr_round_away``, ``div_round_away``,
  ``mul_shr_away``): the round-half-AWAY-FROM-ZERO shift / divide / 128-bit
  product the arm chain's "no slide" rule needs (design 6.3). The C++ kit has
  this rounding only at fixed shifts (``narrow_round_signed`` at 16,
  ``q30_to_q16_round`` at 14) and as a divide (``mean_round``); these are the
  general forms, held equal to those at the shifts they share. P6c adds them
  to ``fixed_point.h`` -- each docstring states the C++ recipe that must
  reproduce it (``mul_shr_away`` without a 128-bit add, through
  ``mul128_shr``).

Integers only: no float anywhere. The results are Python ints; the twins of
int64-returning C++ functions wrap to int64 exactly as the C++ cast does
(``wrap_i64``), so an overflow the C++ would commit is visible, never hidden.

Import-light (stdlib only): the reference, tests and tools read it.
"""
from __future__ import annotations

import math

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

FP_SHIFT = 16
FP_ONE = 1 << FP_SHIFT


def wrap_i64(v: int) -> int:
    """Two's-complement wrap to int64 (what a C++ ``(int64_t)`` cast of a
    wider value does)."""
    v &= (1 << 64) - 1
    return v - (1 << 64) if v >= (1 << 63) else v


def wrap_i32(v: int) -> int:
    """Two's-complement wrap to int32 (a C++ ``(int32_t)`` / ``(q16)`` cast)."""
    v &= (1 << 32) - 1
    return v - (1 << 32) if v >= (1 << 31) else v


def _trunc_div(n: int, d: int) -> int:
    """C++ ``/`` on integers: the quotient rounded toward ZERO."""
    q = abs(n) // abs(d)
    return q if (n >= 0) == (d >= 0) else -q


def _trunc_mod(n: int, d: int) -> int:
    """C++ ``%``: the remainder with the sign of the dividend."""
    return n - _trunc_div(n, d) * d


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------
def mul_wide(a: int, b: int) -> int:
    """fixed_point.h ``mul_wide``: ``(int64)a * (int64)b`` of two int32 --
    exact, it cannot overflow."""
    return a * b


def mul128_shr(a: int, b: int, shift: int) -> int:
    """fixed_point.h ``mul128_shr``: the full 128-bit signed product,
    arithmetic-shifted right (floor toward -inf), cast to int64. ``shift == 0``
    is the plain product's low 64 bits (the P-E3 special case)."""
    return wrap_i64((a * b) >> shift)


# ---------------------------------------------------------------------------
# Divisions
# ---------------------------------------------------------------------------
def floordiv_q(n: int, d: int) -> int:
    """fixed_point.h ``floordiv_q``: C++ ``n / d`` (toward zero) corrected by
    one where the remainder is non-zero and the signs disagree -- the floor.
    ``d != 0`` (the caller's contract)."""
    q = _trunc_div(n, d)
    if _trunc_mod(n, d) != 0 and (n ^ d) < 0:
        q -= 1
    return q


def mean_round(total: int, count: int) -> int:
    """fixed_point.h ``mean_round``: ``total / count`` rounded half away from
    zero (the dividend biased by +/- ``count / 2``, then C++ ``/``), cast to
    q16. ``count <= 0`` returns 0."""
    if count <= 0:
        return 0
    half = count // 2
    m = _trunc_div(total + half, count) if total >= 0 else _trunc_div(total - half, count)
    return wrap_i32(m)


def reciprocal_q16(denom_q: int) -> int:
    """fixed_point.h ``reciprocal_q16``: the Newton reciprocal of a Q16.16
    denominator -- a two-bit power-of-two seed, four round-to-nearest Newton
    trips. Self-guards ``denom_q <= 0 -> 0`` and floors {1, 2} to 3.
    (The arm chain never calls it -- design 6.2 -- it is twinned so the kit's
    Python side has one home; sweep_ref_q's copy is held equal to it.)"""
    if denom_q <= 0:
        return 0
    if denom_q < 3:
        denom_q = 3
    bitlen = denom_q.bit_length()
    shift = 32 - bitlen
    if shift >= 1:
        base = 1 << shift
        upper_half = bitlen >= 2 and ((denom_q >> (bitlen - 2)) & 1)
        r = base if upper_half else base + (base >> 1)
    else:
        r = (1 << shift) if shift >= 0 else 1
        if r < 1:
            r = 1
    two_q = FP_ONE << 1
    half = 1 << (FP_SHIFT - 1)
    for _ in range(4):
        dr = (denom_q * r + half) >> FP_SHIFT
        r = (r * (two_q - dr) + half) >> FP_SHIFT
    return wrap_i32(r)


# ---------------------------------------------------------------------------
# Rounding narrows
# ---------------------------------------------------------------------------
def narrow_round_signed(wide: int) -> int:
    """fixed_point.h ``narrow_round_signed``: ``wide >> 16`` rounded half away
    from zero (``narrow_round`` on the magnitude, the sign re-applied), cast
    to q16."""
    half = 1 << (FP_SHIFT - 1)
    if wide >= 0:
        return wrap_i32((wide + half) >> FP_SHIFT)
    return wrap_i32(-(((-wide) + half) >> FP_SHIFT))


def q30_to_q16_round(v30: int) -> int:
    """fixed_point.h ``q30_to_q16_round`` (the trig kit's one narrow): Q.30 ->
    Q16.16, round half away from zero."""
    if v30 >= 0:
        return wrap_i32((v30 + (1 << 13)) >> 14)
    return wrap_i32(-(((-v30) + (1 << 13)) >> 14))


# ---- the CHAIN's rounding kit (design 6.3; P6c adds these to fixed_point.h) --
def shr_round_away(x: int, s: int) -> int:
    """``x / 2^s`` rounded half AWAY FROM ZERO: the magnitude rounded, the sign
    re-applied, so ``shr_round_away(-x, s) == -shr_round_away(x, s)`` exactly.
    ``s == 0`` is the identity. The general form of ``narrow_round_signed``
    (s = 16) and ``q30_to_q16_round`` (s = 14). C++ recipe (int64 ``x``,
    ``|x| + 2^(s-1)`` must not overflow -- the caller's bound)::

        s == 0 ? x : (x >= 0 ? (x + h) >> s : -(((-x) + h) >> s)),  h = 1 << (s-1)
    """
    if s == 0:
        return x
    h = 1 << (s - 1)
    return (x + h) >> s if x >= 0 else -(((-x) + h) >> s)


def div_round_away(n: int, d: int) -> int:
    """``n / d`` rounded half AWAY FROM ZERO for ``d > 0``: ``mean_round``'s
    rule without its int32 cast (an int64 result). Symmetric:
    ``div_round_away(-n, d) == -div_round_away(n, d)``. C++ recipe::

        n >= 0 ? (n + d / 2) / d : -(((-n) + d / 2) / d)
    """
    if d <= 0:
        raise ValueError(f"div_round_away: divisor {d} must be > 0")
    h = d // 2
    return (n + h) // d if n >= 0 else -(((-n) + h) // d)


def mul_shr_away(a: int, b: int, s: int) -> int:
    """``(a * b) / 2^s`` on the EXACT 128-bit product, rounded half away from
    zero (the sign of the product re-applied to the rounded magnitude).
    ``s >= 1``. C++ recipe, with no 128-bit add (``m = |a| |b|``)::

        q = mul128_shr(|a|, |b|, s - 1);   r = (q + 1) >> 1;   sign(a b) * r

    which equals ``floor((m + 2^(s-1)) / 2^s)``: with ``m = q 2^(s-1) + e``,
    ``0 <= e < 2^(s-1)``, both sides are ``floor((q + 1) / 2)``.
    """
    if s < 1:
        raise ValueError(f"mul_shr_away: shift {s} must be >= 1")
    m = abs(a) * abs(b)
    r = (m + (1 << (s - 1))) >> s
    return -r if (a < 0) != (b < 0) else r


# ---------------------------------------------------------------------------
# Square root
# ---------------------------------------------------------------------------
def sqrt_q16_digits(x_q32: int) -> int:
    """fixed_point.h ``sqrt_q16``, transcribed: the FIXED 32-trip binary digit
    recurrence on the uint64 radicand. ``x <= 0 -> 0``; a result above
    INT32_MAX clamps to it. ``x_q32`` is an int64."""
    if x_q32 <= 0:
        return 0
    x = x_q32
    res = 0
    bit = 1 << 62
    for _ in range(32):
        t = res + bit
        res >>= 1
        if x >= t:
            x -= t
            res += bit
        bit >>= 2
    if res > INT32_MAX:
        return INT32_MAX
    return res


def sqrt_q16(x_q32: int) -> int:
    """fixed_point.h ``sqrt_q16``: ``floor(sqrt(x))`` for an int64 radicand,
    ``x <= 0 -> 0``, clamped to INT32_MAX. The floor of a square root is
    unique, so this computes it with ``math.isqrt`` (the reference calls it in
    every constraint); ``sqrt_q16_digits`` is the literal transcription, and
    the twin test holds C++ == digits == this."""
    if x_q32 <= 0:
        return 0
    r = math.isqrt(x_q32)
    return INT32_MAX if r > INT32_MAX else r


# ---------------------------------------------------------------------------
# Trig (Q2-LIFT): atan2_q16 / sin_q16 / cos_q16
# ---------------------------------------------------------------------------
PI_2_Q30 = 1686629713
PI_Q30 = 3373259426
TWO_PI_Q30 = 6746518852
ATAN_C_Q30 = (1073741697, -357897613, 214393620, -150359183,
              105966136, -63167966, 25534137, -4896039)
SIN_C_Q30 = (1073741819, -178956877, 8947544, -212698, 2797)
COS_C_Q30 = (1073741774, -536869890, 44735921, -1487522, 24860)
#: ``quantize(pi)`` in Q16 rad -- what ``atan2_q16`` returns at the cut.
PI_Q16 = 205887


def _horner(coeffs, s30: int) -> int:
    """The kit's unrolled Horner: acc = c_n; acc = c_k + ((acc * s30) >> 30)."""
    acc = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        acc = c + ((acc * s30) >> 30)
    return acc


def sin_core_q30(a30u: int) -> int:
    """fixed_point.h ``sin_core_q30``: wrap a non-negative Q.30 angle into
    [0, 2pi), split into quadrant + remainder, evaluate the k-parity
    polynomial, narrow once, negate for k >= 2."""
    a30u %= TWO_PI_Q30
    k = a30u // PI_2_Q30
    r30 = a30u - k * PI_2_Q30
    s30 = (r30 * r30) >> 30
    if k & 1:
        m30 = _horner(COS_C_Q30, s30)
    else:
        m30 = (r30 * _horner(SIN_C_Q30, s30)) >> 30
    q = q30_to_q16_round(m30)
    return -q if k >= 2 else q


def sin_q16(a: int) -> int:
    """fixed_point.h ``sin_q16``: Q16 radians in, Q16 out; the sign stripped
    first and re-applied (exact odd symmetry)."""
    au = -a if a < 0 else a
    q = sin_core_q30(au << 14)
    return -q if a < 0 else q


def cos_q16(a: int) -> int:
    """fixed_point.h ``cos_q16``: ``sin(|a| + pi/2)`` through the one core."""
    au = -a if a < 0 else a
    return sin_core_q30((au << 14) + PI_2_Q30)


def atan2_q16(y: int, x: int) -> int:
    """fixed_point.h ``atan2_q16``: int32 y, x (only the ratio and the signs
    matter) -> Q16 radians in [-205887, 205887]. Octant fold through ONE
    rounded divide, the degree-7 Horner, the octant and quadrant unfold, one
    round-half-away narrow, then sign(y). ``atan2_q16(0, 0) == 0``."""
    if y == 0 and x == 0:
        return 0
    ay = -y if y < 0 else y
    ax = -x if x < 0 else x
    swap = ay > ax
    mn = ax if swap else ay
    mx = ay if swap else ax
    t30 = ((mn << 30) + (mx >> 1)) // mx
    s30 = (t30 * t30) >> 30
    acc = _horner(ATAN_C_Q30, s30)
    a30 = (t30 * acc) >> 30
    if swap:
        a30 = PI_2_Q30 - a30
    if x < 0:
        a30 = PI_Q30 - a30
    r = q30_to_q16_round(a30)
    return -r if y < 0 else r


__all__ = [
    "INT32_MIN", "INT32_MAX", "INT64_MIN", "INT64_MAX", "FP_SHIFT", "FP_ONE",
    "wrap_i64", "wrap_i32", "mul_wide", "mul128_shr", "floordiv_q",
    "mean_round", "reciprocal_q16", "narrow_round_signed", "q30_to_q16_round",
    "shr_round_away", "div_round_away", "mul_shr_away", "sqrt_q16_digits",
    "sqrt_q16", "PI_2_Q30", "PI_Q30", "TWO_PI_Q30", "PI_Q16", "sin_core_q30",
    "sin_q16", "cos_q16", "atan2_q16",
]
