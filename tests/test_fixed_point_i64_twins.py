"""The kit's int64 twins (ray-engine-v2 P1, design v3 §3 and §2.8).

Two FP_HD functions joined cpp/src/fixed_point.h, unused by the live path yet:

  * ``shr_round0_i64`` — the int64 twin of the q16 symmetric round-toward-0
    shift. The Pass-1 fold applies ``shr_round0`` to the SIGNED ``rad_net``,
    so when that plane widens (P3) the fold must call a function that returns
    the SAME integers on every int32-range input — asserted here, so the
    widening is a type change and not a rounding change.
  * ``deposit_dT_wide_i64`` — the STAGED wide chain
    ``mul128_shr(mul128_shr(deposit, recip_n, 16), recip_cv, 32)`` for a wide
    (int64) first operand. It floors TWICE where ``deposit_dT_wide_q16`` floors
    once at 48 bits: a DIFFERENT rounding, declared to differ by at most one
    LSB on int32-range deposits, and finite (no int64 overflow) where the
    one-narrow chain's plain-int64 first product is not: deposit = E°[3999].

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_fixed_point_i64_twins.py -q
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

INT32_MIN, INT32_MAX = -(2 ** 31), 2 ** 31 - 1
FP_SHIFT, RECIP_SHIFT = 16, 32


def _py_shr_round0(x: int, s: int) -> int:
    return -((-x) >> s) if x < 0 else (x >> s)


def _py_staged(deposit: int, recip_n: int, recip_cv: int) -> int:
    """The staged chain in exact Python ints (floor at each stage)."""
    stage1 = (deposit * recip_n) >> FP_SHIFT
    return (stage1 * recip_cv) >> RECIP_SHIFT


def _py_one_narrow(deposit: int, recip_n: int, recip_cv: int) -> int:
    return (deposit * recip_n * recip_cv) >> (FP_SHIFT + RECIP_SHIFT)


def test_shr_round0_i64_equals_q16_on_every_int32_range_input_tried():
    """PROPERTY: shr_round0_i64(x, s) == shr_round0(x, s) for int32 x, s in 0..31
    (edges + random), so widening the fold's plane cannot move a golden.

    BREAKS IF: the twin's rounding is anything but the symmetric round-toward-0
    (a plain arithmetic `>>` gives -2 for -3 >> 1 where the twin gives -1).
    """
    rng = random.Random(20260916)
    # INT32_MIN is deliberately NOT in the equality set: see the next test.
    edges = [0, 1, -1, 2, -2, 3, -3, INT32_MAX, INT32_MIN + 1,
             65535, 65536, -65535, -65536, 2 ** 30, -(2 ** 30), 12345, -12345]
    n_checked = 0
    for s in range(0, 32):
        xs = edges + [rng.randint(INT32_MIN + 1, INT32_MAX) for _ in range(300)]
        for x in xs:
            got64 = bp.fp_shr_round0_i64(x, s)
            got32 = bp.fp_shr_round0(x, s)
            assert got64 == got32 == _py_shr_round0(x, s), (x, s, got64, got32)
            n_checked += 1
    assert n_checked > 9000
    # NON-VACUITY: the symmetric shift really differs from the plain floor shift
    # on negative operands, so the equality above is not the trivial one.
    assert bp.fp_shr_round0_i64(-3, 1) == -1 and (-3 >> 1) == -2
    assert bp.fp_shr_round0_i64(-(2 ** 40) - 1, 4) == -(2 ** 36)


def test_q16_shr_round0_overflows_at_int32_min_and_the_twin_does_not():
    """FINDING, pinned (ray-engine-v2 P1): the q16 shr_round0's `-x` overflows
    at x == INT32_MIN (undefined behaviour; MSVC wraps it), so shr_round0
    (INT32_MIN, 1) returns +2^30 while the int64 twin returns the correct
    -2^30. The fold can only ever see INT32_MIN in `rad_net` if the old cast
    WRAPPED (its documented out-of-band contract), which the A/B scenario is
    asserted not to do (tests/test_radiation_sweep_shadow_wiring.py). Pinned
    here so the widening at P3 is understood to REMOVE this edge, not to
    preserve it.

    BREAKS IF: the q16 form is fixed (then the two agree at INT32_MIN — and
    this test should be retired with the note), or the twin regresses.
    """
    assert bp.fp_shr_round0_i64(INT32_MIN, 1) == -(2 ** 30)
    assert bp.fp_shr_round0_i64(INT32_MIN, 0) == INT32_MIN
    assert bp.fp_shr_round0(INT32_MIN, 1) != bp.fp_shr_round0_i64(INT32_MIN, 1)


def test_shr_round0_i64_is_symmetric_past_int32():
    """PROPERTY: for |x| beyond int32, shr_round0_i64(-x, s) == -shr_round0_i64(x, s)
    == -(x >> s) — the int64 range is where the sweep's planes live.

    BREAKS IF: the twin narrows its operand (an int32 truncation would fold
    2^40 + 5 to 5) or loses the sign symmetry.
    """
    for x in (2 ** 40 + 5, 2 ** 62 - 1, 3 * 2 ** 33 + 7, 2 ** 41 + 12345):
        for s in (1, 3, 16, 24, 40):
            assert bp.fp_shr_round0_i64(x, s) == x >> s
            assert bp.fp_shr_round0_i64(-x, s) == -(x >> s)


def _shipped_recips():
    """(recip_n_q, recip_cv) pairs in the ranges the fold actually uses: N from
    the shipped n_floor_heat (0.01) up to a full cell (~1.0) through the kit's
    own reciprocal_q16, and c_v reciprocals from make_recip over [1, 100]
    (the one-LSB claim presumes c_v >= 1: for c_v < 1 the second floor can
    drop a further 1/c_v, which the test below states rather than hides)."""
    recip_ns = [bp.fp_reciprocal_q16(int(n * 65536)) for n in
                (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)]
    recip_cvs = [bp.fp_make_recip(cv) for cv in (1.0, 1.5, 2.0, 4.0, 10.0, 100.0)]
    return recip_ns, recip_cvs


def test_staged_wide_chain_within_one_lsb_of_the_one_narrow_chain():
    """PROPERTY: |deposit_dT_wide_i64 - deposit_dT_wide_q16| <= 1 on every
    int32-range deposit tried, with c_v >= 1 — and the two are NOT identical
    (the staged chain floors twice), so the bound is a bound on something.

    BREAKS IF: a stage narrows more than once, uses a rounding other than the
    floor, or the shifts are re-split (a 24/24 split would drift by orders of
    magnitude).
    """
    rng = random.Random(7)
    recip_ns, recip_cvs = _shipped_recips()
    deposits = [0, 1, 65536, 12345678, INT32_MAX, 2 ** 30, 999, 1 << 20]
    deposits += [rng.randint(0, INT32_MAX) for _ in range(200)]
    n_diff = 0
    n_checked = 0
    for dep in deposits:
        for rn in recip_ns:
            for rcv in recip_cvs:
                staged = bp.fp_deposit_dT_wide_i64(dep, rn, rcv)
                narrow = bp.fp_deposit_dT_wide_q16(dep, rn, rcv)
                assert staged == _py_staged(dep, rn, rcv), (dep, rn, rcv)
                assert narrow == _py_one_narrow(dep, rn, rcv), (dep, rn, rcv)
                assert abs(staged - narrow) <= 1, (dep, rn, rcv, staged, narrow)
                n_diff += int(staged != narrow)
                n_checked += 1
    assert n_checked > 5000
    assert n_diff > 0, "the two chains never differed: the one-LSB bound is vacuous"


def test_staged_wide_chain_is_finite_and_exact_at_the_table_top():
    """PROPERTY: at deposit = E°[3999] (~2^41.7, far past int32) the staged chain
    returns the exact big-integer value — no int64 overflow anywhere in it —
    including with the smallest-N reciprocal the fold can produce, where the
    one-narrow chain's plain-int64 first product would overflow (critique 3
    §1c: 2^64.4 .. 2^72.1).

    BREAKS IF: either stage forms a plain 64-bit product instead of the
    128-bit primitive.
    """
    sys.path.insert(0, str(ROOT / "docs" / "ray_engine_v2_scheme_study_2026-09-13"))
    import sweep_ref_q as R  # noqa: E402
    e_top = int(R.E[R.E_TABLE_SIZE - 1])
    assert e_top > INT32_MAX
    recip_ns, recip_cvs = _shipped_recips()
    recip_ns.append(bp.fp_reciprocal_q16(3))       # the kit's self-floor: the widest reciprocal
    for rn in recip_ns:
        for rcv in recip_cvs:
            expect = _py_staged(e_top, rn, rcv)
            got = bp.fp_deposit_dT_wide_i64(e_top, rn, rcv)
            assert got == expect, (rn, rcv, got, expect)
            # The first product of the ONE-narrow chain overflows int64 here for
            # the wide reciprocals -- the reason the staged twin exists.
            if e_top * rn >= 2 ** 63:
                assert True   # documented: deposit_dT_wide_q16 could not take this operand


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
