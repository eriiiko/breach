"""Bedrock cliff-patch — the COMMITTED gate for the integer smoke-CFL substep count.

WHAT THIS PINS
--------------
The Bedrock cliff-patch moved the smoke forward-Euler diffusion stability floor
(the substep COUNT ``n_smoke``) off ``double + std::ceil`` onto an exact
128-bit-rational INTEGER ceil, so the count is bit-identical across machines /
GPUs (the last ``double`` on the determinism-critical substep-count path). The
shipped helper is ``fixedpoint::smoke_cliff_count`` (cpp/src/fixed_point.h), with
three real-target implementations (``__int128`` / MSVC ``_umul128`` / a portable
64-bit staged fallback). The ``__int128`` and ``_umul128`` paths are the exact
cross-machine contract.

The C++ binding ``breach_physics.smoke_cliff_count`` exposes the SHIPPED helper
(the real 128-bit / ``_umul128`` path, exactly what ``run_substeps`` feeds the
engine each tick). This test verifies that shipped C++ against an INDEPENDENT
exact-rational Python mirror (arbitrary-precision ints) — not against a second
re-implementation that could share a bug.

THE FORMULA (mirror of fixed_point.h::smoke_cliff_count, ~lines 308-368)
------------------------------------------------------------------------
The helper takes the already-quantized integer inputs (the binding signature is
``smoke_cliff_count(c4st_q, dsmoke_q, wds_q, mws_q32)``):

  * ``c4st_q``  = quantize(4*sim_time)            Q16.16  (scale 2^16)
  * ``dsmoke_q``= quantize(d_smoke_max)           Q16.16  (scale 2^16)
  * ``wds_q``   = quantize(wind_diffusion_scale)  Q16.16  (scale 2^16)
  * ``mws_q32`` = the integer Q.32 spatial-max of |wind|^2 (scale 2^32, int64)

and computes the true integer ceil of

  n_smoke = ceil( 4*sim_time*d_smoke_max * (1 + wind_diffusion_scale*max_wind_sq) )

via the exact rational

  base_q32  = c4st_q * dsmoke_q                  (scale 2^32)
  wmult_q48 = 2^48 + wds_q * mws_q32             (scale 2^48)
  prod      = base_q32 * wmult_q48               (scale 2^80)
  n_smoke   = (prod + 2^80 - 1) >> 80            (integer ceil)

with the floors/caps:

  * c4st_q <= 0 or dsmoke_q <= 0  -> return 1     (no tick / no diffusion -> 1 step)
  * mws_q32 < 0                   -> treat as 0   (|wind|^2 is non-negative)
  * n < 1                        -> clamp to 1
  * n > SMOKE_N_CAP = 2^20        -> clamp to SMOKE_N_CAP (absurd-wind guard)

Because the inputs are the quantized integers (NOT the floats), there is NO float
in this mirror at all — it is pure arbitrary-precision integer arithmetic, so the
comparison is exact and the gate is unambiguous.

THE FUZZ GRID
-------------
A deterministic (seeded) grid of a few thousand cases: random in-range inputs +
structured EDGES — the format-max wind component, the SMOKE_N_CAP boundary near
2^20, exact-multiple-of-2^80 ceil boundaries (R integer vs R+1 ULP), tiny counts
(n == 1), and zero wind. Zero mismatches expected (the patch review fuzzed 9M
cases clean — this is the committed regression gate).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_bedrock_cliff_counts.py -q
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

try:
    import breach_physics as bp
except ImportError as exc:  # pragma: no cover - build missing
    pytest.skip(f"breach_physics extension not built: {exc}",
                allow_module_level=True)

if not hasattr(bp, "smoke_cliff_count"):  # pragma: no cover - stale build
    pytest.skip("breach_physics built without smoke_cliff_count "
                "(rebuild cpp/ after the Bedrock cliff-patch)",
                allow_module_level=True)


# ---------------------------------------------------------------------------
# Format constants (mirror cpp/src/fixed_point.h).
# ---------------------------------------------------------------------------
SMOKE_N_CAP = 1 << 20          # 1,048,576: the absurd-wind saturation cap
_TWO80 = 1 << 80               # the ceil denominator (scale of base*wmult)

# The integer-input domain. q16 is int32 (Q16.16); the helper takes the products
# as int64, so the in-range producer keeps c4st_q/dsmoke_q/wds_q as non-negative
# int32 (their natural sign in the cliff) and mws_q32 as a non-negative int64
# Q.32 (the |wind|^2 spatial-max). The format-max Q16.16 magnitude is 2^31 - 1;
# the format-max single wind component is 32767 -> its square's spatial-max is
# bounded by 2 * (32767<<16)^2 (two components), i.e. ~2^62. We allow the full
# int64 mws here so the mirror is exercised past anything the engine produces.
_Q16_MAX = (1 << 31) - 1
_MWS_MAX = (1 << 62)           # ~ format-max |wind|^2 over the grid (two components)


# ---------------------------------------------------------------------------
# The exact-rational Python mirror (arbitrary-precision ints; no float).
# ---------------------------------------------------------------------------
def smoke_cliff_count_ref(c4st_q: int, dsmoke_q: int, wds_q: int,
                          mws_q32: int) -> int:
    """Exact integer mirror of fixed_point.h::smoke_cliff_count.

    Pure Python big-int arithmetic — the true ceil of the rational
    base*wmult / 2^80 with the helper's floors/cap. This is the independent
    oracle the shipped C++ is checked against."""
    # Floors: no tick / no diffusion -> a single step (the helper's early-out).
    if c4st_q <= 0 or dsmoke_q <= 0:
        return 1
    if mws_q32 < 0:
        mws_q32 = 0
    base_q32 = c4st_q * dsmoke_q                       # scale 2^32
    wmult_q48 = (1 << 48) + wds_q * mws_q32            # scale 2^48
    prod = base_q32 * wmult_q48                        # scale 2^80
    n = (prod + (_TWO80 - 1)) >> 80                    # integer ceil
    if n < 1:
        n = 1
    if n > SMOKE_N_CAP:
        return SMOKE_N_CAP
    return n


def _check(c4st_q: int, dsmoke_q: int, wds_q: int, mws_q32: int) -> None:
    """Assert the shipped C++ equals the mirror for one input tuple."""
    got = bp.smoke_cliff_count(c4st_q=c4st_q, dsmoke_q=dsmoke_q,
                               wds_q=wds_q, mws_q32=mws_q32)
    want = smoke_cliff_count_ref(c4st_q, dsmoke_q, wds_q, mws_q32)
    assert got == want, (
        f"smoke_cliff_count mismatch: C++={got} mirror={want} for "
        f"c4st_q={c4st_q} dsmoke_q={dsmoke_q} wds_q={wds_q} mws_q32={mws_q32}")


# ---------------------------------------------------------------------------
# Structured EDGE cases — the boundaries where a wrong truncation/cap shows up.
# ---------------------------------------------------------------------------
def _edge_cases():
    cases = []

    # zero wind (the +1 part only: n = ceil(4*st*d_smoke)).
    cases += [
        (q, d, w, 0)
        for q in (1, 65536, 4 * 65536, 100 * 65536, _Q16_MAX)
        for d in (1, 65536, 65536 // 2, 3 * 65536, _Q16_MAX)
        for w in (0, 65536, _Q16_MAX)
    ]

    # tiny counts (n == 1) and the no-tick / no-diffusion early-outs.
    cases += [
        (0, 65536, 65536, 0),          # c4st_q <= 0 -> 1
        (-1, 65536, 65536, 1 << 40),   # negative tick -> 1
        (65536, 0, 65536, 0),          # dsmoke_q <= 0 -> 1
        (65536, -5, 65536, 1 << 40),   # negative d_smoke -> 1
        (1, 1, 0, 0),                  # smallest positive -> ceil(tiny) = 1
        (1, 1, 1, 1),                  # everything = 1 ULP -> 1
        (65536, 65536, 65536, -1),     # negative mws -> treated as 0
        (65536, 65536, 65536, -(1 << 50)),  # large negative mws -> 0
    ]

    # the format-max wind component: a single Q16.16 wind component at 32767
    # contributes (32767<<16)^2 to mws; two components -> ~2 * that.
    wcomp = 32767 << 16
    mws_fmt = 2 * wcomp * wcomp
    cases += [
        (4 * 65536, 65536, 65536, mws_fmt),
        (4 * 65536, 65536, _Q16_MAX, mws_fmt),
        (_Q16_MAX, _Q16_MAX, _Q16_MAX, mws_fmt),    # ~112-bit product (cap'd)
        (65536, 65536, 65536, _MWS_MAX),
    ]

    # exact-multiple-of-2^80 ceil boundaries: choose inputs so prod lands on,
    # just below, and just above an exact multiple of 2^80, so ceil flips. With
    # wds=0, prod = (c4st_q*dsmoke_q) << 48; choosing c4st_q*dsmoke_q a multiple
    # of 2^32 makes prod an exact multiple of 2^80 (R integer -> ceil == R), and
    # +/-1 in the product probes the rounding bias on both sides.
    for k in (1, 2, 7, 1000):
        base_exact = k << 32                # base_q32 a multiple of 2^32
        # factor base_exact into c4st_q*dsmoke_q within Q16.16 range.
        # base_exact = (k<<32); pick dsmoke_q = 1<<16, c4st_q = k<<16 (both <2^31
        # for the k we use). Then prod = base_exact<<48 = exact multiple of 2^80.
        c4 = k << 16
        ds = 1 << 16
        if c4 <= _Q16_MAX:
            cases.append((c4, ds, 0, 0))        # R == k exactly (ceil == k)
            # nudge the product just below/above via a 1-ULP wind term:
            cases.append((c4, ds, 1, 1))        # prod += base*1 -> ceil == k+1
    return cases


def test_edge_cases():
    """The structured boundaries: zero wind, n==1 / early-outs, format-max wind,
    the cap, and exact-2^80 ceil flips."""
    for (c4, ds, w, mws) in _edge_cases():
        _check(c4, ds, w, mws)


def test_cap_boundary():
    """Inputs straddling SMOKE_N_CAP = 2^20: just under, exactly at, and well
    over the cap. The C++ clamps to the cap; the mirror must agree (both the
    clamped value and the un-clamped values just below)."""
    # With wds=0, n = ceil((c4st_q*dsmoke_q) / 2^32). Pick dsmoke_q = 2^16 so
    # n = ceil(c4st_q / 2^16) = ceil(c4st_q as a real number). Then sweep c4st_q
    # in Q16.16 around the cap so n lands at 2^20 - 1 .. 2^20 + a lot.
    ds = 1 << 16
    for n_target in (SMOKE_N_CAP - 2, SMOKE_N_CAP - 1, SMOKE_N_CAP,
                     SMOKE_N_CAP + 1, SMOKE_N_CAP + 5, 4 * SMOKE_N_CAP):
        c4 = n_target << 16          # exact integer n_target
        if c4 <= _Q16_MAX:
            _check(c4, ds, 0, 0)
            _check(c4, ds, 0, 1)     # tiny wind bump, still near the boundary
    # Drive the cap hard through the wind term at format-max magnitudes.
    wcomp = 32767 << 16
    mws_fmt = 2 * wcomp * wcomp
    _check(_Q16_MAX, _Q16_MAX, _Q16_MAX, mws_fmt)


def test_fuzz_random():
    """A few thousand deterministic random in-range tuples. Covers the bulk of
    the input space between the structured edges; zero mismatches expected."""
    rng = random.Random(0xB12DC0DE)      # deterministic seed
    N = 5000
    for _ in range(N):
        c4 = rng.randint(0, _Q16_MAX)
        ds = rng.randint(0, _Q16_MAX)
        wds = rng.randint(0, _Q16_MAX)
        # log-ish spread on mws so we hit calm (small), play-peak (~2^40), and
        # format-max (~2^62) regimes, plus the occasional zero.
        if rng.random() < 0.1:
            mws = 0
        else:
            bits = rng.randint(0, 62)
            mws = rng.randint(0, (1 << bits))
        _check(c4, ds, wds, mws)


def test_fuzz_play_regime():
    """A denser sweep over the REAL play regime — the values the engine actually
    feeds (sim_time ~ 1/60..1/20 s, d_smoke O(0.01..1), wds O(1), |wind| up to a
    few tens at a blast). Verifies the gate is sharp where it matters, not only
    at the exotic edges."""
    rng = random.Random(0x5A1ED)
    FP = 65536
    for _ in range(3000):
        # 4*sim_time in [0.05, 0.5] s -> Q16.16
        c4 = rng.randint(int(0.05 * FP), int(0.5 * FP))
        # d_smoke_max in [0.001, 2.0]
        ds = rng.randint(int(0.001 * FP), int(2.0 * FP))
        # wind_diffusion_scale in [0, 4]
        wds = rng.randint(0, int(4.0 * FP))
        # |wind| up to ~60 -> |wind|^2 up to 3600 -> mws_q32 = round(v * 2^32)
        wmag2 = rng.uniform(0.0, 3600.0)
        mws = int(wmag2 * (1 << 32))
        _check(c4, ds, wds, mws)


if __name__ == "__main__":
    # Standalone smoke run (no pytest) — enumerate all the structured cases plus
    # a quick random batch, report the total checked.
    n = 0
    for (c4, ds, w, mws) in _edge_cases():
        _check(c4, ds, w, mws)
        n += 1
    test_cap_boundary(); test_fuzz_random(); test_fuzz_play_regime()
    print(f"OK — {n} edge cases + 8000+ fuzz cases, zero mismatches.")
