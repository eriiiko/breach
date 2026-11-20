"""Exhaustive proof that ``action_spec_v1.onto_reach_q`` moved no quantized
reach when its root changed from ``float(d2) ** 0.5`` (the C runtime's
``pow``) to ``math.sqrt`` (IEEE, correctly rounded) -- #29 P8.0b.

The lob reach feeds the throw latch (synced game state), so the change is
allowed only if ``quantize_scalar`` of both forms agrees on EVERY d2 the
function can receive.

The bound. ``onto_reach_q``'s one caller (``control_policy`` decoder, the
"onto" throw) passes a unit's centre tile and a team-picture record's centre
tile: both are integer tile coordinates of a unit position, which is a
Q16.16 int32 (``Unit.x_q`` / ``y_q``), so each lies in [0, 32768) and
``|dx|, |dy| <= 32767``: ``d2 <= 2 * 32767**2 = 2_147_352_578 < 2**31``.
The sweep covers every integer d2 in [0, 2**31) (sums of two squares or
not), a superset of the reachable set on any map the game can represent;
a d2 whose quantized reach differs is then tested for REACHABILITY (a sum
dx**2 + dy**2 with both |dx|, |dy| <= 32767) and only a reachable one
fails the sweep. (First run, 2026-11-20: 3 of the 2**31 integers differ,
1455325460, 1716802626, 1878698752: none is such a sum, and every one has
a root above 38000 tiles, past the int32 Q16 range of the reach column
(d2 < 2**30) as well.)

The old form is reproduced in bulk by ``np.power(k, <array of 0.5>)``: with
an ARRAY exponent numpy calls the same C ``pow`` CPython's float ``**``
calls (a scalar 0.5 exponent takes numpy's sqrt shortcut instead). The tool
checks that equivalence against real ``float(k) ** 0.5`` first (every k
below 4e6 plus 2e6 random k over the whole range) and refuses to sweep if a
single value differs.

Quantization is ``unit_fixed.quantize_scalar``'s arithmetic, vectorised: for
x >= 0, ``floor(x * 65536 + 0.5)``; below 2**31 the product is exact and the
+0.5 is exact (ulp <= 2**-21), so the vector form equals the scalar one.

Run (about two to four minutes, single thread):
    python tools/reach_sqrt_sweep.py
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

BOUND = 1 << 31            # d2 in [0, BOUND)
CHUNK = 1 << 24
FP_ONE = 65536.0


def quantize(x: np.ndarray) -> np.ndarray:
    return np.floor(x * FP_ONE + 0.5).astype(np.int64)


def old_root(k: np.ndarray) -> np.ndarray:
    """``float(k) ** 0.5`` for every k, through the C runtime's ``pow``."""
    return np.power(k, np.full(k.shape, 0.5))


def check_old_root_is_python_pow() -> None:
    first = np.arange(4_000_000, dtype=np.float64)
    rng = np.random.default_rng(29_080)
    sample = rng.integers(0, BOUND, 2_000_000).astype(np.float64)
    for k in (first, sample):
        ref = np.array([float(v) ** 0.5 for v in k.tolist()])
        bad = int(np.count_nonzero(old_root(k) != ref))
        if bad:
            sys.exit(f"np.power(array exponent) is not CPython's ** on {bad} "
                     "values: the sweep would not test the old form; STOP")
    n_pow_quirks = int(np.count_nonzero(old_root(first) != np.sqrt(first)))
    print(f"old-root stand-in == CPython float ** 0.5 on 6e6 values "
          f"(pow != sqrt on {n_pow_quirks} of k < 4e6)")


def reachable(d2: int) -> list:
    """The offsets ``(dx, dy)``, ``0 <= dx <= dy <= 32767``, with
    ``dx*dx + dy*dy == d2``."""
    out = []
    for a in range(0, math.isqrt(d2 // 2) + 1):
        b2 = d2 - a * a
        b = math.isqrt(b2)
        if b * b == b2 and b <= 32767:
            out.append((a, b))
    return out


def main() -> int:
    t0 = time.perf_counter()
    check_old_root_is_python_pow()
    raw_diff = 0
    q_diff = 0
    q_diff_q16 = 0          # d2 < 2**30: the reach fits the int32 Q16 column
    q_examples = []
    for lo in range(0, BOUND, CHUNK):
        k = np.arange(lo, min(lo + CHUNK, BOUND), dtype=np.float64)
        old = old_root(k)
        new = np.sqrt(k)
        raw_diff += int(np.count_nonzero(old != new))
        dq = np.nonzero(quantize(old) != quantize(new))[0]
        q_diff += int(dq.size)
        q_diff_q16 += int(np.count_nonzero(lo + dq < (1 << 30)))
        q_examples.extend(int(lo + i) for i in dq)
    dt = time.perf_counter() - t0
    print(f"d2 in [0, 2**31): {BOUND} values; root differs before quantizing "
          f"on {raw_diff}; quantized reach differs on {q_diff} "
          f"({q_diff_q16} of them below 2**30) ({dt:.0f} s)")
    bad = 0
    for d2 in q_examples:
        offs = reachable(d2)
        print(f"  d2 = {d2}: root {math.sqrt(d2):.3f} tiles, offsets with "
              f"|dx|, |dy| <= 32767: {offs[:4] or 'none (unreachable)'}")
        bad += bool(offs)
    print(f"reachable d2 whose quantized reach differs: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
