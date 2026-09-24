"""Q16.16 fixed-point helpers for swarm units (arc #63 P2, engine/17 §4.4).

Swarm units share the sim-wide Q16.16 scale (2^16 == 65536) that every other
fixed-point boundary module uses. Per the canonical-systems rule (Q16
boundary modules: "never hardcode 65536, never re-derive a shift/round/
reciprocal") this module re-exports the rounding helpers from
:mod:`simulation.gas_fixed` — the existing cross-module source
(``temperature_scale.py`` already imports ``gas_fixed.quantize_scalar``) —
instead of adding a SEVENTH hand-copy (survey §E flag 6,
``docs/canonical_systems_survey_2026-08-22.md`` counts six byte-identical
copies already in the tree).

It adds only what no Python module has yet: the checked-in Q16.16 π
constant, the swarm heading's wrap PERIOD (deliberately not
``philox32.TWO_PI_Q16``, the draw RANGE — see :func:`wrap_heading_q16`), and
the wrap law itself. No ``breach_physics`` import, no transcendental (π is a
checked-in constant, door 2 of the ingress rule) — this module never calls
into the C++ kit.
"""
from __future__ import annotations

import numpy as np

from simulation.gas_fixed import (  # noqa: F401  re-export — survey §E flag 6: no 7th copy
    FP_SHIFT, FP_ONE, FP_ONE_F, quantize, quantize_scalar, dequantize, dequantize_f32,
)

# The kit's narrowed Q16.16 pi (cpp/src/fixed_point.h:900-903): round(pi*2^16)
# checked against the Q.30 pipeline's own narrow-consistency static_assert
# (((PI_Q30 + (1 << 13)) >> 14) == 205887). No Python PI_Q16 existed before
# this module (the only prior copy was the test-local `PI_Q` at
# tests/test_fixed_trig.py:53) — this is now THE checked-in constant.
PI_Q16 = 205887

# Swarm heading WRAP PERIOD == 2 * PI_Q16 == 411774, NOT quantize(2*pi) ==
# 411775 (philox32.TWO_PI_Q16, the Philox DRAW range). At Q16 the three pi
# constants are inconsistent (fixed_point.h:890-894: pi == 2*(pi/2) holds at
# Q.30 but NOT at Q16.16 -- round(pi*2^16) = 205887 while 2*round(pi/2*2^16)
# = 205888). Only a period of EXACTLY twice the half-range makes (-P, P] a
# complete residue system (one representation per direction), so this value
# is forced, not a free choice. `philox32.TWO_PI_Q16` (411775) is a DIFFERENT
# number for a DIFFERENT purpose (the draw range) -- they combine as
# draw-then-wrap; nobody "unifies" them (engine/17 P2 doc §2.4/§9).
HEADING_PERIOD_Q16 = 2 * PI_Q16


def wrap_heading_q16(h):
    """Canonicalize a raw Q16.16 heading to (-PI_Q16, PI_Q16].

    ``wrap(h) = PI_Q16 - ((PI_Q16 - h) mod HEADING_PERIOD_Q16)``, with the
    floor-mod computed in an UNBOUNDED-or-int64 domain. A scalar Python int
    in gives a Python int out (Python ints are unbounded and ``%`` floors);
    an ndarray in gives an int64 ndarray out (``np.mod`` floors). The int64
    upcast is load-bearing for the array path: an int32 ``PI_Q16 - h`` wraps
    SILENTLY in numpy (probe: ``PI_Q16 - INT32_MIN`` gives -2147277761 in
    int32 vs. the correct 2147689535 in int64).

    Verified (NumPy 2.4.6, Python 3.11.7): wrap(+-PI_Q16) == PI_Q16,
    wrap(+-3*PI_Q16) == PI_Q16, wrap(411774) == 0, wrap(411775) == 1,
    wrap(INT32_MAX) == 82237, wrap(INT32_MIN) == -82238, and wrap is
    idempotent.
    """
    if isinstance(h, np.ndarray):
        h64 = h.astype(np.int64, copy=False)
        return PI_Q16 - np.mod(PI_Q16 - h64, HEADING_PERIOD_Q16)
    return int(PI_Q16 - ((PI_Q16 - int(h)) % HEADING_PERIOD_Q16))
