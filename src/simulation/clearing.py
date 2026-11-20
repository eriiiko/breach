"""The clearing step -- what a unit's slow state loses in one tick when it
drains by a fraction of itself (#215, ``docs/body_temperature_design_2026-11-
17.md`` section 2b, Systems (b)). Lifted from ``poison.step`` so the poison law
and the stress level clear by ONE step; a unit state that drains by a fraction
a tick uses it, never a second one.

    drain(x, a, shift, flush_q) = 0                   if x <= 0
                                = x                   if 0 < x <= flush_q
                                = max(1, (x * a) >> shift)   otherwise

``a`` is the per-tick fraction in the caller's own currency (``2^shift`` is
ONE): the poison law's Q16 (``shift`` 16), the stress level's finer
:data:`RATE_SHIFT` (a half-life of minutes rounds to a handful of Q16 counts at
24 ticks a second). The floor of one count means a level never stalls above
zero; the flush (``flush_q`` > 0) empties a tail at once, so the level returns
to EXACTLY 0 and the unit hashes as never touched. ``flush_q`` 0 never
flushes. Integer only, no random draw; the device twin is
``unit_lib::clearing_drain`` (``cpp/src/cuda_unit_lib.cuh``), integer for
integer.

:func:`half_life_rate` is the door that turns an authored half-life into the
per-tick fraction: ``round(ln 2 x 2^shift / (seconds x tps))`` with ``ln 2`` an
exact rational of 30 digits (no libm), rounded once. The first-order rate of a
half-life (``1 - 2^(-1/n)`` is ``ln 2 / n`` to a part in ``2n``): the
half-life the integer step then keeps is the authored one to well inside 1 %
(``tests/test_stress.py`` measures it at 120 s and 1200 s).
"""
from __future__ import annotations

import math
from fractions import Fraction

from simulation import unit_fixed

#: The stress level's rate currency: a per-tick fraction in units of 2^-24
#: (a 120 s half-life at 24 tps is 4038 counts, a 20-minute one 404; in Q16
#: they would be 16 and 2).
RATE_SHIFT = 24
RATE_ONE = 1 << RATE_SHIFT

#: ln 2 to 30 digits, an exact rational (the door's constant: no libm).
LN2 = Fraction("0.693147180559945309417232121458")


def drain(x: int, a: int, shift: int = unit_fixed.FP_SHIFT,
          flush_q: int = 0) -> int:
    """What leaves a level ``x`` this tick at the fraction ``a`` (``2^shift``
    = ONE): 0 at or below 0; all of it at or below ``flush_q`` (when
    ``flush_q`` > 0); else ``max(1, (x * a) >> shift)``."""
    if x <= 0:
        return 0
    if flush_q > 0 and x <= flush_q:
        return x
    m = (x * a) >> shift
    return m if m >= 1 else 1


def half_life_rate(seconds: float, tps: int, shift: int = RATE_SHIFT) -> int:
    """The per-tick fraction (``2^shift`` = ONE) of a half-life of
    ``seconds``: ``round(ln 2 x 2^shift / (seconds x tps))``, the authored
    float taken as its exact rational, rounded once half away from zero.
    Refuses a half-life below one tick or not finite."""
    s = float(seconds)                        # p3e-door: the authored time
    if not math.isfinite(s) or s * int(tps) < 1.0:
        raise ValueError(f"a half-life must be at least one tick "
                         f"(1/{int(tps)} s; got {seconds!r})")
    t = Fraction(s)
    num = LN2 * (1 << int(shift)) / (t * int(tps))
    return unit_fixed.div_round(num.numerator, num.denominator)


def rate_per_tick(rate_per_s: float, tps: int, shift: int = RATE_SHIFT) -> int:
    """The per-tick fraction (``2^shift`` = ONE) of a first-order RATE
    authored per second (arc #63 P5b, the corpse's infestation decay):
    ``round(rate x 2^shift / tps)``, the authored float taken as its exact
    rational, rounded once half away from zero. Refuses a negative or
    non-finite rate. (A half-life goes through :func:`half_life_rate`.)"""
    r = float(rate_per_s)                     # p3e-door: the authored rate
    if not math.isfinite(r) or r < 0.0:
        raise ValueError(f"a rate must be finite and >= 0 (got {rate_per_s!r})")
    num = Fraction(r) * (1 << int(shift)) / int(tps)
    return unit_fixed.div_round(num.numerator, num.denominator)


__all__ = ["RATE_SHIFT", "RATE_ONE", "LN2", "drain", "half_life_rate",
           "rate_per_tick"]
