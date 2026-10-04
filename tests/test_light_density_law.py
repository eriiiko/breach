"""The light channels' density law (#12 smoke x light, handle 2, 2026-10-04):
Beer-Lambert per tile, a_gas = ONE - exp(-tau), in pure integer arithmetic --
the law the renderer draws the medium's opacity with, so smoke that looks black
blocks the light. The C++ twin (radiation_sweep.h::exp_neg_q16, shared by the
CUDA sweep) is held bit for bit to the reference by the light sweep gates
(tests/test_radiation_sweep_light.py, cuda_radiation_sweep_check.py).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from _radiation_sweep_harness import ONE, R  # noqa: E402


def test_exp_neg_is_monotone_exact_at_zero_and_close_to_exp():
    """PROPERTY: exp_neg_q16 never rises with tau (denser smoke never lets more
    light through), is exactly ONE at tau = 0 (no smoke, no extinction) and 0
    from tau = 16, and stays within 3 Q16 counts of exp(-tau) everywhere.

    BREAKS IF: the integer chain loses monotonicity (a rounding change), the
    endpoints move, or the approximation drifts from Beer-Lambert.
    """
    assert R.exp_neg_q16(0) == ONE
    assert R.exp_neg_q16(16 * ONE) == 0 and R.exp_neg_q16(40 * ONE) == 0
    prev, worst = ONE, 0.0
    for x_q in range(0, 17 * ONE, 29):
        e = R.exp_neg_q16(x_q)
        assert e <= prev, x_q
        prev = e
        worst = max(worst, abs(e - math.exp(-x_q / ONE) * ONE))
    assert worst <= 3.0, worst


def test_light_uses_beer_lambert_and_heat_keeps_the_thin_law():
    """PROPERTY: for the same density sum, the LIGHT extinction is
    ONE - exp(-tau) and the HEAT extinction stays min(ONE, tau) -- handle 2
    changed the light's law only. At tau = 1 light lets ~37 % through where
    the thin law would block all of it.

    BREAKS IF: the light term falls back to the thin law, or the heat term is
    moved onto Beer-Lambert (that would move the sim and its golden).
    """
    dens, absorb, n_bulk = [ONE // 64], [64 * ONE], ONE        # tau = 1.0 exactly
    a_light = R.gas_extinction_exp_q(dens, absorb, n_bulk)
    a_heat = R.gas_extinction_q(dens, absorb, n_bulk)
    assert a_heat == ONE
    assert abs(a_light - (1 - math.exp(-1.0)) * ONE) <= 3
    assert R.gas_extinction_exp_q(dens, absorb, 0) == 0       # the N_EPS floor
