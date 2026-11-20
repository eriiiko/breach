"""#217 finding 1 / #72: the EOS odd-even (checkerboard) mode.

PROPERTY: in a sealed box of still, ambient air, a 0.1 % every-other-tile
density seed does not grow. The air's own pressure C*N*T_abs and the stored
pressure (`atmosphere`) stay within a few times the seed of each other, so no
cell-to-cell pattern of density or temperature builds that the stored
pressure cannot see.

TODAY THIS FAILS (strict xfail): at 0.333 m tiles the gap reaches ~4 % rms
within ~1 s and stays there, whatever the seed (0.01 % to 1 %), with a fixed
grid parity. Measured and located in docs/eos_checkerboard_design_2026-11-20.md
(the bench: tests/_eos_checkerboard_bench.py).

WHAT MUST BREAK IT: the remedy Erik chooses for #217 finding 1 turns this
into an XPASS, which strict mode reports as a failure: the remedy's patch
removes the xfail marker. From then on, undoing the remedy (taking the
pressure coupling of the odd-even mode back out of the face mass flux, the
kick or the solve, or reintroducing a parity-locked residue in step 3's
integer result) must make it fail again.

Cheap on purpose (one small level, two simulated seconds): it runs in the
not-slow tier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import _eos_checkerboard_bench as bench  # noqa: E402

SEED = 1e-3            # +/-0.1 % bulk air, every other tile
TILE_M = 0.333         # the shipped tile size
TICKS = 48             # 2 s at 24 tps; today the gap passes 4 % by tick ~25
GAP_BOUND = 5 * SEED   # rms of (P_stored - P_state)/P_state, any tick


@pytest.mark.xfail(strict=True, reason=(
    "#217 finding 1 / #72: the EOS odd-even mode grows to ~4 % rms at "
    "0.333 m tiles; remedy is Erik's call (docs/eos_checkerboard_design_"
    "2026-11-20.md)"))
def test_checkerboard_seed_does_not_grow_in_still_sealed_air():
    sim = bench.build(TILE_M, "cb", SEED)
    pr = bench.Probe(sim)
    start = pr.row()
    # the seed went in (guards the fixture: a vacuous seed would pass)
    assert start["cbN"] == pytest.approx(SEED, rel=0.05)
    worst = 0.0
    for _ in range(TICKS):
        bench.step(sim)
        worst = max(worst, pr.row()["gap_rms"])
    assert np.isfinite(worst)
    assert worst <= GAP_BOUND, (
        f"state pressure and stored pressure drifted apart: worst rms gap "
        f"{100 * worst:.2f} % > {100 * GAP_BOUND:.2f} % (seed {100 * SEED:.2f} %)")
