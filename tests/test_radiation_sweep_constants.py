"""The sweep's checked-in per-ordinate constants (ray-engine-v2 P1, design v3
§2.3 / §8.1, critique 3 §5b).

`s_m` (the major share of the downwind split), `sx`/`sy` (the direction signs)
and `x_major` are INTEGER LITERALS in cpp/src/radiation_sweep.cpp for S16 and
S12, both transports — never a libm cos/sin at load in a sim TU. This test
recomputes them from math.cos/math.sin exactly as the integer reference does
and asserts the literals within one count.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_radiation_sweep_constants.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from _radiation_sweep_harness import R, TRANSPORTS  # noqa: E402


@pytest.mark.parametrize("n_ord", [16, 12])
@pytest.mark.parametrize("transport", ["shear", "step"])
def test_literals_match_a_recompute_within_one_count(n_ord, transport):
    """PROPERTY: for every ordinate the checked-in (sx, sy, x_major, s_m) equals
    the reference's ordinate_constants() on this machine's libm — sx/sy exactly,
    s_m within one Q16 count, x_major exactly except on the S12 diagonal
    ordinates where |mu| and |eta| differ only in the last ULP (there s_m is 0
    and the sweep's integers do not depend on x_major).

    BREAKS IF: a literal is mistyped, the half-offset changes, or the shear/step
    share formula changes (s_m would move by thousands of counts).
    """
    got = bp.RadiationSweep.ordinate_constants(n_ord, TRANSPORTS[transport])
    assert len(got) == n_ord
    for m, (mu, eta) in enumerate(R.ordinates(n_ord)):
        x_major, s_m, sx, sy = R.ordinate_constants(mu, eta, transport)
        gsx, gsy, gxm, gsm = got[m]
        assert (gsx, gsy) == (sx, sy), (transport, n_ord, m)
        assert abs(gsm - s_m) <= 1, (transport, n_ord, m, gsm, s_m)
        degenerate = abs(abs(mu) - abs(eta)) < 1e-9
        if degenerate and transport == "shear":
            # the S12 diagonal under shear: the minor/major ratio is 1 up to the
            # last ULP, so the major share is 0 and x_major is libm's coin toss
            assert gsm == 0 and s_m == 0
        else:
            # step's diagonal share is |mu|/(|mu|+|eta|) == 1/2 exactly (32768)
            # and step is always x-major; nothing is degenerate there
            assert bool(gxm) == bool(x_major), (transport, n_ord, m)
            if degenerate:
                assert gsm == 32768


def test_constants_are_not_degenerate_and_differ_by_transport():
    """NON-VACUITY: the shear and step tables are different objects (the shares
    differ), every shear share is in (0, ONE], and the four quadrants are all
    present — so the recompute test above compares real numbers.

    BREAKS IF: one table is a copy of the other or an ordinate set collapses.
    """
    shear = bp.RadiationSweep.ordinate_constants(16, bp.RadiationSweep.SHEAR)
    step = bp.RadiationSweep.ordinate_constants(16, bp.RadiationSweep.STEP)
    assert shear != step
    assert {(r[0], r[1]) for r in shear} == {(1, 1), (-1, 1), (-1, -1), (1, -1)}
    assert all(0 < r[3] <= R.ONE for r in shear)
    assert all(0 < r[3] < R.ONE for r in step)
    # a share that is not a multiple of 4096 (the S16 weight) — the transport
    # constants are cosine ratios, not dyadic weights
    assert any(r[3] % 4096 != 0 for r in shear)
    with pytest.raises(ValueError):
        bp.RadiationSweep.ordinate_constants(8, bp.RadiationSweep.SHEAR)


def test_half_offset_keeps_every_ordinate_off_the_axes():
    """PROPERTY (design §2.5): no S16 ordinate lies on an axis, so no step
    ordinate degenerates into a pencil (s_m in {0, ONE}).

    BREAKS IF: the ordinate set loses its half-offset.
    """
    for mu, eta in R.ordinates(16):
        assert abs(mu) > 1e-6 and abs(eta) > 1e-6
        assert abs(abs(mu) - abs(eta)) > 1e-6
    assert math.isclose(sum(abs(mu) for mu, _ in R.ordinates(16)), 16 * 2 / math.pi, rel_tol=0.05)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
