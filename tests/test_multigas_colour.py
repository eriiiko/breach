"""Multi-gas colour on the LIGHT SWEEP -- mixing is a density-weighted sum.

Rewritten at ray-engine-v2 P6c (design v3 §11.2 row "test_multigas_colour.py:
rewrite 4, delete 4"). Until P6c this file cast the old C++ render march
through coloured gas and checked its float Beer-Lambert ``exp`` identities;
the march is deleted. The four properties that still mean something are now
properties of the sweep's INTEGER per-gas light extinction term (design §4.1,
§6.3: ``a_c = min(ONE, Σ_g light_absorb_q16[g][c] · N_g >> 16)`` and the glow
coefficient ``g_c`` likewise over ``light_glow_q16``):

  * a gas tints the light by its per-channel absorption, and
  * a dense grey soot dims every channel -- both held on the engine by
    tests/test_radiation_sweep_light.py::test_smoke_tints_by_its_rgb_absorption_
    and_dims_with_density (P6a);
  * two gases sharing a cell MIX as the density-weighted SUM of their terms,
    per channel, for extinction and glow alike -- here;
  * an empty gas stack is optically nothing: no extinction, no glow, the same
    light field as no gas group at all -- here.

Every test's docstring names its property and the change that breaks it.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_multigas_colour.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from _radiation_sweep_harness import ONE, R, cpp_sweep, live_table  # noqa: E402

Q = R.quant
N = 9            # a 9 x 9 transparent room
SRC_T = 3000     # a hot column at x = 0 (all three channels emit)
O2_Q, N2_Q = 13763, 51773   # an ambient air cell's bulk pair


def _run(cells, table):
    """The engine's sweep on the room, the gas planes filled per `cells`
    ({(y, x): {gas_id: density_q}}; None = no gas group at all), the SHIPPED
    light columns; returns (light_q, light_glow, a_eff (3, h, w), gcoef
    (3, h, w))."""
    from simulation.gases import N_GASES, O2, INERT_N2
    a = [[0] * N for _ in range(N)]
    d = [[0] * N for _ in range(N)]
    T = [[0] * N for _ in range(N)]
    his = [[3] * N for _ in range(N)]
    ts = [[0] * N for _ in range(N)]
    la = [[[0] * N for _ in range(N)] for _ in range(3)]
    ld = [[[0] * N for _ in range(N)] for _ in range(3)]
    for y in range(N):
        T[y][0] = SRC_T << 16
        ts[y][0] = 1
        for c in range(3):
            la[c][y][0] = ld[c][y][0] = ONE
    light = dict(la=la, ld=ld)
    kw = {}
    if cells is not None:
        planes = [[[0] * N for _ in range(N)] for _ in range(N_GASES)]
        for y in range(N):
            for x in range(N):
                planes[O2][y][x] = O2_Q
                planes[INERT_N2][y][x] = N2_Q
        for (y, x), dens in cells.items():
            for g, q in dens.items():
                planes[g][y][x] = q
        nb = [[planes[O2][y][x] + planes[INERT_N2][y][x] for x in range(N)]
              for y in range(N)]
        light.update(light_absorb_q=table.light_absorb_q16.tolist(),
                     light_glow_q=table.light_glow_q16.tolist())
        kw = dict(gas=planes, hq=[0] * N_GASES, n_bulk=nb)
    got = cpp_sweep(a, d, 0, T, his, ts, transport="shear", n_ord=16,
                    table=live_table(), light=light, **kw)
    sweep, out = got[5], got[6]
    a_eff = np.moveaxis(np.asarray(sweep.light_a_eff_plane()), -1, 0)
    gco = np.moveaxis(np.asarray(sweep.light_gcoef_plane()), -1, 0)
    return out["light_q"], out["light_glow"], a_eff, gco


def test_two_gases_in_one_cell_mix_as_the_density_weighted_sum():
    """PROPERTY (design §6.3 "mixing falls out of the sum", on the light
    channels): with the SHIPPED light columns, a cell holding poison at dP AND
    soot at dS reads, per channel, the light extinction
    ONE - exp(-((absP_c·dP + absS_c·dS) >> 16)) -- the ONE density-weighted
    sum, shifted once, through the light's Beer-Lambert law (#12 handle 2;
    the reference's exp_neg_q16) -- which is MORE than either gas alone gives, and its glow
    coefficient is the same sum over the glow column; the same two gases in
    two separate cells each read only their own term.

    BREAKS IF: the smoke term takes the MAX over gases instead of the sum,
    shifts per gas, drops a channel, or mixes the absorption and glow columns.
    """
    from simulation.gases import GasTable, POISON, SMOKE
    tbl = GasTable.from_config()
    lab, lgl = tbl.light_absorb_q16, tbl.light_glow_q16
    # Densities chosen FROM the table so each gas alone reaches ~30 % of ONE in
    # its strongest channel: the sum stays below the cap whatever the shipped
    # coefficients are (soot's derived light_absorb is ~100x the old dial
    # product -- #12 handle 1 -- and a fixed density would cap).
    dP = (3 * ONE * ONE) // (10 * int(max(lab[POISON])))
    dS = (3 * ONE * ONE) // (10 * int(max(lab[SMOKE])))
    y, xp, xs, xm = 4, 3, 5, 7
    _lq, _gl, a_eff, gco = _run({(y, xp): {POISON: dP}, (y, xs): {SMOKE: dS},
                                 (y, xm): {POISON: dP, SMOKE: dS}}, tbl)
    for c in range(3):
        want_a = ONE - R.exp_neg_q16((int(lab[POISON][c]) * dP + int(lab[SMOKE][c]) * dS) >> 16)
        want_g = min(ONE, (int(lgl[POISON][c]) * dP + int(lgl[SMOKE][c]) * dS) >> 16)
        assert want_a < ONE, "the scene must stay below the cap to test the sum"
        assert int(a_eff[c, y, xm]) == want_a, (c, int(a_eff[c, y, xm]), want_a)
        assert int(gco[c, y, xm]) == want_g, (c, int(gco[c, y, xm]), want_g)
        assert int(a_eff[c, y, xp]) == ONE - R.exp_neg_q16((int(lab[POISON][c]) * dP) >> 16)
        assert int(a_eff[c, y, xs]) == ONE - R.exp_neg_q16((int(lab[SMOKE][c]) * dS) >> 16)
        assert int(a_eff[c, y, xm]) > max(int(a_eff[c, y, xp]), int(a_eff[c, y, xs]))


def test_an_empty_gas_stack_is_optically_nothing():
    """PROPERTY (the old "empty gas leaves light untouched and no glow", on the
    sweep): a gas group whose trace planes are all zero -- ambient air only --
    gives every gas cell zero light extinction and zero glow, and the light
    field is EXACTLY the one computed with no gas group at all. PAIR: one thin
    soot cell in the same room does absorb and glow.

    BREAKS IF: the bulk pair (O2, N2) carries optics, the smoke term reads a
    nonzero floor, or the glow is computed where there is no absorber.
    """
    from simulation.gases import GasTable, SMOKE
    tbl = GasTable.from_config()
    lq_none, gl_none, _a0, _g0 = _run(None, tbl)
    lq_air, gl_air, a_air, g_air = _run({}, tbl)
    assert np.array_equal(lq_air, lq_none), "empty gas changed the light"
    assert not np.any(gl_air) and not np.any(gl_none)
    assert not np.any(a_air[:, :, 1:]) and not np.any(g_air)
    assert np.any(lq_air[:, :, 1:] > 0), "the room must be lit for this to mean anything"
    lq_s, gl_s, a_s, g_s = _run({(4, 4): {SMOKE: Q(0.05)}}, tbl)
    assert np.all(a_s[:, 4, 4] > 0) and np.all(gl_s[:, 4, 4] > 0)
