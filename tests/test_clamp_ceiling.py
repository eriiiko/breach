"""P5d -- THE CLAMP'S CEILING on the ENGINE's Pass-1 fold (ray-engine-v2, issue
#12; Erik's ruling of 2026-09-24, docs/ray_engine_v2_p5d_clamp_headroom_brief_
2026-09-24.md §5.2).

The Pass-1 maximum-principle clamp bounds the radiative sub-step, on thermal
solids and on accountable gas cells alike, at

    T_new = min(T_after, max(T_before, e_ceiling_q(Phi)))

where e_ceiling_q(Phi) is the TOP of the first E° bucket whose emission exceeds
Phi (emissive_table.h; it was E°⁻¹(Phi), the low edge of Phi's own bucket).
The integer reference pins the arithmetic (sweep_ref_q_gates G5, G15 (c), G16)
and tests/test_temperature_gas_radiation.py holds the C++ fold to it bit for
bit; THIS file states the fold's three properties directly on the engine's
output, on both branches, so they are asserted where they live:

  (1) the radiative sub-step never carries a cell above max(T_before,
      e_ceiling_q(Phi)) -- gate 5 restated;
  (2) it never clips a COOLING step;
  (3) a cell already above its ceiling (a burning crate, a flame held hot by
      combustion) is never pulled down to it -- it cools by its own step, or is
      held at T_before.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_clamp_ceiling.py -q
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from _radiation_sweep_harness import R, live_table, reference_table  # noqa: E402
import sweep_ref_q_gates as G  # noqa: E402
from test_temperature_gas_radiation import (  # noqa: E402  (the direct-binding fold)
    RN_PHYS_MAX, _cpp_fold, _solver)


@pytest.mark.parametrize("table_name", ["live", "resolving"])
def test_the_engine_fold_clamps_at_the_headroom_ceiling_on_both_branches(table_name):
    """PROPERTY (P5d, brief §5.2; the ENGINE's fold, solid AND gas branch): on
    one Pass-1 fold of TemperatureSolver.step over random ACCOUNTABLE gas cells
    (row 0: a stored residual E mod N, bulk from the N_EPS edge to 3 atm,
    rad_net of both signs up to 2^44, fluences whose ceiling sits above and
    below the cell) and random thermal solids (row 1, thin rows included), with
    T_before and the unclamped T_after derived from the INPUTS (never from the
    engine) and the ceiling from the binding's e_ceiling_q:
      (1) every cell ends at or below max(T_before, e_ceiling_q(Phi));
      (2) every COOLING step lands exactly where it would with no clamp (on a
          solid, after the low rail at 0 -- a rail, not the clamp);
      (3) every cell that started ABOVE its ceiling ends at min(T_after,
          T_before): it cools by its own step or is held, never pulled down.
    NON-VACUOUS on both branches: the clamp binds and lands a cell EXACTLY ON
    e_ceiling_q(Phi); cooling steps and above-ceiling cells occur.

    BREAKS IF: the fold's ceiling is reverted to e_inv_q (no clamped cell lands
    on e_ceiling_q -- the non-vacuity count goes to 0 on both branches), the
    clamp is removed or placed after the booking ((1)), it clips a cooling step
    ((2)), or it loses its max(T_before, .) arm -- the bare v2 form ((3)).
    """
    live = table_name == "live"
    tbl = live_table() if live else reference_table()
    tref = R.E_LIVE if live else R.E
    rng = random.Random(20260924 + int(live))
    n = 600
    T, Eg, rn, phi, nb = G._gas_fold_cells(rng, n, tref)
    for i in range(n):                  # the floor counter's cells, at the physical bound
        if nb[0][i] < R.N_FLOOR_Q_LIVE and abs(rn[0][i]) > RN_PHYS_MAX:
            rn[0][i] = RN_PHYS_MAX if rn[0][i] > 0 else -RN_PHYS_MAX
    Ts = [rng.choice([0, 5 << 16, 290 << 16, 804 << 16, 1263 << 16, 5000 << 16])
          for _ in range(n)]
    hs = [rng.choice([-4, -3, -2, 0, 3, 5]) for _ in range(n)]
    rs = [rng.choice([1, -1]) * rng.choice([7, 1 << 14, 1 << 24, 1 << 34]) for _ in range(n)]
    ps = [tref[rng.choice([0, 1, 72, 201, 315, 1000, 3999])] + rng.choice([0, 1, 999])
          for _ in range(n)]
    T_all = np.asarray([T[0], Ts], dtype=np.int64)
    E_all = np.asarray([Eg[0], [0] * n], dtype=np.int64)
    rn_all = np.asarray([rn[0], rs], dtype=np.int64)
    phi_all = np.asarray([phi[0], ps], dtype=np.int64)
    nb_all = np.asarray([nb[0], [0] * n], dtype=np.int64)
    his_all = np.asarray([[0] * n, hs], dtype=np.int64)
    ts = np.zeros((2, n), dtype=bool)
    ts[1] = True
    Tc = np.ascontiguousarray(T_all.astype(np.int32))
    Ec = np.ascontiguousarray(E_all.copy())
    _cpp_fold(_solver(), Tc, Ec, rn_all, phi_all, nb_all.astype(np.int32), ts, his_all, tbl)

    seen = {m: dict(at_ceiling=0, cooling=0, above=0) for m in ("gas", "solid")}
    for row, medium in ((0, "gas"), (1, "solid")):
        for i in range(n):
            r = int(rn_all[row, i])
            if r == 0:
                continue
            if medium == "gas":
                N = int(nb_all[0, i])
                t0 = R.gas_mirror_q(int(E_all[0, i]), max(0, N))
                t_after = R.sat_add_q16(t0, R.gas_rad_dT_q(r, N))
                railed = t_after                    # the fold's gas side has no low rail
            else:
                t0 = int(T_all[1, i])
                t_after = R.sat_add_q16(t0, R.shr_round0_signed(r, int(his_all[1, i])))
                railed = max(t_after, 0)            # the solid low rail (a rail, counted)
            ceiling = int(tbl.e_ceiling_q(int(phi_all[row, i])))
            t_new = int(Tc[row, i])
            # (1) the maximum principle, at the headroom ceiling
            assert t_new <= max(t0, ceiling), (medium, i, t0, t_after, ceiling, t_new)
            # (2) a cooling step is never clipped
            if t_after <= t0:
                seen[medium]["cooling"] += 1
                assert t_new == railed, (medium, i, t0, t_after, t_new)
            # (3) a cell above its ceiling is never pulled down to it
            if t0 > ceiling:
                seen[medium]["above"] += 1
                want = min(t_after, t0)
                if medium == "solid":
                    want = max(want, 0)             # the solid low rail, as in (2)
                assert t_new == want, (medium, i, t0, t_after, ceiling, t_new)
            if t_after > max(t0, ceiling) and ceiling >= t0 and t_new == ceiling:
                seen[medium]["at_ceiling"] += 1
    for medium, s in seen.items():
        assert s["at_ceiling"] > 0, f"{medium}: no cell landed on the headroom ceiling: vacuous"
        assert s["cooling"] > 0 and s["above"] > 0, (medium, s)
    print(f"\n{table_name} table: {seen}")
