"""#78 -- THE SWEEP'S FINE HEAT CURRENCY, held on the ENGINE.

docs/sweep_fine_heat_currency_brief_78_2026-09-25.md. At the live scale a
room-temperature black body is only E°[0] = 125 heat counts per tick, so the
sweep's per-ordinate terms were single-digit integers and every floor lost up
to one count -- ~10 K of emission-equivalent near ambient. The ruled fix
(Erik, 2026-09-24; option 1 on #78) is SCALING: the engine's E° table is baked
2^E_FINE_BITS finer than one heat count (cpp/src/emissive_table.h), Phi, the
ambient level and the four sweep planes are all in that currency, and every
reader converts ONCE through the kit's fixedpoint::fine_heat_shr (the gas chain
at its final narrow). The integer reference specifies it first
(sweep_ref_q.py; gates G11 and G17); this file holds the ENGINE to the
properties of the brief's section 5 that need the engine to state:

  * the one constant is one currency everywhere the engine carries it (5, 6);
  * the Python twins of the kit's conversion are the kit's, value for value;
  * near-ambient cells cool radiatively -- the sweep books it and the fold
    lands it -- where the pre-#78 table books nothing (5.2);
  * the uniform ambient is still an exact per-cell fixed point and the
    sweep's identity is exact, in the fine currency (5.3);
  * the int64 headroom holds on an over-driven scene at the fine live scale,
    the engine agreeing with the reference's Python ints there (4, 5.7);
  * a marine beside a fire loses the same HP per second within 1 % (5.4).

Every test's docstring names its property and the change that breaks it; each
was validated by breaking the code once (noted per test).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_sweep_fine_currency.py -q
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "tests", ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _radiation_sweep_harness import (  # noqa: E402  (also sets up the study path)
    F_ONE, ONE, R, cpp_sweep, live_table, random_gas, random_scene, ref_sweep,
    ts_from_a)
import breach_physics as bp  # noqa: E402
import sweep_ref_q_gates as G  # noqa: E402
from simulation import optics_fixed as OF  # noqa: E402

Q = R.quant
K_LEAK = Q(0.10)


def _coarse_live_table():
    """The LIVE calibration in the pre-#78 currency (fine_bits = 0): the pair
    every "breaks with k = 0" property below is measured against."""
    tbl = bp.EmissiveTable()
    tbl.rad_scale = R.RAD_SCALE_LIVE
    tbl.kelvin_ambient = float(R.K_AMB)
    tbl.k_temp_to_kelvin = float(R.K_SLOPE)
    tbl.fine_bits = 0
    tbl.bake()
    return tbl


def _playground_sim():
    from level_loader import load as load_level
    from simulation import Simulation
    return Simulation(load_level("playground"), seed=1, breach_physics=bp,
                      enable_recorder=False)


# ---------------------------------------------------------------------------
# 1. one constant, one currency
# ---------------------------------------------------------------------------
def test_the_engine_runs_one_currency_the_reference_specifies():
    """PROPERTY: the engine's ONE constant is the reference's (E_FINE_BITS ==
    sweep_ref_q.FINE_BITS, > 0), and every place a live engine carries the
    currency carries THAT one: its emissive table is baked in it -- entry for
    entry sweep_ref_q.E_LIVE -- the vacuum ambient level PhysicsRunner hands the
    sweep is in it (the default 295 K lands on E°[0] integer for integer), and
    the GameMap reports its sweep planes in it (optics_fixed.sweep_fine_bits).

    BREAKS IF: the C++ constant and the reference's drift apart, the engine's
    table is baked in another currency than its default, or the vacuum level
    or the planes' reported currency stop following the table's (a vacuum
    level baked coarse would trip the sweep's ambient invariant, or read a
    fine table in whole counts). Validated: the vacuum level baked without the
    2^k -- the sweep then radiates every vacuum cell against a level 2^11 below
    E°[0], i.e. a near-0-K sky, and the equality below goes red.
    """
    assert bp.E_FINE_BITS == R.FINE_BITS > 0
    sim = _playground_sim()
    eng = sim.physics_runner.engine
    assert int(eng.emissive.fine_bits) == bp.E_FINE_BITS
    table = np.asarray(eng.emissive.table(), dtype=np.int64)
    assert np.array_equal(table, np.asarray(R.E_LIVE, dtype=np.int64))
    assert int(sim.physics_runner.rad_amb_vacuum_q) == int(table[0])
    assert OF.sweep_fine_bits(sim.gmap) == bp.E_FINE_BITS
    # the currency follows the table: flip it, and the level follows (the pair)
    eng.emissive.fine_bits = 0
    coarse0 = int(np.asarray(eng.emissive.table(), dtype=np.int64)[0])
    assert int(sim.physics_runner.rad_amb_vacuum_q) == coarse0 == R.bake_e_table(
        rad_scale=R.RAD_SCALE_LIVE)[0]
    assert OF.sweep_fine_bits(sim.gmap) == 0


# ---------------------------------------------------------------------------
# 2. the Python twins ARE the kit's conversion
# ---------------------------------------------------------------------------
def test_the_python_twins_equal_the_kits_one_conversion():
    """PROPERTY: the kit's ONE conversion out of the fine currency,
    fixedpoint::fine_heat_shr, equals its two Python twins --
    sweep_ref_q.fine_heat_shr (the spec) and optics_fixed.fine_heat_shr (the sim
    path's, scalar and array, a per-cell shift included) -- value for value, over
    both signs, zero, the int64 range the planes reach, and every shift class
    (a thin row's negative his, whole counts, the counters' -16, a heavy row);
    it is SYMMETRIC (f(-x) == -f(x)); at k = 0 it IS shr_round0_signed_i64. The
    gas chain converts at its final narrow: fp_deposit_dT_wide_i64(.., k) is
    the reference's, and equals the coarse chain floored by 2^k (nested floors).

    BREAKS IF: a twin drifts from the kit, the conversion becomes one-sided (a
    plain >> would round -x away from zero), or the gas chain converts anywhere
    but its final narrow (a whole-count rounding first breaks the nested-floor
    identity). Validated: optics_fixed.fine_heat_shr with a plain >> fails the
    negative cases.
    """
    rng = random.Random(78)
    xs = [0, 1, -1, 2047, -2047, 2048, -2048, 4097, -4097, 1 << 30, -(1 << 30),
          (1 << 41) + 12345, -((1 << 41) + 12345)]
    xs += [rng.randint(-(1 << 44), 1 << 44) for _ in range(300)]
    for k in (0, 1, R.FINE_BITS):
        for s in (-16, -5, -3, 0, 3, 5):
            arr = OF.fine_heat_shr(np.asarray(xs, dtype=np.int64),
                                   np.full(len(xs), s), k)
            for x, a in zip(xs, arr):
                want = bp.fp_fine_heat_shr(x, s, k)
                assert R.fine_heat_shr(x, s, k) == want, (x, s, k)
                assert OF.fine_heat_shr(x, s, k) == want, (x, s, k)
                assert int(a) == want, (x, s, k)
                assert bp.fp_fine_heat_shr(-x, s, k) == -want, (x, s, k)
                if k == 0:
                    assert want == bp.fp_shr_round0_signed_i64(x, s)
    recips = [(bp.fp_reciprocal_q16(n), bp.fp_make_recip(c)) for n in (3, 655, ONE, 3 * ONE)
              for c in (R.C_V_Q_LIVE / 65536.0, 0.5, 1.0)]
    for dep in [0, 1, 7, 2048, 2049, 1 << 20, (1 << 41) + 99]:
        for rn, rc in recips:
            coarse = bp.fp_deposit_dT_wide_i64(dep, rn, rc)
            fine = bp.fp_deposit_dT_wide_i64(dep, rn, rc, R.FINE_BITS)
            assert fine == R.deposit_dT_wide_i64(dep, rn, rc, R.FINE_BITS)
            assert fine == coarse >> R.FINE_BITS, (dep, rn, rc)


# ---------------------------------------------------------------------------
# 3. near-ambient cells cool radiatively (brief 5.2, the engine's half)
# ---------------------------------------------------------------------------
def _fold_solid(T, rn, phi, his, table):
    """One Pass-1 fold of the C++ solver's direct binding on one row of thermal
    solids, the clamp on. Returns the new temperatures."""
    from test_temperature_gas_radiation import _cpp_fold, _solver
    n = len(T)
    Tc = np.ascontiguousarray(np.asarray([T], dtype=np.int32))
    Ec = np.zeros((1, n), dtype=np.int64)
    ts = np.ones((1, n), dtype=bool)
    _cpp_fold(_solver(), Tc, Ec, np.asarray([rn], dtype=np.int64),
              np.asarray([phi], dtype=np.int64), np.zeros((1, n), dtype=np.int32),
              ts, np.asarray([his], dtype=np.int64), table)
    return Tc[0].astype(np.int64)


def test_near_ambient_cells_cool_radiatively_in_the_engine():
    """PROPERTY (brief 5.2): in the ENGINE, one cell of every shipped absorbing
    row at a temperature in bucket 1 or 2 (4 .. 12 game), alone in a
    transparent room of ambient air, books rad_net < 0 on the live (fine)
    table -- both transports, S16 and S12, the leak on and off -- and a cell at
    or below ambient books exactly 0 on every plane. The fold LANDS it: every
    thin row (his < 0) that booked it cools in one Pass-1 fold. The PAIR: the
    same scenes on the pre-#78 table (the live scale in whole heat counts) book
    rad_net == 0 on every S16 case, and the fold does not move the thin rows --
    Croci & Giles' stagnation.

    BREAKS IF: the engine's table is baked coarse (k = 0): every assertion on
    the live table then fails exactly as its pair does. Validated by running the
    live half on _coarse_live_table(): red.
    """
    fine, coarse = live_table(), _coarse_live_table()
    temps = [4 << 16, 6 << 16, 8 << 16, 10 << 16, (12 << 16) - 1]
    n_case = n_s16 = n_s16_zero = 0
    for a_q, his, _nm in G.SHIPPED_ROWS:
        for transport in ("shear", "step"):
            for n_ord in (16, 12):
                for kq in (0, K_LEAK):
                    for T_q in temps:
                        a, d, _k, T, (cy, cx) = G._lone_cell_scene(7, a_q, T_q, kq)
                        ts = ts_from_a(a)
                        rn = cpp_sweep(a, d, kq, T, his, ts, transport=transport,
                                       n_ord=n_ord, table=fine)[0]
                        rc = cpp_sweep(a, d, kq, T, his, ts, transport=transport,
                                       n_ord=n_ord, table=coarse)[0]
                        assert rn[cy, cx] < 0, (a_q, his, transport, n_ord, kq, T_q)
                        n_case += 1
                        if n_ord == 16:
                            n_s16 += 1
                            n_s16_zero += int(rc[cy, cx] == 0)
                    for T_q in (0, -(4 << 16)):
                        a, d, _k, T, _c = G._lone_cell_scene(7, a_q, T_q, kq)
                        for tbl in (fine, coarse):
                            out = cpp_sweep(a, d, kq, T, his, ts_from_a(a),
                                            transport=transport, n_ord=n_ord, table=tbl)
                            assert all(not np.any(p) for p in out[:3])
    assert n_case > 0 and n_s16_zero == n_s16 > 0, (n_s16_zero, n_s16)
    # the fold lands it on the thin rows, and the pre-#78 table does not
    thin = [(a_q, his) for a_q, his, _nm in G.SHIPPED_ROWS if his < 0]
    assert thin
    for a_q, his in thin:
        for tbl, want_cool in ((fine, True), (coarse, False)):
            a, d, _k, T, (cy, cx) = G._lone_cell_scene(7, a_q, 10 << 16, K_LEAK)
            rn, _rf, _ra, rl, _f, _s = cpp_sweep(a, d, K_LEAK, T, his, ts_from_a(a),
                                                 table=tbl)
            t_new = _fold_solid([T[cy][cx]], [int(rn[cy, cx])], [int(rl[cy, cx])],
                                [his], tbl)
            if want_cool:
                assert t_new[0] < T[cy][cx], (a_q, his, int(rn[cy, cx]))
            else:
                assert t_new[0] == T[cy][cx], (a_q, his, int(rn[cy, cx]))


# ---------------------------------------------------------------------------
# 4. the fixed point and the identity, in the fine currency (brief 5.3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("transport", ["shear", "step"])
@pytest.mark.parametrize("n_ord", [16, 12])
def test_uniform_ambient_and_the_identity_are_exact_in_the_fine_currency(transport, n_ord):
    """PROPERTY (brief 5.3): on the FINE live table the engine's sweep keeps the
    uniform ambient an exact PER-CELL fixed point -- gate 2a's scene: non-dyadic
    a, bodies, the leak on, with and without absorbing smoke -- every cell of
    all three planes exactly 0; and on randomised fine scenes (hot, sub-ambient,
    table-top cells, bodies, smoke) Sum(rad_net) + Sum(rad_flux) + Sum(rad_amb)
    == 0 exactly in int64, each sum non-zero.

    BREAKS IF: absorption and emission are formed in two currencies (an ambient
    cell then books the difference), the per-cell ambient stops being the
    table's own E°[0] in the fine currency, or any writer books one side of a
    transfer without the other. Validated, and kept as the test's own pair: a
    coarse ambient level handed to a fine table (amb = E°[0] >> 11) breaks the
    fixed point.
    """
    fine = live_table()
    rng = random.Random(4)
    h, w = 8, 10
    a = [[rng.choice([0, Q(0.37), Q(0.91), ONE]) for _ in range(w)] for _ in range(h)]
    d = [[min(ONE, a[y][x] + rng.choice([0, Q(0.5), ONE - a[y][x]]))
          for x in range(w)] for y in range(h)]
    T = R.plane(h, w, 0)
    ts = ts_from_a(a)
    gas, hq, nb = random_gas(random.Random(40), h, w)
    for kw in ({}, dict(gas=gas, hq=hq, n_bulk=nb)):
        out = cpp_sweep(a, d, K_LEAK, T, 3, ts, transport=transport, n_ord=n_ord,
                        table=fine, **kw)
        assert all(not np.any(p) for p in out[:3]), kw.keys()
    # the pair: a coarse ambient level on the fine table is NOT ambient
    out = cpp_sweep(a, d, K_LEAK, T, 3, ts, transport=transport, n_ord=n_ord,
                    table=fine, amb=int(R.E_LIVE[0]) >> R.FINE_BITS)
    assert np.any(out[0])
    rng1 = random.Random(20260925 + n_ord)
    for trial in range(3):
        aa, dd, TT, his, tss = random_scene(rng1, 7, 9)
        kw = {} if trial == 0 else dict(zip(("gas", "hq", "n_bulk"), random_gas(rng1, 7, 9)))
        rn, rf, ra, _rl, _f, _s = cpp_sweep(aa, dd, K_LEAK, TT, his, tss,
                                            transport=transport, n_ord=n_ord,
                                            table=fine, **kw)
        sums = [int(p.astype(object).sum()) for p in (rn, rf, ra)]
        assert sum(sums) == 0 and all(s != 0 for s in sums), sums


# ---------------------------------------------------------------------------
# 5. headroom on an over-driven scene (brief section 4 / 5.7)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("transport", ["shear", "step"])
@pytest.mark.parametrize("n_ord", [16, 12])
@pytest.mark.parametrize("k_q", [0, K_LEAK])
def test_the_fine_live_scale_keeps_int64_headroom_over_driven(transport, n_ord, k_q):
    """PROPERTY (brief section 4): on the table the ENGINE bakes -- the live
    calibration in its fine currency -- an over-driven scene (a = 1 everywhere at
    the table top, the Fleck damping undamped: the widest source there is; and
    three table-top emitters in a transparent room under a 0-K sky: the stream
    travels) stays inside design v3's int64 bounds: every per-cell sum
    (|rad_net|, rad_fluence, |rad_amb|) below 2^46, every plain int64 product of
    the sweep loop and the pre-pass's a * ex below 2^58 (>= 2^5 of margin), the
    128-bit Fleck product below 2^63 -- measured by the reference in Python ints
    -- and the engine's int64 sweep EQUALS the reference bit for bit there, so
    nothing in it wrapped. The fold's conversion of that rad_net at the thinnest
    representable row (his = -16, an exact left shift by 16 - k) and at a heavy
    one is the reference's too.

    BREAKS IF: E_FINE_BITS grows past what the bounds allow (measured on the
    reference: at 12 the pre-pass's a * ex is 2^58.1, at 13 the Fleck product
    2^63.5), or a conversion moves to where it overflows. Validated: with
    E_FINE_BITS and the reference's FINE_BITS both set to 12 the pre-pass's
    a * ex bound goes red (65536 x E°[top] = 2^58.1).
    """
    fine = live_table()
    tref = R.E_LIVE
    h, w = 9, 11
    top = R.T_TABLE_TOP_GAME << 16
    scenes = []
    a = R.plane(h, w, ONE)
    scenes.append((a, [r[:] for r in a], R.plane(h, w, top), None))
    a2 = R.plane(h, w, 0)
    T2 = R.plane(h, w, 0)
    for (y, x) in ((1, 1), (h - 2, w - 2), (h // 2, 1)):
        a2[y][x] = ONE
        T2[y][x] = top
    scenes.append((a2, [r[:] for r in a2], T2, 0))
    for a_s, d_s, T_s, amb in scenes:
        ts = ts_from_a(a_s)
        got = cpp_sweep(a_s, d_s, k_q, T_s, 3, ts, transport=transport, n_ord=n_ord,
                        table=fine, fleck=False, amb=amb)
        k = R.plane(h, w, k_q)
        res = R.sweep_q(a_s, d_s, k, T_s, n_ord=n_ord, transport=transport,
                        table=tref, e_ref=amb)
        for g, e in zip(got[:4], (res.rad_net, res.rad_flux, res.rad_amb, res.rad_fluence)):
            assert np.array_equal(np.asarray(g, dtype=np.int64),
                                  np.asarray(e, dtype=np.int64))
        assert max(res.max_abs_net, res.max_fluence, R.plane_max_abs(res.rad_amb)) < 2 ** 46
        assert res.max_plain_product < 2 ** 58 and res.max_fleck_product < 2 ** 63
        # the fold's conversions of that rad_net, at the widest shifts
        rn_row = [int(v) for v in np.asarray(got[0]).ravel()[:32]]
        phi_row = [int(v) for v in np.asarray(got[3]).ravel()[:32]]
        for his in (-16, 5):
            t0 = [0] * len(rn_row)
            t_new = _fold_solid(t0, rn_row, phi_row, [his] * len(rn_row), fine)
            T_ref = [t0[:]]
            R.fold_pass1_solid(T_ref, [rn_row], [phi_row], [[his] * len(rn_row)],
                               [[1] * len(rn_row)], R.FoldCounters(), table=tref)
            assert [int(v) for v in t_new] == T_ref[0], his
    assert ONE * tref[-1] < 2 ** 58                    # the pre-pass's a * ex, zero sky


# ---------------------------------------------------------------------------
# 6. a marine burns exactly as before (brief 5.4)
# ---------------------------------------------------------------------------
def _marine_hp_loss(fine_bits, ticks):
    """HP a marine loses over `ticks` beside the s3c burner (a held 443-game wood
    patch one tile away), with the engine's table in the given currency."""
    import test_s3c_unit_state_digest as S
    from simulation import fire_fixed
    sim = S._fire_kill_sim()
    runner = sim.physics_runner
    runner.engine.emissive.fine_bits = int(fine_bits)
    assert OF.sweep_fine_bits(sim.gmap) == int(fine_bits)
    g = sim.gmap
    u = sim.units[0]
    u.current_hp = 1.0e6
    seed_q = fire_fixed.quantize_scalar(0.7)
    flame_q = fire_fixed.quantize_scalar(443.0)
    hp0 = float(u.current_hp)
    flux = []
    for _ in range(ticks):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y, x = S.FY + dy, S.FX + dx
                if 1 <= y < 15 and 1 <= x < 15:
                    g.fire[y, x] = max(int(g.fire[y, x]), seed_q)
                    g.temperature[y, x] = max(int(g.temperature[y, x]), flame_q)
        sim.set_paused(False)
        sim.step()
        flux.append(OF.fine_heat_shr(int(g.rad_flux_sweep[S.UNIT_Y, S.UNIT_X]), 0,
                                     OF.sweep_fine_bits(g)))
    return hp0 - float(u.current_hp), flux


def test_a_marine_beside_a_fire_loses_the_same_hp_per_second():
    """PROPERTY (brief 5.4): k buys precision, not physics. A marine one tile
    from a held 443-game fire loses the same HP per second -- within 1 % -- with
    the engine's table in the fine currency as with the pre-#78 table: the unit
    coupling (exchange.py) reads the sweep's body channel through the kit's twin
    at s = 0, i.e. in whole heat counts, so the fine bits refine the exchange
    without scaling it. Non-vacuous: the marine really burns, and the sensor it
    reads really is the sweep's (non-zero every tick).

    BREAKS IF: exchange.py reads rad_flux_sweep without converting (the fine
    plane read as heat counts: 2^11 x the exposure), converts twice, or the
    sweep's body channel stops being the heat the marine feels. Validated:
    exchange.py with the conversion removed -- the fine run loses HP many times
    faster and this goes red.
    """
    ticks = 72
    fine_loss, fine_flux = _marine_hp_loss(bp.E_FINE_BITS, ticks)
    coarse_loss, coarse_flux = _marine_hp_loss(0, ticks)
    assert coarse_loss > 0 and fine_loss > 0
    assert all(v > 0 for v in fine_flux) and all(v > 0 for v in coarse_flux)
    rel = abs(fine_loss - coarse_loss) / coarse_loss
    print(f"\nmarine HP lost over {ticks} ticks: fine {fine_loss:.6f}, pre-#78 "
          f"{coarse_loss:.6f} ({rel * 100:.3f} %); sensor (heat counts) fine "
          f"{fine_flux[-1]} vs {coarse_flux[-1]}")
    assert rel <= 0.01, (fine_loss, coarse_loss, rel)
