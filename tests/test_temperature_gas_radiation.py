"""P5c -- SMOKE'S HEAT RADIATION, LIVE: the temperature fold's GAS branch
(ray-engine-v2, issue #12; design v3 §2.8 / §6.3 / §8.4).

The sweep has booked a gas cell's rad_net since P5a (the smoke term) and damped
its emission since P5b (the gas Fleck arm); P5c opens the Pass-1 fold to it. An
ACCOUNTABLE gas cell with rad_net != 0 takes it into the conserved field:

    dT       = sign(rn) * deposit_dT_wide_i64(|rn|, recip_N, recip_cv)
    T_before = mirror_q(E, N)
    T_target = min(sat(T_before + dT), max(T_before, E°⁻¹(Φ)))
    dE       = N * (T_target - T_before)  -> deposit_railed, e_gas_deposit_sum

and the clamp's withheld step is counted, on gas AND solids, in
TemperatureSolver.e_rad_clamp_drop_sum. The arithmetic is the integer
reference's (sweep_ref_q.fold_pass1_gas, gate G15); this file holds the ENGINE
to it and pins the properties the patch exists for, each named with the change
that must break it:

  (0) the C++ fold equals the reference, bit for bit, cell and counter -- one
      tick on random cells, and tick for tick with the C++ sweep in the loop;
  (a) the #54 closure identity and the P-G5 total ledger still close in int64
      with gas radiation live -- no new group;
  (b) smoke SHIELDS, in the live engine, against a clear-air control;
  (c) hot smoke COOLS RADIATIVELY in the live engine, never below ambient;
  (d) a smoke-free scene is bit-identical to the pre-P5c fold -- the branch acts
      only where a_gas > 0 (via rad_net != 0);
  (f) design §8.4's sweep->fold boundary holds on the engine's own counters.

(e), CPU == CUDA at tol 0 for the temperature twin with gas radiation, is
tests/cuda_temperature_gas_radiation_check.py (tests/
test_cuda_temperature_gas_radiation.py runs it through cuda_harness).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_temperature_gas_radiation.py -q
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools", ROOT / "tests",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _radiation_sweep_harness import (  # noqa: E402  (also sets up the study path)
    ONE, R, cpp_sweep, live_table, reference_table)
import breach_physics as bp  # noqa: E402
import sweep_ref_q_gates as G  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation, gas_fixed  # noqa: E402
from simulation.gases import SMOKE  # noqa: E402
from simulation.materials import MaterialTable  # noqa: E402

AIR, HULL = 0, 1
T_AMB_Q = R.T_AMB_Q
COUNTERS = ("t_max_phys_hits", "t_low_rail_hits", "rad_clamp_hits",
            "e_rad_clamp_drop_sum", "e_gas_deposit_sum", "e_gas_rail_sum",
            "e_solid_deposit_sum")


# ---------------------------------------------------------------------------
# the direct-binding fold (conduction off: every face NO_FACE; heat 0)
# ---------------------------------------------------------------------------
def _solver():
    s = bp.TemperatureSolver()
    s.no_face = int(MaterialTable.from_config().no_face)
    s.c_v = R.C_V_LIVE
    s.n_floor_heat = R.N_FLOOR_HEAT_LIVE
    return s


def _counters(s):
    return {k: int(getattr(s, k)) for k in COUNTERS}


def _cpp_fold(solver, T, E, rn, phi, nb, ts, his, table, *, solid=None, vac=None,
              ring=None, clamp=True):
    """One Pass-1 fold through TemperatureSolver.step's direct binding, energy
    form. Every plane is a (h, w) numpy array; T and E are MUTATED. Returns the
    counter DELTAS of this call."""
    h, w = T.shape
    face = np.full((h, w, 4), solver.no_face, dtype=np.int32)
    solid = ts.copy() if solid is None else solid
    vac = np.zeros((h, w), dtype=bool) if vac is None else vac
    before = _counters(solver)
    solver.step(T, np.zeros((h, w), np.int32), np.ascontiguousarray(his, dtype=np.int32),
                face, np.ascontiguousarray(solid), np.ascontiguousarray(vac),
                np.full((h, w), ONE, np.int32),
                n_bulk=np.ascontiguousarray(nb, dtype=np.int32),
                thermal_solid=np.ascontiguousarray(ts),
                rad_net=np.ascontiguousarray(rn, dtype=np.int64),
                rad_fluence=np.ascontiguousarray(phi, dtype=np.int64), e_table=table,
                clamp_enabled=clamp, gas_energy=E, t_amb_q=T_AMB_Q,
                is_ambient=None if ring is None else np.ascontiguousarray(ring))
    after = _counters(solver)
    return {k: after[k] - before[k] for k in COUNTERS}


# ---------------------------------------------------------------------------
# (0) the engine's fold IS the reference's
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table_name", ["live", "resolving"])
@pytest.mark.parametrize("clamp", [True, False])
def test_cpp_fold_equals_the_reference_on_gas_and_solids_bit_for_bit(table_name, clamp):
    """PROPERTY (gate 0 across the fold, P5c): one Pass-1 fold of the C++
    TemperatureSolver on a grid carrying (row 0) random ACCOUNTABLE gas cells --
    a stored residual E mod N, bulk from the N_EPS edge through the n_floor
    density to 3 atm, rad_net of both signs to 2^44, the cap above and below
    each -- (row 1) random thermal solids, including thin rows (negative
    heat_inv_shift), and (row 2) NON-accountable gas cells with rad_net != 0 --
    vacuum and the ambient ring -- equals sweep_ref_q.fold_pass1_solid +
    fold_pass1_gas on the same planes: every temperature, every stored energy,
    and all seven counters, with the clamp on and off, on the live and the
    resolving table. Row 2 is untouched on both. Non-vacuous: the clamp binds on
    some gas cells and not on others, and on some solids.

    BREAKS IF: temperature_solver.cpp's gas branch differs from the reference
    in any step -- the staged chain or its floor, the mirror as T_before, the
    energy form of the clamp, the rail -- or reaches a non-accountable cell, or
    the drop is priced at anything but cap_real.
    """
    live = table_name == "live"
    tbl = live_table() if live else reference_table()
    tref = R.E_LIVE if live else R.E
    rng = random.Random(20260925 + int(clamp) + 2 * int(live))
    n = 600
    T, Eg, rn, phi, nb = G._gas_fold_cells(rng, n, tref)
    # row 1: thermal solids; row 2: vacuum / ring gas cells (non-accountable)
    Ts = [rng.choice([0, 290 << 16, 804 << 16, 1263 << 16]) for _ in range(n)]
    hs = [rng.choice([-4, -2, 0, 3, 5]) for _ in range(n)]
    rs = [rng.choice([1, -1]) * rng.choice([7, 1 << 14, 1 << 24, 1 << 34]) for _ in range(n)]
    ps = [tref[rng.choice([0, 72, 201, 315, 1000])] for _ in range(n)]
    T3 = [(300 << 16)] * n
    E3 = [ONE * ((300 << 16) + T_AMB_Q) + 5] * n
    r3 = [1 << 30] * n
    T_all = np.asarray([T[0], Ts, T3], dtype=np.int64)
    E_all = np.asarray([Eg[0], [0] * n, E3], dtype=np.int64)
    rn_all = np.asarray([rn[0], rs, r3], dtype=np.int64)
    phi_all = np.asarray([phi[0], ps, [tref[500]] * n], dtype=np.int64)
    nb_all = np.asarray([nb[0], [0] * n, [ONE] * n], dtype=np.int64)
    his_all = np.asarray([[0] * n, hs, [0] * n], dtype=np.int64)
    ts = np.zeros((3, n), dtype=bool)
    ts[1] = True
    vac = np.zeros((3, n), dtype=bool)
    ring = np.zeros((3, n), dtype=bool)
    vac[2, : n // 2] = True
    ring[2, n // 2:] = True
    # the reference: solids, then accountable gas (independent cells: order-free)
    Tr = T_all.tolist()
    Er = E_all.tolist()
    c = R.FoldCounters()
    R.fold_pass1_solid(Tr, rn_all.tolist(), phi_all.tolist(), his_all.tolist(),
                       ts.astype(int).tolist(), c, clamp_enabled=clamp, table=tref)
    acct = (~ts & ~vac & ~ring).astype(int).tolist()
    R.fold_pass1_gas(Tr, Er, rn_all.tolist(), phi_all.tolist(), nb_all.tolist(),
                     ts.astype(int).tolist(), c, acct=acct, clamp_enabled=clamp, table=tref)
    # the engine
    s = _solver()
    Tc = np.ascontiguousarray(T_all.astype(np.int32))
    Ec = np.ascontiguousarray(E_all.copy())
    d = _cpp_fold(s, Tc, Ec, rn_all, phi_all, nb_all.astype(np.int32), ts, his_all, tbl,
                  vac=vac, ring=ring, clamp=clamp)
    assert np.array_equal(Tc[:2].astype(np.int64), np.asarray(Tr, dtype=np.int64)[:2])
    assert np.array_equal(Ec[:2], np.asarray(Er, dtype=np.int64)[:2])
    # Row 2 is outside the accountable set: Pass 0 wipes a vacuum / ring gas
    # cell to ambient (T = 0, E = 0: hygiene, outside the books -- the reference
    # has no Pass 0 to model it), and the fold must leave it THERE -- a deposit
    # on it would show as a non-zero energy, and in the counters above.
    assert np.all(Ec[2] == 0) and np.all(Tc[2] == 0), "the fold reached row 2"
    want = dict(t_max_phys_hits=c.t_max_phys_hits, t_low_rail_hits=c.t_low_rail_hits,
                rad_clamp_hits=c.rad_clamp_hits, e_rad_clamp_drop_sum=c.e_rad_clamp_drop_sum,
                e_gas_deposit_sum=c.e_gas_deposit_sum, e_gas_rail_sum=c.e_gas_rail_sum,
                e_solid_deposit_sum=c.e_solid_deposit_sum)
    assert d == want, (d, want)
    if clamp:
        gas_hits = sum(1 for i in range(n)
                       if R.sat_add_q16(T_all[0, i], R.gas_rad_dT_q(int(rn_all[0, i]),
                                                                    int(nb_all[0, i])))
                       > max(R.e_inv_q(int(phi_all[0, i]), tref), int(T_all[0, i])))
        assert 0 < gas_hits < n and c.rad_clamp_hits > gas_hits, (gas_hits, c.rad_clamp_hits)
        assert c.e_rad_clamp_drop_sum > 0
    else:
        assert c.rad_clamp_hits == 0 and c.e_rad_clamp_drop_sum == 0


class _EngineScene:
    """A reference Scene with a gas group, mirrored into the C++ sweep + the
    C++ fold (energy form), run tick by tick beside it."""

    def __init__(self, sc):
        self.ref = sc
        self.h, self.w = len(sc.a), len(sc.a[0])
        self.table = live_table()
        self.sweep = bp.RadiationSweep()
        self.solver = _solver()
        self.T = np.ascontiguousarray(np.asarray(sc.T, dtype=np.int64).astype(np.int32))
        self.E = np.ascontiguousarray(np.asarray(sc.Eg, dtype=np.int64))
        self.ts = np.asarray(sc.ts, dtype=np.int64) != 0
        his = sc.his if not isinstance(sc.his, int) else R.plane(self.h, self.w, sc.his)
        self.his = np.asarray(his, dtype=np.int64)
        self.held = (np.asarray(sc.held, dtype=np.int64) != 0) if sc.held is not None \
            else np.zeros((self.h, self.w), dtype=bool)
        self.nb = np.asarray(sc.n_bulk, dtype=np.int64)

    def tick(self):
        sc = self.ref
        rn, _rf, _ra, rl, _fl, _sw = cpp_sweep(
            sc.a, sc.d, sc.k[0][0], self.T.astype(np.int64).tolist(),
            self.his.tolist(), self.ts.tolist(), transport=sc.transport,
            table=self.table, sweep=self.sweep, gas=sc.gas, hq=sc.heat_absorb_q,
            n_bulk=sc.n_bulk)
        rn = rn.copy()
        rn[self.held] = 0                        # a pinned cell takes no update
        before_T, before_E = self.T.copy(), self.E.copy()
        d = _cpp_fold(self.solver, self.T, self.E, rn, rl, self.nb.astype(np.int32),
                      self.ts, self.his, self.table)
        return rn, d, before_T, before_E


def _scenes():
    return {"sealed smoky room": (G._sealed_smoky_room(7, 9, smoke_q=R.quant(0.06),
                                                       hq=R.quant(5.0),
                                                       T_smoke_game=G.T_SRC_GAME), 24),
            "smoke shield": (G._shield_scene(R.quant(0.2))[0], 24)}


@pytest.mark.parametrize("name", ["sealed smoky room", "smoke shield"])
def test_cpp_sweep_and_fold_follow_the_reference_scene_tick_for_tick(name):
    """PROPERTY (gate 0 across the whole tick, P5c): the C++ sweep feeding the
    C++ fold, on G15's two scenes on the LIVE table -- a sealed room of hot
    absorbing smoke, and a held 1263-game wall shining through a smoke layer at
    a held target -- reproduces the reference Scene tick for tick: every
    temperature, every stored gas energy, and the counters. Non-vacuous: the
    smoke's temperature moves, and on the shield scene the clamp binds.

    BREAKS IF: the engine's fold or sweep diverges from the reference anywhere
    in a multi-tick scene (a divergence one tick cannot show -- a mirror left
    stale, an energy residual drained, a counter booked twice).
    """
    sc, ticks = _scenes()[name]
    eng = _EngineScene(sc)
    gas = ~eng.ts
    T0 = eng.T.copy()
    for t in range(ticks):
        sc.tick()
        eng.tick()
        assert np.array_equal(eng.T.astype(np.int64), np.asarray(sc.T, dtype=np.int64)), t
        assert np.array_equal(eng.E[gas], np.asarray(sc.Eg, dtype=np.int64)[gas]), t
        c = sc.counters
        got = _counters(eng.solver)
        for k in ("rad_clamp_hits", "e_rad_clamp_drop_sum", "e_gas_deposit_sum",
                  "e_gas_rail_sum", "e_solid_deposit_sum", "t_max_phys_hits"):
            assert got[k] == getattr(c, k), (t, k, got[k], getattr(c, k))
    assert not np.array_equal(eng.T[gas], T0[gas]), "the smoke never moved: vacuous"
    if name == "smoke shield":
        assert eng.solver.rad_clamp_hits > 0, "the clamp never bound: vacuous"


@pytest.mark.parametrize("name", ["sealed smoky room", "smoke shield"])
def test_the_sweep_fold_boundary_is_bounded_and_counted_on_the_engine(name):
    """PROPERTY (f), design §8.4 on the ENGINE's own planes and counters: on
    every tick of G15's two scenes (C++ sweep -> C++ fold; a cooling room where
    the clamp never binds, and a shield whose smoke heats against its cap every
    tick), what the sweep booked on the fold's cells, sum(rad_net), differs
    from what the fold landed plus what e_rad_clamp_drop_sum says the clamp
    withheld by at most the conversions' own truncation -- < one temperature
    LSB x C per thermal solid (shr_round0), the staged chain's declared
    precision per gas cell (sweep_ref_q_gates._chain_bound) -- with no rail
    engaged, so nothing else crosses the boundary uncounted. On the shield the
    counted drop is non-zero (the clamp binds), so its CURRENCY is in the sum.

    BREAKS IF: the fold drops energy it does not count (a floor the drop
    counter misses, a second truncation), or e_rad_clamp_drop_sum is priced in
    another currency (the books' N * dT instead of cap_real * dT: measured red
    on the shield scene).
    """
    sc = _scenes()[name][0]
    eng = _EngineScene(sc)
    worst = 0.0
    drops = 0
    for _t in range(24):
        rn, d, bT, bE = eng.tick()
        assert d["t_max_phys_hits"] == 0 and d["e_gas_rail_sum"] == 0
        assert d["t_low_rail_hits"] == 0
        landed = bound = 0
        for y, x in zip(*np.nonzero(rn)):
            r = int(rn[y, x])
            if eng.ts[y, x]:
                cap = R.cap_real_q(True, int(eng.his[y, x]), 0)
                landed += (int(eng.T[y, x]) - int(bT[y, x])) * cap
                bound += cap
            else:
                N = int(eng.nb[y, x])
                cap = R.cap_real_q(False, 0, N)
                dE = int(eng.E[y, x]) - int(bE[y, x])
                assert dE % N == 0, "the gas landing is not a whole step of N"
                landed += (dE // N) * cap
                bound += G._chain_bound(r, N, cap, R.gas_rad_dT_q(r, N))
        resid = (int(rn.astype(object).sum()) << 16) - landed - d["e_rad_clamp_drop_sum"]
        assert abs(resid) <= bound, (resid, bound)
        worst = max(worst, abs(resid) / bound if bound else 0.0)
        drops += d["e_rad_clamp_drop_sum"]
    assert worst > 0.0, "the bound was never approached at all: vacuous"
    if name == "smoke shield":
        assert drops > 0, "the clamp never withheld anything: the drop is vacuous here"
    print(f"\n8.4 boundary ({name}): worst |resid| / bound = {worst:.3f}, "
          f"counted drop {drops}")


# ---------------------------------------------------------------------------
# the live engine
# ---------------------------------------------------------------------------
def _room_level(h_in, w_in, extra=()):
    """A sealed hull room on a space map; `extra` is ((y, x), material) pairs."""
    h, w = h_in + 2, w_in + 2
    tm = np.full((h, w), AIR, dtype=np.int32)
    tm[0, :] = tm[-1, :] = HULL
    tm[:, 0] = tm[:, -1] = HULL
    for (y, x), m in extra:
        tm[y, x] = m
    return LevelData(name="p5c_room", version="2", path=Path("."), tilemap=tm,
                     tile_size_m=0.333, diffuse_path=Path("."), boundary="space")


def _sealed_hot_smoky_room(*, smoke=0.06, T_game=1263, hq=None, h_in=8, w_in=12):
    """A sealed hull room whose air is seeded hot through the gas-energy seam
    (seed_gas_temperature -- 'Gas temperature is a mirror'), with smoke at
    `smoke` density; `hq` overrides smoke's heat_absorb (Q16) for a control."""
    sim = Simulation(_room_level(h_in, w_in), seed=1, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    inner = g._gas_energy_accountable()
    if smoke:
        g.gas[SMOKE][inner] = gas_fixed.quantize_scalar(smoke)
    g.seed_gas_temperature(inner, T_game << 16)
    if hq is not None:
        t = g.gases.heat_absorb_q16.copy()
        t[SMOKE] = hq
        g.gases.heat_absorb_q16 = np.ascontiguousarray(t)
    return sim, inner


def _step(sim):
    sim.set_paused(False)
    sim.step()


def test_closure_identity_and_total_ledger_close_with_gas_radiation_live():
    """PROPERTY (a): with smoke's heat radiation LIVE -- the shipped coefficient,
    a sealed room of hot smoke radiating into its hull -- the P-G5 TOTAL ledger
    (gas books + solid books) closes EXACTLY in int64 on every one of 120 ticks
    against the same six groups tests/test_thermostat_books.py sums (EOS, the
    thermal solver's gas side, combustion, the Python seams, water evac, the
    solid side), and the gas-only #54 identity (tests/test_e1_hot_rail's four
    groups) closes too: radiation into gas is GROUP 1 (e_gas_deposit_sum), with
    no new term. Non-vacuous: e_gas_deposit_sum moved (there is no heat deposit
    in this fireless room, so every count of it is the radiation branch), the
    smoke cooled, and the walls warmed.

    BREAKS IF: the gas branch writes gas_energy without booking it, books it
    into a new counter the identities do not sum, or books the clamp's WITHHELD
    energy (e_rad_clamp_drop_sum) as if it had landed.
    """
    from test_thermostat_books import _gas_books, _terms
    sim, inner = _sealed_hot_smoky_room()
    g = sim.gmap
    eos = sim.physics_runner.eos
    tsolver = sim.physics_runner.engine.temperature
    comb = sim.physics_runner.combustion
    engine = sim.physics_runner.engine
    prev_total = _gas_books(g) + int(tsolver.solid_energy_books_sum)
    prev_gas = _gas_books(g)
    prev_terms = _terms(g, eos, tsolver, comb, engine)
    dep0 = int(tsolver.e_gas_deposit_sum)
    walls = g.thermal_solid
    T_wall0 = int(g.temperature[walls].astype(np.int64).sum())
    T_gas0 = int(g.temperature[inner].astype(np.int64).sum())
    bad_total = bad_gas = 0
    for t in range(120):
        _step(sim)
        terms = _terms(g, eos, tsolver, comb, engine)
        gas_now = _gas_books(g)
        total_now = gas_now + int(tsolver.solid_energy_books_sum)
        d = [terms[0], terms[1] - prev_terms[1], terms[2] - prev_terms[2],
             terms[3] - prev_terms[3], terms[4], terms[5] - prev_terms[5]]
        bad_total += int((total_now - prev_total) != sum(d))
        bad_gas += int((gas_now - prev_gas) != sum(d[:5]))
        prev_total, prev_gas, prev_terms = total_now, gas_now, terms
    assert bad_total == 0, f"the P-G5 total ledger broke on {bad_total}/120 ticks"
    assert bad_gas == 0, f"the #54 gas identity broke on {bad_gas}/120 ticks"
    assert int(tsolver.e_gas_deposit_sum) != dep0, "the gas branch never booked: vacuous"
    assert int(g.temperature[inner].astype(np.int64).sum()) < T_gas0
    assert int(g.temperature[walls].astype(np.int64).sum()) > T_wall0


def test_hot_smoke_cools_radiatively_end_to_end_never_below_ambient():
    """PROPERTY (c): in the LIVE engine (Simulation.step, the shipped smoke
    coefficient), a sealed room of hot smoke COOLS RADIATIVELY: over 4 s its
    mean temperature falls, the coldest smoke cell never goes below ambient on
    any tick, and it cools far faster than the SAME room with smoke's
    heat_absorb zeroed, which can only conduct -- so radiation is the channel
    doing it. (Measured at P5c: 1263 -> ~476 game with radiation, 1263 -> ~1215
    without.)

    BREAKS IF: the fold's gas branch is disconnected (both rooms cool alike), a
    cooling step overshoots below ambient (the Fleck gas arm or the chain
    broken), or the branch heats instead of cooling.
    """
    ticks = 96
    live, inner = _sealed_hot_smoky_room()
    dark, _ = _sealed_hot_smoky_room(hq=0)
    g, gd = live.gmap, dark.gmap
    t0 = float(g.temperature[inner].mean())
    min_seen = None
    mean_prev = t0
    rises = 0
    for _t in range(ticks):
        _step(live)
        _step(dark)
        T = g.temperature[inner]
        m = float(T.min())
        min_seen = m if min_seen is None else min(min_seen, m)
        mean_now = float(T.mean())
        rises += int(mean_now > mean_prev)
        mean_prev = mean_now
    cooled_live = t0 - float(g.temperature[inner].mean())
    cooled_dark = t0 - float(gd.temperature[inner].mean())
    assert min_seen >= 0.0, f"a smoke cell went below ambient ({min_seen / ONE:.2f} game)"
    assert rises == 0, f"the room's mean temperature rose on {rises} ticks"
    assert cooled_live > 5.0 * cooled_dark > 0.0, (cooled_live / ONE, cooled_dark / ONE)
    print(f"\nhot smoky room: mean {t0 / ONE:.0f} -> {g.temperature[inner].mean() / ONE:.0f} "
          f"game radiating, -> {gd.temperature[inner].mean() / ONE:.0f} conducting only; "
          f"coldest cell ever {min_seen / ONE:.1f} game")


def _shield_room(smoke):
    """A sealed room with a HULL column (the emitter, held at the 1263-game
    plateau every tick -- a thermal solid is its own truth), a hull target tile
    and a marine across the room, and optionally a 5-column smoke layer
    between them at `smoke` density."""
    from simulation.unit import Unit
    h_in, w_in = 7, 15
    col = [((y, 2), HULL) for y in range(1, h_in + 1)]
    sim = Simulation(_room_level(h_in, w_in, col + [((4, 14), HULL)]), seed=1,
                     breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    emitter = np.zeros(g.material.shape, dtype=bool)
    emitter[1:h_in + 1, 2] = True
    layer = np.zeros_like(emitter)
    layer[1:h_in + 1, 6:11] = True
    layer &= g._gas_energy_accountable()
    if smoke:
        g.gas[SMOKE][layer] = gas_fixed.quantize_scalar(smoke)
    sim.add_unit(Unit("M1", x=13, y=2, team=0))
    return sim, emitter, (4, 14), (2, 13)


def test_smoke_shields_a_target_and_a_marine_against_clear_air():
    """PROPERTY (b), in the LIVE engine: a thick smoke layer (5 columns at density
    0.2 -- opaque at the shipped coefficient) between a held 1263-game hull
    column and a hull target / a marine across the room cuts what they ABSORB
    over 2 s to under a quarter of the clear-air control's -- the target's
    rad_net_sweep, the marine's body share rad_flux_sweep -- also after the
    smoke has heated and re-radiates. (Measured at P5c: marine 8 vs 32 262,
    target 186 vs 2 310 summed counts.)

    BREAKS IF: smoke stops absorbing in the sweep (the smoke term or the
    shipped coefficient gone), or the fold's gas branch re-emits more than the
    smoke absorbed (a mint), or the density law is bypassed.
    """
    sums = {}
    for name, smoke in (("clear", 0.0), ("smoke", 0.2)):
        sim, em, tgt, mar = _shield_room(smoke)
        g = sim.gmap
        net, flux = 0, 0
        for _t in range(48):
            g.temperature[em] = 1263 << 16
            _step(sim)
            net += int(g.rad_net_sweep[tgt])
            flux += int(g.rad_flux_sweep[mar])
        sums[name] = (net, flux)
    (net_c, flux_c), (net_s, flux_s) = sums["clear"], sums["smoke"]
    assert net_c > 0 and flux_c > 0, "the clear room shows no radiation: vacuous"
    assert 4 * net_s < net_c, sums
    assert 4 * flux_s < flux_c, sums
    print(f"\nshield: target rad_net {net_s} vs clear {net_c}; marine rad_flux "
          f"{flux_s} vs clear {flux_c}")


def test_a_smoke_free_room_folds_bit_identically_whatever_smoke_absorbs():
    """PROPERTY (d): the gas branch acts ONLY where a gas cell's rad_net is
    non-zero, and a gas cell's rad_net is non-zero only where its smoke term
    absorbs (a_gas > 0). So a room WITHOUT smoke -- a hot hull column radiating
    across still air, the air warming by conduction -- runs bit-identically with
    smoke's shipped coefficient and with it zeroed, on every GameMap array and
    every temperature counter, tick for tick; rad_net_sweep is exactly 0 on
    every gas cell of it; and on the direct binding a gas cell with rad_net ==
    0 keeps its stored energy AND its (deliberately stale) mirror untouched.
    Together: a smoke-free scene folds exactly as the pre-P5c fold did, which
    never touched a gas cell.

    BREAKS IF: the branch refreshes a mirror or drains a residual on a cell it
    has nothing to fold, reaches a cell whose smoke term is zero, or the smoke
    coefficient leaks into a smoke-free scene some other way.
    """
    def run(hq):
        sim, em, _t, _m = _shield_room(0.0)
        if hq is not None:
            t = sim.gmap.gases.heat_absorb_q16.copy()
            t[SMOKE] = hq
            sim.gmap.gases.heat_absorb_q16 = np.ascontiguousarray(t)
        g = sim.gmap
        frames = []
        for _i in range(30):
            g.temperature[em] = 1263 << 16
            _step(sim)
            gas = ~g.thermal_solid
            assert not np.any(g.rad_net_sweep[gas]), "a smoke-free gas cell booked rad_net"
            frames.append({k: v.copy() for k, v in vars(g).items()
                           if isinstance(v, np.ndarray)})
        ts = sim.physics_runner.engine.temperature
        return frames, _counters(ts)
    live, c_live = run(None)
    zero, c_zero = run(0)
    for t, (a, b) in enumerate(zip(live, zero)):
        for k in a:
            assert np.array_equal(a[k], b[k]), (t, k)
    assert c_live == c_zero
    # the direct-binding half: rn == 0 on a gas cell -> untouched, stale mirror kept
    s = _solver()
    h, w = 2, 3
    T = np.full((h, w), 777 << 16, dtype=np.int32)          # NOT E's mirror
    E = np.full((h, w), ONE * ((100 << 16) + T_AMB_Q) + 12345, dtype=np.int64)
    rn = np.zeros((h, w), dtype=np.int64)
    rn[0, 0] = 1 << 20                                       # one live gas cell
    ts = np.zeros((h, w), dtype=bool)
    E0, T0 = E.copy(), T.copy()
    _cpp_fold(s, T, E, rn, np.full((h, w), R.E_LIVE[100], np.int64),
              np.full((h, w), ONE, np.int32), ts, np.zeros((h, w)), live_table())
    untouched = np.ones((h, w), dtype=bool)
    untouched[0, 0] = False
    assert np.array_equal(E[untouched], E0[untouched])
    assert np.array_equal(T[untouched], T0[untouched])
    assert E[0, 0] != E0[0, 0], "the one live cell was not folded: vacuous"
