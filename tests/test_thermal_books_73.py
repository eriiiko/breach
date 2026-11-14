"""#73 items 1-3 -- the uncounted thermal channels, booked.

Erik's ruling on #73 (2026-09-30): ONE mechanical books patch, counters and
closure terms only, no behaviour change (the golden does not move). Each item
gets a scenario that DRIVES its channel, and each test proves the books close
in int64 WITH the new term / counter and do NOT close without it -- so the
counter is load-bearing, not decoration.

  1. The Pass-1 LOW rail's lift on a thermal solid is the solid identity's
     `TemperatureSolver.e_solid_low_rail_sum` (it used to sit inside
     e_solid_deposit_sum, unnamed; e_solid_deposit_sum now stops at the rail).
  2. A topology seam (destroy_wall / unseal_tiles retire a thermal solid,
     seal_tiles bears one) moves the solid books by the tile's measured
     cap_real*T: booked per channel in `GameMap.solid_energy_books` and summed
     into the identity's `TemperatureSolver.e_solid_seam_sum` (which also
     moves the snapshot, so the identity closes on the tick the seam fired --
     slot 9's burn-through / burst valve fire AFTER the solver's snapshot).
  3. The Pass-1 HEAT deposit's floor share below n_floor_heat is counted in
     `TemperatureSolver.e_dep_floor_drop_sum` -- a section 8.4-style boundary
     counter, NOT a closure term (it never entered gas_energy).

Run:  <breach-py> -m pytest tests/test_thermal_books_73.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
import test_temperature_gas_radiation as TGR  # noqa: E402
from test_thermostat_books import _gas_books, _terms  # noqa: E402

FP_ONE = 1 << 16
HULL = TGR.HULL


def _solid_books_live(g):
    """Sigma cap_real*T over the map's thermal solids, through the map's own
    per-tile helper (the engine's capacity law) -- the LIVE value the
    solver's snapshot must equal at the end of a tick."""
    ys, xs = np.nonzero(g.thermal_solid)
    return sum(g._solid_books_cell(int(y), int(x)) for y, x in zip(ys, xs))


class _Ledger:
    """The P-G5 TOTAL ledger (gas books + solid books) against the six groups
    test_thermostat_books._terms sums, per tick, and the same residual with
    each #73 solid term left OUT, to show the books do not close without it."""

    def __init__(self, sim):
        self.sim = sim
        self.g = sim.gmap
        self.eos = sim.physics_runner.eos
        self.ts = sim.physics_runner.engine.temperature
        self.comb = sim.physics_runner.combustion
        self.engine = sim.physics_runner.engine
        self.reset()

    def _t(self):
        return _terms(self.g, self.eos, self.ts, self.comb, self.engine)

    def total(self):
        return _gas_books(self.g) + int(self.ts.solid_energy_books_sum)

    def reset(self, extra=0):
        """Re-base (after a fixture write the books cannot see; `extra` is that
        write's measured change to the solid books)."""
        self.prev_total = self.total() + extra
        self.prev_terms = self._t()
        self.prev_named = self._named()

    def _named(self):
        return dict(low_rail=int(self.ts.e_solid_low_rail_sum),
                    seam=int(self.ts.e_solid_seam_sum))

    def close(self):
        """Read this tick: (residual with every term, {name: residual without
        that term})."""
        terms, named = self._t(), self._named()
        p = self.prev_terms
        expected = (terms[0] + (terms[1] - p[1]) + (terms[2] - p[2])
                    + (terms[3] - p[3]) + terms[4] + (terms[5] - p[5]))
        total = self.total()
        resid = (total - self.prev_total) - expected
        without = {k: resid + (named[k] - self.prev_named[k]) for k in named}
        self.prev_total, self.prev_terms, self.prev_named = total, terms, named
        return resid, without


def test_the_low_rails_lift_is_a_term_of_the_solid_identity():
    """PROPERTY (#73 item 1): when the Pass-1 low rail lifts a cold thermal
    solid back to ambient, the energy it adds is booked by name
    (e_solid_low_rail_sum, > 0) and the P-G5 total ledger closes EXACTLY in
    int64 on every tick with it -- and does NOT close on the ticks the rail
    fired if the term is left out. The rail's hits and its booked energy move
    together.

    Scene: the sealed hot smoky room; after one tick every hull cell is put a
    sixteenth of a degree below ambient (a fixture write, measured and folded
    into the baseline) -- the state #73 measured conduction leaving hull cells
    in, scaled so the sweep's own warming cannot lift them in one fold.

    BREAKS IF: the rail's lift is priced in neither e_solid_deposit_sum nor a
    named term (the books break by the lift), or in both (double count); or
    the counter books a quantity other than (0 - T)*cap_real."""
    sim, _inner = TGR._sealed_hot_smoky_room()
    TGR._step(sim)
    led = _Ledger(sim)
    g = sim.gmap
    before = _solid_books_live(g)
    g.temperature[g.thermal_solid] -= 4096
    led.reset(extra=_solid_books_live(g) - before)
    hits0, lift0 = int(led.ts.t_low_rail_hits), int(led.ts.e_solid_low_rail_sum)
    bad = broke_without = 0
    for _t in range(12):
        TGR._step(sim)
        resid, without = led.close()
        bad += int(resid != 0)
        broke_without += int(without["low_rail"] != 0)
    hits = int(led.ts.t_low_rail_hits) - hits0
    lift = int(led.ts.e_solid_low_rail_sum) - lift0
    assert hits > 0, "the low rail never fired: vacuous"
    assert lift > 0, f"the rail fired {hits} times but booked no energy"
    assert bad == 0, f"the total ledger broke on {bad} ticks WITH the rail's term"
    assert broke_without > 0, "the ledger closes without the rail's term: not load-bearing"
    # the snapshot is the live sum (nothing wrote a solid behind the solver)
    assert int(led.ts.solid_energy_books_sum) == _solid_books_live(g)


def test_topology_seams_book_the_solid_energy_they_move():
    """PROPERTY (#73 item 2): a wall DESTROYED mid-run while the hull is warm
    (destroy_wall), a door panel SEALED beside it (seal_tiles, born at its
    close-T) and OPENED again (unseal_tiles) each move the solid books by the
    tile's measured cap_real*T, booked in its own GameMap.solid_energy_books
    channel and summed into TemperatureSolver.e_solid_seam_sum; the P-G5 total
    ledger closes EXACTLY on every tick with that term, in BOTH timings a
    seam fires in -- after the solver's snapshot inside a tick (slot 9: the
    burn-through, the burst valve; here: between step() and the books read)
    and before the next tick's physics (slot 2: explosions, chew) -- and
    does NOT close without it. The snapshot stays the live sum throughout.

    BREAKS IF: a seam changes thermal-solid membership without booking the
    tile's energy; books it without moving the snapshot (red on the slot-9
    timing, by exactly the tile's energy); or books a formula instead of the
    measured change (seal's close-T)."""
    sim, _inner = TGR._sealed_hot_smoky_room()
    for _w in range(30):                       # warm the hull above ambient
        TGR._step(sim)
    g = sim.gmap
    h, w = g.temperature.shape
    assert int(g.temperature[h // 2, w - 1]) > 0 and int(g.temperature[h // 2, 0]) > 0
    led = _Ledger(sim)
    seam0 = int(led.ts.e_solid_seam_sum)
    door = (h // 2, 1)                         # an air tile beside the west hull
    # (tick, timing, action): "in" = after this tick's step, before the read
    # (the snapshot-adjust case); "next" = after the read (before the next
    # tick's physics).
    events = {3: ("in", lambda: g.destroy_wall(h // 2, w - 1)),
              6: ("next", lambda: g.destroy_wall(h // 2 - 2, w - 1)),
              9: ("in", lambda: g.seal_tiles([door], HULL)),
              14: ("next", lambda: g.unseal_tiles([door]))}
    bad = broke_without = 0
    for t in range(20):
        TGR._step(sim)
        ev = events.get(t)
        if ev and ev[0] == "in":
            ev[1]()
        resid, without = led.close()
        bad += int(resid != 0)
        broke_without += int(without["seam"] != 0)
        if ev and ev[0] == "next":
            ev[1]()
        assert int(led.ts.solid_energy_books_sum) == _solid_books_live(g), t
    books = g.solid_energy_books
    for ch in ("destroy_wall", "seal", "unseal"):
        assert books.get(ch, 0) != 0, f"channel {ch} never moved: vacuous ({books})"
    assert books["destroy_wall"] < 0 and books["seal"] > 0 and books["unseal"] < 0
    assert int(led.ts.e_solid_seam_sum) - seam0 == g.solid_energy_seam_net()
    assert bad == 0, f"the total ledger broke on {bad} ticks WITH the seam term"
    assert broke_without > 0, "the ledger closes without the seam term: not load-bearing"


def _thin_gas_scene(n_raw, seed=73):
    """An all-gas grid with NO conduction faces (every face_shift is no_face),
    so each cell's stored-energy change is its own Pass-1 heat deposit and
    nothing else: bulk N = `n_raw` (scalar or plane), air at +100 game, a heat
    deposit on every cell."""
    rng = np.random.default_rng(seed)
    h, w = 4, 6
    s = TGR._solver()
    nb = np.broadcast_to(np.asarray(n_raw, dtype=np.int32), (h, w)).copy()
    t_amb = int(TGR.T_AMB_Q)
    E = nb.astype(np.int64) * (t_amb + (100 << 16))
    T = np.full((h, w), 100 << 16, dtype=np.int32)
    heat = rng.integers(1 << 14, 1 << 20, size=(h, w)).astype(np.int32)
    return s, dict(T=T, E=E, nb=nb, heat=heat)


def _deposit(s, sc):
    """One TemperatureSolver.step (direct binding, energy form) of the scene's
    heat; returns (counter deltas, per-cell landed heat dT*cap_real)."""
    h, w = sc["T"].shape
    names = ("e_dep_floor_drop_sum", "e_deposit_drop_sum", "e_gas_deposit_sum",
             "e_gas_rail_sum", "e_gas_cond_sum", "t_max_phys_hits")
    c0 = {k: int(getattr(s, k)) for k in names}
    E0 = sc["E"].copy()
    zeros_b = np.zeros((h, w), dtype=bool)
    s.step(sc["T"], np.ascontiguousarray(sc["heat"]), np.zeros((h, w), np.int32),
           np.full((h, w, 4), s.no_face, dtype=np.int32), zeros_b, zeros_b,
           np.full((h, w), FP_ONE, np.int32),
           n_bulk=np.ascontiguousarray(sc["nb"]), thermal_solid=zeros_b,
           gas_energy=sc["E"], t_amb_q=int(TGR.T_AMB_Q))
    d = {k: int(getattr(s, k)) - c0[k] for k in names}
    dE = sc["E"] - E0
    nb = sc["nb"].astype(np.int64)
    assert np.all(dE % nb == 0), "a cell's dE is not N*dT: something else moved it"
    dT = dE // nb
    n_floor_q = int(bp.fp_quantize(float(s.n_floor_heat)))
    c_v_q = int(bp.fp_quantize(float(s.c_v)))
    landed = 0
    for (y, x), n in np.ndenumerate(sc["nb"]):
        _cu, cr = bp.conduction_cell_capacity_q(False, 0, int(n), n_floor_q, c_v_q)
        landed += int(dT[y, x]) * int(cr)
    return d, landed, n_floor_q


def test_the_heat_deposits_floor_share_is_counted_on_thin_gas():
    """PROPERTY (#73 item 3): on accountable gas cells below n_floor_heat a
    Pass-1 heat deposit lands only ~N/n_floor of what the cell absorbed; the
    unlanded floor share is counted in e_dep_floor_drop_sum, so per step,
    EXACTLY in int64 and in the landing's heat currency,
        Sigma deposit * FP_ONE == Sigma landed (dT * cap_real)
                                  + e_deposit_drop_sum * FP_ONE   (attenuation)
                                  + e_dep_floor_drop_sum          (the floor)
    -- and without the new counter the boundary is short by most of the
    absorbed energy. It is a COUNTER, not a #54 term: the gas books close on
    the landing alone (Sigma dE == e_gas_deposit_sum). Control: above the
    floor the counter does not move.

    BREAKS IF: the floor share leaves the deposit uncounted (the pre-#73
    state), is counted on cells above the floor, is booked into gas_energy,
    or is priced in another currency than the landing."""
    n_floor_q = int(bp.fp_quantize(float(TGR._solver().n_floor_heat)))
    rng = np.random.default_rng(7)
    # every cell at most half the floor, so the floor withholds at least half
    # of what each cell absorbed
    s, sc = _thin_gas_scene(rng.integers(n_floor_q // 8, n_floor_q // 2, size=(4, 6)))
    deposit = int(sc["heat"].astype(np.int64).sum())
    d, landed, _ = _deposit(s, sc)
    assert d["t_max_phys_hits"] == 0 and d["e_gas_rail_sum"] == 0
    assert d["e_dep_floor_drop_sum"] > 0, "no floor share counted: vacuous"
    boundary = deposit * FP_ONE - landed - d["e_deposit_drop_sum"] * FP_ONE
    assert boundary == d["e_dep_floor_drop_sum"], (boundary, d)
    # load-bearing: without the counter the boundary misses most of e_abs
    absorbed = deposit * FP_ONE - d["e_deposit_drop_sum"] * FP_ONE
    assert boundary > absorbed // 2, (boundary, absorbed)
    # not a #54 term: the gas books close on the landing alone
    assert int((sc["E"].astype(object)).sum()) - int(
        (sc["nb"].astype(np.int64) * (int(TGR.T_AMB_Q) + (100 << 16))).astype(object).sum()
    ) == d["e_gas_deposit_sum"] + d["e_gas_rail_sum"] + d["e_gas_cond_sum"]

    # control: at or above the floor (still thin, so the attenuation counts)
    s2, sc2 = _thin_gas_scene(rng.integers(n_floor_q, FP_ONE // 2, size=(4, 6)))
    d2, _landed2, _ = _deposit(s2, sc2)
    assert d2["e_dep_floor_drop_sum"] == 0, d2
    assert d2["e_deposit_drop_sum"] > 0
