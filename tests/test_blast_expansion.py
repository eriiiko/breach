"""The blast expansion source (#31, src/simulation/blast_expansion.py +
EOSSolver.div_source), on a comparison row (expansion_* columns): a blast tells
the pressure solve the air around it is expanding, and the solve finds the wind.
tests/test_explosion.py holds the generic explosion built on it. Each test names
the property it protects and the change that must break it. No test reads a row's
value; those are Erik's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bp = pytest.importorskip("breach_physics")

from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.blast_expansion import reach_tiles  # noqa: E402
from simulation.gases import INERT_N2, O2  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.payloads import execute_payload  # noqa: E402
from simulation.weapons import PayloadDef  # noqa: E402
from test_heat_source import _closure_bad, _step  # noqa: E402

CH = (15, 15)


def _sim():
    tm = np.full((30, 30), MAT_HULL, dtype=np.int32)
    tm[1:29, 1:29] = MAT_AIR
    return Simulation(LevelData(name="expansion", version="2", path=Path("."),
                                tilemap=tm, tile_size_m=0.333,
                                diffuse_path=Path("."), entities=[], wires=[]),
                      seed=3, breach_physics=bp, enable_recorder=False)


def _row(v=20.0):
    return PayloadDef("t", heat_amount=30.0, heat_radius=3.0, expansion_m3=v,
                      expansion_radius=3.0, expansion_ticks=2)


def _fire(s, v=20.0):
    execute_payload(s.gmap, s.edit_queue, s.units, CH[0], CH[1], _row(v), s.rng)


def _bulk(g):
    return int(g.gas[O2].astype(np.int64).sum() + g.gas[INERT_N2].astype(np.int64).sum())


def _peak_wind(s, ticks=8):
    g = s.gmap
    peak = 0.0
    for _ in range(ticks):
        _step(s)
        peak = max(peak, float(np.hypot(g.wind_x / 65536.0, g.wind_y / 65536.0).max()))
    return peak


def test_a_source_adds_no_gas():
    """PROPERTY: the source moves air and creates none: in a sealed room the bulk
    N total stays exact to the bit through the blast. That is the whole reason to
    prefer it to hot mass, which leaves a room over-pressured.

    BREAKS IF: the source writes N, or its flow leaves the conservative transport.
    """
    s = _sim()
    for _ in range(2):
        _step(s)
    n0 = _bulk(s.gmap)
    _fire(s)
    for _ in range(24):
        _step(s)
    assert _bulk(s.gmap) == n0


def test_the_energy_books_close_through_a_blast():
    """PROPERTY: the closure identity stays exact in int64 on every tick of a
    source-driven blast, because the source only changes velocity and energy still
    moves through the face fluxes.

    BREAKS IF: the source reaches energy outside the books, for example by
    touching the expansion-work term.
    """
    s = _sim()

    def blast(i, sim):
        if i == 2:
            _fire(sim)

    bad = _closure_bad(s, 16, before_step=blast)
    assert not bad, f"closure broken on (step, residual) {bad[:5]}"


def test_more_expansion_gives_more_wind():
    """PROPERTY: wind keeps growing with the source, so the strength number stays
    a real dial. Heat alone has strongly diminishing returns here.

    BREAKS IF: the source saturates, or stops reaching the solve.
    """
    s1, s2 = _sim(), _sim()
    for s in (s1, s2):
        _step(s)
    _fire(s1, 20.0)
    _fire(s2, 80.0)
    assert _peak_wind(s2) > 1.5 * _peak_wind(s1)


def test_a_wall_stops_the_source_and_a_gap_does_not():
    """PROPERTY: the source walks open floor: it does not reach through a wall, but
    it does pass through a gap in that wall.

    BREAKS IF: reach goes back to a plain disc, which would push air in the next
    room through solid hull.
    """
    solid = np.zeros((11, 11), dtype=bool)
    solid[:, 6] = True
    assert not any(x > 6 for _, x in reach_tiles(~solid, 5, 4, 25))
    solid[5, 6] = False
    assert any(x > 6 for _, x in reach_tiles(~solid, 5, 4, 25))
