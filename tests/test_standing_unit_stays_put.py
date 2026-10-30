"""#116 -- a standing unit nobody pushes stays exactly where it stands, and its
stamped footprint lies under the body the renderer draws.

THE BUG (found by #114's GPU check, 2026-10-27). On a quiet playground the
pressure field jitters a few Q16 counts across a standing marine's 3x3
footprint; ``exchange.apply_wave_push`` turned each count into a ~3e-6
tile/tick nudge, so from tick 3 the float position crept just below its
integer (13.0 -> 12.99997). ``tile_x = int(x)`` then stamped the footprint
-- obstacles, permeability, wave absorb, heat and light extinction -- one
tile north-west of the drawn body. The fix is the push's noise floor
(``[exchange] push_grad_floor_atm``).

PROPERTIES PROTECTED
  1. A unit with no orders, on a field nobody perturbs, keeps its EXACT float
     position and its tile over a whole round's worth of ticks. Breaks if the
     noise floor is removed (or set to 0), or if any other per-tick push on a
     standing unit appears.
  2. The tiles the sim stamps for a unit are exactly the tiles whose centres
     lie inside the square the renderer draws it on. The draw rule is
     ``renderer/unit_model_renderer.py::_draw_one``: centre
     ``(x + fp/2, y + fp/2)``, side ``fp`` tiles. Breaks if the stamp anchor
     (``Unit.tile_x``/``occupied_tiles``) or the draw centre moves without the
     other, or if property 1 regresses (a drifted unit stamps one tile off).
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


def _playground_marines():
    from config import CFG
    from level_loader import load as load_level
    from simulation import Simulation
    from simulation.unit import Unit
    level = load_level("playground", levels_dir=str(ROOT / "levels"))
    sim = Simulation(level, seed=1, breach_physics=bp, enable_recorder=False)
    # The marines main.py spawns (zombies walk on their own -- not "standing").
    for s in level.spawns:
        if s.team == 0:
            sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team,
                              footprint=s.footprint))
    assert sim.units, "playground lost its marine spawns"
    sim.set_paused(False)
    # Stay inside one round: the round-end teardown rounds positions, which
    # would hide a drift rather than show its absence.
    n_ticks = int(CFG.clock.ticks_per_round) - 1
    return sim, n_ticks


def _drawn_tiles(u):
    """Tiles whose centres lie inside the drawn body square (renderer rule)."""
    fp = float(u.footprint)
    cx, cy = float(u.x) + fp / 2.0, float(u.y) + fp / 2.0
    lo_x, hi_x = cx - fp / 2.0, cx + fp / 2.0
    lo_y, hi_y = cy - fp / 2.0, cy + fp / 2.0
    out = set()
    for ty in range(int(np.floor(lo_y)) - 1, int(np.ceil(hi_y)) + 1):
        for tx in range(int(np.floor(lo_x)) - 1, int(np.ceil(hi_x)) + 1):
            if lo_x <= tx + 0.5 < hi_x and lo_y <= ty + 0.5 < hi_y:
                out.add((tx, ty))
    return out


def test_standing_units_keep_exact_position_and_tile_for_a_round():
    sim, n_ticks = _playground_marines()
    start = {u.id: (u.x, u.y, u.tile_x, u.tile_y) for u in sim.units}
    for t in range(n_ticks):
        if sim.is_paused():          # phase boundary: keep the clock running
            sim.set_paused(False)
        sim.step()
        for u in sim.units:
            assert (u.x, u.y, u.tile_x, u.tile_y) == start[u.id], (
                f"tick {t + 1}: standing {u.name} moved from {start[u.id]} "
                f"to {(u.x, u.y, u.tile_x, u.tile_y)}")


def test_stamp_covers_exactly_the_drawn_tiles():
    sim, n_ticks = _playground_marines()
    g = sim.gmap
    for t in range(n_ticks):
        if sim.is_paused():
            sim.set_paused(False)
        sim.step()
        if t % 12:
            continue
        g.stamp_units(sim.units)     # the stamp at the CURRENT positions
        stamped = np.argwhere(g.dyn_heat_atten_q != g.heat_atten_q)
        stamped = {(int(x), int(y)) for (y, x) in stamped}
        drawn = set()
        for u in sim.units:
            d = _drawn_tiles(u)
            assert d == set(u.occupied_tiles()), (
                f"tick {t + 1}: {u.name} at ({u.x!r}, {u.y!r}) stamps "
                f"{sorted(u.occupied_tiles())} but is drawn on {sorted(d)}")
            drawn |= d
        assert stamped == drawn, f"tick {t + 1}: stamp plane != drawn bodies"
