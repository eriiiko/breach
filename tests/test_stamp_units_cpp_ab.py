"""0-ULP A/B gate for the stamp_units C++ port.

The per-tick dynamic-field rebuild (``GameMap.stamp_units``) moved from Python
into the C++ ``PhysicsEngine`` (``stamp-units-cpp`` branch). This is a PURE-
STRUCTURE move that must stay behavior-identical: the C++ path does only copies,
a boolean compare, and per-cell min/max — NO float arithmetic — so it is bit-
identical to the Python reference by construction.

This test is the field-level gate (mirrors ``field_ab_harness.py`` / the Patch-1
S-step gates): it captures one trajectory with the Python ``stamp_units`` and one
with the C++ ``stamp_units`` on the SAME seed + the SAME deterministic unit
driver, and asserts per-FIELD per-CELL equality (0-ULP) over the whole trajectory
— for ``obstacles`` / ``dyn_permeability`` / ``dyn_wave_absorb`` /
``dyn_light_atten`` AND every downstream field (the ``dyn_*`` fields feed the
solvers, so a stamp slip shows up everywhere).

CRUCIALLY the scenario has units whose footprints MOVE and DIE tick-to-tick (a
static unit would make the stamp trivially constant and hide a reset bug). The
driver below walks two marines across the interior and kills one partway through.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_stamp_units_cpp_ab.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pytest

import breach_physics as bp
from level_loader import LevelData
from simulation import Simulation
from simulation.unit import Unit

from field_ab_harness import SIM_FIELDS, diff_trajectories

SEED = 20260623
N_STEPS = 40


def _scenario_level() -> LevelData:
    """A 20x20 hull-walled room with a carved interior — room for two 3x3
    marines to walk around without leaving the air pocket."""
    h = w = 20
    tm = np.ones((h, w), dtype=np.int32)   # all hull
    tm[1:19, 1:19] = 4                       # carve interior air
    return LevelData(name="stamp_ab", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _make_sim() -> Simulation:
    """Scenario with two marines whose footprints will move + one that dies.

    Seeds smoke + fire + water + a wave pulse + a hull breach (so every solver
    is active and the dyn_* fields actually feed downstream physics), then spawns
    two 3x3 marines. The driver (_drive) walks them and kills one — so the stamp
    changes every tick."""
    sim = Simulation(_scenario_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    interior = (~g.solid) & (~g.is_vacuum)
    g.smoke[interior] = 0.6
    g.fire[10, 10] = 0.8
    g.fire[10, 11] = 0.5
    g.water_depth[12, 12] = 0.3
    g.water_depth[12, 13] = 0.3
    g.wave_source[5, 5] = 8.0
    sim.add_unit(Unit("M1", x=4, y=4, team=0))
    sim.add_unit(Unit("M2", x=14, y=14, team=0))
    g.destroy_wall(10, 0)            # hull breach -> vacuum (venting)
    sim.set_paused(False)
    return sim


def _drive(sim: Simulation, tick: int) -> None:
    """Deterministic per-tick unit driver: move both marines on a fixed path
    and kill M2 at tick 20. Pure function of ``tick`` (no RNG) so the Python and
    C++ runs see byte-identical unit footprints each tick.

    Mutates ``u.x`` / ``u.y`` directly (the stamp reads ``tile_x``/``tile_y``)
    rather than going through the order/AI system — the stamp only cares about
    the resulting footprint, and a direct drive keeps the two A/B runs in
    lockstep without depending on movement-system determinism."""
    units = sim.units
    if len(units) >= 1:
        # M1 walks a diagonal box, staying inside [2, 16].
        m1 = units[0]
        m1.x = float(2 + (tick % 14))
        m1.y = float(2 + ((tick // 2) % 14))
    if len(units) >= 2:
        m2 = units[1]
        if tick >= 20:
            m2.alive = False           # dies -> footprint must STOP stamping
        else:
            m2.x = float(16 - (tick % 14))
            m2.y = float(2 + (tick % 14))


def _capture(use_cpp_stamp: bool):
    """Run the scenario for N_STEPS with the given stamp path, driving the units
    each tick. Returns a per-tick list of field-snapshot dicts."""
    sim = _make_sim()
    sim.gmap.use_cpp_stamp = use_cpp_stamp
    traj = []
    for t in range(N_STEPS):
        _drive(sim, t)
        sim.set_paused(False)
        sim.step()
        traj.append({name: np.copy(getattr(sim.gmap, name))
                     for name in SIM_FIELDS if hasattr(sim.gmap, name)})
    return traj


def test_stamp_units_cpp_matches_python_0ulp():
    """The C++ stamp_units path is bit-identical to the Python reference over a
    trajectory with moving + dying units."""
    py_traj = _capture(use_cpp_stamp=False)
    cpp_traj = _capture(use_cpp_stamp=True)
    diffs = diff_trajectories(py_traj, cpp_traj, tol=0.0)
    assert not diffs, (
        "stamp_units C++ != Python (0-ULP gate failed):\n  "
        + "\n  ".join(diffs[:20]))


def test_stamp_changes_tick_to_tick():
    """Guard the gate itself: confirm the scenario's stamp is NON-trivial — the
    dyn_* fields and obstacles must actually change as units move/die, else the
    0-ULP match above would be vacuous."""
    traj = _capture(use_cpp_stamp=True)
    # dyn_permeability must differ between an early and a late tick (units moved).
    assert not np.array_equal(traj[2]["dyn_permeability"],
                              traj[18]["dyn_permeability"]), \
        "dyn_permeability never changed — scenario units are not moving"
    # dyn_light_atten must also vary (the per-channel MAX stamp).
    assert not np.array_equal(traj[2]["dyn_light_atten"],
                              traj[18]["dyn_light_atten"]), \
        "dyn_light_atten never changed — stamp is trivial"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
