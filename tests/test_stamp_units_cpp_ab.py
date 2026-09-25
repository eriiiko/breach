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
from simulation import fire_fixed, gas_fixed, water_fixed, wave_fixed
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
    # These fields are all int32 Q16.16, so a bare float assignment TRUNCATES
    # to 0 — every seed here was silently landing on an empty map (found in the
    # 2026-09-12 fixture sweep; unrelated to R3, just the same class of fixture
    # that quietly stopped meaning what it said). The A/B property under test is
    # unit stamping, which these fields only decorate, so the comparison stayed
    # valid — but it was comparing two blank scenes. Quantize properly.
    g.smoke[interior] = gas_fixed.quantize_scalar(0.6)
    g.fire[10, 10] = fire_fixed.quantize_scalar(0.8)
    g.fire[10, 11] = fire_fixed.quantize_scalar(0.5)
    g.water_depth[12, 12] = water_fixed.quantize_scalar(0.3)
    g.water_depth[12, 13] = water_fixed.quantize_scalar(0.3)
    g.wave_source[5, 5] = wave_fixed.quantize_scalar(8.0)
    m1 = Unit("M1", x=4, y=4, team=0)
    m2 = Unit("M2", x=14, y=14, team=0)
    # Ray-engine-v2 P1: one marine keeps the default heat extinction (1.0, an
    # opaque body), the other declares a partial one — so the fourth stamp
    # output exercises BOTH the quantize-at-the-boundary and the MAX (a 0.5
    # body over air stamps 32768; over a wall it must not lower the wall's ONE).
    m2.heat_atten = 0.5
    # Ray-engine-v2 P6a: M2 also declares a TINTED light extinction, so the
    # fifth stamp output's per-channel MAX sees a partial triple as well as the
    # default opaque one (and the float stamp beside it the same triple).
    m2.light_atten = (0.25, 0.5, 1.0)
    sim.add_unit(m1)
    sim.add_unit(m2)
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
    # Ray-engine-v2 P1: the fourth output, the Q16 heat-extinction plane, must
    # vary too, carry BOTH unit values (M1's opaque 65536 and M2's partial
    # 32768 over air), and never fall below the static material plane (MAX).
    assert not np.array_equal(traj[2]["dyn_heat_atten_q"],
                              traj[18]["dyn_heat_atten_q"]), \
        "dyn_heat_atten_q never changed — the heat stamp is trivial"
    early = traj[2]["dyn_heat_atten_q"]
    static = traj[2]["heat_atten_q"]
    assert np.any(early == 32768) and np.any(early == 65536)
    assert np.all(early >= static), "a body lowered a material's extinction"
    assert np.all(early <= 65536)
    # after M2 dies (tick 20) no 32768 stamp remains
    late = traj[30]["dyn_heat_atten_q"]
    assert not np.any(late == 32768)
    # Ray-engine-v2 P6a: the FIFTH output, the Q16 light twin (h, w, 3), must
    # vary, carry M2's tint per channel (16384, 32768, 65536 over air) and M1's
    # opaque triple, never fall below the static plane (MAX), and lose the tint
    # once M2 is dead -- so the 0-ULP match above covers a real integer stamp.
    lq_early = traj[2]["dyn_light_atten_q"]
    lq_static = traj[2]["light_atten_q"]
    assert lq_early.shape[-1] == 3
    assert not np.array_equal(lq_early, traj[18]["dyn_light_atten_q"]), \
        "dyn_light_atten_q never changed — the integer light stamp is trivial"
    tint = np.all(lq_early == np.array([16384, 32768, 65536]), axis=-1)
    assert np.any(tint), "M2's tinted light extinction was never stamped"
    assert np.any(np.all(lq_early == 65536, axis=-1) & ~np.all(lq_static == 65536, axis=-1))
    assert np.all(lq_early >= lq_static), "a body lowered a material's light extinction"
    assert not np.any(np.all(traj[30]["dyn_light_atten_q"]
                             == np.array([16384, 32768, 65536]), axis=-1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
