"""Tests for over-pressure wall failure — the emergent pressure-relief valve
(ch.04 §5, docs/architecture/engine/04_atmosphere_and_pressure.md).

A sealed room that keeps absorbing grenades would otherwise build pressure
without limit. Instead, each tick (after physics) any wall holding a pressure
differential above its material's ``burst_threshold`` fails and vents. Over-
pressured clusters self-breach in a chain until the gradient relaxes.

Two deterministic, headless cases:

1. POPS UNDER OVER-PRESSURE: a small sealed room pumped far above any
   threshold loses at least one wall, and the peak interior pressure afterwards
   is LOWER than before (relief happened).
2. NO SPURIOUS POPS: the SAME room at normal pressure (~1.0), stepped the same
   number of ticks, loses ZERO walls (the threshold is not tripped by a
   normally-pressurised ship).

Run:
    C:/Users/steen/anaconda3/python.exe tests/test_wall_failure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import numpy as np

import breach_physics as bp
from level_loader import LevelData
from simulation import Simulation
from simulation.gamemap import GameMap

SEED = 7
N_STEPS = 5


def _sealed_room_level():
    """Build a tiny level: a hull-walled room with interior air, surrounded by
    outside vacuum. CSV codes: 0 = vacuum, 1 = hull, 4 = interior air.

    Layout (12x12), '.' = vacuum, '#' = hull, ' ' = interior air:

        ............
        ............
        ..########..
        ..#      #..
        ..#      #..
        ..#      #..
        ..#      #..
        ..#      #..
        ..#      #..
        ..########..
        ............
        ............

    The hull ring is one tile thick. Each hull tile borders interior air on one
    side and outside vacuum (which contributes 0 to the spread) on the other, so
    a pressurised interior makes every hull tile hold ~p_room of differential.
    """
    h = w = 12
    tm = np.zeros((h, w), dtype=np.int32)  # all vacuum
    # Hull ring at rows/cols 2..9.
    tm[2:10, 2:10] = 1                      # fill the box with hull...
    tm[3:9, 3:9] = 4                        # ...then carve interior air.
    return LevelData(
        name="sealed_room_test",
        version="1",
        path=Path("."),
        tilemap=tm,
        tile_size_m=1.0,
        diffuse_path=Path("."),
    )


def _interior_mask(gmap: GameMap):
    """Boolean mask of the interior air tiles (not solid, not vacuum)."""
    return (~gmap.solid) & (~gmap.is_vacuum)


def _make_sim():
    level = _sealed_room_level()
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    sim.set_paused(False)
    return sim


def test_pops_under_overpressure():
    """A sealed room pumped far over threshold loses >=1 wall and relieves."""
    sim = _make_sim()
    g = sim.gmap

    interior = _interior_mask(g)
    assert interior.any(), "test level has no interior air"

    # The hull now ships at burst_threshold=0 (never pressure-collapses; it
    # breaches via damage/explosions — ch.04). Re-enable the relief valve on the
    # test's wall material so we exercise the mechanism on a burstable wall.
    for wid in np.unique(g.material[g.solid]):
        g.materials.burst_threshold[int(wid)] = 6.0

    walls_before = int(g.solid.sum())
    # Pump the interior far above the (re-enabled) burst_threshold. EOS P3:
    # `atmosphere` (P) is solver-materialized from (N,T) every tick — a
    # direct P write would be overwritten. Create the overpressure through
    # the REAL state: scale the bulk O2/N2 up 50x (p* = C*N*T -> P = 50).
    from simulation import atmosphere_fixed
    from simulation.gases import O2, INERT_N2
    g.gas[O2][interior] = g.gas[O2][interior] * 50
    g.gas[INERT_N2][interior] = g.gas[INERT_N2][interior] * 50
    g.atmosphere[interior] = atmosphere_fixed.quantize_scalar(50.0)  # P_prev seed (solver refreshes)
    peak_before = atmosphere_fixed.dequantize(g.atmosphere.max())

    for _ in range(N_STEPS):
        sim.step()

    walls_after = int(g.solid.sum())
    peak_after = atmosphere_fixed.dequantize(g.atmosphere.max())

    assert walls_after < walls_before, (
        f"expected a wall to fail: before={walls_before} after={walls_after}")
    assert peak_after < peak_before, (
        f"expected pressure relief: peak {peak_before:.3f} -> {peak_after:.3f}")
    print(f"OK: pops_under_overpressure "
          f"(walls {walls_before}->{walls_after}, "
          f"peak {peak_before:.2f}->{peak_after:.2f})")


def test_no_spurious_pops_at_normal_pressure():
    """The same room at ~1.0 atm loses ZERO walls over the same ticks."""
    sim = _make_sim()
    g = sim.gmap

    interior = _interior_mask(g)

    # Same burstable wall as the pop test (hull ships at 0=never), so this
    # meaningfully checks that a normal-pressure ship does NOT trip the valve.
    for wid in np.unique(g.material[g.solid]):
        g.materials.burst_threshold[int(wid)] = 6.0

    from simulation import atmosphere_fixed
    g.atmosphere[interior] = atmosphere_fixed.quantize_scalar(1.0)  # normal interior pressure

    wall_set_before = g.solid.copy()

    for _ in range(N_STEPS):
        sim.step()

    assert np.array_equal(g.solid, wall_set_before), (
        f"normal-pressure room lost walls: "
        f"{int(wall_set_before.sum())} -> {int(g.solid.sum())}")
    print(f"OK: no_spurious_pops_at_normal_pressure "
          f"({int(wall_set_before.sum())} walls intact)")


if __name__ == "__main__":
    test_pops_under_overpressure()
    test_no_spurious_pops_at_normal_pressure()
    print("\nAll wall-failure tests passed.")
