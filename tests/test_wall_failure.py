"""Tests for over-pressure wall failure — the emergent pressure-relief valve
(ch.04 §5, docs/architecture/engine/04_atmosphere_and_pressure.md).

A sealed room that keeps absorbing grenades would otherwise build pressure
without limit. Instead, each tick (after physics) any wall holding a pressure
differential above its material's ``burst_threshold`` fails and vents. Over-
pressured clusters self-breach in a chain until the gradient relaxes.

The spread is a TRUE DIFFERENTIAL across a wall's open sides: solid
neighbours are skipped (they are more wall, not a side), exposed vacuum is a
real side holding 0. Consequences under test: equal pressure on both sides
never bursts, and only 1-tile-deep membranes can burst at all (a >=2-thick
slab has no tile with two open sides).

Four deterministic, headless cases:

1. POPS UNDER OVER-PRESSURE: a small sealed room pumped far above any
   threshold loses at least one wall, and the peak interior pressure afterwards
   is LOWER than before (relief happened).
2. NO SPURIOUS POPS: the SAME room at normal pressure (~1.0), stepped the same
   number of ticks, loses ZERO walls (the threshold is not tripped by a
   normally-pressurised ship).
3. EQUAL PRESSURE HOLDS / DIFFERENTIAL POPS: two rooms split by a burstable
   1-thick wood divider. Both rooms pumped equally -> the divider holds
   (regression: the old spread counted solid neighbours as 0, so a wall
   pressurised equally on both sides still burst). Only one room pumped ->
   the divider pops.
4. THICK WALL HOLDS: the same two rooms with a 2-thick divider, one room
   pumped -> zero pops (no divider tile has two open sides).

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
    # arc #54 P-G1b: a direct bulk-N write is a seam violation, and under D1
    # it has a VISIBLE consequence -- `gas_energy` is the stored truth now, so
    # multiplying N by 50 without it divides this room's `E/N` by 50: the room
    # gets 50x COLDER instead of 50x denser and `p = C*N*T_abs` does not move
    # at all, so no wall ever pops. Re-derive the energy over exactly the
    # cells this scenario just authored (design 2.2's seeding act, restricted
    # to a selection) and the intended 50-atm overpressure exists again.
    g.reseed_gas_energy(interior)
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


def _two_room_level(divider_thick: int = 1):
    """Two rooms split by a wood divider (v1 code 2 = MAT_WOOD), hull ring
    outside, vacuum around. The hull stays at its shipped
    ``burst_threshold = 0`` (never bursts) so the tests isolate the divider.

    Layout (16x12), '.' = vacuum, '#' = hull, 'w' = wood divider:

        ................
        ................
        ..############..
        ..#    w     #..
        ..#    w     #..   divider column(s) at x = 7 (+8 when 2-thick)
        ..#    w     #..
        ..#    w     #..
        ..#    w     #..
        ..#    w     #..
        ..############..
        ................
        ................
    """
    h, w = 12, 16
    tm = np.zeros((h, w), dtype=np.int32)   # all vacuum
    tm[2:10, 2:14] = 1                       # hull box
    tm[3:9, 3:13] = 4                        # carve interior air
    tm[3:9, 7:7 + divider_thick] = 2         # wood divider
    return LevelData(
        name="two_room_test",
        version="1",
        path=Path("."),
        tilemap=tm,
        tile_size_m=1.0,
        diffuse_path=Path("."),
    )


def _make_two_room_sim(divider_thick: int = 1):
    from simulation.materials import MAT_HULL, MAT_WOOD
    level = _two_room_level(divider_thick)
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    sim.set_paused(False)
    g = sim.gmap
    # Shipped-intent thresholds, pinned explicitly so config re-tuning can't
    # silently change what these tests exercise: hull never pressure-bursts,
    # the wood divider bursts above 6.0 of REAL differential.
    g.materials.burst_threshold[MAT_HULL] = 0.0
    g.materials.burst_threshold[MAT_WOOD] = 6.0
    return sim


def _pump_rooms(g, left_only: bool, factor: int = 50):
    """Scale the bulk gas up in the left room (``left_only``) or the whole
    interior (EOS P3: pressure is materialized from (N, T), so overpressure
    is created through N). The divider starts at x=7, so ``xs < 7`` is the
    left room for any divider thickness."""
    from simulation import atmosphere_fixed
    from simulation.gases import O2, INERT_N2
    interior = _interior_mask(g)
    xs = np.arange(g.atmosphere.shape[1])[None, :]
    room = interior & (xs < 7) if left_only else interior
    for sp in (O2, INERT_N2):
        g.gas[sp][room] = g.gas[sp][room] * factor
    # arc #54 P-G1b: re-derive the stored energy over the cells just authored
    # -- see the note in test_pops_under_overpressure for why a direct bulk-N
    # write without this makes the room COLDER rather than denser.
    g.reseed_gas_energy(room)
    g.atmosphere[room] = atmosphere_fixed.quantize_scalar(float(factor))
    return room


def test_equal_pressure_holds_differential_pops():
    """The divider holds when both rooms are pumped equally (the old spread
    counted solid neighbours as p=0 and burst it — regression), and pops when
    only one room is pumped (a real differential)."""
    from simulation.materials import MAT_WOOD

    # Both rooms pumped equally -> zero pops anywhere.
    sim = _make_two_room_sim(divider_thick=1)
    g = sim.gmap
    _pump_rooms(g, left_only=False)
    walls_before = g.solid.copy()
    for _ in range(N_STEPS):
        sim.step()
    assert np.array_equal(g.solid, walls_before), (
        "equal pressure on both sides burst a wall (differential regression): "
        f"{int(walls_before.sum())} -> {int(g.solid.sum())}")

    # Only the left room pumped -> the wood divider pops; the hull holds.
    sim = _make_two_room_sim(divider_thick=1)
    g = sim.gmap
    _pump_rooms(g, left_only=True)
    wood_before = int((g.material == MAT_WOOD).sum())
    hull_walls_before = int((g.solid & (g.material != MAT_WOOD)).sum())
    for _ in range(N_STEPS):
        sim.step()
    wood_after = int((g.material == MAT_WOOD).sum())
    hull_walls_after = int((g.solid & (g.material != MAT_WOOD)).sum())
    assert wood_after < wood_before, (
        f"expected the divider to pop under differential: "
        f"wood {wood_before} -> {wood_after}")
    assert hull_walls_after == hull_walls_before, (
        f"hull (burst_threshold=0) must never pressure-burst: "
        f"{hull_walls_before} -> {hull_walls_after}")
    print(f"OK: equal_pressure_holds_differential_pops "
          f"(wood {wood_before}->{wood_after}, hull intact)")


def test_thick_wall_holds():
    """A 2-thick divider has no tile with two open sides -> spread 0 -> it
    holds any differential (thick walls breach via damage, not pressure)."""
    sim = _make_two_room_sim(divider_thick=2)
    g = sim.gmap
    _pump_rooms(g, left_only=True)
    walls_before = g.solid.copy()
    for _ in range(N_STEPS):
        sim.step()
    assert np.array_equal(g.solid, walls_before), (
        f"2-thick divider lost tiles to over-pressure: "
        f"{int(walls_before.sum())} -> {int(g.solid.sum())}")
    print(f"OK: thick_wall_holds ({int(walls_before.sum())} walls intact)")


if __name__ == "__main__":
    test_pops_under_overpressure()
    test_no_spurious_pops_at_normal_pressure()
    test_equal_pressure_holds_differential_pops()
    test_thick_wall_holds()
    print("\nAll wall-failure tests passed.")
