"""Tests for the mobility coefficient (mobility design — movement as a
coefficient, replacing the old ``passable`` boolean).

Covers the three behavioural seams of the v1 build:
  - the walkability predicates read ``mobility > 0`` (furniture enterable,
    walls not) — the predicate redirect (design §2/§8)
  - the movement-cadence penalty: an all-furniture footprint step costs 2.5x
    the all-air base ticks, pinned to exact integers through the §3 half-up math
  - the speed_fn determinism contract (§4.1): returns ``int``, run-to-run
    identical, takes a baked int speed_class (NEVER the unit object)

Run:
    C:/Users/steen/anaconda3/python.exe tests/test_mobility.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from level_loader import load as load_level
from simulation.gamemap import GameMap
from simulation.materials import MAT_AIR, MAT_FURNITURE, MAT_HULL, MAT_WOOD
from simulation.movement import FootprintSamples, default_speed, half_up, MOBILITY_ONE


# ---------------------------------------------------------------------------
# Enterability: the predicates read mobility > 0
# ---------------------------------------------------------------------------
def test_furniture_tile_is_enterable():
    """A single furniture tile (mobility 400 > 0) is enterable; a wall is not."""
    g = GameMap(load_level("unhcr_vessel"))
    ys, xs = np.where((g.material == MAT_AIR) & ~g.is_vacuum)
    y, x = int(ys[len(ys) // 2]), int(xs[len(ys) // 2])

    # Air: enterable.
    assert g.is_passable(y, x)

    # Furniture: now enterable (the headline behavioural change).
    g.material[y, x] = MAT_FURNITURE
    g.on_tile_changed(y, x)
    assert g.is_passable(y, x), "furniture (mobility 400) must be enterable"

    # Wall (hull, mobility 0): NOT enterable.
    g.material[y, x] = MAT_HULL
    g.on_tile_changed(y, x)
    assert not g.is_passable(y, x), "hull (mobility 0) must NOT be enterable"
    print("OK: furniture_tile_is_enterable")


def test_block_enterable_requires_every_tile_positive():
    """is_passable_block is geometry: ANY single mobility-0 tile blocks the
    footprint, but an all-furniture block IS enterable (§4)."""
    g = GameMap(load_level("unhcr_vessel"))
    # Find a 3x3 all-air block.
    ys, xs = np.where((g.material == MAT_AIR) & ~g.is_vacuum)
    found = None
    for y, x in zip(ys.tolist(), xs.tolist()):
        if g.is_passable_block(y, x, 3):
            found = (y, x)
            break
    assert found is not None, "need an enterable 3x3 air block to test"
    y, x = found

    # Paint the whole 3x3 to furniture: still enterable (all mobility 400 > 0).
    for dy in range(3):
        for dx in range(3):
            g.material[y + dy, x + dx] = MAT_FURNITURE
            g.on_tile_changed(y + dy, x + dx)
    assert g.is_passable_block(y, x, 3), "all-furniture block must be enterable"

    # Flip ONE tile to a wall: the whole block is now blocked.
    g.material[y + 1, x + 1] = MAT_WOOD
    g.on_tile_changed(y + 1, x + 1)
    assert not g.is_passable_block(y, x, 3), "one wall tile blocks the footprint"
    print("OK: block_enterable_requires_every_tile_positive")


# ---------------------------------------------------------------------------
# Cadence penalty — exact integers through the half-up math (§3, §4)
# ---------------------------------------------------------------------------
def test_half_up_is_pure_integer():
    """half_up((num+den//2)//den): rounds .5 up, never round()-on-float."""
    assert half_up(45 * 2, 2) == 45      # exact
    assert half_up(3, 2) == 2            # 1.5 -> 2 (up)
    assert half_up(1, 2) == 1            # 0.5 -> 1 (up)
    assert half_up(162000, 9000) == 18   # all-air @ base 18
    assert isinstance(half_up(5, 3), int)
    print("OK: half_up_is_pure_integer")


def _samples(mobilities):
    return FootprintSamples(mobility=list(mobilities))


def test_all_air_footprint_keeps_base():
    """An all-air 3x3 footprint leaves the base cadence untouched (multiplier 1)."""
    base = 18  # marine_attack @ 24 tps (0.75 s * 24)
    cost = default_speed(_samples([MOBILITY_ONE] * 9), base)
    assert cost == 18, f"all-air must keep base 18, got {cost}"
    print("OK: all_air_footprint_keeps_base")


def test_all_furniture_footprint_costs_2_5x():
    """An all-furniture 3x3 step costs 2.5x the base: base 18 -> 45 (the
    spec-pinned integer), exactly via half_up(18*9*1000, 9*400)."""
    base = 18
    cost = default_speed(_samples([400] * 9), base)
    # half_up(18*9*1000, 3600) = half_up(162000, 3600) = (162000+1800)//3600 = 45
    assert cost == 45, f"all-furniture base 18 must cost 45 (2.5x), got {cost}"

    # Pin a second base too (cover move = 12 @ 24 tps -> 30).
    assert default_speed(_samples([400] * 9), 12) == 30
    print("OK: all_furniture_footprint_costs_2_5x")


def test_mixed_footprint_area_averages():
    """A mixed footprint averages per §4 (size is not a liability): 8 air + 1
    furniture dilutes the obstacle by body area, staying near base speed."""
    base = 18
    # 8x1000 + 1x400 = 8400; half_up(18*9*1000, 8400) = half_up(162000, 8400)
    # = (162000 + 4200)//8400 = 166200//8400 = 19
    cost = default_speed(_samples([MOBILITY_ONE] * 8 + [400]), base)
    assert cost == 19, f"8-air+1-furniture must be ~base (19), got {cost}"

    # Half air, half furniture (5 air + 4 furniture): 5000+1600 = 6600;
    # half_up(162000, 6600) = (162000+3300)//6600 = 165300//6600 = 25
    cost2 = default_speed(_samples([MOBILITY_ONE] * 5 + [400] * 4), base)
    assert cost2 == 25, f"5-air+4-furniture must be 25, got {cost2}"
    print("OK: mixed_footprint_area_averages")


# ---------------------------------------------------------------------------
# Determinism contract (§4.1)
# ---------------------------------------------------------------------------
def test_speed_fn_returns_int_and_is_deterministic():
    """speed_fn is pure / integer-out / run-to-run identical."""
    samples = _samples([MOBILITY_ONE, 400, 400, MOBILITY_ONE])
    first = default_speed(samples, 9)
    assert isinstance(first, int), "tick_cost must be a Python int"
    # Run-to-run identical (no hidden state, no float).
    for _ in range(50):
        assert default_speed(samples, 9) == first
    print("OK: speed_fn_returns_int_and_is_deterministic")


def test_speed_fn_takes_baked_int_not_unit_object():
    """The determinism contract: speed_fn takes a baked int speed_class, never
    the unit object (a float field on the unit must not leak in)."""
    import inspect

    sig = inspect.signature(default_speed)
    params = list(sig.parameters)
    assert params == ["footprint_samples", "speed_class"], params

    # A unit-like object as speed_class is a misuse and must fail loudly (it is
    # multiplied as an int) — proving the function expects an int, not a unit.
    class FakeUnit:
        base_speed = 1.5  # the float field §4.1 forbids leaking in

    try:
        default_speed(_samples([MOBILITY_ONE] * 9), FakeUnit())
        raised = False
    except TypeError:
        raised = True
    assert raised, "speed_class must be an int; a unit object must not work"
    print("OK: speed_fn_takes_baked_int_not_unit_object")


def test_speed_fn_floors_at_one_tick():
    """A step is never instantaneous: cost floors at 1 even for fast tiles
    (the §3 mobility>1 coarse-quantisation floor)."""
    # mobility 1000000 (a hypothetical conveyor) with base 1 -> sub-1, floored.
    assert default_speed(_samples([10_000_000]), 1) == 1
    print("OK: speed_fn_floors_at_one_tick")


if __name__ == "__main__":
    test_furniture_tile_is_enterable()
    test_block_enterable_requires_every_tile_positive()
    test_half_up_is_pure_integer()
    test_all_air_footprint_keeps_base()
    test_all_furniture_footprint_costs_2_5x()
    test_mixed_footprint_area_averages()
    test_speed_fn_returns_int_and_is_deterministic()
    test_speed_fn_takes_baked_int_not_unit_object()
    test_speed_fn_floors_at_one_tick()
    print("\nAll mobility tests passed.")
