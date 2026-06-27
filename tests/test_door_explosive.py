"""Regression: detonating a door-explosive must not crash.

The FieldEdit migration (65aebb5) changed ``apply_explosion`` /
``add_explosion_smoke`` to take the ``EditQueue`` and updated the grenade path,
but MISSED ``process_door_explosives`` -> detonating a door-explosive raised
``TypeError: apply_explosion() missing 1 required positional argument:
'wall_damage'`` (fixed in f6f8753). This locks that call path so a future
signature drift can't silently re-break it.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_door_explosive.py -q
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
from simulation.unit import Unit
from simulation.orders import Order, ORDER_EXPLOSIVE, DET_START_PHASE1
from simulation.combat import process_door_explosives


def _room_level():
    """A small hull-walled room (CSV: 0 vacuum, 1 hull, 4 air)."""
    h = w = 12
    tm = np.zeros((h, w), dtype=np.int32)
    tm[2:10, 2:10] = 1   # hull box
    tm[3:9, 3:9] = 4     # interior air
    return LevelData(
        name="door_explosive_test", version="1", path=Path("."),
        tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."),
    )


def test_door_explosive_detonates_without_crash():
    """process_door_explosives runs (was a TypeError) and damages walls in radius."""
    sim = Simulation(_room_level(), seed=7, breach_physics=bp, enable_recorder=False)

    u = Unit("breacher", x=5, y=5, team=0)   # interior air tile
    sim.add_unit(u)
    # explosive order targeting a hull tile on the ring: Order(type, target_fx, target_fy, phase, ...)
    u.orders.append(Order(ORDER_EXPLOSIVE, 5, 2, 0, det_slot=DET_START_PHASE1))

    # S3b: wall_hp is int32 Q16.16 — sum in int64 so a hull-dense map (HP*65536 per
    # tile) can't overflow the default int32 accumulator before the compare.
    hp_before = float(sim.gmap.wall_hp.astype(np.int64).sum())

    # THE REGRESSION: this raised before the fix.
    process_door_explosives(
        sim.gmap, sim.edit_queue, sim.units, DET_START_PHASE1, sim.rng)
    sim.edit_queue.flush(sim.gmap, sim.rng)

    hp_after = float(sim.gmap.wall_hp.astype(np.int64).sum())
    assert hp_after < hp_before, (
        f"door explosive did not damage any wall: {hp_before} -> {hp_after}")


def test_full_step_with_door_explosive_order():
    """A full Simulation.step() with a queued door-explosive order must not crash."""
    sim = Simulation(_room_level(), seed=7, breach_physics=bp, enable_recorder=False)
    sim.set_paused(False)
    u = Unit("breacher", x=5, y=5, team=0)
    sim.add_unit(u)
    u.orders.append(Order(ORDER_EXPLOSIVE, 5, 2, 0, det_slot=DET_START_PHASE1))
    for _ in range(3):
        sim.step()   # exercises the DET_START_PHASE1 call site in step()
