"""#121 — the pressure push must let a unit walk through a legacy door.

A legacy ``MAT_DOOR`` tile is walkable but solid to gas: it stores
atmosphere 0. ``exchange.apply_wave_push`` once read that 0 as a pressure in
the footprint gradient and shoved any unit whose footprint edge stood on the
door back off it, 0.2 tile a tick against a walk of 0.056 -- no unit under
MOVE_DIR could cross a door (found by the bot team on ship_demo). Written
through ``level_lib``; driven through the per-tick intent facade.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import level_lib  # noqa: E402
import level_loader  # noqa: E402
from control_source import quantize_stick_direction  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.materials import MAT_AIR, MAT_DOOR, MAT_HULL  # noqa: E402
from simulation.orders import ORDER_MOVE_ATTACK  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.unit import Unit  # noqa: E402

W, H, DOOR_X = 40, 16, 20


def _level(tmp_path):
    lvl = tmp_path / "doorway"
    lvl.mkdir()
    level_lib.write_level_header(lvl, name="doorway", tile_size_m=1.0)
    tm = np.full((H, W), MAT_HULL, dtype=np.int32)
    tm[1:H - 1, 1:W - 1] = MAT_AIR
    tm[:, DOOR_X] = MAT_HULL                 # a wall across the box ...
    tm[5:11, DOOR_X] = MAT_DOOR              # ... with a 6-tile legacy door
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (W, H)).save(lvl / "diffuse.png")
    return level_loader.load("doorway", str(tmp_path))


def test_a_walking_unit_crosses_a_legacy_door_on_a_quiet_map(tmp_path):
    """Property: on a quiet map (both sides at the same pressure), a unit
    holding MOVE_DIR east walks clean through a legacy MAT_DOOR tile and
    ends well past it. Breaks if the push's footprint gradient reads the
    door tile's stored atmosphere (0) as a pressure again -- the unit then
    stalls with its footprint edge on the door."""
    lvl = _level(tmp_path)
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    u = Unit("walker", x=12.5, y=6.5, team=0, footprint=3)
    sim.add_unit(u)
    assert sim.gmap.solid[7, DOOR_X]          # solid to gas ...
    assert sim.gmap.is_passable_block(6, DOOR_X, 3)   # ... yet walkable
    dx, dy, _ = quantize_stick_direction(1.0, 0.0, 0.0)
    for _ in range(24 * 12):                  # 12 s ~ 16 tiles of walking
        sim.set_move_dir(u.id, dx, dy, ORDER_MOVE_ATTACK)
        sim.step()
    assert u.x > DOOR_X + 3, u.x              # footprint wholly past it
