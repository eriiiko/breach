"""#140 end to end -- a squad at the door of an over-pressured room is not
blown out of it.

THE BUG (#139/#140, ship_demo match seed 1): a room at ~1.44 atm vented
through its doorway and ``exchange.apply_wave_push`` turned the pressure
gradient into a drift velocity: the bots stood in the doorway holding
MOVE_DIR for ~28 s and were shoved back. Erik's ruling (2026-10-31): wind
moves no unit's position at all; it only slows a walker (above
``wind_q_slow_pa``) or lays it down (``wind_q_fall_pa``).

The scene, rebuilt through the real ``Simulation`` (no stub): ship_demo; the
zombie room (the match's fragged room) raised by +0.44 atm through a
FieldEdit; every door of the ship OUTSIDE that room opened with
``destroy_wall`` (so the room vents into the ship, not into a closet); then
the room's two west doors opened. A squad stands in the antechamber in front
of the upper door with no orders; one walker holds MOVE_DIR west out through
the lower door.

PROPERTIES PROTECTED
  1. Every standing unit keeps its EXACT float position for the whole vent.
     Breaks if any displacement term returns to a wind / pressure row.
  2. The walker is never moved faster than its calm cadence (each tick's
     step is at most ``1 / calm ticks-per-tile`` tiles, west only) and still
     gets out of the room. Breaks if the row pushes a walker along the jet,
     or if the law freezes walkers for good.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bp = pytest.importorskip("breach_physics")

# The zombie room of ship_demo (its [[spawn_room]] team 255 at x 41, y 18,
# 18 x 19): interior rows 18..36, cols 41..58; walls at col 40 / 59 and rows
# 17 / 37. Its west wall carries two 4-tile doors, rows 21..24 and 29..32.
ROOM_ROWS = (18, 36)
ROOM_COLS = (41, 58)
WEST_WALL = 40
UPPER_DOOR = range(21, 25)
LOWER_DOOR = range(29, 33)
OVERPRESSURE_ATM = 0.44          # 1.0 -> ~1.44 atm, the #139 measurement
VENT_TICKS = 240                  # 10 s at 24 tps; the jet is over in < 1 s


def _scene():
    from level_loader import load as load_level
    from simulation import Simulation
    from simulation.field_edit import EditMode, FieldEdit, Region
    from simulation.materials import MAT_DOOR
    from simulation.ruleset import ContinuousRealtime
    from simulation.unit import Unit

    level = load_level("ship_demo", levels_dir=str(ROOT / "levels"))
    sim = Simulation(level, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    g = sim.gmap
    for r in list(UPPER_DOOR) + list(LOWER_DOOR):
        assert g.material[r, WEST_WALL] == MAT_DOOR, "ship_demo's room moved"
    sim.edit_queue.enqueue(FieldEdit(
        "atmosphere", Region.RECT,
        (ROOM_ROWS[0], ROOM_COLS[0], ROOM_ROWS[1], ROOM_COLS[1]),
        OVERPRESSURE_ATM, EditMode.ADD))
    sim.set_paused(False)
    for _ in range(30):
        sim.step()
    # Open every door OUTSIDE the room's walls: the room will vent into the
    # whole deck through its west doors.
    for (r, c) in np.argwhere(g.material == MAT_DOOR).tolist():
        on_room = (ROOM_ROWS[0] - 1 <= r <= ROOM_ROWS[1] + 1
                   and ROOM_COLS[0] - 1 <= c <= ROOM_COLS[1] + 1)
        if not on_room:
            g.destroy_wall(r, c)
    for _ in range(24):
        sim.step()
    squad = [Unit("sq_a", x=37.0, y=20.0, team=0),
             Unit("sq_b", x=37.0, y=22.0, team=0),
             Unit("sq_c", x=34.0, y=21.0, team=0)]
    walker = Unit("walker", x=42.0, y=29.0, team=0)
    for u in squad + [walker]:
        sim.add_unit(u)
    p_room = int(g.atmosphere[27, 50]) / 65536.0  # display only
    assert p_room > 1.3, f"the room is not over-pressured ({p_room:.3f} atm)"
    for r in list(UPPER_DOOR) + list(LOWER_DOOR):
        g.destroy_wall(r, WEST_WALL)        # the door opens
    return sim, squad, walker


def test_a_squad_at_a_venting_door_is_not_moved_and_a_walker_keeps_cadence():
    from control_source import quantize_stick_direction
    from config import CFG
    from simulation.orders import ORDER_MOVE_ATTACK

    sim, squad, walker = _scene()
    start = {u.id: (u.x, u.y) for u in squad}
    calm_step = 1.0 / int(CFG.movement.marine_attack_ticks_per_tile)
    dx_q, dy_q, _ = quantize_stick_direction(-1.0, 0.0, 0.0)
    x0 = walker.x
    for t in range(VENT_TICKS):
        wx, wy = walker.x, walker.y
        sim.set_move_dir(walker.id, dx_q, dy_q, ORDER_MOVE_ATTACK)
        sim.step()
        for u in squad:
            assert (u.x, u.y) == start[u.id], (
                f"tick {t + 1}: standing {u.name} moved from {start[u.id]} "
                f"to {(u.x, u.y)}")
        step_x = walker.x - wx
        assert walker.y == wy, f"tick {t + 1}: the walker drifted in y"
        assert -calm_step - 1e-12 <= step_x <= 0.0, (
            f"tick {t + 1}: walker step {step_x} vs calm {calm_step}")
    assert walker.x < WEST_WALL - 3, (
        f"the walker never got out of the room (x {x0} -> {walker.x})")
