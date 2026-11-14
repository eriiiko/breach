"""#140 end to end -- a squad at the door of an over-pressured room is not
blown out of it (re-stated at #183 G5p-b for the unit impulse law).

THE BUG (#139/#140, ship_demo match seed 1): a room at ~1.44 atm vented
through its doorway and ``exchange.apply_wave_push`` turned the pressure
gradient into a drift velocity: the bots stood in the doorway holding
MOVE_DIR for ~28 s and were shoved back. Erik's ruling (2026-10-31): wind
below a threshold moves no unit's position at all. His correction
(2026-11-10, the impulse law): SOFT wind never displaces a unit; a force
ABOVE the footing limit displaces it by its excess. Since G5p-b the wind is
a force on the unit impulse law (docs/unit_impulse_design_2026-11-10.md §4,
§7), so the door's half-second JET -- measured at the squad at up to ~5700
Pa, twenty holds -- floors the squad and pushes it by that excess (about a
third of a metre), and the vent's LASTING wind, far under the hold, moves
nobody: the #140 drift cannot come back.

The scene, rebuilt through the real ``Simulation`` (no stub): ship_demo; the
zombie room (the match's fragged room) raised by +0.44 atm through a
FieldEdit; every door of the ship OUTSIDE that room opened with
``destroy_wall`` (so the room vents into the ship, not into a closet); then
the room's two west doors opened. A squad stands in the antechamber in front
of the upper door with no orders; one walker holds MOVE_DIR west out through
the lower door.

PROPERTIES PROTECTED
  1. With the wind's input off (``[impulse] wind_push_mult = 0``) every
     standing unit keeps its EXACT position for the whole vent: nothing but
     the wind's force can move it. Breaks if any other displacement term
     returns to a wind / pressure row.
  2. With it on, a standing unit moves only while the wind at its body is
     over its hold, and stops within a second of the last such tick: its
     position is exact from then to the end of the vent (~9 s), its whole
     displacement under half a metre. Breaks if a sub-hold wind displaces
     (the #140 drift), or the law lets a pushed unit run on.
  3. A teammate's shove in the doorway during the lasting vent moves the
     shoved unit as far as the same shove with the wind off, to 5 cm, and it
     comes to rest. Breaks if the vent's wind switches the footing off
     behind a shove (critique A, B1: 23 m in draft 1).
  4. The walker leaving through the lower door gets out, and once the jet is
     over it is never moved faster than its calm cadence nor sideways.
     Breaks if the row pushes a walker along the lasting jet, or if the law
     freezes walkers for good.
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
SETTLE_TICKS = 24                 # a pushed unit is at rest within 1 s
SHOVE_TICK = 80                   # the lasting vent (the squad is up again)


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


def _vent(monkeypatch, wind_on, shove=False):
    """Run the vent; per tick each unit's position and whether the wind at
    its body (the 9c2 read of that tick) is over its standing hold."""
    from config import CFG
    from control_source import quantize_stick_direction
    from simulation import unit_impulse as ui
    from simulation.turning import place
    if not wind_on:
        monkeypatch.setattr(CFG.impulse, "wind_push_mult", 0.0)
    sim, squad, walker = _scene()
    t = sim.impulse_table
    shover = squad[2]
    place(shover, 0)                       # sq_c faces the door (east)
    dx_q, dy_q, _ = quantize_stick_direction(-1.0, 0.0, 0.0)
    rows = []
    for tick in range(VENT_TICKS):
        if shove and tick == SHOVE_TICK:
            sim.shove_intent(shover.id)
        sim.set_move_dir(walker.id, dx_q, dy_q, 0)  # throttle 0: the walk
        sim.step()
        row = {}
        for u in squad + [walker]:
            w = sim.gmap.footprint_wind_load(u.tile_y, u.tile_x, u.footprint)
            over = ui.wind_impulse_q(t, w) > t.hold_q(u) * u.mass_g
            row[u.name] = (u.x, u.y, over)
        rows.append(row)
    return sim, squad, walker, rows


def test_with_the_wind_off_nothing_moves_a_standing_unit(monkeypatch):
    """Property 1."""
    sim, squad, walker, rows = _vent(monkeypatch, wind_on=False)
    for u in squad:
        start = (rows[0][u.name][0], rows[0][u.name][1])
        assert start == (37.0, 20.0) or start == (37.0, 22.0) \
            or start == (34.0, 21.0)
        for tick, row in enumerate(rows):
            assert row[u.name][:2] == start, (u.name, tick)
    assert walker.x < WEST_WALL - 3


def test_the_jet_floors_the_squad_by_its_excess_and_the_vent_moves_nobody(
        monkeypatch):
    """Property 2."""
    from simulation.status import KNOCKED_DOWN
    sim, squad, walker, rows = _vent(monkeypatch, wind_on=True)
    starts = {"sq_a": (37.0, 20.0), "sq_b": (37.0, 22.0),
              "sq_c": (34.0, 21.0)}
    tile = sim.gmap.tile_size_m
    for u in squad:
        over = [i for i, r in enumerate(rows) if r[u.name][2]]
        assert over, f"the jet never reached {u.name}: the scene is vacuous"
        rest = over[-1] + SETTLE_TICKS
        assert rest < 2 * 24, (u.name, over[-1])
        final = rows[-1][u.name][:2]
        for tick in range(rest, VENT_TICKS):
            assert rows[tick][u.name][:2] == final, (u.name, tick)
        x0, y0 = starts[u.name]
        moved_m = float(np.hypot(final[0] - x0, final[1] - y0)) * tile
        assert 0.0 < moved_m < 0.5, (u.name, moved_m)
        assert not any(s.kind == KNOCKED_DOWN for s in u.statuses), (
            f"{u.name} is still down at the end of the vent")


def test_a_shove_in_the_lasting_vent_is_only_the_shoves(monkeypatch):
    """Property 3."""
    out = {}
    for wind_on in (False, True):
        with monkeypatch.context() as mp:
            sim, squad, walker, rows = _vent(mp, wind_on=wind_on, shove=True)
        tile = sim.gmap.tile_size_m
        moved = {}
        for u in squad[:2]:
            a = rows[SHOVE_TICK - 1][u.name][:2]
            b = rows[SHOVE_TICK + 2 * 24][u.name][:2]
            assert rows[-1][u.name][:2] == b, (wind_on, u.name)   # at rest
            moved[u.name] = float(np.hypot(b[0] - a[0], b[1] - a[1])) * tile
        out[wind_on] = moved
    target = max(out[False], key=out[False].get)
    assert out[False][target] > 0.2          # the shove landed (stumble)
    assert abs(out[True][target] - out[False][target]) <= 0.05, out


def test_the_walker_gets_out_and_keeps_its_cadence_after_the_jet(
        monkeypatch):
    """Property 4."""
    sim, squad, walker, rows = _vent(monkeypatch, wind_on=True)
    calm_step = 65536 / sim.speed_table.walk_cost_q   # the [speeds] walk
    over = [i for i, r in enumerate(rows) if r["walker"][2]]
    after = (over[-1] if over else 0) + SETTLE_TICKS
    for tick in range(after + 1, VENT_TICKS):
        (x1, y1, _), (x0, y0, _) = rows[tick]["walker"], rows[tick - 1]["walker"]
        assert y1 == y0, f"tick {tick + 1}: the walker drifted in y"
        assert -calm_step - 1e-12 <= x1 - x0 <= 0.0, (
            f"tick {tick + 1}: walker step {x1 - x0} vs calm {calm_step}")
    assert walker.x < WEST_WALL - 3, (
        f"the walker never got out of the room (x -> {walker.x})")
