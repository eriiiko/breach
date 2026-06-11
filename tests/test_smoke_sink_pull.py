"""Smoke-side sink-pull toward the nearest breach (smoke v2, step 2).

With aggressive atmosphere diffusion (``d_atm = 200``) the interior pressure
flattens fast, so the wind dies deep in a room and smoke just sits there even
when the hull is breached to vacuum. The fix (ch.05 smoke v2) is a *dial-able*
smoke-side sink-pull: a per-cell unit vector field (``GameMap.sink_x/sink_y``),
built by a BFS over air cells toward the nearest exposed-vacuum tile, is added
(scaled by ``smoke_sink_strength``) to the smoke solver's advecting velocity in
its semi-Lagrangian back-trace. It is a bias inside *smoke* transport only — it
never touches the pressure field.

Safe by construction: with no breach the sink field is all-zero, so a sealed
room behaves bit-identically to the plain semi-Lagrangian advection (step 1).

Three cases:

1. BREACHED ROOM CLEARS — a smoke-filled room with one hull tile opened to
   vacuum drops below 10 % of its initial smoke within a reasonable number of
   ticks at the default sink strength.
2. SEALED ROOM RETAINS — the SAME room with NO breach keeps its smoke; the
   result is exactly the no-sink (step-1) behaviour, i.e. no extra loss from
   the sink.
3. SINK FIELD SHAPE — the sink field is zero everywhere with no breach, and
   non-zero pointing toward the opening when there is one.

Run:
    C:/Users/steen/anaconda3/python.exe tests/test_smoke_sink_pull.py
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


def _edge_room_level():
    """A hull-walled room whose LEFT wall sits on the map edge, so destroying a
    left-wall tile produces a real hull breach to vacuum.

    CSV codes: 0 = vacuum, 1 = hull, 4 = interior air. Layout (12x12),
    '.' = vacuum, '#' = hull, ' ' = interior air:

        ############
        #          #
        #          #
        #          #
        #          #
        #          #
        #          #
        #          #
        #          #
        #          #
        #          #
        ############

    The whole border is hull on the map edge, so ``destroy_wall`` on any border
    tile flips it to exposed vacuum (a breach), and the room then vents. The
    interior (rows/cols 1..10) is air.
    """
    h = w = 12
    tm = np.ones((h, w), dtype=np.int32)   # all hull
    tm[1:11, 1:11] = 4                       # carve interior air
    return LevelData(
        name="edge_room_test",
        version="1",
        path=Path("."),
        tilemap=tm,
        tile_size_m=1.0,
        diffuse_path=Path("."),
    )


def _interior_mask(gmap: GameMap):
    """Boolean mask of interior air tiles (not solid, not vacuum)."""
    return (~gmap.solid) & (~gmap.is_vacuum)


def _make_sim():
    level = _edge_room_level()
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    sim.set_paused(False)
    return sim


def _fill_smoke(g, interior, value=1.0):
    g.smoke[:] = 0.0
    g.smoke[interior] = value


# --------------------------------------------------------------------------
# 1. Breached room clears
# --------------------------------------------------------------------------
def test_breached_room_clears():
    """A breached, smoke-filled room vents below 10 % within a dozen-ish ticks."""
    sim = _make_sim()
    g = sim.gmap
    interior = _interior_mask(g)
    assert interior.any(), "test level has no interior air"

    _fill_smoke(g, interior, 1.0)
    total0 = float(g.smoke.sum())
    assert total0 > 0.0

    # Open one hull tile on the map edge -> exposed vacuum (a breach).
    g.destroy_wall(6, 0)
    assert g.is_vacuum[6, 0] and not g.solid[6, 0], "breach did not expose vacuum"

    n_ticks = 0
    max_ticks = 40
    while float(g.smoke.sum()) > 0.10 * total0 and n_ticks < max_ticks:
        sim.step()
        n_ticks += 1

    frac = float(g.smoke.sum()) / total0
    assert frac < 0.10, (
        f"breached room did not clear: {frac*100:.1f}% remains "
        f"after {n_ticks} ticks")
    print(f"OK: breached_room_clears "
          f"({frac*100:.1f}% smoke remains after {n_ticks} ticks)")


# --------------------------------------------------------------------------
# 2. Sealed room retains (and matches the no-sink behaviour exactly)
# --------------------------------------------------------------------------
def test_sealed_room_retains():
    """The SAME room with NO breach keeps its smoke, and the sink adds NO loss.

    We run two sealed sims for the same number of ticks: one at the default
    sink strength, one with the sink disabled (strength 0). With no breach the
    sink field is all-zero, so the two must produce an IDENTICAL smoke field —
    proving the sink introduces no extra loss in a sealed room.
    """
    n_ticks = 20

    # (a) default sink strength.
    sim = _make_sim()
    g = sim.gmap
    interior = _interior_mask(g)
    _fill_smoke(g, interior, 1.0)
    total0 = float(g.smoke.sum())
    for _ in range(n_ticks):
        sim.step()
    sealed_default = g.smoke.copy()
    frac = float(sealed_default.sum()) / total0

    # (b) same room, sink disabled — the step-1 (no-sink) baseline.
    sim0 = _make_sim()
    g0 = sim0.gmap
    sim0.physics_runner.smoke.sink_strength = 0.0
    interior0 = _interior_mask(g0)
    _fill_smoke(g0, interior0, 1.0)
    for _ in range(n_ticks):
        sim0.step()
    sealed_nosink = g0.smoke.copy()

    # Sealed room must hold most of its smoke (it does not vent).
    assert frac > 0.9, (
        f"sealed room lost smoke unexpectedly: {frac*100:.1f}% remains")
    # And the sink must have made NO difference (no breach -> sink field zero).
    assert np.array_equal(sealed_default, sealed_nosink), (
        "sink-pull changed a sealed room (should be identical to no-sink)")
    print(f"OK: sealed_room_retains "
          f"({frac*100:.1f}% smoke remains; identical to no-sink)")


# --------------------------------------------------------------------------
# 3. Sink-field shape
# --------------------------------------------------------------------------
def test_sink_field_zero_without_breach_nonzero_with():
    """No breach -> sink field all-zero. Breach -> non-zero, toward the opening."""
    sim = _make_sim()
    g = sim.gmap

    # No breach yet: sink field must be exactly zero everywhere.
    sx, sy = g.sink_fields()
    assert not np.any(sx) and not np.any(sy), (
        "sink field is non-zero with no breach")

    # Open a breach on the LEFT edge; the field must rebuild (dirty flag) and
    # point air cells toward the opening.
    g.destroy_wall(6, 0)
    sx, sy = g.sink_fields()
    interior = _interior_mask(g)
    assert np.any(sx[interior]) or np.any(sy[interior]), (
        "sink field stayed zero after a breach")

    # The air tile just inside the breach (6, 1) should point LEFT toward the
    # opening at (6, 0): sink_x negative (toward smaller column), sink_y ~ 0.
    assert sx[6, 1] < 0.0, (
        f"breach-adjacent cell does not point at the opening: "
        f"sink_x={sx[6, 1]}, sink_y={sy[6, 1]}")

    # An interior cell further from the breach should also have a net leftward
    # pull (the BFS gradient descends toward the single opening).
    assert sx[6, 5] <= 0.0, (
        f"interior cell points away from the breach: sink_x={sx[6, 5]}")

    print("OK: sink_field_zero_without_breach_nonzero_with "
          "(zero with no breach; points toward opening with one)")


if __name__ == "__main__":
    test_breached_room_clears()
    test_sealed_room_retains()
    test_sink_field_zero_without_breach_nonzero_with()
    print("\nAll smoke sink-pull tests passed.")
