"""Units are soft, porous bodies for gas/pressure (ch.04 §3b).

`stamp_units` no longer paints a living unit into the boolean `obstacles`
mask (which drives the C++ hard-zeroing wall BCs). Instead it stamps the
unit's footprint into the *continuous* `dyn_permeability` field with a
PARTIAL value (`CFG.physics.unit_permeability`, default 0.5, overridable
per unit via a `unit.permeability` hook). Smoke/air therefore seep past a
body (slowed by the `face = min(perm)` flux weighting) rather than being
perfectly blocked.

Units still cast solid light shadows: the `dyn_light_atten` unit stamp is
unchanged (default [1,1,1] = opaque).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_unit_permeability.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import CFG
from level_loader import load as load_level
from simulation.gamemap import GameMap
from simulation.unit import Unit


def _clear_air_anchor(g: GameMap, footprint: int = 3):
    """Interior anchor whose footprint-sized block is fully passable air."""
    h, w = g.material.shape
    for y in range(2, h - footprint - 2):
        for x in range(2, w - footprint - 2):
            if g.is_passable_block(y, x, footprint):
                if not g.light_atten[y:y + footprint, x:x + footprint].any():
                    return y, x
    raise AssertionError("no clear-air footprint found in level")


def test_unit_is_soft_body_not_obstacle():
    """(a) unit tiles NOT in obstacles, (b) dyn_permeability == partial value,
    (c) dyn_light_atten still opaque (shadow unchanged)."""
    g = GameMap(load_level("unhcr_vessel"))
    ay, ax = _clear_air_anchor(g)
    u = Unit("U1", x=ax, y=ay, team=0)

    g.stamp_units([u])

    expected_perm = float(getattr(CFG.physics, "unit_permeability", 0.5))
    assert expected_perm == 0.5, "config default unit_permeability should be 0.5"

    for (tx, ty) in u.occupied_tiles():
        # (a) NOT an obstacle anymore.
        assert not g.obstacles[ty, tx], \
            f"unit tile ({tx},{ty}) still marked as obstacle"
        # (b) partial permeability (porous body), not 0 (sealed) nor 1 (open).
        assert g.dyn_permeability[ty, tx] == expected_perm, \
            f"unit tile ({tx},{ty}) perm {g.dyn_permeability[ty, tx]} != {expected_perm}"
        # (c) still casts a solid (opaque) shadow.
        assert np.array_equal(g.dyn_light_atten[ty, tx], [1.0, 1.0, 1.0]), \
            f"unit tile ({tx},{ty}) shadow changed: {g.dyn_light_atten[ty, tx]}"

    # obstacles is now walls-only == solid set.
    assert np.array_equal(g.obstacles, g.solid), \
        "obstacles must equal the solid (wall) set after stamping a unit"


def test_per_unit_permeability_hook():
    """A unit may override permeability via a `unit.permeability` attribute."""
    g = GameMap(load_level("unhcr_vessel"))
    ay, ax = _clear_air_anchor(g)
    u = Unit("U1", x=ax, y=ay, team=0)
    u.permeability = 0.2

    g.stamp_units([u])

    expected = np.float32(0.2)
    for (tx, ty) in u.occupied_tiles():
        assert g.dyn_permeability[ty, tx] == expected, \
            f"per-unit permeability hook ignored at ({tx},{ty})"


def test_dead_unit_leaves_permeability_open():
    """A dead unit is neither an obstacle nor a flow-restriction."""
    g = GameMap(load_level("unhcr_vessel"))
    ay, ax = _clear_air_anchor(g)
    u = Unit("U1", x=ax, y=ay, team=0)
    u.alive = False

    g.stamp_units([u])

    assert np.array_equal(g.dyn_permeability, g.permeability), \
        "dead unit must not alter the permeability field"
    assert np.array_equal(g.obstacles, g.solid), \
        "dead unit must not add obstacles"
