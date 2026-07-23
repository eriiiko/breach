"""Free-aim directional shooting (free_aim_shooting_design_2026-07-23).

F1 — the ``aim_angle`` seam through the fire resolvers:

  - the CONVENTION proof: ``u.facing`` is math-style (y-up); the march bearing
    is screen-style (y-down); the free-aim dispatch feeds the resolvers
    ``aim_angle = -facing`` (the single y-up->y-down flip). This test pins that
    ``-facing`` reproduces the tile-derived march bearing within the Q16.16
    quantization floor for a spread of known facing<->tile pairs;
  - the SEAM byte-identity: ``fire_burst`` / ``fire_beam`` called with an
    explicit ``aim_angle`` equal to the tile-derived bearing produce the
    IDENTICAL hit/event stream as the tile-derived (``aim_angle=None``) call —
    i.e. the seam is a pure substitution, nothing downstream changed.

F2/F3 directional end-to-end behaviour lives in tests/test_p3_direct_e2e.py.

Run:
    PYTHONPATH="...cpp/build;...cpp/build/Release" pytest tests/test_free_aim.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from level_loader import LevelData  # noqa: E402
from simulation import unit_fixed  # noqa: E402
from simulation.combat import fire_beam, fire_burst  # noqa: E402
from simulation.events import UnitHitEvent  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import get_tables  # noqa: E402

SEED = 20260723


def _room(h=24, w=24):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    return GameMap(LevelData(name="free_aim", version="2", path=Path("."),
                             tilemap=tm, tile_size_m=1.0,
                             diffuse_path=Path(".")))


def test_aim_angle_reproduces_tile_bearing():
    """CONVENTION PROOF (design §6): starting from a facing that points at a
    tile (what the AIM intent sets), the free-aim march bearing ``-facing``
    reproduces the tile-derived march bearing within quantization — the step
    vectors agree to the Q16.16 floor. Guards against a silent y-up/y-down
    sign inversion (escalation trigger (c))."""
    cx, cy = 5.0, 5.0
    pairs = [(15, 5), (5, 15), (15, 15), (1, 5), (5, 1), (1, 1),
             (12, 3), (3, 12), (14, 9), (2, 10)]
    for tx, ty in pairs:
        dx = tx - cx
        dy = ty - cy
        # What the AIM intent stores on the unit (unit.face_towards convention).
        facing = unit_fixed.atan2_rad(-dy, dx)
        # What the free-aim dispatch feeds the resolvers (march convention).
        aim_angle = -facing
        # What the tile-derived (aim_angle=None) path computes.
        tile_base = unit_fixed.atan2_rad(dy, dx)

        # Step vectors must agree — this is what the march actually consumes.
        sx_a = unit_fixed.cos_rad(aim_angle)
        sy_a = unit_fixed.sin_rad(aim_angle)
        sx_t = unit_fixed.cos_rad(tile_base)
        sy_t = unit_fixed.sin_rad(tile_base)
        err = math.hypot(sx_a - sx_t, sy_a - sy_t)
        # Quantization floor of the kit is ~9e-6 per component; allow a couple
        # ULP-scale slack for the double negation / atan2 y-oddness.
        assert err < 5e-5, (tx, ty, err)

        # And the world-direction the free-aim step encodes must match the
        # facing direction (cos f, -sin f) — the deleted band-aid's direction.
        fdx = unit_fixed.cos_rad(facing)
        fdy = -unit_fixed.sin_rad(facing)
        assert math.hypot(sx_a - fdx, sy_a - fdy) < 5e-5, (tx, ty)


def _fire(resolver, weapon, aim_angle):
    """Fire one zero-spread trigger east from a shooter at (2,8) into an enemy
    at (8,8); return the sorted (unit_id, amount) hit tuples."""
    gmap = _room()
    shooter = Unit("S", x=2, y=8, team=0)
    enemy = Unit("E", x=8, y=8, team=1)
    units = [shooter, enemy]
    t = get_tables()
    ammo = t.ammo_for_weapon(weapon)
    events: list = []
    rng = np.random.default_rng(SEED)
    cx, cy = shooter.center_tile_x(), shooter.center_tile_y()
    # Tile target due east at the weapon's nominal range (only used when
    # aim_angle is None); when aim_angle is provided the tile is ignored.
    fx2, fy2 = cx + float(weapon.range_tiles), cy
    resolver(gmap, units, shooter, cx, cy, fx2, fy2, tick=0, shots=[],
             real_time=0.0, rng=rng, events=events, weapon=weapon, ammo=ammo,
             spread_deg=0.0, aim_angle=aim_angle)
    hits = sorted((e.unit_id, e.source, round(float(e.damage), 6))
                  for e in events if isinstance(e, UnitHitEvent))
    return hits


def test_fire_burst_aim_angle_matches_tile_derived():
    """SEAM byte-identity: fire_burst(aim_angle=<tile bearing>) reproduces the
    tile-derived (aim_angle=None) hit stream exactly — the provided path is a
    pure substitution for the internal base_angle."""
    weapon = get_tables().weapons.by_name["k5_carbine"]
    # The march bearing the tile target (due east) produces.
    tile_base = unit_fixed.atan2_rad(0.0, float(weapon.range_tiles))
    tile_hits = _fire(fire_burst, weapon, aim_angle=None)
    aim_hits = _fire(fire_burst, weapon, aim_angle=tile_base)
    assert tile_hits, "control: the tile-derived shot must hit the enemy"
    assert aim_hits == tile_hits


def test_fire_beam_aim_angle_matches_tile_derived():
    """SEAM byte-identity for the hitscan resolver (Lance beam)."""
    weapon = get_tables().weapons.by_name["lance_3"]
    tile_base = unit_fixed.atan2_rad(0.0, float(weapon.range_tiles))
    tile_hits = _fire(fire_beam, weapon, aim_angle=None)
    aim_hits = _fire(fire_beam, weapon, aim_angle=tile_base)
    assert tile_hits, "control: the tile-derived beam must skewer the enemy"
    assert aim_hits == tile_hits
