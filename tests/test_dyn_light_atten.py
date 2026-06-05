"""Dynamic per-channel light-attenuation field (Slice 3).

`stamp_units` rebuilds, in one pass with `obstacles`, a live per-channel field
`gmap.dyn_light_atten` (h, w, 3) = the static material attenuation combined via
per-channel MAX with each living unit's opacity stamped over its footprint.
This restores unit shadows (S2 dropped them) and sets up per-colour dynamic
occlusion. The ray march reads THIS field (read-only) instead of the
static-only `gmap.light_atten`.

Verifies:
  - on a clear-air footprint, dyn_light_atten == [1,1,1] (default unit opacity)
    and equals the static field everywhere else;
  - the buffer is filled IN-PLACE (same object id across ticks — no realloc
    that would stale a C++ view);
  - per-channel MAX: an occluder only ADDS opacity (static glass under a unit
    stays at least as opaque as both);
  - the per-unit colour hook (`unit.light_atten`) is honoured;
  - a dead unit leaves no shadow;
  - downstream: a C++ ray crossing a stamped-unit tile is blocked, while the
    same scene without the unit is not (proves the field actually occludes).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_dyn_light_atten.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

from level_loader import load as load_level
from simulation.gamemap import GameMap
from simulation.unit import Unit


def _clear_air_anchor(g: GameMap, footprint: int = 3):
    """Find an interior anchor (tile_y, tile_x) whose footprint-sized block is
    fully passable air (so the static atten there is [0,0,0])."""
    h, w = g.material.shape
    for y in range(2, h - footprint - 2):
        for x in range(2, w - footprint - 2):
            if g.is_passable_block(y, x, footprint):
                # double-check static atten is clear over the whole block
                if not g.light_atten[y:y + footprint, x:x + footprint].any():
                    return y, x
    raise AssertionError("no clear-air footprint found in level")


def test_dyn_field_default_unit_shadow_and_matches_static_elsewhere():
    g = GameMap(load_level("unhcr_vessel"))
    ay, ax = _clear_air_anchor(g)
    # Footprint offsets are top-left anchored (0..2), so anchor at (ax, ay)
    # makes the footprint exactly the verified clear-air block.
    u = Unit("U1", x=ax, y=ay, team=0)
    static_before = g.light_atten.copy()

    g.stamp_units([u])

    # Footprint tiles are fully opaque [1,1,1] (default unit opacity -> shadow).
    occ = u.occupied_tiles()
    for (tx, ty) in occ:
        assert np.array_equal(g.dyn_light_atten[ty, tx], [1.0, 1.0, 1.0]), \
            f"unit tile ({tx},{ty}) not opaque: {g.dyn_light_atten[ty, tx]}"

    # Everywhere NOT in the footprint, dyn == static.
    mask = np.ones((g._h, g._w), dtype=bool)
    for (tx, ty) in occ:
        mask[ty, tx] = False
    assert np.array_equal(g.dyn_light_atten[mask], g.light_atten[mask]), \
        "dyn diverged from static outside the unit footprint"
    # Static field itself was not mutated.
    assert np.array_equal(g.light_atten, static_before), "static light_atten mutated"


def test_dyn_field_filled_in_place_no_realloc():
    g = GameMap(load_level("unhcr_vessel"))
    ay, ax = _clear_air_anchor(g)
    buf_id = id(g.dyn_light_atten)
    u = Unit("U1", x=ax, y=ay, team=0)
    g.stamp_units([u])
    assert id(g.dyn_light_atten) == buf_id, "dyn_light_atten was reassigned (would stale a C++ view)"
    # And again on a second tick.
    g.stamp_units([u])
    assert id(g.dyn_light_atten) == buf_id, "dyn_light_atten reassigned on second tick"


def test_per_channel_max_occluder_only_adds_opacity():
    g = GameMap(load_level("unhcr_vessel"))
    ay, ax = _clear_air_anchor(g)
    # Paint glass-ish static atten on the anchor tile (inside the footprint).
    g.light_atten[ay, ax] = [0.5, 0.5, 0.5]
    u = Unit("U1", x=ax, y=ay, team=0)
    # Give the unit a partial, per-colour opacity (passes some green).
    u.light_atten = (0.2, 0.0, 0.8)
    g.stamp_units([u])

    # At (ay, ax): MAX([0.5,0.5,0.5], [0.2,0.0,0.8]) = [0.5,0.5,0.8].
    assert np.allclose(g.dyn_light_atten[ay, ax], [0.5, 0.5, 0.8]), \
        f"per-channel max wrong: {g.dyn_light_atten[ay, ax]}"
    # At a clear footprint tile: MAX([0,0,0], unit) = unit opacity.
    assert np.allclose(g.dyn_light_atten[ay + 2, ax + 2], [0.2, 0.0, 0.8])


def test_dead_unit_casts_no_shadow():
    g = GameMap(load_level("unhcr_vessel"))
    ay, ax = _clear_air_anchor(g)
    u = Unit("U1", x=ax, y=ay, team=0)
    u.alive = False
    g.stamp_units([u])
    assert np.array_equal(g.dyn_light_atten, g.light_atten), \
        "dead unit must not alter the dynamic field"


def test_dyn_field_blocks_a_cpp_ray_downstream():
    """A ray crossing a stamped-unit tile is blocked; same scene w/o the unit
    is not. Synthetic 1-row grid + the real C++ directional cast, but using the
    real stamp_units output as the attenuation field (built on a real gmap, the
    relevant row sliced into the headless cast)."""
    import breach_physics as bp

    g = GameMap(load_level("unhcr_vessel"))
    ay, ax = _clear_air_anchor(g, footprint=3)
    w = g._w

    # Stamp a real unit; read back the dynamic field on the unit's centre row.
    u = Unit("U1", x=ax, y=ay, team=0)
    g.stamp_units([u])
    centre_row = ay + 1
    occ_xs = sorted({tx for (tx, ty) in u.occupied_tiles() if ty == centre_row})
    assert occ_xs, "unit has no tiles on its centre row"

    # Baseline row: clear air everywhere (so the source at x=0 has an
    # unobstructed path — isolate the unit's contribution from ship walls).
    row_no_unit = np.zeros((1, w, 3), np.float32)
    # Unit row: copy the REAL stamped-unit opacity (proves stamp_units output
    # occludes) onto an otherwise-clear path.
    row_unit = np.zeros((1, w, 3), np.float32)
    for tx in occ_xs:
        row_unit[0, tx] = g.dyn_light_atten[centre_row, tx]
    # Sanity: stamp_units made those tiles opaque.
    assert np.allclose(row_unit[0, occ_xs[0]], [1.0, 1.0, 1.0])

    def cast(atten_row):
        rc = bp.Raycaster()
        rgb = np.zeros((1, w, 3), np.float32)
        dx = np.zeros((1, w), np.float32)
        dy = np.zeros((1, w), np.float32)
        smoke = np.zeros((1, w), np.float32)
        s = bp.LightSource()
        s.x, s.y = 0.0, 0.0
        s.max_range = float(w * 2)
        s.intensity = 1.0
        s.angle_center = 0.0
        s.angle_spread = 0.05
        s.ray_count = 1
        s.color = (1.0, 1.0, 1.0)
        rc.cast_source_directional(s, rgb, dx, dy, smoke, atten_row)
        return rgb

    rgb_no = cast(row_no_unit)
    rgb_yes = cast(row_unit)

    # Pick a tile just past the unit footprint on this row.
    past = occ_xs[-1] + 1
    assert past < w
    # Without the unit, light reaches `past`; with the opaque unit it is killed.
    assert rgb_no[0, past].sum() > 0.0, "baseline ray should reach past the unit tile"
    assert rgb_yes[0, past].sum() == 0.0, \
        f"ray leaked past the opaque unit shadow: {rgb_yes[0, past]}"
