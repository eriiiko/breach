"""Pressure overlay reads ``atmosphere`` alone (issue #62).

Regression net for the double-counted DC offset: since EOS P3, ``wave_p`` is
the previous tick's pressure buffer carrying its own ~1 atm DC offset, so
``renderer/pressure_overlay.py``'s old ``atmosphere + wave_p`` packing made a
quiet 1 atm room read as ~2 atm on the overlay. The property this protects:
the packed colour for a quiet room depends on ``atmosphere`` only — changing
``wave_p`` must never move it. A regression that reintroduces
``+ gmap.wave_p`` (or any other read of ``wave_p``) in the packing breaks
this test.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_pressure_overlay_atmosphere_only.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402,F401  (loads the pyd GameMap needs)
from level_loader import LevelData  # noqa: E402
from simulation import atmosphere_fixed  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from renderer.pressure_overlay import pack_pressure_rgba  # noqa: E402


def _sealed_room_level(h=12, w=12) -> LevelData:
    """A hull-walled sealed box, interior air, no vacuum/breach anywhere."""
    tm = np.ones((h, w), dtype=np.int32)      # all hull
    tm[1:h - 1, 1:w - 1] = 4                   # carve interior air
    return LevelData(name="p62_sealed_room", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _make_gmap(h=12, w=12) -> GameMap:
    return GameMap(_sealed_room_level(h, w))


# A colour-stops table with one interior stop so the "1 atm" and "2 atm"
# buckets paint visibly different colours (real config.toml stops idiom).
_STOPS = np.array([
    [0.0,   0,   0,   0,   0],
    [1.5,  10,  20,  30,  40],   # ~1 atm should land inside this bucket
    [2.5, 200, 210, 220, 230],   # ~2 atm (the old double-counted bug) here
], dtype=np.float32)


def test_quiet_room_packs_to_one_atm_not_two():
    """A quiet 1 atm room packs to the 1-atm colour bucket, regardless of
    what wave_p holds — the property #62 exists to protect: the overlay
    reads atmosphere ALONE. Setting wave_p to a 1-atm-worth DC offset (the
    real-world shape of the bug: the previous tick's pressure buffer) must
    not move the packed colour into the 2-atm bucket."""
    gmap = _make_gmap()
    interior = ~(gmap.solid | gmap.is_vacuum)
    assert np.any(interior)

    # Sanity: the freshly-built room is at exactly 1 atm.
    assert np.allclose(
        atmosphere_fixed.dequantize_f32(gmap.atmosphere)[interior], 1.0)

    rgba_before = pack_pressure_rgba(
        gmap.atmosphere, gmap.solid, gmap.is_vacuum, _STOPS, pressure_scale=1.0)

    # Simulate the bug's real shape: wave_p carrying a ~1 atm DC offset (as
    # it does mid-run, per EOS P3), on top of the quiet 1-atm atmosphere.
    gmap.wave_p[:] = atmosphere_fixed.quantize(1.0)

    rgba_after = pack_pressure_rgba(
        gmap.atmosphere, gmap.solid, gmap.is_vacuum, _STOPS, pressure_scale=1.0)

    # The packing must be identical: it never reads wave_p at all.
    np.testing.assert_array_equal(rgba_before, rgba_after)

    # And it must land in the 1-atm bucket (interpolated between the [0.0]
    # and [1.5] stops), not the 2-atm bucket the old sum would have hit.
    iy, ix = np.argwhere(interior)[0]
    assert rgba_after[iy, ix, 0] < _STOPS[1, 1]  # below the 1.5-atm stop's R
    assert rgba_after[iy, ix, 0] != 0 or _STOPS[0, 1] == 0  # sane bucket edge


def test_solid_and_vacuum_tiles_stay_transparent():
    """Walls and vacuum are masked out regardless of atmosphere/wave_p —
    the property that must keep holding after the #62 fix: a solid tile's
    packed alpha is always 0. Removing the ``solid | is_vacuum`` mask
    breaks this."""
    gmap = _make_gmap()
    rgba = pack_pressure_rgba(
        gmap.atmosphere, gmap.solid, gmap.is_vacuum, _STOPS, pressure_scale=1.0)
    assert np.all(rgba[gmap.solid][..., 3] == 0)
