"""#150 -- a sealed hull tile is drawn as a wall, not as a hole.

``is_vacuum`` / ``is_ambient`` on a SOLID tile is the sealed-hull state of
engine/04 §2.3 (the flag says the tile touches the outside; ``seal_tiles``
keeps it so a re-opened patch rejoins space instead of becoming an interior
air pocket at the map edge). EXPOSED vacuum / sky -- what the physics drains
into and what the picture shows as a hole -- is the flag AND open. Before
#150 the wall-destruction render seam and the tile inspector read the flag
alone, so a hull door closed after a decompression was drawn as space and
reported as "vacuum".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from level_loader import LevelData  # noqa: E402
from simulation.ambient import derive_ambient  # noqa: E402
from simulation.gamemap import MAT_HULL, GameMap  # noqa: E402
from renderer.hover_readout import _material_name  # noqa: E402
from renderer.tile_art import TileArtSeam  # noqa: E402

BREACH = (0, 5)         # an edge hull tile: destroying it opens the room to space


def _space_room(h=12, w=12) -> GameMap:
    tm = np.ones((h, w), dtype=np.int32)       # hull
    tm[1:h - 1, 1:w - 1] = 4                    # air
    return GameMap(LevelData(name="s150", version="1", path=Path("."),
                             tilemap=tm, tile_size_m=1.0,
                             diffuse_path=Path(".")))


def _planetside_room(h=12, w=12) -> GameMap:
    """A sky ring (v1 code 0 -> is_ambient) around a hull-walled room."""
    tm = np.zeros((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 1                    # hull
    tm[2:h - 2, 2:w - 2] = 4                    # air
    return GameMap(LevelData(name="s150_sky", version="1", path=Path("."),
                             tilemap=tm, tile_size_m=1.0,
                             diffuse_path=Path("."), boundary="ambient",
                             ambient=derive_ambient()))


def _patch_for(seam, g, tile):
    (p,) = [p for p in seam.consume(g) if p.tile == tile]
    return p


def test_a_sealed_breach_is_drawn_as_a_wall_and_reopens_onto_space():
    """PROPERTY (#150): breach -> seal -> the render seam's patch for the
    tile is NOT a hole and the inspector names its material, although the
    sim keeps ``is_vacuum`` on the solid tile (engine/04 §2.3) -- which is
    what lets a second unseal rejoin space with no gas seeded. BREAKS IF a
    reader takes the flag alone for a hole again, or if ``seal_tiles``
    starts clearing the flag (the reopened edge patch would seed air)."""
    g = _space_room()
    seam = TileArtSeam()
    seam.consume(g)                              # snapshot the load-time art
    g.destroy_wall(*BREACH)
    assert bool(g.is_vacuum[BREACH]) and not bool(g.solid[BREACH])
    assert _patch_for(seam, g, BREACH).vacuum, "a breach must be a hole: vacuous"
    assert _material_name(g, BREACH[0], BREACH[1]) == "vacuum"

    g.seal_tiles([BREACH], MAT_HULL)
    assert bool(g.solid[BREACH]) and bool(g.is_vacuum[BREACH])   # sealed hull
    assert not _patch_for(seam, g, BREACH).vacuum, \
        "a sealed hull tile drawn as space (#150)"
    assert _material_name(g, BREACH[0], BREACH[1]) != "vacuum"

    g.unseal_tiles([BREACH])
    assert bool(g.is_vacuum[BREACH]) and not bool(g.solid[BREACH])
    assert all(int(g.gas[i][BREACH]) == 0 for i in range(g.gas.shape[0]))
    assert _patch_for(seam, g, BREACH).vacuum


def test_the_load_time_art_snapshot_treats_only_open_flagged_tiles_as_holes():
    """PROPERTY: the seam's donor snapshot marks a hole where the flag is on
    an OPEN tile; a solid tile flagged before the snapshot (a level loaded
    from a save after a hull patch) is a donor like any wall. BREAKS IF the
    snapshot reads the flag alone."""
    g = _space_room()
    g.destroy_wall(*BREACH)
    g.seal_tiles([BREACH], MAT_HULL)
    g.drain_render_dirty()
    seam = TileArtSeam()
    seam.consume(g)
    assert seam.base_material[BREACH] == MAT_HULL
    assert (seam.base_material == -1).sum() == 0     # no open vacuum tile here


def test_a_sealed_sky_breach_is_drawn_as_a_wall():
    """PROPERTY (#150, the planetside twin): on an ambient map a breach joins
    the sky (``is_ambient``); sealed, the tile keeps the flag and is drawn as
    a wall. BREAKS IF the seam reads ``is_ambient`` alone."""
    g = _planetside_room()
    seam = TileArtSeam()
    seam.consume(g)
    hull = (1, 5)                               # the hull ring's top edge
    assert bool(g.solid[hull]) and bool(g.is_ambient[0, 5])
    g.destroy_wall(*hull)
    assert bool(g.is_ambient[hull]) and not bool(g.solid[hull])
    assert _patch_for(seam, g, hull).vacuum

    g.seal_tiles([hull], MAT_HULL)
    assert bool(g.solid[hull]) and bool(g.is_ambient[hull])
    assert not _patch_for(seam, g, hull).vacuum
