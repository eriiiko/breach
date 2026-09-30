"""#9 -- the wall-destruction render seam: ONE path from a topology change to
the picture, whatever destroyed (or sealed, or opened) the tile.

Sim end: ``GameMap.on_tile_changed`` marks ``gmap.render_dirty``; every
runtime topology writer funnels through it. Render end:
``renderer.tile_art.TileArtSeam.consume`` drains it (``GameRenderer.
upload_state``) and ``compose_world`` re-blits the tiles
(``GameRenderer._apply_tile_patches``, GPU -- exercised end to end by
``tools/e2e_drive.py``, not here). These tests hold the pure parts.
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

import breach_physics as bp  # noqa: E402
from config import CFG, ticks_from_seconds  # noqa: E402
from level_loader import EntityInstance, LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402
from simulation.materials import (MAT_AIR, MAT_DOOR_CLOSED,  # noqa: E402
                                  MAT_HULL)
from renderer.tile_art import (TileArtSeam, TilePatch, find_donor,  # noqa: E402
                               patch_block, tile_pixel_rect)

TPS = int(CFG.clock.ticks_per_second)


def _room(hh=14, entities=()):
    """A hull box (1-tile hull ring, space outside it is the grid edge) with an
    interior hull pillar at (6, 6) to break and an air interior."""
    tm = np.full((hh, hh), MAT_HULL, dtype=np.int32)
    tm[1:hh - 1, 1:hh - 1] = MAT_AIR
    tm[6, 6] = MAT_HULL
    return LevelData(name="p4_seam", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."),
                     entities=list(entities), wires=[])


def _sim(level=None):
    return Simulation(level or _room(), seed=5, breach_physics=bp,
                      enable_recorder=False)


# ---------------------------------------------------------------------------
# Sim end: every topology writer reaches the seam
# ---------------------------------------------------------------------------
def _destroy(g):
    g.destroy_wall(6, 6)
    return [(6, 6)]


def _seal(g):
    g.seal_tiles([(8, 8)], MAT_DOOR_CLOSED)
    return [(8, 8)]


def _seal_then_unseal(g):
    g.seal_tiles([(8, 8)], MAT_DOOR_CLOSED)
    g.drain_render_dirty()
    g.unseal_tiles([(8, 8)])
    return [(8, 8)]


@pytest.mark.parametrize("writer", [_destroy, _seal, _seal_then_unseal],
                         ids=lambda f: f.__name__)
def test_a_topology_writer_marks_every_tile_it_changes(writer):
    """Property: after any topology writer runs, every tile whose `material`
    it changed is in the drained dirty set. Breaks if a writer stops going
    through `on_tile_changed`, or if `on_tile_changed` stops marking."""
    g = _sim().gmap
    g.drain_render_dirty()
    before = g.material.copy()
    touched = writer(g)
    changed = {tuple(map(int, t)) for t in np.argwhere(g.material != before)}
    changed |= set(touched)            # seal-then-unseal ends where it began
    assert changed, "the writer changed nothing: vacuous"
    assert changed <= set(g.drain_render_dirty())


def test_drain_returns_the_marks_once_and_clears_them():
    """Property: a drain hands over the marked tiles and empties the set, so a
    tile is re-blitted once per change, not every frame. Breaks if the drain
    stops clearing (per-frame re-blits) or clears before returning."""
    g = _sim().gmap
    g.drain_render_dirty()
    g.destroy_wall(6, 6)
    assert (6, 6) in g.drain_render_dirty()
    assert g.drain_render_dirty() == []
    assert not g.render_dirty


def test_every_material_change_of_a_real_run_is_drained():
    """Property: over a real run in which a charge blows the hull open (the
    explosion path, then the solver ticks after it), EVERY tile whose material
    differs from the level's is in the union of the per-tick drains -- whatever
    code path changed it. Breaks if any runtime path writes `material` without
    `on_tile_changed` (a new destruction cause that bypasses destroy_wall)."""
    fields = {f.name: f.default for f in REGISTRY["timed_charge"].FIELDS}
    fields.update(x=6, y=2, payload="studio_hull", first_at_s=0.25,
                  period_s=0.0, enabled=True)
    charge = EntityInstance(id="c1", class_name="timed_charge", ordinal=0,
                            tags=(), fields=fields)
    sim = _sim(_room(entities=[charge]))
    g = sim.gmap
    mat0 = g.material.copy()
    g.drain_render_dirty()
    drained = set()
    for _ in range(ticks_from_seconds(0.25, TPS) + 6):
        sim.set_paused(False)
        sim.step()
        drained |= set(g.drain_render_dirty())
    changed = {tuple(map(int, t)) for t in np.argwhere(g.material != mat0)}
    assert changed, "the charge broke nothing: vacuous"
    assert changed <= drained


def test_the_dirty_set_is_no_synced_field():
    """Property: the dirty set is render-only bookkeeping -- it is not an
    array field and no digest, A/B harness or recorder field list names it, so
    marking tiles can never move a golden. Breaks if `render_dirty` is made a
    grid field or added to a synced field list."""
    from field_ab_harness import SIM_FIELDS
    from field_digest import DIGEST_FIELDS
    from simulation.recorder import PhysicsRecorder as Recorder
    g = _sim().gmap
    assert not isinstance(g.render_dirty, np.ndarray)
    assert "render_dirty" not in {n for n, _ in DIGEST_FIELDS}
    assert "render_dirty" not in SIM_FIELDS
    assert "render_dirty" not in Recorder.DEFAULT_FIELDS


# ---------------------------------------------------------------------------
# Render end: the consumer
# ---------------------------------------------------------------------------
def test_consume_clears_the_set_and_a_broken_wall_takes_floor_art():
    """Property: the renderer's drain empties the sim's set, and a destroyed
    wall is patched from a tile that was loaded as floor (AIR, not a vacuum
    hole) -- never from its own wall art. Breaks if consume stops draining, or
    the donor rule stops keying on the tile's material now."""
    g = _sim().gmap
    seam = TileArtSeam()
    seam.consume(g)                      # the first sight: the load snapshot
    g.destroy_wall(6, 6)
    patches = seam.consume(g)
    assert not g.render_dirty
    (p,) = [q for q in patches if q.tile == (6, 6)]
    assert p.donor is not None and p.donor != (6, 6)
    assert int(g.material[p.donor]) == MAT_AIR
    assert not g.is_vacuum[p.donor]
    assert p.vacuum == bool(g.is_vacuum[6, 6])


def test_a_hull_hole_to_space_is_punched_through():
    """Property: a destroyed edge-hull tile that joins vacuum comes out of the
    consumer flagged vacuum (the shader then shows the backdrop). Breaks if
    the consumer reads the tile's state before destroy_wall's vacuum join."""
    g = _sim().gmap
    seam = TileArtSeam()
    seam.consume(g)
    g.destroy_wall(0, 6)                 # the edge of the grid: hull -> space
    assert g.is_vacuum[0, 6], "fixture: this breach must join vacuum"
    (p,) = [q for q in seam.consume(g) if q.tile == (0, 6)]
    assert p.vacuum


def test_a_tile_back_to_its_load_material_shows_its_own_art():
    """Property: a tile whose material returns to its load-time material (a
    door span that opens and closes again) is patched from its OWN pristine
    art. Breaks if the donor search skips the tile itself."""
    g = _sim().gmap
    seam = TileArtSeam()
    seam.consume(g)
    g.seal_tiles([(8, 8)], MAT_DOOR_CLOSED)
    g.unseal_tiles([(8, 8)])
    (p,) = [q for q in seam.consume(g) if q.tile == (8, 8)]
    assert p.donor == (8, 8)


def test_find_donor_takes_the_nearest_ring_and_gives_up_past_the_radius():
    """Property: the donor is at the smallest Chebyshev distance holding the
    material, and None beyond the search radius. Breaks if the search returns
    a farther tile first or never terminates."""
    base = np.full((9, 9), 1, dtype=np.int16)
    base[4, 7] = 0                        # distance 3
    base[1, 4] = 0                        # distance 3, earlier row-major
    base[6, 6] = 0                        # distance 2 -- the nearest
    assert find_donor(base, (4, 4), 0, radius=8) == (6, 6)
    assert find_donor(base, (4, 4), 0, radius=1) is None
    assert find_donor(base, (4, 4), 1, radius=8) == (4, 4)


@pytest.mark.parametrize("art_align,art_wh,tex_wh", [
    (None, (40, 30), (40, 30)),                      # legacy full stretch
    (None, (40, 30), (80, 60)),                      # a 2x normal map
    ((4.0, 2.0, 8.0, 8.0), (48, 36), (48, 36)),     # explicit [art.align]
])
def test_patch_block_copies_the_donor_pristine_rect_onto_the_tile(
        art_align, art_wh, tex_wh):
    """Property: the re-blit writes, at the tile's rect under the draw's own
    art -> grid mapping, exactly the donor's pristine pixels. Breaks if the
    rect math diverges from `art_src_and_uv_rect` (the draw) or the copy
    reads the wrong tile."""
    grid_h, grid_w = 3, 4
    tw, th = tex_wh
    rng = np.random.default_rng(0)
    pristine = rng.integers(0, 256, size=(th, tw, 4), dtype=np.uint8)
    p = TilePatch(tile=(1, 2), vacuum=False, donor=(2, 0))
    (x, y, w, h), px = patch_block(pristine, p, grid_h, grid_w, art_align,
                                   art_wh[0], art_wh[1])
    assert (x, y, w, h) == (lambda r: (r[0], r[1], r[2] - r[0], r[3] - r[1]))(
        tile_pixel_rect((1, 2), grid_h, grid_w, art_align, art_wh[0],
                        art_wh[1], tw, th))
    dx0, dy0, dx1, dy1 = tile_pixel_rect((2, 0), grid_h, grid_w, art_align,
                                         art_wh[0], art_wh[1], tw, th)
    np.testing.assert_array_equal(px, pristine[dy0:dy1, dx0:dx1])
    assert px.flags["C_CONTIGUOUS"]


def test_no_donor_means_no_blit():
    """Property: a patch without a donor leaves the art alone. Breaks if a
    missing donor is blitted from some default tile."""
    pristine = np.zeros((8, 8, 3), dtype=np.uint8)
    assert patch_block(pristine, TilePatch((0, 0), False, None),
                       2, 2, None, 8, 8) is None
