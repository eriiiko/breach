"""The wall-destruction render seam (#9) -- pure numpy, no GPU calls.

ONE path from a topology change to the picture, whatever caused it (the burst
valve, an explosion, bullet or beam chew, fire burn-through, a door's death,
a door opening or closing):

    GameMap.on_tile_changed          marks the tile in ``gmap.render_dirty``
      (every runtime topology writer -- destroy_wall / seal_tiles /
       unseal_tiles -- funnels through it)
    TileArtSeam.consume(gmap)        drains the set once per frame
                                     (GameRenderer.upload_state) and turns each
                                     tile into a :class:`TilePatch`
    GameRenderer.compose_world       applies the patches before the lit-ship
                                     draw: the vacuum mask texel, and the
                                     tile's rect of every static art layer
                                     re-blitted from PRISTINE art

What a changed tile shows is decided by one rule, with no per-cause code:
**a tile shows the load-time art of the nearest tile whose load-time material
equals its material now** -- its own art when its material is back to what it
was (a door that closes again), else the nearest tile that was loaded as that
material (a destroyed wall shows the level's own floor art: no rubble art
exists in any shipped tileset, and the level format's ``[art.destroyed]``
layer is parsed but no level carries one). A tile that is now vacuum or open
sky is punched out through the vacuum mask, so the backdrop shows through the
hole. The "load-time material" is the renderer's snapshot of
``gmap.material`` at its first drain, which is what the art depicts
(door and prop stamps included); load-time vacuum / sky tiles are never
donors (their art is the hole).

Nothing here scans the grid per frame: work is proportional to the drained
tiles (plus a bounded donor search per tile). The sim never imports this; it
only fills a plain set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .lighting import art_src_and_uv_rect

# How far (Chebyshev rings, in tiles) the donor search looks for a tile loaded
# as the changed tile's new material. A tile with no such tile within the
# radius keeps its own art (and says so through ``TilePatch.donor``).
DONOR_SEARCH_RADIUS = 8


@dataclass(frozen=True)
class TilePatch:
    """What one changed tile must look like now.

    ``vacuum``: the tile is vacuum / open sky -> discarded by the ship shader.
    ``donor``: the tile whose PRISTINE art it now shows (itself when its
    material is its load-time one), or ``None`` when no donor exists within
    :data:`DONOR_SEARCH_RADIUS` (the art is left as it is)."""
    tile: Tuple[int, int]
    vacuum: bool
    donor: Optional[Tuple[int, int]]


def find_donor(base_material: np.ndarray, tile, material_now: int,
               radius: int = DONOR_SEARCH_RADIUS):
    """The nearest tile (Chebyshev ring by ring, row-major within a ring --
    deterministic) whose load-time material is ``material_now``; the tile
    itself first. ``None`` when none lies within ``radius``."""
    h, w = base_material.shape
    fy, fx = int(tile[0]), int(tile[1])
    if int(base_material[fy, fx]) == int(material_now):
        return (fy, fx)
    for r in range(1, int(radius) + 1):
        y0, y1 = max(0, fy - r), min(h - 1, fy + r)
        x0, x1 = max(0, fx - r), min(w - 1, fx + r)
        win = base_material[y0:y1 + 1, x0:x1 + 1]
        hits = np.argwhere(win == material_now)
        for dy, dx in hits:                       # row-major already
            y, x = y0 + int(dy), x0 + int(dx)
            if max(abs(y - fy), abs(x - fx)) == r:
                return (y, x)
    return None


def tile_pixel_rect(tile, grid_h: int, grid_w: int, art_align,
                    art_w: int, art_h: int, tex_w: int, tex_h: int):
    """The (x0, y0, x1, y1) pixel rect, in a texture of ``tex_w`` x ``tex_h``,
    that the draw maps onto ``tile`` -- the SAME art -> grid mapping as
    ``LightingPass.draw_lit_world`` (``art_src_and_uv_rect``: the full stretch
    without ``[art.align]``, the explicit transform with it), taken in art-UV
    space so a layer at another resolution (a normal map) maps the same way.
    Clipped to the texture; may be empty (x1 <= x0) for a tile off the art."""
    _src, uv = art_src_and_uv_rect(art_align, grid_w, grid_h,
                                   float(art_w), float(art_h))
    ux, uy, uw, uh = uv
    fy, fx = int(tile[0]), int(tile[1])
    x0 = int(round((ux + uw * fx / grid_w) * tex_w))
    x1 = int(round((ux + uw * (fx + 1) / grid_w) * tex_w))
    y0 = int(round((uy + uh * fy / grid_h) * tex_h))
    y1 = int(round((uy + uh * (fy + 1) / grid_h) * tex_h))
    return (max(0, x0), max(0, y0), min(tex_w, x1), min(tex_h, y1))


def resample_nearest(block: np.ndarray, h: int, w: int) -> np.ndarray:
    """``block`` resized to (h, w) by nearest index (tile rects from rounded
    boundaries can differ by a pixel); contiguous, same dtype."""
    bh, bw = block.shape[:2]
    if (bh, bw) == (h, w):
        return np.ascontiguousarray(block)
    ys = (np.arange(h) * bh) // max(h, 1)
    xs = (np.arange(w) * bw) // max(w, 1)
    return np.ascontiguousarray(block[ys][:, xs])


def patch_block(pristine: np.ndarray, patch: TilePatch, grid_h: int,
                grid_w: int, art_align, art_w: int, art_h: int):
    """The (rect, pixels) re-blit for ``patch`` into a layer whose PRISTINE
    pixels are ``pristine`` (H, W, C): the donor's pristine rect resampled to
    the tile's rect. ``None`` when there is nothing to blit (no donor, or the
    tile maps off the art)."""
    if patch.donor is None:
        return None
    th, tw = pristine.shape[:2]
    x0, y0, x1, y1 = tile_pixel_rect(patch.tile, grid_h, grid_w, art_align,
                                     art_w, art_h, tw, th)
    dx0, dy0, dx1, dy1 = tile_pixel_rect(patch.donor, grid_h, grid_w,
                                         art_align, art_w, art_h, tw, th)
    if x1 <= x0 or y1 <= y0 or dx1 <= dx0 or dy1 <= dy0:
        return None
    return ((x0, y0, x1 - x0, y1 - y0),
            resample_nearest(pristine[dy0:dy1, dx0:dx1], y1 - y0, x1 - x0))


class TileArtSeam:
    """The renderer's consumer end of ``gmap.render_dirty`` (see module doc)."""

    def __init__(self, radius: int = DONOR_SEARCH_RADIUS):
        self.radius = int(radius)
        self.base_material: Optional[np.ndarray] = None

    def consume(self, gmap) -> list:
        """Drain ``gmap``'s dirty tiles (clearing them) and return one
        :class:`TilePatch` per tile, read from the tile's state NOW. The first
        call snapshots ``gmap.material`` as the load-time material."""
        is_ambient = getattr(gmap, "is_ambient", None)
        if self.base_material is None:
            base = np.array(gmap.material, dtype=np.int16, copy=True)
            # A load-time vacuum / sky tile is AIR in `material` but its art
            # is the hole (transparent), never floor: it is no donor.
            hole = np.asarray(gmap.is_vacuum, dtype=bool).copy()
            if is_ambient is not None:
                hole |= np.asarray(is_ambient, dtype=bool)
            base[hole] = -1
            self.base_material = base
        tiles = gmap.drain_render_dirty()
        if not tiles:
            return []
        out = []
        for fy, fx in tiles:
            vac = bool(gmap.is_vacuum[fy, fx]) or (
                is_ambient is not None and bool(is_ambient[fy, fx]))
            donor = find_donor(self.base_material, (fy, fx),
                               int(gmap.material[fy, fx]), self.radius)
            out.append(TilePatch(tile=(fy, fx), vacuum=vac, donor=donor))
        return out
