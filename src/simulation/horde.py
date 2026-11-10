"""The horde field (gameplay pack v1 G9p; the residency design's P3d).

``docs/residency_batching_design_2026-11-07.md`` §4.9 and
``docs/bots_zombies_speeds_design_2026-11-10.md`` §3 (Erik's ruling of
2026-11-10: the distance field moves forward, and the horde is rewritten
once, onto it).

THE RULE: the horde paths ONLY by an integer distance field. A field is a
breadth-first flood over the ANCHOR grid of a footprint ``fp`` (cell ``(y,
x)`` is the top-left tile of an ``fp x fp`` body; it is passable when that
whole block is in bounds and every tile of it is ``tile_ok``), from a set of
seed cells, 4-connected; a cell the flood never reaches holds
:data:`UNREACHED`. A zombie DESCENDS a field: of its four neighbours it steps to the lowest one
strictly below its own cell's value that it can enter now, ties in the cell's
fixed order (:data:`DESCENT_ORDER`) -- or it stays. Its every step is to a
strictly lower value. No per-zombie search exists anywhere in the law.

- THE PREY FIELD: seeded at every living non-zombie (each prey's centre tile
  in the zombie's anchor space, ``centre - fp // 2``: the anchor that would
  centre the zombie on the prey, which is A*'s old goal when the footprints
  are equal), so a zombie walks to the PATH-nearest prey.
- The flood itself is C++ (``PhysicsEngine.horde_field``, ``cpp/src/
  horde_field.cpp``) whenever the map has an engine bound -- the played game
  and every match; :func:`flood_reference` is the Python reference of the
  same law for a bare ``GameMap`` (tests, tools), the ``stamp_units``
  precedent. The device kernel is residency P7d.
- REFRESH (:class:`HordeState`): every field is dropped at the zombie slot of
  a tick ``t`` with ``t % field_refresh_ticks == 0`` and whenever the map's
  topology serial (``GameMap._tile_serial``: a wall destroyed, a door opened
  or closed) moved since the last slot; a field is flooded the first time a
  zombie needs it after a drop, from the state of THAT tick. So the field a
  zombie reads at tick ``t`` was flooded at the first tick since the last
  drop on which one was needed -- a pure function of the synced history.
  The cache is never synced and never digested (a derived quantity, the
  vision cache's precedent). ACCEPTED GAP: a save / load between two
  refreshes would re-flood from the loaded tick's state.

Integer throughout; no random number; nothing here reads a float field.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral

import numpy as np

#: The value of a cell the flood never reached (no path: the zombie waits).
#: Equal to ``horde::UNREACHED`` in ``cpp/src/horde_field.h``.
UNREACHED = 0x7FFFFFFF

#: The four neighbours, ``(dy, dx)``: north, west, east, south (the
#: row-major order of the four cells) -- the flood's order (irrelevant to
#: its distances) and the wander's direction index.
NEIGHBOURS = ((-1, 0), (0, -1), (0, 1), (1, 0))

#: The descent's TIE order, by the parity of the cell's ``x + y``: vertical
#: first (north, south, west, east) on an even cell, horizontal first (west,
#: east, north, south) on an odd one. On a 4-connected field every cell
#: toward a diagonal goal has two equal lower neighbours; one fixed order
#: would walk an L (all north, then all east), this one walks the staircase
#: (north, east, north, ...), which is the straight line the 4-grid allows.
#: A fixed rule per cell, so the device port is the same two tables.
DESCENT_ORDER = (((-1, 0), (1, 0), (0, -1), (0, 1)),
                 ((0, -1), (0, 1), (-1, 0), (1, 0)))


# ---------------------------------------------------------------------------
# The settings door
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HordeSettings:
    """``[zombie]``'s horde dials, quantized ONCE for one map
    (:func:`horde_settings`). Radii are integer tiles, compared as squares
    (the law's old ``math.sqrt`` gates, integer now)."""
    trigger_r2: int
    propagation_r2: int
    field_refresh_ticks: int


def _int_key(section, key, lo=None):
    if section is None or not hasattr(section, key):
        raise ValueError(f"[zombie] is missing {key!r}")
    v = getattr(section, key)
    if isinstance(v, bool) or not isinstance(v, Integral):
        raise ValueError(f"[zombie] {key} must be an integer, got {v!r}")
    if lo is not None and v < lo:
        raise ValueError(f"[zombie] {key} must be >= {lo}, got {v!r}")
    return int(v)


def horde_settings(cfg, tile_size_m: float = 1.0) -> HordeSettings:
    """``[zombie]`` of a loaded config through the refusing door: every key
    present and of its type (ValueError naming it otherwise), quantized for a
    map of ``tile_size_m`` metres per tile."""
    z = getattr(cfg, "zombie", None)
    if z is None:
        raise ValueError("config has no [zombie] table")
    if not float(tile_size_m) > 0:
        raise ValueError(f"tile_size_m must be > 0, got {tile_size_m!r}")
    trig = _int_key(z, "trigger_radius", 0)
    prop = _int_key(z, "propagation_radius", 0)
    return HordeSettings(
        trigger_r2=trig * trig,
        propagation_r2=prop * prop,
        field_refresh_ticks=_int_key(z, "field_refresh_ticks", 1),
    )


# ---------------------------------------------------------------------------
# The flood
# ---------------------------------------------------------------------------
def block_passable(tile_ok: np.ndarray, fp: int) -> np.ndarray:
    """The anchor grid's passability: ``True`` at ``(y, x)`` when the ``fp x
    fp`` block from there is in bounds and all ``tile_ok`` (an integral
    image of the bad tiles; integer)."""
    h, w = tile_ok.shape
    bad = np.zeros((h + 1, w + 1), dtype=np.int64)
    bad[1:, 1:] = np.cumsum(np.cumsum(~tile_ok.astype(bool), axis=0), axis=1)
    out = np.zeros((h, w), dtype=bool)
    if fp <= h and fp <= w:
        s = (bad[fp:, fp:] - bad[:h + 1 - fp, fp:]
             - bad[fp:, :w + 1 - fp] + bad[:h + 1 - fp, :w + 1 - fp])
        out[:h + 1 - fp, :w + 1 - fp] = s == 0
    return out


def flood_reference(tile_ok: np.ndarray, fp: int, seeds) -> np.ndarray:
    """The flood's Python reference (a bare map without an engine): the
    law of ``horde::flood`` -- seeds labelled 0 whether passable or not,
    the frontier entering passable anchor cells only, 4-connected; every
    unreached cell :data:`UNREACHED`."""
    h, w = tile_ok.shape
    passable = block_passable(tile_ok, fp)
    dist = np.full((h, w), UNREACHED, dtype=np.int32)
    q = deque()
    for y, x in seeds:
        y, x = int(y), int(x)
        if 0 <= y < h and 0 <= x < w and dist[y, x] != 0:
            dist[y, x] = 0
            q.append((y, x))
    while q:
        y, x = q.popleft()
        nd = int(dist[y, x]) + 1
        for dy, dx in NEIGHBOURS:
            ny, nx = y + dy, x + dx
            if (0 <= ny < h and 0 <= nx < w and passable[ny, nx]
                    and dist[ny, nx] == UNREACHED):
                dist[ny, nx] = nd
                q.append((ny, nx))
    return dist


def flood(gmap, tile_ok: np.ndarray, fp: int, seeds) -> np.ndarray:
    """THE flood: ``PhysicsEngine.horde_field`` when ``gmap`` has an engine
    bound, else :func:`flood_reference` (the same law)."""
    seeds_arr = np.ascontiguousarray(
        np.asarray(seeds, dtype=np.int32).reshape(-1, 2))
    eng = getattr(gmap, "_physics_engine", None)
    if eng is None:
        return flood_reference(tile_ok, fp, seeds_arr)
    out = np.empty(tile_ok.shape, dtype=np.int32)
    eng.horde_field(np.ascontiguousarray(tile_ok, dtype=np.uint8), int(fp),
                    seeds_arr, out)
    return out


def walkable_tiles(gmap) -> np.ndarray:
    """The tiles a body may overlap: the mobility table's ``mobility > 0``
    projected through the material grid -- ``GameMap.is_passable_block``'s
    own predicate, tile by tile."""
    return gmap.materials.mobility[gmap.material] > 0


def anchor_of(cy: int, cx: int, fp: int) -> tuple:
    """The anchor cell that centres an ``fp`` body on tile ``(cy, cx)``."""
    return (int(cy) - fp // 2, int(cx) - fp // 2)


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------
class HordeState:
    """One map's field cache (never synced, never digested): the fields
    flooded since the last drop, keyed ``("prey", fp)``; the tile mask they
    were flooded over. A Simulation keeps one per reset; a direct caller of
    the law with none gets a fresh one per call."""

    def __init__(self):
        self.fields: dict = {}
        self.tile_ok = None
        self.serial = None
        self.floods = 0              # diagnostics: floods computed

    def begin_tick(self, gmap, tick: int, settings: HordeSettings) -> None:
        """The slot head: drop every field at a refresh tick or when the
        topology moved since the last slot."""
        serial = getattr(gmap, "_tile_serial", 0)
        if (int(tick) % settings.field_refresh_ticks == 0
                or serial != self.serial):
            self.fields.clear()
            self.tile_ok = None
            self.serial = serial

    def mask(self, gmap) -> np.ndarray:
        """The tile mask of this refresh (built once per drop)."""
        if self.tile_ok is None:
            self.tile_ok = walkable_tiles(gmap)
        return self.tile_ok

    def prey_field(self, gmap, fp: int, prey) -> np.ndarray:
        """The prey field of footprint ``fp``: flooded from every living
        non-zombie in ``prey`` (their centres, in anchor space)."""
        key = ("prey", int(fp))
        f = self.fields.get(key)
        if f is None:
            seeds = [anchor_of(p.center_tile_y(), p.center_tile_x(), fp)
                     for p in prey]
            f = flood(gmap, self.mask(gmap), fp, seeds)
            self.fields[key] = f
            self.floods += 1
        return f


# ---------------------------------------------------------------------------
# The descent
# ---------------------------------------------------------------------------
def candidates(field: np.ndarray, y: int, x: int) -> list:
    """The cells a zombie at anchor ``(y, x)`` may step to on ``field``:
    the in-bounds 4-neighbours strictly below its own cell's value (an
    out-of-bounds or unreached own cell counts as :data:`UNREACHED`), as
    ``(value, order, ny, nx)`` lowest first, ties in the cell's
    :data:`DESCENT_ORDER`."""
    h, w = field.shape
    cur = int(field[y, x]) if (0 <= y < h and 0 <= x < w) else UNREACHED
    out = []
    for k, (dy, dx) in enumerate(DESCENT_ORDER[(x + y) & 1]):
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            v = int(field[ny, nx])
            if v < cur:
                out.append((v, k, ny, nx))
    out.sort()
    return out


def descend(gmap, field: np.ndarray, y: int, x: int, fp: int):
    """The step a zombie at anchor ``(y, x)`` takes on ``field``: the first
    of :func:`candidates` it can enter NOW (``is_passable_block`` on the
    live map -- the field may be a refresh old), as ``(ny, nx)``; ``None``
    = it waits."""
    for _v, _k, ny, nx in candidates(field, y, x):
        if gmap.is_passable_block(ny, nx, fp):
            return (ny, nx)
    return None


__all__ = ["UNREACHED", "NEIGHBOURS", "DESCENT_ORDER", "HordeSettings", "horde_settings",
           "block_passable", "flood_reference", "flood", "walkable_tiles",
           "anchor_of", "HordeState", "candidates", "descend"]
