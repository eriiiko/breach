"""GameMap — world state container (tiles + physics fields).

Lifted from ``game.py:GameMap`` (lines 292-542 in the legacy file) and merged
with the level-driven constructor that previously lived as a shim in
``main.py``. Canonical signature is ``GameMap(level_data)`` — the no-arg
form and ``_build_ship`` fallback from the legacy implementation are gone
(CSV loading via :mod:`level_loader` is the only path now).

Owns the cached arrays the physics systems read and write:

    material, wall_hp, is_wall, is_vacuum, flammable,
    atmosphere, wave_p, wave_v, wave_source, wind_x, wind_y,
    smoke, fire, obstacles, light_map

Plus ``self.level`` (the :class:`level_loader.LevelData` instance) and the
methods needed by combat / pathfinding / physics (``stamp_units``,
``is_passable``, ``is_passable_block``, ``has_los``, ``destroy_wall``).

No pygame, no pyray — pure numpy + config.
"""
from __future__ import annotations

import numpy as np

from config import CFG
from level_loader import materials_from_tilemap


# ---------------------------------------------------------------------------
# Material IDs (gameplay constants; renderer uses its own color table)
# ---------------------------------------------------------------------------
MAT_AIR = 0
MAT_HULL = 1
MAT_WOOD = 2
MAT_DOOR = 3


class GameMap:
    """2D grid map at fine-tile resolution, sized from the loaded level."""

    def __init__(self, level_data):
        """Build a GameMap from a :class:`level_loader.LevelData`.

        Grid dimensions come from ``level_data.tilemap`` — no longer from
        ``CFG.display.fine_w/fine_h``. The CSV decides the world size.
        """
        self.level = level_data
        h, w = int(level_data.tilemap.shape[0]), int(level_data.tilemap.shape[1])
        self._h, self._w = h, w

        # Field grids (allocate up front; populate from level + caches below)
        self.material     = np.zeros((h, w), dtype=np.int8)
        self.wall_hp      = np.zeros((h, w), dtype=np.float32)
        self.is_wall      = np.zeros((h, w), dtype=bool)
        self.is_vacuum    = np.zeros((h, w), dtype=bool)
        self.flammable    = np.zeros((h, w), dtype=bool)
        self.atmosphere   = np.ones((h, w), dtype=np.float32)
        self.wave_p       = np.zeros((h, w), dtype=np.float32)
        self.wave_v       = np.zeros((h, w), dtype=np.float32)
        self.wave_source  = np.zeros((h, w), dtype=np.float32)
        self.wind_x       = np.zeros((h, w), dtype=np.float32)
        self.wind_y       = np.zeros((h, w), dtype=np.float32)
        self.smoke        = np.zeros((h, w), dtype=np.float32)
        self.fire         = np.zeros((h, w), dtype=np.float32)
        self.obstacles    = np.zeros((h, w), dtype=bool)
        self.light_map    = np.zeros((h, w), dtype=np.float32)

        # Populate material + vacuum from the level's CSV.
        mat, vac = materials_from_tilemap(level_data.tilemap)
        self.material[:] = mat
        self.is_vacuum[vac] = True

        self._update_caches()

    # ------------------------------------------------------------------
    # Cache rebuild
    # ------------------------------------------------------------------
    def _update_caches(self):
        """Rebuild cached arrays from the material grid.

        Atmosphere starts at 1.0 in interior air, 0.0 at walls and vacuum.
        ``is_wall`` covers hull + wood + door (the latter is temporary —
        doors occlude smoke/light until the proper door system lands,
        even though ``is_passable_block`` still lets units walk through).
        Flammable covers wood only. HP comes from
        ``CFG.materials.<name>[0]`` per material ID.
        """
        m = self.material
        # TODO: drop MAT_DOOR from is_wall when the dynamic door system
        # is implemented — for now they occlude like static walls.
        self.is_wall = np.isin(m, [MAT_HULL, MAT_WOOD, MAT_DOOR])
        self.flammable = (m == MAT_WOOD)
        self.wall_hp = np.zeros_like(self.wall_hp)

        # HP from config (only walls with positive HP get one stamped in).
        mat_props = {
            MAT_AIR: CFG.materials.air,
            MAT_HULL: CFG.materials.hull,
            MAT_WOOD: CFG.materials.wood,
            MAT_DOOR: CFG.materials.door,
        }
        for mat_id, props in mat_props.items():
            hp = props[0]
            if hp > 0:
                self.wall_hp[m == mat_id] = hp

        # Atmosphere: 1.0 in interior air, 0.0 at walls and vacuum.
        self.atmosphere = np.where(
            self.is_wall | self.is_vacuum, 0.0, 1.0
        ).astype(np.float32)

        # Obstacles == walls until stamp_units paints unit footprints over it.
        self.obstacles = self.is_wall.copy()

    # ------------------------------------------------------------------
    # Per-tick rebuild: units act as walls for all physics
    # ------------------------------------------------------------------
    def stamp_units(self, units):
        """Rebuild ``obstacles`` = static walls + living unit footprints.

        Uses ``unit.occupied_tiles()`` so the footprint contract (spec §6)
        is the only dependency — no assumption about storage representation.
        When tiles transition from blocked to free (unit moved away or
        died), fill them with the neighbor mean of ``atmosphere`` to
        avoid spurious vacuum pulses.
        """
        h, w = self._h, self._w
        prev_obstacles = self.obstacles
        self.obstacles = self.is_wall.copy()
        for u in units:
            if not u.alive:
                continue
            for (tx, ty) in u.occupied_tiles():
                if 0 <= ty < h and 0 <= tx < w:
                    self.obstacles[ty, tx] = True

        freed = prev_obstacles & ~self.obstacles
        if freed.any():
            for fy, fx in zip(*np.where(freed)):
                if not self.is_vacuum[fy, fx]:
                    self.atmosphere[fy, fx] = self._neighbor_mean(
                        self.atmosphere, fy, fx)

    # ------------------------------------------------------------------
    # Pure queries (used by AI, combat, pathfinding)
    # ------------------------------------------------------------------
    def is_passable(self, fy, fx):
        """True if (fy, fx) is in-bounds and not a solid wall."""
        if fy < 0 or fy >= self._h or fx < 0 or fx >= self._w:
            return False
        return self.material[fy, fx] in (MAT_AIR, MAT_DOOR)

    def is_passable_block(self, fy, fx, footprint: int = 3):
        """True if a footprint-sized block at (fy, fx) is fully passable."""
        if fy < 0 or fx < 0 or fy + footprint > self._h or fx + footprint > self._w:
            return False
        block = self.material[fy:fy + footprint, fx:fx + footprint]
        return bool(np.all((block == MAT_AIR) | (block == MAT_DOOR)))

    def has_los(self, fy1, fx1, fy2, fx2):
        """Bresenham line-of-sight check. Stops on ``is_wall``."""
        h, w = self._h, self._w
        dx = abs(fx2 - fx1)
        dy = abs(fy2 - fy1)
        sx = 1 if fx1 < fx2 else -1
        sy = 1 if fy1 < fy2 else -1
        err = dx - dy
        x, y = fx1, fy1
        while True:
            if x == fx2 and y == fy2:
                return True
            if 0 <= y < h and 0 <= x < w and self.is_wall[y, x]:
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _neighbor_mean(self, field, fy, fx):
        """Mean of field values from passable (non-wall, non-vacuum) 4-neighbors."""
        h, w = field.shape
        total = 0.0
        count = 0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = fy + dy, fx + dx
            if (0 <= ny < h and 0 <= nx < w
                    and not self.is_wall[ny, nx]
                    and not self.is_vacuum[ny, nx]):
                total += field[ny, nx]
                count += 1
        return total / count if count > 0 else 0.0

    # ------------------------------------------------------------------
    # Mutators (used by explosions, fire wall burn-through)
    # ------------------------------------------------------------------
    def destroy_wall(self, fy, fx):
        """Convert (fy, fx) to air. Handles hull breach (edge => vacuum).

        Interior walls and non-edge hulls are refilled with the neighbor
        mean of ``atmosphere`` so we don't open with an artificial vacuum
        pulse. Edge hull tiles become vacuum and rely on relaxation BCs
        to drain smoothly.
        """
        h, w = self._h, self._w
        if not (0 <= fy < h and 0 <= fx < w):
            return
        was_hull = (self.material[fy, fx] == MAT_HULL)
        if self.material[fy, fx] in (MAT_HULL, MAT_WOOD, MAT_DOOR):
            self.material[fy, fx] = MAT_AIR
            self.wall_hp[fy, fx] = 0
            self.is_wall[fy, fx] = False
            self.flammable[fy, fx] = False
            if was_hull:
                if (fy < 1 or fy >= h - 1
                        or fx < 1 or fx >= w - 1):
                    # True hull breach — wall tile is on the map edge.
                    self.is_vacuum[fy, fx] = True
                    # Don't hard-zero — let relaxation BC drain smoothly.
                    self.atmosphere[fy, fx] = self._neighbor_mean(
                        self.atmosphere, fy, fx)
                else:
                    # Interior hull: fill with neighbor mean.
                    self.atmosphere[fy, fx] = self._neighbor_mean(
                        self.atmosphere, fy, fx)
            else:
                # Interior wall: fill with neighbor mean.
                self.atmosphere[fy, fx] = self._neighbor_mean(
                    self.atmosphere, fy, fx)
