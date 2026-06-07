"""GameMap — world state container (tiles + physics fields).

Lifted from ``game.py:GameMap`` (lines 292-542 in the legacy file) and merged
with the level-driven constructor that previously lived as a shim in
``main.py``. Canonical signature is ``GameMap(level_data)`` — the no-arg
form and ``_build_ship`` fallback from the legacy implementation are gone
(CSV loading via :mod:`level_loader` is the only path now).

Owns the cached arrays the physics systems read and write:

    material, wall_hp, is_wall, is_vacuum, flammable,
    atmosphere, wave_p, wave_v, wave_source, wind_x, wind_y,
    smoke, fire, obstacles, light_map, heat, smoke_glow

Plus ``self.level`` (the :class:`level_loader.LevelData` instance) and the
methods needed by combat / pathfinding / physics (``stamp_units``,
``is_passable``, ``is_passable_block``, ``has_los``, ``destroy_wall``).

No pygame, no pyray — pure numpy + config.
"""
from __future__ import annotations

import numpy as np

from config import CFG
from level_loader import materials_from_tilemap

# Material IDs are defined once in :mod:`simulation.materials` (the single
# source of truth) and re-exported here so existing
# ``from simulation.gamemap import MAT_*`` imports keep working.
from simulation.materials import (  # noqa: F401  (re-exported)
    MAT_AIR,
    MAT_HULL,
    MAT_WOOD,
    MAT_DOOR,
    MAT_STEEL,
    MAT_GLASS,
    MaterialTable,
)


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

        # Material-property table (ch.02): single source of every per-material
        # constant. Derived caches below are projections of this table indexed
        # by the ``material`` grid. Rebuilt on config hot-reload via
        # :meth:`reload_material_table`.
        self.materials = MaterialTable.from_config(CFG)

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
        # Scalar light field (legacy: fire raycaster output + render unit/smoke
        # tinting). Kept alongside light_rgb during the RGB migration.
        self.light_map    = np.zeros((h, w), dtype=np.float32)
        # RGB light field (ch.03 render byproduct): total light colour reaching
        # each tile, summed over all sources. Shape (h, w, 3), f32 accumulator
        # down-converted to the RGBA16F render textures at pack time (ch.05).
        self.light_rgb    = np.zeros((h, w, 3), dtype=np.float32)
        # Per-tile STATIC light attenuation (ch.02/03): the material table's
        # ``light_atten`` projected onto the grid, shape (h, w, 3) f32. This is
        # the static half of ``total_atten = material(static) × dynamic(live)``
        # — a structural-change cache (rebuilt in _update_caches, patched per
        # tile in on_tile_changed), NOT recomputed each tick. The directional
        # ray march reads it per channel: opaque [1,1,1] kills the ray (== old
        # wall hard-stop), air [0,0,0] passes untouched, glass [0.1,..] dims.
        self.light_atten  = np.zeros((h, w, 3), dtype=np.float32)
        # Per-tile DYNAMIC light attenuation (ch.02 §static×dynamic, ch.03
        # §units): the live per-channel field the ray march actually reads.
        # Rebuilt every tick in ``stamp_units`` = static ``light_atten`` (copy)
        # combined per-channel via MAX with each living unit's opacity stamped
        # over its footprint (default [1,1,1] = opaque → unit shadow, restoring
        # pre-S2 behaviour). An occluder can only ADD opacity, never remove it.
        # Allocated once here and filled IN-PLACE each tick (never reassigned)
        # so a C++ view of the buffer never goes stale (project gotcha:
        # in-place writes vs reassignment). Away from units it equals the
        # static field, so behaviour matches S2 in unoccupied regions.
        self.dyn_light_atten = np.zeros((h, w, 3), dtype=np.float32)
        # Per-tile thermal conductivity (table-derived). Allocated + populated
        # now; consumed later by the temperature/conduction pass (ch.04).
        self.conductivity = np.zeros((h, w), dtype=np.float32)
        # Per-tile gas/smoke PERMEABILITY (table-derived): 0 = sealed wall,
        # 1 = open air, partial = porous. The physics-solid boundary derives
        # from this (a tile is solid to flow iff permeability == 0), replacing
        # the old occlusion-flag (is_wall) as the gas/wave boundary source. For
        # the current materials it equals the is_wall set, so behaviour is
        # unchanged; the gas/smoke solver consuming it as a *continuous* field
        # (partial units/grills) lands in a later step (ch.04).
        self.permeability = np.ones((h, w), dtype=np.float32)
        # Per-tile DYNAMIC gas/smoke permeability (ch.04 §3a): the live field
        # the C++ flux gather actually reads, = static ``permeability`` (copy)
        # with each living unit's footprint stamped to 0 (a unit fully blocks
        # flow this step — identical to today's obstacle stamp). Rebuilt IN-PLACE
        # each tick in ``stamp_units`` (never reassigned, so a C++ view of the
        # buffer never goes stale). Away from units it equals the static field,
        # so behaviour matches the pre-3a obstacle-mirror in unoccupied regions.
        self.dyn_permeability = np.ones((h, w), dtype=np.float32)
        # Heat deposit buffer (ch.03 output / ch.04 §Fixed-point format): the
        # only SIM-affecting ray output. Q16.16 FIXED-POINT int32 — 16 integer
        # bits, 16 fractional bits, so 1.0 energy == 65536 raw counts (the C++
        # HEAT_SCALE constant). The ray march SATURATING-adds into it (clamp at
        # INT32_MAX, never wrap). Integer += is order-independent -> determinism
        # (cross-machine / future lockstep multiplayer). Nothing READS it this
        # slice — the temperature pass (ch.04) consumes it non-destructively and
        # the per-tick deposit is cleared at cleanup. Allocated once, written
        # IN-PLACE (never reassigned) so any C++ view stays valid.
        self.heat = np.zeros((h, w), dtype=np.int32)
        # Smoke-glow buffer (ch.03 C16 / ch.05 §God-rays): RENDER-ONLY god-ray
        # glow. The light each tile's smoke ABSORBS is deposited here per
        # channel by the march (energy-conserving). Shape (h, w, 3) f32 ->
        # packed into render Texture B at pack time (ch.05). Supersedes the old
        # surface-tint light_modulation path (no double-count). float (no
        # downstream sim threshold). Allocated once, written IN-PLACE.
        self.smoke_glow = np.zeros((h, w, 3), dtype=np.float32)

        # Populate material + vacuum from the level's CSV.
        mat, vac = materials_from_tilemap(level_data.tilemap)
        self.material[:] = mat
        self.is_vacuum[vac] = True

        self._update_caches()

    # ------------------------------------------------------------------
    # Cache rebuild
    # ------------------------------------------------------------------
    def _update_caches(self):
        """Rebuild all table-derived caches from the material grid.

        Every cache is a projection of the material-property table
        (``self.materials``) indexed by ``material`` — no hardcoded material
        lists. The two distinct masks are preserved (ch.02 §two masks):

        - ``is_wall`` — the **occlusion** mask (physics/light/smoke/vision
          boundary). Derived from ``light_atten`` (a tile occludes if it
          attenuates any channel), so it includes doors (``[1,1,1]``) but not
          air (``[0,0,0]``) — exactly the old ``{HULL, WOOD, DOOR}`` set for
          the current materials.
        - ``is_passable`` (the walkability predicate, AIR+DOOR) lives in the
          query methods and is derived from the table's ``passable`` column.

        ``flammable`` and ``wall_hp`` come from the table; ``conductivity`` is
        populated for the later thermal pass. Atmosphere starts at 1.0 in
        interior air, 0.0 at walls and vacuum.
        """
        m = self.material
        tbl = self.materials

        # Occlusion mask from the table (doors occlude; air does not). Always
        # boolean-typed regardless of the input grid's dtype.
        self.is_wall = np.asarray(tbl.occludes(m), dtype=bool)
        # Static per-channel light attenuation: table column projected onto the
        # grid (ch.03 march input). C-contiguous f32 so it crosses to C++ as a
        # plain (h, w, 3) buffer with no copy.
        self.light_atten = np.ascontiguousarray(tbl.light_atten[m], dtype=np.float32)
        self.flammable = tbl.flammable[m]
        self.wall_hp = tbl.hp[m].astype(np.float32, copy=True)
        self.conductivity = tbl.conductivity[m].astype(np.float32, copy=True)
        # Gas/smoke permeability projected onto the grid (0 sealed, 1 open).
        self.permeability = tbl.permeability[m].astype(np.float32, copy=True)

        # Atmosphere: 1.0 in interior air, 0.0 at walls and vacuum.
        self.atmosphere = np.where(
            self.is_wall | self.is_vacuum, 0.0, 1.0
        ).astype(np.float32)

        # Obstacles (the physics solid boundary) == solid tiles (permeability
        # == 0) until stamp_units paints unit footprints over it. Sourced from
        # permeability, not the occlusion flag, so flow and optics can diverge.
        self.obstacles = self.permeability <= 0.0

    # ------------------------------------------------------------------
    # Incremental cache patch (single structural-edit seam — ch.02 review #10)
    # ------------------------------------------------------------------
    def on_tile_changed(self, fy, fx):
        """Patch ALL table-derived static caches for one tile after a
        structural edit to ``material[fy, fx]``.

        Centralizes cache invalidation so callers (``destroy_wall``, the future
        laser pre-phase) never patch caches inline. O(1) per tile — never an
        O(grid) ``_update_caches`` rebuild (which won't scale when a firestorm
        melts many walls per tick). Does NOT touch atmosphere/vacuum — those
        carry edit-specific semantics owned by the caller (see
        ``destroy_wall``).
        """
        if not (0 <= fy < self._h and 0 <= fx < self._w):
            return
        mat_id = int(self.material[fy, fx])
        tbl = self.materials
        self.is_wall[fy, fx] = bool(tbl.light_atten[mat_id].max() > 0.0)
        self.light_atten[fy, fx] = tbl.light_atten[mat_id]
        self.flammable[fy, fx] = bool(tbl.flammable[mat_id])
        self.wall_hp[fy, fx] = float(tbl.hp[mat_id])
        self.conductivity[fy, fx] = float(tbl.conductivity[mat_id])
        self.permeability[fy, fx] = float(tbl.permeability[mat_id])

    # ------------------------------------------------------------------
    # Config hot-reload: rebuild the table + static caches (ch.02 §14)
    # ------------------------------------------------------------------
    def reload_material_table(self):
        """Re-read the material table from config and rebuild static caches.

        Call after ``CFG.reload()``. Preserves the live ``material``/vacuum
        grids; only table-derived caches change. (A GPU material-mirror re-sync
        wires in here when CUDA lands — ch.02 §14.)
        """
        self.materials = MaterialTable.from_config(CFG)
        # Rebuild only the table-derived caches; keep atmosphere/obstacles as
        # the running sim left them by snapshotting and restoring them.
        atmosphere = self.atmosphere
        obstacles = self.obstacles
        self._update_caches()
        self.atmosphere = atmosphere
        self.obstacles = obstacles

    # ------------------------------------------------------------------
    # Per-tick rebuild: units act as walls for all physics
    # ------------------------------------------------------------------
    def stamp_units(self, units):
        """Rebuild ``obstacles`` = static walls (units are NO LONGER stamped
        here), and in the SAME pass rebuild the dynamic per-channel
        light-attenuation field ``dyn_light_atten`` and the dynamic gas/smoke
        permeability field ``dyn_permeability`` from each living unit's
        footprint (ch.04 §3b, ch.03 §units, ch.02 §static×dynamic).

        Three outputs of one pass:

        * ``obstacles`` = solid set (``permeability == 0``), i.e. WALLS ONLY.
          The C++ hard-zeroing BCs (zero pressure / Neumann skip) key off
          ``obstacles``/``is_wall``, so walls keep their hard wall behaviour
          and a unit is no longer force-zeroed — gas/pressure may exist in a
          unit's cell.
        * ``dyn_permeability`` = static ``permeability`` with each living unit's
          footprint set to a PARTIAL value ``unit_perm`` (ch.04 §3b). A unit is
          a *soft, porous body*: smoke/air seep past it (slowed by the
          ``face = min(perm)`` flux weighting), not perfectly blocked. The value
          comes from an optional per-unit ``unit.permeability`` hook, defaulting
          to ``CFG.physics.unit_permeability`` (0.5 = "slows flow, doesn't
          seal"). 0 would restore the old hard wall; 1 would be invisible.
        * ``dyn_light_atten`` = static material attenuation combined per-channel
          via MAX with each living unit's opacity (UNCHANGED — units still cast
          solid shadows). Because the field is RGB a unit can occlude *per
          colour* via an optional ``unit.light_atten`` (default ``[1,1,1]`` =
          full block → a shadow). An occluder can only ADD opacity, never remove
          it.

        Uses ``unit.occupied_tiles()`` so the footprint contract (spec §6)
        is the only dependency — no assumption about storage representation.

        When *wall* tiles transition from blocked to free (wall destroyed),
        fill them with the neighbor mean of ``atmosphere`` to avoid a spurious
        vacuum pulse. ``freed`` keys off ``prev_obstacles & ~obstacles``, which
        now only changes on wall destruction (units are no longer in
        ``obstacles``), so a moving/dying unit triggers no fill — correct, since
        a unit no longer zeros its cell's atmosphere.
        """
        h, w = self._h, self._w
        prev_obstacles = self.obstacles
        # Base = solid tiles (permeability == 0) = WALLS ONLY. Units are no
        # longer painted into ``obstacles`` (3b): they are soft bodies, not
        # hard walls, so the C++ hard-zeroing BCs must not fire on them.
        self.obstacles = self.permeability <= 0.0
        # Reset the dynamic attenuation field to the static material baseline
        # IN-PLACE (no reassignment — keeps any C++ view valid). Units then
        # raise opacity per-channel below.
        self.dyn_light_atten[:] = self.light_atten
        # Reset the dynamic permeability field to the static material baseline
        # IN-PLACE (no reassignment — keeps any C++ view valid). Units then
        # lower their footprint to a PARTIAL value below (3b: porous body).
        self.dyn_permeability[:] = self.permeability
        default_atten = (1.0, 1.0, 1.0)
        default_perm = float(getattr(CFG.physics, "unit_permeability", 0.5))
        for u in units:
            if not u.alive:
                continue
            # Per-unit opacity hook: a unit may declare ``light_atten`` (RGB)
            # to occlude per colour; default is fully opaque (a shadow).
            u_atten = getattr(u, "light_atten", default_atten)
            # Per-unit permeability hook (mirrors light_atten): a unit may
            # declare ``permeability`` (e.g. a denser/looser body); default is
            # the config value (porous, slows flow but does not seal).
            u_perm = float(getattr(u, "permeability", default_perm))
            for (tx, ty) in u.occupied_tiles():
                if 0 <= ty < h and 0 <= tx < w:
                    # Unit is a soft body: partial permeability, NOT an obstacle.
                    self.dyn_permeability[ty, tx] = u_perm
                    # Per-channel MAX: opacity can only increase.
                    cell = self.dyn_light_atten[ty, tx]
                    cell[0] = cell[0] if cell[0] >= u_atten[0] else u_atten[0]
                    cell[1] = cell[1] if cell[1] >= u_atten[1] else u_atten[1]
                    cell[2] = cell[2] if cell[2] >= u_atten[2] else u_atten[2]

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
    # Over-pressure wall failure — the emergent pressure-relief valve (ch.04 §5)
    # ------------------------------------------------------------------
    def find_burst_walls(self, max_pops: int | None = None):
        """Find wall tiles holding a pressure differential above their material's
        ``burst_threshold``. Pure scan — does NOT mutate state.

        A sealed room that keeps absorbing grenades builds pressure without
        limit; this is the emergent relief valve (ch.04 §5). For each wall tile,
        the differential it holds is the **spread across its opposing sides**:
        ``max(neighbour atmosphere) - min(neighbour atmosphere)`` over its
        in-bounds 4-neighbours, where a *solid or sealed-vacuum* neighbour
        contributes 0 (so a hull between a pressurised room and outside-vacuum
        sees ``p_room - 0``). A wall between two equal-pressure rooms has ~0
        spread and never pops — correct.

        A material with ``burst_threshold <= 0`` is treated as never-bursting
        (air, or any material omitting the column).

        Parameters
        ----------
        max_pops
            Optional cap. When set, only the ``max_pops`` worst-differential
            tiles are returned (sorted descending), so a mistuned threshold
            cannot nuke the whole ship in one tick.

        Returns
        -------
        list of (int, int)
            ``(fy, fx)`` wall tiles that should fail this tick. Caller runs
            :meth:`destroy_wall` on each (mirrors fire burn-through plumbing).
        """
        h, w = self._h, self._w
        atm = self.atmosphere
        is_wall = self.is_wall
        is_vacuum = self.is_vacuum
        thresh = self.materials.burst_threshold

        failing = []  # (differential, fy, fx)
        ys, xs = np.where(is_wall)
        for fy, fx in zip(ys.tolist(), xs.tolist()):
            mat_id = int(self.material[fy, fx])
            t = float(thresh[mat_id])
            if t <= 0.0:
                continue  # n/a material (e.g. air) never bursts
            lo = None
            hi = None
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = fy + dy, fx + dx
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                # A solid neighbour (wall, incl. sealed-hull which is also
                # is_wall) or an exposed-vacuum breach holds no air → 0; an
                # air tile contributes its atmosphere.
                if is_wall[ny, nx] or is_vacuum[ny, nx]:
                    p = 0.0
                else:
                    p = float(atm[ny, nx])
                lo = p if lo is None or p < lo else lo
                hi = p if hi is None or p > hi else hi
            if lo is None:
                continue
            spread = hi - lo
            if spread > t:
                failing.append((spread, fy, fx))

        if not failing:
            return []
        # Worst differentials first; apply the per-tick cap.
        failing.sort(key=lambda r: r[0], reverse=True)
        if max_pops is not None:
            failing = failing[:max_pops]
        return [(fy, fx) for _, fy, fx in failing]

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
        # A wall is anything the occlusion mask marks (hull/wood/door today,
        # plus steel/glass when placed) — replaces the hardcoded id list.
        if self.is_wall[fy, fx]:
            self.material[fy, fx] = MAT_AIR
            # Patch ALL table-derived caches for this tile through the single
            # incremental seam (is_wall, flammable, wall_hp, conductivity) —
            # no inline cache fixups, no O(grid) rebuild.
            self.on_tile_changed(fy, fx)
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
