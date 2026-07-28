"""Cover runtime — continuous-space rectangles that eat bullets (design §7).

The sim-side half of :mod:`simulation.entities.cover` (which stays schema-only,
because the entities package is import-light by contract). Mirrors
:mod:`simulation.door_system`: the class declares the schema and the canonical
quantization, this module builds runtime objects and owns their behaviour.

The whole point, restated from §7: a bullet's fate is decided by GEOMETRY. The
march asks "did I cross this rectangle?", and if so the round stops there and
chews it. There is no exposure roll, no cover bonus, no directional bookkeeping
— a flanking approach simply does not have the crate in the way.

Determinism: the rectangle is derived once at load from the authored meters
(the door-span quantization rule), and every runtime test is plain IEEE
comparison and arithmetic on tile floats — the same discipline the bullet march
already runs on. No RNG, no libm, ordinal iteration order everywhere.
"""
from __future__ import annotations

from simulation.entities.cover import quantize_extent_tiles
from simulation import wall_fixed


def _base_tile_size(level_data) -> float:
    """The a6 §3 recovery: ``tile_size_m_base`` when a ``--res`` run already
    divided the live tile size, else the unscaled value."""
    base = getattr(level_data, "tile_size_m_base", None)
    return float(base) if base is not None else float(level_data.tile_size_m)


class CoverRuntime:
    """One live cover rectangle.

    Geometry is a half-open AABB in TILE coordinates — ``[x0, x1) x [y0, y1)``
    — the same convention a unit's footprint uses (``tile_x <= ix < tile_x +
    footprint``), so "inside a crate" and "inside a body" mean the same thing
    to the marcher and there is no off-by-one seam between them.

    Wraps the parsed ``EntityInstance`` and exposes the serializer duck-type
    (``ordinal`` / ``id`` / ``class_name`` / ``fields`` + ``alive``), exactly as
    :class:`~simulation.door_system.DoorRuntime` does, so cover rows ride the
    existing entity digest with no special-casing.
    """

    __slots__ = ("inst", "x0", "y0", "x1", "y1", "hp_now", "hp_max",
                 "blocks_los", "alive")

    def __init__(self, inst, x0, y0, w_tiles, h_tiles):
        self.inst = inst
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.x1 = float(x0 + w_tiles)
        self.y1 = float(y0 + h_tiles)
        self.hp_max = int(inst.fields.get("hp", 60))
        self.hp_now = self.hp_max
        self.blocks_los = bool(inst.fields.get("blocks_los", False))
        self.alive = True

    # -- serializer duck-type (the DoorRuntime precedent) ---------------
    @property
    def ordinal(self):
        return self.inst.ordinal

    @property
    def id(self):
        return self.inst.id

    @property
    def class_name(self):
        return self.inst.class_name

    @property
    def fields(self):
        return self.inst.fields

    # -- geometry -------------------------------------------------------
    def contains(self, x: float, y: float) -> bool:
        """Is the continuous point inside this rectangle? Half-open, so a
        rectangle ending at x=10.0 does not own x=10.0 — which is what keeps
        two abutting crates from both claiming the seam between them."""
        return (self.alive and self.x0 <= x < self.x1
                and self.y0 <= y < self.y1)

    def blocks_segment(self, x0: float, y0: float, x1: float,
                       y1: float) -> bool:
        """Does the segment cross this rectangle? (slab test)

        Used by :func:`simulation.vision.ray_clear` for the ``blocks_los``
        rectangles only. The bullet march does NOT use this — it tests its own
        marched positions with :meth:`contains`, so that a round stops AT the
        crate's face and chews the thing it actually hit, rather than being
        told after the fact that its whole flight was obstructed.

        The classic slab algorithm, in plain IEEE arithmetic: clip the
        parametric range ``t in [0, 1]`` against each axis's slab; an empty
        range means no crossing. Axis-parallel segments are handled by the
        degenerate branch (no divide by zero).
        """
        if not self.alive:
            return False
        t0, t1 = 0.0, 1.0
        for (p, d, lo, hi) in ((x0, x1 - x0, self.x0, self.x1),
                               (y0, y1 - y0, self.y0, self.y1)):
            if d == 0.0:
                if p < lo or p >= hi:
                    return False
                continue
            inv = 1.0 / d
            ta = (lo - p) * inv
            tb = (hi - p) * inv
            if ta > tb:
                ta, tb = tb, ta
            if ta > t0:
                t0 = ta
            if tb < t1:
                t1 = tb
            if t0 > t1:
                return False
        return True

    # -- damage ---------------------------------------------------------
    def chew(self, damage) -> bool:
        """Take structural damage; return True if this broke the cover.

        §7: cover is static-but-DESTRUCTIBLE. When it breaks it stops being
        cover in every sense at once — rounds pass, ``blocks_segment`` stops
        occluding, and the exposed-profile metric immediately reports the
        target as more exposed, with no separate bookkeeping to keep in sync.
        """
        if not self.alive or damage <= 0:
            return False
        self.hp_now -= int(damage)
        if self.hp_now <= 0:
            self.hp_now = 0
            self.alive = False
            return True
        return False

    def __repr__(self):
        return (f"CoverRuntime({self.id!r}, [{self.x0},{self.x1}) x "
                f"[{self.y0},{self.y1}), hp={self.hp_now}/{self.hp_max}, "
                f"los={self.blocks_los})")


def cover_instances(level_data) -> list:
    """The level's cover ``EntityInstance``s, in ordinal order."""
    ents = getattr(level_data, "entities", None) or []
    rows = [e for e in ents if e.class_name == "cover"]
    rows.sort(key=lambda e: int(e.ordinal))
    return rows


def build_cover(level_data) -> list:
    """Build the runtime cover list (ordinal order) for a level.

    Extents quantize at BASE resolution and are then multiplied by
    ``res_factor``, never re-quantized at the scaled tile size — the same rule
    door spans follow (a6 §3), and for the same reason: a float recompute is
    only exact for power-of-two factors, so the base is carried.
    """
    ts = _base_tile_size(level_data)
    rf = int(getattr(level_data, "res_factor", 1) or 1)
    out = []
    for inst in cover_instances(level_data):
        ctx = f"cover entity '{inst.id}'"
        w = quantize_extent_tiles(inst.fields.get("width_m", 1.0), ts,
                                  context=ctx) * rf
        h = quantize_extent_tiles(inst.fields.get("height_m", 1.0), ts,
                                  context=ctx) * rf
        out.append(CoverRuntime(inst, int(inst.fields["x"]) * rf,
                                int(inst.fields["y"]) * rf, w, h))
    return out


def cover_at(cover_list, x: float, y: float):
    """The first live cover rectangle containing the point, or ``None``.

    Ordinal order, so overlapping authored rectangles resolve deterministically
    (and the same way every tick).
    """
    for c in cover_list:
        if c.contains(x, y):
            return c
    return None


__all__ = ["CoverRuntime", "build_cover", "cover_at", "cover_instances"]
