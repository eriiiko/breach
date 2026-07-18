"""The `light` exemplar class — the first registered entity (Arc A patch 1).

Registry-only: it mirrors today's :class:`level_loader.LightEntry` vocabulary
(engine/15 §2.2, P4 — render-only, Erik's locked call 2026-07-07) field for
field, proving the schema machinery end to end. The legacy ``[[light]]``
loader path is untouched; the A3 ``[[entity]]`` patch makes ``[[light]]`` a
legacy alias of this class.

Every field is :data:`~simulation.entities.schema.KIND_FLOAT_RENDER` (or
enum/color): level lights never enter synced sim state — no Q16.16 snap, same
class as ``light_rgb``. The forbidden ``heat``/``jitter`` knobs (P4 critique
M2) simply do not exist in this schema, mirroring the loader's rejection.
"""
from __future__ import annotations

from simulation.entities.schema import (
    Entity, Field, KIND_COLOR_RGB, KIND_ENUM, KIND_FLOAT_RENDER, register,
)


@register
class light(Entity):
    """One level light: static lamp or rotating beacon (render-only)."""

    INTANGIBLE = True   # a lamp never occupies the grid (design §5)

    FIELDS = (
        Field("x", KIND_FLOAT_RENDER, default=0.0,
              doc="tile coords (tile centers at .5)"),
        Field("y", KIND_FLOAT_RENDER, default=0.0),
        Field("color", KIND_COLOR_RGB, default=(255, 255, 255),
              doc="toml carries 0-255 ints; render converts to 0-1 floats"),
        Field("intensity", KIND_FLOAT_RENDER, default=1.0, minimum=0.0),
        Field("range", KIND_FLOAT_RENDER, default=12.0, minimum=0.0,
              doc="tiles"),
        Field("kind", KIND_ENUM, default="static",
              choices=("static", "beacon")),
        Field("period_s", KIND_FLOAT_RENDER, default=2.0, minimum=0.0,
              doc="beacon: seconds per full rotation"),
        Field("beam_deg", KIND_FLOAT_RENDER, default=30.0, minimum=0.0,
              maximum=360.0, doc="beacon: cone width in degrees"),
        Field("phase", KIND_FLOAT_RENDER, default=0.0, minimum=0.0,
              maximum=1.0,
              doc="beacon: fraction of a turn; a red/blue cop-car pair = "
                  "two beacons, phase 0.0 / 0.5"),
    )
    # No inputs, no class signals in v1 — a light emits only the free
    # `alive` and is driven by nothing until Arc B wires exist.
