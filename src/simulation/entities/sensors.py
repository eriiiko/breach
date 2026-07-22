"""The v1 sensor catalog — Arc B patch B3 (impl doc §4, sensor catalog).

Design: docs/arc_b_impl_2026-07-21.md §4 (v2, D7/D8/D9/D13 folded). This module
carries the SCHEMA (fields / signals) for the L0 sensor classes only; the
sim-side RUNTIME (reading the physics fields through the §5a accessor and the
unit list) lives in :mod:`simulation.sensor_system`, exactly as the door / node
runtimes live sim-side. This package stays IMPORT-LIGHT (stdlib only — design
§3b, CI-tested in tests/test_entities_import_light.py): NOTHING here imports the
accessor, the gmap, or numpy — a sensor class is pure declaration.

Every sensor is ``SENSOR = True``: its free-standing ``value`` signal is SAMPLED
from the world at slot-9e(a) and published to the bus (NOT a node output — never
swapped; refreshed every tick, and a DEAD sensor writes 0 — fail-deadly, D13).
Sensors are physical by default (they occupy their mount tile like a `button`),
`intangible` per instance where the author wants a free-floating probe (§4);
`clock` is the exception — it reads no field and holds no tile, so it defaults
intangible like a logic node.

The catalog (§4):

- **Field sensors** ``pressure`` / ``smoke`` / ``water_depth`` / ``o2`` /
  ``temperature`` / ``fire`` — one per FROZEN accessor channel (§3). Each reads
  ONE tile of ONE gmap field through the accessor and emits it as ``value``
  (Q16.16, NO dequantize). **The sample family is per-channel (D7/D8):**

  - AIR family (``pressure``/``smoke``/``water_depth``/``o2``): the sensor is
    mounted in a solid wall body and FACES an adjacent air tile via its
    ``sample_dx``/``sample_dy`` offset; it samples that FACED AIR tile (dodging
    a solid body that would read 0, design §5).
  - BODY family (``temperature``/``fire``, D7/D8): those fields live on the
    SOLID / flammable matter, not the air — so the sensor samples its OWN mount
    (anchor) tile, the body whose heat / fire it reads. The faced air tile would
    read 0 forever (the D7/D8 regression this fixes).

  Optional ``area_m`` radius → an area-MEAN over a disc (``dist < radius``
  strict), masked to currently-non-solid tiles via the frozen ``solid`` channel
  (D6) and floored (§4/§6). ``area_m`` is authoring-bound (KIND_LENGTH_M,
  quantized once at load by the canonical rule — never hashed, never raw in the
  sim path); its synced consequence is the sampled ``value``.

- **``clock``** — emits ``value`` = an integer counter derived from the tick
  (``tick // period``), ``period`` snapped ≥ 1 at load. Reads no field.
- **``sensor_motion``** — emits ``value`` = the count of living units within a
  quantized ``radius`` (``dist² ≤ r_tiles²``, unit ANCHOR CORNER `(tile_y,
  tile_x)` — the pinned canon, N3) optionally gated on line-of-sight (the sensor
  as the Bresenham ORIGIN — asymmetric; a shut door on the ray blocks sight),
  a faction filter, and a minimum footprint. Reads ``sim.units`` in unit-id
  order — deterministic, integer.

``hp_below`` / hostiles-of / ``chip.carried_by`` / ``win(...)`` are stack-2
riders, format-reserved, NOT built here (§4).
"""
from __future__ import annotations

from simulation.entities.schema import (
    Entity, Field, KIND_BOOL, KIND_INT, KIND_LENGTH_M, Signal, register,
)

# The two per-channel sample families (D7/D8) — see the module doc.
SAMPLE_AIR = "air"     # sample the FACED air tile (anchor + offset)
SAMPLE_BODY = "body"   # sample the mount (anchor) tile — temperature/fire


class _FieldSensor(Entity):
    """Shared schema for the six field sensors (NOT registered directly).

    Subclasses fix ``CHANNEL_NAME`` (a frozen §3 :class:`Channel` value, mapped
    to the enum sim-side so this file imports nothing heavy) and
    ``SAMPLE_FAMILY``. All inherit the same FIELDS / SIGNALS: an anchor tile
    ``x``/``y`` (base resolution, like a door/button), a facing offset
    ``sample_dx``/``sample_dy`` (base tiles — used only by the AIR family; the
    BODY family samples the anchor), and an optional ``area_m`` radius.
    """

    INTANGIBLE = False   # a placed probe occupies its mount tile (design §5)
    SENSOR = True
    CHANNEL_NAME = None   # set per subclass (a Channel value string, §3)
    SAMPLE_FAMILY = SAMPLE_AIR

    FIELDS = (
        Field("x", KIND_INT, default=None, minimum=0,
              doc="anchor tile COL at base resolution (the mount tile) — "
                  "REQUIRED"),
        Field("y", KIND_INT, default=None, minimum=0,
              doc="anchor tile ROW at base resolution (the mount tile) — "
                  "REQUIRED"),
        Field("sample_dx", KIND_INT, default=0,
              doc="AIR-family faced-tile COL offset from the anchor (base "
                  "tiles); ignored by the BODY family (temperature/fire read "
                  "the anchor)"),
        Field("sample_dy", KIND_INT, default=0,
              doc="AIR-family faced-tile ROW offset from the anchor (base "
                  "tiles); ignored by the BODY family"),
        Field("area_m", KIND_LENGTH_M, default=0.0, minimum=0.0,
              doc="optional area-mean radius in meters (quantized once at load "
                  "by the canonical rule); 0 = a single-tile probe. The disc "
                  "is masked to currently-non-solid tiles (D6) and the mean is "
                  "floored (§4/§6)"),
    )
    SIGNALS = (Signal("value", "the sampled field value (Q16.16, no "
                      "dequantize) — a single tile or the floored area-mean"),)

    @classmethod
    def resolve_sample_tile(cls, fields) -> tuple:
        """The BASE-resolution ``(fy, fx)`` this sensor samples (§4, D7/D8).

        AIR family → the FACED tile ``(y + sample_dy, x + sample_dx)``; BODY
        family (``temperature``/``fire``) → the anchor ``(y, x)`` itself. The
        sim-side runtime scales this by ``res_factor`` to the gmap grid.
        """
        x, y = int(fields["x"]), int(fields["y"])
        if cls.SAMPLE_FAMILY == SAMPLE_AIR:
            return (y + int(fields["sample_dy"]), x + int(fields["sample_dx"]))
        return (y, x)


@register
class pressure(_FieldSensor):
    """Air-pressure probe — the materialized ``atmosphere`` P (air, §3)."""
    CHANNEL_NAME = "pressure"
    SAMPLE_FAMILY = SAMPLE_AIR


@register
class smoke(_FieldSensor):
    """Black-smoke density probe — ``gas[SMOKE]`` (air, §3)."""
    CHANNEL_NAME = "smoke"
    SAMPLE_FAMILY = SAMPLE_AIR


@register
class water_depth(_FieldSensor):
    """Standing-water depth probe — ``water_depth`` (air/floor, §3)."""
    CHANNEL_NAME = "water_depth"
    SAMPLE_FAMILY = SAMPLE_AIR


@register
class o2(_FieldSensor):
    """Oxygen MASS-DENSITY probe — ``gas[O2]`` (air, D9).

    NOT a partial pressure: no ``p_O2`` field exists and deriving one needs a
    forbidden ×T dequantize (§5a). This is the pure-gather density channel; its
    temperature dependence is documented, not derived (D9).
    """
    CHANNEL_NAME = "o2"
    SAMPLE_FAMILY = SAMPLE_AIR


@register
class temperature(_FieldSensor):
    """Temperature probe — samples the SOLID BODY tile (D7).

    Temperature lives on solids only; a faced-air sample would read 0 forever,
    so this BODY-family sensor reads its own mount (anchor) tile.
    """
    CHANNEL_NAME = "temperature"
    SAMPLE_FAMILY = SAMPLE_BODY


@register
class fire(_FieldSensor):
    """Fire-intensity probe — samples the flammable BODY tile (D8).

    Fire only burns flammable matter, not air; like ``temperature`` this
    BODY-family sensor reads its own mount (anchor) tile.
    """
    CHANNEL_NAME = "fire"
    SAMPLE_FAMILY = SAMPLE_BODY


@register
class clock(Entity):
    """A tick-driven counter — emits ``value`` = ``tick // period`` (§4).

    Reads no field and holds no tile (pure logic, like a node — intangible by
    default). ``period`` is snapped ≥ 1 at load; the counter resets each round
    with the sim tick (a documented v1 property).
    """

    INTANGIBLE = True    # no field, no tile — pure tick logic
    SENSOR = True

    FIELDS = (
        Field("period", KIND_INT, default=1, minimum=1,
              doc="ticks per counter increment (value = tick // period); "
                  "snapped >= 1 at load"),
    )
    SIGNALS = (Signal("value", "integer counter = tick // period"),)


@register
class sensor_motion(Entity):
    """A proximity / presence plate — counts living units in range (§4, N3).

    ``value`` = the count of living units whose ANCHOR CORNER ``(tile_y,
    tile_x)`` (the pinned canon, N3) is within the quantized ``radius``
    (``dist² ≤ r_tiles²``), optionally gated on ``needs_los`` (the sensor as
    the Bresenham ORIGIN — asymmetric; a shut door on the ray blocks sight), a
    ``faction_filter`` (team int, or -1 = any), and ``min_footprint``. Physical
    by default (it occupies its mount tile). Reads ``sim.units`` in unit-id
    order — deterministic, integer.
    """

    INTANGIBLE = False   # a presence plate occupies its mount tile
    SENSOR = True

    FIELDS = (
        Field("x", KIND_INT, default=None, minimum=0,
              doc="plate anchor tile COL at base resolution — REQUIRED"),
        Field("y", KIND_INT, default=None, minimum=0,
              doc="plate anchor tile ROW at base resolution — REQUIRED"),
        Field("radius", KIND_LENGTH_M, default=1.0, minimum=0.0,
              doc="detection radius in meters (quantized to r_tiles once at "
                  "load); a unit counts iff dist² <= r_tiles²"),
        Field("needs_los", KIND_BOOL, default=False,
              doc="when true a unit must have unblocked line-of-sight FROM the "
                  "sensor (Bresenham origin = the sensor; a shut door blocks "
                  "sight — intended)"),
        Field("faction_filter", KIND_INT, default=-1,
              doc="team int to count (0 = marine, 1 = zombie), or -1 = any "
                  "team"),
        Field("min_footprint", KIND_INT, default=0, minimum=0,
              doc="ignore units whose footprint side is below this (0 = count "
                  "all)"),
    )
    SIGNALS = (Signal("value", "count of living units in range "
                      "[with LOS] matching the filter"),)
