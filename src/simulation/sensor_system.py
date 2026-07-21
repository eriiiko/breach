"""Sensor RUNTIME — the slot-9e step-(a) sample (Arc B patch B3).

Design: docs/arc_b_impl_2026-07-21.md §3/§4 (v2, D6-D9/D13). This is the
sim-side sampling of the v1 sensor catalog: each sensor reads THIS tick's
post-physics world — field sensors through the ONE §5a accessor
(:class:`simulation.sensor_accessor.EntityFieldAccessor`), ``sensor_motion``
over ``sim.units``, ``clock`` off the sim tick — and publishes ``value`` to the
SignalBus at 9e(a), BEFORE any node reads it (§2b). The SCHEMA (fields /
signals / the per-channel sample family) lives in the import-light
:mod:`simulation.entities.sensors`; everything that touches the accessor / the
gmap / the unit list lives HERE, exactly as the door + node runtimes do.

Fail-deadly (D13): a DEAD sensor (``alive == 0``) publishes 0 — never a stale
last reading. The direction-dependent fail-safe framing (a `p<0.8 → close` wire
slams on a dead sensor; adding `require_alive` makes it fail-passive) is the
author's per-wire choice (§2d) — this module only guarantees the honest 0.

Determinism (§9): integer-only (Q16.16 / counts); the sample tiles + area discs
+ r_tiles are quantized ONCE at load by exact rules (no float in the synced
path — the disc/threshold arithmetic is pure integer); ``sensor_motion`` reads
units in unit-id order; no RNG, no dequantize (§5a — the accessor returns the
raw mirror integer).
"""
from __future__ import annotations

from simulation.entities import REGISTRY
from simulation.entities.door import quantize_span_tiles
from simulation.sensor_accessor import (
    Channel, EntityFieldAccessor, build_site_index,
)


def is_sensor(class_name) -> bool:
    """True iff ``class_name`` is a registered SENSOR class (§4 marker)."""
    cls = REGISTRY.get(class_name)
    return bool(cls is not None and getattr(cls, "SENSOR", False))


def _base_tile_size(level) -> float:
    """The BASE ``tile_size_m`` (S1 recovery, mirrors door_system): the
    unscaled size a ``--res`` run divided away, so load-time quantization
    happens at base resolution and replicates by ``res_factor``."""
    base = getattr(level, "tile_size_m_base", None)
    return float(base) if base is not None else float(level.tile_size_m)


def _disc_tiles(cy, cx, r_tiles, h, w) -> list:
    """Absolute ``(fy, fx)`` tiles of the disc of radius ``r_tiles`` centered
    at ``(cy, cx)`` — the STRICT ``dist < radius`` rule (design §4, FieldEdit),
    i.e. ``dy² + dx² < r_tiles²``, clipped to the ``[0, h) × [0, w)`` grid so
    the accessor never indexes out of bounds."""
    tiles = []
    r2 = r_tiles * r_tiles
    for dy in range(-r_tiles, r_tiles + 1):
        for dx in range(-r_tiles, r_tiles + 1):
            if dy * dy + dx * dx < r2:
                fy, fx = cy + dy, cx + dx
                if 0 <= fy < h and 0 <= fx < w:
                    tiles.append((fy, fx))
    return tiles


# ---------------------------------------------------------------------------
# Per-sensor runtime objects — one per instance, ordinal order. Each carries
# the entity instance (for the live `alive` check) + its precomputed sampling
# geometry, and evaluates to an integer `value` for 9e(a).
# ---------------------------------------------------------------------------

class _FieldSensorRuntime:
    """A field sensor: sample ONE gmap channel through the §5a accessor (§4).

    ``disc`` is None for a single-tile probe (``accessor.sample``) or the
    precomputed disc tiles for an area-mean (``accessor.area`` → floored). NO
    dequantize — the accessor returns the raw Q16.16 mirror integer.
    """

    __slots__ = ("inst", "value_slot", "channel", "sample_tile", "disc")

    def __init__(self, inst, value_slot, channel, sample_tile, disc):
        self.inst = inst
        self.value_slot = int(value_slot)
        self.channel = channel                 # a Channel enum member
        self.sample_tile = (int(sample_tile[0]), int(sample_tile[1]))
        self.disc = disc                       # None or list of (fy, fx)

    def evaluate(self, sim) -> int:
        acc = sim._sensor_accessor
        if self.disc is None:
            fy, fx = self.sample_tile
            return acc.sample(self.channel, fy, fx)
        total, n = acc.area(self.channel, self.disc)
        return total // n if n > 0 else 0      # floored area-mean (§6)


class _ClockSensorRuntime:
    """A ``clock``: ``value`` = ``tick // period`` — no field, no unit read."""

    __slots__ = ("inst", "value_slot", "period")

    def __init__(self, inst, value_slot, period):
        self.inst = inst
        self.value_slot = int(value_slot)
        self.period = max(1, int(period))      # snapped >= 1 at load

    def evaluate(self, sim) -> int:
        return int(sim.tick) // self.period


class _MotionSensorRuntime:
    """A ``sensor_motion`` plate: count living units in range (§4, N3).

    ``dist²(sensor_tile, unit_anchor_corner) ≤ r2`` (inclusive, per §4) with
    the unit's ANCHOR CORNER ``(tile_y, tile_x)`` (the pinned canon, N3);
    optional LOS with the SENSOR as Bresenham origin (asymmetric — a shut door
    blocks sight); ``faction`` (-1 = any) and ``min_footprint`` filters. Reads
    ``sim.units`` in unit-id order.
    """

    __slots__ = ("inst", "value_slot", "sy", "sx", "r2", "needs_los",
                 "faction", "min_footprint")

    def __init__(self, inst, value_slot, sy, sx, r2, needs_los, faction,
                 min_footprint):
        self.inst = inst
        self.value_slot = int(value_slot)
        self.sy = int(sy)
        self.sx = int(sx)
        self.r2 = int(r2)
        self.needs_los = bool(needs_los)
        self.faction = int(faction)
        self.min_footprint = int(min_footprint)

    def evaluate(self, sim) -> int:
        gmap = sim.gmap
        count = 0
        for u in sorted(sim.units, key=lambda u: int(u.id)):
            if not u.alive:
                continue
            uy, ux = int(u.tile_y), int(u.tile_x)
            if (uy - self.sy) ** 2 + (ux - self.sx) ** 2 > self.r2:
                continue
            if self.faction != -1 and int(u.team) != self.faction:
                continue
            if int(u.footprint) < self.min_footprint:
                continue
            if self.needs_los and not gmap.has_los(self.sy, self.sx, uy, ux):
                continue                       # sensor is the Bresenham origin
            count += 1
        return count


# ---------------------------------------------------------------------------
# Build + sweep — wired into simulation.py's slot-9e block.
# ---------------------------------------------------------------------------

def build_sensors(sim) -> list:
    """Build the ordinal-ordered sensor runtime list for the 9e(a) sample and
    the §5a accessor (with its frozen site index) — attached to ``sim`` as
    ``_sensors`` / ``_sensor_accessor`` (§3/§4).

    Sample tiles + area discs + motion radii are quantized ONCE here from the
    authored base-resolution fields, scaled to the gmap grid by ``res_factor``
    (the door pattern — quantize at base, replicate by rf). ``sim._signal_bus``
    must exist (the caller gates on it); every sensor's ``value`` slot is
    present (build_signal_bus added it).
    """
    bus = sim._signal_bus
    gmap = sim.gmap
    h, w = gmap.solid.shape
    level = sim.level
    base_ts = _base_tile_size(level)
    rf = int(getattr(level, "res_factor", 1) or 1)

    sensor_ents = sorted((e for e in sim.entities if is_sensor(e.class_name)),
                         key=lambda e: int(e.ordinal))
    sensors: list = []
    field_runtimes: list = []             # feed the frozen site index (§3)
    for e in sensor_ents:
        cls = REGISTRY[e.class_name]
        value_slot = bus.slot(int(e.ordinal), "value")
        cn = e.class_name
        if getattr(cls, "CHANNEL_NAME", None) is not None:   # a field sensor
            channel = Channel(cls.CHANNEL_NAME)
            by, bx = cls.resolve_sample_tile(e.fields)       # base tiles
            fy, fx = rf * int(by), rf * int(bx)              # → gmap grid
            disc = None
            area_m = e.fields.get("area_m", 0.0)
            if area_m and float(area_m) > 0.0:
                # Quantize the radius at BASE resolution, replicate by rf — the
                # door rule; an area smaller than one tile falls back to a
                # single-tile probe (r_tiles clamps to >= 1).
                r_tiles = rf * quantize_span_tiles(
                    area_m, base_ts, context=f"sensor '{e.id}' area")
                disc = _disc_tiles(fy, fx, r_tiles, h, w)
            rt = _FieldSensorRuntime(e, value_slot, channel, (fy, fx), disc)
            sensors.append(rt)
            field_runtimes.append(rt)
        elif cn == "clock":
            sensors.append(_ClockSensorRuntime(
                e, value_slot, e.fields["period"]))
        elif cn == "sensor_motion":
            sy, sx = rf * int(e.fields["y"]), rf * int(e.fields["x"])
            radius_m = e.fields["radius"]
            r_tiles = (rf * quantize_span_tiles(
                radius_m, base_ts, context=f"sensor '{e.id}' radius")
                if float(radius_m) > 0.0 else 0)
            sensors.append(_MotionSensorRuntime(
                e, value_slot, sy, sx, r_tiles * r_tiles,
                bool(e.fields["needs_los"]), int(e.fields["faction_filter"]),
                int(e.fields["min_footprint"])))
        else:                             # pragma: no cover - registry drift
            raise ValueError(
                f"sensor class {cn!r} has no B3 runtime — the registry and "
                f"sensor_system.build_sensors disagree")

    sim._sensors = sensors
    sim._sensor_accessor = EntityFieldAccessor(gmap, build_site_index(
        field_runtimes))
    return sensors


def sample_sensors(sim) -> None:
    """9e(a): sample every sensor in ORDINAL order and publish ``value`` to
    ``pub`` (§2b step a). A DEAD sensor (``alive == 0``) publishes 0 —
    fail-deadly, never a stale reading (D13)."""
    bus = sim._signal_bus
    for s in sim._sensors:
        alive = bool(getattr(s.inst, "alive", True))
        bus.set_pub(s.value_slot, s.evaluate(sim) if alive else 0)
