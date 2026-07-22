"""The §5a sensor accessor seam — ``EntityFieldAccessor`` (impl doc §3).

Design: docs/arc_b_impl_2026-07-21.md §3 (v2). Sensors read the physics fields
through exactly ONE accessor so the backing can change WITHOUT touching sensor
code — the forever-contract (escalation trigger 1):

- **CPU path (THIS arc, the stub):** ``sample`` / ``area`` read the host mirror
  (the ``gmap`` field arrays) directly. Values are already Q16.16 integers on
  the mirror — **no dequantize**, ever (the sim-path rule).
- **Resident path (later, NOT this arc — S8c concurrency constraint):** the
  same two methods are served by an ``(n_sites × n_channels)`` int32 gather
  buffer. Arc B calls only ``sample`` / ``area`` and MUST NOT care which
  backing answers. The ordinal site order + the int32 Q16.16 contract are
  FROZEN here; nothing in this arc builds the kernel or touches
  ``physics_runner``.

The **channel set is frozen** (§3, v2) — including the ``solid`` channel (D6),
so ``area()``'s live non-solid count is derivable from the buffer alone, and
adding a channel to this enumeration is NOT a change to the frozen
shape/dtype/order (only reordering/retyping would be). Per-channel notes:

- ``pressure``/``smoke``/``water_depth``/``o2`` live on AIR tiles.
- ``temperature``/``fire`` live on the SOLID/flammable body tile (D7/D8) — air
  carries no temperature, fire only burns flammable matter.
- ``o2`` is the ``gas[O2]`` **mass-density** channel (D9), a pure gather — NOT
  a partial pressure (no ``p_O2`` field exists; the T-dependence is documented,
  not derived, since deriving it would need a forbidden ×T dequantize).
- ``solid`` is the occupancy/solidity mask (D6) — 0/1, so a currently-non-solid
  tile count is derivable without a second field.

★ **``is_ambient`` is not static** (S8a finding): the site→tile map is static
(sample tiles are authored), but VALUES are always read live each tick — no
per-level field snapshot is cached; nothing combat can mutate is memoized.

No sensor consumes this until B3; the class + the channel enum + the interface
are the B1 deliverable (the seam), per the patch breakdown (§10).
"""
from __future__ import annotations

from enum import Enum


class Channel(Enum):
    """The FROZEN sensor field-channel set (impl doc §3, v2).

    Each member names the gmap field it gathers and its tile family. Ordering
    here defines the resident buffer's channel axis order (frozen); B1 only
    uses the CPU mirror path, but the order is pinned now so the later gather
    kernel is a drop-in. ``clock`` / ``door.is_open`` / ``sensor_motion`` do
    NOT go through this accessor (tick counter / a signal / units).
    """
    PRESSURE = "pressure"        # gmap.atmosphere — materialized P (air)
    SMOKE = "smoke"              # gmap.gas[SMOKE] (air)
    WATER_DEPTH = "water_depth"  # gmap.water_depth (air/floor)
    TEMPERATURE = "temperature"  # gmap.temperature — solid body tile (D7)
    FIRE = "fire"                # gmap.fire — flammable body tile (D8)
    O2 = "o2"                    # gmap.gas[O2] mass density (D9), NOT p_O2
    SOLID = "solid"             # gmap.solid occupancy/solidity mask (D6)


# The frozen channel order (resident-buffer channel axis). A tuple, not just
# the Enum iteration, so a reordering is a deliberate edit reviewers can see.
CHANNEL_ORDER = (
    Channel.PRESSURE, Channel.SMOKE, Channel.WATER_DEPTH, Channel.TEMPERATURE,
    Channel.FIRE, Channel.O2, Channel.SOLID,
)


class SiteIndex:
    """The static site→(tile, channel) map (impl doc §3): ordinal-ordered
    sample sites.

    Built once at load from the FIELD sensors' resolved sample tiles (B3;
    ``clock`` / ``sensor_motion`` do not go through the accessor, §3/§4).
    ``sites`` is a tuple of ``(fy, fx)`` and ``channels`` a parallel tuple of
    :class:`Channel`, both in ordinal (file) order — the frozen site ordering
    the later resident ``(n_sites × n_channels)`` gather buffer will mirror.
    The CPU-mirror path samples with an explicit ``(channel, fy, fx)`` and does
    NOT consult this map; it is the FROZEN seam artifact the resident kernel
    binds to (escalation trigger 1).
    """

    def __init__(self, sites=(), channels=()):
        self.sites = tuple((int(fy), int(fx)) for fy, fx in sites)
        self.channels = tuple(channels)
        if self.channels and len(self.channels) != len(self.sites):
            raise ValueError("SiteIndex sites / channels length mismatch")

    def __len__(self) -> int:
        return len(self.sites)


def build_site_index(sensors=()) -> SiteIndex:
    """The static site index for a FIELD-sensor list, in ordinal order (§3).

    Each sensor contributes its resolved ``sample_tile`` ``(fy, fx)`` and its
    ``channel`` (a :class:`Channel`). Callers pass the field sensors already in
    ordinal order (``clock`` / ``sensor_motion`` are excluded — they never read
    a field). B1 passed an empty list (no sensor classes yet); the signature +
    ordering rule were frozen then and are unchanged.
    """
    sites = []
    channels = []
    for s in sensors:                     # ordinal order (caller pre-sorts)
        tile = getattr(s, "sample_tile", None)
        if tile is None:
            continue
        sites.append((int(tile[0]), int(tile[1])))
        channels.append(getattr(s, "channel", None))
    return SiteIndex(sites, channels)


class EntityFieldAccessor:
    """The ONE seam sensors read the physics fields through (§5a, §3).

    CPU-mirror stub for Arc B: ``sample`` / ``area`` read ``gmap`` arrays
    directly, returning Q16.16 integers with NO dequantize. The resident gather
    backing (later) serves the identical two methods; sensor code is agnostic.
    """

    def __init__(self, gmap, site_index: SiteIndex = None):
        self._gmap = gmap
        self.site_index = site_index if site_index is not None else SiteIndex()

    # -- channel → live gmap array (the CPU mirror; no dequantize) ----------
    def _channel_array(self, channel: Channel):
        g = self._gmap
        if channel is Channel.PRESSURE:
            return g.atmosphere
        if channel is Channel.SMOKE:
            from simulation.gases import SMOKE
            return g.gas[SMOKE]
        if channel is Channel.WATER_DEPTH:
            return g.water_depth
        if channel is Channel.TEMPERATURE:
            return g.temperature
        if channel is Channel.FIRE:
            return g.fire
        if channel is Channel.O2:
            from simulation.gases import O2
            return g.gas[O2]
        if channel is Channel.SOLID:
            return g.solid
        raise KeyError(f"unknown sensor channel {channel!r}")

    def sample(self, channel: Channel, fy: int, fx: int) -> int:
        """One site's Q16.16 value on ``channel`` at tile ``(fy, fx)``.

        Read live off the host mirror — no dequantize (the sim-path rule). The
        ``solid`` channel returns 0/1 (its mask is boolean on the mirror).
        """
        return int(self._channel_array(channel)[int(fy), int(fx)])

    def area(self, channel: Channel, disc_tiles) -> tuple:
        """Integer ``(sum, n)`` over a disc, masked to currently NON-SOLID
        tiles (impl doc §4/§6): the area-mean numerator + its live non-solid
        denominator, so a caller floors ``sum // n`` for the mean.

        ``disc_tiles`` is an iterable of ``(fy, fx)`` (the sensor quantizes its
        radius once at load — the editor §4 rule — so this method never sees
        meters). The non-solid mask legitimately shrinks when a wall in the
        disc is destroyed (physics sensed, not drift, §4). No dequantize; the
        ``solid`` channel (D6) supplies the mask so the buffer alone backs it.
        """
        arr = self._channel_array(channel)
        solid = self._channel_array(Channel.SOLID)
        total = 0
        n = 0
        for (fy, fx) in disc_tiles:
            fy, fx = int(fy), int(fx)
            if int(solid[fy, fx]):
                continue                  # currently-solid tiles excluded (D6)
            total += int(arr[fy, fx])
            n += 1
        return total, n
