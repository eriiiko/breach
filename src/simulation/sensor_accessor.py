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

**The observation crop (#29 P1a, residency design §4.5).** ``crop`` is the
accessor's third method: a per-unit ``(U, C, 2r+1, 2r+1)`` float32 window of
the frozen channels (any subset, in the caller's order) plus the crop-only
planes :data:`CROP_EXTRA_CHANNELS` (``material`` and unit occupancy by team,
relative to the observing unit). It is a READ-SIDE product like a render read:
each plane is dequantized ONCE at the read through its Q16 boundary module
(:func:`_crop_plane`), never written back, never inside the sim path. This is
the CPU backing; #25's gather kernel is its device backing (P8b). ``light``
is refused until #81 separates light's inputs (ACCEPTED GAP).

**The observation door (#195 1g).** ``observe(spec, units, out, **kw)`` is
THE one way an observation layout is served: it dispatches by spec name
(:data:`OBS_SPECS`) into the caller's ``(U, ...)`` views -- ``"crop"`` the
placeholder rows above (the crop plus the run file's vector fields),
``"v1"`` ``simulation.obs_spec_v1`` (the spec as code: the soldier's vector,
the list). The env calls only this; a new layout is a new spec name here,
never an edit of one a trained network uses.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

#: The observation specs ``observe`` serves, by name (``[obs] spec``).
OBS_SPECS = ("crop", "v1")
#: The value network's full view (#195 1g (c), obs spec v1 section 5):
#: ``observe`` serves it beside ``"v1"`` into ``info["full_view"]``'s
#: buffers; it is NOT in :data:`OBS_SPECS`, so no run file can name it as
#: the policy's observation.
FULL_VIEW_SPEC = "full_v1"


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

#: The crop's planes beyond the frozen field channels (#29 P1a): the tile
#: material id, and occupancy by team RELATIVE TO THE OBSERVING UNIT -- 1.0
#: on every footprint tile of a living unit of its own team (itself
#: included), of another marine team, or of the horde; 0.0 elsewhere.
CROP_EXTRA_CHANNELS = ("material", "units_own", "units_enemy",
                       "units_zombie")
#: Planes the crop refuses by name, with the reason.
CROP_REFUSED_CHANNELS = {
    "light": "light is not sim-pure until #81 (Light as Game State) "
             "separates its inputs -- ACCEPTED GAP: no light channel in "
             "observations",
}
_UNIT_PLANES = ("units_own", "units_enemy", "units_zombie")


def crop_channel_names() -> tuple:
    """Every plane name ``crop`` accepts: the frozen channels in
    ``CHANNEL_ORDER``, then :data:`CROP_EXTRA_CHANNELS`."""
    return tuple(c.value for c in CHANNEL_ORDER) + CROP_EXTRA_CHANNELS


def resolve_crop_channels(channels) -> tuple:
    """Plane names (``Channel`` members or their strings) -> a tuple of
    names, refusing an unknown, a refused (``light``) or a repeated one by
    name."""
    known = crop_channel_names()
    out = []
    for c in channels:
        name = c.value if isinstance(c, Channel) else c
        if not isinstance(name, str):
            raise ValueError(f"crop channel {c!r} is not a name")
        if name in CROP_REFUSED_CHANNELS:
            raise ValueError(f"crop channel {name!r} is refused: "
                             f"{CROP_REFUSED_CHANNELS[name]}")
        if name not in known:
            raise ValueError(f"unknown crop channel {name!r} (known: "
                             f"{list(known)})")
        if name in out:
            raise ValueError(f"crop channel {name!r} named twice")
        out.append(name)
    return tuple(out)


def _crop_plane(gmap, name: str):
    """One field plane as float32 real units, dequantized ONCE through its
    Q16 boundary module (the read-side bridge; nothing is written back)."""
    from simulation import (atmosphere_fixed, fire_fixed, gas_fixed,
                            water_fixed)
    from simulation.gases import O2, SMOKE
    if name == "pressure":
        return atmosphere_fixed.dequantize_f32(gmap.atmosphere)
    if name == "smoke":
        # the trace currency's own helper (#88), never a bulk name
        return gas_fixed.dequantize_trace_f32(gmap.gas[SMOKE])
    if name == "water_depth":
        return water_fixed.dequantize_f32(gmap.water_depth)
    if name == "temperature":
        return gas_fixed.dequantize_q16(gmap.temperature).astype(np.float32)
    if name == "fire":
        return fire_fixed.dequantize_f32(gmap.fire)
    if name == "o2":
        return gas_fixed.dequantize_bulk_f32(gmap.gas[O2])
    if name == "solid":
        return np.asarray(gmap.solid, dtype=np.float32)
    if name == "material":
        return np.asarray(gmap.material, dtype=np.float32)
    raise KeyError(f"no crop plane {name!r}")


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
            # The smoke plane is in the TRACE currency (#88); the sensor signal
            # contract and every authored KIND_Q16 threshold are Q16, so the
            # channel converts HERE, once, where both `sample` and `area` see
            # it (critique A7). Floor: invariant on S_old << TRACE_FINE_BITS.
            from simulation import gas_fixed
            from simulation.gases import SMOKE
            return gas_fixed.trace_to_q16(g.gas[SMOKE])
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

    def sample_many(self, channel: Channel, fys, fxs) -> np.ndarray:
        """:meth:`sample` at many tiles at once (#183 G8p, the bots' position
        features): an int64 array of the Q16.16 values at ``(fys[i],
        fxs[i])``, the channel converted once per call. No dequantize."""
        arr = self._channel_array(channel)
        return np.asarray(arr[np.asarray(fys, dtype=np.int64),
                              np.asarray(fxs, dtype=np.int64)],
                          dtype=np.int64)

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

    def observe(self, spec: str, units, out: dict, **kw) -> None:
        """THE observation door (#195 1g): write the layout ``spec`` names
        for ``units`` (one unit or ``None`` per row) into ``out``'s
        ``(U, ...)`` views. ``"crop"``: ``out["crop"]`` through :meth:`crop`
        (``channels``, ``r``, ``all_units``) and ``out["vector"]`` through
        the callables ``vector_fields`` (the run file's placeholder rows);
        ``"v1"``: ``simulation.obs_spec_v1.observe_rows`` (``sim``,
        ``context``) into ``vector`` / ``list`` / ``list_mask`` /
        ``list_uid``. The row flags are the env's. An unknown spec is
        refused by name."""
        if spec == "crop":
            out["crop"][...] = self.crop(units, kw["channels"], kw["r"],
                                         all_units=kw.get("all_units"))
            vec = out.get("vector")
            if vec is not None:
                vec[...] = 0.0
                fns = tuple(kw.get("vector_fields", ()))
                for i, u in enumerate(units):
                    if u is not None:
                        for j, f in enumerate(fns):
                            vec[i, j] = f(u)
            return
        if spec == "v1":
            from simulation.obs_spec_v1 import observe_rows
            observe_rows(kw["sim"], kw["context"], units, out)
            return
        if spec == FULL_VIEW_SPEC:
            # the value network's full view (spec 5): served beside "v1" by
            # this door into the farm's OWN buffers (info["full_view"]),
            # never a run-file spec -- ``units`` are the rows' ROSTER units
            from simulation.obs_spec_v1 import observe_full_rows
            observe_full_rows(kw["sim"], kw["context"], units, out)
            return
        raise ValueError(f"unknown observation spec {spec!r} (known: "
                         f"{list(OBS_SPECS)} and the full view "
                         f"{FULL_VIEW_SPEC!r})")

    def crop(self, units, channels, r: int, *, all_units=None):
        """The observation crop (#29 P1a): ``(U, C, 2r+1, 2r+1)`` float32,
        row ``i`` the window centred on ``units[i]``'s centre tile
        (``center_tile_y/x``), plane ``c`` the ``channels[c]`` (a frozen
        channel or a :data:`CROP_EXTRA_CHANNELS` name; see
        :func:`resolve_crop_channels`). A ``None`` unit is an all-zero row;
        cells outside the grid are zero. The unit planes read every living
        unit of ``all_units`` (default: the observed ``units``) by its
        footprint (``offsets`` from ``tile_x/y``), relative to each observer's
        team. Each field plane is dequantized once per call, at the read."""
        names = resolve_crop_channels(channels)
        r = int(r)
        if r < 0:
            raise ValueError(f"crop radius must be >= 0, got {r}")
        k = 2 * r + 1
        units = list(units)
        out = np.zeros((len(units), len(names), k, k), dtype=np.float32)
        if not units or not names:
            return out
        g = self._gmap
        h, w = g.material.shape
        planes = {n: _crop_plane(g, n) for n in names
                  if n not in _UNIT_PLANES}
        team_at = zombie_at = None
        if any(n in _UNIT_PLANES for n in names):
            # team id per tile (-1 empty) and the horde mask, living units only
            team_at = np.full((h, w), -1, dtype=np.int32)
            zombie_at = np.zeros((h, w), dtype=bool)
            if all_units is None:
                pop = [u for u in units if u is not None]
            else:
                pop = list(all_units)
            for u in pop:
                if u is None or not u.alive:
                    continue
                x0, y0 = u.tile_x, u.tile_y
                for dx, dy in u.offsets:
                    x, y = x0 + int(dx), y0 + int(dy)
                    if 0 <= x < w and 0 <= y < h:
                        if u.is_zombie:
                            zombie_at[y, x] = True
                        else:
                            team_at[y, x] = int(u.team)
        for i, u in enumerate(units):
            if u is None:
                continue
            cy, cx = u.center_tile_y(), u.center_tile_x()
            y0, x0 = cy - r, cx - r
            sy0, sy1 = max(y0, 0), min(y0 + k, h)
            sx0, sx1 = max(x0, 0), min(x0 + k, w)
            if sy0 >= sy1 or sx0 >= sx1:
                continue
            dy0, dy1 = sy0 - y0, sy1 - y0
            dx0, dx1 = sx0 - x0, sx1 - x0
            for c, name in enumerate(names):
                dst = out[i, c, dy0:dy1, dx0:dx1]
                if name in planes:
                    dst[...] = planes[name][sy0:sy1, sx0:sx1]
                    continue
                tw = team_at[sy0:sy1, sx0:sx1]
                if name == "units_own":
                    dst[...] = tw == int(u.team)
                elif name == "units_enemy":
                    dst[...] = (tw >= 0) & (tw != int(u.team))
                else:
                    dst[...] = zombie_at[sy0:sy1, sx0:sx1]
        return out
