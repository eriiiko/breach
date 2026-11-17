"""#29 P7d -- THE HORDE'S FIELD CACHE on the device, and slot 5's LATCH
(residency design 4.4c, the cut table's P7d row; §4.9's refresh rule).

``horde.HordeState`` is cross-tick state the zombie law reads and writes (the
fields flooded since the last drop, the mask they were flooded over, the
stored topology serial): on the device it is a per-env HEADER row, the field
KEYS in the dict's order, one (h, w) int32 DISTANCE PLANE per key and the
three MASK planes (``applier.HORDE_*``; the generated ``APPLIER_HORDE_*``).
:class:`DeviceHordeCache` holds them (CuPy arrays: device state that lives
across ticks) with the HORDE stage's scratch and the latch; the stage reads
and writes them in place (``cpp/src/cuda_unit_horde.cuh``).

- :func:`cache_records` is the cache's ONE record function: the host's
  ``HordeState`` as :class:`CacheRecords` -- a header record and one record
  per field, each with a digest of its plane(s) (the planes themselves ride
  along, so a cache is staged and landed whole). The device's
  :meth:`DeviceHordeCache.records` gives the same form.
- :func:`to_host` lands the device's cache on a sim's ``HordeState`` in place
  (the chained form).
- :func:`latch_of` is the per-env LATCH row the host writes for the stage:
  the tick, the map's topology serial and the zombie request set (its mode,
  then one column per unit row) -- an input, never a host slot in the tick.

CLASS T (training only): nothing in the played tick imports this module.
"""
from __future__ import annotations

import hashlib

import numpy as np

from simulation import applier

#: A cache record: (tag, a, b, c, d, e, digest) -- tag 0 the header
#: ``(mask_built, has_serial, serial, floods, n_fields)``, tag 1 a field
#: ``(kind, fp, sy, sx, slot)``.
REC_W = 7
HDR_TAG, FIELD_TAG = 0, 1
_HDR = {n: i for i, n in enumerate(applier.HORDE_HEADER)}
_KEY = {n: i for i, n in enumerate(applier.HORDE_KEY)}
_KINDS = {n: i for i, n in enumerate(applier.HORDE_FIELD_KINDS)}
_MODE = dict(applier.HORDE_MODES)


def digest(*arrays) -> int:
    """A 56-bit digest of the arrays' bytes (C order): what a record compares
    in place of a whole plane."""
    h = hashlib.blake2b(digest_size=7)
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return int.from_bytes(h.digest(), "little")


def key_of(k: tuple) -> tuple:
    """A ``HordeState.fields`` key as ``(kind, fp, sy, sx)``."""
    if k[0] == "prey":
        kind = "PREY_FEARLESS" if len(k) > 2 and k[2] == "fearless" else "PREY"
        return (_KINDS[kind], int(k[1]), 0, 0)
    if k[0] == "noise":
        return (_KINDS["NOISE"], int(k[1]), int(k[2]), int(k[3]))
    raise ValueError(f"horde cache: no device form of the field key {k!r}")


def host_key(kind: int, fp: int, sy: int, sx: int) -> tuple:
    """The inverse of :func:`key_of`."""
    name = applier.HORDE_FIELD_KINDS[int(kind)]
    if name == "PREY":
        return ("prey", int(fp))
    if name == "PREY_FEARLESS":
        return ("prey", int(fp), "fearless")
    return ("noise", int(fp), int(sy), int(sx))


class CacheRecords:
    """One env's cache as records (a sequence of int tuples, header first)
    with the planes it digests: ``header`` (HDR_W,) int64, ``keys`` (F, 4)
    int64, ``dist`` (F, h, w) int32, ``masks`` (3, h, w) uint8 (zeros while
    not built)."""

    def __init__(self, header, keys, dist, masks):
        self.header = np.asarray(header, dtype=np.int64)
        self.keys = np.asarray(keys, dtype=np.int64).reshape(-1, len(applier.HORDE_KEY))
        self.dist = np.asarray(dist, dtype=np.int32)
        self.masks = np.asarray(masks, dtype=np.uint8)
        h = self.header
        built = int(h[_HDR["mask_built"]])
        rows = [(HDR_TAG, built, int(h[_HDR["has_serial"]]), int(h[_HDR["serial"]]),
                 int(h[_HDR["floods"]]), int(h[_HDR["n_fields"]]),
                 digest(self.masks) if built else 0)]
        for s in range(self.keys.shape[0]):
            k = self.keys[s]
            rows.append((FIELD_TAG, int(k[0]), int(k[1]), int(k[2]), int(k[3]), s,
                         digest(self.dist[s])))
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, i):
        return self._rows[i]

    def __eq__(self, other):
        return list(self) == list(other)

    def __ne__(self, other):
        return not self.__eq__(other)


def cache_records(sim, shape=None) -> CacheRecords:
    """THE record function of the cache: ``sim.horde_state`` as
    :class:`CacheRecords` (``shape`` the map's (h, w); the sim's own by
    default)."""
    st = sim.horde_state
    h, w = shape if shape is not None else sim.gmap.material.shape
    keys = [key_of(k) for k in st.fields]
    dist = (np.stack([np.asarray(st.fields[k], dtype=np.int32) for k in st.fields])
            if st.fields else np.zeros((0, h, w), dtype=np.int32))
    built = st.tile_ok_fearless is not None
    masks = np.zeros((len(applier.HORDE_MASKS), h, w), dtype=np.uint8)
    if built:
        masks[0] = np.asarray(st.tile_ok, dtype=bool)
        masks[1] = np.asarray(st.tile_ok_fearless, dtype=bool)
        if st.bashable is not None:
            masks[2] = np.asarray(st.bashable, dtype=bool)
    hdr = np.zeros(len(applier.HORDE_HEADER), dtype=np.int64)
    hdr[_HDR["mask_built"]] = int(built)
    hdr[_HDR["has_serial"]] = int(st.serial is not None)
    hdr[_HDR["serial"]] = int(st.serial) if st.serial is not None else 0
    hdr[_HDR["floods"]] = int(st.floods)
    hdr[_HDR["n_fields"]] = len(keys)
    return CacheRecords(hdr, keys, dist, masks)


def to_host(sim, recs: CacheRecords) -> None:
    """The device's cache onto ``sim.horde_state`` IN PLACE (the chained
    form's landing): its fields in the device's order, the masks (bashable
    None while ``bash_doors`` is off, as the host keeps it), the serial, the
    floods counter."""
    st = sim.horde_state
    hdr = recs.header
    st.fields = {}
    for s in range(recs.keys.shape[0]):
        k = recs.keys[s]
        st.fields[host_key(*k)] = np.array(recs.dist[s], dtype=np.int32, copy=True)
    if int(hdr[_HDR["mask_built"]]):
        st.tile_ok = recs.masks[0].astype(bool)
        st.tile_ok_fearless = recs.masks[1].astype(bool)
        s = getattr(sim, "horde_settings", None)
        st.bashable = recs.masks[2].astype(bool) if (s is not None and s.bash_doors) else None
    else:
        st.tile_ok = st.tile_ok_fearless = st.bashable = None
    st.serial = int(hdr[_HDR["serial"]]) if int(hdr[_HDR["has_serial"]]) else None
    st.floods = int(hdr[_HDR["floods"]])
    st.settings = getattr(sim, "horde_settings", st.settings)
    st.doors = getattr(sim, "_doors", st.doors)


def latch_of(sim, n_units: int) -> np.ndarray:
    """THE LATCH row of ``sim`` for the HORDE stage at slot 5's entry: the
    tick, ``GameMap._tile_serial``, the request set's mode (CONDUCTOR: the
    conductor drives every zombie; EMPTY: a controller named none -- the slot
    returns; SET) and, per unit row, 1 when the row's id is in the set."""
    row = np.zeros(len(applier.HORDE_LATCH) + int(n_units), dtype=np.int64)
    row[0] = int(sim.tick)
    row[1] = int(getattr(sim.gmap, "_tile_serial", 0))
    reqs = sim._zombie_requests
    if reqs is None:
        row[2] = _MODE["CONDUCTOR"]
    elif not reqs:
        row[2] = _MODE["EMPTY"]
    else:
        row[2] = _MODE["SET"]
        base = len(applier.HORDE_LATCH)
        for r, u in enumerate(sim.units):
            if r < n_units and u.id in reqs:
                row[base + r] = 1
    return row


class DeviceHordeCache:
    """N envs' field caches (device state across ticks), the HORDE stage's
    scratch and its latch buffer, on (h, w) maps of up to ``n_units`` rows
    (CuPy arrays; the stage takes their addresses through :meth:`reach`).
    The planes are allocated for the map's (h, w) on first use (a staged
    cache's, or the stage's planes: :meth:`ensure`). ``persistent``: the
    cache is never reset after its first :meth:`begin` (the CACHE gate's
    device state across ticks, never restaged from the host)."""

    def __init__(self, n_env: int, n_units: int, fcap: int = applier.HORDE_FIELD_CAP,
                 h: int | None = None, w: int | None = None, persistent: bool = False):
        self.n_env, self.n_units = int(n_env), int(n_units)
        self.fcap = int(fcap)
        if self.n_env < 1 or self.fcap < 1 or self.n_units < 1:
            raise ValueError("horde cache: n_env, fcap and n_units must be >= 1")
        self.persistent = bool(persistent)
        self._begun = False
        self.h = self.w = None
        if h is not None:
            self.ensure(h, w)

    def ensure(self, h: int, w: int) -> None:
        """The planes for (h, w) maps (allocated once; another shape refused)."""
        import cupy as cp
        if self.h is not None:
            if (self.h, self.w) != (int(h), int(w)):
                raise ValueError(f"horde cache: allocated for {(self.h, self.w)} maps, not "
                                 f"{(int(h), int(w))}")
            return
        self.h, self.w = int(h), int(w)
        N, n, U = self.n_env, self.h * self.w, self.n_units
        self.hdr = cp.zeros((N, len(applier.HORDE_HEADER)), dtype=cp.int64)
        self.keys = cp.zeros((N, self.fcap, len(applier.HORDE_KEY)), dtype=cp.int64)
        self.dist = cp.zeros((N, self.fcap, self.h, self.w), dtype=cp.int32)
        self.masks = cp.zeros((N, len(applier.HORDE_MASKS), self.h, self.w), dtype=cp.uint8)
        self.pass_ = cp.zeros((N, n), dtype=cp.uint8)
        self.flags = cp.zeros((N, 2, U), dtype=cp.uint8)
        self.queues = cp.zeros((N, 2, n), dtype=cp.int32)
        self.live = cp.zeros((N, U, 3), dtype=cp.int64)
        self.latch = cp.zeros((N, len(applier.HORDE_LATCH) + U), dtype=cp.int64)

    @property
    def shape(self):
        return (self.n_env,)

    def begin(self) -> None:
        """Every env's cache fresh (``HordeState()``: no serial, no field); a
        persistent cache only the first time."""
        if self.persistent and self._begun:
            return
        self._begun = True
        if self.h is not None:
            self.hdr[...] = 0
            self.keys[...] = 0

    def stage_lists(self, per_env) -> None:
        """Set every env's cache from its host :class:`CacheRecords`."""
        import cupy as cp
        if len(per_env) != self.n_env:
            raise ValueError(f"horde cache: {len(per_env)} envs staged, it holds {self.n_env}")
        self.ensure(*per_env[0].masks.shape[1:])
        hdr = np.zeros(self.hdr.shape, dtype=np.int64)
        keys = np.zeros(self.keys.shape, dtype=np.int64)
        dist = np.zeros(self.dist.shape, dtype=np.int32)
        masks = np.zeros(self.masks.shape, dtype=np.uint8)
        for e, rc in enumerate(per_env):
            f = rc.keys.shape[0]
            if f > self.fcap:
                raise applier.HordeCacheOverflow(
                    f"horde cache: env {e} holds {f} fields past its capacity {self.fcap}")
            if rc.masks.shape[1:] != (self.h, self.w):
                raise ValueError(f"horde cache: env {e}'s map is {rc.masks.shape[1:]}, the "
                                 f"cache's {(self.h, self.w)}")
            hdr[e] = rc.header
            keys[e, :f] = rc.keys
            dist[e, :f] = rc.dist
            masks[e] = rc.masks
        self.hdr[...] = cp.asarray(hdr)
        self.keys[...] = cp.asarray(keys)
        self.dist[...] = cp.asarray(dist)
        self.masks[...] = cp.asarray(masks)

    def set_latch(self, rows) -> None:
        """The per-env latch rows (:func:`latch_of`), padded to the cache's U
        (the planes must be allocated: :meth:`ensure`)."""
        import cupy as cp
        a = np.zeros(self.latch.shape, dtype=np.int64)
        for e, r in enumerate(rows):
            r = np.asarray(r, dtype=np.int64)
            a[e, :min(len(r), a.shape[1])] = r[:a.shape[1]]
        self.latch[...] = cp.asarray(a)

    def records(self) -> list:
        """Per env, its cache as :class:`CacheRecords` (downloaded)."""
        if self.h is None:
            raise RuntimeError("horde cache: never allocated (no map seen)")
        hdr = self.hdr.get()
        keys = self.keys.get()
        dist = self.dist.get()
        masks = self.masks.get()
        out = []
        for e in range(self.n_env):
            f = int(hdr[e, _HDR["n_fields"]])
            built = int(hdr[e, _HDR["mask_built"]])
            m = masks[e] if built else np.zeros_like(masks[e])
            out.append(CacheRecords(hdr[e], keys[e, :f], dist[e, :f], m))
        return out

    def check(self, what: str = "") -> None:
        pass

    def reach(self) -> np.ndarray:
        """The ten uint64 the stage takes (``applier_run(horde=)``)."""
        return np.array([int(a.data.ptr) for a in (
            self.hdr, self.keys, self.dist, self.masks, self.pass_, self.flags,
            self.queues, self.live, self.latch)] + [self.fcap], dtype=np.uint64)


__all__ = ["REC_W", "digest", "key_of", "host_key", "CacheRecords", "cache_records",
           "to_host", "latch_of", "DeviceHordeCache"]
