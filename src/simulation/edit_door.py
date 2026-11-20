"""THE DEVICE EDIT DOOR, step (a) (#29 P7a.3;
``docs/residency_batching_design_2026-11-07.md`` §4.4c decision 7, §4.8).

Where a DEVICE producer -- an effect function of the per-env serial applier
(``simulation.applier``; SPRAY_CELLS first, P7c's payloads next) -- hands its
FieldEdits: as the rows the device edit queue reads
(``field_edit.pack_for_device``'s EDIT and TILE rows, ``cpp/src/cuda_resident.h``
``EDQ_*``), written by device forms of the door's quantization and of the
region walk for the regions the producer uses (``cpp/src/cuda_edit_door.cuh``:
TILE today). Per env (``cpp/src/cuda_edit_door.h``'s ``EditDoor``):

- the EDIT rows: pack_for_device's 16 columns, then the edit's sort key's
  ``source_id`` and ``region`` (:data:`SRC_COL`, :data:`REGION_COL`), which
  the packed row does not carry and step (c)'s per-env sort ``(field, source,
  region, seq)`` needs;
- the TILE rows, their edit index LOCAL to the env's own list;
- ``seq``: the per-env edit counter (the queue's ``seq``, the noise key). It
  RESETS AT THE 6b FLUSH (:meth:`DeviceEditDoor.begin`), never at the tick
  end, and is set by :meth:`DeviceEditDoor.set_seq` when its base is a harness
  input (the host window's ``EditQueue._seq`` at the walk's start).

Until step (c) the rows stay in ENQUEUE order and the played edit queue
(Class P) is untouched. Two rules (decision 7):

- GATED AT THE DOOR: a producer's rows equal :func:`door_rows` of the edits
  the host producer enqueued (``pack_for_device`` of them, ``seq`` absolute
  when the base is a harness input, the edit indices rebased to the
  producer's first edit). A host edit of ANOTHER producer inside the same
  walk (an interleave) is refused by name (:class:`DoorInterleave`), never
  compared loosely.
- A CHAINED run lands the device's rows through the host's own enqueue
  (:func:`replay`, the request list's ``apply_to_host`` pattern), which must
  reproduce them.

NOT PART OF THE PLAYED TICK: Class T (training only). Import-light: numpy,
``field_edit``; ``breach_physics`` only inside :class:`DeviceEditDoor`.
"""
from __future__ import annotations

import numpy as np

from simulation import field_edit as _fe

#: The device door's EDIT row: the packed columns, then the sort key's two,
#: then (#29 P7c.1) the region's four arguments (:data:`ARG_COLS`): a TILE
#: ``(r, c, 0, 0)``, a DISC ``(cy, cx, r_q, falloff)`` -- the radius as the
#: region walk quantizes it --, so the chained landing rebuilds the very edit
#: (its TILE rows alone cannot name a clipped disc's centre).
PACKED_COLS = _fe.EDIT_COLS
SRC_COL = PACKED_COLS
REGION_COL = PACKED_COLS + 1
ARG_COL = PACKED_COLS + 2
ARG_COLS = 4
EDIT_COLS = PACKED_COLS + 2 + ARG_COLS
TILE_COLS = _fe.TILE_COLS
#: Edits per env per window: STORAGE (a sprayed cell enqueues at most two).
EDIT_CAP = 8192
TILE_CAP = 8192


#: The keyed draws the host pack is handed for a noisy edit's rows (the EDIT
#: and TILE rows do not depend on them: only the flush draws).
from simulation.draws import MatchDraws as _MatchDraws  # noqa: E402
_ROWS_DRAWS = _MatchDraws(0)


class DoorInterleave(RuntimeError):
    """A host producer's edit inside a walk whose device producer's rows are
    gated (decision 7: refused by name, never compared loosely)."""


class DoorOverflow(RuntimeError):
    """More edit or tile rows than the door's capacity (named: list, env)."""


def region_args(edit) -> tuple:
    """#29 P7c.1: the region's four arguments in the door's EDIT row (see
    :data:`ARG_COL`); a region the device door does not write is refused."""
    if edit.region == _fe.Region.TILE:
        return (int(edit.coords[0]), int(edit.coords[1]), 0, 0)
    if edit.region == _fe.Region.DISC:
        return (int(edit.coords[0]), int(edit.coords[1]),
                int(_fe._ufx.quantize_scalar(float(edit.coords[2]))),  # p3e-door: _iter_region's
                int(edit.falloff))
    if edit.region == _fe.Region.TILES:
        # #29 P7c.2: the fixed-total deposit -- its tile count and whether it
        # is scaled (the TILE rows carry the tiles, weights and factors)
        return (len(edit.coords), 0 if edit.scale is None else 1, 0, 0)
    raise DoorInterleave(f"edit door: a {_fe.Region(edit.region).name} edit has no "
                         f"device form at the door (TILE, DISC and TILES do)")


def door_rows(gmap, edits_seq, owned_sources=None):
    """THE HOST REFERENCE of a producer's door rows: ``edits_seq`` the
    ``(FieldEdit, seq)`` pairs the host producer enqueued, in ENQUEUE order;
    returns ``(edits (E, EDIT_COLS), tiles (T, TILE_COLS))`` int64 --
    ``pack_for_device`` of them (its own door and region walk), the sort
    key's ``source_id`` / ``region`` appended. ``owned_sources``: the producer's
    source ids; an edit of another source is an interleave, refused by name.
    A pack the device could not serve (its fallback reason) is refused too."""
    pairs = list(edits_seq)
    if owned_sources is not None:
        for e, _s in pairs:
            if int(e.source_id) not in owned_sources:
                raise DoorInterleave(
                    f"edit door: an edit of source {int(e.source_id)} ({e.field!r}) "
                    f"inside the walk of the producers {sorted(owned_sources)}: "
                    f"host and device producers interleave (decision 7 refuses "
                    f"such a tick)")
    # #29 P7c.2: a noisy edit (the blast smoke) packs only beside the keyed
    # draws; the rows never read them (the draw is the flush's), so any seed
    pack, reason = _fe.pack_for_device(gmap, pairs, _ROWS_DRAWS)
    if pack is None:
        raise DoorInterleave(f"edit door: the host pack refuses the tick ({reason})")
    edits = np.zeros((pack.edits.shape[0], EDIT_COLS), dtype=np.int64)
    edits[:, :PACKED_COLS] = pack.edits
    for k, (e, _s) in enumerate(pairs):
        edits[k, SRC_COL] = int(e.source_id)
        edits[k, REGION_COL] = int(e.region)
        edits[k, ARG_COL:ARG_COL + ARG_COLS] = region_args(e)
    return edits, np.asarray(pack.tiles, dtype=np.int64).reshape(-1, TILE_COLS)


_FIELD_OF_ORDINAL = {k: n for n, k in _fe.DEVICE_FIELD_ORDINALS.items()}


def _clamp_row(pol):
    """The (flags, lo, hi) the door writes for the policy's own clamp."""
    pair = _fe._clamp_q(pol.clamp, _fe._CURRENCY[pol.dtype])
    flags = lo = hi = 0
    if pair is not None:
        if pair[0] is not None:
            flags, lo = flags | 1, int(pair[0])
        if pair[1] is not None:
            flags, hi = flags | 2, int(pair[1])
    return flags, lo, hi


def _unscaled_den(r) -> tuple:
    """#29 P7c.2: an EDIT row's denominator less its noise's and its scale's
    ONE factors, and whether it is scaled: ``(den, scaled)``. A DISC's base
    is ONE, so its scale is what is left; a TILES row names it (arg 1)."""
    one = int(_fe._ufx.FP_ONE)
    den = r[_fe.E_DEN]
    if r[_fe.E_NOISY]:
        if den % one:
            raise DoorInterleave(f"edit door: a noisy device row {r} whose den is not a "
                                 f"multiple of ONE")
        den //= one
    if r[REGION_COL] == int(_fe.Region.TILES):
        scaled = bool(r[ARG_COL + 1])
    else:
        scaled = den == one * one
    if scaled:
        if den % one:
            raise DoorInterleave(f"edit door: a scaled device row {r} whose den is not a "
                                 f"multiple of ONE")
        den //= one
    return den, scaled


def edit_of(gmap, row, tiles) -> tuple:
    """A device EDIT row (and its TILE rows) as the host FieldEdit it stands
    for and its seq: the inverse of :func:`door_rows` for the forms the device
    door writes (a TILE or -- #29 P7c.1 -- a DISC edit, a clamp equal to its
    field policy's: the replayed edit carries none, which the door quantizes
    to the same row; #29 P7c.2 -- a TILES edit, and a DISC or TILES edit with
    its noise (rebuilt from the row's low bound) and its scale (a plane of ONE
    with each TILE row's factor at its tile)). Anything else is refused by
    name."""
    r = [int(x) for x in row]
    field = _FIELD_OF_ORDINAL.get(r[_fe.E_FIELD])
    region = r[REGION_COL]
    one = int(_fe._ufx.FP_ONE)
    if field is None or region not in (int(_fe.Region.TILE), int(_fe.Region.DISC),
                                       int(_fe.Region.TILES)):
        raise DoorInterleave(f"edit door: a device row {r} the door does not write")
    den, scaled = _unscaled_den(r)
    if region != int(_fe.Region.TILES) and den != one:
        raise DoorInterleave(f"edit door: a device row {r} the door does not write")
    if region == int(_fe.Region.TILE) and (r[_fe.E_NOISY] or scaled):
        raise DoorInterleave(f"edit door: a noisy or scaled TILE row {r} (the door "
                             f"writes none)")
    pol = _fe._policy(field)
    if (r[_fe.E_CLAMP_FLAGS], r[_fe.E_LO], r[_fe.E_HI]) != _clamp_row(pol) \
            or any(r[_fe.E_O2_LO:_fe.E_N2_HI + 1]):
        raise DoorInterleave(f"edit door: a device row {r} whose clamp is not its "
                             f"field's policy clamp (not replayed)")
    t = [tuple(int(x) for x in tr) for tr in tiles]
    w = int(gmap._w)
    a = r[ARG_COL:ARG_COL + ARG_COLS]
    scale = None
    if scaled:
        scale = np.full((int(gmap._h), w), one, dtype=np.int64)
        for tr in t:
            y, x = divmod(tr[1], w)
            scale[y, x] = tr[3]
    weight_den = 0
    if region == int(_fe.Region.DISC):
        coords = (a[0], a[1], _fe._ufx.dequantize_scalar(a[2]))
        falloff = _fe.Falloff(a[3])
    elif region == int(_fe.Region.TILES):
        if len(t) != a[0]:
            raise DoorInterleave(f"edit door: a TILES edit naming {a[0]} tiles with "
                                 f"{len(t)} tile rows")
        coords = tuple((tr[1] // w, tr[1] % w, tr[2]) for tr in t)
        falloff, weight_den = _fe.Falloff.FLAT, den
    else:
        if len(t) != 1:
            raise DoorInterleave(f"edit door: a TILE edit with {len(t)} tile rows "
                                 f"(a tile outside the grid is not replayed)")
        y, x = divmod(t[0][1], w)
        coords, falloff = (y, x), _fe.Falloff.FLAT
    a_q = r[_fe.E_AMOUNT]
    if pol.dtype == "gas":
        from simulation import gas_fixed
        amount = float(gas_fixed.dequantize_trace(a_q))
    else:
        amount = _fe._ufx.dequantize_scalar(a_q)
    noise = 0.0
    if r[_fe.E_NOISY]:
        noise = _fe._ufx.dequantize_scalar(one - r[_fe.E_LOW])   # p3e-view: re-quantized exactly
    edit = _fe.FieldEdit(field=field, region=_fe.Region(region), coords=coords,
                         falloff=falloff,
                         amount=amount, amount_q=a_q, mode=_fe.EditMode(r[_fe.E_MODE]),
                         channel=(r[_fe.E_SLICE] if field == "gas" else None),
                         source_id=r[SRC_COL], noise=noise, scale=scale,
                         weight_den=weight_den)
    return edit, r[_fe.E_KEY]


def replay(sim, edits, tiles) -> None:
    """The chained form's landing: the device's rows, in order, through the
    host's own ``EditQueue.enqueue`` -- each must land at the seq its row was
    keyed with, and :func:`door_rows` of the replayed edits must reproduce the
    rows (refused by name otherwise)."""
    q = sim.edit_queue
    edits = np.asarray(edits, dtype=np.int64).reshape(-1, EDIT_COLS)
    tiles = np.asarray(tiles, dtype=np.int64).reshape(-1, TILE_COLS)
    pairs = []
    for k in range(edits.shape[0]):
        e, key = edit_of(sim.gmap, edits[k], tiles[tiles[:, 0] == k])
        if q._seq != key:
            raise DoorInterleave(f"edit door: a replayed edit keyed {key} would land "
                                 f"at the queue's seq {q._seq}")
        q.enqueue(e)
        pairs.append((e, key))
    if pairs:
        e2, t2 = door_rows(sim.gmap, pairs)
        if not (np.array_equal(e2, edits) and np.array_equal(t2, tiles)):
            raise DoorInterleave("edit door: the replayed edits do not reproduce the "
                                 "device's rows")


class DeviceEditDoor:
    """THE device edit door of ``n_env`` envs (``breach_physics.EditDoorBuffer``:
    two record lists and the seq counters; the CUDA build only). Refuses a
    compiled module whose rows are not this module's."""

    def __init__(self, n_env: int, edit_cap: int = EDIT_CAP, tile_cap: int = TILE_CAP,
                 bp=None):
        if bp is None:
            import breach_physics as bp
        if not getattr(bp, "HAS_CUDA", False):
            raise RuntimeError("edit door: the device door needs the CUDA build")
        lay = bp.edit_door_layout()
        if (int(lay["edit_cols"]), int(lay["packed_cols"]), int(lay["tile_cols"]),
                int(lay["src_col"]), int(lay["region_col"]),
                int(lay.get("arg_col", -1))) != (
                EDIT_COLS, PACKED_COLS, TILE_COLS, SRC_COL, REGION_COL, ARG_COL):
            raise ValueError(f"edit door: the compiled rows {dict(lay)} are not "
                             f"simulation.edit_door's")
        self.bp = bp
        self.n_env = int(n_env)
        self.edit_cap, self.tile_cap = int(edit_cap), int(tile_cap)
        self._buf = bp.EditDoorBuffer(self.n_env, self.edit_cap, self.tile_cap)

    @property
    def buffer(self):
        return self._buf

    def begin(self) -> None:
        """The 6b flush's reset: every env's rows empty, its seq 0."""
        self._buf.begin()

    def set_seq(self, seq) -> None:
        """Every env's seq (``(N,)``): the window's base, a harness input."""
        self._buf.set_seq(np.ascontiguousarray(seq, dtype=np.int64).reshape(self.n_env))

    def seq(self) -> np.ndarray:
        return np.asarray(self._buf.seq(), dtype=np.int64)

    def rows(self):
        """Per env ``(edits (E, EDIT_COLS), tiles (T, TILE_COLS))`` int64."""
        er, ec, _ = self._buf.edits.download()
        tr, tc, _ = self._buf.tiles.download()
        return [(er[e, :min(int(ec[e]), self.edit_cap)].copy(),
                 tr[e, :min(int(tc[e]), self.tile_cap)].copy()) for e in range(self.n_env)]

    def check(self, what: str = "") -> None:
        """Refuse by name an env whose EDIT or TILE rows overflowed."""
        for name, lst, cap in (("EDIT rows", self._buf.edits, self.edit_cap),
                               ("TILE rows", self._buf.tiles, self.tile_cap)):
            _c, drp = lst.counts()
            bad = [e for e in range(self.n_env) if int(drp[e])]
            if bad:
                raise DoorOverflow(f"the edit door's {name}{' ' + what if what else ''}: "
                                   f"env {bad[0]} dropped {int(drp[bad[0]])} past its "
                                   f"capacity {cap}")


    def reset_env(self, e: int) -> None:
        """Env ``e``'s reset alone (an env reset inside a batch): its rows
        empty, its seq 0; the other envs' windows untouched."""
        self._buf.reset_env(int(e))

    def append_probe(self, e: int, edits, tiles) -> None:
        """TEST-ONLY (#29 P7e.3): host rows (``door_rows``' form, the tiles'
        edit index local to ``edits``) appended to env ``e`` as a device
        producer writes them: each row's key must continue the env's seq
        (refused by name otherwise, nothing written)."""
        ok = self.bp.edit_door_append_probe(
            self._buf, int(e), np.ascontiguousarray(edits, dtype=np.int64).reshape(-1, EDIT_COLS),
            np.ascontiguousarray(tiles, dtype=np.int64).reshape(-1, TILE_COLS))
        if not ok:
            raise DoorInterleave(f"edit door: rows appended to env {int(e)} do not continue "
                                 f"its seq or do not fit its capacity")


# ---------------------------------------------------------------------------
# #29 P7e.3 -- STEP (c): slot 6b's flush FROM THE DEVICE DOOR
# ---------------------------------------------------------------------------
#: The build's refusals, by code (``cpp/src/cuda_edit_flush.h``
#: ``EditFlushFault``): each is a row ``field_edit.pack_for_device`` refuses,
#: raises on or falls back on -- inside a device tick there is no host to
#: fall back to, so the env is refused by name. The pack's "noisy edits
#: without MatchDraws" has no device form: the tick row always carries the
#: sim's MatchDraws key (``unit_draws.tick_row`` reads ``sim.draws``), so a
#: sim without them is refused at the latch, never at the flush.
FLUSH_FAULTS = {
    1: ("door_overflow", "the door dropped rows past its capacity (DoorOverflow)"),
    2: ("field", "a field ordinal with no device policy"),
    3: ("water_depth", "a water_depth edit (the host pack's fallback: "
                       "field_edit.DEVICE_REFUSED_FIELDS)"),
    4: ("gas_slice", "a gas edit on a slice outside [0, G) (the host pack's fallback: "
                     "numpy's index wrap)"),
    5: ("conservative", "a CONSERVATIVE slice through the gas policy (the host raises)"),
    6: ("mode", "an EditMode outside ADD / REMOVE / MAX"),
    7: ("den", "a denominator outside (0, 2^62) (the host pack's fallback)"),
    8: ("noise_low", "a noisy row's low bound outside [0, ONE]"),
    9: ("key", "keys that are not the window's consecutive seqs (a lost or foreign row)"),
    10: ("tile", "a tile outside its env's edits or the grid"),
    11: ("product", "an integer product |amount x weight x mult x scale| >= 2^125 (the "
                    "host pack's fallback)"),
    12: ("contrib", "a contribution >= 2^55 (the host pack's fallback)"),
}

#: The planes the flush reads and writes, in the binding's order (``None``
#: is_ambient: a space map).
FLUSH_PLANES = (("gas", np.int32), ("temperature", np.int32), ("gas_energy", np.int64),
                ("fire", np.int32), ("heat", np.int32), ("solid", np.bool_),
                ("is_vacuum", np.bool_), ("is_ambient", np.bool_),
                ("thermal_solid", np.bool_), ("flammable", np.bool_))


class EditFlushRefused(RuntimeError):
    """An env whose window the device flush refuses (:data:`FLUSH_FAULTS`):
    nothing of that env was applied. ``env``, ``code``, ``name``, ``row``."""

    def __init__(self, msg, env, code, row):
        super().__init__(msg)
        self.env, self.code, self.row = int(env), int(code), int(row)
        self.name = FLUSH_FAULTS.get(self.code, ("?", ""))[0]


class EditQueueFailed(RuntimeError):
    """K2's failure slot (``cuda_edit_queue.cu``): an edit result left its
    field's range (an int32 store, an int64 energy, a negative trace) -- the
    host flush raises there."""


class DeviceEditFlush:
    """THE DEVICE FLUSH of slot 6b (#29 P7e.3, ``breach_physics.
    EditFlushBuffer``): ``door``'s window -- every row a device producer wrote
    since the last flush, across the tick boundary -- sorted per env into the
    host's applied order ``(field, source, region, seq)``, range-checked as
    ``pack_for_device`` checks it, laid out as ``_edit_buffer`` lays it out
    (tick word and key from THE TICK ROW, the ambient from the tables) and fed
    to the device edit queue's K1 / K2, then the door reset (rows empty, seq
    0): all on the device, nothing downloaded until :meth:`books`, THE ONE
    READ (the books and the refusals together)."""

    def __init__(self, door: DeviceEditDoor, bp=None):
        bp = door.bp if bp is None else bp
        lay = dict(bp.edit_flush_layout())
        names = {int(lay[name]): name for _c, (name, _w) in FLUSH_FAULTS.items()}
        if names != {c: n for c, (n, _w) in FLUSH_FAULTS.items()} \
                or int(lay["n_codes"]) != len(FLUSH_FAULTS) + 1:
            raise ValueError(f"edit flush: the compiled fault codes {lay} are not "
                             f"simulation.edit_door's")
        self.bp, self.door = bp, door
        self.n_env = door.n_env
        self.slots, self.fail_slot = int(lay["slots"]), int(lay["fail"])
        self._buf = bp.EditFlushBuffer(self.n_env, door.edit_cap, door.tile_cap)

    def _ptrs(self, planes: dict):
        out = np.zeros(len(FLUSH_PLANES), dtype=np.uint64)
        for k, (name, dt) in enumerate(FLUSH_PLANES):
            a = planes.get(name)
            if a is None:
                if name == "is_ambient":
                    continue
                raise ValueError(f"edit flush: plane {name!r} missing")
            if np.dtype(a.dtype) != np.dtype(dt) or int(a.shape[0]) != self.n_env:
                raise ValueError(f"edit flush: plane {name!r} is {a.dtype} {a.shape}, "
                                 f"want {np.dtype(dt)} with {self.n_env} envs")
            out[k] = int(a.data.ptr)
        h, w = (int(x) for x in planes["solid"].shape[1:3])
        return out, h, w

    def run(self, tick_row, tables, planes: dict, n_gases: int, o2_idx: int,
            n2_idx: int) -> None:
        """Slot 6b: the flush from the door (see the class). ``tick_row`` a
        ``unit_draws.TickRow``, ``tables`` the batch's device tables
        (``DeviceStore.tables``), ``planes`` :data:`FLUSH_PLANES` as device
        arrays. Launches only."""
        from simulation import unit_draws
        from simulation.rng_salts import Salt
        ptr = unit_draws.row_ptr(tick_row, self.n_env, "edit flush")
        p, h, w = self._ptrs(planes)
        self._buf.run(self.door.buffer, tables, ptr, p, h, w, int(n_gases), int(o2_idx),
                      int(n2_idx), int(Salt.FIELD_NOISE))

    def run_host_pack(self, packs, planes: dict, n_gases: int, o2_idx: int,
                      n2_idx: int) -> None:
        """THE GATE'S REFERENCE: the same K1 / K2 fed from the host packs
        (``field_edit.pack_for_device`` per env, ``None`` for an env with no
        edit) as the resident tick feeds them (``physics_runner._edit_buffer``)."""
        from simulation.physics_runner import _edit_buffer
        from simulation.rng_salts import Salt
        buf, n_e, n_t = _edit_buffer(list(packs), self.n_env)
        p, h, w = self._ptrs(planes)
        self._buf.run_host_pack(buf, n_e, n_t, p, h, w, int(n_gases), int(o2_idx),
                                int(n2_idx), int(Salt.FIELD_NOISE))

    def books(self) -> np.ndarray:
        """THE ONE READ: ``(N, EDQ_SLOTS)`` books; refuses by name an env the
        build refused (:class:`EditFlushRefused`, the lowest env first) or
        whose results K2 refused (:class:`EditQueueFailed`)."""
        b, f = self._buf.books()
        b = np.asarray(b, dtype=np.int64)
        for e, word in enumerate(np.asarray(f, dtype=np.uint64)):
            word = int(word)
            if word != (1 << 64) - 1:
                code, row = word >> 32, word & 0xffffffff
                name, why = FLUSH_FAULTS.get(code, ("?", "an unknown code"))
                raise EditFlushRefused(f"edit flush: env {e} refused ({name}: {why}; row "
                                       f"{row})", e, code, row)
        for e in range(self.n_env):
            if int(b[e, self.fail_slot]):
                raise EditQueueFailed(
                    f"device edit queue: env {e}: {int(b[e, self.fail_slot])} edit "
                    f"result(s) left their field's range (an int32 store, an int64 "
                    f"energy, a negative trace) -- the host flush raises here")
        return b

    def fold(self, gmaps, rngs) -> np.ndarray:
        """:meth:`books`, then each env's folded into its map and draws as
        the host flush writes them (``physics_runner.fold_edit_queue_books``,
        the resident tick's own fold)."""
        from simulation.physics_runner import fold_edit_queue_books
        b = self.books()
        for e, (g, r) in enumerate(zip(gmaps, rngs)):
            fold_edit_queue_books(self.bp.EDIT_QUEUE_LAYOUT, g, b[e], r)
        return b

    def rows(self, n_edits):
        """The gate's download of the built buffers, per env: ``(edits (E,
        16), tiles (T, 4), env scalars (4,))`` with the tiles' edit index made
        env-local again (``n_edits`` the env's edit count, which the device
        never hands the host)."""
        er, tr, off, env = self._buf.rows()
        cap = self.door.edit_cap
        out = []
        for e in range(self.n_env):
            n = int(n_edits[e])
            t = tr[int(off[e]):int(off[e + 1])].copy()
            t[:, 0] -= e * cap
            out.append((er[e * cap:e * cap + n].copy(), t, env[e].copy()))
        return out


__all__ = ["PACKED_COLS", "SRC_COL", "REGION_COL", "ARG_COL", "ARG_COLS",
           "region_args", "EDIT_COLS", "TILE_COLS",
           "EDIT_CAP", "TILE_CAP", "DoorInterleave", "DoorOverflow", "door_rows",
           "edit_of", "replay", "DeviceEditDoor", "FLUSH_FAULTS", "FLUSH_PLANES",
           "EditFlushRefused", "EditQueueFailed", "DeviceEditFlush"]
