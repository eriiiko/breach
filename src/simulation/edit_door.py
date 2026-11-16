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
    raise DoorInterleave(f"edit door: a {_fe.Region(edit.region).name} edit has no "
                         f"device form at the door (TILE and DISC do)")


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
    pack, reason = _fe.pack_for_device(gmap, pairs, None)
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


def edit_of(gmap, row, tiles) -> tuple:
    """A device EDIT row (and its TILE rows) as the host FieldEdit it stands
    for and its seq: the inverse of :func:`door_rows` for the forms the device
    door writes (a TILE or -- #29 P7c.1 -- a DISC edit, no noise, no scale, a
    clamp equal to its field policy's: the replayed edit carries none, which
    the door quantizes to the same row). Anything else is refused by name."""
    r = [int(x) for x in row]
    field = _FIELD_OF_ORDINAL.get(r[_fe.E_FIELD])
    region = r[REGION_COL]
    if field is None or region not in (int(_fe.Region.TILE), int(_fe.Region.DISC)) \
            or r[_fe.E_NOISY] or r[_fe.E_DEN] != int(_fe._ufx.FP_ONE):
        raise DoorInterleave(f"edit door: a device row {r} the door does not write")
    pol = _fe._policy(field)
    if (r[_fe.E_CLAMP_FLAGS], r[_fe.E_LO], r[_fe.E_HI]) != _clamp_row(pol) \
            or any(r[_fe.E_O2_LO:_fe.E_N2_HI + 1]):
        raise DoorInterleave(f"edit door: a device row {r} whose clamp is not its "
                             f"field's policy clamp (not replayed)")
    t = [tuple(int(x) for x in tr) for tr in tiles]
    w = int(gmap._w)
    a = r[ARG_COL:ARG_COL + ARG_COLS]
    if region == int(_fe.Region.DISC):
        coords = (a[0], a[1], _fe._ufx.dequantize_scalar(a[2]))
        falloff = _fe.Falloff(a[3])
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
    edit = _fe.FieldEdit(field=field, region=_fe.Region(region), coords=coords,
                         falloff=falloff,
                         amount=amount, amount_q=a_q, mode=_fe.EditMode(r[_fe.E_MODE]),
                         channel=(r[_fe.E_SLICE] if field == "gas" else None),
                         source_id=r[SRC_COL])
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


__all__ = ["PACKED_COLS", "SRC_COL", "REGION_COL", "ARG_COL", "ARG_COLS",
           "region_args", "EDIT_COLS", "TILE_COLS",
           "EDIT_CAP", "TILE_CAP", "DoorInterleave", "DoorOverflow", "door_rows",
           "edit_of", "replay", "DeviceEditDoor"]
