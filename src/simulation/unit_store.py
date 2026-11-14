"""THE device unit store (#29 P6.1; ``docs/residency_batching_design_2026-11-07.md``
§4.4b, decisions 1 to 3).

A unit is a ROW of W int64: the unit schema's group-S columns
(:data:`simulation.unit_schema.COLUMNS`) in schema order, each at its ENCODED
width (``Column.encode`` -- an int, or a tuple of ints for a ``block`` /
``optional`` column), then ``valid`` (1 = a unit occupies the row, 0 = a
padding row). A sim is ``(U, W)``, a batch of sims ``(N, U, W)``: row r of env
e is the r-th unit of that env's ``sim.units`` -- id (spawn) order, the order
every reference loop walks -- and an env with fewer units than the batch's U
pads with all-zero rows whose ``valid`` is 0. ``(team, slot)`` are columns,
never the row order.

Three parts, one module:

- **The layout** (:func:`layout`): each column's offset and width, W, the
  ``valid`` column and a schema FINGERPRINT, derived from the schema (and the
  config tables its widths read: the status registry, the damage types, the
  hand-grenade rows, the hooks). C++ reads it from the GENERATED, checked-in
  ``cpp/src/unit_store_layout.h`` (:func:`header_text`, written by
  ``tools/gen_unit_layout.py --write``; ``tests/test_unit_store.py`` holds the
  file equal to the schema). A new schema column is: regenerate, commit.
- **The bridge** (:func:`pack` / :func:`pack_batch` / :func:`unpack`), built
  on the schema's own ``encode_unit`` / ``decode_into``; and the unit digest
  record read from a ROW (:func:`record_from_row`, :func:`section_bytes`), so
  a row can be checked against the reference's ``unit_section_bytes``.
- **The device buffer** (:class:`DeviceStore`): ONE ``(N, U, W)`` int64 device
  buffer allocated once (``breach_physics.UnitStoreBuffer``, the CUDA build
  only), uploaded and downloaded on THE tick stream through pinned staging,
  and the pilot kernel (slot 1c, exposure decay; ``cpp/src/cuda_unit_store.cu``).

NOT PART OF THE PLAYED TICK (decision 3): nothing in ``Simulation.step``,
``ResidentBatch`` or ``PhysicsRunner.run_resident_batch`` calls this module;
the unit kernels are Class T (training only). The gates use it now
(``tests/test_unit_store.py`` IDENTITY, ``tests/cuda_unit_store_check.py`` the
device round trip and the SHADOW LOCKSTEP through ``tests/cuda_unit_lockstep.py``);
the ``device`` backend uses it from P8b.

Import-light: numpy and the schema at module level; ``breach_physics`` only
inside :class:`DeviceStore`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from simulation import unit_schema
from simulation.unit_schema import (COLUMNS, KIND_BLOCK, KIND_OPTIONAL,
                                    column_width)

#: The store's last column: 1 = a unit occupies the row, 0 = padding.
VALID = "valid"
I64_MIN = -(2 ** 63)
I64_MAX = 2 ** 63 - 1
#: The generated header (C++'s copy of the layout).
HEADER_REL = "cpp/src/unit_store_layout.h"
GENERATOR = "tools/gen_unit_layout.py --write"
_TUPLE_KINDS = (KIND_BLOCK, KIND_OPTIONAL)


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StoreColumn:
    """One schema column's place in a row: ``offset`` .. ``offset + width``.
    ``tuple_valued``: its encoding is a tuple (``block`` / ``optional``),
    else one int."""
    name: str
    offset: int
    width: int
    tuple_valued: bool


@dataclass(frozen=True)
class Layout:
    columns: tuple            # StoreColumn, schema order
    width: int                # W: every column, then `valid`
    valid: int                # the `valid` column's offset (== W - 1)
    fingerprint: str          # sha256 hex of the layout's text

    def column(self, name: str) -> StoreColumn:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"unit store: no column {name!r}")

    def column_at(self, offset: int) -> str:
        """The column name holding row offset ``offset`` (``valid`` last)."""
        if offset == self.valid:
            return VALID
        for c in self.columns:
            if c.offset <= offset < c.offset + c.width:
                return c.name if c.width == 1 else \
                    f"{c.name}[{offset - c.offset}]"
        raise IndexError(offset)


_LAYOUT_CACHE: dict = {}


def layout() -> Layout:
    """THE store layout from the live schema (cached on the column widths,
    which read config tables)."""
    widths = tuple(column_width(c) for c in COLUMNS)
    hit = _LAYOUT_CACHE.get(widths)
    if hit is not None:
        return hit
    cols, off = [], 0
    for c, w in zip(COLUMNS, widths):
        if w < 1:
            raise ValueError(f"unit store: column {c.name!r} has width {w}")
        cols.append(StoreColumn(c.name, off, w, c.kind in _TUPLE_KINDS))
        off += w
    text = "".join(f"{c.name}:{c.offset}:{c.width}\n" for c in cols)
    text += f"{VALID}:{off}:1\nW:{off + 1}\n"
    lay = Layout(columns=tuple(cols), width=off + 1, valid=off,
                 fingerprint=hashlib.sha256(text.encode("ascii")).hexdigest())
    _LAYOUT_CACHE[widths] = lay
    return lay


def header_text(lay: Layout | None = None) -> str:
    """The generated C++ header's full text (``cpp/src/unit_store_layout.h``)."""
    lay = lay or layout()
    lines = [
        "#pragma once",
        "// ==========================================================================",
        "// GENERATED by tools/gen_unit_layout.py from src/simulation/unit_schema.py",
        "// (#29 P6.1, residency design 4.4b decision 1) -- NEVER EDIT BY HAND.",
        "// The device unit store's row layout: a unit is a row of UNIT_STORE_W",
        "// int64, the schema's group-S columns in schema order at their encoded",
        "// widths, then `valid`. A schema column added, removed or re-widened",
        "// (a config row that moves a block's width included) means: run",
        f"// `{GENERATOR}` and commit; tests/test_unit_store.py refuses a",
        "// stale copy. Read by cpp/src/cuda_unit_store.cu (Class T).",
        "// ==========================================================================",
        "#include <cstdint>",
        "",
        "namespace unit_store {",
        "",
        f'constexpr const char* UNIT_STORE_FINGERPRINT = "{lay.fingerprint}";',
        f"constexpr int UNIT_STORE_W = {lay.width};",
        f"constexpr int UNIT_STORE_N_COLUMNS = {len(lay.columns)};",
        f"constexpr int UNIT_COL_VALID = {lay.valid};",
        "",
        "// <column>: its first int64 in the row; <column>_W: its width.",
    ]
    for c in lay.columns:
        n = c.name.upper()
        lines.append(f"constexpr int UNIT_COL_{n} = {c.offset};")
        lines.append(f"constexpr int UNIT_COL_{n}_W = {c.width};")
    lines += ["", "}  // namespace unit_store", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------------
def _check_order(units) -> None:
    prev = None
    for u in units:
        uid = int(u.id)
        if prev is not None and uid <= prev:
            raise ValueError(
                f"unit store: sim.units is not in ascending id order (id "
                f"{uid} after {prev}); a row is the r-th unit in the laws' "
                f"id order")
        prev = uid


def _flat(u, lay: Layout) -> list:
    enc = unit_schema.encode_unit(u)
    out = []
    for c in lay.columns:
        v = enc[c.name]
        if c.tuple_valued:
            if not isinstance(v, tuple) or len(v) != c.width:
                raise ValueError(
                    f"unit store: unit {u.id} column {c.name!r} encodes to "
                    f"{v!r}, not a tuple of width {c.width}")
            vals = v
        else:
            if isinstance(v, tuple):
                raise ValueError(f"unit store: unit {u.id} column "
                                 f"{c.name!r} encodes to a tuple {v!r}")
            vals = (v,)
        for x in vals:
            x = int(x)
            if not I64_MIN <= x <= I64_MAX:
                raise ValueError(
                    f"unit store: unit {u.id} column {c.name!r} holds {x}, "
                    f"outside int64")
            out.append(x)
    out.append(1)                       # valid
    return out


def pack(units, n_units: int | None = None, out=None) -> np.ndarray:
    """``units`` (a ``sim.units``, ascending id) as ``(n_units, W)`` int64
    rows; rows past ``len(units)`` are zero with ``valid`` 0. Refused by
    name: more units than ``n_units``, an id order that is not ascending, a
    value outside int64."""
    lay = layout()
    units = list(units)
    U = len(units) if n_units is None else int(n_units)
    if len(units) > U:
        raise ValueError(f"unit store: {len(units)} units do not fit "
                         f"{U} rows")
    _check_order(units)
    if out is None:
        out = np.zeros((U, lay.width), dtype=np.int64)
    else:
        if out.shape != (U, lay.width) or out.dtype != np.int64:
            raise ValueError(f"unit store: out must be ({U}, {lay.width}) "
                             f"int64, got {out.shape} {out.dtype}")
        out[len(units):] = 0
    if units:
        out[:len(units)] = np.array([_flat(u, lay) for u in units],
                                    dtype=np.int64)
    return out


def pack_batch(unit_lists, n_units: int | None = None) -> np.ndarray:
    """N unit lists as ``(N, U, W)``, U the largest (or ``n_units``); env
    i's rows are exactly :func:`pack` of env i alone at that U."""
    lists = [list(us) for us in unit_lists]
    U = max((len(us) for us in lists), default=0) if n_units is None \
        else int(n_units)
    out = np.zeros((len(lists), U, layout().width), dtype=np.int64)
    for e, us in enumerate(lists):
        pack(us, U, out=out[e])
    return out


def row_values(row) -> dict:
    """One row as ``{column: encoding}`` (an int, or a tuple of ints)."""
    lay = layout()
    r = [int(x) for x in row]
    return {c.name: (tuple(r[c.offset:c.offset + c.width]) if c.tuple_valued
                     else r[c.offset]) for c in lay.columns}


def unpack(rows, units) -> None:
    """Write ``rows`` (``(U, W)``) onto ``units`` (row r onto the r-th):
    the bridge's direction store -> unit, through ``decode_into``. A row lands
    on its own unit (same id) or on a blank one (id at the schema default,
    :func:`blank_units`). Refused by name: more units than rows, a padding
    row under a unit, a valid row with no unit, another unit's row."""
    lay = layout()
    rows = np.asarray(rows)
    units = list(units)
    if rows.ndim != 2 or rows.shape[1] != lay.width:
        raise ValueError(f"unit store: rows must be (U, {lay.width}), got "
                         f"{rows.shape}")
    if len(units) > rows.shape[0]:
        raise ValueError(f"unit store: {len(units)} units, {rows.shape[0]} "
                         f"rows")
    blank = unit_schema.COLUMN_BY_NAME["id"].host_default()
    id_off = lay.column("id").offset
    for r in range(rows.shape[0]):
        valid = int(rows[r, lay.valid])
        if r >= len(units):
            if valid:
                raise ValueError(f"unit store: row {r} holds unit "
                                 f"{int(rows[r, id_off])} but no unit was "
                                 f"given for it")
            continue
        u = units[r]
        if not valid:
            raise ValueError(f"unit store: row {r} is padding (valid 0) "
                             f"under unit {getattr(u, 'id', '?')}")
        rid = int(rows[r, id_off])
        uid = getattr(u, "id", blank)
        if uid != rid and uid != blank:
            raise ValueError(f"unit store: row {r} is unit {rid}'s, not "
                             f"unit {uid}'s")
        unit_schema.decode_into(u, row_values(rows[r]))


def blank_units(n: int) -> list:
    """``n`` units holding only their schema defaults (no constructor: the
    group-D attributes stay unset) -- what :func:`unpack` writes a store into
    to prove the rows alone carry the state."""
    from simulation.unit import Unit
    out = []
    for _ in range(int(n)):
        u = Unit.__new__(Unit)
        unit_schema.apply_defaults(u)
        out.append(u)
    return out


class _RowView:
    """A row read as a unit for a column's ``digest_of`` (each attribute a
    column's DECODED value, by ``Column.attr``)."""

    def __init__(self, values: dict):
        self._v = values

    def __getattr__(self, name):
        for c in COLUMNS:
            if c.attr == name or c.name == name or c.store == name:
                return c.decode(self._v[c.name])
        raise AttributeError(name)


def record_from_row(row) -> dict:
    """The unit digest record (``unit_schema.digest_record``'s) read from a
    ROW alone: every column under its key, an *always* column always, every
    other while its value differs from its default's."""
    vals = row_values(row)
    view = None
    rec = {}
    for c in COLUMNS:
        if c.digest_of is not None:
            if view is None:
                view = _RowView(vals)
            v = c.digest_of(view)
        else:
            v = vals[c.name]
        if c.digest == unit_schema.DIGEST_ALWAYS or v != c.digest_default():
            rec[c.key] = v
    return rec


def section_bytes(rows) -> bytes:
    """The ``UNIT_SECT_V2`` bytes of a store's valid rows (row order = id
    order), equal to the reference's ``unit_section_bytes``."""
    lay = layout()
    rows = np.asarray(rows)
    return unit_schema.unit_section_bytes(
        record_from_row(r) for r in rows if int(r[lay.valid]))


# ---------------------------------------------------------------------------
# The device buffer
# ---------------------------------------------------------------------------
class LayoutMismatch(RuntimeError):
    """The compiled module's layout is not the schema's."""


class DeviceStore:
    """THE ``(N, U, W)`` int64 device unit store, allocated ONCE (never per
    tick), on the CUDA build's tick stream: :meth:`upload` / :meth:`download`
    stage through pinned host memory (async copies), and the slot kernels run
    on it in place. Refuses a compiled module whose layout fingerprint is
    not the schema's (a header regenerated without a rebuild, or the reverse)."""

    def __init__(self, n_env: int, n_units: int, bp=None):
        if bp is None:
            import breach_physics as bp
        if not getattr(bp, "HAS_CUDA", False):
            raise RuntimeError("unit store: the device store needs the CUDA "
                               "build (breach_physics.HAS_CUDA)")
        lay = layout()
        compiled = bp.unit_store_layout()
        if compiled["fingerprint"] != lay.fingerprint \
                or int(compiled["width"]) != lay.width:
            raise LayoutMismatch(
                f"unit store: the compiled layout (W {compiled['width']}, "
                f"{compiled['fingerprint'][:12]}) is not the schema's (W "
                f"{lay.width}, {lay.fingerprint[:12]}): run `{GENERATOR}` "
                f"and rebuild cpp/build_cuda")
        if int(n_env) < 1 or int(n_units) < 1:
            raise ValueError(f"unit store: n_env {n_env} and n_units "
                             f"{n_units} must be >= 1")
        self.layout = lay
        self.shape = (int(n_env), int(n_units), lay.width)
        self._buf = bp.UnitStoreBuffer(int(n_env), int(n_units), lay.width)

    @property
    def device_ptr(self) -> int:
        """The buffer's device address (a CuPy view's ``UnownedMemory``)."""
        return int(self._buf.ptr)

    def upload(self, rows) -> None:
        """Copy ``rows`` (``(N, U, W)`` int64) into the device store."""
        a = np.ascontiguousarray(rows)
        if a.shape != self.shape or a.dtype != np.int64:
            raise ValueError(f"unit store: upload needs {self.shape} int64, "
                             f"got {a.shape} {a.dtype}")
        self._buf.upload(a)

    def download(self) -> np.ndarray:
        """The device store as a fresh ``(N, U, W)`` int64 array."""
        return self._buf.download()

    def exposure_decay(self, decay_q: int) -> np.ndarray:
        """Slot 1c on the device (``exposure.decay_step`` on every valid row's
        exposure block); returns, per env, the rows whose block changed."""
        return self._buf.exposure_decay(int(decay_q))


__all__ = ["VALID", "HEADER_REL", "GENERATOR", "StoreColumn", "Layout",
           "layout", "header_text", "pack", "pack_batch", "row_values",
           "unpack", "blank_units", "record_from_row", "section_bytes",
           "LayoutMismatch", "DeviceStore"]
