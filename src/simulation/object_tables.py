"""THE OBJECT TABLES -- what flies and what is placed, as device rows (#29
P7.0b; ``docs/residency_batching_design_2026-11-07.md`` §4.4c decision 3).

A host list of world objects (``Simulation.planted_charges`` today; the
grenades and rounds in flight, the multi-tick blast sources and each entity
class later) is a TABLE: per env ``cap`` rows of W int64 and a count. Row i
is the i-th object of the host list -- the order every reference loop walks
-- and the rows past the count are padding (all zero), never read as live.
EVERY fact of a table lives in its schema (``simulation.unit_schema.
ObjectTable``: the columns with their encode / decode, the capacity, the
column whose 0 marks a row dead, the constructor ``unpack`` rebuilds an
object with); this module holds none, so a new table is a client: a schema
and a row of ``unit_schema.OBJECT_TABLES``.

Four parts, one module, over the unit store's machinery:

- **The layout** (:func:`table_layout`, :func:`layout`): each column's offset
  and width (``unit_store.StoreColumn``), W, the live column, the capacity and
  a FINGERPRINT over every table. C++ reads it from the GENERATED, checked-in
  ``cpp/src/object_tables_layout.h`` (:func:`header_text`, written by
  ``tools/gen_unit_layout.py --write``; ``tests/test_object_tables.py``
  refuses a stale copy; :class:`DeviceTable` refuses a compiled module whose
  fingerprint is not the schema's).
- **The bridge** (:func:`pack` / :func:`pack_batch` / :func:`unpack`) over the
  schema's own encode / decode; :func:`records` is THE record function of a
  table's host list (the harness's, the gates').
- **The host reference of the two list operations** (:func:`append_rows`,
  :func:`compact_rows`): what the device does, in numpy, row for row.
- **The device buffer** (:class:`DeviceTable`): ONE ``(N, cap, W)`` int64
  buffer with a per-env count and a per-env OVERFLOW count, allocated once
  (``breach_physics.ObjectTableBuffer``, the CUDA build only). It IS a record
  list (``cuda_unit_events.h``'s ``UnitRecordList``, never a parallel
  mechanism): a serial per-env producer appends through the record list's
  sink (``emit_record``), a per-row kernel stages per row and the record
  list's ONE gather appends per env in row order, both AFTER what the table
  holds -- so an APPEND keeps the order it is called in. Two operations are
  the table's own: the upload of a host state and the STABLE COMPACTION (per
  env, the rows whose live column is 0 dropped, the rest kept in order, the
  freed rows zeroed). Past the capacity nothing is written, the env's
  overflow count rises and the Python face refuses the run naming the table
  and the env (:class:`ObjectTableOverflow`).

NOT PART OF THE PLAYED TICK: Class T (training only), like the unit store.
Import-light: numpy and the schema; ``breach_physics`` only inside
:class:`DeviceTable`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from simulation import unit_schema
from simulation.unit_store import I64_MAX, I64_MIN, StoreColumn

#: The generated header (C++'s copy of every table's layout).
HEADER_REL = "cpp/src/object_tables_layout.h"
GENERATOR = "tools/gen_unit_layout.py --write"


class ObjectTableOverflow(RuntimeError):
    """More objects than a table's capacity (named: the table and the env)."""


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TableLayout:
    name: str
    columns: tuple            # StoreColumn, schema order
    width: int                # W
    live: int                 # the live column's offset
    capacity: int

    def column(self, name: str) -> StoreColumn:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"object table {self.name!r}: no column {name!r}")


@dataclass(frozen=True)
class Layout:
    tables: tuple             # TableLayout, unit_schema.OBJECT_TABLES order
    fingerprint: str

    @property
    def width(self) -> int:
        """The widest table's W (the generator's report)."""
        return max((t.width for t in self.tables), default=0)

    def table(self, name: str) -> TableLayout:
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(f"object tables: no table {name!r}")


def _schema(table) -> unit_schema.ObjectTable:
    if isinstance(table, str):
        for t in unit_schema.OBJECT_TABLES:
            if t.name == table:
                return t
        raise KeyError(f"object tables: no table {table!r}")
    return table


def table_layout(table) -> TableLayout:
    """One table's layout from its schema."""
    t = _schema(table)
    cols, off = [], 0
    for c in t.columns:
        if c.width < 1:
            raise ValueError(f"object table {t.name!r}: column {c.name!r} "
                             f"has width {c.width}")
        cols.append(StoreColumn(c.name, off, c.width, c.width > 1))
        off += c.width
    live = [c for c in cols if c.name == t.live]
    if len(live) != 1 or live[0].width != 1:
        raise ValueError(f"object table {t.name!r}: its live column "
                         f"{t.live!r} is not one int64 column")
    if int(t.capacity) < 1:
        raise ValueError(f"object table {t.name!r}: capacity {t.capacity}")
    return TableLayout(name=t.name, columns=tuple(cols), width=off,
                       live=live[0].offset, capacity=int(t.capacity))


def layout() -> Layout:
    """Every table's layout and THE fingerprint over them."""
    tabs = tuple(table_layout(t) for t in unit_schema.OBJECT_TABLES)
    if len({t.name for t in tabs}) != len(tabs):
        raise ValueError("object tables: a table name repeats")
    text = ""
    for t in tabs:
        text += f"table:{t.name}:W:{t.width}:cap:{t.capacity}:live:{t.live}\n"
        text += "".join(f"{t.name}.{c.name}:{c.offset}:{c.width}\n"
                        for c in t.columns)
    return Layout(tables=tabs,
                  fingerprint=hashlib.sha256(text.encode("ascii")).hexdigest())


def header_text(lay: Layout | None = None) -> str:
    """The generated C++ header's full text (``cpp/src/object_tables_layout.h``)."""
    lay = lay or layout()
    lines = [
        "#pragma once",
        "// ==========================================================================",
        "// GENERATED by tools/gen_unit_layout.py from src/simulation/unit_schema.py",
        "// (OBJECT_TABLES; #29 P7.0b, residency design 4.4c decision 3) -- NEVER",
        "// EDIT BY HAND. The device OBJECT TABLES: per table, per env (CAP, W)",
        "// int64 rows of the host list's objects in list order, the schema's",
        "// columns in schema order. A table, column or capacity added, removed",
        f"// or re-widened means: run `{GENERATOR}` and commit;",
        "// tests/test_object_tables.py refuses a stale copy. Class T.",
        "// ==========================================================================",
        "#include <cstdint>",
        "",
        "namespace object_tables {",
        "",
        f'constexpr const char* OBJECT_TABLES_FINGERPRINT = "{lay.fingerprint}";',
        f"constexpr int OBJECT_TABLES_N = {len(lay.tables)};",
    ]
    for k, t in enumerate(lay.tables):
        n = t.name.upper()
        lines += [
            "",
            f"// table {t.name!r}: its index, W, capacity and live column;",
            "// <column>: its first int64 in the row; <column>_W: its width.",
            f"constexpr int OBJ_{n} = {k};",
            f"constexpr int OBJ_{n}_W = {t.width};",
            f"constexpr int OBJ_{n}_CAP = {t.capacity};",
            f"constexpr int OBJ_{n}_LIVE = {t.live};",
        ]
        for c in t.columns:
            lines.append(f"constexpr int OBJ_{n}_COL_{c.name.upper()} = {c.offset};")
            lines.append(f"constexpr int OBJ_{n}_COL_{c.name.upper()}_W = {c.width};")
    lines += ["", "}  // namespace object_tables", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------------
def records(table, objects) -> list:
    """THE record function of a table's host list: each object as its int
    row (a tuple, schema order), list order."""
    t = _schema(table)
    lay = table_layout(t)
    out = []
    for obj in objects:
        r = unit_schema.table_record(t, obj)
        if len(r) != lay.width:
            raise ValueError(f"object table {t.name!r}: an object encodes to "
                             f"{len(r)} ints, not W {lay.width}")
        for x in r:
            if not I64_MIN <= int(x) <= I64_MAX:
                raise ValueError(f"object table {t.name!r}: {x} is outside "
                                 f"int64")
        out.append(tuple(int(x) for x in r))
    return out


def pack(table, objects, cap: int | None = None, out=None, env=None):
    """``objects`` (a host list, its order) as ``(cap, W)`` int64 rows and
    their count; the rows past the count are zero. More objects than ``cap``
    (default: the schema's capacity) is refused by name
    (:class:`ObjectTableOverflow`)."""
    lay = table_layout(table)
    cap = lay.capacity if cap is None else int(cap)
    recs = records(table, objects)
    if len(recs) > cap:
        where = "" if env is None else f" in env {env}"
        raise ObjectTableOverflow(
            f"object table {lay.name!r}: {len(recs)} objects{where} do not "
            f"fit its capacity {cap}")
    if out is None:
        out = np.zeros((cap, lay.width), dtype=np.int64)
    else:
        if out.shape != (cap, lay.width) or out.dtype != np.int64:
            raise ValueError(f"object table {lay.name!r}: out must be "
                             f"({cap}, {lay.width}) int64, got {out.shape} "
                             f"{out.dtype}")
        out[:] = 0
    if recs:
        out[:len(recs)] = np.array(recs, dtype=np.int64)
    return out, len(recs)


def pack_batch(table, object_lists, cap: int | None = None):
    """N host lists as ``((N, cap, W) rows, (N,) counts)``; env i's rows are
    exactly :func:`pack` of env i alone."""
    lay = table_layout(table)
    cap = lay.capacity if cap is None else int(cap)
    lists = [list(o) for o in object_lists]
    rows = np.zeros((len(lists), cap, lay.width), dtype=np.int64)
    counts = np.zeros(len(lists), dtype=np.int64)
    for e, objs in enumerate(lists):
        _, counts[e] = pack(table, objs, cap, out=rows[e], env=e)
    return rows, counts


def row_values(table, row) -> dict:
    """One row as ``{column: encoding}`` (an int, or a tuple of ints)."""
    lay = table_layout(table)
    r = [int(x) for x in row]
    return {c.name: (tuple(r[c.offset:c.offset + c.width]) if c.tuple_valued
                     else r[c.offset]) for c in lay.columns}


def unpack(table, rows, count) -> list:
    """Fresh host objects from the first ``count`` rows (the schema's decode
    and constructor); the rows past ``count`` are never read."""
    t = _schema(table)
    lay = table_layout(t)
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != lay.width:
        raise ValueError(f"object table {t.name!r}: rows must be (cap, "
                         f"{lay.width}), got {rows.shape}")
    count = int(count)
    if not 0 <= count <= rows.shape[0]:
        raise ValueError(f"object table {t.name!r}: count {count} outside "
                         f"0 .. {rows.shape[0]}")
    out = []
    for i in range(count):
        vals = row_values(t, rows[i])
        out.append(t.build({c.name: c.decode(vals[c.name])
                            for c in t.columns}))
    return out


# ---------------------------------------------------------------------------
# The host reference of the list operations
# ---------------------------------------------------------------------------
def append_rows(table, rows, count, recs):
    """``rows`` / ``count`` with ``recs`` appended in order: ``(rows,
    count, dropped)``; a record past the capacity is not written and is
    counted (the device's semantics)."""
    rows = np.array(rows, dtype=np.int64, copy=True)
    n, lost = int(count), 0
    for r in recs:
        if n >= rows.shape[0]:
            lost += 1
            continue
        rows[n] = np.asarray(r, dtype=np.int64)
        n += 1
    return rows, n, lost


def compact_rows(table, rows, count):
    """The STABLE COMPACTION: the first ``count`` rows whose live column is
    not 0, in order, then zeros; ``(rows, count)``."""
    lay = table_layout(table)
    rows = np.asarray(rows, dtype=np.int64)
    keep = [rows[i] for i in range(int(count)) if int(rows[i, lay.live]) != 0]
    out = np.zeros_like(rows)
    if keep:
        out[:len(keep)] = np.array(keep, dtype=np.int64)
    out[int(count):] = rows[int(count):]        # past the count: untouched
    return out, len(keep)


# ---------------------------------------------------------------------------
# The device buffer
# ---------------------------------------------------------------------------
class LayoutMismatch(RuntimeError):
    """The compiled module's object tables are not the schema's."""


class DeviceTable:
    """ONE table's ``(N, cap, W)`` int64 device buffer with per-env counts
    and overflow counts (see the module docstring), on the CUDA build's tick
    stream. ``n_units`` / ``row_cap``: the per-row staging a per-row kernel
    appends through (the record list's gather). Refuses a compiled module
    whose object tables fingerprint is not the schema's."""

    def __init__(self, table, n_env: int, n_units: int = 1, row_cap: int = 1,
                 cap: int | None = None, bp=None):
        if bp is None:
            import breach_physics as bp
        if not getattr(bp, "HAS_CUDA", False):
            raise RuntimeError("object tables: the device table needs the "
                               "CUDA build (breach_physics.HAS_CUDA)")
        self.schema = _schema(table)
        self.layout = table_layout(self.schema)
        compiled = bp.object_tables_layout()
        if compiled["fingerprint"] != layout().fingerprint:
            raise LayoutMismatch(
                f"object tables: the compiled layout "
                f"({compiled['fingerprint'][:12]}) is not the schema's "
                f"({layout().fingerprint[:12]}): run `{GENERATOR}` and "
                f"rebuild cpp/build_cuda")
        cap = self.layout.capacity if cap is None else int(cap)
        if int(n_env) < 1 or cap < 1:
            raise ValueError(f"object table {self.schema.name!r}: n_env "
                             f"{n_env} and cap {cap} must be >= 1")
        self.bp = bp
        self.name = self.schema.name
        self.shape = (int(n_env), cap, self.layout.width)
        self._buf = bp.ObjectTableBuffer(int(n_env), cap, self.layout.width,
                                         self.layout.live, int(n_units),
                                         int(row_cap))

    @property
    def buffer(self):
        """The bound ``ObjectTableBuffer`` (a ``UnitRecordListBuffer``: what
        a kernel appending to the table takes)."""
        return self._buf

    def upload(self, rows, counts) -> None:
        """Set every env's table: ``rows`` ``(N, cap, W)``, ``counts``
        ``(N,)`` (each in 0 .. cap); the overflow counts back to 0."""
        a = np.ascontiguousarray(rows, dtype=np.int64)
        c = np.ascontiguousarray(counts, dtype=np.int64)
        if a.shape != self.shape or c.shape != (self.shape[0],):
            raise ValueError(f"object table {self.name!r}: upload needs "
                             f"{self.shape} rows and ({self.shape[0]},) "
                             f"counts, got {a.shape} / {c.shape}")
        if (c < 0).any() or (c > self.shape[1]).any():
            raise ValueError(f"object table {self.name!r}: a count outside "
                             f"0 .. {self.shape[1]}")
        self._buf.upload(a, c)

    def download(self):
        """``(rows (N, cap, W), counts (N,), overflow (N,))`` int64."""
        return self._buf.download()

    def check(self, what: str = "") -> None:
        """Refuse by name an env whose table overflowed."""
        _rows, _counts, over = self.download()
        bad = [e for e in range(len(over)) if int(over[e])]
        if bad:
            e = bad[0]
            raise ObjectTableOverflow(
                f"object table {self.name!r}{' ' + what if what else ''}: "
                f"env {e} lost {int(over[e])} row(s) past its capacity "
                f"{self.shape[1]}")

    def append(self, recs, n) -> None:
        """Append, per env e, its first ``n[e]`` records of ``recs`` ``(N, k,
        W)`` in order after what the table holds (one thread per env through
        the record list's sink); then :meth:`check`."""
        a = np.ascontiguousarray(recs, dtype=np.int64)
        c = np.ascontiguousarray(n, dtype=np.int64)
        if a.ndim != 3 or a.shape[0] != self.shape[0] \
                or a.shape[2] != self.shape[2] or c.shape != (self.shape[0],):
            raise ValueError(f"object table {self.name!r}: append needs "
                             f"(N, k, {self.shape[2]}) records and (N,) "
                             f"counts, got {a.shape} / {c.shape}")
        if (c < 0).any() or (c > a.shape[1]).any():
            raise ValueError(f"object table {self.name!r}: an append count "
                             f"outside 0 .. {a.shape[1]}")
        self._buf.append(a, c)
        self.check("append")

    def compact(self) -> None:
        """The stable compaction of every env (rows whose live column is 0
        dropped, the rest in order, the freed rows zeroed)."""
        self._buf.compact()

    def stage_lists(self, per_env) -> None:
        """#29 P7a.2 (the harness's list face): set every env's table from
        its ``(R_e, W)`` host records (:func:`records` of the env's list);
        more than the capacity is refused by name."""
        n, cap, w = self.shape
        rows = np.zeros(self.shape, dtype=np.int64)
        counts = np.zeros(n, dtype=np.int64)
        for e, r in enumerate(per_env):
            r = np.asarray(r, dtype=np.int64).reshape(-1, w)
            if r.shape[0] > cap:
                raise ObjectTableOverflow(
                    f"object table {self.name!r}: {r.shape[0]} objects in env "
                    f"{e} do not fit its capacity {cap}")
            rows[e, :r.shape[0]] = r
            counts[e] = r.shape[0]
        self.upload(rows, counts)

    def begin(self) -> None:
        """Every env's table emptied (the harness's output-only list)."""
        self.upload(np.zeros(self.shape, dtype=np.int64),
                    np.zeros(self.shape[0], dtype=np.int64))

    def records(self) -> list:
        """Per env, its rows as int tuples (list order)."""
        rows, counts, _ = self.download()
        return [[tuple(int(x) for x in rows[e, i])
                 for i in range(int(counts[e]))] for e in range(rows.shape[0])]


__all__ = ["HEADER_REL", "GENERATOR", "ObjectTableOverflow", "TableLayout",
           "Layout", "table_layout", "layout", "header_text", "records",
           "pack", "pack_batch", "row_values", "unpack", "append_rows",
           "compact_rows", "LayoutMismatch", "DeviceTable"]
