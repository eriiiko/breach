"""#29 P7.0b -- the object tables' host side (``simulation.object_tables``):
the layout and its generated header, the bridge, the host reference of the
append and the stable compaction. The device side is
``tests/cuda_object_table_check.py`` (wrapper ``tests/test_cuda_object_table.py``).

Each test names the property it protects and the change that must break it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import object_tables as ot  # noqa: E402
from simulation import unit_schema  # noqa: E402
from simulation.charges import PlantedCharge  # noqa: E402

T = unit_schema.CHARGE_TABLE
SLOTS = PlantedCharge.__slots__


def _charges():
    """Charges over every encoding branch: a timer and a remote one, an
    approach and none, live and dead."""
    a = PlantedCharge(3, 4, "demo_c4", 7, det_tick=120, approach=(5, 3))
    b = PlantedCharge(10, 2, "demo_breach", -1, det_tick=None, approach=None)
    c = PlantedCharge(1, 1, "demo_c4", 2, det_tick=0, approach=(0, 0))
    c.live = False
    return [a, b, c]


def _same(x, y):
    return all(getattr(x, s) == getattr(y, s) for s in SLOTS)


def test_the_header_is_the_schema():
    """The checked-in header equals the schema's text. Breaks: a schema
    column, capacity or live column changed without regenerating."""
    text = (ROOT / ot.HEADER_REL).read_text(encoding="utf-8")
    assert text == ot.header_text(), \
        f"{ot.HEADER_REL} is stale: run `{ot.GENERATOR}` and commit"


def test_every_table_is_in_the_header_and_the_fingerprint_moves():
    """Every table of OBJECT_TABLES reaches the header, and a change to a
    table's facts moves the fingerprint. Breaks: a table left out of the
    layout, or a fact left out of the fingerprint text."""
    text = ot.header_text()
    for t in unit_schema.OBJECT_TABLES:
        assert f"OBJ_{t.name.upper()}_W" in text
    base = ot.layout().fingerprint
    old = unit_schema.OBJECT_TABLES
    try:
        from dataclasses import replace
        unit_schema.OBJECT_TABLES = tuple(
            replace(t, capacity=t.capacity + 1) for t in old)
        assert ot.layout().fingerprint != base
    finally:
        unit_schema.OBJECT_TABLES = old
    assert ot.layout().fingerprint == base


def test_the_module_holds_no_table_fact():
    """No table's name, column or capacity is written in the module: a
    table is a schema (the clients P7b / P6b add need no module change).
    Breaks: a charge column or capacity hardcoded in object_tables.py."""
    src = (ROOT / "src" / "simulation" / "object_tables.py").read_text(
        encoding="utf-8")
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    for t in unit_schema.OBJECT_TABLES:
        for c in t.columns:
            if len(c.name) > 2:
                assert f'"{c.name}"' not in code and f"'{c.name}'" not in code
        assert f"CHARGE_CAPACITY" not in code
        assert "PlantedCharge" not in code


def test_unpack_of_pack_is_the_list():
    """unpack(pack(list)) rebuilds every object attribute for attribute, in
    order, from the rows alone. Breaks: a column whose decode is not its
    encode's inverse, or a dropped column."""
    cs = _charges()
    rows, n = ot.pack(T, cs)
    assert n == len(cs) and rows.shape == (T.capacity, ot.table_layout(T).width)
    back = ot.unpack(T, rows, n)
    assert len(back) == len(cs)
    for x, y in zip(cs, back):
        assert _same(x, y), (x, y)


def test_rows_are_the_digest_rows():
    """A packed row IS the charge's digest row (``charge_record``), so the
    device form hashes what the reference hashes. Breaks: the module or the
    schema encoding a column another way than the digest."""
    cs = _charges()
    rows, n = ot.pack(T, cs)
    for i, c in enumerate(cs):
        assert tuple(int(v) for v in rows[i]) == unit_schema.charge_record(c)


def test_padding_is_zero_and_never_read():
    """Rows past the count are zero after pack, and unpack never reads them
    (garbage with live = 1 there changes nothing). Breaks: pack leaving old
    rows in ``out``, or unpack reading past the count."""
    cs = _charges()
    out = np.full((T.capacity, ot.table_layout(T).width), 9, dtype=np.int64)
    rows, n = ot.pack(T, cs, out=out)
    assert not rows[n:].any()
    rows[n:, ot.table_layout(T).live] = 1
    assert len(ot.unpack(T, rows, n)) == n


def test_pack_refuses_past_capacity_by_name():
    """More objects than the capacity is refused naming the table (and the
    env in a batch). Breaks: a silent truncation."""
    cs = [PlantedCharge(1, 1, "demo_c4", 1, det_tick=5)] * (T.capacity + 1)
    with pytest.raises(ot.ObjectTableOverflow, match="'charge'"):
        ot.pack(T, cs)
    with pytest.raises(ot.ObjectTableOverflow, match="env 1"):
        ot.pack_batch(T, [cs[:2], cs])


def test_batch_env_is_env_alone():
    """Env i of pack_batch equals pack of env i alone, envs of different
    lengths. Breaks: a batch layout that shifts rows between envs."""
    lists = [_charges(), [], _charges()[:1]]
    rows, counts = ot.pack_batch(T, lists)
    for e, cs in enumerate(lists):
        r, n = ot.pack(T, cs)
        assert counts[e] == n and np.array_equal(rows[e], r)


def test_compaction_is_stable_in_every_position():
    """The host compaction keeps the live rows in order and drops every
    dead one: dead in the middle, at both ends, all, none; the freed rows
    are zero and rows past the count untouched. Breaks: an unstable keep
    (a swap-with-last), a dead row kept, a live row dropped."""
    lay = ot.table_layout(T)
    patterns = ([1, 0, 1, 0, 1], [0, 1, 1, 1, 0], [0, 0, 0], [1, 1, 1], [])
    for live in patterns:
        rows = np.zeros((T.capacity, lay.width), dtype=np.int64)
        for i, l in enumerate(live):
            rows[i, 0] = 100 + i
            rows[i, lay.live] = l
        rows[len(live) + 1, lay.live] = 1          # garbage past the count
        out, n = ot.compact_rows(T, rows, len(live))
        want = [100 + i for i, l in enumerate(live) if l]
        assert n == len(want)
        assert [int(v) for v in out[:n, 0]] == want
        assert not out[n:len(live)].any()
        assert np.array_equal(out[len(live):], rows[len(live):])


def test_append_keeps_order_and_counts_overflow():
    """The host append writes in call order after the count and counts what
    does not fit. Breaks: an append that overwrites, reorders or drops
    silently."""
    lay = ot.table_layout(T)
    rows = np.zeros((4, lay.width), dtype=np.int64)
    rows[0, 0] = 1
    recs = [np.full(lay.width, v, dtype=np.int64) for v in (7, 8, 9, 10)]
    out, n, lost = ot.append_rows(T, rows, 1, recs)
    assert n == 4 and lost == 1
    assert [int(v) for v in out[:, 0]] == [1, 7, 8, 9]
