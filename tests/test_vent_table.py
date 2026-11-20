"""#29 P6b.2 -- the duct table and the vent table (``unit_schema.DUCT_TABLE`` /
``VENT_TABLE``, the second and third entity tables), on the host (no CUDA):
the schemas, the record functions, the landing back onto the sim's runtimes,
the entity rows read from the tables and the host twin of the sweep's range
check. The device side is tests/cuda_unit_vents_check.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tools") not in sys.path:
    sys.path.append(str(_ROOT / "tools"))

import level_loader  # noqa: E402
from simulation import object_tables as ot, unit_kernels, unit_schema  # noqa: E402

VENT_LEVELS = ("ship_demo", "test_arena", "fire_smoke_studio")


def _sim(name):
    from simulation import Simulation
    return Simulation(level_loader.load(name, str(_ROOT / "levels")), seed=1,
                      enable_recorder=False)


def _mutate(sim):
    """Runtime state a duct and a vent carry across ticks, varied row by row
    (Python's unbounded ints at int64's edges, negative energies)."""
    for k, d in enumerate(sim._ducts):
        d.o2_raw, d.n2_raw = 1000 + k, (1 << 61) - k
        d.trace_raw = [k * 7 + i for i in range(unit_schema.VENT_N_TRACE)]
        d.sink = [(1 << 62) - i for i in range(unit_schema.VENT_N_TRACE)]
        d.e_plenum, d.e_wipe = -(1 << 62) + k, (1 << 63) - 1 - k
        d.rail_lo_hits, d.rail_hi_hits = k, 3 * k + 1
    for k, v in enumerate(sim._vents):
        v.accum = k % max(1, v.tps)
        v.guard_skips = 11 * k
        v.alive = bool(k % 2 == 0)


def _duct_state(d):
    return (int(d.ordinal), d.inst.fields["filter"], d.o2_raw, d.n2_raw, list(d.trace_raw),
            d.e_plenum, list(d.sink), d.e_wipe, d.rail_lo_hits, d.rail_hi_hits, bool(d.alive))


def _vent_state(v):
    return (int(v.ordinal), unit_schema._vent_duct_ordinal(v),
            "return" if v.role == "return" else "supply", v.aperture_y, v.aperture_x,
            v.rate_raw_per_s, v.tps, v.accum, v.guard_skips, bool(v.alive))


@pytest.mark.parametrize("name", VENT_LEVELS)
def test_duct_and_vent_tables_round_trip_are_the_runtimes(name):
    """Property: unpack(records(sim._ducts)) and unpack(records(sim._vents))
    hold every attribute the 9e(d) sweep reads or writes, in ordinal order,
    with the ledgers at int64's edges. Breaks if a column is dropped, a
    duct's filter or a vent's duct / role is encoded wrongly, or the order
    is not the runtimes'."""
    sim = _sim(name)
    assert sim._ducts and sim._vents
    _mutate(sim)
    ducts = unit_kernels.duct_objects(unit_kernels.ducts_of(sim))
    vents = unit_kernels.vent_objects(unit_kernels.vents_of(sim))
    assert [(r.ordinal, r.filter, r.o2_raw, r.n2_raw, r.trace_raw, r.e_plenum, r.sink,
             r.e_wipe, r.rail_lo_hits, r.rail_hi_hits, r.alive) for r in ducts] == \
        [_duct_state(d) for d in sim._ducts]
    assert [(r.ordinal, r.duct_ordinal, r.role, r.aperture_y, r.aperture_x,
             r.rate_raw_per_s, r.tps, r.accum, r.guard_skips, r.alive) for r in vents] == \
        [_vent_state(v) for v in sim._vents]
    assert [int(d.ordinal) for d in sim._ducts] == sorted(int(d.ordinal) for d in sim._ducts)


@pytest.mark.parametrize("name", VENT_LEVELS)
def test_tables_land_on_the_sims_own_runtimes(name):
    """Property: ducts_to_host / vents_to_host write a table's state onto the
    sim's OWN runtime objects (the entity list holds them) and refuse rows
    that are another sim's. Breaks if the landing builds new objects, skips a
    ledger, or accepts a foreign ordinal."""
    sim = _sim(name)
    keep_d, keep_v = list(sim._ducts), list(sim._vents)
    other = _sim(name)
    _mutate(other)
    unit_kernels.ducts_to_host(sim, unit_kernels.ducts_of(other))
    unit_kernels.vents_to_host(sim, unit_kernels.vents_of(other))
    assert all(a is b for a, b in zip(keep_d, sim._ducts))
    assert all(a is b for a, b in zip(keep_v, sim._vents))
    assert [_duct_state(d) for d in sim._ducts] == [_duct_state(d) for d in other._ducts]
    assert [_vent_state(v) for v in sim._vents] == [_vent_state(v) for v in other._vents]
    bad = unit_kernels.ducts_of(other).copy()
    bad[0, 0] += 1000
    with pytest.raises(unit_kernels.VentTableMismatch):
        unit_kernels.ducts_to_host(sim, bad)
    badv = unit_kernels.vents_of(other).copy()
    badv[0, ot.table_layout(unit_schema.VENT_TABLE).column("aperture_x").offset] += 1
    with pytest.raises(unit_kernels.VentTableMismatch):
        unit_kernels.vents_to_host(sim, badv)


@pytest.mark.parametrize("name", VENT_LEVELS)
def test_entity_rows_read_from_the_tables_are_the_serializers(name):
    """Property: duct_entity_rows / vent_entity_rows (P6b.4's export) read
    from a table row equal the serializer's ENTITY_SECT_V1 rows of that
    entity (the free ``alive`` row, then its ``runtime_digest_rows``).
    Breaks if a row is missing, renamed or out of order."""
    from simulation.entities import vents as vent_schema
    sim = _sim(name)
    _mutate(sim)
    for d, rec in zip(sim._ducts, unit_kernels.ducts_of(sim)):
        want = (("alive", 1 if d.alive else 0),) + vent_schema.duct.runtime_digest_rows(d)
        assert unit_kernels.duct_entity_rows(rec) == want
    for v, rec in zip(sim._vents, unit_kernels.vents_of(sim)):
        want = (("alive", 1 if v.alive else 0),) + vent_schema.vent.runtime_digest_rows(v)
        assert unit_kernels.vent_entity_rows(rec) == want


def test_a_duct_past_the_proved_range_is_named_and_one_step_inside_is_not():
    """Property: the host twin of the sweep's head check
    (``vent_ledger_range``) accepts every ledger AT its bound (which depends
    on the duct's returns' quanta and its vent counts) and names the ledger
    one past it, on both sides of a signed ledger. Breaks if a bound is off by
    one or a ledger family is not checked."""
    sim = _sim("test_arena")
    drec = unit_kernels.ducts_of(sim)
    vrec = unit_kernels.vents_of(sim)
    lay = ot.table_layout(unit_schema.DUCT_TABLE)
    row = drec[0].copy()
    assert unit_kernels.vent_ledger_range(row, vrec) == []
    ord0 = int(row[lay.column("ordinal").offset])
    vl = ot.table_layout(unit_schema.VENT_TABLE)
    b_n = r = s = 0
    for v in vrec:
        if int(v[vl.column("duct_ordinal").offset]) != ord0:
            continue
        if int(v[vl.column("role").offset]) == 1:
            q = (int(v[vl.column("accum").offset]) + int(v[vl.column("rate_raw_per_s").offset])) \
                // int(v[vl.column("tps").offset])
            b_n += min(q, 1 << 32)
            r += 1
        else:
            s += 1
    assert r and s
    bounds = {"o2_raw": ((1 << 61) - b_n, None), "n2_raw": ((1 << 61) - b_n, None),
              "e_plenum": ((1 << 61) - (b_n << 29), -((1 << 61) - (b_n << 29))),
              "e_wipe": (1 << 62, -(1 << 62)),
              "rail_lo_hits": ((1 << 62) - s, None), "rail_hi_hits": ((1 << 62) - s, None),
              "trace_raw": ((1 << 54) - r * (1 << 31), None),
              "sink": ((1 << 62) - r * (1 << 31), None)}
    for name, (hi, lo) in bounds.items():
        c = lay.column(name)
        for edge, step in ((hi, 1), (lo, -1)):
            if edge is None:
                edge, step = 0, -1                   # a count: below zero is out
            x = row.copy()
            x[c.offset + c.width - 1] = edge
            assert unit_kernels.vent_ledger_range(x, vrec) == [], (name, edge)
            x[c.offset + c.width - 1] = edge + step
            got = unit_kernels.vent_ledger_range(x, vrec)
            assert [g[0] for g in got] == [name], (name, edge + step, got)


def test_capacities_refuse_by_name():
    """Property: a duct or vent list past its table's capacity is refused by
    name (``ObjectTableOverflow``). Breaks if a table silently truncates."""
    sim = _sim("test_arena")
    for table, objs, cap in ((unit_schema.DUCT_TABLE, sim._ducts, unit_schema.DUCT_CAPACITY),
                             (unit_schema.VENT_TABLE, sim._vents, unit_schema.VENT_CAPACITY)):
        many = list(objs) * (cap // len(objs) + 1)
        with pytest.raises(ot.ObjectTableOverflow, match=table.name):
            ot.pack(table, many[:cap + 1])
        ot.pack(table, many[:cap])
