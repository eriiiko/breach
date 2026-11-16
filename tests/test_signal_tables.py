"""#29 P6b.3 -- the signal tables (``unit_schema.SIGNAL_TABLE``, ``SENSOR_TABLE``
+ ``SENSOR_TILE_TABLE``, ``NODE_TABLE`` + ``NODE_IN_TABLE``, ``PUMP_TABLE`` +
``PUMP_IN_TABLE``, ``DOOR_DRIVE_TABLE``) on the host (no CUDA): the record
functions, the landing back onto the sim's runtimes, the digest and entity
rows read from the tables, the codes and the host twin of the filter's range
check. The device side is tests/cuda_unit_signals_check.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tools") not in sys.path:
    sys.path.append(str(_ROOT / "tools"))

import level_loader  # noqa: E402
from simulation import object_tables as ot, unit_kernels as uk, unit_schema as us  # noqa: E402


def _sim(name="airlock_demo", ticks=0):
    from simulation import Simulation
    sim = Simulation(level_loader.load(name, str(_ROOT / "levels")), seed=1,
                     enable_recorder=False)
    for _ in range(ticks):
        sim.step()
    return sim


def _round_trip(table, rows):
    recs = uk._records_of(table, rows)
    sides = None
    if table.spans:
        side = uk._side_of(table, rows, table.spans[0].name)
        sides = {table.spans[0].name: (side, side.shape[0])}
    return ot.unpack(table, recs, recs.shape[0], sides=sides)


@pytest.mark.parametrize("ticks", (0, 30))
def test_signal_tables_round_trip_are_the_runtimes(ticks):
    """Property: unpack(records(rows)) is every signal table's row objects on
    airlock_demo (the bus in slot order, the rest in ordinal order), before
    and after the tick has moved the bus and the latches. Breaks if a column
    is dropped, mis-encoded or a span points at another row."""
    sim = _sim(ticks=ticks)
    for fn, table in ((uk.signal_rows, us.SIGNAL_TABLE), (uk.sensor_rows, us.SENSOR_TABLE),
                      (uk.node_rows, us.NODE_TABLE), (uk.pump_rows, us.PUMP_TABLE),
                      (uk.drive_rows, us.DOOR_DRIVE_TABLE)):
        rows = fn(sim)
        assert rows, f"{table.name}: airlock_demo holds none (vacuous)"
        assert _round_trip(table, rows) == rows, table.name


def test_the_bus_table_is_the_bus():
    """Property: the bus table holds the bus's slots in order with their pub
    and stg, its digest rows are SignalBus.digest_rows, and landing rows back
    writes pub and stg. Breaks if the slot order, the alive exclusion or the
    landing moves."""
    sim = _sim(ticks=12)
    bus = sim._signal_bus
    recs = uk.signals_of(sim)
    assert [(r.ordinal, r.name) for r in uk.signal_objects(recs)] == list(bus.slots)
    assert uk.signal_digest_rows(recs) == bus.digest_rows()
    lay = ot.table_layout(us.SIGNAL_TABLE)
    moved = recs.copy()
    moved[:, lay.column("pub").offset] += 5
    moved[:, lay.column("stg").offset] -= 3
    uk.signals_to_host(sim, moved)
    assert uk.signals_of(sim).tolist() == moved.tolist()


def test_node_and_pump_landing_and_entity_rows():
    """Property: the node table's ema / state / dwell and the pump table's
    at_target land on the sim's own evaluators, and their entity rows are the
    serializer's. Breaks if a landing misses a runtime attribute or an entity
    row is read from the wrong column."""
    from simulation.entities import REGISTRY
    sim = _sim(ticks=8)
    lay = ot.table_layout(us.NODE_TABLE)
    recs = uk.nodes_of(sim)
    moved = recs.copy()
    for r in moved:
        if us.NODE_KINDS[int(r[lay.column("kind").offset])] == "airlock_controller":
            r[lay.column("state").offset] = 4
            r[lay.column("dwell").offset] = 17
    uk.nodes_to_host(sim, moved)
    ctrl = [n for n in sim._logic_nodes if hasattr(n, "dwell")][0]
    assert (ctrl.state, ctrl.dwell) == (4, 17)
    by = {int(e.ordinal): e for e in sim.entities}
    for rec in uk.nodes_of(sim):
        e = by[int(rec[0])]
        assert uk.node_entity_rows(rec) == (("alive", 1),) + tuple(
            REGISTRY[e.class_name].runtime_digest_rows(e))
    prec = uk.pumps_of(sim)
    pl = ot.table_layout(us.PUMP_TABLE)
    prec2 = prec.copy()
    prec2[:, pl.column("at_target").offset] = 1 - prec2[:, pl.column("at_target").offset]
    uk.pumps_to_host(sim, prec2)
    assert [p.at_target for p in sim._pumps] == list(prec2[:, pl.column("at_target").offset])
    for rec, p in zip(uk.pumps_of(sim), sim._pumps):
        assert uk.pump_entity_rows(rec) == (("alive", 1),) + tuple(
            REGISTRY["pump"].runtime_digest_rows(p))


def test_constant_tables_refuse_other_rows():
    """Property: the load-constant tables (sensors, drives) land only by being
    the sim's own: another row is refused by name. Breaks if a landing
    silently accepts a mismatched table."""
    sim = _sim()
    for fn, of in ((uk.sensors_to_host, uk.sensors_of), (uk.drives_to_host, uk.door_drives_of)):
        bad = of(sim).copy()
        bad[0, 0] += 1
        with pytest.raises(uk.SignalTableMismatch):
            fn(sim, bad)


def test_the_codes_cover_every_class():
    """Property: every registered logic-node class is a NODE_KINDS code, every
    sensor class a SENSOR_KINDS one (field sensors through their frozen
    channel), and the comparator codes are the decider's tokens. Breaks if a
    node or sensor class is added without its device code (the tables would
    refuse it at the first record)."""
    from simulation import logic_nodes as ln
    from simulation.entities import REGISTRY
    from simulation.sensor_accessor import CHANNEL_ORDER
    nodes = {n for n, c in REGISTRY.items() if getattr(c, "LOGIC_NODE", False)}
    assert nodes <= set(us.NODE_KINDS)
    for n, c in REGISTRY.items():
        if getattr(c, "SENSOR", False):
            assert (getattr(c, "CHANNEL_NAME", None) in us.sensor_channels()
                    or n in ("clock", "sensor_motion")), n
    assert us.sensor_channels() == tuple(c.value for c in CHANNEL_ORDER)
    assert set(us.comparators()) == set(ln._COMPARATORS)


def test_filter_range_twin_names_the_operand():
    """Property: the host twin of the LOGIC stage's head names a filter whose
    ema, input or k is one past the proved range and passes it at the bound.
    Breaks if a bound moves without its twin."""
    from simulation import logic_nodes as ln
    from types import SimpleNamespace
    L = uk.FILTER_LIM
    inst = SimpleNamespace(ordinal=7, id="f", class_name="filter", fields={})
    for ema, x, k, bad in ((L, L, 62, set()), (L + 1, 0, 3, {"ema"}),
                           (0, -L - 1, 3, {"in"}), (0, 0, 63, {"k"})):
        f = ln.FilterRuntime(inst, 1, 0, k)
        f.ema = ema
        fake = SimpleNamespace(entities=[inst], _logic_nodes=[f])
        rec = uk._records_of(us.NODE_TABLE, uk.node_rows(fake))
        ins = uk._side_of(us.NODE_TABLE, uk.node_rows(fake), "ins")
        bus = uk._records_of(us.SIGNAL_TABLE, [us.SignalRow(
            ordinal=0, name="value", pub=x, stg=0, node=False, emit="none", emit_ref=0)])
        assert {n for n, *_ in uk.filter_range(rec[0], ins, bus)} == bad
