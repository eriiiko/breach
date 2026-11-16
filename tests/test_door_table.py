"""#29 P6b.1 -- the door table (``unit_schema.DOOR_TABLE``, the first entity
table) and its tile side table, on the host (no CUDA): the schema, the side
spans of ``simulation.object_tables``, the record functions and the entity
rows read from the table. The device side is tests/cuda_unit_doors_check.py.
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

DOOR_LEVELS = ("playground", "test_level", "door_test", "airlock_demo")


def _sim(name):
    from simulation import Simulation
    return Simulation(level_loader.load(name, str(_ROOT / "levels")), seed=1,
                      enable_recorder=False)


def _mutate(sim):
    """Runtime state a door carries across ticks, varied door by door."""
    for k, d in enumerate(sim._doors):
        d.want_open = bool(k % 2)
        d.state = k % 3
        d.alive = d.state != 2
        d.hp = [((k + 1) * 4099 + i * 17) * (1 if i % 2 else -1) for i in range(len(d.hp))]


@pytest.mark.parametrize("name", DOOR_LEVELS)
def test_door_table_round_trip_is_the_doors(name):
    """Property: unpack(pack(sim._doors)) over the door table AND its tile
    side table rebuilds every door's ordinal, alive, state, want_open, span
    (row-major, the hp_i <-> tile pin) and hp. Breaks if a column's encode /
    decode, the side span's start / count bookkeeping or the tile rows'
    order changes."""
    sim = _sim(name)
    _mutate(sim)
    recs, tiles = unit_kernels.doors_of(sim), unit_kernels.door_tiles_of(sim)
    back = unit_kernels.door_objects(recs, tiles)
    assert len(back) == len(sim._doors) > 0
    for d, b in zip(sim._doors, back):
        assert (int(d.ordinal), bool(d.alive), int(d.state), bool(d.want_open),
                [tuple(t) for t in d.span], list(d.hp)) == \
            (b.ordinal, b.alive, b.state, b.want_open, b.span, b.hp)


@pytest.mark.parametrize("name", DOOR_LEVELS)
def test_door_spans_point_into_the_side_table_in_door_order(name):
    """Property: each door row's tile span is the running sum of the spans
    before it and its rows are that door's span tiles with their hp, so the
    side table is the doors' spans concatenated in ordinal order. Breaks if a
    span's start is not the running sum or the side rows' order changes."""
    sim = _sim(name)
    _mutate(sim)
    recs, tiles = unit_kernels.doors_of(sim), unit_kernels.door_tiles_of(sim)
    lay = ot.table_layout(unit_schema.DOOR_TABLE)
    s, c = lay.column("tile_start").offset, lay.column("tile_count").offset
    run = 0
    for k, d in enumerate(sim._doors):
        assert int(recs[k, s]) == run and int(recs[k, c]) == len(d.span)
        for i, ((ty, tx), hp) in enumerate(zip(d.span, d.hp)):
            assert tuple(int(v) for v in tiles[run + i, :3]) == (ty, tx, hp)
        run += len(d.span)
    assert run == tiles.shape[0]


@pytest.mark.parametrize("name", DOOR_LEVELS)
def test_door_entity_rows_from_the_table_are_the_serializers(name):
    """Property: a door row's ENTITY_SECT_V1 part read from the table (the
    free alive row, then state, want_open, hp_i) equals what the one
    serializer writes for that door. Breaks if the table loses a runtime row
    or reads an hp from another tile."""
    from simulation.entities import door as door_schema
    sim = _sim(name)
    _mutate(sim)
    recs, tiles = unit_kernels.doors_of(sim), unit_kernels.door_tiles_of(sim)
    for k, d in enumerate(sim._doors):
        want = (("alive", 1 if d.alive else 0),) + door_schema.door.runtime_digest_rows(d)
        assert unit_kernels.door_entity_rows(recs[k], tiles) == want


def test_doors_land_on_the_sims_own_runtimes():
    """Property: landing table rows writes alive / state / want_open / hp
    onto the SAME door objects (their identity kept: the observation and the
    entity serializer hold them), and rows that are another door (an
    ordinal or a span that differ, another count) are refused by name.
    Breaks if the landing builds new objects or skips an attribute."""
    sim = _sim("playground")
    objs = list(sim._doors)
    donor = _sim("playground")
    _mutate(donor)
    unit_kernels.doors_to_host(sim, unit_kernels.doors_of(donor),
                               unit_kernels.door_tiles_of(donor))
    assert all(a is b for a, b in zip(objs, sim._doors))
    assert unit_kernels.doors_of(sim).tolist() == unit_kernels.doors_of(donor).tolist()
    assert unit_kernels.door_tiles_of(sim).tolist() == \
        unit_kernels.door_tiles_of(donor).tolist()
    bad = unit_kernels.doors_of(donor)
    bad[0, ot.table_layout(unit_schema.DOOR_TABLE).column("ordinal").offset] += 99
    with pytest.raises(unit_kernels.DoorTableMismatch):
        unit_kernels.doors_to_host(sim, bad, unit_kernels.door_tiles_of(donor))
    with pytest.raises(unit_kernels.DoorTableMismatch):
        unit_kernels.doors_to_host(sim, unit_kernels.doors_of(donor)[:-1],
                                   unit_kernels.door_tiles_of(donor))


def test_a_span_past_its_bound_and_a_table_past_its_capacity_are_refused():
    """Property: one door's span longer than the tile table's capacity is
    refused at its encode, and a door list or a tile list past its table's
    capacity at the pack, each naming what overflowed. Breaks if a capacity
    check is dropped or stops naming its table."""
    class D:
        def __init__(self, n, o=0):
            self.ordinal, self.alive, self.state, self.want_open = o, True, 0, False
            self.span = [(1, j) for j in range(n)]
            self.hp = [65536] * n
    with pytest.raises(ValueError, match="tiles"):
        ot.records(unit_schema.DOOR_TABLE, [D(unit_schema.DOOR_TILE_CAPACITY + 1)])
    with pytest.raises(ot.ObjectTableOverflow, match="'door'"):
        ot.pack(unit_schema.DOOR_TABLE, [D(1, i) for i in range(unit_schema.DOOR_CAPACITY + 1)])
    with pytest.raises(ot.ObjectTableOverflow, match="'door_tile'"):
        ot.pack_sides(unit_schema.DOOR_TABLE, [D(9, i) for i in range(unit_schema.DOOR_CAPACITY)])


def test_a_list_books_host_twin_reads_its_index():
    """Property: a book whose host is a LIST attribute (``trace_clip_books:g``)
    reads element g, and every trace-clip book names its own slice. Breaks if
    host_book stops indexing a list or a slice's book points at another."""
    from simulation import applier
    from simulation.gases import N_GASES

    class G:
        trace_clip_books = [10 * g + 3 for g in range(N_GASES)]
    for g in range(N_GASES):
        assert applier.host_book(G, f"trace_clip_books:{g}") == 10 * g + 3
        row = [b for b in applier.BOOKS if b.name == f"trace_clip_{g}"]
        assert len(row) == 1 and row[0].host == f"trace_clip_books:{g}"
