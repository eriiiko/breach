"""#29 P6.4b -- the per-env serial applier's one table (``simulation.applier``):
its generated C++ header, its planes against the GameMap they point into, its
books against the host quantities they twin, and the effect list's packing.

Pure Python (no CUDA): the device side is held to these by
``tests/cuda_unit_topology_check.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bp = pytest.importorskip("breach_physics")

from simulation import applier  # noqa: E402


@pytest.fixture(scope="module")
def ship_gmap():
    import level_loader
    from simulation import Simulation
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    return Simulation(level, seed=1, breach_physics=bp,
                      enable_recorder=False).gmap


def test_the_checked_in_header_is_the_applier_table():
    """Property: ``cpp/src/applier_layout.h`` is exactly the text
    ``applier.header_text()`` generates (the record, the kinds, the books, the
    planes' order, the clears). Breaks when a kind, a book or a plane is added
    or reordered without regenerating: the device would read the launch's
    pointer table or write the books in another order than the host packs."""
    path = ROOT / applier.HEADER_REL
    have = path.read_text(encoding="utf-8") if path.is_file() else ""
    assert have == applier.header_text(), (
        f"{applier.HEADER_REL} is stale: run `C:/Users/steen/anaconda3/envs/"
        f"data/python.exe {applier.GENERATOR}` and commit the header")


def test_every_plane_is_the_gamemap_field_it_names(ship_gmap):
    """Property: every ``applier.PLANES`` row is a GameMap array attribute in
    the row's dtype with the row's per-cell layout (``(h, w)``, ``(h, w, c)``
    or the gas planes ``(G, h, w)``), C-contiguous -- so the device's typed
    rebase of the launch's pointer table reads the host's bytes. Breaks if a
    plane's dtype or shape moves on the GameMap (e.g. ``fuel_recip`` to
    int32) without the table."""
    g = ship_gmap
    h, w = g.material.shape
    for name, dt, cw in applier.PLANES:
        a = getattr(g, name)
        assert isinstance(a, np.ndarray), name
        assert a.dtype == np.dtype(dt), (name, a.dtype, dt)
        want = ((g.gas.shape[0], h, w) if cw == "G" else
                (h, w) if cw == 1 else (h, w, int(cw)))
        assert a.shape == want, (name, a.shape, want)
        assert a.flags.c_contiguous, name
    for name, dt in applier.CLEAR_PLANES:
        a = getattr(g, name)
        assert a.dtype == np.dtype(dt) and a.shape == (h, w), name


def test_every_book_twins_a_host_quantity(ship_gmap):
    """Property: each book with a host twin names a GameMap integer attribute
    or a key of a GameMap dict of integers (absent = 0), so the shadow harness
    can take the host's delta and the chained harness land the device's.
    Breaks if a twin is renamed on the GameMap (``n_destruction_seed_sum``,
    the seam channels, ``_tile_serial``)."""
    g = ship_gmap
    for b in applier.BOOKS:
        if b.host is None:
            continue
        attr = b.host.split(":", 1)[0]
        assert hasattr(g, attr), (b.name, b.host)
        assert isinstance(applier.host_book(g, b.host), int), b.name


def test_the_non_resident_planes_are_exactly_those_outside_the_resident_set():
    """Property: ``non_resident_planes`` names a plane iff it is not a
    resident field (``GameMap._RESIDENT_FIELD_NAMES``) -- the list P8b must
    bring onto the device. Breaks if a plane joins or leaves the resident set
    and the list does not follow."""
    from simulation.gamemap import GameMap
    res = set(GameMap._RESIDENT_FIELD_NAMES)
    nr = set(applier.non_resident_planes())
    for n in applier.PLANE_NAMES + applier.CLEAR_NAMES:
        assert (n in nr) == (n not in res), n


def test_a_list_past_its_capacity_keeps_its_first_entries_and_its_count():
    """Property: ``pack`` keeps each env's first ``cap`` records in order and
    its REQUESTED count (which the device compares with ``cap`` to count the
    dropped), padding the rest with no-record rows. Breaks if the packing
    truncates the count (the overflow would go uncounted) or reorders."""
    recs = applier.destroy_records([(i, i + 1) for i in range(10)])
    out, counts = applier.pack([recs, recs[:3], recs[:0]], cap=4)
    assert out.shape == (3, 4, applier.F)
    assert counts.tolist() == [10, 3, 0]
    assert np.array_equal(out[0], recs[:4])
    assert np.array_equal(out[1, :3], recs[:3]) and not out[1, 3].any()
    assert not out[2].any()
    assert (recs[:, 0] == applier.KIND_DESTROY_WALL).all()


# ---------------------------------------------------------------------------
# #29 P7a.1: the effect list's kinds and its ONE record function
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ship_sim():
    import level_loader
    from simulation import Simulation
    from simulation.unit import Unit
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    sim = Simulation(level, seed=1, breach_physics=bp, enable_recorder=False)
    for i in range(2):
        sim.add_unit(Unit(f"U{i}", x=10.0 + 4 * i, y=10.0, team=i))
    return sim


def test_every_kind_fits_the_record_and_names_its_fields_once():
    """Property: each kind's fields fit the record's arg slots, are unique
    within the kind, and ``kind_field`` / ``make_record`` place them where the
    generated header says (column 1 + position). Breaks if a kind grows past
    ARG_SLOTS without widening the record, or the two placements disagree."""
    text = applier.header_text()
    for kind, fields in applier.KIND_FIELDS:
        assert len(fields) <= applier.ARG_SLOTS, kind
        assert len(set(fields)) == len(fields), kind
        vals = {f: 100 + i for i, f in enumerate(fields)}
        rec = applier.make_record(kind, **vals)
        assert rec[0] == applier.KINDS.index(kind) + 1
        for f in fields:
            col = applier.kind_field(kind, f)
            assert rec[col] == vals[f]
            assert f"APPLIER_{kind}_{f.upper()} = {col};" in text
    with pytest.raises(ValueError):
        applier.make_record("HIT", row=0)


def test_the_unapplied_kinds_and_the_stages_are_rows_of_the_table():
    """Property: every kind the applier refuses to apply is a kind of the
    table, every stage has a landing mode, and the resolve stage lands its
    HITs while the topology stage does not (decision 2: the mode is per
    (stage, kind)). Breaks if a refused kind is renamed away from the table,
    or the resolve stage stops landing."""
    for k in applier.UNAPPLIED_KINDS:
        assert k in applier.KINDS
    modes = dict(applier.STAGES)
    assert modes["RESOLVE"] is True and modes["TOPOLOGY"] is False
    assert applier.STAGE_NAMES.index("RESOLVE") == applier.STAGE_RESOLVE


def test_the_host_effects_become_records_carrying_what_each_call_receives(ship_sim):
    """Property: ``record_of`` turns each SimultaneousEffects entry into the
    record of the call it stands for -- a HIT's packet at its door (Q16
    amount, AP quantized, the attacker normalized, the cause's index), a
    STATUS's quantized magnitude and None source as -1, a PUSH's five
    integers, ``chew_wall`` as CHEW with its per-call quantized damage (and
    ``live`` 0 for no damage), ``execute_payload`` as PAYLOAD -- and refuses by
    name FACE, any other world call and an effect on another sim's objects.
    Breaks if a field is dropped, re-quantized differently from the
    reference's door, or a world call is recognised by anything but its
    function."""
    from simulation import combat, payloads, unit_fixed, wall_fixed
    from simulation.damage import DamagePacket
    from simulation.events import CAUSE_NAMES
    from simulation.simultaneous import SimultaneousEffects
    sim = ship_sim
    a, t = sim.units[0], sim.units[1]
    fx = SimultaneousEffects()
    fx.hit(t, DamagePacket(amount=7.25, dtype=2, attacker_id=a.id, ap=1.5,
                           expose=12345, dir_x_q=-3, dir_y_q=4),
           sim.tick_events, source="laser")
    fx.status(t, 2, magnitude=0.5, duration_ticks=9, source_id=None)
    fx.push(sim.impulse_table, t, 11, -12, 13, 70000)
    fx.world(combat.chew_wall, sim.gmap, 3, 4, 12.3)
    fx.world(combat.chew_wall, sim.gmap, 3, 4, 0)
    payload = next(iter(sim.weapons_tables.payloads.by_name.values()))
    fx.world(payloads.execute_payload, sim.gmap, sim.edit_queue, sim.units, 5, 6,
             payload, sim.draws, events=sim.tick_events, kind="grenade",
             owner_id=a.id, approach=None)
    recs = [tuple(int(x) for x in r) for r in applier.records_of(sim, fx._pending)]
    f = applier.kind_field
    hit = recs[0]
    assert hit[0] == applier.KIND_HIT and hit[f("HIT", "row")] == 1
    assert hit[f("HIT", "amount_q")] == unit_fixed.quantize_scalar(7.25)
    assert hit[f("HIT", "ap_q")] == unit_fixed.quantize_scalar(1.5)
    assert hit[f("HIT", "cause")] == CAUSE_NAMES.index("laser")
    assert (hit[f("HIT", "expose")], hit[f("HIT", "dir_x_q")]) == (12345, -3)
    st = recs[1]
    assert st[f("STATUS", "magnitude_q")] == unit_fixed.quantize_scalar(0.5)
    assert st[f("STATUS", "source")] == -1 and st[f("STATUS", "duration")] == 9
    assert recs[2][1:6] == (1, 11, -12, 13, 70000)
    assert recs[3][f("CHEW", "damage_q")] == wall_fixed.quantize_scalar(12.3)
    assert recs[3][f("CHEW", "live")] == 1 and recs[4][f("CHEW", "live")] == 0
    pl = recs[5]
    assert pl[0] == applier.KIND_PAYLOAD and pl[f("PAYLOAD", "has_approach")] == 0
    assert pl[f("PAYLOAD", "kind")] == applier.PAYLOAD_KINDS.index("grenade")
    for make in (lambda fx: fx.face(a, 3), lambda fx: fx.world(print, 1),
                 lambda fx: fx.hit(t, DamagePacket(amount=1.0, dtype=0), [],
                                   source="bullet")):
        bad = SimultaneousEffects()
        make(bad)
        with pytest.raises(applier.EffectRefused):
            applier.records_of(sim, bad._pending)


def test_only_synced_events_count_as_ring_events():
    """Property: ``unit_events.is_synced`` holds exactly for the event types
    the ring holds (the hashed unit hit / kill), so the shadow harness drops
    the render-only ones (MeleeEvent, ShotFired, ...) and nothing else.
    Breaks if a synced type is dropped or a render-only one compared."""
    from simulation import unit_events
    from simulation.events import MeleeEvent, UnitHitEvent, UnitKilledEvent
    assert unit_events.is_synced(UnitHitEvent(unit_id=1, damage=1.0, source="melee"))
    assert unit_events.is_synced(UnitKilledEvent(unit_id=1, killed_by="melee"))
    assert not unit_events.is_synced(MeleeEvent(unit_id=1, target_id=2, ticks=3))
