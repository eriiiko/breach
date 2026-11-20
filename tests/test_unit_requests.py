"""#29 P6.4a -- the request list's record layout (``simulation.unit_requests``),
its generated C++ header, the records <-> host objects bridge, slot 1b's
per-tick inputs, and the P6.4a rows of the device tables.

Pure Python (no CUDA): the device side is held to these by
``tests/cuda_unit_verbs_check.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bp = pytest.importorskip("breach_physics")

from simulation import charges, unit_requests, unit_tables  # noqa: E402
from simulation.unit_schema import CHARGE_CAPACITY  # noqa: E402

ONE = 1 << 16


@pytest.fixture()
def ship_session():
    import level_loader
    from simulation.match_session import MatchSession
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    return MatchSession(level, seed=1, breach_physics=bp)


def test_the_checked_in_header_is_the_request_layout():
    """Property: ``cpp/src/unit_requests_layout.h`` is exactly the text
    ``unit_requests.header_text()`` generates (the fields, the kinds, the row
    cap). Breaks when a field or a kind is added or reordered without
    regenerating: the kernel would write records the host misreads."""
    path = ROOT / unit_requests.HEADER_REL
    have = path.read_text(encoding="utf-8") if path.is_file() else ""
    assert have == unit_requests.header_text(), (
        f"{unit_requests.HEADER_REL} is stale: run "
        f"`C:/Users/steen/anaconda3/envs/data/python.exe "
        f"{unit_requests.GENERATOR}` and commit the header")


def test_the_row_cap_holds_one_request_of_each_kind():
    """Property: a row's staging holds one request of every kind (a unit
    releases at most one grenade, lays at most one charge and flips at most
    one door in one slot 1b). Breaks if a kind is added without widening
    the staging, or the cap shrinks below the kinds."""
    assert unit_requests.ROW_CAP == len(unit_requests.KINDS) == 3
    assert unit_requests.layout().row_cap == unit_requests.ROW_CAP


def test_records_become_the_hosts_objects_and_back(ship_session):
    """Property: a throw record turned into the host's ``Projectile``
    (``apply_to_host``) reads back as the same record (start tile, Q16
    target, fuse, thrown tick, kind, owner); a plant record is offered to
    ``charges.place`` and reads back the same, its ``accepted`` the door's
    answer; a use record flips that door's latch. Breaks if the bridge drops
    or rescales a field, bypasses ``charges.place``, or flips another door."""
    sim = ship_session.sim
    u = [x for x in sim.units if not x.is_zombie][0]
    tick = sim.tick
    throw = unit_requests._blank(unit_requests.KIND_THROW, u.id, tick)
    throw[5:9] = [10 * ONE + 32768 + 123, 12 * ONE + 32768 - 77, 48, 1]
    throw[12] = 1
    # #215: a Q16 start (the hand), not a tile
    throw[unit_requests.FIELDS.index("start_x_q")] = 10 * ONE + 40000
    throw[unit_requests.FIELDS.index("start_y_q")] = 12 * ONE + 7
    n0 = len(sim.projectiles)
    plant = unit_requests.plant_record(charges.PlantedCharge(
        5, 6, "demo_c4", u.id, det_tick=tick + 300, approach=(6, 5)), tick, True)
    recs = [tuple(throw), plant]
    if sim._doors:
        recs.append(unit_requests.use_record(u.id, tick, 0))
        was = sim._doors[0].want_open
    got = unit_requests.apply_to_host(sim, recs)
    assert got["throw"] == 1 and got["plant"] == 1
    assert unit_requests.throw_record(sim, sim.projectiles[n0]) == tuple(throw)
    assert unit_requests.plant_record(sim.planted_charges[-1], tick, True) == plant
    if sim._doors:
        assert sim._doors[0].want_open != was


def test_a_plant_record_must_agree_with_the_charge_tables_door(ship_session):
    """Property: the bridge refuses a plant whose ``accepted`` contradicts
    ``charges.place`` (a full table refuses at ``CHARGE_CAPACITY``, counted in
    ``sim.charges_refused``), so a device that placed a charge the host
    refuses (or the reverse) shows instead of passing. Breaks if the bridge
    appends without asking the door."""
    sim = ship_session.sim
    while unit_requests.charges_live(sim) < CHARGE_CAPACITY:
        charges.place(sim, charges.PlantedCharge(1, 1, "demo_c4", -1, det_tick=10 ** 9))
    ref0 = getattr(sim, "charges_refused", 0)
    rec = unit_requests.plant_record(charges.PlantedCharge(
        2, 2, "demo_c4", 0, det_tick=10 ** 9, approach=(2, 2)), sim.tick, False)
    assert unit_requests.apply_to_host(sim, [rec])["refused"] == 1
    assert sim.charges_refused == ref0 + 1
    bad = rec[:12] + (1,) + rec[13:]
    with pytest.raises(ValueError, match="placed"):
        unit_requests.apply_to_host(sim, [bad])


def test_door_spans_are_the_live_doors_the_use_verb_scans(ship_session):
    """Property: ``door_spans`` lists every LIVE door's span tiles with its
    ordinal in ``sim._doors``, door order then span order (what
    ``verb_gates.use_door`` walks), a dead door absent; ``pad_door_spans``
    pads a short env with ordinal -1. Breaks if a dead door is listed, the
    order changes, or a pad row names a door."""
    sim = ship_session.sim
    spans = unit_requests.door_spans(sim)
    want = [(k, int(ty), int(tx)) for k, d in enumerate(sim._doors) if d.alive
            for (ty, tx) in d.span]
    assert [tuple(int(x) for x in r) for r in spans] == want
    if sim._doors:
        sim._doors[0].alive = False
        assert all(int(r[0]) != 0 for r in unit_requests.door_spans(sim))
        sim._doors[0].alive = True
    pad = unit_requests.pad_door_spans([spans, spans[:0]])
    assert pad.shape[0] == 2 and (pad[1, :, 0] == unit_requests.NO_DOOR).all()
    assert (pad[0, :len(spans)] == spans).all()


def test_the_p64a_table_rows_read_the_reference_objects(ship_session):
    """Property: the P6.4a device-table rows are the reference's own values:
    each weapon row's ``mag_size`` / ``reload_ticks`` in the unit schema's
    weapons-table order, the hand grenade's wind-up, reach clamp and fuse
    range at the sim's tick rate, the plant row's ticks, the charge capacity,
    the shove's impulse / cone / cooldown and the body radius from the map's
    impulse table, the match's ``limit_ticks`` (0 for a sim that plays no
    match). Breaks if a row is re-derived from config, read from another
    table order, or the limit read from anything but the sim's MatchRules."""
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    from simulation.unit_schema import TABLES
    from simulation.weapons import PLANT_WEAPON, fuse_ticks_from_seconds
    sim = ship_session.sim
    v = unit_tables.values(sim)
    W = sim.weapons_tables.weapons.by_name
    names = tuple(TABLES["weapons"]())
    assert unit_tables.read(v, "weapon_mag_size") == tuple(W[n].mag_size for n in names)
    assert unit_tables.read(v, "weapon_reload_ticks") == tuple(W[n].reload_ticks
                                                                for n in names)
    hand = W["hand_grenade"]
    assert unit_tables.read(v, "hand_windup_class_ticks") == hand.throw_windup_ticks
    assert unit_tables.read(v, "hand_max_reach_q") == int(hand.max_throw_range) << 16
    assert unit_tables.read(v, "hand_fuse_lo_ticks") == fuse_ticks_from_seconds(
        hand.fuse_min_seconds, sim._tps)
    assert unit_tables.read(v, "hand_fuse_hi_ticks") == fuse_ticks_from_seconds(
        hand.fuse_max_seconds, sim._tps)
    assert unit_tables.read(v, "plant_class_ticks") == W[PLANT_WEAPON].plant_ticks
    assert unit_tables.read(v, "charge_capacity") == CHARGE_CAPACITY
    it = sim.impulse_table
    assert unit_tables.read(v, "shove_q") == it.shove_q
    assert unit_tables.read(v, "shove_cone_q") == it.shove_cone_q
    assert unit_tables.read(v, "shove_cooldown_ticks") == it.shove_cooldown_ticks
    assert unit_tables.read(v, "impulse_r_body_q") == it.r_body_q
    assert unit_tables.read(v, "match_limit_ticks") == ship_session.rules.limit_ticks(sim)
    bare = Simulation(ship_session.level, seed=1, breach_physics=None,
                      enable_recorder=False, ruleset=ContinuousRealtime())
    assert unit_tables.read(unit_tables.values(bare), "match_limit_ticks") == 0
