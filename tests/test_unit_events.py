"""#29 P6.3 -- the event ring's record layout (``simulation.unit_events``), its
generated C++ header, the records <-> host events bridge, and the P6.3 rows of
the device tables (``simulation.unit_tables``) and claims table
(``simulation.unit_kernels``).

Pure Python (no CUDA): the device side is held to these by
``tests/cuda_unit_damage_check.py``.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bp = pytest.importorskip("breach_physics")

from simulation import unit_events, unit_kernels, unit_tables  # noqa: E402
from simulation.events import (CAUSE_NAMES, UnitHitEvent,  # noqa: E402
                               UnitKilledEvent, ShotFiredEvent)


@pytest.fixture(scope="module")
def ship_sim():
    import level_loader
    from simulation.match_session import MatchSession
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    return MatchSession(level, seed=1, breach_physics=bp).sim


def test_the_checked_in_header_is_the_ring_layout():
    """Property: ``cpp/src/unit_events_layout.h`` is exactly the text
    ``unit_events.header_text()`` generates (the fields, the kinds, the causes,
    the damage types, the row cap, the credit channels). Breaks when a field,
    an event kind, a cause or a damage type is added or reordered (or the
    status block, and with it the row cap, re-widened) without regenerating:
    the kernels would write records the host misreads."""
    path = ROOT / unit_events.HEADER_REL
    have = path.read_text(encoding="utf-8") if path.is_file() else ""
    assert have == unit_events.header_text(), (
        f"{unit_events.HEADER_REL} is stale: run "
        f"`C:/Users/steen/anaconda3/envs/data/python.exe "
        f"{unit_events.GENERATOR}` and commit the header")


def test_a_record_is_the_event_the_digest_hashes_and_back():
    """Property: a hit or a kill of every cause, with any attacker and a
    direction, goes to its ring record and back to an event whose digest
    record (``field_ab_harness._event_record``) and direction are the
    original's -- the integer record is lossless. Breaks if a field is
    dropped, a cause's index is taken from another vocabulary, or the damage
    leaves the Q16 grid."""
    from field_ab_harness import _event_record
    for i, cause in enumerate(CAUSE_NAMES):
        for att in (-1, 0, 7):
            hit = UnitHitEvent(unit_id=3 + i, damage=(12345 + i) / 65536.0,
                               source=cause, attacker_id=att,
                               dir_x_q=-70000 + i, dir_y_q=5)
            kill = UnitKilledEvent(unit_id=3 + i, killed_by=cause, attacker_id=att)
            for ev in (hit, kill):
                back = unit_events.event_of(unit_events.record_of(ev))
                assert type(back) is type(ev)
                assert _event_record(back) == _event_record(ev)
                assert getattr(back, "dir_x_q", 0) == getattr(ev, "dir_x_q", 0)
                assert getattr(back, "dir_y_q", 0) == getattr(ev, "dir_y_q", 0)
                assert repr(_event_record(back)) == repr(_event_record(ev))


def test_the_ring_refuses_an_event_it_does_not_hold():
    """Property: ``record_of`` refuses (naming the type) an event that is not
    a ring kind, so a slot that appended one shows as a difference instead of
    a silently skipped event. Breaks if the bridge drops unknown events."""
    with pytest.raises(ValueError, match="ShotFiredEvent"):
        unit_events.record_of(ShotFiredEvent(unit_id=1, from_tile=(0, 0),
                                             to_tile=(1, 1)))


def test_the_row_cap_holds_slot_2b_worst_case():
    """Property: one unit's statuses tick emits at most ``row_cap()`` events
    -- a full status block of damage-over-time kinds on a unit that dies of
    them (one hit per status and one kill). Breaks if the row staging were
    sized below the 2b worst case (a legitimate tick would overflow)."""
    from simulation import status as S
    from simulation.unit import Unit
    from simulation.unit_schema import STATUS_STACK_CAP, status_capacity
    u = Unit("U", x=2, y=2, team=0)
    u.id = 1
    u.current_hp = 1.0
    for _ in range(STATUS_STACK_CAP):
        S.apply_status(u, S.POISONED, 0.25, 9, None)
    for kind in (S.BURNING, S.SUFFOCATING, S.REGEN, S.KNOCKED_DOWN,
                 S.IMMOBILIZED, S.STUNNED, S.PARALYZED, S.BLINDED):
        S.apply_status(u, kind, 0.25, 9, None)
    assert len(u.statuses) == status_capacity()
    events = []
    S.tick_statuses([u], events)
    assert not u.alive
    assert len(events) <= unit_events.row_cap()


def test_the_p63_rows_are_the_reference_objects_own(ship_sim):
    """Property: the P6.3 rows of the device tables read the objects the
    reference laws read -- the zombie mitigation, the exposure settings, the
    heat row's dials, the poison law's class values. Breaks if a row
    re-derives its value (from config, or another quantization)."""
    from config import CFG
    from simulation import exchange, exposure, poison
    from simulation.species import ZOMBIE_MITIGATION
    v = unit_tables.values(ship_sim)
    r = lambda name: unit_tables.read(v, name)  # noqa: E731
    assert r("mitig_resist_zombie_q") == tuple(ZOMBIE_MITIGATION.resist_q)
    assert r("expose_k_q") == exposure.get_settings().k_q
    assert (r("heat_gain_q"), r("heat_ambient_q")) == \
        exchange.heat_dials_q(exchange.heat_dials(CFG))
    ps = poison.get_settings()
    assert r("poison_harm_class_q") == ps.class_harm_q  # traits-class: the table's reference value
    assert r("poison_dose_den") == 16777216 * ship_sim._tps
    assert r("heat_tps") == ship_sim._tps


def test_a_species_whose_footprint_is_not_row_major_is_refused(ship_sim, monkeypatch):
    """Property: the device walks a footprint as the row-major square; a
    species whose default offsets are any other order is refused at the
    tables' door, naming the row. Breaks if the check is dropped, so a kernel
    would walk tiles in another order than ``Unit.occupied_tiles`` (the first
    densest poison tile, the first hottest heat tile would differ)."""
    from simulation import species
    human = species.SPECIES_REGISTRY["human"]
    odd = replace(human, default_offsets=tuple(reversed(human.default_offsets)))
    monkeypatch.setitem(species.SPECIES_REGISTRY, "human", odd)
    with pytest.raises(ValueError, match="footprint_rowmajor"):
        unit_tables.values(ship_sim)


def test_the_claims_table_is_the_tick_credit(ship_sim):
    """Property: ``claims_of`` lists exactly the tick's claims, the heat
    channel as CREDIT_HEAT and a gas slice as its index, and ``pad_claims``
    pads the short envs with rows that match no tile. Breaks if a claim is
    lost, a channel mis-coded, or the padding could match a real tile."""
    from simulation.damage import CREDIT_HEAT, tick_credit
    tc = tick_credit(ship_sim.gmap)
    tc.clear()
    try:
        tc.claim(CREDIT_HEAT, [(3, 4), (3, 5)], 7)
        tc.claim(2, [(6, 1)], 9)
        tc.claim(CREDIT_HEAT, [(3, 4)], 8)          # the first claim stays
        rows = unit_kernels.claims_of(ship_sim)
        got = {tuple(int(x) for x in r) for r in rows}
        assert got == {(unit_events.CREDIT_HEAT, 3, 4, 7),
                       (unit_events.CREDIT_HEAT, 3, 5, 7), (2, 6, 1, 9)}
        pad = unit_kernels.pad_claims([rows, rows[:1]])
        assert pad.shape == (2, 3, 4)
        assert all(int(pad[1, i, 0]) == unit_events.CREDIT_PAD for i in (1, 2))
    finally:
        tc.clear()
