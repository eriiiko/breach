"""A payload charge owned by the tests, not by config.toml.

Erik's ruling (2026-10-03): tests must never break on retuning a config row.
``[payloads.frag_standard]`` is shared by the hand grenade and the 40 mm
round and Erik keeps retuning it live, so any test that needs a payload with
FIXED numeric properties (to assert an exact blast falloff, an exact wall
chew, "this payload deposits energy") must own its own charge instead of
reading — or worse, pinning a literal copy of — a shipped row.

``TEST_CHARGE`` is a :class:`~simulation.weapons.PayloadDef` built in code,
never a `[payloads.*]` config row, so nothing Erik does in config.toml can
ever move its numbers. ``install_test_charge`` wires it into the live
code paths (the payload table + one ammo row's payload ref) for the
duration of a single test, via ``monkeypatch`` so nothing leaks to the next
test.
"""
from __future__ import annotations

from simulation.weapons import PayloadDef, get_tables

# Fixed values the tests own. Chosen so a round carrying this payload still
# behaves like a real blast (nonzero pressure -> an atmosphere/wave_source/
# temperature energy deposit; nonzero wall_damage/unit_damage -> a real
# falloff to assert against) but with NO heat (heat_amount 0.0) and no gas,
# so it never ignites and never emits gas.
TEST_CHARGE = PayloadDef(
    name="test_charge",
    radius=5,
    pressure=10.0,
    wall_damage=200,
    unit_damage=60,
    clear_smoke=True,
    emit_blast_smoke=True,
    heat_amount=0.0,
    heat_radius=0.0,
)


def install_test_charge(monkeypatch, ammo_name="40mm_frag"):
    """Register :data:`TEST_CHARGE` in the shared payload table and rebind
    ``[ammo.<ammo_name>]``'s payload ref onto it, for the duration of the
    calling test (``monkeypatch`` undoes both at teardown, so the shared
    module-level tables — :func:`simulation.weapons.get_tables` — are left
    exactly as they were for the next test). Returns :data:`TEST_CHARGE`.
    """
    t = get_tables()
    monkeypatch.setitem(t.payloads.by_name, "test_charge", TEST_CHARGE)
    ammo = t.ammo.by_name[ammo_name]
    monkeypatch.setattr(ammo, "payload", "test_charge")
    return TEST_CHARGE
