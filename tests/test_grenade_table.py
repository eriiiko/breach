"""#29 P7b -- the grenade table's host side (CPU, no device): the schema
``unit_schema.GRENADE_TABLE`` over ``Simulation.projectiles``, the P7b device
table rows and the device door ``unit_kernels.refuse_wego``.
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation import object_tables as ot
from simulation import unit_kernels, unit_schema, unit_tables
from simulation.combat import Projectile
from simulation.orders import ORDER_GRENADE

GT = unit_schema.GRENADE_TABLE


def _sim():
    from pathlib import Path

    from level_loader import LevelData
    from simulation import Simulation
    from simulation.materials import MAT_AIR, MAT_HULL
    from simulation.ruleset import ContinuousRealtime
    tm = np.full((30, 40), MAT_HULL, dtype=np.int32)
    tm[1:29, 1:39] = MAT_AIR
    level = LevelData(name="p7b_cpu", version="3", path=Path("."), tilemap=tm,
                      tile_size_m=1.0, diffuse_path=Path("."), entities=[])
    sim = Simulation(level, seed=3, enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def test_unpack_of_pack_is_the_host_grenade_list_attribute_for_attribute():
    """PROPERTY: every grenade a throw creates -- every kind, in flight, landed,
    detonated -- comes back from ``unpack(pack(list))`` with exactly its
    attributes (``vars``), in list order. BREAKS IF a Projectile attribute the
    flight or the fuse-out reads is missing from the schema, a column's encode
    and decode disagree, or the constructor ``build`` derives a value the row
    does not hold."""
    from simulation.unit import Unit
    sim = _sim()
    kinds = sim.weapons_tables.hand_grenade_rows
    for i, _name in enumerate(kinds):
        u = Unit(f"T{i}", x=4.0 + 5 * i, y=20.0, team=0)
        sim.add_unit(u)
        u.grenade_counts = [2] * len(kinds)
        sim._release_grenade(u, i, 10 + 3 * i, (3 + i) << 16, 0, -65536)
    sim.projectiles[1].detonated = True
    for t in (0, 2, 9):
        for p in sim.projectiles:
            p.update_position(t)
        rows, n = ot.pack(GT, sim.projectiles)
        back = ot.unpack(GT, rows, n)
        assert [vars(p) for p in back] == [vars(p) for p in sim.projectiles]
    assert len({p.ammo_name for p in sim.projectiles}) == len(kinds)


def test_the_wego_grenade_round_trips_and_the_door_refuses_it():
    """PROPERTY: a WEGO-order grenade (``fuse_ticks`` None, a float
    ``fuse_seconds``) is still exactly representable, and the device door
    refuses it by name; a fuse equal to the NO_TIMER sentinel and a
    ``fuse_seconds`` that is not an exact Q16 value are refused at the
    encode. BREAKS IF the sentinel collides silently with a real fuse, a
    non-Q16 float is rounded quietly, or the door lets a WEGO grenade through."""
    sim = _sim()
    p = Projectile(ORDER_GRENADE, 3, 3, 9.5, 3.5, 2.5, 7, owner_id=4)
    rows, n = ot.pack(GT, [p])
    assert vars(ot.unpack(GT, rows, n)[0]) == vars(p)
    sim.projectiles.append(p)
    with pytest.raises(unit_kernels.ObjectSlotRefused, match="WEGO"):
        unit_kernels.refuse_wego([sim])
    bad = Projectile(ORDER_GRENADE, 3, 3, 9.5, 3.5, 0.0, 7, fuse_ticks=1)
    bad.fuse_ticks = unit_schema.NO_TIMER
    with pytest.raises(ValueError, match="NO_TIMER"):
        ot.pack(GT, [bad])
    odd = Projectile(ORDER_GRENADE, 3, 3, 9.5, 3.5, 0.1, 7)
    with pytest.raises(ValueError, match="Q16"):
        ot.pack(GT, [odd])


def test_the_device_door_refuses_a_ruleset_that_drives_the_units():
    """PROPERTY: the continuous path passes the grenade / charge slots' device
    door and a ruleset that drives the units (OnePhaseWEGO: its timeline
    runs ``charges.detonate_owned``) is refused naming detonate_owned.
    BREAKS IF the door stops checking ``drives_units``."""
    from simulation.ruleset import OnePhaseWEGO
    sim = _sim()
    unit_kernels.refuse_wego([sim])
    sim.ruleset = OnePhaseWEGO()
    with pytest.raises(unit_kernels.ObjectSlotRefused, match="detonate_owned"):
        unit_kernels.refuse_wego([sim])


def test_the_p7b_table_rows_are_what_the_flight_and_the_fuse_out_read():
    """PROPERTY: the device tables' P7b rows hold, per throw kind, the round a
    throw creates and that round's ``Projectile.travel_speed_q``, and per
    ammo row the index of ``payload_for_ammo``'s row (-1 for a round with
    none). BREAKS IF a row re-derives a value instead of reading the object
    the reference law reads (a kind appended to the family, a payload
    renamed, a speed re-quantized)."""
    sim = _sim()
    vec = unit_tables.values(sim)
    wt = sim.weapons_tables
    ammo = list(wt.ammo.by_name)
    names = list(wt.payloads.by_name)
    for k, name in enumerate(wt.hand_grenade_rows):
        p = Projectile(ORDER_GRENADE, 0, 0, 1, 1, 0.0, 0, ammo_name=name)
        assert unit_tables.read(vec, "hand_grenade_ammo_row")[k] == ammo.index(name)
        assert unit_tables.read(vec, "hand_grenade_speed_q")[k] == p.travel_speed_q
    rows = unit_tables.read(vec, "ammo_payload_row")
    for i, name in enumerate(ammo):
        want = (names.index(wt.payload_for_ammo(name).name)
                if wt.ammo.by_name[name].payload else -1)
        assert rows[i] == want
    assert -1 in rows and any(r >= 0 for r in rows)
