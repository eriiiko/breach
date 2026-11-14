"""#185 -- every hit and every stagger carries the SIM's own direction.

``DamagePacket`` / ``UnitHitEvent`` carry ``dir_x_q`` / ``dir_y_q``: the
direction the hit TRAVELLED, a Q16 tile vector (x right, y down), set by
each producer from sim state in integers; ``StaggerEvent`` carries the
overflow of the sway that staggered the unit. None of it is hashed: the
synced event record reads named fields only. Properties only; every test
names the property it protects and the change that must break it.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_hit_direction.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.unit_fixed import quantize_scalar  # noqa: E402
from simulation import unit_impulse as ui  # noqa: E402
from simulation.combat import fire_beam, fire_burst, melee_blow  # noqa: E402
from simulation.events import StaggerEvent, UnitHitEvent  # noqa: E402
from simulation.exchange import (apply_blast_damage,  # noqa: E402
                                 apply_environmental_damage)
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.turning import PI_Q16, place  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.unit_fixed import FP_ONE as ONE  # noqa: E402

EAST, WEST = 0, PI_Q16


def _arena(h=48, w=48):
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    return LevelData(name="hitdir", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."))


def _sim():
    sim = Simulation(_arena(), seed=1, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _unit(sim, x, y, facing=EAST, team=0, name="M"):
    u = sim.get_unit(sim.add_unit(Unit(name, x=x, y=y, team=team)))
    place(u, facing)
    u.current_hp = 10_000.0                      # nothing here may die
    return u


def _hits(events, uid):
    return [e for e in events if isinstance(e, UnitHitEvent) and e.unit_id == uid]


def _angle(dx, dy):
    return math.atan2(dy, dx)


# ---------------------------------------------------------------------------
# producers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("east", [True, False])
def test_a_bullet_hit_travels_along_its_flight(east):
    """PROPERTY: a round's hit carries its flight direction -- the march's
    own Q16 step: fired east the vector is (ONE, 0), fired west (-ONE, 0).

    BREAKS IF: the bullet packet drops the direction, carries the shooter ->
    target bearing of another convention (y up), or its sign is flipped."""
    sim = _sim()
    if east:
        s, t, aim, tx = _unit(sim, 4, 19, name="S"), _unit(sim, 14, 19, name="T"), 0.0, 30
    else:
        s, t, aim, tx = (_unit(sim, 24, 19, WEST, name="S"),
                         _unit(sim, 14, 19, name="T"), math.pi, 1)
    events = []
    fire_burst(sim.gmap, sim.units, s, s.center_tile_x(), 20, tx, 20, sim.tick, [],
               0.0, sim.draws, events, weapon=sim.weapons_tables.weapons.by_name["lr50"],
               spread_deg=0.0, aim_q=quantize_scalar(aim))
    hits = _hits(events, t.id)
    assert len(hits) == 1 and hits[0].source == "bullet"
    sign = 1 if east else -1
    assert (hits[0].dir_x_q, hits[0].dir_y_q) == (sign * ONE, 0)


def test_a_diagonal_bullet_hit_points_along_the_shot():
    """PROPERTY: off the axes too, the hit's vector is the flight's angle
    (screen convention, y down): a round fired toward the lower right
    carries a vector with both components positive at that bearing.

    BREAKS IF: one component is dropped or the y axis is flipped."""
    sim = _sim()
    s, t = _unit(sim, 4, 4, name="S"), _unit(sim, 14, 14, name="T")
    events = []
    fire_burst(sim.gmap, sim.units, s, 5, 5, 25, 25, sim.tick, [], 0.0, sim.draws,
               events, weapon=sim.weapons_tables.weapons.by_name["lr50"],
               spread_deg=0.0)
    (h,) = _hits(events, t.id)
    assert h.dir_x_q > 0 and h.dir_y_q > 0
    assert abs(_angle(h.dir_x_q, h.dir_y_q) - math.pi / 4) < 1e-3


def test_a_beam_hit_travels_along_the_beam():
    """PROPERTY: a hitscan beam's hit carries the beam's own step, as a
    round's does (fired west: (-ONE, 0)).

    BREAKS IF: the laser packet carries no direction or the wrong sign."""
    sim = _sim()
    s, t = _unit(sim, 24, 19, WEST, name="S"), _unit(sim, 10, 19, name="T")
    events = []
    fire_beam(sim.gmap, sim.units, s, 24, 20, 1, 20, sim.tick, [], 0.0, sim.draws,
              events, weapon=sim.weapons_tables.weapons.by_name["lance_3"],
              spread_deg=0.0, aim_q=quantize_scalar(math.pi))
    (h,) = _hits(events, t.id)
    assert h.source == "laser"
    assert (h.dir_x_q, h.dir_y_q) == (-ONE, 0)


def test_a_melee_blow_travels_from_striker_to_victim():
    """PROPERTY: a blow's hit points from the striker's centre to the
    victim's (Q16, exact): striker west of the victim -> (+3 tiles, 0);
    striker below it -> (0, -3 tiles).

    BREAKS IF: the vector is victim -> striker, or the melee packet drops
    it."""
    sim = _sim()
    t = _unit(sim, 20, 20, name="T")
    weapon = sim.weapons_tables.weapons.by_name["rifle_butt"]
    for (ax, ay), want in (((17, 20), (3 * ONE, 0)), ((20, 23), (0, -3 * ONE))):
        a = _unit(sim, ax, ay, name="A")
        events = []
        melee_blow(a, t, sim.draws, events, weapon, None, snap_facing=False)
        (h,) = _hits(events, t.id)
        assert h.source == "melee" and h.attacker_id == a.id
        assert (h.dir_x_q, h.dir_y_q) == want


def test_a_zombie_bite_travels_from_the_biter_to_its_prey():
    """PROPERTY: the zombie's bite (its own law, ``ai_zombie``) points from
    the biter's centre to the prey's: a zombie east of the marine bites
    along -x.

    BREAKS IF: the bite's packet carries no direction or the reversed one."""
    sim = _sim()
    m = _unit(sim, 5, 9, name="M")
    z = _unit(sim, 8, 9, team=ZOMBIE_TEAM, name="Z")
    z.zombie_activated = True
    sim.step()
    (h,) = _hits(sim.tick_events, m.id)
    assert h.source == "melee" and h.attacker_id == z.id
    assert (h.dir_x_q, h.dir_y_q) == (-3 * ONE, 0)


def test_an_explosion_hit_points_from_the_detonation_to_the_victim():
    """PROPERTY: a blast's direct damage points from the detonation tile to
    the victim's centre tile, the geometry its falloff reads -- a unit
    north-east of the blast gets a vector up and to the right (y down:
    negative y), and a unit whose centre tile IS the detonation tile gets
    (0, 0).

    BREAKS IF: the vector is reversed (victim -> blast), the axes are
    swapped, or a blast on the victim's own tile invents a direction."""
    a = Unit("A", x=24, y=14, team=0)            # centre tile (25, 15)
    b = Unit("B", x=19, y=19, team=0)            # centre tile (20, 20)
    for i, u in enumerate((a, b)):
        u.id = i + 1
        u.current_hp = 10_000.0
    events = []
    apply_blast_damage([a, b], 20, 20, 12, 200, events=events, attacker_id=-1)
    (ha,) = _hits(events, a.id)
    (hb,) = _hits(events, b.id)
    assert (ha.dir_x_q, ha.dir_y_q) == (5 * ONE, -5 * ONE)
    assert (hb.dir_x_q, hb.dir_y_q) == (0, 0)


class _HeatStub:
    def __init__(self, h, w):
        self.heat = np.zeros((h, w), dtype=np.int32)


def test_a_heat_hit_carries_no_direction():
    """PROPERTY: heat damage (per tick, from the tiles) carries (0, 0): no
    direction, so it never picks a directional flinch.

    BREAKS IF: a heat packet is given a direction."""
    from simulation.exchange import HEAT_SCALE
    stub = _HeatStub(40, 40)
    m = Unit("M", x=10, y=10, team=0)
    m.id = 1
    m.current_hp = 100.0
    for (tx, ty) in m.occupied_tiles():
        stub.heat[ty, tx] = int(50.0 * HEAT_SCALE)
    events = []
    apply_environmental_damage([m], stub, ticks_per_second=24, events=events)
    (h,) = _hits(events, m.id)
    assert h.source == "heat"
    assert (h.dir_x_q, h.dir_y_q) == (0, 0)


# ---------------------------------------------------------------------------
# the stagger
# ---------------------------------------------------------------------------
def test_a_stagger_carries_the_push_that_overflowed_the_sway():
    """PROPERTY: a ``StaggerEvent`` carries the overflow that became the
    pushed velocity on that tick -- exactly the ``pv`` the push added (the
    unit was at rest), along the applied push (a kick east: x > 0, y 0).

    BREAKS IF: the event's vector is the sway left behind, the input
    before the sway took its share, or the reverse direction."""
    sim = _sim()
    u = _unit(sim, 20, 20, name="M")
    ui.add(sim.impulse_table, u, 4000 * u.mass_g, 0)
    events = []
    ui.integrate(sim.impulse_table, u, events)
    (st,) = [e for e in events if isinstance(e, StaggerEvent)]
    assert st.dir_x_q > 0 and st.dir_y_q == 0
    # before the resist works pv off (the reaction delay has just armed)
    assert u.react_ticks > 0 and (st.dir_x_q, st.dir_y_q) == (u.pvx, u.pvy)


def test_a_diagonal_stagger_points_along_the_push():
    """PROPERTY: the stagger's vector has the push's angle (a push to the
    lower left: both components negative x, positive y, at 135 deg in the
    y-down frame).

    BREAKS IF: a component is dropped or swapped."""
    sim = _sim()
    u = _unit(sim, 20, 20, name="M")
    ui.add(sim.impulse_table, u, -4000 * u.mass_g, 4000 * u.mass_g)
    events = []
    ui.integrate(sim.impulse_table, u, events)
    (st,) = [e for e in events if isinstance(e, StaggerEvent)]
    assert st.dir_x_q < 0 < st.dir_y_q
    assert abs(_angle(st.dir_x_q, st.dir_y_q) - 3 * math.pi / 4) < 1e-3


# ---------------------------------------------------------------------------
# no digest moves
# ---------------------------------------------------------------------------
def test_the_direction_is_not_in_the_synced_event_record():
    """PROPERTY: two hit events that differ ONLY in their direction give
    the same synced record (the per-tick unit digest), and the record names
    no direction field -- the direction is render-only, so no pinned
    digest can move with it.

    BREAKS IF: ``_event_record`` starts hashing every field of an event
    (e.g. ``dataclasses.asdict``) or names the direction."""
    from field_ab_harness import _event_record
    a = UnitHitEvent(unit_id=3, damage=10, source="bullet", attacker_id=1,
                     dir_x_q=ONE, dir_y_q=0)
    b = UnitHitEvent(unit_id=3, damage=10, source="bullet", attacker_id=1,
                     dir_x_q=-7, dir_y_q=12345)
    assert _event_record(a) == _event_record(b)
    assert not any("dir" in k for k in _event_record(a))


def test_the_replay_format_records_no_event():
    """PROPERTY: a match recording holds intents and checkpoints, never
    events, so a new event field needs no replay format bump: no recorded
    intent kind is an event.

    BREAKS IF: events join the recording (then a field change is a format
    change and must bump ``FORMAT_VERSION``)."""
    import match_replay
    assert not any("Event" in k for k in match_replay.INTENT_KINDS)
