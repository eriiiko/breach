"""#119 P1 — team semantics, roster slots, loadouts (properties, not snapshots).

What each test protects, and the change that must break it:

- a unit's roster slot is fixed at spawn and never reused, even after deaths
  and further spawns — breaks if ``add_unit`` derives the slot from the LIVING
  members (the RL observation layout would shift under a policy);
- a converted marine joins ``ZOMBIE_TEAM`` and is an enemy of its old squad,
  a friend of the zombies — breaks if conversion keeps the old team, or if
  hostility is read from a team literal again;
- two marine teams (no zombies) see and shoot each other and the zombie AI
  never touches them — breaks if any "team 1 means zombie" site survives;
- a loadout row arms a unit; the door refuses an unknown weapon / profile;
- a legacy level ``[[spawn]] team = 1`` still produces a zombie, on
  ``ZOMBIE_TEAM`` — breaks if the loader stops mapping the legacy code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import LEGACY_ZOMBIE_TEAM, load  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import engagement, vision  # noqa: E402
from simulation.ai_zombie import convert_marines_to_zombies  # noqa: E402
from simulation.environment import EnvironmentProfile  # noqa: E402
from simulation.events import ShotFiredEvent  # noqa: E402
from simulation.loadouts import LoadoutTables, get_tables  # noqa: E402
from simulation.orders import ORDER_MOVE_ATTACK, Order  # noqa: E402
from simulation.unit import ZOMBIE_TEAM, Unit  # noqa: E402


def _sim(level="playground", seed=7):
    return Simulation(load(level), seed=seed, breach_physics=bp,
                      enable_recorder=False)


def _first_marine_spawn(level):
    return next(s for s in level.spawns if s.team != ZOMBIE_TEAM)


def _facing_pair(sim, ax, ay, fp=3, lo=5, hi=9):
    """A second open footprint east/west/north/south of (ax, ay) with LOS
    between the centres, inside rifle range."""
    g = sim.gmap
    for d in range(lo, hi + 1):
        for bx, by in ((ax + d, ay), (ax - d, ay), (ax, ay + d), (ax, ay - d)):
            if not g.is_passable_block(by, bx, fp):
                continue
            c = fp // 2
            if g.has_los(ay + c, ax + c, by + c, bx + c):
                return bx, by
    raise AssertionError("no open facing footprint near the marine spawn")


# ---------------------------------------------------------------------------
# Constant + legacy door
# ---------------------------------------------------------------------------
def test_zombie_team_cannot_collide_with_a_marine_team():
    # Marine teams are small ints counted from 0; the legacy level code is the
    # ONLY place 1 means zombie.
    assert ZOMBIE_TEAM not in range(0, 16)
    assert LEGACY_ZOMBIE_TEAM == 1


def test_legacy_level_team_1_spawns_zombies_on_zombie_team():
    level = load("playground")
    raw_zombies = [s for s in level.spawns if s.team == ZOMBIE_TEAM]
    assert raw_zombies, "the playground's penned zombies must still load"
    assert not any(s.team == LEGACY_ZOMBIE_TEAM for s in level.spawns)
    sim = _sim()
    for s in level.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team,
                          footprint=s.footprint))
    zombies = [u for u in sim.units if u.is_zombie]
    assert len(zombies) == len(raw_zombies)
    assert all(u.team == ZOMBIE_TEAM for u in zombies)
    assert all(not u.weapon_id and u.has_grenade == 0 for u in zombies)
    assert sim.zombies() == zombies


def test_team_is_any_int_and_only_zombie_team_is_a_zombie():
    for team in (0, 1, 2, 7):
        u = Unit("M", x=5, y=5, team=team)
        assert not u.is_zombie and u.weapon_id and u.faction_id == team
    z = Unit("Z", x=5, y=5, team=ZOMBIE_TEAM)
    assert z.is_zombie and z.faction_id == ZOMBIE_TEAM


# ---------------------------------------------------------------------------
# Roster slots
# ---------------------------------------------------------------------------
def test_slot_is_fixed_and_never_reused_after_deaths_and_more_spawns():
    sim = _sim()
    first = [Unit(f"A{i}", x=5, y=5, team=0) for i in range(3)]
    other = [Unit(f"B{i}", x=5, y=5, team=1) for i in range(2)]
    for u in (first[0], other[0], first[1], other[1], first[2]):
        sim.add_unit(u)
    assert [u.slot for u in first] == [0, 1, 2]       # spawn order within team
    assert [u.slot for u in other] == [0, 1]
    slots_before = {u.id: u.slot for u in sim.units}

    first[1].alive = False                            # a death mid-match
    late = Unit("A3", x=5, y=5, team=0)
    sim.add_unit(late)
    assert late.slot not in {u.slot for u in first}   # the dead slot stays taken
    assert {u.id: u.slot for u in sim.units if u is not late} == slots_before
    assert first[1] in sim.units                      # dead units stay listed

    roster = sim.roster(0)
    assert [r.slot for r in roster] == list(range(len(roster)))
    assert roster[first[1].slot] is first[1]          # indexed by slot
    padded = sim.roster(0, n_slots=8)
    assert len(padded) == 8 and padded[len(roster):] == [None] * (8 - len(roster))
    assert sim.roster(42) == []


def test_conversion_keeps_the_slot_in_the_old_teams_roster():
    sim = _sim()
    a, b = Unit("A0", x=5, y=5, team=0), Unit("A1", x=5, y=5, team=0)
    sim.add_unit(a)
    sim.add_unit(b)
    a.alive = False
    a.killed_by_zombie = True
    convert_marines_to_zombies(sim.units)
    assert a.slot == 0 and sim.roster(0)[0] is a      # slot kept
    c = Unit("A2", x=5, y=5, team=0)
    sim.add_unit(c)
    assert c.slot == 2                                # ... and never reused


# ---------------------------------------------------------------------------
# Hostility
# ---------------------------------------------------------------------------
def test_a_converted_marine_is_hostile_to_its_old_team_not_to_zombies():
    sim = _sim()
    level = load("playground")
    s = _first_marine_spawn(level)
    ax, ay = int(s.x), int(s.y)
    bx, by = _facing_pair(sim, ax, ay)
    old_mate = Unit("M0", x=ax, y=ay, team=0)
    turned = Unit("M1", x=bx, y=by, team=0)
    zombie = Unit("Z0", x=bx, y=by, team=ZOMBIE_TEAM)
    for u in (old_mate, turned, zombie):
        sim.add_unit(u)
    old_mate.face_towards(float(bx), float(by))
    assert turned not in engagement._enemies_visible_to(sim, old_mate)

    turned.alive = False
    turned.killed_by_zombie = True
    convert_marines_to_zombies(sim.units)
    assert turned.is_zombie and turned.alive and turned.team == ZOMBIE_TEAM
    assert turned in engagement._enemies_visible_to(sim, old_mate)
    zombie.face_towards(float(bx), float(by))
    assert turned.team == zombie.team                 # a friend of the zombies
    assert turned not in engagement._enemies_visible_to(sim, zombie)


def test_two_marine_teams_engage_each_other_and_the_zombie_ai_ignores_them():
    sim = _sim()
    level = load("playground")
    s = _first_marine_spawn(level)
    ax, ay = int(s.x), int(s.y)
    bx, by = _facing_pair(sim, ax, ay)
    red = Unit("R", x=ax, y=ay, team=0)
    blue = Unit("B", x=bx, y=by, team=1)
    sim.add_unit(red)
    sim.add_unit(blue)
    red.face_towards(float(bx), float(by))
    blue.face_towards(float(ax), float(ay))

    assert blue.id in sim.visible_enemy_ids(0)
    assert red.id in sim.visible_enemy_ids(1)

    for u, (tx, ty) in ((red, (ax, ay)), (blue, (bx, by))):
        assert sim.apply_action(u.id, Order(ORDER_MOVE_ATTACK, target_fx=tx,
                                            target_fy=ty, phase=0))
    sim.set_paused(False)
    shooters = set()
    for _ in range(int(CFG.clock.ticks_per_phase)):
        sim.step()
        shooters |= {e.unit_id for e in sim.tick_events
                     if isinstance(e, ShotFiredEvent)}
    assert {red.id, blue.id} <= shooters              # both sides fire
    for u in (red, blue):                             # zombie AI never ran
        assert not u.is_zombie
        assert not u.zombie_activated and u.zombie_path == []
        assert u.last_melee_tick == -999


# ---------------------------------------------------------------------------
# Loadouts + env profiles
# ---------------------------------------------------------------------------
def test_default_marine_is_armed_by_the_marine_loadout_row():
    row = get_tables().loadout(None)
    assert row.name == CFG.marine.loadout
    u = Unit("M", x=5, y=5)
    assert (u.weapon_id, u.has_grenade, u.has_explosive) == \
        (row.weapon, row.grenades, row.explosives)
    assert u.loadout[0] == row.weapon


def test_a_loadout_row_arms_the_unit_with_its_weapon_and_grenades():
    names = list(get_tables().loadouts.by_name)
    for name in names:
        row = get_tables().loadout(name)
        u = Unit("M", x=5, y=5, team=3, loadout_name=name)
        assert u.weapon_id == row.weapon and u.loadout[0] == row.weapon
        assert u.has_grenade == row.grenades
        assert u.has_explosive == row.explosives
        assert u.loadout_name == name


def test_a_zombie_ignores_any_loadout():
    z = Unit("Z", x=5, y=5, team=ZOMBIE_TEAM,
             loadout_name=CFG.marine.loadout)
    assert z.weapon_id == "" and z.has_grenade == 0 and z.loadout == []


def test_env_profile_rows_dress_the_unit_and_default_is_the_baseline():
    tables = get_tables()
    assert tables.env_profile("default") == EnvironmentProfile()
    plain = Unit("M", x=5, y=5)
    assert plain.environment == tables.env_profile("default")
    for name in tables.env_profiles.by_name:
        u = Unit("M", x=5, y=5, env_profile=name)
        assert u.environment == tables.env_profile(name)
        assert u.env_profile_name == name


def _weapon_names():
    from simulation.weapons import get_tables as weapon_tables
    return list(weapon_tables().weapons.by_name)


def test_the_door_refuses_an_unknown_weapon():
    w = _weapon_names()
    with pytest.raises(ValueError, match="no_such_gun"):
        LoadoutTables({"x": {"weapon": "no_such_gun", "grenades": 0,
                             "explosives": 0}}, {}, w, "x")
    with pytest.raises(ValueError):
        LoadoutTables({"x": {"weapon": w[0], "grenades": -1,
                             "explosives": 0}}, {}, w, "x")
    with pytest.raises(ValueError):                  # default names no row
        LoadoutTables({"x": {"weapon": w[0], "grenades": 0,
                             "explosives": 0}}, {}, w, "rifleman_typo")


def test_the_door_refuses_an_unknown_profile_or_field():
    w = _weapon_names()
    row = {"x": {"weapon": w[0], "grenades": 0, "explosives": 0}}
    with pytest.raises(ValueError, match="o2_reserv"):
        LoadoutTables(row, {"suit": {"o2_reserv": 5.0}}, w, "x")
    with pytest.raises(ValueError):
        LoadoutTables(row, {"suit": {"submersion": "floats"}}, w, "x")
    with pytest.raises(KeyError, match="no_such_suit"):
        Unit("M", x=5, y=5, env_profile="no_such_suit")
    with pytest.raises(KeyError, match="no_such_kit"):
        Unit("M", x=5, y=5, loadout_name="no_such_kit")
