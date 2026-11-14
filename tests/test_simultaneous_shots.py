"""#169 -- simultaneous shot resolution: an attack slot decides every
shooter's eligibility and every shot's geometry against its START state and
applies the hits together (``simulation.simultaneous``), so no unit's
ability to act in a tick depends on its id relative to another unit's.

Properties gated here (each test names the change that must break it):

(a) two mirrored shooters with sight of each other on the same tick BOTH
    fire, their hp stay equal tick for tick, and swapping their unit ids
    gives the mirrored result (the outcome is a function of position);
(b) a mutual kill: both die on one tick, each credited to the other, one
    UnitKilled each;
(c) eligibility is slot-start: a shooter another unit STUNS (a baton) or
    KILLS on tick T, from a lower id, still fires on tick T;
(d) geometry is slot-start: rounds of two shooters stop on the first unit
    on their line even after an earlier round of the tick killed it -- the
    unit behind is untouched -- and the corpse dies once, credited to the
    packet that took it to zero; two rounds already in flight that land on
    one tick (slot 2) behave the same;
(e) the damage door kills once: a packet on a corpse emits its UnitHit but
    no second UnitKilled (two zombies biting one marine to death on one
    tick included);
(f) a tick with no same-tick interaction is bit-identical to the one-pass
    (pre-#169) slot: the conductor with every effect applied at once
    (``simultaneous.IMMEDIATE``) and with phase A / phase B run in lockstep
    on fields, unit state, the hashed events and the RNG stream.

The levels are synthetic (``LevelData``), the test_credited_events shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tests", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from field_ab_harness import assert_trajectories_match, capture_snapshot  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation.draws import MatchDraws as _MatchDraws  # noqa: E402 (#29 P3b)
from simulation import Simulation  # noqa: E402
from simulation import simulation as simulation_mod  # noqa: E402
from simulation import simultaneous  # noqa: E402
from simulation import unit_fixed  # noqa: E402
from simulation import weapons as weapons_mod  # noqa: E402
from simulation.ai_zombie import update_zombies_tick  # noqa: E402
from simulation.combat import fire_burst, process_shooting  # noqa: E402
from simulation.damage import KINETIC, DamagePacket, apply_packet  # noqa: E402
from simulation.events import (  # noqa: E402
    ShotFiredEvent, UnitHitEvent, UnitKilledEvent)
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.intents import FP_ONE  # noqa: E402
from simulation.materials import MAT_HULL  # noqa: E402
from simulation.orders import ORDER_FIRE, Order  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.status import STUNNED, composed_flags  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import AmmoDef, WeaponDef  # noqa: E402
from simulation.turning import place as _place_facing  # noqa: E402
from simulation.unit_fixed import quantize_scalar as _q_rad  # noqa: E402

SEED = 20261106
HULL, AIR = 1, 4


@pytest.fixture(autouse=True)
def _k5_fires_the_five_round_burst(monkeypatch):
    """These properties are about ONE tick's burst being resolved together,
    and need a burst of several rounds; the shipped carbine is one round a
    pull since #183 G2p (Erik's 12 rounds a second), so the rows the tests
    fire are the carbine with the five-round burst it had. The burst size is
    the tests' own arrangement; the simultaneity law does not read it."""
    from config import CFG
    # the config row (a Simulation rebuilds its tables from it) and the
    # shared table (the direct callers read it)
    monkeypatch.setattr(CFG.weapons.k5_carbine, "shots_per_trigger", 5)
    row = weapons_mod.get_tables().weapons.by_name["k5_carbine"]
    monkeypatch.setattr(row, "shots_per_trigger", 5)
W = 30                     # the duel lane's width: tile i mirrors 29 - i


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------
def _lane(h=9, w=W) -> LevelData:
    """One 7-row air lane in hull, mirror-symmetric about x = w/2."""
    tm = np.full((h, w), HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = AIR
    return LevelData(name="simultaneous_lane", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _sim(level=None) -> Simulation:
    sim = Simulation(level or _lane(), seed=SEED, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _add(sim, name, x, y, team, hp=None):
    u = sim.get_unit(sim.add_unit(Unit(name, x=x, y=y, team=team)))
    if hp is not None:
        u.current_hp = float(hp)
    return u


def _step(sim):
    sim.step()
    sim.set_paused(False)
    return list(sim.tick_events)


def _launches(events, unit):
    return [e for e in events if isinstance(e, ShotFiredEvent)
            and e.unit_id == unit.id and e.launch]


def _kills(events, unit=None):
    return [e for e in events if isinstance(e, UnitKilledEvent)
            and (unit is None or e.unit_id == unit.id)]


def _hits(events, unit=None):
    return [e for e in events if isinstance(e, UnitHitEvent)
            and (unit is None or e.unit_id == unit.id)]


def _duel(hp, left_first, n_ticks):
    """Two k5 riflemen 16 tiles apart on the lane's mirror line, aiming at
    each other, triggers held. ``left_first`` spawns the left one first (the
    lower id). At 16 tiles a 3 deg cone keeps every round inside the 3x3
    footprint, so each burst lands 5 x 10 whatever the cone draws are -- the
    duel is geometrically symmetric. Returns per-tick (left, right) records."""
    sim = _sim()
    spawn = [("L", 5, 0), ("R", W - 5 - 3, 1)]
    if not left_first:
        spawn.reverse()
    who = {name: _add(sim, name, x, 2, team=team, hp=hp)
           for (name, x, team) in spawn}
    L, R = who["L"], who["R"]
    # G1p: a facing turns at the turn-rate cap; this duel is about the slot,
    # so both already face their aim (the turn is test_turn_rate's). West is
    # the kit's pi, exactly what the aim (-1, 0) turns to.
    from simulation.turning import PI_Q16
    _place_facing(L, 0)
    _place_facing(R, PI_Q16)
    sim.set_aim(L.id, FP_ONE, 0)          # east
    sim.set_aim(R.id, -FP_ONE, 0)         # west
    sim.set_trigger(L.id, True)
    sim.set_trigger(R.id, True)
    rows = []
    for _ in range(n_ticks):
        ev = _step(sim)
        rows.append({
            "hp": (L.current_hp, R.current_hp),
            "alive": (L.alive, R.alive),
            "fired": (len(_launches(ev, L)), len(_launches(ev, R))),
            "killed": (_kills(ev, L), _kills(ev, R)),
        })
    return sim, L, R, rows


# ---------------------------------------------------------------------------
# (a) mirrored shooters both fire; the id swap mirrors the result
# ---------------------------------------------------------------------------
def test_mirrored_shooters_both_fire_and_the_id_swap_mirrors_the_result(
        monkeypatch):
    """Property: two shooters with sight of each other on the same tick both
    fire on that tick, a mirrored pair stays mirrored tick for tick (equal
    hp, equal life), and the result is the same per POSITION whichever of
    them holds the lower id. Breaks if the shooting slot applies a hit
    before a later shooter's turn (the pre-#169 walk: the lower id's second
    burst kills the other before it fires, so the duel is won by id).
    (G5p-a2: rounds now push their target and recoil their shooter along
    their own flight, so the cone draws -- different for the two shooters --
    would move the pair apart unequally and break the mirror the property
    reads; the cone is closed here, so every push is mirrored exactly. The
    draws are still taken, in the same order.)"""
    from config import CFG
    monkeypatch.setattr(CFG.weapons.k5_carbine, "spread_deg", 0.0)
    row = weapons_mod.get_tables().weapons.by_name["k5_carbine"]
    monkeypatch.setattr(row, "spread_deg", 0.0)
    _s, L1, R1, run_lo = _duel(hp=100, left_first=True, n_ticks=12)
    _s, L2, R2, run_hi = _duel(hp=100, left_first=False, n_ticks=12)
    assert L1.id < R1.id and L2.id > R2.id       # the swap happened
    for rows in (run_lo, run_hi):
        assert rows[0]["fired"] == (5, 5)        # both fire on the first tick
        for r in rows:
            assert r["hp"][0] == r["hp"][1]      # mirrored, tick for tick
            assert r["alive"][0] == r["alive"][1]
            assert r["fired"][0] == r["fired"][1]
    # The same result per position across the id swap.
    assert [r["hp"] for r in run_lo] == [r["hp"] for r in run_hi]
    assert [r["fired"] for r in run_lo] == [r["fired"] for r in run_hi]
    assert not run_lo[-1]["alive"][0]            # the duel did end


# ---------------------------------------------------------------------------
# (b) a mutual kill
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("left_first", [True, False])
def test_two_shooters_can_kill_each_other_on_one_tick(left_first):
    """Property: two shooters who each kill the other with this tick's burst
    BOTH die on this tick, each with exactly one UnitKilled, credited to the
    other. Breaks if the lower id's kill stops the other's shot, or a corpse
    is killed twice (the door's alive -> dead transition)."""
    _s, L, R, rows = _duel(hp=40, left_first=left_first, n_ticks=1)
    r = rows[0]
    assert r["fired"] == (5, 5)
    assert r["alive"] == (False, False)
    kl, kr = r["killed"]
    assert len(kl) == 1 and len(kr) == 1
    assert (kl[0].attacker_id, kl[0].killed_by) == (R.id, "bullet")
    assert (kr[0].attacker_id, kr[0].killed_by) == (L.id, "bullet")


# ---------------------------------------------------------------------------
# (c) eligibility is slot-start
# ---------------------------------------------------------------------------
def _room_tilemap(h=20, w=24):
    """A hull-walled room, level format v2 (codes ARE material ids)."""
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 1
    return tm


def _room() -> GameMap:
    return GameMap(LevelData(name="simultaneous_room", version="2",
                             path=Path("."), tilemap=_room_tilemap(),
                             tile_size_m=1.0, diffuse_path=Path(".")))


def _unit(name, x, y, team, uid, weapon, hp=None):
    u = Unit(name, x=x, y=y, team=team)
    u.id = uid
    u.weapon_id = weapon
    if hp is not None:
        u.current_hp = float(hp)
    return u


def _order_on(u, target):
    u.orders = [Order(ORDER_FIRE, target_fx=target.center_tile_x(),
                      target_fy=target.center_tile_y(), phase=0)]


@pytest.mark.parametrize("striker_weapon, victim_hp", [
    ("arc_baton", None),          # the baton STUNS the victim this tick
    ("k5_carbine", 10.0),         # the burst KILLS the victim this tick
])
def test_a_shooter_stunned_or_killed_this_tick_still_fires_this_tick(
        striker_weapon, victim_hp):
    """Property: a unit eligible at the start of the shooting slot fires in
    it, even when a LOWER id stuns it (arc baton) or kills it (a k5 burst)
    earlier in the same walk -- and the stun / death takes effect after
    (no fire on the stunned ticks that follow). Breaks if a status or a
    death applied mid-walk gates a later shooter's can_act / alive check."""
    gmap = _room()
    striker = _unit("A", 4, 8, team=0, uid=1, weapon=striker_weapon)
    victim = _unit("B", 7, 8, team=1, uid=2, weapon="k5_carbine",
                   hp=victim_hp)
    _order_on(striker, victim)
    _order_on(victim, striker)
    rng = _MatchDraws(SEED)
    events = []
    process_shooting(gmap, [striker, victim], 0, [], 0.0, rng, events=events)
    assert _launches(events, victim)                 # B fired on tick 0
    assert _hits(events, striker)                    # and its rounds landed
    assert _hits(events, victim)                     # A's attack landed too
    if victim_hp is None:
        assert any(st.kind == STUNNED for st in victim.statuses)
        assert victim.alive
        # The stun is real from the next tick on: B holds its fire.
        for t in range(1, 30):
            ev = []
            process_shooting(gmap, [striker, victim], t, [], 0.0, rng,
                             events=ev)
            assert not composed_flags(victim).can_act
            assert not _launches(ev, victim)
    else:
        assert not victim.alive
        assert len(_kills(events, victim)) == 1


# ---------------------------------------------------------------------------
# (d) geometry is slot-start; the corpse dies once
# ---------------------------------------------------------------------------
def test_rounds_stop_on_a_unit_killed_this_tick_and_it_dies_once():
    """Property: every round of the tick stops on the first unit on its
    line, even after an earlier round of the same tick killed it -- the
    unit standing behind it takes nothing -- and the corpse has one
    UnitKilled, credited to the attacker of the packet that took its hp to
    zero. Breaks if a round reads a death applied mid-walk (it flies through
    the corpse into the unit behind) or the door kills a corpse again."""
    gmap = _room()
    a = _unit("A", 2, 4, team=0, uid=1, weapon="k5_carbine")
    c = _unit("C", 2, 12, team=0, uid=2, weapon="k5_carbine")
    x = _unit("X", 12, 7, team=1, uid=3, weapon="", hp=10)
    y = _unit("Y", 17, 7, team=1, uid=4, weapon="", hp=1e9)
    units = [a, c, x, y]
    # Both shooters fire straight east along X's rows (spread 0, origins on
    # X's footprint rows 7..9), Y directly behind X.
    rng = _MatchDraws(SEED)
    events = []
    fx = simultaneous.SimultaneousEffects()
    for shooter, row in ((a, 7.5), (c, 9.5)):
        fire_burst(gmap, units, shooter, 4.5, row, 25.0, row, tick=0,
                   shots=[], real_time=0.0, rng=rng, events=events,
                   spread_deg=0.0, effects=fx)
    fx.resolve()
    hits_x = _hits(events, x)
    assert len(hits_x) == 10                        # all ten rounds stopped on X
    assert not _hits(events, y)                     # none reached Y
    kills = _kills(events, x)
    assert len(kills) == 1
    first = events.index(hits_x[0])
    assert events[first + 1] is kills[0]            # the packet that took it to 0
    assert kills[0].attacker_id == hits_x[0].attacker_id
    assert not x.alive and y.current_hp == 1e9


def _slow_weapon(name="w169_slug"):
    w = WeaponDef(name, "projectile", ammo_family=name + "_fam",
                  spread_deg=0.0, spread_snap_deg=0.0, range_tiles=26,
                  shots_per_trigger=1)
    a = AmmoDef(name + "_std", name + "_fam", "kinetic", damage=10,
                wall_damage=0, speed_tiles_per_tick=2.0)
    a.speed_q16 = unit_fixed.quantize_scalar(2.0)
    return w, a


def test_two_rounds_landing_on_one_tick_both_stop_on_the_first_unit():
    """Property: rounds already in flight (slot 2) that reach a unit on the
    same tick both stop on it, even when the first kills it; the unit
    behind is untouched and the corpse dies once. Breaks if the in-flight
    advance applies a round's hit before the next round marches."""
    sim = _sim(LevelData(name="simultaneous_slot2", version="2",
                         path=Path("."), tilemap=_room_tilemap(),
                         tile_size_m=1.0, diffuse_path=Path(".")))
    a = _add(sim, "A", 2, 2, team=0)
    c = _add(sim, "C", 2, 14, team=0)
    x = _add(sim, "X", 12, 7, team=1, hp=10)
    y = _add(sim, "Y", 17, 7, team=1, hp=1e9)
    w, am = _slow_weapon()
    events = []
    rng = _MatchDraws(SEED)
    for shooter, row in ((a, 7.5), (c, 9.5)):
        fire_burst(sim.gmap, sim.units, shooter, 4.5, row, 25.0, row, tick=0,
                   shots=[], real_time=0.0, rng=rng, events=events,
                   weapon=w, ammo=am, spread_deg=0.0, bullets=sim.bullets)
    assert len(sim.bullets) == 2                  # both in flight
    landed = []
    for _ in range(10):
        landed += _step(sim)
        if not sim.bullets:
            break
    assert not sim.bullets
    assert len(_hits(landed, x)) == 2             # both stopped on X
    assert not _hits(landed, y)
    assert len(_kills(landed, x)) == 1
    assert not x.alive and y.current_hp == 1e9


# ---------------------------------------------------------------------------
# (e) the door kills once
# ---------------------------------------------------------------------------
def test_a_packet_on_a_corpse_emits_its_hit_but_no_second_kill():
    """Property: a unit dies once -- the packet that takes hp to zero emits
    the UnitKilled; a later packet on the corpse still lands (UnitHit, hp)
    but emits no second UnitKilled. Two zombies biting one marine to death
    on one tick (the zombie slot's tick-start prey list) credit one kill.
    Breaks if apply_packet runs the death transition on a dead unit."""
    u = Unit("U", x=2, y=2, team=0)
    u.current_hp = 5.0
    events = []
    apply_packet(u, DamagePacket(10, KINETIC, 7), events, source="bullet")
    apply_packet(u, DamagePacket(10, KINETIC, 8), events, source="bullet")
    assert [type(e).__name__ for e in events] == [
        "UnitHitEvent", "UnitKilledEvent", "UnitHitEvent"]
    assert _kills(events)[0].attacker_id == 7
    assert u.current_hp == -15.0 and not u.alive

    gmap = _room()
    m = Unit("M", x=8, y=8, team=0)
    m.id, m.current_hp = 1, 1.0
    zs = []
    for uid, zx in ((2, 5), (3, 11)):              # both adjacent to M
        z = Unit(f"Z{uid}", x=zx, y=8, team=ZOMBIE_TEAM)
        z.id, z.zombie_activated = uid, True
        zs.append(z)
    events = []
    update_zombies_tick(gmap, [m] + zs, tick=100, events=events)
    assert len(_hits(events, m)) == 2              # both bites landed
    assert len(_kills(events, m)) == 1


# ---------------------------------------------------------------------------
# (f) no same-tick interaction: bit-identical to the one-pass slot
# ---------------------------------------------------------------------------
_TARGET_HP = 1e9


def _quiet_firefight() -> Simulation:
    """Three shooters, three lanes walled apart, each shooting a target that
    neither shoots nor dies, and one rifle chewing the lane's end hull (300
    hp: 40 ticks of bursts never break a tile): hits, chew, beams and RNG
    draws every few ticks, and no attack whose effect another attack of the
    same tick reads."""
    h, w = 31, 32
    tm = np.full((h, w), HULL, dtype=np.int32)
    for top in (1, 11, 21):
        tm[top:top + 7, 1:w - 1] = AIR
    sim = _sim(LevelData(name="simultaneous_quiet", version="1",
                         path=Path("."), tilemap=tm, tile_size_m=1.0,
                         diffuse_path=Path(".")))
    rifle = _add(sim, "R", 3, 3, team=0)
    _add(sim, "TR", 16, 3, team=1, hp=_TARGET_HP)
    chewer = _add(sim, "C", 3, 13, team=0)
    laser = _add(sim, "L", 3, 23, team=0)
    laser.weapon_id = "lance_3"
    _add(sim, "TL", 16, 23, team=1, hp=_TARGET_HP)
    for u in (rifle, chewer, laser):
        sim.set_aim(u.id, FP_ONE, 0)
        sim.set_trigger(u.id, True)
    return sim


def _rng_tail(sim):
    return sim.draws.drawn


def test_a_tick_without_interaction_is_bit_identical_to_the_one_pass_slot(
        monkeypatch):
    """Property: when no attack's effect is read by another decision of the
    same slot, phase A / phase B applies exactly what the one-pass slot
    applied, in the same order, with the same RNG draws -- fields, unit
    state and the hashed hit / kill events match the conductor run with
    every effect applied at once (``simultaneous.IMMEDIATE``, the pre-#169
    behaviour) in lockstep. Breaks if the deferral reorders effects, moves
    an RNG draw, or loses / duplicates a packet, a chew or a facing write.

    The focus-fire law is DORMANT here (``k_expose = 0``, #183 G4p): with it
    live, a burst's rounds on one target DO interact -- one-pass, each hit's
    exposure raise is read by the next round's damage; deferred, every round
    reads the slot-start level (the start-of-slot rule,
    ``tests/test_exposure.py``) -- so this scenario would no longer be one
    "without interaction"."""
    from config import CFG
    from field_ab_harness import capture_trajectory
    from simulation import exposure
    # (the shared settings object is restored at teardown: the sims below
    # rebuild it from the patched CFG)
    monkeypatch.setattr(exposure, "_SETTINGS", exposure.get_settings())
    monkeypatch.setattr(CFG.damage.exposure, "k_expose", 0.0)

    def run(immediate):
        if immediate:
            monkeypatch.setattr(simulation_mod, "SimultaneousEffects",
                                lambda: simultaneous.IMMEDIATE)
        else:
            monkeypatch.setattr(simulation_mod, "SimultaneousEffects",
                                simultaneous.SimultaneousEffects)
        sims = []

        def make():
            sims.append(_quiet_firefight())
            return sims[-1]
        traj = capture_trajectory(make, n_steps=40)
        return traj, sims[0]

    traj_b, sim_b = run(immediate=False)
    traj_a, sim_a = run(immediate=True)
    assert_trajectories_match(traj_a, traj_b)
    assert _rng_tail(sim_a) == _rng_tail(sim_b)
    # The scenario exercised what it claims: hits on both targets, a chewed
    # wall, and no deaths.
    hp = {u.name: u.current_hp for u in sim_b.units}
    assert hp["TR"] < _TARGET_HP and hp["TL"] < _TARGET_HP
    assert all(u.alive for u in sim_b.units)
    end_wall = (slice(11, 18), 31)                     # lane 2's east hull
    fresh = _quiet_firefight().gmap.wall_hp[end_wall]
    assert (sim_b.gmap.wall_hp[end_wall] < fresh).any()
    assert (sim_b.gmap.material[end_wall] == MAT_HULL).all()  # none broke
