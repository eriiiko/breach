"""Gameplay pack G4p -- the focus-fire law, EXPOSURE (#181, #183).

``simulation/exposure.py`` (the law, its door and the slot's landing), the
``DamagePacket.expose`` column, ``damage.apply_packet``'s scale, the
``SimultaneousEffects`` landing, ``[ammo.*] expose``, ``[env_profiles.*] /
[zombie] expose_resist``, ``[damage.exposure]`` and the bench
``tools/bench_focus_fire.py``. Design:
``docs/gameplay_pack_v1_design_2026-11-10.md`` §3 G4p.

Properties (each test's docstring names the change that must break it): with
the law dormant (``k_expose = 0``) or a round that exposes nothing, damage is
exactly what it was for every damage type; exposure returns to exactly 0
without hits in a bounded number of ticks and never leaves [0, x_max]; damage
is monotone non-decreasing in exposure; every hit of one attack slot reads
the slot-start exposure (two shooters in one tick take the same damage per
round, and swapping their ids changes nothing); a hit raises only its own
type, an ``expose = 0`` round and an immune target raise nothing; the bench's
rounds-to-kill falls strictly from one shooter to two and does not rise
after; the block is zero after death and after conversion; the doors refuse
by name; a unit nothing has exposed records in the digest as it did before
G4p, and with the law dormant the pinned match is the pre-G4p match.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from config import CFG  # noqa: E402
from field_ab_harness import _unit_record  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import exposure as ex  # noqa: E402
from simulation import unit_fixed  # noqa: E402
from simulation.ai_zombie import convert_marines_to_zombies  # noqa: E402
from simulation.damage import (DAMAGE_TYPE_NAMES, ENERGY, HEAL,  # noqa: E402
                               KINETIC, N_DAMAGE_TYPES, DamagePacket,
                               apply_packet, mitigate, mitigation_for)
from simulation.environment import EnvironmentProfile  # noqa: E402
from simulation.events import UnitHitEvent  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.loadouts import EnvProfileTable  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.simultaneous import SimultaneousEffects  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.unit_fixed import FP_ONE  # noqa: E402
from simulation.weapons import AmmoTable, get_tables  # noqa: E402
from simulation.turning import place as _place_facing  # noqa: E402
from simulation.unit_fixed import quantize_scalar as _q_rad  # noqa: E402

CARBINE_ROUND = get_tables().ammo.by_name["rifle_556_standard"]


@pytest.fixture(autouse=True)
def _shipped_settings():
    """Every test starts from the shipped dials (a test that edits CFG
    rebuilds them itself) and leaves them shipped."""
    ex.rebuild_settings()
    yield
    ex.rebuild_settings()


def _marine(**kw) -> Unit:
    u = Unit("M", x=4, y=4, team=0, **kw)
    u.id = 1
    return u


def _zombie() -> Unit:
    u = Unit("Z", x=4, y=4, team=ZOMBIE_TEAM)
    u.id = 2
    return u


def _old_damage(unit, packet) -> float:
    """The pre-G4p door's applied delta for ``packet`` on ``unit``."""
    return unit_fixed.quantize_hp_delta(mitigate(
        packet.amount, packet.dtype, mitigation_for(unit), ap=packet.ap))


def _settings(cfg_overrides: dict) -> ex.ExposureSettings:
    base = dict(half_life_seconds=0.125, k_expose=1.0, x_max=2.5)
    base.update(cfg_overrides)
    cfg = SimpleNamespace(damage=SimpleNamespace(exposure=base),
                          clock=SimpleNamespace(ticks_per_second=24),
                          zombie=SimpleNamespace())
    return ex.settings_from_config(cfg)


# ---------------------------------------------------------------------------
# A. Dormant = the pre-G4p damage, every type
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", sorted(DAMAGE_TYPE_NAMES))
def test_with_k_expose_zero_damage_is_exactly_what_it_was(monkeypatch, dtype):
    """Property: with ``k_expose = 0`` a packet of ANY type deals exactly the
    pre-G4p damage, even on a unit whose exposure block is full, and raises
    nothing. Breaks if the scale ignores k, if the dormant law still keeps a
    level, or if the door's quantize split is not the old snap."""
    monkeypatch.setattr(CFG.damage.exposure, "k_expose", 0.0)
    ex.rebuild_settings()
    for amount in (0, 1, 2, 7.5, 10, 60, 123.456):
        u = _marine()
        u.current_hp = 1e6                    # alive after the packet
        u.exposure[:] = [ex.get_settings().x_max_q] * N_DAMAGE_TYPES
        before = list(u.exposure)
        p = DamagePacket(amount=amount, dtype=dtype, expose=FP_ONE)
        expected = _old_damage(u, p)
        hp0 = u.current_hp
        assert apply_packet(u, p, source="bullet") == expected
        assert u.current_hp == hp0 - expected
        assert u.exposure == before


@pytest.mark.parametrize("dtype", sorted(DAMAGE_TYPE_NAMES))
def test_an_unexposed_unit_takes_exactly_the_old_damage(dtype):
    """Property: at the shipped dials, a unit with exposure 0 takes exactly
    the pre-G4p damage of every type, and a packet with ``expose = 0``
    leaves its block at 0. Breaks if the scale is not the identity at 0 or a
    packet without an ``expose`` raises anything."""
    for amount in (0, 1, 2, 7.5, 10, 60, 123.456):
        u = _marine()
        p = DamagePacket(amount=amount, dtype=dtype)
        assert apply_packet(u, p, source="bullet") == _old_damage(u, p)
        assert not any(u.exposure)


# ---------------------------------------------------------------------------
# B. Decay: back to exactly 0, bounded, inside [0, x_max]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half_s", [1 / 24, 0.125, 0.5, 2.0, 10.0])
def test_exposure_decays_to_exactly_zero_in_bounded_ticks(half_s):
    """Property: from ``x_max`` with no hits the level falls every tick,
    never below 0, and is exactly 0 within ``h * (log2(x_max) + 1) +
    ONE / d + 1`` ticks (the halving phase, then the one-count floor's
    tail); after ``h`` ticks it is at most half. Breaks if the floor
    ``max(1, .)`` is dropped (the tail never reaches 0), the step can
    overshoot below 0, or the derived decay stops meaning the half-life."""
    s = _settings({"half_life_seconds": half_s, "x_max": 4.0})
    x, n = s.x_max_q, 0
    h = s.half_life_ticks
    bound = h * (int(np.log2(s.x_max_q)) + 1) + FP_ONE // s.decay_q + 1
    while x > 0:
        nx = ex.decay_step(x, s.decay_q)
        assert 0 <= nx < x
        n += 1
        if n == h:
            assert nx <= s.x_max_q // 2
        x = nx
        assert n <= bound
    assert x == 0


def test_in_the_sim_exposure_returns_to_zero_and_stays_inside_its_range():
    """Property: a unit hit by many rounds at once is capped at ``x_max``
    and, with no further hit, the sim's slot 1c takes it back to exactly 0
    and keeps it there. Breaks if the cap is not applied at the landing,
    the decay slot is not run every tick, or it runs on a zero block."""
    sim = Simulation(_arena(), seed=3, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    z = sim.get_unit(sim.add_unit(Unit("Z", x=20, y=8, team=ZOMBIE_TEAM)))
    z.current_hp = 1e6
    fx = SimultaneousEffects()
    for _ in range(50):
        fx.hit(z, DamagePacket(amount=2, dtype=KINETIC,
                               expose=CARBINE_ROUND.expose_q),
               None, source="bullet")
    fx.resolve()
    s = ex.get_settings()
    assert z.exposure[KINETIC] == s.x_max_q
    seen = []
    for _ in range(400):
        sim.step()
        sim.set_paused(False)
        seen.append(z.exposure[KINETIC])
    assert all(0 <= v <= s.x_max_q for v in seen)
    assert seen[-1] == 0 and seen.index(0) < 400 and \
        all(v == 0 for v in seen[seen.index(0):])


# ---------------------------------------------------------------------------
# C. Monotone in exposure
# ---------------------------------------------------------------------------
def test_damage_taken_is_monotone_non_decreasing_in_exposure():
    """Property: for a fixed packet, the damage applied never falls as the
    target's exposure of that type rises from 0 to ``x_max``, and is larger
    at ``x_max`` than at 0 (k > 0). Breaks if the scale's sign or order of
    operations makes a higher level deal less."""
    s = ex.get_settings()
    assert s.k_q > 0
    for amount in (1, 2, 10, 37.25):
        prev = None
        for x in range(0, s.x_max_q + 1, max(1, s.x_max_q // 257)):
            u = _zombie()
            u.exposure[KINETIC] = x
            d = apply_packet(u, DamagePacket(amount=amount, dtype=KINETIC),
                             source="bullet")
            assert prev is None or d >= prev
            prev = d
        u = _zombie()
        d0 = apply_packet(u, DamagePacket(amount=amount, dtype=KINETIC),
                          source="bullet")
        assert prev > d0


# ---------------------------------------------------------------------------
# D. The start-of-slot rule
# ---------------------------------------------------------------------------
def test_every_hit_of_one_slot_reads_the_slot_start_exposure():
    """Property: hits recorded in one attack slot on one exposed target each
    take the damage of the slot-START level, whatever their order, and their
    raises land together after the slot. Breaks if a hit's raise lands
    before the next hit of the same slot is applied (the later hit would
    take more, and the order would matter)."""
    def run(order):
        z = _zombie()
        z.current_hp = 1e6
        z.exposure[KINETIC] = FP_ONE // 2
        packets = [DamagePacket(amount=2, dtype=KINETIC, attacker_id=a,
                                expose=CARBINE_ROUND.expose_q)
                   for a in order]
        events = []
        fx = SimultaneousEffects()
        for p in packets:
            fx.hit(z, p, events, source="bullet")
        start = z.exposure[KINETIC]
        fx.resolve()
        dmg = [e.damage for e in events if isinstance(e, UnitHitEvent)]
        return dmg, z.current_hp, z.exposure[KINETIC], start
    a = run([10, 11, 12])
    b = run([12, 10, 11])
    assert len(set(a[0])) == 1 and a[0] == b[0]
    assert a[1:] == b[1:]
    add = ex.raise_amount(_zombie(), DamagePacket(
        amount=2, dtype=KINETIC, expose=CARBINE_ROUND.expose_q))
    assert a[2] == min(ex.get_settings().x_max_q, a[3] + 3 * add)


def _arena(h=17, w=40) -> LevelData:
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return LevelData(name="g4p_arena", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _duel(swap: bool):
    """Two marines mirrored about a zombie, both holding the trigger from
    one tick; ``swap`` adds them in the other order (their ids swapped).
    Returns per tick: the target's hp and the sorted hit damages."""
    sim = Simulation(_arena(), seed=11, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.hand_zombies_to_controller()
    z = sim.get_unit(sim.add_unit(Unit("Z", x=19, y=7, team=ZOMBIE_TEAM)))
    spots = [(7, 7, FP_ONE), (31, 7, -FP_ONE)]
    if swap:
        spots.reverse()
    shooters = []
    for (x, y, ax) in spots:
        u = sim.get_unit(sim.add_unit(Unit("S", x=x, y=y, team=0)))
        _place_facing(u, 0 if ax > 0 else unit_fixed.quantize_scalar(np.pi))
        shooters.append((u, ax))
    sim.set_paused(False)
    trace = []
    for _ in range(200):
        for u, ax in shooters:
            sim.set_aim(u.id, ax, 0)
            sim.set_trigger(u.id, True)
        sim.step()
        sim.set_paused(False)
        hits = sorted(e.damage for e in sim.tick_events
                      if isinstance(e, UnitHitEvent) and e.unit_id == z.id)
        trace.append((z.current_hp, tuple(hits), z.alive))
        if not z.alive:
            break
    return trace


def test_two_shooters_in_one_tick_take_the_same_damage_and_ids_do_not_matter():
    """Property (the start-of-slot rule, in the sim): two marines firing at
    one zombie on the same ticks deal the SAME damage per round in every tick
    (exposure above 0 included), and swapping their ids gives the same
    trajectory of the target's hp and hits. Breaks if a slot's raises land
    between its hits, or the landing depends on the walk order."""
    a, b = _duel(False), _duel(True)
    assert a == b
    assert not a[-1][2]                              # the zombie died
    doubles = [hits for _, hits, _ in a if len(hits) == 2]
    assert doubles and all(h[0] == h[1] for h in doubles)
    # ... and the law was live: a later volley hit harder than the first.
    assert max(h[0] for h in doubles) > doubles[0][0]


# ---------------------------------------------------------------------------
# E. What a hit raises
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [KINETIC, ENERGY])
def test_a_hit_raises_only_its_own_type(dtype):
    """Property: a packet of type T with an ``expose`` raises ``x[T]`` and
    no other slot of the block. Breaks if the raise is written to a fixed
    slot or to every type."""
    u = _marine()
    apply_packet(u, DamagePacket(amount=1, dtype=dtype, expose=FP_ONE // 4),
                 source="bullet")
    assert u.exposure[dtype] > 0
    assert all(v == 0 for t, v in enumerate(u.exposure) if t != dtype)


def test_a_round_that_exposes_nothing_and_an_immune_target_raise_nothing():
    """Property: ``expose = 0`` raises nothing; a target whose
    ``expose_resist`` for the type is 1 raises nothing; a resistance of 0.5
    halves the raise. Breaks if the resistance is not applied, or a zero
    column still adds the one-count floor."""
    u = _marine()
    apply_packet(u, DamagePacket(amount=1, dtype=KINETIC), source="bullet")
    assert not any(u.exposure)
    immune = _marine()
    immune.environment = EnvironmentProfile(
        expose_resist=ex.resist_block(1.0, "t"))
    apply_packet(immune, DamagePacket(amount=1, dtype=KINETIC, expose=FP_ONE),
                 source="bullet")
    assert not any(immune.exposure)
    half = _marine()
    half.environment = EnvironmentProfile(
        expose_resist=ex.resist_block({"kinetic": 0.5}, "t"))
    apply_packet(half, DamagePacket(amount=1, dtype=KINETIC,
                                    expose=FP_ONE // 2), source="bullet")
    assert half.exposure[KINETIC] == FP_ONE // 4


def test_the_zombie_states_resistance_is_its_own_column(monkeypatch):
    """Property: a zombie reads ``[zombie] expose_resist`` (selected on the
    zombie state, as its mitigation is), not its environment profile. Breaks
    if the zombie's column is ignored or a marine reads it."""
    monkeypatch.setattr(CFG.zombie, "expose_resist", 1.0, raising=False)
    ex.rebuild_settings()
    z, m = _zombie(), _marine()
    for u in (z, m):
        apply_packet(u, DamagePacket(amount=1, dtype=KINETIC, expose=FP_ONE),
                     source="bullet")
    assert not any(z.exposure) and m.exposure[KINETIC] > 0


def test_in_v1_only_kinetic_rounds_expose():
    """Property (the ruled v1 scope): every ``[ammo.*]`` row with a non-zero
    ``expose`` is a kinetic round with direct damage, each exposing in
    proportion to its damage; fire, gas, blast and energy rows expose
    nothing. Breaks if a non-kinetic row gains an ``expose`` without the
    rule moving, or a bullet row's value leaves the proportion."""
    rows = get_tables().ammo.by_name.values()
    exposing = [a for a in rows if a.expose_q > 0]
    assert exposing
    per_damage = {round(a.expose / a.damage, 6) for a in exposing}
    assert len(per_damage) == 1
    for a in rows:
        if a.expose_q > 0:
            assert a.dtype == "kinetic" and a.damage > 0
        if a.dtype == "kinetic" and a.damage > 0:
            assert a.expose_q > 0


# ---------------------------------------------------------------------------
# F. The bench: focus fire pays
# ---------------------------------------------------------------------------
def test_rounds_to_kill_fall_from_one_shooter_to_two_and_never_rise_after():
    """Property (the point of the law): on the bench, continuous fire at a
    zombie needs strictly fewer rounds IN ALL from two shooters than from
    one, and no more from three or four than from two / three. Breaks if
    the raises stop landing together, the decay outlasts the kill (no gain
    from a second shooter) or the cap makes a larger squad worse."""
    import bench_focus_fire as bench
    r = [bench.run_case("zombie", n, "continuous") for n in (1, 2, 3, 4)]
    assert all(c.killed for c in r)
    rounds = [c.rounds for c in r]
    assert rounds[1] < rounds[0]
    assert rounds[2] <= rounds[1] and rounds[3] <= rounds[2]


# ---------------------------------------------------------------------------
# G. Death and conversion
# ---------------------------------------------------------------------------
def test_the_block_is_zero_after_death_in_a_slot_and_outside_one():
    """Property: a unit killed by a packet -- applied at once, or inside an
    attack slot whose other hits raise it -- ends with a zero block. Breaks
    if a death does not zero, or a slot's raises land on a corpse."""
    u = _marine()
    u.exposure[KINETIC] = FP_ONE
    apply_packet(u, DamagePacket(amount=1e6, dtype=KINETIC, expose=FP_ONE),
                 source="bullet")
    assert not u.alive and not any(u.exposure)
    z = _zombie()
    z.current_hp = 3
    z.exposure[KINETIC] = FP_ONE
    fx = SimultaneousEffects()
    for _ in range(4):
        fx.hit(z, DamagePacket(amount=2, dtype=KINETIC, expose=FP_ONE),
               None, source="bullet")
    fx.resolve()
    assert not z.alive and not any(z.exposure)


def test_the_block_is_zero_after_conversion():
    """Property: a bitten marine that turns into a zombie starts with no
    exposure, whatever its block held. Breaks if the conversion keeps it."""
    m = _marine()
    apply_packet(m, DamagePacket(amount=1e6, dtype=KINETIC),
                 source="melee", mark_killed_by_zombie=True)
    m.exposure[KINETIC] = FP_ONE          # however it got there
    convert_marines_to_zombies([m])
    assert m.is_zombie and m.alive and not any(m.exposure)


# ---------------------------------------------------------------------------
# H. The doors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [-0.1, 1.5, "x", True,
                                 {"kinetic": 2.0}, {"plasma": 0.5}])
def test_the_profile_door_refuses_a_resistance_outside_zero_one(bad):
    """Property: ``[env_profiles.*] expose_resist`` is refused naming the
    profile and the column when outside [0, 1], not a number, or naming no
    damage type. Breaks if the door clamps or ignores it."""
    with pytest.raises(ValueError, match=r"env_profiles\.p\.expose_resist"):
        EnvProfileTable({"p": {"expose_resist": bad}})


def test_the_profile_door_takes_a_scalar_or_a_table():
    """Property: a scalar applies to every type, a table to the named types
    only. Breaks if either form is misread."""
    t = EnvProfileTable({"a": {"expose_resist": 0.25},
                         "b": {"expose_resist": {"kinetic": 1.0}}})
    assert t.get("a").expose_resist == (FP_ONE // 4,) * N_DAMAGE_TYPES
    b = t.get("b").expose_resist
    assert b[KINETIC] == FP_ONE and sum(b) == FP_ONE


def test_the_ammo_door_refuses_a_negative_expose():
    """Property: ``[ammo.*] expose`` is refused naming the row when negative
    or not a number. Breaks if the door clamps it to 0."""
    row = {"family": "f", "dtype": "kinetic", "damage": 1}
    for bad in (-0.5, "x"):
        with pytest.raises(ValueError, match=r"ammo\.r\.expose"):
            AmmoTable({"r": dict(row, expose=bad)})


@pytest.mark.parametrize("key", ex.SETTING_KEYS)
def test_the_settings_door_refuses_a_missing_key(key):
    """Property: a ``[damage.exposure]`` table missing any of its keys is
    refused naming that key. Breaks if a key gets a silent default."""
    full = dict(half_life_seconds=0.125, k_expose=1.0, x_max=2.5)
    del full[key]
    cfg = SimpleNamespace(damage=SimpleNamespace(exposure=full),
                          clock=SimpleNamespace(ticks_per_second=24))
    with pytest.raises(ValueError, match=key):
        ex.settings_from_config(cfg)


@pytest.mark.parametrize("over", [{"half_life_seconds": 0},
                                  {"k_expose": -1}, {"x_max": 0},
                                  {"x_max": "big"}, {"spare": 1}])
def test_the_settings_door_refuses_a_bad_value_or_unknown_key(over):
    """Property: a non-positive half-life or cap, a negative k, a value that
    is not a number and an unknown key are refused naming the key. Breaks if
    the door lets one through."""
    with pytest.raises(ValueError, match=next(iter(over))):
        _settings(over)


def test_the_zombie_column_is_refused_outside_zero_one():
    """Property: ``[zombie] expose_resist`` goes through the same door.
    Breaks if the zombie's column is read unchecked."""
    cfg = SimpleNamespace(
        damage=SimpleNamespace(exposure=dict(half_life_seconds=0.125,
                                             k_expose=1.0, x_max=2.5)),
        clock=SimpleNamespace(ticks_per_second=24),
        zombie=SimpleNamespace(expose_resist=-1.0))
    with pytest.raises(ValueError, match=r"\[zombie\] expose_resist"):
        ex.settings_from_config(cfg)


# ---------------------------------------------------------------------------
# I. The digest
# ---------------------------------------------------------------------------
def test_a_unit_with_no_exposure_records_as_before_and_one_with_some_does_not():
    """Property (absence-transparent): the unit digest record carries an
    ``exposure`` key only while the block is non-zero, so a match nothing
    exposes hashes exactly as before G4p, and an exposure desync is caught.
    Breaks if the key is always written (every pin would move) or never."""
    u = _marine()
    rec = _unit_record(u)
    assert "exposure" not in rec
    u.exposure[KINETIC] = 5
    rec2 = _unit_record(u)
    assert rec2["exposure"][KINETIC] == 5
    assert {k: v for k, v in rec2.items() if k != "exposure"} == rec


#: ship_demo seed 1 as it played before G4p (the G3p pin, win 1 at 246).
#: RE-MEASURED at #183 G5p-a1 commit 1 (2026-11-10, the turning law's
#: representation and routing at unlimited acceleration): the same run on
#: the new tree. Outcome, ticks and field_digest UNCHANGED; tick_digest and
#: the unit stream move. SOLE CAUSES, verified on the same tree: with every
#: non-aim facing writer placing the facing at once again and the old float
#: representation hashed, the run gives exactly the previous values; with
#: the representation alone undone only the unit stream moves (the zombie
#: walk's facing lands a tick later, through slot 1a).
#: (was tick e4e42ce9320d57c1782bc625289e3429937300e1fbebd767a0440c9099e53076,
#: units 6e30ef3f7df49f556d8d0b1e43bc55d2499303302095935a26b522385e7d1cfe)
#: RE-MEASURED at #183 G5p-a1 commit 2 (2026-11-10, the shipped turning
#: acceleration, 30 rad/s^2): the same run with the turns now speeding up
#: and braking. SOLE CAUSE, verified on the same tree: with
#: `turn_accel_rad_per_s2` >= 2w (1000) the run gives exactly the values of
#: re-pin 1 (in the comment above).
#: (was win 1 at 246; field 5175fd4cddd17c018e84bc9548b1b81b5a92ebac0d1c25
#: ba06c3b14528bd6175, tick d207abc729f885076ddcabffcf7b9bcdb2797f4950534032
#: ca4f1e220f49c7c4, units 4c26af9bdab372237cd4570021296285c73dc93d878a9b60
#: 3dc7c6cd2aa05672)
#: RE-MEASURED at #183 G5p-a2 re-pin 1 (2026-11-10, the unit impulse law built with every input at zero -- [impulse]
#: bullet_push_mult = recoil_mult = 0, no shove -- and the MOVE_DIR walk
#: on the Q16 grid (the step is the speed law's rational rounded ONCE to
#: Q16 tiles per tick, critique B M6)):
#: the same dormant-exposure run on the new tree. Outcome, ticks and
#: field_digest UNCHANGED (win 0 at 266); tick_digest and the unit stream
#: move. SOLE CAUSE, verified on the same tree: with the pre-G5p-a2 float walk
#: step reinstated (a scratch monkeypatch of Simulation._step_move_dir)
#: this run gives EXACTLY the previous values -- slot 1a's integration,
#: slot 3's displacement loop, the effort member, the digest keys, the
#: events and the door change nothing while every input is off.
#: (was tick e6a38e5f509d1a91ed74ac11057f90dd2e6f2f791c3f46dd11321be4fc62e314,
#: units c5108241c836c8e7a6d85636f9ad9048f3b2119bb94f1a3c499716b9c77006ee)
#: RE-MEASURED at #183 G5p-a2 re-pin 2 (2026-11-10, hits push and guns recoil: [impulse] bullet_push_mult =
#: recoil_mult = 1.0, the RULED real momentum -- a carbine round 3.6 N s on
#: the target, its recoil on the shooter):
#: the same dormant-exposure run with hits pushing and guns recoiling.
#: Outcome and ticks UNCHANGED (win 0 at 266); field_digest, tick_digest and
#: the unit stream move. SOLE CAUSE, verified on the same tree: with both
#: multipliers at 0 this run gives exactly re-pin 1's values (the comment above).
#: (was field 226202de75e88edd60e00666a884a9be860e5d5981f9d183e251edf1782fca89,
#: tick f88fb88162203c4dc219f31d57a204be1d3d912a45da8e8eed3faa9f6e759121,
#: units 262cc298e43a73ca7e6b668e4419008cdd4b7a8436da0ae0827accd8e78c0863)
#: RE-MEASURED at #183 G5p-b (2026-11-10, the wind and the blast onto the
#: unit impulse law: the dynamic pressure at each footprint is a force
#: q * drag_area through unit_impulse.add_wind; #140's cadence scale and
#: bespoke knockdown DELETED in the same commit).
#: The same dormant-exposure run. Outcome and ticks UNCHANGED (win 0 at
#: 266); all three digests and the unit stream move.
#: SOLE CAUSE, verified on the same build: the pre-G5p-b tree (0b57fc6e)
#: with #140's wind law off (wind_q_slow_pa / wind_q_fall_pa = 1e15 /
#: 2e15) and this tree with the wind's input off ([impulse]
#: wind_push_mult = 0) give exactly the same values (win 0 at 266, 7ec78192 / a53745f5 / ecc897c9) --
#: replacing the law is the whole change.
#: (was field e54b96fa081f8d6588c6a309d23bcfeea99c17eff5d4323e1192308
#: 99aabcb5e, tick 7b1fdf8920e54d262717284ab693dd6eb22526464aeec0b11b6
#: 10428573cb6be, units 5c6236e4261b2c897c9787389b4938490af60ce74a242bf
#: e461fa5712ee3788d)
#: RE-MEASURED at #183 G9p STAGE 1 (2026-11-11, the horde field law,
#: residency P3d: the horde descends an integer BFS field, no A*; see
#: tests/test_zombie_control.py's pins for the law and its proof). This
#: gate's dial is untouched; the horde under it walks the field now, so
#: the match it pins is re-measured on this tree with the gate's own
#: settings -- the law is the whole move.
#: (was win 0 at 266; field
#: 8f63758be630e4af3dd834777cd4f28abdb17924628ee1fbf0e27c53da5f733c; tick
#: da215754064fd37cc5c6f7b6a5e48e8e953665a316bf0e29ab7a548f89bed24e; units
#: 977423cbbc572f03ad30183d6956713a86b4eae2261524be65912ec6da21e2d2)
#: RE-MEASURED at #183 G9p STAGE 2 (2026-11-11, the horde behaviours
#: on the field: wander, hearing, fear of fire, door bashing and the
#: sprint, [zombie]; see tests/test_zombie_control.py's pins). This
#: gate's dial is untouched; the match it pins is re-measured on this
#: tree with the gate's own settings -- the behaviours are the whole
#: move (tests/test_horde_behaviours.py: with all of them off, the
#: stage-1 law exactly).
#: (was win 1 at 254; field
#: d383d16b434a271891b5ef6a03d09b0bd971eaa123e44a9f5c5e078665107ac8; tick
#: 20d79396d6b267ceee64bc4f206798f7e7d56be439e4b035956f3008abc3696a; units
#: 8c3375ebc62fd939608932221143d0f5b01da48bc24c13edbe32c08aaf75349a)
_PRE_G4P_SHIP_DEMO = dict(
    outcome="win", team=0, ticks=264,
    # #29 P3e re-pin (2026-11-12, the match path to integer): the shot
    # cone, the exposure / crit rolls and the FieldEdit noise are integer draws
    # (simulation.draws: new random values), and the blast falloff, the payload
    # deposits, the heat and poison rows, mitigation and the hp are integer
    # (re-rounded) -- the GAME moved by design, not only the hash.
    # docs/p3e_ab_2026-11-12.md: win rate, survivors, rounds per kill and the
    # blast outcomes within seed noise over 24-72 seeds.
    # (was a7f9fd78c8a13c2bf9fc38657b05eae042993affcd85bc777493724808fe2b8b)
    # #29 P3b re-pin (2026-11-12, Philox at the match draw sites): every
    # draw is keyed Philox (seed; tick, index, salt, id) in place of one
    # numpy stream -- WHICH random numbers are drawn moved, nothing else.
    # SOLE CAUSE, verified on the same tree: tests/_p3b_sole_cause.py feeds
    # main's stream words through the keyed sites and this pin's old value
    # comes back bit for bit; A/B in docs/p3b_ab_2026-11-12.md.
    # (was 2db5b2eaac6dcfafdbcfd85ec91772dfe838b1fddff573e90a09eba1de6bcb08)
    field_digest="d5e0e90969e747038c0a8d7391303b21"
                 "804bae0d0cc7c9bf7d4ded8c258c4174",
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was ea975da670c3cec503c4f756b3a4b5c61e60bf49a0721638e4944ec1220c255d)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was dfd9854cf32b78da456df843352039371cfa4d8ec3b33d7043cd4cc3a67c21cf)
    # #29 P3e re-pin (2026-11-12, the match path to integer): the shot
    # cone, the exposure / crit rolls and the FieldEdit noise are integer draws
    # (simulation.draws: new random values), and the blast falloff, the payload
    # deposits, the heat and poison rows, mitigation and the hp are integer
    # (re-rounded) -- the GAME moved by design, not only the hash.
    # docs/p3e_ab_2026-11-12.md: win rate, survivors, rounds per kill and the
    # blast outcomes within seed noise over 24-72 seeds.
    # (was 38fb265a937975bff666bf133f5ebc38c72be4396c32f514678b202df7424c83)
    # #29 P3b re-pin (2026-11-12, Philox at the match draw sites): every
    # draw is keyed Philox (seed; tick, index, salt, id) in place of one
    # numpy stream -- WHICH random numbers are drawn moved, nothing else.
    # SOLE CAUSE, verified on the same tree: tests/_p3b_sole_cause.py feeds
    # main's stream words through the keyed sites and this pin's old value
    # comes back bit for bit; A/B in docs/p3b_ab_2026-11-12.md.
    # (was 892b5eafaec42203142dd4565f8e2203f5eaa57b0197d049e2f9809c1b07d533)
    tick_digest="58fd01b50f17f712913ba0148507e426"
                "a752b5e4a3d5a174565021aa54547c8e",
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was 6795601371a29973269791c8dc67e4539bf08cd219a18fb7ae2cadab2c4ba565)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was 7bbfcee472970bce0238e63b255e5f7dd1fa74d7b8141dea65105d8886dab25e)
    # #29 P3e re-pin (2026-11-12, the match path to integer): the shot
    # cone, the exposure / crit rolls and the FieldEdit noise are integer draws
    # (simulation.draws: new random values), and the blast falloff, the payload
    # deposits, the heat and poison rows, mitigation and the hp are integer
    # (re-rounded) -- the GAME moved by design, not only the hash.
    # docs/p3e_ab_2026-11-12.md: win rate, survivors, rounds per kill and the
    # blast outcomes within seed noise over 24-72 seeds.
    # (was befb598018c7f135c75662f49a6365e42f11d518be6d4ea097c2abc7ccfe04f0)
    # #29 P3b re-pin (2026-11-12, Philox at the match draw sites): every
    # draw is keyed Philox (seed; tick, index, salt, id) in place of one
    # numpy stream -- WHICH random numbers are drawn moved, nothing else.
    # SOLE CAUSE, verified on the same tree: tests/_p3b_sole_cause.py feeds
    # main's stream words through the keyed sites and this pin's old value
    # comes back bit for bit; A/B in docs/p3b_ab_2026-11-12.md.
    # (was 11f23789903ddcfef78450a17ce5f9c91a267f86a46d8c5bbf4c573d96e3ee8f)
    unit_stream="2519f4e1da0cb281ce420bbfeb05b9ca"
                "12706e833f165909b4aee49663e3f9c5")


def test_with_the_law_dormant_the_pinned_match_is_the_pre_g4p_match(
        monkeypatch):
    """Property (the cause of the G4p re-pin): with ``k_expose = 0``,
    ship_demo seed 1 plays exactly the match it played before G4p, tick for
    tick -- the new column on the rounds, the packet field, the slot's
    landing, the decay slot and the digest key change nothing in it, so the
    LAW is the only cause of the pin's move. Breaks if any of those is not
    neutral while the law is dormant. The next DELIBERATE re-pin of this
    match by another cause retires this gate's values with it."""
    import level_loader
    from run_match import run_match
    from test_zombie_control import _UnitStream
    monkeypatch.setattr(CFG.damage.exposure, "k_expose", 0.0)
    from test_bots_v1 import v1_spec
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    stream = _UnitStream()
    # G8p: the first bots (bots_v1) played this pin; the layered bots
    # moved the shipped match, so this gate plays it with them.
    o = run_match(level, v1_spec(level), on_tick=stream)
    got = dict(outcome=o.result.outcome, team=o.result.team,
               ticks=o.result.ticks, field_digest=o.field_digest,
               tick_digest=o.tick_digest, unit_stream=stream.h.hexdigest())
    assert got == _PRE_G4P_SHIP_DEMO
