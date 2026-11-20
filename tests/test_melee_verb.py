"""G6p (#183) -- the MELEE VERB: ``Simulation.melee_intent``, a loadout's
``melee`` row, ``combat.process_melee_intents`` (slot 4's phase A, recorded on
``SimultaneousEffects``, applied at 4c), the right mouse button / pad R3.

PROPERTIES PROTECTED (each test names its own and the change that breaks it)
  1. The doors refuse by name: a loadout's melee naming no melee-archetype
     row, a melee row with a crit (the verb draws no number), the weapon
     row's push and arc.
  2. A blow needs an ADJACENT ENEMY inside the facing ARC: no strike without
     adjacency, none outside the arc, none on a teammate (zombies are enemies),
     the nearest the facing wins, ties to the lower id.
  3. The cadence of the row holds, a whiff spends none, and the latch is
     consumed (never kept for a later tick).
  4. The blow is a ``melee``-caused packet credited to the striker; its stun
     and push LAND AT THE RESOLVE, not in phase A.
  5. Nobody's blow depends on its id (two units striking each other on one
     tick).
  6. The verb cannot be used by a stunned unit, during a throw wind-up, by a
     unit with no melee row or by a zombie; a reload stall does not stop it.
  7. Dormant: with no controller calling it, a match plays as without the
     verb; a recorded match that uses it replays exactly.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import control_source  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
import match_replay as mr  # noqa: E402
from config import CFG  # noqa: E402
from control_source import ControlSource, quantize_stick_direction  # noqa: E402
from level_loader import LevelData  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import Simulation, combat  # noqa: E402
from simulation.events import MeleeEvent, UnitHitEvent  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.loadouts import LoadoutTable  # noqa: E402
from simulation.match import MatchSpec, MatchTeam, SpawnRoom  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.simultaneous import SimultaneousEffects  # noqa: E402
from simulation.status import STUNNED, apply_status, composed_flags  # noqa: E402
from simulation.turning import PI_Q16, place  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.unit_fixed import quantize_scalar  # noqa: E402
from simulation.weapons import WeaponTable, get_tables  # noqa: E402

TILE = 1.0 / 3.0
EAST, NORTH, WEST, SOUTH = 0, PI_Q16 // 2, PI_Q16, -(PI_Q16 // 2)
BUTT = "rifle_butt"


def _arena(h=48, w=48):
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    return LevelData(name="melee", version="1", path=Path("."), tilemap=tm,
                     tile_size_m=TILE, diffuse_path=Path("."))


def _sim():
    sim = Simulation(_arena(), seed=1, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _unit(sim, x, y, facing=EAST, team=0, name="U"):
    u = sim.get_unit(sim.add_unit(Unit(name, x=x, y=y, team=team)))
    place(u, facing)
    return u


def _pair(dx=3, dy=0, foe_team=1, facing=EAST):
    """A striker at (20, 20) facing ``facing`` and an enemy ``(dx, dy)``
    tiles away (3 apart = footprints touching), made tough enough to be hit
    many times."""
    sim = _sim()
    s = _unit(sim, 20, 20, facing=facing, name="S")
    e = _unit(sim, 20 + dx, 20 + dy, team=foe_team, name="E")
    e.current_hp = 10000.0
    return sim, s, e


def _blows(sim):
    return [ev for ev in sim.tick_events if isinstance(ev, MeleeEvent)]


def _butt():
    return get_tables().weapons.by_name[BUTT]


# ---------------------------------------------------------------------------
# 1 -- the doors
# ---------------------------------------------------------------------------
def test_the_loadout_door_refuses_a_melee_that_is_unknown_or_not_a_clean_melee():
    """Property: a loadout's ``melee`` must name a melee-archetype weapon row
    with no crit; an unknown row, a ranged row and a row with a crit are
    refused naming the column; absent means no melee. Breaks if the door
    lets any weapon row be a melee or lets a crit (a random draw) in."""
    rows = {n: float(w.crit_chance) for n, w in get_tables().weapons.by_name.items()
            if w.archetype == "melee"}
    base = dict(weapon="k5_carbine", grenades=0, explosives=0)
    ok = LoadoutTable({"t": dict(base, melee=BUTT)}, ["k5_carbine"],
                      melee_rows=rows)
    assert ok.by_name["t"].melee == BUTT
    assert LoadoutTable({"t": dict(base)}, ["k5_carbine"],
                        melee_rows=rows).by_name["t"].melee == ""
    for bad, needle in (("no_such_row", "loadouts.t.melee"),
                        ("k5_carbine", "loadouts.t.melee"),
                        ("combat_knife", "crit_chance")):
        with pytest.raises(ValueError, match=needle):
            LoadoutTable({"t": dict(base, melee=bad)}, ["k5_carbine"],
                         melee_rows=rows)
    with pytest.raises(ValueError, match="melee"):
        LoadoutTable({"t": dict(base, melee=BUTT)}, ["k5_carbine"])


@pytest.mark.parametrize("column, value", [
    ("melee_push_ns", -1.0), ("melee_push_ns", "x"),
    ("melee_arc_half_deg", 0.0), ("melee_arc_half_deg", 180.5),
    ("melee_arc_half_deg", True)])
def test_the_weapon_door_refuses_a_bad_push_or_arc_by_name(column, value):
    """Property: a weapon row's push is a finite number >= 0 and its arc a
    number in (0, 180] degrees; anything else is refused naming the row and
    column. Breaks if a typo in config.toml silently becomes an arc of 0
    (a verb that never lands) or a negative push."""
    row = dict(archetype="melee", melee_damage=5, melee_dtype="kinetic",
               **{column: value})
    with pytest.raises(ValueError, match=f"weapons.t.{column}"):
        WeaponTable({"t": row}, 24)


def test_the_shipped_rifle_butt_is_a_melee_row_with_the_placeholder_numbers():
    """Property: the rifleman's loadout names the rifle-butt row; the row is
    a no-crit melee with a stun shorter than its cadence (no chain-stun) and
    a push; the arc is quantized once into Q16 radians. Breaks if the row is
    renamed away from the loadout, gains a crit, or its stun outlasts the
    cadence (the baton's chain-stun trap)."""
    from simulation.loadouts import get_tables as lt
    assert lt().loadout("rifleman").melee == BUTT
    w = _butt()
    assert w.archetype == "melee" and w.crit_chance == 0.0
    assert w.status_kind_id is not None and 0 < w.status_ticks < w.rof_interval_ticks
    assert w.melee_push_ns > 0
    assert w.melee_arc_half_q == quantize_scalar(
        np.radians(w.melee_arc_half_deg).item())


# ---------------------------------------------------------------------------
# 2 -- who is struck
# ---------------------------------------------------------------------------
def test_an_adjacent_enemy_in_front_is_struck_and_the_blow_is_credited():
    """Property: with an enemy adjacent dead ahead, one melee_intent costs it
    exactly the row's damage as ONE hit event of cause ``melee`` credited to
    the striker, and emits a MeleeEvent naming both with the row's cadence.
    Breaks if the blow is not the row's packet, is credited to the
    environment, or emits no event."""
    sim, s, e = _pair()
    hp0 = e.current_hp
    sim.melee_intent(s.id)
    sim.step()
    hits = [ev for ev in sim.tick_events if isinstance(ev, UnitHitEvent)
            and ev.unit_id == e.id]
    assert len(hits) == 1
    assert hits[0].source == "melee" and hits[0].attacker_id == s.id
    assert hp0 - e.current_hp == pytest.approx(_butt().melee_damage)
    ev, = _blows(sim)
    assert (ev.unit_id, ev.target_id, ev.ticks) == (
        s.id, e.id, _butt().rof_interval_ticks)


def test_no_strike_without_adjacency():
    """Property: an enemy whose footprint is not within one tile of the
    striker's is never struck, however it is faced. Breaks if the verb
    reaches (a range) instead of touching (``melee_adjacent``)."""
    for dx in (5, 8, 12):
        sim, s, e = _pair(dx=dx)
        hp0 = e.current_hp
        sim.melee_intent(s.id)
        sim.step()
        assert e.current_hp == hp0 and not _blows(sim)
        assert s.last_blow_tick == -1


def test_the_arc_is_inclusive_at_its_half_angle_and_exclusive_beyond():
    """Property: an adjacent enemy at bearing 0 is struck when the striker's
    facing is within the row's half-angle of it (either side) and not struck
    one degree beyond. Breaks if the arc is a different angle, one-sided, or
    compared in the wrong currency."""
    half = _butt().melee_arc_half_deg
    one_deg = quantize_scalar(np.radians(1.0).item())
    arc_q = _butt().melee_arc_half_q
    for sign in (1, -1):
        for off_q, struck in ((arc_q - one_deg, True), (arc_q, True),
                              (arc_q + one_deg, False)):
            sim, s, e = _pair(facing=sign * off_q)
            hp0 = e.current_hp
            sim.melee_intent(s.id)
            sim.step()
            assert (e.current_hp < hp0) == struck, (sign, off_q, half)


def test_an_enemy_behind_is_never_struck():
    """Property: an adjacent enemy directly behind the striker is outside
    any arc below 180 degrees. Breaks if the arc test ignores the facing."""
    sim, s, e = _pair(facing=WEST)
    hp0 = e.current_hp
    sim.melee_intent(s.id)
    sim.step()
    assert e.current_hp == hp0 and not _blows(sim)


def test_a_teammate_is_never_struck_and_never_hides_an_enemy():
    """Property: a teammate adjacent dead ahead takes nothing, and an enemy
    further round the arc is struck past it. Breaks if the target filter
    reads distance or bearing before team."""
    sim, s, e = _pair(dx=3, dy=3)               # the enemy at 45 deg
    mate = _unit(sim, 23, 20, team=0, name="T")  # a teammate dead ahead
    hp_mate = mate.current_hp
    hp_e = e.current_hp
    sim.melee_intent(s.id)
    sim.step()
    assert mate.current_hp == hp_mate
    assert e.current_hp < hp_e
    sim2 = _sim()
    s2 = _unit(sim2, 20, 20, name="S")
    mate2 = _unit(sim2, 23, 20, team=0, name="T")
    sim2.melee_intent(s2.id)
    sim2.step()
    assert mate2.current_hp == 100.0 and not _blows(sim2)


def test_a_zombie_is_an_enemy():
    """Property: a zombie (its own team) adjacent in the arc is struck like
    any enemy -- the verb reads team inequality, never a team literal.
    Breaks if the filter only admits marine teams."""
    sim, s, z = _pair(foe_team=ZOMBIE_TEAM)
    hp0 = z.current_hp
    sim.melee_intent(s.id)
    sim.step()
    assert z.current_hp < hp0


def test_the_nearest_the_facing_wins_and_ties_go_to_the_lower_id():
    """Property: of two adjacent enemies in the arc the one whose bearing is
    nearest the facing is struck; at equal bearings the lower id. Breaks if
    the choice is by distance, list order or the higher id."""
    sim = _sim()
    s = _unit(sim, 20, 20, name="S")
    far = _unit(sim, 23, 22, team=1, name="far")      # ~34 deg off
    near = _unit(sim, 23, 21, team=1, name="near")    # ~18 deg off
    far.current_hp = near.current_hp = 10000.0
    sim.melee_intent(s.id)
    sim.step()
    assert near.current_hp < 10000.0 and far.current_hp == 10000.0
    # ties: mirror positions about the facing; the lower id is struck
    for order in ((0, 1), (1, 0)):
        sim = _sim()
        s = _unit(sim, 20, 20, name="S")
        ys = (21, 19)
        foes = [None, None]
        for k in order:
            foes[k] = _unit(sim, 23, ys[k], team=1, name=f"f{k}")
            foes[k].current_hp = 10000.0
        sim.melee_intent(s.id)
        sim.step()
        lower = min(foes, key=lambda f: f.id)
        higher = max(foes, key=lambda f: f.id)
        assert lower.current_hp < 10000.0 and higher.current_hp == 10000.0


# ---------------------------------------------------------------------------
# 3 -- the cadence and the latch
# ---------------------------------------------------------------------------
def test_blows_land_exactly_one_cadence_apart_while_the_verb_is_held():
    """Property: with melee_intent issued every tick (a held button) blows
    land exactly ``rof_interval_ticks`` apart, the first at once. Breaks if
    the cadence is not the weapon row's, or a held verb stacks blows."""
    sim, s, e = _pair()
    ticks = []
    for _ in range(4 * _butt().rof_interval_ticks):
        sim.melee_intent(s.id)
        sim.step()
        if _blows(sim):
            ticks.append(sim.tick)
    gaps = {b - a for a, b in zip(ticks, ticks[1:])}
    assert len(ticks) >= 4 and gaps == {_butt().rof_interval_ticks}


def test_a_whiff_spends_no_cadence_and_the_latch_is_not_kept():
    """Property: a melee_intent with nobody in the arc spends no cooldown
    (the next tick's verb with an enemy there lands at once) and is consumed
    (an enemy arriving a tick later is NOT struck without a new verb).
    Breaks if a whiff is charged or the latch waits for a target."""
    sim = _sim()
    s = _unit(sim, 20, 20, name="S")
    sim.melee_intent(s.id)
    sim.step()
    assert s.last_blow_tick == -1 and not s.pending_melee
    e = _unit(sim, 23, 20, team=1, name="E")
    e.current_hp = 10000.0
    sim.step()
    assert e.current_hp == 10000.0                       # no kept latch
    sim.melee_intent(s.id)
    sim.step()
    assert e.current_hp < 10000.0                        # lands at once


# ---------------------------------------------------------------------------
# 4 -- phase A decides, phase B lands
# ---------------------------------------------------------------------------
def test_the_hit_the_stun_and_the_push_land_at_the_resolve_not_in_phase_a():
    """Property: ``process_melee_intents`` alone changes no target state --
    no hp, no status, no impulse; it records the three effects (hit, stun,
    push) and spends the striker's cadence. ``resolve()`` then applies them:
    hp down by the row's damage, STUNNED for the row's ticks, a push along
    striker -> target. Breaks if any effect is applied in phase A (a
    same-slot decision would then read it)."""
    sim, s, e = _pair()
    hp0 = e.current_hp
    s.pending_melee = True
    fx = SimultaneousEffects()
    combat.process_melee_intents(sim.units, sim.tick, sim.weapons_tables,
                                 sim.impulse_table, events=[], effects=fx)
    assert e.current_hp == hp0 and not e.statuses
    assert (e.imp_x, e.imp_y) == (0, 0)
    assert len(fx) == 3 and s.last_blow_tick == sim.tick
    fx.resolve()
    assert hp0 - e.current_hp == pytest.approx(_butt().melee_damage)
    stun = [st for st in e.statuses if st.kind == STUNNED]
    assert len(stun) == 1 and stun[0].remaining_ticks == _butt().status_ticks
    assert e.imp_x > 0 and e.imp_y == 0                  # pushed east, away


def test_a_killing_blow_pushes_and_stuns_no_corpse():
    """Property: a blow that kills leaves the target with no stun and no
    impulse (a corpse takes neither). Breaks if the effects ignore death."""
    sim, s, e = _pair()
    e.current_hp = 1.0
    sim.melee_intent(s.id)
    sim.step()
    assert not e.alive
    assert not any(st.kind == STUNNED for st in e.statuses)
    assert (e.imp_x, e.imp_y) == (0, 0)


def test_a_zombie_takes_several_blows_to_die():
    """Property (the placeholder's brief): a spawned 100 hp zombie takes
    between 4 and 6 blows of the rifleman's butt. Breaks if the row's damage
    leaves the design's range (retune on purpose and change this)."""
    sim, s, z = _pair(foe_team=ZOMBIE_TEAM)
    z.current_hp = 100.0
    n = 0
    for _ in range(40 * _butt().rof_interval_ticks):
        if not z.alive:
            break
        sim.melee_intent(s.id)
        sim.step()
        n += len(_blows(sim))
        s.current_hp = 100.0
    assert not z.alive and 4 <= n <= 6, n


# ---------------------------------------------------------------------------
# 5 -- no id dependence
# ---------------------------------------------------------------------------
def _duel(first):
    """Two marines of opposing teams face to face; ``first`` names which one
    is added (and so gets the lower id)."""
    sim = _sim()
    spec = {"A": dict(x=20, y=20, facing=EAST, team=0),
            "B": dict(x=23, y=20, facing=WEST, team=1)}
    us = {}
    for name in first:
        d = spec[name]
        us[name] = _unit(sim, d["x"], d["y"], facing=d["facing"],
                         team=d["team"], name=name)
    return sim, us


def test_two_units_striking_each_other_on_one_tick_do_not_depend_on_ids():
    """Property: A and B strike each other on one tick; whichever was given
    the lower id, each takes the blow once and ends in the same state
    (hp, push, stun) by ROLE. Breaks if the second striker of the walk reads
    the first one's effects (a stun, a death) before deciding."""
    out = []
    for first in ("AB", "BA"):
        sim, us = _duel(first)
        for u in us.values():
            sim.melee_intent(u.id)
        sim.step()
        out.append({n: (u.current_hp, u.imp_x, u.imp_y, u.last_blow_tick,
                        tuple((st.kind, st.remaining_ticks)
                              for st in u.statuses))
                    for n, u in us.items()})
        assert all(u.current_hp < 100.0 for u in us.values())
    assert out[0] == out[1]


def test_a_unit_stunned_by_a_blow_of_the_same_tick_still_strikes_that_tick():
    """Property (#169): the blow is decided against the slot-start state, so
    a unit that is stunned (or killed) by a blow of this very tick still
    lands its own. Breaks if the verb reads post-resolve state."""
    sim, us = _duel("AB")
    us["A"].current_hp = us["B"].current_hp = 5.0     # each blow kills
    for u in us.values():
        sim.melee_intent(u.id)
    sim.step()
    assert not us["A"].alive and not us["B"].alive


# ---------------------------------------------------------------------------
# 6 -- who cannot strike
# ---------------------------------------------------------------------------
def test_a_stunned_a_winding_up_or_unarmed_unit_cannot_strike():
    """Property: no blow from a unit that cannot act (STUNNED), is winding up
    a throw, has no melee row, or is a zombie; a blow that did not happen
    spends no cadence. Breaks if any gate of the verb is dropped."""
    cases = []
    sim, s, e = _pair()
    apply_status(s, STUNNED, magnitude=0, duration_ticks=50, source_id=-1)
    cases.append((sim, s, e))
    sim, s, e = _pair()
    s.throw_release_tick = sim.tick + 10
    cases.append((sim, s, e))
    sim, s, e = _pair()
    s.melee_id = ""
    cases.append((sim, s, e))
    sim = _sim()
    z = _unit(sim, 20, 20, team=ZOMBIE_TEAM, name="Z")
    z.melee_id = BUTT                       # even if it somehow had the row
    m = _unit(sim, 23, 20, team=0, name="M")
    m.current_hp = 10000.0
    cases.append((sim, z, m))
    for k, (sim, striker, foe) in enumerate(cases):
        hp0 = foe.current_hp
        sim.melee_intent(striker.id)
        sim.step()
        assert not _blows(sim) and striker.last_blow_tick == -1
        if k < 3:        # (a zombie's own BITE hurts the marine: its law)
            assert foe.current_hp == hp0


def test_a_reload_stall_does_not_stop_a_blow():
    """Property: a rifle butt needs no magazine: a unit mid-reload strikes.
    Breaks if the verb is gated on the reload stall."""
    sim, s, e = _pair()
    s.current_mag, s.spare_mags = 10, 2                  # a part-spent gun
    sim.reload_intent(s.id)
    sim.step()
    assert s.reload_done_tick > sim.tick                 # stalled
    hp0 = e.current_hp
    sim.melee_intent(s.id)
    sim.step()
    assert e.current_hp < hp0


# ---------------------------------------------------------------------------
# 7 -- dormant, and recordable
# ---------------------------------------------------------------------------
def _digest_after(patch_out, ticks=120):
    """Unit + field digests of a short ship_demo-free scene: a marine and a
    horde in the arena, the marine shooting, nobody calling the verb."""
    from field_ab_harness import _capture_unit_state
    sim = _sim()
    s = _unit(sim, 20, 20, name="S")
    for k in range(3):
        z = _unit(sim, 30 + 4 * k, 20, team=ZOMBIE_TEAM, name=f"Z{k}")
        z.zombie_activated = True
    if patch_out:
        sim_mod = sys.modules["simulation.simulation"]
        sim_mod.process_melee_intents = lambda *a, **k: None
    sim.set_aim(s.id, 65536, 0)
    sim.set_trigger(s.id, True)
    hashes = []
    for _ in range(ticks):
        sim.step()
        hashes.append(_capture_unit_state(sim)["hash"])
    return hashes


def test_with_no_caller_the_verb_changes_nothing_of_a_match(monkeypatch):
    """Property (gameplay pack rule 6): a scene in which nothing calls
    melee_intent plays digest for digest as with the verb's slot removed.
    Breaks if the verb's walk touches state, draws a number or emits a
    digested event when no latch is set."""
    sim_mod = sys.modules["simulation.simulation"]
    monkeypatch.setattr(sim_mod, "process_melee_intents",
                        sim_mod.process_melee_intents)   # restored after
    with_verb = _digest_after(False)
    without = _digest_after(True)
    assert with_verb == without


def _close_box(tmp_path, spec):
    name = "meleebox"
    lvl = tmp_path / name
    lvl.mkdir()
    level_lib.write_level_header(lvl, name=name, tile_size_m=1.0)
    w, h = 30, 14
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (w, h)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", [
        SpawnRoom(team=0, x=2, y=3, w=8, h=8, max_per_team=1),
        SpawnRoom(team=1, x=15, y=3, w=8, h=8, max_per_team=1),
        SpawnRoom(team=ZOMBIE_TEAM, x=24, y=3, w=4, h=8, max_per_team=1)])
    level_lib.write_match(lvl, spec)
    return level_loader.load(name, str(tmp_path))


class _BrawlControl(ControlSource):
    """Test controller: walk at the nearest foe and strike it every tick."""

    def __init__(self, team):
        self.team = team

    def handle_frame(self, sim, renderer=None):
        foes = [u for u in sim.units if u.alive and u.team != self.team
                and not u.is_zombie]
        for u in sim.units:
            if not (u.alive and u.team == self.team) or not foes:
                continue
            f = min(foes, key=lambda f: ((f.x - u.x) ** 2 + (f.y - u.y) ** 2,
                                         f.id))
            dx, dy, _ = quantize_stick_direction(f.x - u.x, f.y - u.y, 0.0)
            sim.set_aim(u.id, dx, dy)
            sim.set_move_dir(u.id, dx, dy, 65536)
            sim.melee_intent(u.id)


def test_a_recorded_brawl_replays_exactly_and_was_decided_by_blows(
        tmp_path, monkeypatch):
    """Property: a match whose only attack is the melee verb is decided by
    it (one marine eliminates the other well before the time limit), records
    the verb and replays to every checkpoint and the original digests. Breaks if the
    verb is not on the tap, or a blow depends on anything the recording does
    not fix."""
    monkeypatch.setitem(control_source._MATCH_CONTROLLERS, "test_brawler",
                        _BrawlControl)
    t = MatchTeam(n=1, loadout="rifleman", env_profile="default",
                  controller="test_brawler")
    spec = MatchSpec(teams=(t, t), seed=5, zombies=0, time_limit_s=40.0)
    lvl = _close_box(tmp_path, spec)
    a = run_match(lvl, record=True, checkpoint_every_ticks=10)
    assert a.result.ticks < 40 * 24, a.result            # someone fell
    kinds = {k for v in a.recording.intents.values() for k, _ in v}
    assert "melee_intent" in kinds
    rec = mr.decode_recording(mr.encode_recording(a.recording))
    b = run_match(lvl, replay=rec)
    assert b.checkpoints_verified == len(a.recording.checkpoints)
    assert (b.result, b.field_digest, b.tick_digest) == \
        (a.result, a.field_digest, a.tick_digest)


def test_the_melee_intent_is_on_the_tap_and_the_policy_row_issues_it():
    """Property: ``melee_intent`` is a recordable kind with a latch rule in
    the policy's verb table and a flag column in the action row; a row with
    the flag issues exactly it. Breaks if the verb skips the tap (a replay
    would lose its blows) or a policy cannot reach it."""
    import control_policy as cp
    assert mr.INTENT_KINDS["melee_intent"] == ("int",)
    assert cp.VERB_RULES["melee_intent"] == ("latch", "flag")
    schema = cp.action_schema()
    col = schema.index("melee_intent")
    row = np.zeros(len(schema), dtype=np.int32)
    row[col] = 1
    calls = []

    class _Sim:
        tick = 0

        def __getattr__(self, name):
            return lambda *a: calls.append((name, a))

    plan = cp._Plan(schema)
    cp.issue_row(_Sim(), 7, row, True, plan)
    assert ("melee_intent", (7,)) in calls
    calls.clear()
    row[col] = 0
    cp.issue_row(_Sim(), 7, row, True, plan)
    assert not any(k == "melee_intent" for k, _ in calls)
