"""#195 1d -- the team picture (``simulation/team_picture.py``).

Properties (``docs/rl_agent_rung1_note_1d_2026-11-12.md``, "Property
tests"): the conductor's one update per tick, never on demand; a record is
plain values taken at its sighting, its motion from two consecutive
sightings, kept exactly ``memory_ticks``; nothing the team never saw; the
one lane rasteriser. Step (ii): a record outlives an unseen death or
conversion and goes at a seen death; the fire lanes are read from state
(``fire_lanes``). The bit-for-bit gate of step (i) is the pins under
``docs/rl_agent_rung1_1d_pins/`` (step (ii) moves them by design); step
(ii)'s gate is the tournament the note names.

Scenes are the bots' test boxes (``test_bots_v1._level``: an open hull box,
optionally split by a wall), the sight rule ``all_round`` where a scene
needs a unit seen whichever way the observer faces.
"""
from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from config import CFG, Namespace  # noqa: E402
from control_bots import BotTeam  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import Simulation, vision  # noqa: E402
from simulation import team_picture as tp  # noqa: E402
from simulation.match import MatchSpec, MatchTeam  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.damage import KINETIC, DamagePacket, apply_packet  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.turning import PI_Q16  # noqa: E402
from simulation.turning import place as place_facing  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.unit_fixed import quantize_scalar  # noqa: E402
from test_bots import _match_level  # noqa: E402
from test_bots_v1 import _level  # noqa: E402

ONE = 1 << 16

BOTS = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                 controller="bots")


def _scene(tmp_path, name, units, *, wall_x=None, memory_ticks=None):
    """A running sim on the bots' box holding ``units`` (team, x, y), the
    sight rule all-round, the picture started for teams 0 and 1."""
    lvl = _level(tmp_path, name, wall_x=wall_x)
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    sim.sight = "all_round"
    out = []
    for i, (team, x, y) in enumerate(units):
        u = Unit(f"u{i}", x=x, y=y, team=team, footprint=3)
        sim.add_unit(u)
        out.append(u)
    sim.start_team_picture((0, 1), memory_ticks=memory_ticks)
    return sim, out


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def test_updated_once_per_step_at_the_sims_tick(tmp_path, monkeypatch):
    """Property: through a whole match the picture is at ``sim.tick`` after
    every step, and the conductor called ``update`` exactly once per step
    (plus the one at ``start_team_picture``). Breaks if the slot is missing,
    doubled, or placed before the tick advance."""
    calls = []
    real = tp.TeamPicture.update

    def counting(self, sim):
        calls.append(int(sim.tick))
        return real(self, sim)
    monkeypatch.setattr(tp.TeamPicture, "update", counting)
    spec = MatchSpec(teams=(BOTS, BOTS), seed=1, time_limit_s=4.0, zombies=2)
    lvl = _match_level(tmp_path, "once", spec)
    seen = []

    def watch(sim):
        assert sim.team_picture.tick == sim.tick
        seen.append(int(sim.tick))
    o = run_match(lvl, on_tick=watch)
    assert seen == list(range(1, o.result.ticks + 1))
    assert calls == list(range(0, o.result.ticks + 1))


def test_no_picture_off_a_match(tmp_path):
    """Property: a sim that never spawned a match has no picture and steps
    without one. Breaks if the picture runs on a game path."""
    lvl = _level(tmp_path, "off")
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    sim.add_unit(Unit("a", x=10, y=10, team=0, footprint=3))
    sim.add_unit(Unit("b", x=30, y=10, team=1, footprint=3))
    for _ in range(3):
        sim.step()
    assert sim.team_picture is None


def test_a_bot_team_needs_the_picture(tmp_path):
    """Property: a ``BotTeam`` on a sim with no picture refuses, naming
    ``start_team_picture``. Breaks if the bots build a picture of their own
    (a second picture)."""
    lvl = _level(tmp_path, "need")
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    sim.add_unit(Unit("a", x=10, y=10, team=0, footprint=3))
    with pytest.raises(RuntimeError, match="start_team_picture"):
        BotTeam(0).handle_frame(sim, None)


def test_never_on_demand(tmp_path):
    """Property: ``update`` refuses a skipped tick and ``records`` refuses a
    stale picture. Breaks if a reader brings the picture up to date itself
    (an on-demand picture is a second update path)."""
    sim, _ = _scene(tmp_path, "strict", [(0, 10, 10), (1, 30, 10)])
    pic = sim.team_picture
    pic.update(sim)                      # current: a no-op
    sim.team_picture = None              # the conductor skips a tick ...
    sim.step()
    with pytest.raises(RuntimeError, match="team picture"):
        pic.records(0)                   # ... the stale picture refuses
    sim.step()
    with pytest.raises(RuntimeError, match="once every tick"):
        pic.update(sim)


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------
def test_a_record_holds_plain_values(tmp_path):
    """Property: every field of every record is an int, a bool, a str or a
    tuple of ints. Breaks if a live ``Unit`` (or any object) is stored."""
    sim, _ = _scene(tmp_path, "plain", [(0, 10, 10), (1, 30, 10),
                                        (1, 40, 20)])
    for _ in range(3):
        sim.step()
    recs = sim.team_picture.records(0)
    assert len(recs) == 2
    for rec in recs.values():
        for f in fields(rec):
            v = getattr(rec, f.name)
            if isinstance(v, tuple):
                assert all(type(x) is int for x in v), (f.name, v)
            else:
                assert type(v) in (int, bool, str), (f.name, v)


def test_a_record_is_the_sighting_and_stays_put(tmp_path):
    """Property: a record equals the unit's state at its sighting tick;
    once the unit is out of sight and moves, the record does not move.
    Breaks if the update or a reader takes the live position."""
    sim, (me, foe) = _scene(tmp_path, "stay", [(0, 10, 20), (1, 40, 20)])
    sim.step()
    rec = sim.team_picture.records(0)[foe.id]
    assert rec.tick == sim.tick
    assert rec.tile == (foe.tile_x, foe.tile_y)
    assert rec.centre == (foe.center_tile_x(), foe.center_tile_y())
    assert rec.pos_q == (quantize_scalar(foe.x), quantize_scalar(foe.y))
    assert (rec.uid, rec.team, rec.zombie) == (foe.id, 1, False)
    assert rec.loadout == (foe.loadout_name or "")
    # a fixture puts the foe behind a wall the observer cannot see through
    sim2, (me2, foe2) = _scene(tmp_path, "stay2", [(0, 10, 20), (1, 40, 20)],
                               wall_x=30)
    sim2.team_picture._write(sim2, 0, foe2)        # seen now
    before = sim2.team_picture.records(0)[foe2.id]
    sim2.set_move_dir(foe2.id, ONE, 0, 0)
    for _ in range(24):
        sim2.step()
    assert quantize_scalar(foe2.x) > before.pos_q[0]   # it walked
    assert sim2.team_picture.records(0)[foe2.id] == before


def test_motion_is_two_consecutive_sightings(tmp_path):
    """Property: ``vel_q`` is the displacement between two sightings one
    tick apart, ``(0, 0)`` on a first sighting and after a gap. Breaks if
    motion is taken across an unseen tick or from the live unit."""
    sim, (me, foe) = _scene(tmp_path, "vel", [(0, 10, 20), (1, 30, 20)])
    pic = sim.team_picture
    assert pic.records(0)[foe.id].vel_q == (0, 0)  # the first sighting
    sim.set_move_dir(foe.id, ONE, 0, 0)
    for _ in range(6):
        p0 = pic.records(0)[foe.id].pos_q
        sim.step()
        rec = pic.records(0)[foe.id]
        assert rec.vel_q == (rec.pos_q[0] - p0[0], rec.pos_q[1] - p0[1])
    assert rec.vel_q[0] > 0
    # after a gap (the record two ticks old) a sighting moves at 0
    t = int(sim.tick)
    rec2 = tp._sighting(foe, t + 2, rec)
    assert rec2.vel_q == (0, 0)


def test_memory_kept_to_the_tick(tmp_path):
    """Property: a record unseen since its tick is kept while ``tick -
    rec.tick == memory_ticks`` and gone one tick later. Breaks on an
    off-by-one in the expiry."""
    m = 5
    sim, (me, foe) = _scene(tmp_path, "mem", [(0, 10, 20), (1, 40, 20)],
                            wall_x=30, memory_ticks=m)
    pic = sim.team_picture
    rec = pic._write(sim, 0, foe)
    for _ in range(m):
        sim.step()
    assert sim.tick - rec.tick == m and foe.id in pic.records(0)
    sim.step()
    assert foe.id not in pic.records(0)


def test_the_memory_setting_is_the_tables():
    """Property: ``[team_picture] memory_s`` is quantized once to whole
    ticks (at least one) and the door refuses a missing or mistyped key by
    name. Breaks if the memory falls back to a default or the bots keep a
    copy of the key."""
    s = tp.picture_settings(CFG)
    tps = float(CFG.clock.ticks_per_second)
    assert tp.memory_ticks(s, tps) == max(1, int(round(s.memory_s * tps)))
    assert not hasattr(CFG.bots, "memory_s")
    for bad, msg in (({}, "missing 'memory_s'"),
                     ({"memory_s": "6"}, "must be a number"),
                     ({"memory_s": 6.0, "x": 1}, "unknown key")):
        with pytest.raises(ValueError, match=msg):
            tp.picture_settings(Namespace({"team_picture": bad}))
    with pytest.raises(ValueError, match=r"no \[team_picture\]"):
        tp.picture_settings(Namespace({}))


def test_only_what_the_team_saw(tmp_path):
    """Property: through a whole match every record names a unit that was
    in ``visible_by_team[team]`` at the record's tick. Breaks if the picture
    shows a unit the team never saw."""
    spec = MatchSpec(teams=(BOTS, BOTS), seed=2, time_limit_s=6.0, zombies=2)
    lvl = _match_level(tmp_path, "saw", spec)
    visible = {}
    checked = [0]

    def watch(sim):
        vis = vision.state_for(sim)
        for team in (0, 1):
            visible[(team, int(sim.tick))] = set(
                vis.visible_by_team.get(team, ()))
            for uid, rec in sim.team_picture.records(team).items():
                assert uid in visible[(team, rec.tick)], (team, uid, rec)
                checked[0] += 1
    run_match(lvl, on_tick=watch)
    assert checked[0] > 0


# ---------------------------------------------------------------------------
# the lanes' drawing routine
# ---------------------------------------------------------------------------
def test_draw_lanes_order_free_and_shared_cells():
    """Property: ``draw_lanes`` gives the same grid whatever the segments'
    order; a cell on one owner's lane holds its id, on two owners' lanes
    -2, elsewhere -1. Breaks if the routine's -1 / -2 / owner law
    changes."""
    segs = [(3, (2, 10), (40, 10)), (7, (20, 2), (20, 30)),
            (9, (5, 25), (35, 25))]
    a = tp.draw_lanes(32, 48, segs, 4, 1)
    b = tp.draw_lanes(32, 48, segs[::-1], 4, 1)
    assert np.array_equal(a, b)
    assert a.shape == (32 // 4 + 1, 48 // 4 + 1)
    assert a[10 // 4, 20 // 4] == -2           # 3 and 7 cross
    assert a[10 // 4, 36 // 4] == 3
    assert a[18 // 4, 20 // 4] == 7
    assert a[0, 47 // 4] == -1
    assert set(np.unique(a)) <= {-2, -1, 3, 7, 9}


# ---------------------------------------------------------------------------
# step (ii): honest records
# ---------------------------------------------------------------------------
def _kill_in_next_step(monkeypatch, sim, victim, *, by_zombie=False):
    """Kill ``victim`` INSIDE the next step (its ``UnitKilledEvent`` on that
    step's ``tick_events``, before the picture's slot 11; a zombie's kill
    converts in the same ``on_tick_end``, as a bite does)."""
    rules = sim.ruleset
    real = type(rules).on_tick_end
    done = []

    def hooked(sim_):
        if not done:
            done.append(True)
            apply_packet(victim, DamagePacket(10_000, KINETIC, -1),
                         sim_.tick_events, source="bullet",
                         mark_killed_by_zombie=by_zombie)
        return real(rules, sim_)
    monkeypatch.setattr(rules, "on_tick_end", hooked)


def test_an_unseen_death_keeps_the_record_its_memory_time(tmp_path,
                                                          monkeypatch):
    """Property: a unit killed while out of the team's sight (its record
    older than the observation before the step it died in) keeps its record,
    unchanged, for the record's whole memory time, and loses it one tick
    later. Breaks if the death is detected by liveness (``alive``) or by any
    kill event, seen or not."""
    m = 8
    sim, (me, foe) = _scene(tmp_path, "unseen", [(0, 10, 20), (1, 40, 20)],
                            wall_x=30, memory_ticks=m)
    pic = sim.team_picture
    rec = pic._write(sim, 0, foe)         # seen now, then never again
    sim.step()
    _kill_in_next_step(monkeypatch, sim, foe)
    sim.step()
    assert not foe.alive
    assert pic.records(0)[foe.id] == rec
    while sim.tick - rec.tick < m:
        sim.step()
        assert pic.records(0)[foe.id] == rec
    sim.step()
    assert foe.id not in pic.records(0)


def test_a_seen_death_drops_the_record(tmp_path, monkeypatch):
    """Property: a unit the team saw at the observation just before the step
    it died in loses its record on that step's update. Breaks if a seen
    death is kept for the memory time (the event is not read) or the
    "seen" test reads another tick."""
    sim, (me, foe) = _scene(tmp_path, "seen", [(0, 10, 20), (1, 30, 20)])
    pic = sim.team_picture
    sim.step()
    assert pic.records(0)[foe.id].tick == sim.tick
    _kill_in_next_step(monkeypatch, sim, foe)
    sim.step()
    assert not foe.alive
    assert foe.id not in pic.records(0)


def test_an_unseen_conversion_keeps_the_marine_record(tmp_path, monkeypatch):
    """Property: a marine converted out of the team's sight stays a marine
    record (class and side as seen); once the team sees it again the record
    is the zombie's. Breaks if the class or side is read from the live
    unit."""
    sim, (me, foe) = _scene(tmp_path, "conv", [(0, 10, 20), (1, 24, 20)])
    sim.sight = "cone"
    place_facing(me, PI_Q16)              # looking west, away from the foe
    sim.step()
    pic = sim.team_picture
    rec = pic.records(0)[foe.id]          # seen all-round at the start ...
    assert rec.tick < sim.tick            # ... and not since
    _kill_in_next_step(monkeypatch, sim, foe, by_zombie=True)
    sim.step()
    assert foe.is_zombie and foe.team == ZOMBIE_TEAM
    r = pic.records(0)[foe.id]
    assert r == rec and (r.zombie, r.team) == (False, 1)
    sim.set_aim(me.id, ONE, 0)            # turn east: see it again
    for _ in range(40):
        sim.step()
        r = pic.records(0).get(foe.id)
        if r is not None and r.tick == sim.tick:
            break
    assert r is not None and r.tick == sim.tick
    assert (r.zombie, r.team) == (True, ZOMBIE_TEAM)


# ---------------------------------------------------------------------------
# step (ii): the fire lanes from state
# ---------------------------------------------------------------------------
def test_a_lane_is_a_held_trigger_cut_at_the_wall(tmp_path):
    """Property: a unit holding its trigger, facing east, owns the lane
    cells along its row from its centre tile to the last open tile before
    the wall, and no others; with the trigger released there is no lane.
    Breaks if the lanes read anything but the units' state (trigger,
    facing, weapon range) or the walk does not stop at the first solid
    tile."""
    sim, (me, foe) = _scene(tmp_path, "lane", [(0, 10, 20), (1, 50, 20)],
                            wall_x=30)
    place_facing(me, 0)
    rng = sim.weapons_tables.weapons.by_name[me.weapon_id].range_tiles
    assert rng > 30 - me.center_tile_x()  # the wall cuts the lane
    cx, cy = me.center_tile_x(), me.center_tile_y()
    sim.set_trigger(me.id, True)
    g = tp.fire_lanes(sim, 0, 1, 0)
    lane = {(int(y), int(x)) for y, x in zip(*np.nonzero(g == me.id))}
    assert lane == {(cy, x) for x in range(cx, 30)}
    assert set(np.unique(g)) == {-1, me.id}
    assert (tp.fire_lanes(sim, 1, 1, 0) == -1).all()  # the other team's
    sim.set_trigger(me.id, False)
    assert (tp.fire_lanes(sim, 0, 1, 0) == -1).all()


def test_a_bot_reads_the_lanes_any_controller_would(tmp_path):
    """Property: the layered bots' lanes are ``fire_lanes`` of the units'
    state at the team update -- the same grid an ``idle`` team's unit in
    that state has, whoever set its trigger. Breaks if the bots draw lanes
    from their own memory (last tick's targets)."""
    sim, (a, b, foe) = _scene(tmp_path, "botlane",
                              [(0, 10, 12), (0, 10, 26), (1, 50, 20)],
                              wall_x=30)
    place_facing(a, 0)
    sim.set_trigger(a.id, True)           # set as an idle team's would be
    bots = BotTeam(0)
    s = bots.s
    want = tp.fire_lanes(sim, 0, s.lane_cell, s.lane_halfwidth)
    assert (want == a.id).any()
    bots.handle_frame(sim, None)          # the first frame: a team update
    assert np.array_equal(bots.brain.lanes, want)


# ---------------------------------------------------------------------------
# #195 1g: the trackers (obs spec 1.6) -- members of the picture on the
# same slot-11 driver, never memory on demand
# ---------------------------------------------------------------------------
def _hit_in_next_step(monkeypatch, sim, victim, dmg, attacker_id=-1,
                      direction=(0, 0), source="bullet"):
    """Land ONE hit on ``victim`` inside the next step (its ``UnitHitEvent``
    on that step's ``tick_events``, before the picture's slot 11)."""
    rules = sim.ruleset
    real = type(rules).on_tick_end
    done = []

    def hooked(sim_):
        if not done:
            done.append(True)
            apply_packet(victim, DamagePacket(dmg, KINETIC, attacker_id,
                                              dir_x_q=direction[0],
                                              dir_y_q=direction[1]),
                         sim_.tick_events, source=source)
        return real(rules, sim_)
    monkeypatch.setattr(rules, "on_tick_end", hooked)


def test_motion_is_the_last_ticks_displacement(tmp_path):
    """Property: ``motion[uid]`` is the unit's Q16 position at this update
    and the one before, so ``velocity_q`` is the last tick's displacement,
    ``(0, 0)`` at the first update; a standing unit reads ``(0, 0)``.
    Breaks if the tracker is taken across a skipped tick or on demand."""
    sim, (me, foe) = _scene(tmp_path, "motion", [(0, 10, 20), (1, 30, 20)])
    pic = sim.team_picture
    assert pic.velocity_q(me.id) == (0, 0)
    assert pic.velocity_q(foe.id) == (0, 0)
    sim.set_move_dir(me.id, ONE, 0, 0)
    for _ in range(6):
        p0 = quantize_scalar(me.x), quantize_scalar(me.y)
        sim.step()
        p1 = quantize_scalar(me.x), quantize_scalar(me.y)
        assert pic.motion[me.id] == (p0, p1)
        assert pic.velocity_q(me.id) == (p1[0] - p0[0], p1[1] - p0[1])
    assert pic.velocity_q(me.id)[0] > 0
    assert pic.velocity_q(foe.id) == (0, 0)
    assert pic.velocity_q(999) == (0, 0)


def test_last_hit_is_the_last_damaging_hit_and_never_clears(tmp_path,
                                                            monkeypatch):
    """Property: ``last_hit[uid]`` is the tick, travel vector and cause of
    the last ``damage > 0`` hit on the unit, kept as its age grows; a hit
    of no damage moves nothing. Breaks if a hit is dropped, a heal counted
    or the memory cleared."""
    sim, (me, foe) = _scene(tmp_path, "lasthit", [(0, 10, 20), (1, 30, 20)])
    pic = sim.team_picture
    assert me.id not in pic.last_hit
    _hit_in_next_step(monkeypatch, sim, me, 5, foe.id, (ONE, -ONE),
                      source="bullet")
    sim.step()
    t = int(sim.tick)
    assert pic.last_hit[me.id] == (t, ONE, -ONE, "bullet")
    for _ in range(10):
        sim.step()
    assert pic.last_hit[me.id] == (t, ONE, -ONE, "bullet")   # age grows
    _hit_in_next_step(monkeypatch, sim, me, 3, -1, (0, 0), source="heat")
    sim.step()
    assert pic.last_hit[me.id] == (int(sim.tick), 0, 0, "heat")


def test_damage_window_sums_what_the_team_dealt_or_saw(tmp_path,
                                                        monkeypatch):
    """Property: a team's window sums the hits it DEALT (the attacker a
    living marine of the team) and the hits on a unit it SAW at the
    previous tick -- a THIRD side's hit on a watched enemy included -- and
    nothing else; entries fall out after ``window_ticks``. Breaks if the
    window reads an unseen hit, keys the saw rule on this tick's sighting,
    or forgets too early or late."""
    # me (team 0) sees foe (team 1) across the open box; a third side's unit
    # (team 2) stands beside the foe, no picture of its own
    sim, (me, foe, third) = _scene(tmp_path, "window",
                                   [(0, 10, 20), (1, 30, 20), (2, 30, 30)])
    pic = sim.team_picture
    W = pic.window_ticks
    assert W == max(1, int(round(float(sim._tps))))
    sim.step()
    assert pic.records(0)[foe.id].tick == sim.tick       # seen now
    # (1) the third side hits the foe my team watches: SEEN by team 0
    _hit_in_next_step(monkeypatch, sim, foe, 7, third.id)
    sim.step()
    t1 = int(sim.tick)
    assert pic.damage_last_s(0, foe.id) == 7.0
    assert pic.damage_last_s(1, foe.id) == 0.0           # team 1 dealt nothing
    # (2) my team deals a hit on the third side's unit, which team 1 saw too
    _hit_in_next_step(monkeypatch, sim, third, 4, me.id)
    sim.step()
    assert pic.damage_last_s(0, third.id) == 4.0         # dealt
    assert pic.damage_last_s(1, third.id) == 4.0         # saw
    # (3) a hit on a unit nobody watched: the foe behind a wall
    sim2, (me2, foe2) = _scene(tmp_path, "window2", [(0, 10, 20), (1, 40, 20)],
                               wall_x=30)
    sim2.step()
    assert foe2.id not in sim2.team_picture.records(0)
    _hit_in_next_step(monkeypatch, sim2, foe2, 9, -1)
    sim2.step()
    assert sim2.team_picture.damage_last_s(0, foe2.id) == 0.0
    # (4) the window: the first hit leaves W ticks after it landed
    while sim.tick - t1 < W - 1:
        sim.step()
    assert pic.damage_last_s(0, foe.id) == 7.0
    sim.step()
    assert pic.damage_last_s(0, foe.id) == 0.0


def test_bars_are_the_spawn_hp_and_the_turning_hp(tmp_path, monkeypatch):
    """Property: ``bars[uid]`` is the unit's hp at the first update at which
    it exists (its spawn: an enemy first seen wounded reads its true ratio)
    and ``[zombie] hp`` once it has turned. Breaks if the bar is taken at
    the first sighting or never re-taken at the conversion."""
    from simulation import unit_traits
    sim, (me, foe) = _scene(tmp_path, "bars", [(0, 10, 20), (1, 40, 20)],
                            wall_x=30)
    pic = sim.team_picture
    assert pic.bars[foe.id] == float(foe.current_hp) == (
        unit_traits.max_hp_q(foe) / unit_traits.ONE)
    _hit_in_next_step(monkeypatch, sim, foe, 30, -1)   # wounded, unseen
    sim.step()
    assert foe.current_hp < pic.bars[foe.id]           # the bar stays
    # a late spawn exists from its first update
    late = Unit("late", x=20, y=30, team=1, footprint=3)
    sim.add_unit(late)
    sim.step()
    assert pic.bars[late.id] == float(late.current_hp)
    # a conversion re-takes the bar
    _kill_in_next_step(monkeypatch, sim, foe, by_zombie=True)
    sim.step()
    assert foe.is_zombie
    assert pic.bars[foe.id] == float(CFG.zombie.hp)
    assert pic.death_tick[foe.id] == int(sim.tick)


def test_the_object_tracker_gives_a_seen_object_one_id_for_its_life(
        tmp_path):
    """Property: a grenade or charge a team-mate has a line of sight to gets
    the team's next object id at its first sighting and keeps it while it
    lives (seen again after a gap: the same id); the next first-seen object
    gets the next id; an object behind a wall gets none; a spent object is
    forgotten. Breaks if the id is a list index, a sim index or a live
    position."""
    from simulation import charges as ch
    sim, (me, foe) = _scene(tmp_path, "objects", [(0, 10, 20), (1, 40, 20)],
                            wall_x=30)
    pic = sim.team_picture
    seen = ch.PlantedCharge(20, 20, "demo_c4", foe.id, det_tick=None)
    hidden = ch.PlantedCharge(45, 20, "demo_c4", foe.id, det_tick=None)
    assert ch.place(sim, hidden) and ch.place(sim, seen)
    sim.step()
    ids0 = pic.object_ids(0)
    assert ids0[seen] == tp.OBJECT_UID_BASE and hidden not in ids0
    assert pic.seen_now[0] == [(seen, tp.OBJECT_UID_BASE)]
    assert pic.seen_now[1] == [(hidden, tp.OBJECT_UID_BASE)]   # team 1's own
    # the foe lobs a grenade: team 1 sees it, team 0 does not; the next
    # thing team 0 sees gets its NEXT id, not the sim's index
    sim.throw_grenade_intent(foe.id, ONE, 0, 24, 3 << 16, 0)
    for _ in range(10):
        sim.step()
    assert len(sim.projectiles) == 1
    assert sim.projectiles[0] not in pic.object_ids(0)
    assert pic.object_ids(1)[sim.projectiles[0]] == tp.OBJECT_UID_BASE + 1
    seen2 = ch.PlantedCharge(22, 24, "demo_c4", foe.id, det_tick=None)
    assert ch.place(sim, seen2)
    sim.step()
    assert pic.object_ids(0)[seen2] == tp.OBJECT_UID_BASE + 1
    assert pic.object_ids(0)[seen] == tp.OBJECT_UID_BASE
    # a spent charge is forgotten, the ids of the others unmoved
    seen.live = False
    sim.step()
    assert seen not in pic.object_ids(0)
    assert pic.object_ids(0)[seen2] == tp.OBJECT_UID_BASE + 1


def test_a_record_carries_the_facing_at_the_sighting(tmp_path):
    """Property: ``Record.facing_q`` is the unit's facing when it was last
    seen and does not follow a later turn out of sight. Breaks if the
    facing is read live."""
    sim, (me, foe) = _scene(tmp_path, "facing", [(0, 10, 20), (1, 40, 20)],
                            wall_x=30)
    place_facing(foe, 0)
    rec = sim.team_picture._write(sim, 0, foe)
    assert rec.facing_q == 0 == foe.facing_q
    sim.set_aim(foe.id, 0, -ONE)                 # turn north, unseen
    for _ in range(12):
        sim.step()
    assert foe.facing_q != 0
    assert sim.team_picture.records(0)[foe.id].facing_q == 0
