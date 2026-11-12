"""#195 1d -- the team picture (``simulation/team_picture.py``).

Step (i) properties (``docs/rl_agent_rung1_note_1d_2026-11-12.md``, "Property
tests"): the conductor's one update per tick, never on demand; a record is
plain values taken at its sighting, its motion from two consecutive
sightings, kept exactly ``memory_ticks``; nothing the team never saw; the
one lane rasteriser. The bit-for-bit gate of step (i) is the pins under
``docs/rl_agent_rung1_1d_pins/`` (``tools/bot_tournament.py``) plus the two
pinned matches (``test_zombie_control`` / ``test_bots_v1``).

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
