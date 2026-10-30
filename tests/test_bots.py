"""#121 — the rule-based bot team (``control_bots.BotTeam``, controller "bots").

Properties of the five rules, the seed gate and the registry, on small
levels written through ``level_lib`` (never a hand-written toml). Matches run
through ``run_match`` (the one match loop); single-rule checks step a
``Simulation`` with one ``BotTeam`` and hand-placed units.

Geometry note: a unit faces NORTH at spawn, so the scenes put what a bot
must see north of it (its 55 deg cone), and what it must not see behind a
wall.
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
from config import CFG, Namespace  # noqa: E402
from control_bots import BotTeam, bot_settings  # noqa: E402
from field_ab_harness import capture_snapshot  # noqa: E402
from field_digest import tick_digest  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.events import ShotFiredEvent  # noqa: E402
from simulation.match import MatchSpec, MatchTeam, SpawnRoom  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.stats import effective_vitality  # noqa: E402
from simulation.unit import Unit  # noqa: E402

W, H = 60, 40
BOTS = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                 controller="bots")
# Team 0 bottom-left, team 1 top-right: in the open box they see each other
# once they face the enemy room.
ROOMS = [
    SpawnRoom(team=0, x=2, y=28, w=8, h=8, max_per_team=4),
    SpawnRoom(team=1, x=48, y=2, w=8, h=8, max_per_team=4),
]


def _level(tmp_path, name, *, wall_x=None, gap=None, spec=None):
    """An open hull box (``W`` x ``H``, 1 m tiles) through the level_lib
    writers; ``wall_x`` adds a full-height, two-tile hull wall at that
    column, so nothing sees across it -- and nothing crosses it unless
    ``gap = (y0, y1)`` opens rows ``[y0, y1)`` of it."""
    lvl = tmp_path / name
    lvl.mkdir()
    level_lib.write_level_header(lvl, name=name, tile_size_m=1.0)
    tm = np.full((H, W), MAT_HULL, dtype=np.int32)
    tm[1:H - 1, 1:W - 1] = MAT_AIR
    if wall_x is not None:
        tm[:, wall_x:wall_x + 2] = MAT_HULL
        if gap is not None:
            tm[gap[0]:gap[1], wall_x:wall_x + 2] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (W, H)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", ROOMS)
    if spec is not None:
        level_lib.write_match(lvl, spec)
    return level_loader.load(name, str(tmp_path))


def _sim(level, units):
    """A running ContinuousRealtime sim holding ``units`` (team, x, y)."""
    sim = Simulation(level, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    out = []
    for i, (team, x, y) in enumerate(units):
        u = Unit(f"u{i}", x=x, y=y, team=team, footprint=3)
        sim.add_unit(u)
        out.append(u)
    return sim, out


def _run(sim, bots, ticks, on_tick=None):
    for _ in range(ticks):
        for b in bots:
            b.handle_frame(sim, None)
        sim.step()
        if on_tick is not None:
            on_tick(sim)


def _room_centre(team):
    r = next(r for r in ROOMS if r.team == team)
    return (r.x + r.w // 2, r.y + r.h // 2)


def _dist(u, p):
    cx, cy = u.x + u.footprint // 2, u.y + u.footprint // 2
    return float(np.hypot(cx - p[0], cy - p[1]))


# ---------------------------------------------------------------------------
# (a) the seed gate
# ---------------------------------------------------------------------------
def test_bot_match_one_seed_one_digest_another_seed_differs(tmp_path):
    """Property: a bots-vs-bots match is a pure function of config + seed --
    two runs of one seed give one digest, another seed another (the bots'
    bullets draw from sim.rng). Breaks if a bot decision reads anything
    unsynced (wall clock, set order, an OS-seeded draw) or if the bots never
    fire (seed 1 == seed 2)."""
    spec = MatchSpec(teams=(BOTS, BOTS), seed=1, time_limit_s=6.0)
    lvl = _level(tmp_path, "duel", spec=spec)
    a = run_match(lvl)
    b = run_match(lvl)
    c = run_match(lvl, seed=2)
    assert (a.tick_digest, a.result) == (b.tick_digest, b.result)
    assert c.tick_digest != a.tick_digest


def test_the_window_plays_the_runner_s_match(tmp_path):
    """Property: `main.py --control match`'s MatchObserver, driven the way
    main.py drives it (initial_seed / initial_ruleset / spawn_units, then
    on_tick before every step), ends in the SAME state as tools/run_match.py
    -- Erik watches the match the bench measures. Breaks if the observer
    calls the controllers per frame instead of per tick, spawns other units
    or seeds the sim differently."""
    from control_match import MatchObserver
    spec = MatchSpec(teams=(BOTS, BOTS), seed=3, time_limit_s=3.0)
    lvl = _level(tmp_path, "watch", spec=spec)
    ref = run_match(lvl)
    obs = control_source.create_control_source("match", level=lvl)
    assert isinstance(obs, MatchObserver)
    sim = Simulation(lvl, seed=obs.initial_seed(42), breach_physics=bp,
                     enable_recorder=False, ruleset=obs.initial_ruleset())
    sim.set_paused(obs.starts_paused())
    assert obs.spawn_units(sim, lvl)
    while not sim.is_paused():
        obs.on_tick(sim)
        sim.step()
    assert obs.rules.result(sim) == ref.result
    assert tick_digest(capture_snapshot(sim)) == ref.tick_digest


# ---------------------------------------------------------------------------
# (b) shoot on sight
# ---------------------------------------------------------------------------
def _shots_by(sim, uid):
    n = [0]

    def watch(s):
        n[0] += sum(1 for e in s.tick_events
                    if isinstance(e, ShotFiredEvent) and e.unit_id == uid
                    and e.launch)
    return n, watch


def test_a_bot_fires_at_an_enemy_in_sight_and_not_without_one(tmp_path):
    """Property: a bot with an enemy in its sight fires within half a second;
    a bot whose enemy is behind a wall never fires. Breaks if the trigger is
    held without a seen target (wasted fire, a friendly-fire hose) or if a
    seen enemy does not pull it."""
    lvl = _level(tmp_path, "sight")
    sim, (bot, foe) = _sim(lvl, [(0, 10, 25), (1, 10, 8)])   # foe due north
    n, watch = _shots_by(sim, bot.id)
    _run(sim, [BotTeam(0)], 12, watch)
    assert n[0] > 0

    lvl2 = _level(tmp_path, "blind", wall_x=29)
    sim2, (bot2, foe2) = _sim(lvl2, [(0, 10, 25), (1, 45, 8)])
    n2, watch2 = _shots_by(sim2, bot2.id)
    _run(sim2, [BotTeam(0)], 48, watch2)
    assert n2[0] == 0


def test_a_bot_holds_fire_with_a_teammate_on_the_line(tmp_path):
    """Property: a bot does not shoot through a teammate standing on its
    line of fire a few tiles ahead (friendly fire is ON in the game; the bot
    just does not aim at its own). Breaks if the line check is lost."""
    lvl = _level(tmp_path, "line")
    sim, (bot, mate, foe) = _sim(lvl, [(0, 10, 30), (0, 10, 24),
                                       (1, 10, 8)])
    # The mate stands on the line (6 tiles north); only `bot` is driven,
    # the mate idles (a BotTeam would also drive it), so a stand-in team
    # controller for the shooter alone:
    team = BotTeam(0)
    n, watch = _shots_by(sim, bot.id)
    for _ in range(12):
        team.handle_frame(sim, None)
        sim.clear_move_dir(mate.id)
        sim.set_trigger(mate.id, False)
        mate.x, mate.y = 10.0, 24.0       # pin the mate on the line
        sim.step()
        watch(sim)
    assert n[0] == 0


# ---------------------------------------------------------------------------
# (c) advance
# ---------------------------------------------------------------------------
def test_a_bot_with_nothing_in_sight_advances_on_the_enemy_spawn(tmp_path):
    """Property: with no enemy ever seen, a bot walks toward the enemy spawn
    room -- its distance to the room's centre shrinks over 10 s by at least
    half the distance a walking marine covers. Breaks if the advance rule
    stands still, walks elsewhere, or loses its path."""
    lvl = _level(tmp_path, "advance")
    sim, (bot,) = _sim(lvl, [(0, 4, 30)])
    goal = _room_centre(1)
    d0 = _dist(bot, goal)
    _run(sim, [BotTeam(0)], 240)
    walked = 10.0 / float(CFG.movement.marine_attack_seconds_per_tile)
    assert d0 - _dist(bot, goal) >= 0.5 * walked


# ---------------------------------------------------------------------------
# (d) grenade a group
# ---------------------------------------------------------------------------
def test_the_grenade_rule_throws_at_a_pair_and_not_near_a_teammate(tmp_path):
    """Property: two enemies close together in sight, far enough away, draw
    a frag aimed at their centre (it lands within two tiles of it); the same
    pair with a teammate inside the danger radius of their centre draws none.
    Breaks if the rule throws without a group, at the wrong point, or blows
    up its own side."""
    s = bot_settings(CFG)
    lvl = _level(tmp_path, "frag")
    pair = [(1, 7, 10), (1, 13, 10)]          # centres (8,11) / (14,11)
    centre = (11, 11)
    sim, units = _sim(lvl, [(0, 10, 28)] + pair)
    bot = units[0]
    g0 = bot.has_grenade
    assert g0 > 0
    _run(sim, [BotTeam(0)], 1)
    assert bot.has_grenade == g0 - 1
    proj = sim.projectiles[-1]
    assert abs(proj.target_fx - 0.5 - centre[0]) <= 2.0
    assert abs(proj.target_fy - 0.5 - centre[1]) <= 2.0

    lvl2 = _level(tmp_path, "frag_mate")
    mate_xy = (centre[0] - 1, centre[1] + s.danger_radius - 3)   # inside
    sim2, units2 = _sim(lvl2, [(0, 10, 28), (0,) + mate_xy] + pair)
    bot2 = units2[0]
    _run(sim2, [BotTeam(0)], 24)
    assert bot2.has_grenade == g0 and not sim2.projectiles


# ---------------------------------------------------------------------------
# (e) fall back
# ---------------------------------------------------------------------------
def test_a_lone_wounded_bot_falls_back_and_a_healthy_one_does_not(tmp_path):
    """Property: alone and below fallback_hp_frac of its vitality, a bot
    walks toward its OWN spawn room; at full health the same bot walks away
    from it (toward the enemy). With a teammate within fallback_r the
    wounded bot advances too. Breaks if the fall-back rule never fires or
    fires with a friend at hand."""
    s = bot_settings(CFG)
    home = _room_centre(0)

    def moved_home(name, wounded, with_mate):
        lvl = _level(tmp_path, name)
        spots = [(0, 28, 18)]
        if with_mate:
            spots.append((0, 28, 18 + s.fallback_r // 2))
        sim, units = _sim(lvl, spots)
        bot = units[0]
        if wounded:
            bot.current_hp = 0.3 * effective_vitality(bot)
        d0 = _dist(bot, home)
        _run(sim, [BotTeam(0)], 96)
        return d0 - _dist(bot, home)

    assert moved_home("hurt", True, False) > 1.0
    assert moved_home("fine", False, False) < -1.0
    assert moved_home("hurt_mate", True, True) < -1.0


# ---------------------------------------------------------------------------
# (f) the [bots] door
# ---------------------------------------------------------------------------
_GOOD = dict(replan_ticks=24, group_radius=6, danger_radius=8,
             grenade_cooldown_s=4.0, fallback_hp_frac=0.5, fallback_r=15,
             friendly_line_tiles=12)


@pytest.mark.parametrize("key", sorted(_GOOD))
def test_bots_settings_refuse_a_missing_key(key):
    """Property: [bots] without any one of its keys is refused, naming the
    key; the shipped config passes. Breaks if a threshold silently falls back
    to a default."""
    bot_settings(CFG)
    cfg = Namespace({"bots": {k: v for k, v in _GOOD.items() if k != key}})
    with pytest.raises(ValueError, match=key):
        bot_settings(cfg)
    with pytest.raises(ValueError, match=r"\[bots\]"):
        bot_settings(Namespace({}))


@pytest.mark.parametrize("key,bad", [("replan_ticks", 2.5),
                                     ("fallback_hp_frac", 1.5),
                                     ("danger_radius", 0),
                                     ("grenade_cooldown_s", True)])
def test_bots_settings_refuse_a_bad_value(key, bad):
    """Property: a mistyped or out-of-range [bots] value is refused."""
    cfg = Namespace({"bots": dict(_GOOD, **{key: bad})})
    with pytest.raises(ValueError, match=key):
        bot_settings(cfg)


# ---------------------------------------------------------------------------
# (g) the registry
# ---------------------------------------------------------------------------
def test_registry_builds_bots_that_drive_only_their_team(tmp_path):
    """Property: "bots" is a registered match controller; the controller
    built for team 0 moves team 0's units and never team 1's. Breaks if the
    bots drive units of another team (one controller per team is the seam
    every later agent plugs into)."""
    assert "bots" in control_source.registered_controllers()
    c = control_source.create_controller("bots", 0)
    assert isinstance(c, BotTeam) and c.team == 0
    lvl = _level(tmp_path, "teams", wall_x=29, gap=(30, 38))
    sim, units = _sim(lvl, [(0, 4, 30), (0, 10, 30), (1, 50, 4),
                            (1, 44, 4)])
    before = {u.id: (u.x, u.y, u.facing) for u in units}
    _run(sim, [c], 48)
    for u in units:
        if u.team == 1:
            assert (u.x, u.y, u.facing) == before[u.id]
        else:
            assert (u.x, u.y) != before[u.id][:2]


def test_shipped_levels_play_bots_by_default():
    """Property: the playground's and ship_demo's match configs put every
    marine team on "bots" (#121), and idle stays registered. Breaks if a
    level reverts to idle teams (Erik's HUMAN-TEST command would show a
    standing match)."""
    assert "idle" in control_source.registered_controllers()
    for name in ("playground", "ship_demo"):
        spec = level_loader.load(name).match
        assert spec is not None
        assert {t.controller for t in spec.teams} == {"bots"}, name
