"""#121 — the first rule-based bot team (``control_bots_v1.BotTeamV1``,
controller "bots_v1" since #183 G8p; moved here from test_bots.py with the
law).

Properties of the five rules, the seed gate and the registry, on small
levels written through ``level_lib`` (never a hand-written toml). The
permanent G8p gate: every team on ``bots_v1`` plays ship_demo's pre-G8p pin
bit for bit (:func:`test_bots_v1_play_the_pre_g8p_pinned_match`). Matches run
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
from control_bots_v1 import BotTeamV1 as BotTeam  # noqa: E402
from control_bots_v1 import bot_v1_settings as bot_settings  # noqa: E402
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
                 controller="bots_v1")
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
    bullets draw from sim.draws). Breaks if a bot decision reads anything
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
        mate.set_pos(10.0, 24.0)       # pin the mate on the line
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
    # 10 s at the [speeds] walk (G1p-c): tiles per tick x ticks
    walked = 240 * 65536 / sim.speed_table.walk_cost_q
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
    # the grenade leaves the hand after the wind-up (G3p), along the facing
    # the bot has then -- which it holds on the lob's direction meanwhile
    windup = sim.weapons_tables.weapons.by_name["hand_grenade"]         .throw_windup_ticks
    _run(sim, [BotTeam(0)], 1 + windup)
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
    """Property: "bots_v1" is a registered match controller; the controller
    built for team 0 moves team 0's units and never team 1's. Breaks if the
    bots drive units of another team (one controller per team is the seam
    every later agent plugs into)."""
    assert "bots_v1" in control_source.registered_controllers()
    c = control_source.create_controller("bots_v1", 0)
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


# ---------------------------------------------------------------------------
# (h) the permanent G8p gate: bots_v1 IS the pre-G8p law
# ---------------------------------------------------------------------------
#: ship_demo seed 1 as the bots played it before #183 G8p (the G9p stage-2
#: pin, main 81cf3581): frozen here, never re-pinned for a bot change -- the
#: proof that the layered bots, not anything else, moved the shipped pin.
#: A change of the SIM (not of the bots) moves it like any other pin: then
#: re-pin it here too, with the rationale.
_PRE_G8P_SHIP_DEMO = dict(
    outcome="win", team=0, ticks=264,
    # #29 P3e re-pin (2026-11-12, the match path to integer): the shot
    # cone, the exposure / crit rolls and the FieldEdit noise are integer draws
    # (simulation.draws: new random values), and the blast falloff, the payload
    # deposits, the heat and poison rows, mitigation and the hp are integer
    # (re-rounded) -- the GAME moved by design, not only the hash.
    # docs/p3e_ab_2026-11-12.md: win rate, survivors, rounds per kill and the
    # blast outcomes within seed noise over 24-72 seeds.
    # (was a7f9fd78c8a13c2bf9fc38657b05eae042993affcd85bc777493724808fe2b8b)
    field_digest="2db5b2eaac6dcfafdbcfd85ec91772df"
                 "e838b1fddff573e90a09eba1de6bcb08",
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was 8df76cf7f0a3681c583c99e10f31624d8803f33c4ecd3adb3fdd0535e5a4cf46)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was b58afffbe86c3c2f47969fc5e29d7a4c4ab522e32bb14b7c7ff8e3ef67c51290)
    # #29 P3e re-pin (2026-11-12, the match path to integer): the shot
    # cone, the exposure / crit rolls and the FieldEdit noise are integer draws
    # (simulation.draws: new random values), and the blast falloff, the payload
    # deposits, the heat and poison rows, mitigation and the hp are integer
    # (re-rounded) -- the GAME moved by design, not only the hash.
    # docs/p3e_ab_2026-11-12.md: win rate, survivors, rounds per kill and the
    # blast outcomes within seed noise over 24-72 seeds.
    # (was cea75eeedc7e953e210c4a45201870a2c36519a67995a3d9c18192f34212dd42)
    tick_digest="48c4be2c99dd3a35dfd64a439732a6f9"
                "e92e77f0ea1aacfd58e8909e6285c524",
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was 1082d3b490f52183e4f047ed136219ecf2e53670e946b20a63ed989d46065a76)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was f9b510c240c4e80554afa85419f305aec6d4de7d4b94ae58c0d85d66d51a42d5)
    # #29 P3e re-pin (2026-11-12, the match path to integer): the shot
    # cone, the exposure / crit rolls and the FieldEdit noise are integer draws
    # (simulation.draws: new random values), and the blast falloff, the payload
    # deposits, the heat and poison rows, mitigation and the hp are integer
    # (re-rounded) -- the GAME moved by design, not only the hash.
    # docs/p3e_ab_2026-11-12.md: win rate, survivors, rounds per kill and the
    # blast outcomes within seed noise over 24-72 seeds.
    # (was 1a3e3d18b044c64c90472ab6a628550e3b7b5c1989f9a36ee36723784941da73)
    unit_stream="4503c2306583d04f3f85bbf87131ef8a"
                "5101b138cbc02fda0d9b5d01a5a9bd2f")


def v1_spec(level):
    """``level``'s match with every marine team on ``bots_v1`` (the horde
    as configured): the pre-G8p match, for the tests that reproduce a
    historical pin."""
    spec = level.match
    return replace(spec, teams=tuple(replace(t, controller="bots_v1")
                                     for t in spec.teams))


def test_bots_v1_play_the_pre_g8p_pinned_match():
    """Property (G8p's proof of cause): ship_demo seed 1 with every team on
    ``bots_v1`` ends exactly as the shipped match did before G8p -- outcome,
    ticks, final fields, tick digest and the per-tick unit stream. Breaks if
    the first bots' law changed when it moved to ``control_bots_v1`` (a
    renamed key read differently, a shared helper edited), or if the sim
    changed (then re-pin with the rationale)."""
    from test_zombie_control import _UnitStream
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    stream = _UnitStream()
    o = run_match(level, v1_spec(level), on_tick=stream)
    got = dict(outcome=o.result.outcome, team=o.result.team,
               ticks=o.result.ticks, field_digest=o.field_digest,
               tick_digest=o.tick_digest, unit_stream=stream.h.hexdigest())
    assert got == _PRE_G8P_SHIP_DEMO
