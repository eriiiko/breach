"""#183 G8p -- the layered bot team (``control_bots.BotTeam``).

Properties of the three layers (team / unit / position), the position
features one by one on hand-built tiles, the plays, the door, determinism and
the tournament's friendly-fire counter. Small levels are written through
``level_lib`` (``test_bots_v1._level``: an open hull box, optionally split by
a wall with a gap); matches run through ``run_match`` (the one match loop);
single-layer checks step a ``Simulation`` with one ``BotTeam`` and
hand-placed units.

Geometry note: a unit faces NORTH at spawn (its 55 deg cone), so the scenes
put what a bot must see north of it.
"""
from __future__ import annotations

import ast
import sys
from dataclasses import replace
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

import control_bots  # noqa: E402
import control_source  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
import match_replay  # noqa: E402
from config import CFG, Namespace  # noqa: E402
from control_bots import (BotTeam, UnitContext,  # noqa: E402
                          bot_settings, evaluate_features)
from field_ab_harness import capture_snapshot  # noqa: E402
from field_digest import tick_digest  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.events import ShoveEvent, UnitHitEvent  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.match import MatchSpec, MatchTeam, SpawnRoom  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.team_picture import draw_lanes  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from test_bots_v1 import ROOMS, _level  # noqa: E402

BOTS = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                 controller="bots")


def _sim(level, units, zombies=()):
    """A running ContinuousRealtime sim holding ``units`` (team, x, y) and
    zombies at ``zombies`` (x, y)."""
    sim = Simulation(level, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    out = []
    for i, (team, x, y) in enumerate(units):
        u = Unit(f"u{i}", x=x, y=y, team=team, footprint=3)
        sim.add_unit(u)
        out.append(u)
    zs = []
    for j, (x, y) in enumerate(zombies):
        z = Unit(f"z{j}", x=x, y=y, team=ZOMBIE_TEAM, footprint=3)
        sim.add_unit(z)
        zs.append(z)
    sim.start_team_picture((0, 1))
    return sim, out, zs


def _run(sim, bots, ticks, on_tick=None):
    for _ in range(ticks):
        for b in bots:
            b.handle_frame(sim, None)
        sim.step()
        if on_tick is not None:
            on_tick(sim)


def _box(tmp_path, name, tm_edit=None):
    """A 60 x 40 open hull box (``test_bots_v1._level``), with
    ``tm_edit(tm)`` applied to its tilemap before writing."""
    lvl = tmp_path / name
    lvl.mkdir()
    level_lib.write_level_header(lvl, name=name, tile_size_m=1.0)
    tm = np.full((40, 60), MAT_HULL, dtype=np.int32)
    tm[1:39, 1:59] = MAT_AIR
    if tm_edit is not None:
        tm_edit(tm)
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (60, 40)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", ROOMS)
    return level_loader.load(name, str(tmp_path))


#: The box's spawn rooms with the horde in the middle.
MATCH_ROOMS = list(ROOMS) + [SpawnRoom(team=ZOMBIE_TEAM, x=26, y=16, w=8,
                                       h=8, max_per_team=4)]


def _match_level(tmp_path, name, spec):
    """The 60 x 40 box with both marine rooms, the horde's room in the
    middle and ``spec`` as its match.toml (all through level_lib)."""
    lvl = tmp_path / name
    lvl.mkdir()
    level_lib.write_level_header(lvl, name=name, tile_size_m=1.0)
    tm = np.full((40, 60), MAT_HULL, dtype=np.int32)
    tm[1:39, 1:59] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (60, 40)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", MATCH_ROOMS)
    level_lib.write_match(lvl, spec)
    return level_loader.load(name, str(tmp_path))


def _ctx(bots, sim, u, cands):
    """The position context of bot ``u`` over the candidate anchors
    ``cands`` [(x, y), ...], after one team update."""
    mine = [m for m in sim.units if m.team == u.team and m.alive
            and not m.is_zombie]
    if bots.brain.updated_at is None:
        bots.handle_frame(sim, None)
    c = UnitContext(sim, u, bots.brain, mine, [], int(sim.tick))
    xs = np.array([p[0] for p in cands], dtype=np.int64)
    ys = np.array([p[1] for p in cands], dtype=np.int64)
    return bots.position_context(c, xs, ys)


def _know(bots, sim, e):
    """Put enemy ``e`` in the team's picture, seen now (the picture's own
    writer)."""
    sim.team_picture._write(sim, bots.team, e)


# ---------------------------------------------------------------------------
# (a) determinism: one seed one match, replayed exactly
# ---------------------------------------------------------------------------
def test_a_bots_match_is_identical_twice_and_replays_exactly(tmp_path):
    """Property: a layered-bots match (the horde in) is a pure function of
    config + seed -- two runs give one digest, its recording replays with
    every checkpoint matching, another seed plays another match. Breaks if a
    bot decision reads anything unsynced (wall clock, a set's order, an
    OS-seeded draw, an object id), or if the bots' tie-break ignores the
    seed (seed 1 == seed 2)."""
    spec = MatchSpec(teams=(BOTS, BOTS), seed=1, time_limit_s=8.0, zombies=2)
    lvl = _match_level(tmp_path, "duel", spec)
    a = run_match(lvl, record=True, checkpoint_every_ticks=12)
    b = run_match(lvl)
    assert (a.tick_digest, a.result) == (b.tick_digest, b.result)
    back = match_replay.read_recording_bytes(
        match_replay.recording_bytes(a.recording)) \
        if hasattr(match_replay, "recording_bytes") else a.recording
    r = run_match(lvl, replay=back)
    assert r.tick_digest == a.tick_digest
    assert r.checkpoints_verified >= 2
    c = run_match(lvl, seed=2)
    assert c.tick_digest != a.tick_digest


def test_the_shipped_levels_play_the_layered_bots():
    """Property: the playground's and ship_demo's match configs put every
    marine team on "bots" -- the layered bots -- and ``bots_v1`` stays
    registered beside them (a ladder of opponents). Breaks if a level reverts
    to idle teams or the old bots lose their registry row."""
    names = control_source.registered_controllers()
    assert "idle" in names and "bots_v1" in names
    assert isinstance(control_source.create_controller("bots", 0), BotTeam)
    for name in ("playground", "ship_demo"):
        spec = level_loader.load(name).match
        assert {t.controller for t in spec.teams} == {"bots"}, name


# ---------------------------------------------------------------------------
# (b) the [bots] door
# ---------------------------------------------------------------------------
def _shipped_bots():
    """The shipped [bots] table as a plain dict (subtables too)."""
    def plain(ns):
        return {k: (plain(v) if isinstance(v, Namespace) else v)
                for k, v in vars(ns).items()}
    return plain(CFG.bots)


@pytest.mark.parametrize("key", sorted(control_bots._SCALARS))
def test_the_door_refuses_a_missing_scalar_key(key):
    """Property: [bots] without any one of the layered bots' scalar keys is
    refused, naming the key; the shipped config passes. Breaks if a
    threshold silently falls back to a default."""
    bot_settings(CFG)
    d = _shipped_bots()
    del d[key]
    with pytest.raises(ValueError, match=key):
        bot_settings(Namespace({"bots": d}))


@pytest.mark.parametrize("sub,rows", [
    ("features", [f.name for f in control_bots.FEATURES]),
    ("actions", [a.name for a in control_bots.ACTIONS]),
    ("plays", [p.name for p in control_bots.PLAYS])])
def test_the_door_refuses_a_missing_or_unknown_weight(sub, rows):
    """Property: each weight table needs a row for every registered feature
    / action / play and refuses one it does not know, by name. Breaks if a
    new registry row runs with no weight, or a typo'd weight is ignored."""
    for name in rows:
        d = _shipped_bots()
        del d[sub][name]
        with pytest.raises(ValueError, match=name):
            bot_settings(Namespace({"bots": d}))
    d = _shipped_bots()
    d[sub]["no_such_row"] = 1.0
    with pytest.raises(ValueError, match="no_such_row"):
        bot_settings(Namespace({"bots": d}))


# ---------------------------------------------------------------------------
# (c) the position features, one by one, on hand-built tiles
# ---------------------------------------------------------------------------
def _feature(ctx, name):
    return list(evaluate_features(ctx)[name])


def test_feature_sees_target(tmp_path):
    """Property: `sees_target` is yes on an open tile with a line of sight
    to the target and no behind a wall. Breaks if the feature stops casting
    the sight oracle's line or reads the wrong end."""
    def wall(tm):
        tm[20:23, 20:30] = MAT_HULL           # a wall south of the target
    lvl = _box(tmp_path, "sees", wall)
    sim, (bot, foe), _ = _sim(lvl, [(0, 24, 30), (1, 24, 10)])
    bots = BotTeam(0)
    bots.handle_frame(sim, None)
    _know(bots, sim, foe)
    ctx = _ctx(bots, sim, bot, [(10, 30), (24, 25)])
    ctx.target = (25, 11)
    assert _feature(ctx, "sees_target") == [True, False]


def test_feature_outside_fire_lanes(tmp_path):
    """Property: `outside_fire_lanes` is no on a tile on a shooting
    teammate's line to its target and yes beside it. Breaks if the lanes are
    not rasterised from the shooters, or a bot's own lane counts against
    it."""
    lvl = _box(tmp_path, "lanes")
    sim, (bot, mate, foe), _ = _sim(lvl, [(0, 40, 30), (0, 10, 30),
                                          (1, 10, 5)])
    bots = BotTeam(0)
    bots.handle_frame(sim, None)
    s = bots.s
    h, w = sim.gmap.material.shape
    bots.brain.lanes = draw_lanes(
        h, w, [(mate.id, (mate.center_tile_x(), mate.center_tile_y()),
                (foe.center_tile_x(), foe.center_tile_y()))],
        s.lane_cell, s.lane_halfwidth)
    ctx = _ctx(bots, sim, bot, [(10, 18), (40, 18)])
    assert _feature(ctx, "outside_fire_lanes") == [False, True]


def test_feature_spaced(tmp_path):
    """Property: `spaced` is no within `spacing_tiles` of a teammate and yes
    beyond. Breaks if teammates (#142's stacking) stop counting."""
    lvl = _box(tmp_path, "spaced")
    sim, (bot, mate), _ = _sim(lvl, [(0, 10, 30), (0, 30, 30)])
    bots = BotTeam(0)
    sp = bots.s.spacing_tiles
    ctx = _ctx(bots, sim, bot, [(30 + 1, 30), (30 - sp - 3, 30)])
    assert _feature(ctx, "spaced") == [False, True]


def test_feature_cover(tmp_path):
    """Property: `cover` is yes on a tile with a wall between it and the
    threat (on the threat's side of the body) and no in the open. Breaks if
    the ring is read on the wrong side or not at all."""
    def wall(tm):
        tm[24, 8:16] = MAT_HULL               # a wall just north of (10, 25)
    lvl = _box(tmp_path, "cover", wall)
    sim, (bot,), _ = _sim(lvl, [(0, 40, 30)])
    bots = BotTeam(0)
    ctx = _ctx(bots, sim, bot, [(10, 25), (40, 25)])
    ctx.threat = (11, 5)
    got = _feature(ctx, "cover")
    ctx.threat = (41, 5)
    assert got[0] and not _feature(ctx, "cover")[1]


def test_feature_hidden_from_others(tmp_path):
    """Property: `hidden_from_others` is no where another known enemy (not
    the target) has a line of sight and yes behind a wall from it. Breaks if
    the other enemies are not cast against."""
    def wall(tm):
        tm[20:23, 30:50] = MAT_HULL
    lvl = _box(tmp_path, "hidden", wall)
    sim, (bot, foe), _ = _sim(lvl, [(0, 10, 30), (1, 40, 10)])
    bots = BotTeam(0)
    ctx = _ctx(bots, sim, bot, [(10, 25), (40, 26)])
    ctx.others = [(41, 11)]
    assert _feature(ctx, "hidden_from_others") == [False, True]


def test_feature_closer_to_objective(tmp_path):
    """Property: `closer_to_objective` is yes on a tile at least
    `progress_steps` nearer the play's objective (by the flood's path
    length) and no on one farther. Breaks if the objective field is not
    read, or its sense is inverted."""
    from simulation import horde
    lvl = _box(tmp_path, "closer")
    sim, (bot,), _ = _sim(lvl, [(0, 30, 20)])
    bots = BotTeam(0)
    ctx = _ctx(bots, sim, bot, [(20, 20), (40, 20)])
    ctx.objective_field = horde.flood(sim.gmap, bots.tile_ok(sim), 3,
                                      [(20, 5)])
    assert _feature(ctx, "closer_to_objective") == [True, False]


def test_feature_safe_ground(tmp_path):
    """Property: `safe_ground` is no on a tile the field accessor reads as
    burning, densely smoky or vacuum and yes on a quiet tile -- read
    through the sensors' seam (``EntityFieldAccessor.sample_many``). Breaks
    if a hazard stops counting or the feature reads a raw plane."""
    from simulation.sensor_accessor import Channel
    lvl = _box(tmp_path, "safe")
    sim, (bot,), _ = _sim(lvl, [(0, 30, 20)])
    bots = BotTeam(0)
    real = bots.accessor(sim)
    hazards = {Channel.FIRE: (21, 11, 1 << 16),
               Channel.SMOKE: (21, 21, bots.smoke_q),
               Channel.PRESSURE: (21, 31, None)}

    class _Acc:
        def sample_many(self, ch, fys, fxs):
            v = real.sample_many(ch, fys, fxs).copy()
            if ch in hazards:
                hy, hx, val = hazards[ch]
                hit = (np.asarray(fys) == hy) & (np.asarray(fxs) == hx)
                v[hit] = 0 if val is None else val
            return v
    bots._acc = (sim.gmap, _Acc())
    ctx = _ctx(bots, sim, bot, [(10, 20), (20, 20), (30, 20), (40, 20)])
    assert _feature(ctx, "safe_ground") == [False, False, False, True]


def test_feature_reachable(tmp_path):
    """Property: `reachable` is yes on a tile a few steps away and no inside
    a walled-off pocket. Breaks if the flood from the bot is not read."""
    def pocket(tm):
        tm[5:15, 25:35] = MAT_HULL
        tm[7:13, 27:33] = MAT_AIR              # sealed inside
    lvl = _box(tmp_path, "reach", pocket)
    sim, (bot,), _ = _sim(lvl, [(0, 40, 20)])
    bots = BotTeam(0)
    ctx = _ctx(bots, sim, bot, [(42, 20), (28, 8)])
    assert _feature(ctx, "reachable") == [True, False]


# ---------------------------------------------------------------------------
# (d) the team layer: focus fire
# ---------------------------------------------------------------------------
def test_focus_fire_the_shooters_share_the_most_exposed_then_weakest(tmp_path):
    """Property: with two enemies in sight, the team's shared target is the
    one the exposure law has raised most, and every shooter picks it; with
    no exposure the weaker (fewer hit points) is the shared target. Breaks
    if each bot picks its own nearest, or the order is not exposure, then
    hit points, then distance."""
    from control_bots import KINETIC
    lvl = _box(tmp_path, "focus")
    sim, (a, b, e1, e2), _ = _sim(lvl, [(0, 20, 30), (0, 32, 30),
                                        (1, 22, 14), (1, 30, 14)])
    bots = BotTeam(0)
    e2.exposure[KINETIC] = 1 << 16            # e2 is under fire
    bots.handle_frame(sim, None)
    assert bots.brain.target is e2
    for u in (a, b):
        assert bots.fire_target(sim, u, [a, b], [e1, e2], bots.brain) is e2
    sim2, (a2, b2, f1, f2), _ = _sim(lvl, [(0, 20, 30), (0, 32, 30),
                                           (1, 22, 14), (1, 30, 14)])
    f1.current_hp -= 30.0                     # f1 is the weaker
    bots2 = BotTeam(0)
    bots2.handle_frame(sim2, None)
    assert bots2.brain.target is f1
    for u in (a2, b2):
        assert bots2.fire_target(sim2, u, [a2, b2], [f1, f2],
                                 bots2.brain) is f1


# ---------------------------------------------------------------------------
# (e) the throw action never flashes a teammate
# ---------------------------------------------------------------------------
def test_no_flashbang_where_a_teammate_would_face_the_flash(tmp_path):
    """Property: the flashbang action targets a known enemy out of the
    team's sight -- and is never selected while a teammate (the thrower
    included) is within the flash's blinding radius with a line of sight to
    where it lands (the flash blinds anyone looking: G3p); with that
    teammate behind the wall the same throw is chosen. Breaks if the
    action ignores teammates, the radius or the line of sight."""
    lvl = _level(tmp_path, "flash", wall_x=29)        # a full-height wall
    for mate_xy, safe in (((40, 26), False), ((20, 30), True)):
        sim, (bot, mate, foe), _ = _sim(lvl, [(0, 20, 20), (0,) + mate_xy,
                                              (1, 36, 20)])
        bots = BotTeam(0)
        bots.handle_frame(sim, None)
        _know(bots, sim, foe)                         # seen a moment ago
        bots.brain.play = "bound"
        c = UnitContext(sim, bot, bots.brain, [bot, mate], [], int(sim.tick))
        p = bots.flash_point(c)
        assert (p is not None) == safe, mate_xy
    # and the shipped radius is the flash row's: a teammate inside it with a
    # line of sight to the point is unsafe, one outside it is not
    landing = (foe.center_tile_x(), foe.center_tile_y())
    r = bots.blind_r
    assert r > 0
    near = SimpleNamespace(center_tile_x=lambda: landing[0] + r - 1,
                           center_tile_y=lambda: landing[1])
    far = SimpleNamespace(center_tile_x=lambda: landing[0] + r + 1,
                          center_tile_y=lambda: landing[1])
    assert not bots.flash_safe(sim, [near], landing)
    assert bots.flash_safe(sim, [far], landing)


# ---------------------------------------------------------------------------
# (f) the Fall back play: backwards, facing the threat, shoving
# ---------------------------------------------------------------------------
def test_fall_back_walks_backwards_facing_the_zombie_and_shoves_it(tmp_path):
    """Property: with a zombie at its side the team falls back: the bot's
    facing stays on the zombie (within the vision cone) while it moves away
    from it -- the backward walk -- and it shoves the zombie when it is
    adjacent. Breaks if the bot turns its back to walk forward, stands, or
    never uses the shove."""
    import math
    lvl = _box(tmp_path, "fallback")
    sim, (bot,), (z,) = _sim(lvl, [(0, 30, 24)], zombies=[(30, 20)])
    bot.current_hp = 1.0e6                    # survive the bites
    bots = BotTeam(0)
    shoves, facing_ok, away = [0], [], []

    def watch(s):
        shoves[0] += sum(1 for ev in s.tick_events
                         if isinstance(ev, ShoveEvent)
                         and ev.unit_id == bot.id)
    for _ in range(96):
        prev = (bot.x, bot.y)
        _run(sim, [bots], 1, watch)
        if bots.brain.play != "fall_back" or not z.alive:
            continue
        dx = z.center_tile_x() - bot.center_tile_x()
        dy = z.center_tile_y() - bot.center_tile_y()
        if dx == 0 and dy == 0:
            continue
        want = math.atan2(-dy, dx)
        off = abs((bot.facing - want + math.pi) % (2 * math.pi) - math.pi)
        facing_ok.append(off <= math.radians(55))
        mx, my = bot.x - prev[0], bot.y - prev[1]
        if mx or my:
            away.append(mx * dx + my * dy < 0)
    assert shoves[0] >= 1
    assert facing_ok and sum(facing_ok) >= 0.8 * len(facing_ok)
    assert away and sum(away) >= 0.6 * len(away)


# ---------------------------------------------------------------------------
# (g) the unit layer commits (#138) and keeps its distance (#142)
# ---------------------------------------------------------------------------
def test_a_bot_keeps_its_position_between_decisions(tmp_path):
    """Property (#138): a bot's chosen position changes only on a tick it
    decides -- never between two decisions. Breaks if the position layer is
    re-run every tick (the judder) or the executor rewrites the goal."""
    from simulation.match_session import MatchSession
    spec = MatchSpec(teams=(BOTS, BOTS), seed=4, time_limit_s=10.0, zombies=2)
    lvl = _match_level(tmp_path, "commit", spec)
    ses = MatchSession(lvl, None, breach_physics=bp)
    bots = [c for c in ses.controllers if isinstance(c, BotTeam)]
    last, changes, decided = {}, 0, 0
    while not ses.is_terminal() and ses.sim.tick < 200:
        ses.tick()
        for b in bots:
            for uid, st in b._state.items():
                fresh = st.decided_at == ses.sim.tick - 1
                if uid in last and st.goal != last[uid]:
                    changes += 1
                    assert fresh, (uid, st.goal)
                decided += fresh
                last[uid] = st.goal
    assert decided > 0 and changes > 0                 # non-vacuous


def _corridor(tm):
    """A corridor 7 tiles high from the team-0 room to the team-1 room."""
    tm[1:39, 12:46] = MAT_HULL
    tm[18:25, 10:48] = MAT_AIR
    tm[18:36, 10:12] = MAT_AIR
    tm[4:25, 46:48] = MAT_AIR


def _overlaps(units) -> int:
    n = 0
    for i, a in enumerate(units):
        for b in units[i + 1:]:
            if (abs(a.tile_x - b.tile_x) < a.footprint
                    and abs(a.tile_y - b.tile_y) < a.footprint):
                n += 1
    return n


@pytest.mark.parametrize("controller,stacks", [("bots_v1", True),
                                               ("bots", False)])
def test_teammates_do_not_stack_in_a_corridor(tmp_path, controller, stacks):
    """Property (#142): four teammates walking a corridor with nobody to
    fight end, after 40 s, with no two footprints overlapping -- where the
    first bots (`bots_v1`, the non-vacuity leg) pile onto one tile. Breaks
    if the spacing feature stops counting teammates or their goals."""
    lvl = _box(tmp_path, f"corr_{controller}", _corridor)
    sim, units, _ = _sim(lvl, [(0, 3, 29), (0, 6, 29), (0, 3, 32),
                               (0, 6, 32)])
    bots = control_source.create_controller(controller, 0)
    _run(sim, [bots], 960)
    assert (_overlaps([u for u in units if u.alive]) > 0) == stacks


# ---------------------------------------------------------------------------
# (h) the controller drives only through the intent facade
# ---------------------------------------------------------------------------
def test_the_bots_change_nothing_but_intents(tmp_path):
    """Property: with no renderer, a bots tick writes no synced state (the
    tick digest is the same before and after ``handle_frame``) and every
    call it makes on the sim goes through the intent facade's tap (a row of
    ``match_replay.INTENT_KINDS``), never the reload verb (RULED); and no
    line of ``control_bots`` assigns to a unit, the sim or the map. Breaks
    if a bot writes a position, a facing, hit points or a field directly,
    or acts outside the facade."""
    from simulation.match_session import MatchSession
    spec = MatchSpec(teams=(BOTS, BOTS), seed=2, time_limit_s=6.0, zombies=2)
    lvl = _match_level(tmp_path, "facade", spec)
    ses = MatchSession(lvl, None, breach_physics=bp)
    kinds = []

    class _Tap:
        def record(self, tick, kind, args):
            kinds.append(kind)
    ses.sim.attach_intent_log(_Tap())
    bots = [c for c in ses.controllers if isinstance(c, BotTeam)]
    assert bots
    while not ses.is_terminal():
        before = tick_digest(capture_snapshot(ses.sim))
        for b in bots:
            b.handle_frame(ses.sim, None)
        assert tick_digest(capture_snapshot(ses.sim)) == before
        ses.tick()
    assert kinds and set(kinds) <= set(match_replay.INTENT_KINDS)
    assert "reload_intent" not in kinds
    src = (ROOT / "src" / "control_bots.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            for t in targets:
                base = t
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(t, (ast.Attribute, ast.Subscript)) and \
                        isinstance(base, ast.Name):
                    assert base.id not in ("u", "unit", "m", "e", "z",
                                           "sim", "gmap", "target"), \
                        ast.unparse(t)


# ---------------------------------------------------------------------------
# (i) the tournament's friendly-fire counter
# ---------------------------------------------------------------------------
def test_the_tournament_counts_a_same_team_hit_as_friendly_fire():
    """Property: ``bot_tournament.TeamLog`` counts a credited hit whose
    attacker and victim are marines of one team as friendly fire for that
    team, a hit on another team's unit as a hit, and a unit's own hit as a
    self hit. Breaks if the counter compares the wrong teams or misses a
    hit."""
    from bot_tournament import TeamLog
    units = [SimpleNamespace(id=i, alive=True, is_zombie=False, team=t)
             for i, t in ((0, 0), (1, 0), (2, 1))]
    sim = SimpleNamespace(units=units, tick_events=[])
    log = TeamLog()
    log(sim)
    sim.tick_events = [
        UnitHitEvent(unit_id=1, damage=10, source="bullet", attacker_id=0),
        UnitHitEvent(unit_id=2, damage=10, source="bullet", attacker_id=0),
        UnitHitEvent(unit_id=0, damage=10, source="explosion",
                     attacker_id=0)]
    log(sim)
    assert log.ff_hits[0] == 1 and log.hits[0] == 1
    assert log.self_hits[0] == 1 and log.ff_hits[1] == 0


# ---------------------------------------------------------------------------
# (j) a closed door on the path is opened (#142)
# ---------------------------------------------------------------------------
def test_a_bot_opens_a_closed_door_on_its_path(tmp_path):
    """Property (#142): a bot whose objective lies beyond a closed (non-hull)
    door paths through it, asks it open with the USE verb once it touches
    it, and walks through. Breaks if the bots' passable grid shuts doors out
    (the bot stands), the door is toggled shut again, or USE is never
    issued."""
    from simulation.entities.door import DOOR_OPEN
    from test_horde_behaviours import _door_level
    sim = Simulation(_door_level(), seed=1, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    bot = sim.get_unit(sim.add_unit(Unit("B", x=8, y=4, team=0)))
    foe = sim.get_unit(sim.add_unit(Unit("F", x=32, y=4, team=1)))
    sim.start_team_picture((0, 1))
    door = sim._doors[0]
    uses = []

    class _Tap:
        def record(self, tick, kind, args):
            if kind == "use_intent":
                uses.append(tick)
    sim.attach_intent_log(_Tap())
    bots = BotTeam(0)
    for _ in range(600):
        _know(bots, sim, foe)          # known beyond the wall, never seen
        bots.handle_frame(sim, None)
        sim.step()
        if bot.tile_x > 21:
            break
    assert uses, "USE never issued"
    assert door.state == DOOR_OPEN
    assert bot.tile_x > 21
