"""#195 patch 1c -- sight for many tiles (``cpp/src/sight_lines.{h,cpp}``,
``PhysicsEngine.sight_rays`` / ``sight_window``, the ``vision.py`` wrappers).

Properties: the C++ routine and its pure-Python reference both equal
``GameMap.has_los`` ray for ray (sight and ``physical``), on generated ships
that carry a grille (the sight screen), a hull box scattered with screens, and
lines that start or end outside the map; the wrappers equal ``vision.ray_clear``
on every map, with a cover entity that blocks sight in the way; a window
equals the many-rays answer for the same pairs in both directions; and under
the match's all-round rule ``vision.compute`` (one many-rays call) gives the
seen-by lists of the per-pair path over a played match.
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

import level_lib  # noqa: E402
import level_loader  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation, vision  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.cover_system import CoverRuntime  # noqa: E402
from simulation.materials import (MAT_AIR, MAT_GRILLE, MAT_HULL)  # noqa: E402
from simulation.match import MatchSpec, MatchTeam, SpawnRoom  # noqa: E402
from simulation.match_session import MatchSession  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from test_bots_v1 import ROOMS, _level as _box_level  # noqa: E402


def _cover(i, x0, y0, w, h, blocks_los=True):
    inst = SimpleNamespace(ordinal=i, id=f"c{i}", class_name="cover",
                           fields={"width_m": 1.0, "height_m": 1.0, "hp": 60,
                                   "blocks_los": blocks_los})
    return CoverRuntime(inst, x0, y0, w, h)


def _sim_of(level, covers=()):
    sim = Simulation(level, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    assert sim.gmap._physics_engine is not None
    sim.cover = list(covers)
    return sim


def _scatter_level():
    """A 48 x 64 hull box with scattered hull blocks and grille tiles and a
    full grille wall: every kind of blocker, none by design on a line."""
    h, w = 48, 64
    rng = np.random.default_rng(7)
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    for _ in range(60):
        y, x = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
        tm[y, x] = MAT_HULL
    for _ in range(40):
        y, x = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
        tm[y, x] = MAT_GRILLE
    tm[10:30, 40] = MAT_GRILLE
    return LevelData(name="sight_scatter", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


@pytest.fixture(scope="module")
def recipe_level(tmp_path_factory):
    import gen_ship as G
    plan = G.generate(1, "medium", recipe="freighter", profile="standard",
                      zombies=4)
    base = tmp_path_factory.mktemp("sight_recipe")
    G.write(plan, base / plan.name)
    return level_loader.load(plan.name, levels_dir=str(base))


@pytest.fixture(scope="module", params=["ship_demo", "freighter", "scatter"])
def world(request, recipe_level):
    """(sim, covers): a Simulation on a generated map with sight-blocking
    cover (and one ordinary crate, which must NOT block) placed on open
    floor."""
    if request.param == "ship_demo":
        lvl = level_loader.load("ship_demo", str(ROOT / "levels"))
    elif request.param == "freighter":
        lvl = recipe_level
    else:
        lvl = _scatter_level()
    sim = _sim_of(lvl)
    g = sim.gmap
    ys, xs = np.nonzero(~g.solid & ~g.sight_screen)
    rng = np.random.default_rng(3)
    pick = rng.choice(len(ys), size=6, replace=False)
    covers = [_cover(i, int(xs[k]), int(ys[k]), 2, 2, blocks_los=(i < 5))
              for i, k in enumerate(pick)]
    sim.cover = covers
    return request.param, sim


def _rays(g, n, seed):
    """n random tile pairs, a fifth reaching outside the map (any side), a
    few degenerate (start == end) and axis-aligned."""
    h, w = g.solid.shape
    rng = np.random.default_rng(seed)
    pairs = np.stack([rng.integers(0, h, n), rng.integers(0, w, n),
                      rng.integers(0, h, n), rng.integers(0, w, n)], axis=1)
    out = rng.random(n) < 0.2
    pairs[out, 0] = rng.integers(-12, h + 12, int(out.sum()))
    pairs[out, 3] = rng.integers(-12, w + 12, int(out.sum()))
    pairs[:5, 2:] = pairs[:5, :2]
    pairs[5:15, 2] = pairs[5:15, 0]
    pairs[15:25, 3] = pairs[15:25, 1]
    return pairs.astype(np.int32)


def _through_cover(sim, n_each, seed):
    """Lines that run through the middle of each sight-blocking cover."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in sim.cover:
        if not c.blocks_los:
            continue
        cy, cx = int(c.y0 + 1), int(c.x0 + 1)
        d = rng.integers(-8, 9, size=(n_each, 2))
        d = d[(d != 0).any(axis=1)]
        for dy, dx in d.tolist():
            rows.append((cy + dy, cx + dx, cy - dy, cx - dx))
    return np.array(rows, dtype=np.int32)


def _has_los(g, pairs, physical):
    return np.array([g.has_los(int(a), int(b), int(c), int(d),
                               physical=physical)
                     for a, b, c, d in pairs.tolist()])


def test_the_routine_and_its_reference_equal_has_los_ray_for_ray(world):
    """Property: on a generated ship (and a screen-scattered box) the C++
    ``sight_rays`` and ``sight_lines_reference`` equal ``GameMap.has_los``
    for 4 000 random lines, sight and ``physical`` alike, lines reaching
    outside the map included; the answers are not vacuous (blocked and clear
    lines, and lines only the sight screen blocks). Breaks if the walk's
    error term, the order of its two axis steps, the end-tile or
    out-of-the-map treatment, or the screen / solid stop changes in any one
    of the three."""
    name, sim = world
    g = sim.gmap
    pairs = _rays(g, 4000, 11)
    for physical in (False, True):
        want = _has_los(g, pairs, physical)
        got = vision.lines_clear(g, pairs, physical).astype(bool)
        assert np.array_equal(got, want), (name, physical)
        solid, screen = vision._planes(g)
        ref = vision.sight_lines_reference(solid, screen, pairs,
                                           physical).astype(bool)
        assert np.array_equal(ref, want), (name, physical)
        assert want.any() and not want.all()
    if name != "ship_demo":
        assert g.sight_screen.any()                    # a real grille
        s = _has_los(g, pairs, False)
        p = _has_los(g, pairs, True)
        assert (p & ~s).any()                          # screen-only blocks
        assert not (s & ~p).any()                      # screen only ever adds


def test_the_wrappers_equal_ray_clear_with_cover_that_blocks_sight(world):
    """Property: ``rays_clear`` equals ``vision.ray_clear`` for every line
    with sight-blocking cover on the map (an ordinary crate does not block),
    for the engine path and the engine-less reference path; cover alone
    blocks some lines the walls leave clear; ``physical=True`` ignores
    cover, as ``has_los(physical=True)`` does. Breaks if cover is applied
    before the walk, to the wrong segment, skipped, or applied to the
    physical question."""
    name, sim = world
    g = sim.gmap
    pairs = np.concatenate([_rays(g, 3000, 12), _through_cover(sim, 80, 5)])
    want = np.array([vision.ray_clear(sim, *map(int, p)) for p in pairs])
    got = vision.rays_clear(sim, pairs)
    assert np.array_equal(got, want), name
    walls_only = _has_los(g, pairs, False)
    assert (walls_only & ~want).any()                  # cover blocked some
    eng = g._physics_engine
    g._physics_engine = None                           # the bare-map path
    try:
        assert np.array_equal(vision.rays_clear(sim, pairs), want), name
    finally:
        g._physics_engine = eng
    assert np.array_equal(vision.rays_clear(sim, pairs, physical=True),
                          _has_los(g, pairs, True)), name


def test_a_window_equals_the_many_rays_answer_in_both_directions(world):
    """Property: ``window_clear`` -- one tile to every tile of a rectangle,
    and every tile of it to the tile -- equals ``rays_clear`` over the same
    pairs cell for cell (cover included), for windows inside the map, over
    its corner and past its edge, and the two directions differ somewhere (the
    walk is not symmetric). Breaks if a window cell walks from the wrong end,
    the rectangle is indexed (row, column) the wrong way round, or its cells
    leave the map's treatment of outside tiles."""
    name, sim = world
    g = sim.gmap
    h, w = g.solid.shape
    asym = False
    for (y, x, wy, wx, wh, ww) in [(h // 2, w // 2, h // 2 - 12, w // 2 - 12,
                                    25, 25),
                                   (3, 4, -10, -9, 25, 25),
                                   (h - 2, w - 3, h - 14, w - 10, 25, 25),
                                   (h // 3, w // 3, 5, 9, 7, 31)]:
        for physical in (False, True):
            both = []
            for to_window in (False, True):
                got = vision.window_clear(sim, y, x, wy, wx, wh, ww,
                                          to_window, physical)
                pairs = vision._window_pairs(y, x, wy, wx, wh, ww, to_window)
                want = vision.rays_clear(sim, pairs, physical).reshape(wh, ww)
                assert got.shape == (wh, ww)
                assert np.array_equal(got, want), (name, y, x, to_window)
                both.append(got)
            asym = asym or not np.array_equal(both[0], both[1])
    assert asym
    # a one-by-one and an empty window are windows too
    one = vision.window_clear(sim, 5, 5, 6, 6, 1, 1)
    assert one.shape == (1, 1)
    assert vision.window_clear(sim, 5, 5, 0, 0, 0, 4).shape == (0, 4)
    assert vision.rays_clear(sim, np.zeros((0, 4), dtype=np.int32)).shape == (0,)


def test_the_all_round_batch_gives_the_pairwise_seen_by_lists(tmp_path):
    """Property: over a played all-round match (two bot teams and a horde,
    thirty ticks, the units moving and dying), ``vision.compute`` -- one
    many-rays call -- returns the same ``seen_by`` and ``visible_by_team`` as
    ``compute_pairwise`` (every pair through ``can_see``) on every tick, and
    the lists are not empty at all times. Breaks if the batch builds a pair
    ``can_see`` would not (a zombie observer, a teammate, a corpse), orders the
    tuples differently, or reads a stale tile."""
    team = MatchTeam(n=3, loadout="rifleman", env_profile="default",
                     controller="bots_v1")
    spec = MatchSpec(teams=(team, team), seed=3, time_limit_s=10.0,
                     sight="all_round", zombies=3)
    _box_level(tmp_path, "batch", wall_x=29, gap=(14, 24))
    lvl_dir = tmp_path / "batch"
    level_lib.write_spawn_rooms(lvl_dir / "level.toml", ROOMS + [
        SpawnRoom(team=ZOMBIE_TEAM, x=36, y=28, w=8, h=8, max_per_team=4)])
    level_lib.write_match(lvl_dir, spec)
    lvl = level_loader.load("batch", str(tmp_path))
    session = MatchSession(lvl, spec, breach_physics=bp)
    sim = session.sim
    assert sim.sight == "all_round"
    seen_any = 0
    for _ in range(30):
        session.tick()
        a = vision.compute(sim)
        b = vision.compute_pairwise(sim)
        assert a.seen_by == b.seen_by
        assert a.visible_by_team == b.visible_by_team
        seen_any += bool(a.seen_by)
    assert seen_any > 5


def test_the_cone_rule_does_not_touch_the_routine(tmp_path, monkeypatch):
    """Property: under cone (the default) ``compute`` is the per-pair path and
    never calls the many-rays entry. Breaks if the cone rule is routed
    through the batch (its pair set is not all-round's)."""
    lvl = _box_level(tmp_path, "cone")
    sim = _sim_of(lvl)

    def boom(*a, **k):
        raise AssertionError("the cone path asked for many rays")
    monkeypatch.setattr(vision, "rays_clear", boom)
    assert vision.compute(sim).seen_by == {}
