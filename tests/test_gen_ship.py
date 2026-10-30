"""Properties of the v1 ship generator (``tools/gen_ship.py``, #120).

Each test names the property it protects and the change that must break it.
Nothing here pins a room count, a door count or a layout: those are the
generator's to change.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_ship  # noqa: E402
from level_airtight import find_leaks  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.match import (  # noqa: E402
    FORMATION_FOOTPRINT, formation_capacity, placement_cells,
)
from simulation.materials import MAT_AIR, MAT_DOOR  # noqa: E402

SEEDS = range(100)


@pytest.fixture(scope="module")
def small_plans():
    return [gen_ship.generate(s, "small") for s in SEEDS]


def test_every_small_seed_passes_the_gates(small_plans):
    """PROPERTY: for 100 seeds the generator returns a small ship that is
    airtight, left-right mirrored (team 1's room the reflection of team
    0's), and re-passes every gate (A* spawn <-> spawn and spawn -> zombie
    room, every compartment reachable, formation cells on floor) when the
    gates are run again on the returned tiles. BREAKS IF a gate is skipped
    or a returned plan is not the validated one (e.g. a later stage edits
    the tiles after ``validate``)."""
    for plan in small_plans:
        tm = plan.tilemap
        assert np.array_equal(tm, tm[:, ::-1]), plan.seed
        mat, vac, solid, _walk = gen_ship._masks(tm)
        _, vac_leak, edge_leak = find_leaks(solid, vac)
        assert not vac_leak and not edge_leak, plan.seed
        r0, r1 = plan.spawn_rooms[0], plan.spawn_rooms[1]
        assert (r0.team, r1.team) == (0, 1)
        assert (r1.x, r1.y, r1.w, r1.h) == (tm.shape[1] - r0.x - r0.w, r0.y,
                                            r0.w, r0.h)
        assert plan.spawn_rooms[2].team == ZOMBIE_TEAM
        assert gen_ship.validate(tm, plan.spawn_rooms,
                                 n_rooms=len(plan.graph.rooms)) == [], plan.seed


def test_rejection_stays_rare(small_plans):
    """PROPERTY: the gates reject few small plans (under one in five over
    100 seeds), so a retry is an exception, not the generator's way of
    working. BREAKS IF a layout change starts failing the gates routinely."""
    tried = sum(p.attempt + 1 for p in small_plans)
    rejected = sum(p.attempt for p in small_plans)
    assert rejected / tried < 0.2, (rejected, tried)


def test_spawn_rooms_seat_their_teams():
    """PROPERTY: each spawn room's formation capacity covers its
    ``max_per_team`` (zombies: the zombie count), and its formation cells
    are disjoint 3x3 footprints on open floor, for every seated team size.
    BREAKS IF the spawn slab is sized below #119's formation rule or fuel /
    walls land inside a spawn room."""
    fp = FORMATION_FOOTPRINT
    for m in (1, 4, 6, 9):
        for seed in range(6):
            plan = gen_ship.generate(seed, "small", max_per_team=m, zombies=4)
            tm = plan.tilemap
            for r in plan.spawn_rooms:
                want = m if r.team != ZOMBIE_TEAM else 4
                assert r.max_per_team == want
                assert formation_capacity(r) >= want
                taken = np.zeros(tm.shape, dtype=int)
                for (x, y) in placement_cells(r, want):
                    taken[y:y + fp, x:x + fp] += 1
                    assert (tm[y:y + fp, x:x + fp] == MAT_AIR).all()
                assert taken.max() == 1


def test_refuses_what_a_size_cannot_seat():
    """PROPERTY: a ``max_per_team`` or zombie count the smallest size can
    never seat is refused up front with ShipGenError (never a 32-attempt
    search, never a level that cannot spawn its teams). BREAKS IF
    ``check_request`` stops guarding the request."""
    with pytest.raises(gen_ship.ShipGenError, match="max_per_team"):
        gen_ship.generate(1, "small", max_per_team=40)
    with pytest.raises(gen_ship.ShipGenError, match="zombies"):
        gen_ship.generate(1, "small", zombies=60)
    with pytest.raises(gen_ship.ShipGenError):
        gen_ship.generate(1, "huge")


def test_the_embedding_reads_only_the_graph():
    """PROPERTY: stage 2 (``embed``) and the gates take any ShipGraph, not
    only v1's mirrored split -- a hand-built, unmirrored three-room graph
    (two spawn rooms joined through a zombie room by two doors) embeds into
    tiles that pass the gates, with every door edge cut as MAT_DOOR. This
    is the seam a graph-grammar stage 1 (#27) plugs into. BREAKS IF the
    embedding reaches back into v1's planner (size presets, the mirror)."""
    G = gen_ship
    rooms = [G.Room(0, 4, 4, 10, 10, "spawn", {"spawn_team": 0}),
             G.Room(1, 15, 4, 10, 10, "section", {"zombie": True,
                                                  "fuel": 1}),
             G.Room(2, 26, 4, 10, 10, "spawn", {"spawn_team": 1})]
    edges = [G.Edge(0, 1, "door"), G.Edge(1, 2, "door", at=6)]
    graph = G.ShipGraph(width=40, height=18, hull=(3, 3, 36, 14),
                        rooms=rooms, edges=edges, mirror_x=False,
                        max_per_team=4, zombies=2)
    emb = G.embed(graph, np.random.Generator(np.random.PCG64(0)))
    assert G.validate(emb.tilemap, emb.spawn_rooms, n_rooms=3,
                      mirrored=False) == []
    for e in edges:
        for (y, x) in G._opening_tiles(graph, e):
            assert emb.tilemap[y, x] == MAT_DOOR
    assert [r.team for r in emb.spawn_rooms] == [0, 1, ZOMBIE_TEAM]


def test_one_seed_one_level(tmp_path):
    """PROPERTY: one seed writes a byte-identical level folder twice; another
    seed writes a different tilemap. BREAKS IF any draw escapes the seeded
    generator (a global RNG, set / dict ordering, a time stamp in a file)."""
    a = gen_ship.write(gen_ship.generate(3, "small", vents=True), tmp_path / "a")
    b = gen_ship.write(gen_ship.generate(3, "small", vents=True), tmp_path / "b")
    c = gen_ship.write(gen_ship.generate(4, "small", vents=True), tmp_path / "c")
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir())
    assert set(gen_ship.DATA_FILES) <= set(names)
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n
    assert (a / "tilemap.csv").read_bytes() != (c / "tilemap.csv").read_bytes()


_CLI = r"""
import sys, dataclasses, tempfile
from pathlib import Path
root = Path.cwd()
sys.path[:0] = [str(root), str(root / 'src'), str(root / 'tools'),
                str(root / 'cpp' / 'build' / 'Release'), str(root / 'tests')]
import gen_ship, level_loader, level_airtight, run_match
import breach_physics as bp
from simulation import Simulation
d = Path(tempfile.mkdtemp())
plan = gen_ship.generate(5, 'small', vents=True)
gen_ship.write(plan, d / 'ship_t')
lv = level_loader.load('ship_t', levels_dir=str(d))
assert level_airtight.check('ship_t', levels_dir=str(d))
sim = Simulation(lv, seed=1, breach_physics=bp, enable_recorder=False)
kinds = sorted(type(e).__name__ for e in sim.entities)
spec = dataclasses.replace(lv.match, time_limit_s=1.0)
out = run_match.run_match(lv, spec, out=lambda *_: None)
bad = [m for m in sys.modules
       if m in ('pyray', 'raylib') or m.split('.')[0] == 'renderer']
print('RESULT', out.result.outcome, out.result.ticks, kinds, bad)
"""


def test_a_written_ship_loads_lints_and_plays():
    """PROPERTY: a written ship (with vents) loads through the level
    loader, passes ``level_airtight.check`` on the real GameMap, builds its
    duct + two vents as runtimes, and ``run_match`` (idle teams, a 1 s
    limit) ends in a result -- all without importing the renderer. BREAKS IF
    the writer emits something the loader, the vent system or the match
    door refuses, or the plan-side airtight check disagrees with the
    GameMap's."""
    proc = subprocess.run([sys.executable, "-c", _CLI], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
    assert line, proc.stdout[-2000:]
    assert line[0].split()[1] in ("win", "draw")
    assert "'DuctRuntime', 'VentRuntime', 'VentRuntime'" in line[0]
    assert line[0].endswith("[]"), line[0]
