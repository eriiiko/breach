"""#195 2c -- the training rooms (``tools/gen_training_rooms.py``; note
``docs/rl_agent_rung2_note_2c_2026-11-17.md`` section 2, design
``docs/rl_agent_rung2_design_2026-11-14.md`` 5.1) and the reward profile
"range" (note section 3).

Each test names the property it protects and the change that must break it.
The rooms are written into the test's own temporary folder, never into the
repository.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import gen_ship as G  # noqa: E402
import gen_training_rooms as TR  # noqa: E402
import level_loader  # noqa: E402
import rl_env  # noqa: E402
from level_airtight import check as airtight_check  # noqa: E402
from simulation import reward_v1 as R  # noqa: E402
from simulation.match_session import MatchSession  # noqa: E402
from simulation.vision import can_see  # noqa: E402

#: The graph shape each hand room's name says (note 2c, 2): rooms, and the
#: kinds of the openings between them.
SHAPES = {"range_hall": (2, ("open",)),
          "range_door": (2, ("entity_door",)),
          "range_corner": (3, ("entity_door", "entity_door"))}


@pytest.fixture(scope="module")
def rooms(tmp_path_factory):
    out = tmp_path_factory.mktemp("training_rooms")
    TR.write_all(out)
    return out


def _base(name):
    return name[:-len(TR.SWAPPED_SUFFIX)] if name.endswith(
        TR.SWAPPED_SUFFIX) else name


def test_every_room_loads_is_airtight_and_joins_its_spawn_rooms(rooms):
    """Protects: every level the tool writes loads through the level loader
    with a room graph at the map's shape, passes the level lint (the
    airtight flood over the loaded GameMap) and the ship generator's gates
    on the LOADED tiles (spawn cells on floor, an A* path for a 3 x 3 unit
    between the two spawn rooms), and is accepted by a "v1" run's level
    door. Breaks when a writer function leaves a hole in the hull, stamps
    a door as wall, or drops the room graph."""
    names = TR.level_names()
    assert set(TR.HAND_ROOMS) <= set(names)
    for name in names:
        lv = level_loader.load(name, str(rooms))
        assert lv.rooms is not None, name
        assert tuple(lv.rooms.grid.shape) == tuple(lv.tilemap.shape), name
        assert airtight_check(name, levels_dir=str(rooms)), name
        assert G.validate(lv.tilemap, list(lv.spawn_rooms),
                          mirrored=False) == [], name
        assert abs(lv.tile_size_m - G.TILE_SIZE_M) < 1e-9


def test_the_graph_has_the_shape_its_name_says(rooms):
    """Protects: the hall is two graph rooms joined by an open edge, the
    door room two rooms joined by an entity door (a door entity standing on
    the opening's tiles), the corner three rooms chained by two doors in an
    L (a path 0-1-2, the doors on walls of different orientation); each
    spawn room inside a graph room, the two in different rooms. Breaks when
    a writer function drops an opening, the door entity, or puts both teams
    in one room."""
    for name in TR.level_names():
        lv = level_loader.load(name, str(rooms))
        n_rooms, kinds = SHAPES[_base(name)]
        g = lv.rooms
        assert len(g.rooms) == n_rooms, name
        assert sorted(o.kind for o in g.openings) == sorted(kinds), name
        doors = [e for e in lv.entities if e.class_name == "door"]
        assert len(doors) == kinds.count("entity_door"), name
        for o in g.openings:
            assert o.a != o.b
            if o.kind == "entity_door":
                assert any(int(e.fields["x"]) == o.x
                           and int(e.fields["y"]) == o.y for e in doors)
        if n_rooms == 3:
            pairs = sorted((o.a, o.b) for o in g.openings)
            assert pairs == [(0, 1), (1, 2)], name
            assert len({e.fields["orientation"] for e in doors}) == 2, name
        homes = []
        for sr in lv.spawn_rooms:
            cell = g.grid[sr.y:sr.y + sr.h, sr.x:sr.x + sr.w]
            assert (cell >= 0).all() and len(np.unique(cell)) == 1, name
            homes.append(int(cell[0, 0]))
        assert homes[0] != homes[1], name


def test_the_match_file_and_what_the_soldier_sees_at_the_start(rooms):
    """Protects: every room's match file is sight "all_round", no zombies,
    a 45 s clock, one learner (team 0) and one target (team 1); in the hall
    the target is in view from the learner's tile, in the door room and the
    corner it is not (note 2c, 2's table). Breaks when the match line
    loses the sight key, gains a zombie, or a room's spawn tiles move onto
    a line of sight they must not have."""
    for name in TR.level_names():
        lv = level_loader.load(name, str(rooms))
        m = lv.match
        assert m.sight == "all_round" and m.zombies == 0, name
        assert m.time_limit_s == TR.TIME_LIMIT_S == 45.0, name
        assert [t.n for t in m.teams] == [1, 1], name
        sim = MatchSession(lv, m, seed=1, breach_physics=bp).sim
        a, b = sim.roster(0)[0], sim.roster(1)[0]
        assert can_see(sim, a, b) == (_base(name) == "range_hall"), name


def test_the_swapped_twin_differs_only_in_the_spawn_rooms(rooms):
    """Protects: ``<name>_swapped`` has the same tiles, room graph,
    entities and match as ``<name>``, and its two spawn rooms are the
    original's exchanged (team 0 stands where team 1 stood). Breaks when
    the twin is drawn anew, or the swap moves anything else."""
    for name in TR.HAND_ROOMS:
        a = level_loader.load(name, str(rooms))
        b = level_loader.load(name + TR.SWAPPED_SUFFIX, str(rooms))
        assert np.array_equal(a.tilemap, b.tilemap)
        assert a.rooms.rooms == b.rooms.rooms
        assert a.rooms.openings == b.rooms.openings
        assert np.array_equal(a.rooms.grid, b.rooms.grid)
        assert [(e.id, e.class_name, e.fields) for e in a.entities] == \
            [(e.id, e.class_name, e.fields) for e in b.entities]
        assert a.match == b.match
        box = {r.team: (r.x, r.y, r.w, r.h) for r in a.spawn_rooms}
        tbox = {r.team: (r.x, r.y, r.w, r.h) for r in b.spawn_rooms}
        assert box[0] != box[1]
        assert tbox[0] == box[1] and tbox[1] == box[0]


def test_two_writes_are_byte_equal(rooms, tmp_path):
    """Protects: the rooms are a pure function of the tool (note 2c, 2:
    made again from the tool, not kept): a second write into another folder
    gives every file byte for byte. Breaks when anything time-, path- or
    order-dependent enters a written file."""
    TR.write_all(tmp_path)
    for name in TR.level_names():
        files = sorted(p.name for p in (rooms / name).iterdir())
        assert files == sorted(p.name for p in (tmp_path / name).iterdir())
        for f in files:
            assert (rooms / name / f).read_bytes() == \
                (tmp_path / name / f).read_bytes(), (name, f)


def test_the_tool_refuses_a_room_that_fails_a_gate(tmp_path):
    """Protects: the gates run before any write: a draft whose two spawn
    rooms no 3 x 3 unit can walk between (its door walled up) is refused by
    name and nothing is written. Breaks when
    :func:`gen_training_rooms.write` stops gating."""
    d = TR.range_door()
    door = d.openings[0]
    d.tm[door["y"]:door["y"] + door["h"], door["x"]:door["x"] + door["w"]] = \
        TR.MAT_HULL
    with pytest.raises(TR.TrainingRoomRefused, match="reach"):
        TR.write(d, tmp_path)
    assert not (tmp_path / d.name).exists()


# ---------------------------------------------------------------------------
# The "range" reward row (note 2c, 3)
# ---------------------------------------------------------------------------
def test_the_range_row_is_fireteam_with_the_unused_terms_zero():
    """Protects: "range" has exactly fireteam's terms, the same weight on
    every term a one-soldier, one-target, no-zombie room can pay (win,
    eliminated, clock, marine_bar, marine_alive) and 0 on the zombie and
    friendly-fire terms. Breaks when a weight drifts from the table it was
    cut from, or a term is added to one row and not the other."""
    rng, ft = R.PROFILES["range"], R.PROFILES["fireteam"]
    assert set(rng) == set(ft)
    for k in ("zombie_bar", "ff_bar", "ff_kill"):
        assert rng[k] == 0.0
    for k in set(ft) - {"zombie_bar", "ff_bar", "ff_kill"}:
        assert rng[k] == ft[k], k


def _runner(rooms, level, profile, tmp_path):
    p = tmp_path / f"run_{level}_{profile}.toml"
    p.write_text(
        "[run]\nseed = 7\nn_envs = 1\ndecimation = 1\n"
        f"levels = [\"{level}\"]\npolicy = [\"bots_door\", \"idle\"]\n"
        f"levels_dir = '{Path(rooms).as_posix()}'\n"
        "[obs]\nspec = \"v1\"\nfull_view = false\n"
        f"[action]\nspec = \"v1\"\n[reward]\nprofile = \"{profile}\"\n",
        encoding="utf-8")
    run = rl_env.load_run(p)
    levels = rl_env.load_levels(run)
    lay = rl_env.make_layout(run, levels)
    r = rl_env.EnvRunner(0, run, lay, levels, breach_physics=bp)
    r.reset()
    out = {k: np.zeros(s[1:], d) for k, (s, d) in lay.obs_shapes().items()}
    return r, lay, out


@pytest.mark.parametrize("level", ["range_hall", "range_door"])
def test_range_pays_what_fireteam_pays_on_every_tick(rooms, tmp_path,
                                                     level):
    """Protects: in a training room (one soldier, one target, no zombie)
    the "range" profile pays every row exactly what "fireteam" pays, group
    for group, on every tick of a played match (the door bots against the
    standing marine, to the match's end), and the match pays something
    (a hit, the kill, the win). Breaks when a weight of the row drifts on a
    term this room pays, or the room gains a zombie or a team-mate the zero
    terms would hide."""
    ra, lay, oa = _runner(rooms, level, "range", tmp_path)
    rf, _, of = _runner(rooms, level, "fireteam", tmp_path)
    acts = np.zeros((lay.n_rows, 5), np.int64)
    paid = 0.0
    for _ in range(int(TR.TIME_LIMIT_S * 24) + 10):
        ra.observe(oa)
        rf.observe(of)
        a, f = ra.step(acts), rf.step(acts)
        assert np.array_equal(a["reward"], f["reward"])
        paid += float(np.abs(a["reward"]).sum())
        assert a["ended"] == f["ended"]
        if a["ended"]:
            break
    assert a["ended"] and a["result"]["outcome"] == "win"
    assert paid > 0.0
