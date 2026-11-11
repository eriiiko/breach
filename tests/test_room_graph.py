"""Properties of the ROOM GRAPH saved with the level (#195 1b; RL agent
design 5.1 / 5.7 / section 10 "Room graph"): the fixed part
(``[[room]]`` + ``[[opening]]`` + ``rooms.npy``, written by gen_ship through
level_lib, read through ``level_loader``'s door as ``LevelData.rooms``) and
the changing part worked out by ``simulation.room_graph``.

Each test names the property it protects and the change that must break it.
The hand-built graphs are minimal fixtures (the shape of
``test_gen_ship_vents_doors``'s), not layouts the generator owes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_ship as G  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
from simulation import room_graph as RG  # noqa: E402

try:
    import breach_physics as _bp  # noqa: E402
except Exception:                  # pragma: no cover - a box without the build
    _bp = None
needs_bp = pytest.mark.skipif(_bp is None,
                              reason="needs the compiled breach_physics")

RNG = lambda: np.random.Generator(np.random.PCG64(0))  # noqa: E731


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _graph(kind: str = "entity_door", *, tags: dict | None = None
           ) -> G.ShipGraph:
    """Spawn 0 | host section | spawn 1 joined by doors, and an ANNEX below
    the host joined to it by one ``kind`` edge (room 3)."""
    tags = tags or {}
    base = {0: {"spawn_team": 0}, 1: {}, 2: {"spawn_team": 1}, 3: {}}
    boxes = {0: (4, 4, 10, 10, "spawn"), 1: (15, 4, 10, 10, "section"),
             2: (26, 4, 10, 10, "spawn"), 3: (15, 15, 10, 9, "room")}
    rooms = [G.Room(i, *boxes[i][:4], boxes[i][4],
                    {**base[i], **tags.get(i, {})}) for i in range(4)]
    edges = [G.Edge(0, 1, "door", at=4), G.Edge(1, 2, "door", at=4),
             G.Edge(1, 3, kind)]
    return G.ShipGraph(width=40, height=28, hull=(3, 3, 36, 24),
                       rooms=rooms, edges=edges, mirror_x=False,
                       max_per_team=4, zombies=0)


def _airlock_graph() -> G.ShipGraph:
    return _graph("airlock", tags={0: {"cell": 0}, 1: {"cell": 0},
                                   2: {"cell": 0}, 3: {"cell": 1}})


def _write_load(graph, tmp_path, name="fixture"):
    emb = G.embed(graph, RNG())
    assert G.validate(emb.tilemap, emb.spawn_rooms, mirrored=False,
                      graph=graph) == []
    plan = G.ShipPlan(seed=0, size="fixture", attempt=0, graph=graph,
                      embedding=emb)
    G.write(plan, tmp_path / name)
    return plan, level_loader.load(name, levels_dir=str(tmp_path))


def _sim(lv):
    from simulation import Simulation
    return Simulation(lv, seed=1, breach_physics=_bp, enable_recorder=False)


_SHIPS = {}


def _ship(kind: str):
    """A generated ship (cached): ``v1`` = levels/ship_demo's inputs, ``b`` =
    a freighter recipe ship (airlocks, a bypass, open / vented doors)."""
    if kind not in _SHIPS:
        if kind == "v1":
            _SHIPS[kind] = G.generate(1, "medium", max_per_team=6, zombies=4,
                                      vents=True)
        else:
            _SHIPS[kind] = G.generate(1, "medium", recipe="freighter",
                                      profile="standard")
    return _SHIPS[kind]


def _walkable(tm) -> np.ndarray:
    _mat, _vac, _solid, walk = G._masks(tm)
    return walk


# ---------------------------------------------------------------------------
# The fixed part
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["v1", "b"])
def test_every_floor_tile_of_a_room_maps_to_that_room(kind):
    """PROPERTY: every walkable tile of a room's rectangle maps to that room
    in the grid -- bar the tiles of an airlock's passage (its vestibule,
    carved out of the host room, belongs to the airlock opening) -- and no
    tile outside every rectangle maps to a room. BREAKS IF the grid is
    built from another rectangle (a box off by one, the wrong id), a room is
    left out, or the vestibule carve-out leaks into another room."""
    plan = _ship(kind)
    rooms, openings, grid = G.room_graph(plan.graph)
    walk = _walkable(plan.tilemap)
    passage = np.zeros(grid.shape, dtype=bool)
    for o in openings:
        if o.kind == "airlock":
            passage[o.y:o.y + o.h, o.x:o.x + o.w] = True
    inside = np.zeros(grid.shape, dtype=bool)
    for r in rooms:
        box = (slice(r.y, r.y + r.h), slice(r.x, r.x + r.w))
        inside[box] = True
        floor = walk[box] & ~passage[box]
        assert (grid[box][floor] == r.id).all(), r
        assert set(np.unique(grid[box])) <= {r.id, -1}, r
    assert (grid[~inside] == -1).all()


@pytest.mark.parametrize("kind", ["v1", "b"])
def test_every_saved_opening_touches_both_its_rooms(kind):
    """PROPERTY: every saved opening's tiles lie outside every room and some
    of them 4-touch a tile of room ``a`` and some a tile of room ``b`` --
    and one row exists per walkable edge (``keep_clear``), none for a
    window, grille, bulkhead or vent-only edge. BREAKS IF an opening's
    rectangle is placed off its wall, an airlock's row is only its outer
    door (it would not touch the host), a row is dropped or a non-walkable
    edge gets one."""
    plan = _ship(kind)
    rooms, openings, grid = G.room_graph(plan.graph)
    h, w = grid.shape
    for o in openings:
        touched = set()
        for (y, x) in o.tiles:
            assert grid[y, x] == -1, o
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] >= 0:
                    touched.add(int(grid[ny, nx]))
        assert {o.a, o.b} <= touched, (o, touched)
    walkable_edges = [e for e in plan.graph.edges if e.spec.keep_clear]
    assert len(openings) == len(walkable_edges)
    assert sorted(o.kind for o in openings) == sorted(
        e.kind for e in walkable_edges)
    assert not {o.kind for o in openings} & {"window", "grille", "bulkhead",
                                            "vent_only"}


def test_the_same_seed_gives_the_same_ids_twice():
    """PROPERTY: an opening's id is its rank under (lower room, higher room,
    first tile row-major): two generations of one seed give identical rows,
    and the ids are exactly that rank. BREAKS IF an id comes from a counter
    in edge order (a grammar reordering its edges would renumber every
    order), or the sort key changes."""
    for kind in ("v1", "b"):
        _rooms, openings, _grid = G.room_graph(_ship(kind).graph)
        if kind == "v1":
            again = G.generate(1, "medium", max_per_team=6, zombies=4,
                               vents=True)
        else:
            again = G.generate(1, "medium", recipe="freighter",
                               profile="standard")
        r2, o2, g2 = G.room_graph(again.graph)
        assert o2 == openings
        keys = [level_loader.opening_order_key(o.a, o.b, o.x, o.y)
                for o in openings]
        assert keys == sorted(keys)
        assert [o.id for o in openings] == list(range(len(openings)))
        assert all(o.a < o.b for o in openings)
    # the ids do not follow edge order on the recipe ship
    g = _ship("b").graph
    edge_order = [(min(e.a, e.b), max(e.a, e.b)) for e in g.edges
                  if e.spec.keep_clear]
    assert edge_order != [(o.a, o.b) for o in openings]


def test_ship_demo_carries_the_graph_and_it_is_the_generators():
    """PROPERTY: the checked-in ``levels/ship_demo`` loads with a room graph
    equal to what the generator makes from its own seed. BREAKS IF the
    level is not regenerated after a change to the graph's rows, or the
    loader reads the rows / grid differently than they were written."""
    lv = level_loader.load("ship_demo")
    rooms, openings, grid = G.room_graph(_ship("v1").graph)
    assert lv.rooms is not None
    assert lv.rooms.rooms == rooms
    assert lv.rooms.openings == openings
    assert np.array_equal(lv.rooms.grid, grid)
    assert not lv.rooms.grid.flags.writeable


def test_a_level_without_the_graph_loads_with_rooms_none():
    """PROPERTY: the graph is DORMANT when absent: a hand-made level (no
    rows, no rooms.npy) loads with ``rooms = None``. BREAKS IF the door
    refuses a level that has none, or invents an empty one."""
    assert level_loader.load("ship_demo").rooms is not None
    hand = [p.parent.name for p in sorted((ROOT / "levels").glob(
        "*/level.toml")) if not (p.parent / "rooms.npy").exists()]
    assert hand
    assert level_loader.load(hand[0]).rooms is None


# ---------------------------------------------------------------------------
# The door's refusals
# ---------------------------------------------------------------------------

def _rows():
    plan = _ship("v1")
    rooms, openings, grid = G.room_graph(plan.graph)
    return ([r.as_dict() for r in rooms], [o.as_dict() for o in openings],
            grid)


def _door(rooms, openings, grid, shape=None):
    return level_loader.parse_room_graph(
        rooms, openings, grid, grid.shape if shape is None else shape, "t")


@pytest.mark.parametrize("breakage, match", [
    (lambda r, o, g: r[0].update(colour=1), r"unknown key\(s\) \['colour'\]"),
    (lambda r, o, g: o[0].update(colour=1), r"unknown key\(s\) \['colour'\]"),
    (lambda r, o, g: r[0].pop("kind"), "missing key 'kind'"),
    (lambda r, o, g: r[3].update(x=95), "room 3 .* leaves the 100x70 map"),
    (lambda r, o, g: r[3].update(y=-1), "room 3 .* leaves the 100x70 map"),
    (lambda r, o, g: o[0].update(b=99), "names room 99, which does not "
                                        "exist"),
    (lambda r, o, g: o[0].update(id=5), "id 5 is not its rank 0"),
    (lambda r, o, g: r[1].update(id=7), "id 7 .*0..n-1"),
])
def test_the_door_refuses_by_name(breakage, match):
    """PROPERTY: the loader's door refuses, naming it, an unknown key, a
    missing key, a room outside the map, an opening naming a room that does
    not exist, an opening id that is not its rank, room ids out of order.
    BREAKS IF a check is dropped or its message stops naming the key /
    room."""
    rooms, openings, grid = _rows()
    breakage(rooms, openings, grid)
    with pytest.raises(ValueError, match=match):
        _door(rooms, openings, grid)


def test_the_door_refuses_a_grid_of_the_wrong_shape_or_dtype():
    """PROPERTY: rooms.npy must be int32 of the map's shape, holding -1 or
    a room id. BREAKS IF a grid of another map (or a scaled one) is
    accepted, or a stray value slips through."""
    rooms, openings, grid = _rows()
    with pytest.raises(ValueError, match=r"shape \(70, 99\) != the map's "
                                         r"\(70, 100\)"):
        _door(rooms, openings, grid[:, :99], shape=grid.shape)
    with pytest.raises(ValueError, match="dtype int32"):
        _door(rooms, openings, grid.astype(np.int64))
    bad = grid.copy()
    bad[0, 0] = 44
    with pytest.raises(ValueError, match="leave -1..43"):
        _door(rooms, openings, bad)


def test_a_level_folder_with_half_a_graph_is_refused(tmp_path):
    """PROPERTY: the three parts come together or not at all: rows without
    rooms.npy, rooms.npy without rows, are refused by name at load. BREAKS
    IF a level with a stale or missing grid loads with a graph the
    generator never made."""
    plan, lv = _write_load(_graph(), tmp_path)
    assert lv.rooms is not None
    npy = tmp_path / "fixture" / "rooms.npy"
    data = npy.read_bytes()
    npy.unlink()
    with pytest.raises(ValueError, match=r"\[\[room\]\] rows without "
                                         r"rooms.npy"):
        level_loader.load("fixture", levels_dir=str(tmp_path))
    npy.write_bytes(data)
    level_lib.write_managed_blocks(
        tmp_path / "fixture" / "level.toml",
        {"room": lambda nl: [], "opening": lambda nl: []})
    with pytest.raises(ValueError, match="without \\[\\[room\\]\\] rows"):
        level_loader.load("fixture", levels_dir=str(tmp_path))


def test_the_writer_runs_the_door_first(tmp_path):
    """PROPERTY: level_lib never writes a graph its loader refuses. BREAKS
    IF the writer formats rows without the loader's door."""
    rooms, openings, grid = _rows()
    openings[0]["b"] = 99
    with pytest.raises(ValueError, match="does not exist"):
        level_lib.room_graph_blocks(rooms, openings, grid)
    with pytest.raises(ValueError, match="does not exist"):
        level_lib.write_rooms_npy(tmp_path, grid, rooms, openings)
    assert not (tmp_path / "rooms.npy").exists()


# ---------------------------------------------------------------------------
# The changing part
# ---------------------------------------------------------------------------

def _annex_opening(lv):
    return next(o for o in lv.rooms.openings if (o.a, o.b) == (1, 3))


@needs_bp
def test_closing_a_door_entity_makes_its_opening_openable(tmp_path):
    """PROPERTY: an entity door's opening is passable while the door is
    open; closed, it is not passable and IS openable (a living, closed,
    unwired, non-hull door), and the two rooms are no longer joined.
    Reopened, it is passable again. BREAKS IF passability is read off the
    edge kind instead of the map, the cache keeps a stale answer across a
    door's change, or a closed door is not recognised as one a unit could
    open."""
    from simulation.door_system import sweep_doors
    _plan, lv = _write_load(_graph("entity_door"), tmp_path)
    sim = _sim(lv)
    op = _annex_opening(lv)
    st = RG.opening_states(lv.rooms, sim.gmap, sim._doors, lv.wires)
    assert (st[op.id].passable, st[op.id].openable) == (True, False)
    assert (1, 3) in RG.joined_pairs(lv.rooms, sim.gmap)
    door = next(d for d in sim._doors if d.id.startswith("door_e"))
    door.want_open = False
    sweep_doors(sim)
    st = RG.opening_states(lv.rooms, sim.gmap, sim._doors, lv.wires)
    assert (st[op.id].passable, st[op.id].openable) == (False, True)
    assert (1, 3) not in RG.joined_pairs(lv.rooms, sim.gmap)
    assert all(s.passable for s in st if s.id != op.id)
    door.want_open = True
    sweep_doors(sim)
    st = RG.opening_states(lv.rooms, sim.gmap, sim._doors, lv.wires)
    assert st[op.id].passable


@needs_bp
def test_a_wired_or_hull_door_is_not_openable(tmp_path):
    """PROPERTY: a closed door that is WIRED (an airlock's: the logic opens
    it, not a unit) or a HULL door (its far side is space) does not make an
    opening openable. BREAKS IF the wired set or the hull-door rule is
    dropped from ``openable``."""
    _plan, lv = _write_load(_airlock_graph(), tmp_path)
    sim = _sim(lv)
    op = _annex_opening(lv)
    assert op.kind == "airlock"
    st = RG.opening_states(lv.rooms, sim.gmap, sim._doors, lv.wires)[op.id]
    assert (st.passable, st.openable) == (False, False)   # outer door shut
    # The same doors, unwired: the outer door would be a unit's to open.
    st = RG.opening_states(lv.rooms, sim.gmap, sim._doors, ())[op.id]
    assert (st.passable, st.openable) == (False, True)
    assert (1, 3) not in RG.joined_pairs(lv.rooms, sim.gmap)

    # A hull door: a room's closed door to space, recorded (artificially)
    # as an opening -- the door rule, not the generator, is under test.
    g = _graph("door", tags={3: {"hull_door": {"width": 3,
                                               "state": "closed"}}})
    _plan, lv = _write_load(g, tmp_path, name="hull")
    sim = _sim(lv)
    hd = next(d for d in sim._doors if d.id.startswith("hull_door"))
    ys = [t[0] for t in hd.span]
    xs = [t[1] for t in hd.span]
    rooms = [r.as_dict() for r in lv.rooms.rooms]
    fake = [dict(id=0, a=2, b=3, kind="entity_door", x=min(xs), y=min(ys),
                 w=max(xs) - min(xs) + 1, h=max(ys) - min(ys) + 1)]
    rg = level_loader.parse_room_graph(rooms, fake, lv.rooms.grid,
                                       lv.rooms.grid.shape, "t")
    st = RG.opening_states(rg, sim.gmap, sim._doors, lv.wires)[0]
    assert (st.passable, st.openable) == (False, False)


@needs_bp
def test_a_hole_through_a_wall_joins_two_rooms_and_sealing_parts_them(
        tmp_path):
    """PROPERTY: with the annex door closed, destroying ONE wall tile
    between the host and the annex joins them (a hole has no opening id:
    the accepted gap), and sealing that tile parts them again; the saved
    openings are untouched by the hole. BREAKS IF the hole detection is
    dropped (only saved openings counted), or the cache is not keyed on
    ``_tile_serial``."""
    from simulation.door_system import sweep_doors
    from simulation.materials import MAT_STEEL
    _plan, lv = _write_load(_graph("entity_door"), tmp_path)
    sim = _sim(lv)
    door = next(d for d in sim._doors if d.id.startswith("door_e"))
    door.want_open = False
    sweep_doors(sim)
    gm = sim.gmap
    assert (1, 3) not in RG.joined_pairs(lv.rooms, gm)
    wy, wx = 14, 16                    # the host / annex wall, off the door
    assert lv.rooms.grid[wy - 1, wx] == 1 and lv.rooms.grid[wy + 1, wx] == 3
    assert not RG.walkable_tiles(gm)[wy, wx]
    before = RG.opening_states(lv.rooms, gm, sim._doors, lv.wires)
    gm.destroy_wall(wy, wx)
    assert (1, 3) in RG.joined_pairs(lv.rooms, gm)
    assert RG.opening_states(lv.rooms, gm, sim._doors, lv.wires) == before
    gm.seal_tiles([(wy, wx)], MAT_STEEL)
    assert (1, 3) not in RG.joined_pairs(lv.rooms, gm)
    assert (0, 1) in RG.joined_pairs(lv.rooms, gm)


@needs_bp
def test_the_graph_is_not_synced_state(tmp_path):
    """PROPERTY: reading the graph writes nothing the game hashes: the field
    and unit digests after the reads equal those before. BREAKS IF a read
    stores anything on the map or a unit that a digest covers."""
    sys.path.insert(0, str(ROOT / "tests"))
    from match_replay import match_digests
    _plan, lv = _write_load(_graph("entity_door"), tmp_path)
    sim = _sim(lv)
    before = match_digests(sim)
    RG.opening_states(lv.rooms, sim.gmap, sim._doors, lv.wires)
    RG.joined_pairs(lv.rooms, sim.gmap)
    assert match_digests(sim) == before
