"""#134 -- the grille edge kinds in the generator: ``grille`` (an air-only
opening, the window's twin) and ``vented_door`` (the legacy door with a
grille in each jamb), and the freighter's breathing spine
(docs/levelgen_zone_vocabulary_2026-10-29.md §0.4, the 2026-11-06 addition).

Each test names the property it protects and the change that must break it.
The hand-built graphs are minimal fixtures, not layouts the generator owes;
no count of rooms, edges or tiles of a recipe ship is pinned.
"""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_ship as G  # noqa: E402
import level_airtight  # noqa: E402
import level_loader  # noqa: E402
import ship_grammar as S  # noqa: E402
import ship_layout as L  # noqa: E402
from level_airtight import find_leaks  # noqa: E402
from simulation.materials import (  # noqa: E402
    MAT_AIR, MAT_DOOR, MAT_GRILLE, MAT_HULL, MAT_STEEL,
)
from test_gen_ship_edges import _graph as _annex_graph  # noqa: E402

RNG = lambda: np.random.Generator(np.random.PCG64(0))  # noqa: E731
P = ("test", 0)
SPINE = (L.SLAB, L.PASSAGEWAY, L.LOBBY, L.SECTION)
SPINE_KEYS = ("slab-passageway", "passageway-lobby", "lobby-section",
              "section-section")


def _ring(walk: str) -> G.ShipGraph:
    """Spawn 0 | zombie section | spawn 1 in a row, two rooms below the
    first two, every walk edge ``walk``, and a GRILLE between the two lower
    rooms (both reached by walking): 0-1, 1-2, 1-3, 0-4 walk, 3-4 grille."""
    rooms = [G.Room(0, 4, 4, 10, 10, "spawn", {"spawn_team": 0}),
             G.Room(1, 15, 4, 10, 10, "section", {"zombie": True}),
             G.Room(2, 26, 4, 10, 10, "spawn", {"spawn_team": 1}),
             G.Room(3, 15, 15, 10, 6, "room"),
             G.Room(4, 4, 15, 10, 6, "room")]
    edges = [G.Edge(0, 1, walk), G.Edge(1, 2, walk, at=6),
             G.Edge(1, 3, walk), G.Edge(0, 4, walk), G.Edge(3, 4, "grille")]
    return G.ShipGraph(width=40, height=25, hull=(3, 3, 36, 21),
                       rooms=rooms, edges=edges, mirror_x=False,
                       max_per_team=4, zombies=2)


def _embed(graph, mirrored=False):
    emb = G.embed(graph, RNG())
    return emb, G.validate(emb.tilemap, emb.spawn_rooms, mirrored=mirrored,
                           graph=graph)


def _beyond(graph, e) -> list:
    """The wall tiles one past each jamb, along the wall."""
    orient, line, at, n, _lo, _hi = G._opening_span(graph, e)
    return G._wall_tiles(orient, line, (at - 2, at + n + 1))


def _check_stamps(tm, graph) -> int:
    """Every grille / vented_door edge stamped as the table says; returns
    how many vented doors were checked."""
    vented = 0
    for e in graph.edges:
        if e.kind == "grille":
            assert G._opening_tiles(graph, e)
            assert all(tm[y, x] == MAT_GRILLE
                       for (y, x) in G._opening_tiles(graph, e)), e
            assert G._jamb_tiles(graph, e) == []
        if e.kind == "vented_door":
            vented += 1
            assert all(tm[y, x] == MAT_DOOR
                       for (y, x) in G._opening_tiles(graph, e)), e
            jambs = G._jamb_tiles(graph, e)
            assert len(jambs) == 2
            assert all(tm[y, x] == MAT_GRILLE for (y, x) in jambs), e
            assert all(tm[y, x] in (MAT_STEEL, MAT_HULL)
                       for (y, x) in _beyond(graph, e)), e
    return vented


# ---------------------------------------------------------------------------
# Hand-built graphs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("walk", ["door", "vented_door"])
def test_grille_and_vented_door_edges_embed_where_the_table_says(walk):
    """PROPERTY: a graph with a ``grille`` edge (between two rooms both
    reached by walking) embeds and passes every gate, with or without its
    walk edges vented; the grille's opening is grille tiles; a vented
    door's opening is door tiles, its two jambs grille, the wall beyond
    steel; the airtight gate still passes (the grilles are interior); both
    kinds bound a compartment (one per room). BREAKS IF a kind stamps the
    wrong material or the wrong tiles, a jamb lands outside the opening's
    two neighbours, or compartments() starts reading a grille as a gap."""
    g = _ring(walk)
    emb, reasons = _embed(g)
    assert reasons == []
    n_vented = _check_stamps(emb.tilemap, g)
    assert n_vented == (4 if walk == "vented_door" else 0)
    assert G.compartments(emb.tilemap, g)[1] == len(g.rooms)
    assert max(g.compartment_of()) + 1 == len(g.rooms)


def test_swapping_door_for_vented_door_moves_only_the_jambs():
    """PROPERTY (the design's "no layout arithmetic changes"): the same
    graph with every ``door`` a ``vented_door`` has the same rooms, the same
    compartment count and labelling, the same gate verdict, and a tilemap
    that differs from the door one EXACTLY at the vented doors' jamb tiles
    (steel -> grille). BREAKS IF a vented door's opening, width or keep-
    clear differ from the door's, the jambs touch anything else, or the
    swap moves a compartment or a reachability verdict."""
    gd, gv = _ring("door"), _ring("vented_door")
    ed, rd = _embed(gd)
    ev, rv = _embed(gv)
    assert rd == rv == []
    assert [r.box for r in gd.rooms] == [r.box for r in gv.rooms]
    assert gd.compartment_of() == gv.compartment_of()
    ld, nd = G.compartments(ed.tilemap, gd)
    lv, nv = G.compartments(ev.tilemap, gv)
    assert nd == nv and np.array_equal(ld, lv)
    jambs = {t for e in gv.edges for t in G._jamb_tiles(gv, e)}
    diff = {tuple(map(int, t)) for t in
            np.argwhere(ed.tilemap != ev.tilemap)}
    assert diff == jambs and jambs
    assert all(ed.tilemap[t] == MAT_STEEL for t in jambs)


@pytest.mark.parametrize("kind", ["grille", "vented_door"])
def test_a_grille_kind_never_crosses_a_cell(kind):
    """PROPERTY (L5; the grille is not a CELL_CROSSING_KINDS row): an edge
    of either grille kind between two pressure cells is refused by name when
    the graph is built, and by the grammar's ``cell_crossing`` invariant.
    BREAKS IF either kind is added to CELL_CROSSING_KINDS or the check stops
    reading the table."""
    assert kind not in G.CELL_CROSSING_KINDS
    with pytest.raises(G.GraphSchemaError, match="crosses cells"):
        _annex_graph(kind, {"cell": 1})
    d = S.Draft()
    d.rooms = {"slab": S.QRoom(id="slab", role=L.SLAB), "a": S.QRoom(id="a")}
    d = S.join(d, "slab", "a", "door", P, "test")
    d = S.leaf(d, "a", S.QRoom(id="c", cell=1), kind, P)
    with pytest.raises(S.InvariantViolation) as err:
        S.check_invariants(d, 1)
    assert err.value.rule == "cell_crossing"


def test_a_room_reached_only_through_a_grille_is_refused():
    """PROPERTY: a grille passes air, not units -- a non-secret room whose
    only edge is a grille is refused by the stage-2 reachability gate
    (``compartments: ... not reachable``) and by the grammar's ``reachable``
    invariant, while a ``vented_door`` reaches it. BREAKS IF ``grille``
    becomes a walk kind (WALK_KINDS) or its row gains mobility."""
    _emb, reasons = _embed(_annex_graph("grille"))
    assert any(r.startswith("compartments: 1 not reachable")
               for r in reasons), reasons
    assert _embed(_annex_graph("vented_door"))[1] == []
    assert "grille" not in S.WALK_KINDS and "vented_door" in S.WALK_KINDS
    for kind, ok in (("grille", False), ("vented_door", True)):
        d = S.Draft()
        d.rooms = {"slab": S.QRoom(id="slab", role=L.SLAB),
                   "a": S.QRoom(id="a")}
        d = S.join(d, "slab", "a", "door", P, "test")
        d = S.leaf(d, "a", S.QRoom(id="c"), kind, P)
        if ok:
            S.check_invariants(d, 3)
        else:
            with pytest.raises(S.InvariantViolation) as err:
                S.check_invariants(d, 3)
            assert err.value.rule == "reachable"


def test_a_jamb_outside_the_shared_wall_is_refused():
    """PROPERTY: a vented door whose opening starts at the very end of its
    shared wall (no wall tile left for a jamb) is rejected, never stamped
    into the corner (whose far side may be a third room's air). BREAKS IF
    _jamb_tiles stops checking the run."""
    g = _ring("vented_door")
    e = g.edges[2]                                   # 1-3, the row-14 wall
    _o, _line, _at, _n, lo, _hi = G._opening_span(g, e)
    g.edges[2] = replace(e, at=lo)
    with pytest.raises(G._Reject, match="jamb"):
        G.embed(g, RNG())


# ---------------------------------------------------------------------------
# The recipe freighter: the spine breathes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def freighter():
    return G.generate(0, "medium", recipe="freighter", profile="standard",
                      zombies=4)


def _spine_pair(g, e) -> bool:
    ra, rb = g.rooms[e.a].tags, g.rooms[e.b].tags
    return (ra.get("role") in SPINE and rb.get("role") in SPINE
            and "dept" not in ra and "dept" not in rb)


def test_the_freighters_spine_edges_are_vented_doors_mirrored(freighter):
    """PROPERTY (change C, the lead's content call): on a recipe freighter
    every walk edge between two spine rooms (slab, passageway, lobby, main
    sections) is the skeleton's spine kind, ``vented_door``, stamped as the
    table says in both wings (the tilemap mirror-symmetric), while every
    department entry stays a plain ``door``. BREAKS IF the skeleton table's
    spine rows revert, a department entry is vented, or a twin's jambs miss
    the reflection."""
    g, tm = freighter.graph, freighter.tilemap
    assert all(L.FREIGHTER.edges[k] == "vented_door" for k in SPINE_KEYS)
    spine = [e for e in g.edges if _spine_pair(g, e)
             and e.kind in S.WALK_KINDS]
    assert spine and all(e.kind == "vented_door" for e in spine), \
        [(e.a, e.b, e.kind) for e in spine]
    # every vented door is checked, the spine's among them (a department
    # may carry its own: the crew quarters' heads, vocabulary §0.6)
    assert _check_stamps(tm, g) == sum(e.kind == "vented_door"
                                       for e in g.edges) >= len(spine)
    assert np.array_equal(tm, tm[:, ::-1])
    for e in g.edges:
        ta, tb = g.rooms[e.a].tags, g.rooms[e.b].tags
        if ("dept" in ta) != ("dept" in tb) and e.kind in S.WALK_KINDS \
                and e.kind != "airlock":
            assert e.kind == "door", (e.a, e.b, e.kind)


def test_swapping_the_spine_back_to_door_moves_no_room_or_verdict(
        freighter, monkeypatch):
    """PROPERTY: the freighter generated with its spine edges reverted to
    ``door`` has the same rooms (every rectangle), the same compartment
    count, the same gate verdicts and the same grammar / layout attempts --
    the vented door is a door to the layout. BREAKS IF the two kinds differ
    in width, keep-clear or walkability, so the layout or the gates read
    them differently."""
    for k in SPINE_KEYS:
        monkeypatch.setitem(L.FREIGHTER.edges, k, "door")
    plain = G.generate(0, "medium", recipe="freighter", profile="standard",
                       zombies=4)
    assert (plain.g, plain.attempt) == (freighter.g, freighter.attempt)
    assert [r.box for r in plain.graph.rooms] == \
        [r.box for r in freighter.graph.rooms]
    for p in (plain, freighter):
        assert G.validate(p.tilemap, p.spawn_rooms, mirrored=True,
                          graph=p.graph) == []
    assert G.compartments(plain.tilemap, plain.graph)[1] == \
        G.compartments(freighter.tilemap, freighter.graph)[1]


def _flood(open_: np.ndarray, start) -> np.ndarray:
    h, w = open_.shape
    seen = np.zeros_like(open_, dtype=bool)
    seen[start] = True
    q = deque([start])
    while q:
        y, x = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and open_[ny, nx] \
                    and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


def test_the_spine_is_one_air_volume_and_departments_are_not_in_it(
        freighter, tmp_path):
    """PROPERTY: on the WRITTEN and LOADED freighter, flooding the level's
    own gas-open tiles (GameMap: not solid, not vacuum -- the material
    table's permeability) from a slab reaches every floor tile of every
    spine room (slab, passageway, lobby, main sections; a lobby's airlock
    vestibule aside), and no floor tile of any department room or of the
    engineering cell. BREAKS IF the spine edges stop passing air (a jamb
    stamped steel, the grille row made solid), or a department's entry or
    the engineering bulkhead / airlock starts passing it."""
    from simulation.gamemap import GameMap
    G.write(freighter, tmp_path / freighter.name)
    gm = GameMap(level_loader.load(freighter.name, levels_dir=str(tmp_path)))
    open_ = (~np.asarray(gm.solid, bool)) & (~np.asarray(gm.is_vacuum, bool))
    g = freighter.graph
    slab = next(r for r in g.rooms if r.tags.get("role") == L.SLAB)
    air = _flood(open_, (slab.y, slab.x))
    carved = G._carved(g)
    tm = freighter.tilemap

    def floor(rm):
        x0, y0, x1, y1 = rm.box
        return [(y, x) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)
                if tm[y, x] == MAT_AIR
                and (y, x) not in carved.get(rm.id, ())]

    spine = [rm for rm in g.rooms if rm.tags.get("role") in SPINE
             and "dept" not in rm.tags]
    assert {rm.tags["role"] for rm in spine} == set(SPINE)
    for rm in spine:
        assert all(air[t] for t in floor(rm)), (rm.id, rm.tags["role"])
    for rm in g.rooms:
        if "dept" in rm.tags or rm.tags.get("cell", 0) != 0:
            assert not any(air[t] for t in floor(rm)), (rm.id, rm.tags)


def test_the_airtight_lint_passes_a_recipe_ship_and_catches_a_hull_grille(
        freighter, tmp_path):
    """PROPERTY: the airtight lint (level_airtight, the GameMap decode)
    passes the written freighter with its interior grilles, and still
    catches a leak when ONE hull tile against a room is replaced by a
    grille. BREAKS IF the lint starts treating a grille as solid (a hull
    grille would pass), or a spine grille ends up touching space."""
    G.write(freighter, tmp_path / freighter.name)
    assert level_airtight.check(freighter.name, levels_dir=str(tmp_path))
    tm = freighter.tilemap.copy()
    x0, y0, _x1, _y1 = freighter.graph.hull
    x = next(x for x in range(tm.shape[1])
             if tm[y0, x] == MAT_HULL and tm[y0 + 1, x] == MAT_AIR)
    tm[y0, x] = MAT_GRILLE
    _mat, vac, solid, _walk = G._masks(tm)
    _, vac_leak, edge_leak = find_leaks(solid, vac)
    assert vac_leak or edge_leak
