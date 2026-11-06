"""Properties of gen_ship's four stage-2 extensions for the family-B grammar
(#133; docs/levelgen_cycle_types_and_recipes_2026-10-31.md §1.3 / §5.2,
docs/levelgen_layout_stage_2026-10-31.md §4.3 / §11): the airlock bypass (a
breakable wall as a cell crossing), the fuel class, the vent list and lamps
per length.

Each test names the property it protects and the change that must break it.
The hand-built graphs are minimal fixtures, not layouts the generator owes.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_ship as G  # noqa: E402
from level_airtight import find_leaks  # noqa: E402
from simulation.materials import (  # noqa: E402
    MAT_AIR, MAT_KINDLING, MAT_STEEL, MAT_WOOD,
)

try:
    import breach_physics as _bp  # noqa: E402
except Exception:                  # pragma: no cover - a box without the build
    _bp = None
needs_bp = pytest.mark.skipif(_bp is None,
                              reason="needs the compiled breach_physics")

RNG = lambda: np.random.Generator(np.random.PCG64(0))  # noqa: E731
BOTH = ["supply", "return"]


def _run(graph):
    emb = G.embed(graph, RNG())
    reasons = G.validate(emb.tilemap, emb.spawn_rooms, mirrored=False,
                         graph=graph)
    return emb, reasons


def _airtight(tm) -> bool:
    _mat, vac, solid, _walk = G._masks(tm)
    _, vac_leak, edge_leak = find_leaks(solid, vac)
    return not vac_leak and not edge_leak


# ---------------------------------------------------------------------------
# 1. The airlock bypass: a breakable wall crossing a pressure cell
# ---------------------------------------------------------------------------

def _bypass_graph(al_at=16, br_at=23, second="breakable",
                  cells=True) -> G.ShipGraph:
    """Spawn 0 | host | spawn 1 (cell 0) joined by doors, and an ANNEX
    below the host (cell 1) sharing its 18-tile wall (x 15..32) with TWO
    edges: an airlock (vestibule on the host's side) and a ``second`` edge,
    each at its own ``at``."""
    c = (lambda k: {"cell": k}) if cells else (lambda k: {})
    rooms = [G.Room(0, 4, 4, 10, 10, "spawn", {"spawn_team": 0, **c(0)}),
             G.Room(1, 15, 4, 18, 10, "section", {**c(0)}),
             G.Room(2, 34, 4, 10, 10, "spawn", {"spawn_team": 1, **c(0)}),
             G.Room(3, 15, 15, 18, 9, "room", {**c(1)})]
    edges = [G.Edge(0, 1, "door", at=4), G.Edge(1, 2, "door", at=4),
             G.Edge(1, 3, "airlock", at=al_at),
             G.Edge(1, 3, second, at=br_at)]
    return G.ShipGraph(width=48, height=28, hull=(3, 3, 44, 24),
                       rooms=rooms, edges=edges, mirror_x=False,
                       max_per_team=4, zombies=0)


def test_an_airlock_bypass_across_two_cells_is_two_cells_with_a_wood_wall():
    """PROPERTY (L5 as amended, layout §4.3): an airlock + a breakable on one
    shared wall between two pressure cells builds, embeds and passes every
    gate; the level is airtight; the two cells stay SEPARATE compartments
    (rooms + the vestibule); the breakable's span is wood that belongs to
    no compartment (structure), framed by steel on both sides, and the outer
    door keeps its two steel wall mounts. BREAKS IF ``breakable`` leaves
    CELL_CROSSING_KINDS, the pair is refused as a multi-edge, the wood is
    labelled as room contents (the cells merge), or the openings are stamped
    into each other's mounts / clearance."""
    g = _bypass_graph()
    emb, reasons = _run(g)
    assert reasons == []
    tm = emb.tilemap
    assert _airtight(tm)
    clab, cn = G.compartments(tm, g)
    assert cn == len(g.rooms) + 1
    host, annex = g.rooms[1], g.rooms[3]
    assert clab[host.y, host.x] != clab[annex.y + annex.h - 1, annex.x]
    wood = G._opening_tiles(g, g.edges[3])
    assert wood and all(tm[y, x] == MAT_WOOD and clab[y, x] == 0
                        for (y, x) in wood)
    (row, x0), (_r, x1) = wood[0], wood[-1]
    assert tm[row, x0 - 1] == MAT_STEEL and tm[row, x1 + 1] == MAT_STEEL
    (_i, al), = G._airlocks(g)
    (orow, ox0), (_r, ox1) = al.outer[0], al.outer[-1]
    assert tm[orow, ox0 - 1] == MAT_STEEL and tm[orow, ox1 + 1] == MAT_STEEL
    assert not set(wood) & set(al.outer)


@needs_bp
def test_an_airlock_bypass_still_cycles_for_a_unit(tmp_path):
    """PROPERTY: with the breakable beside it, the airlock still loads,
    passes the GameMap airtight check, idles empty and opens its outer door
    for a unit standing in the vestibule (the headless check of
    test_gen_ship_vents_doors, on the bypass fixture). BREAKS IF the
    breakable takes a mount tile of the interior recipe (a transfer pump or
    side probe), or its wood lets the cells' pressures leak together."""
    from test_gen_ship_vents_doors import _HEADLESS
    g = _bypass_graph()
    emb, reasons = _run(g)
    assert reasons == []
    plan = G.ShipPlan(seed=0, size="fixture", attempt=0, graph=g,
                      embedding=emb)
    G.write(plan, tmp_path / "bypass")
    (_i, al), = G._airlocks(g)
    x0, y0, _x1, _y1 = al.chamber
    proc = subprocess.run(
        [sys.executable, "-c", _HEADLESS, str(tmp_path), "bypass",
         str(x0 + 1), str(y0 + 1)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
    assert line, proc.stdout[-2000:]
    _r, idle, ticks, _n, bad = line[0].split(" ", 4)
    assert idle == "IDLE"
    assert ticks != "None", "the outer door never opened for a unit inside"
    assert bad == "[]"


@pytest.mark.parametrize("kw, match", [
    ({"br_at": 22}, r"overlap"),                    # clearance on the mount
    ({"br_at": 29}, r"leaves the shared wall"),     # no clearance at the end
    ({"al_at": 15, "br_at": 24}, r"leaves the shared wall"),  # mount outside
    ({"br_at": None}, r"must both be set"),
    ({"second": "door"}, r"carry 2 edges"),
    ({"second": "window"}, r"carry 2 edges"),
    ({"second": "airlock", "br_at": 25}, r"carry 2 edges"),
])
def test_a_bypass_off_its_arithmetic_or_another_multi_edge_is_refused(kw,
                                                                      match):
    """PROPERTY: an airlock bypass whose openings overlap, crowd each
    other's mount / clearance tiles or leave the shared run, or that leaves
    an ``at`` to centring, is refused BY NAME when the graph is built; and
    the only multi-edge a room pair may carry is airlock + breakable (two
    doors, a door or window beside the airlock, two airlocks: refused).
    BREAKS IF such openings are stamped instead (a wood tile on a pump
    mount, a door inside the outer door), or multi-edges open in general."""
    with pytest.raises(G.GraphSchemaError, match=match):
        _bypass_graph(**kw)


def test_plain_kinds_still_may_not_cross_a_cell():
    """PROPERTY (L5): only the airlock, the bulkhead and the breakable wall
    cross a cell; a door, an arch or a window between two cells is refused
    by name. BREAKS IF CELL_CROSSING_KINDS grows past the amended L5."""
    assert "breakable" in G.CELL_CROSSING_KINDS
    for kind in ("door", "open", "window", "entity_door"):
        g = _bypass_graph(cells=False)
        g.rooms[3].tags["cell"] = 1
        g.edges[:] = g.edges[:2] + [G.Edge(1, 3, kind, at=23)]
        with pytest.raises(G.GraphSchemaError, match=r"\(L5\)"):
            g.check()


# ---------------------------------------------------------------------------
# 2. The fuel class
# ---------------------------------------------------------------------------

def _fuel_graph(fuel, extra: dict | None = None) -> G.ShipGraph:
    """Spawn 0 | a 14x10 STORE | spawn 1, one door each side; ``fuel`` on
    the store."""
    rooms = [G.Room(0, 4, 4, 10, 10, "spawn", {"spawn_team": 0}),
             G.Room(1, 15, 4, 14, 10, "room", {"fuel": fuel,
                                                **(extra or {})}),
             G.Room(2, 30, 4, 10, 10, "spawn", {"spawn_team": 1})]
    edges = [G.Edge(0, 1, "door", at=7), G.Edge(1, 2, "door", at=7)]
    return G.ShipGraph(width=44, height=18, hull=(3, 3, 40, 14),
                       rooms=rooms, edges=edges, mirror_x=False,
                       max_per_team=4, zombies=0)


def test_a_kindling_fuel_class_is_its_rooms_contents():
    """PROPERTY (recipe doc §5.2): a room whose fuel is a kindling mix gets
    kindling clusters, every one of them labelled as its room's own
    compartment (contents, never structure), the plan passes the gates with
    one compartment per room, and the floor-site rule (hazards, heat
    sources) keeps kindling out of its site's 3x3 -- one set, read off the
    material table's fuel rows, at all three sites. BREAKS IF compartments()
    / _floor_site / the fuel count go back to a furniture / wood literal
    (the kindling becomes walls, or an entity lands in a kindling heap), or
    the mix is ignored (no kindling stamped)."""
    g = _fuel_graph({"clusters": 6, "mix": {"kindling": 1.0}})
    emb, reasons = _run(g)
    assert reasons == []
    tm = emb.tilemap
    kind = np.argwhere(tm == MAT_KINDLING)
    assert len(kind) > 0, "no kindling stamped: vacuous"
    assert MAT_KINDLING in G.fuel_materials()
    clab, cn = G.compartments(tm, g)
    assert cn == len(g.rooms)
    store = g.rooms[1]
    lab = clab[store.y + store.h // 2, store.x + store.w // 2]
    assert all(clab[y, x] == lab for (y, x) in kind)
    assert emb.fuel_tiles == len(kind)
    # The floor-site rule: put kindling beside the site it would pick.
    free = np.zeros(tm.shape, dtype=bool)
    x, y = G._floor_site(tm, store, g.width - 1, free, free)
    tm2 = tm.copy()
    tm2[y, x + 1] = MAT_KINDLING
    x2, y2 = G._floor_site(tm2, store, g.width - 1, free, free)
    assert not np.isin(tm2[y2 - 1:y2 + 2, x2 - 1:x2 + 2],
                       G.fuel_materials()).any()


def test_the_int_fuel_form_is_v1s_furniture_wood_mix():
    """PROPERTY: the int ``fuel`` tag draws EXACTLY as the mix
    :data:`FUEL_MIX_V1` (furniture 0.6 / wood 0.4) under the same generator
    -- the int form is one point of the mix form, not a second path, and
    v1's ships keep their clusters (ship_demo's byte identity guards the
    rest). BREAKS IF the int form's draw or the mix walk's order changes."""
    a, _ = _run(_fuel_graph(4))
    b, _ = _run(_fuel_graph({"clusters": 4, "mix": dict(G.FUEL_MIX_V1)}))
    assert a.fuel_tiles > 0
    assert np.array_equal(a.tilemap, b.tilemap)


@pytest.mark.parametrize("name", ["steel", "hull", "glass", "door",
                                  "door_closed", "air", "unobtainium"])
def test_a_mix_naming_structure_or_no_material_is_refused_by_name(name):
    """PROPERTY: a fuel mix naming a structural row (hull, steel, glass, the
    doors), air or a name with no ``[materials.*]`` row is refused when the
    graph is built, naming the room and the material; a mix whose fractions
    do not sum to 1 is refused by the tag schema. BREAKS IF a mix could
    stamp a wall into a room as 'fuel', or a typo stamps a default."""
    with pytest.raises(G.GraphSchemaError, match=rf"room 1: .*'{name}'"):
        _fuel_graph({"clusters": 1, "mix": {name: 0.5, "kindling": 0.5}})
    with pytest.raises(G.GraphSchemaError, match=r"'fuel'"):
        _fuel_graph({"clusters": 1, "mix": {"kindling": 0.5, "wood": 0.4}})


# ---------------------------------------------------------------------------
# 3. The vent list
# ---------------------------------------------------------------------------

def _vent_graph(vent1, vent3=None, cell3=None) -> G.ShipGraph:
    """Spawn 0 | host | spawn 1 by doors, an annex below the host by a
    door; ``vent1`` on the host, ``vent3`` on the annex."""
    rooms = [G.Room(0, 4, 4, 10, 10, "spawn", {"spawn_team": 0}),
             G.Room(1, 15, 4, 14, 10, "section", {"vent": vent1}),
             G.Room(2, 30, 4, 10, 10, "spawn", {"spawn_team": 1}),
             G.Room(3, 15, 15, 14, 8, "room", {})]
    if vent3 is not None:
        rooms[3].tags["vent"] = vent3
    if cell3 is not None:
        for rm in rooms:
            rm.tags["cell"] = 0
        rooms[3].tags["cell"] = cell3
    kind = "door" if cell3 is None else "bulkhead"
    edges = [G.Edge(0, 1, "door", at=4), G.Edge(1, 2, "door", at=4),
             G.Edge(1, 3, kind)]
    return G.ShipGraph(width=44, height=27, hull=(3, 3, 40, 23),
                       rooms=rooms, edges=edges, mirror_x=False,
                       max_per_team=4, zombies=0)


def test_a_room_on_two_ducts_gets_a_site_per_duct_and_role():
    """PROPERTY (recipe doc §1.3, the duct link): a room whose ``vent`` tag
    is a list of two ducts gets one vent per (duct, role) on its own wall,
    on the right duct entity, every vent site clear of every other's 3x3
    (whatever its duct), ids unique, and the plan passes the gates. BREAKS
    IF the builder reads only the first duct of the list, two ducts' sites
    collide, or the ids of two ducts clash."""
    g = _vent_graph([{"duct": "dept", "roles": BOTH},
                     {"duct": "link", "roles": BOTH}],
                    vent3={"duct": "link", "roles": BOTH})
    emb, reasons = _run(g)
    assert reasons == []
    vents = [e for e in emb.entities if e.class_name == "vent"]
    ducts = {e.id for e in emb.entities if e.class_name == "duct"}
    assert {v.fields["duct"] for v in vents} <= ducts
    host = g.rooms[1]
    x0, y0, x1, y1 = host.box
    step = {"n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0)}

    def serves_host(v):                  # the air tile the vent faces
        dx, dy = step[v.fields["facing"]]
        x, y = v.fields["x"] + dx, v.fields["y"] + dy
        return x0 <= x <= x1 and y0 <= y <= y1
    mine = {(v.fields["duct"], v.fields["role"]) for v in vents
            if serves_host(v)}
    assert mine == {(f"duct_{d}", r) for d in ("dept", "link") for r in BOTH}
    ids = [v.id for v in vents]
    assert len(ids) == len(set(ids))
    sites = [(v.fields["x"], v.fields["y"]) for v in vents]
    for i, (ax, ay) in enumerate(sites):
        for (bx, by) in sites[i + 1:]:
            assert max(abs(ax - bx), abs(ay - by)) >= 3


def test_the_vent_rules_hold_per_duct_of_a_list():
    """PROPERTY: on a list tag the air-volume vent rule (vocabulary §0.6)
    runs PER DUCT (a room in its own air volume with a lone role on its
    second duct is refused naming that duct, the volume and the missing
    role, though its first duct carries both) and so does 'a duct never
    crosses a cell' (a list duct shared with a room of another cell is
    refused naming the duct); two entries on one duct are refused by the
    schema. BREAKS IF either rule checks only a list's first entry."""
    with pytest.raises(G.GraphSchemaError,
                       match=r"duct 'link': the air volume of rooms \[1\].*"
                             r"no 'return'"):
        _vent_graph([{"duct": "dept", "roles": BOTH},
                     {"duct": "link", "roles": ["supply"]}],
                    vent3={"duct": "link", "roles": BOTH})
    with pytest.raises(G.GraphSchemaError, match=r"duct 'link' spans 2 cells"):
        _vent_graph([{"duct": "dept", "roles": BOTH},
                     {"duct": "link", "roles": BOTH}],
                    vent3={"duct": "link", "roles": BOTH}, cell3=1)
    with pytest.raises(G.GraphSchemaError, match=r"'vent'"):
        _vent_graph([{"duct": "dept", "roles": ["supply"]},
                     {"duct": "dept", "roles": ["return"]}])


# ---------------------------------------------------------------------------
# 4. Lamps per length
# ---------------------------------------------------------------------------

def _lamp_graph(span: int | None) -> G.ShipGraph:
    """A mirrored strip: a 25-long corridor and its twin, a 50-long axis
    corridor (6 wide), and an 8x8 room and its twin; ``lamp_span`` on all
    of them (None: untagged). No edges: only the lamps are read."""
    W = 60
    t = {} if span is None else {"lamp_span": span}

    def pair(i, x, y, w, h):
        return [G.Room(i, x, y, w, h, "room", dict(t), mirror=i + 1),
                G.Room(i + 1, W - x - w, y, w, h, "room", dict(t), mirror=i)]
    rooms = pair(0, 4, 4, 25, 6) + [G.Room(2, 5, 12, 50, 6, "section",
                                           dict(t), mirror=2)] \
        + pair(3, 4, 20, 8, 8)
    return G.ShipGraph(width=W, height=32, hull=(3, 3, 56, 30),
                       rooms=rooms, edges=[], mirror_x=True,
                       max_per_team=4, zombies=0)


def _lamps_in(lights, rm):
    return [(l.x, l.y) for l in lights
            if rm.x <= l.x <= rm.x + rm.w and rm.y <= l.y <= rm.y + rm.h]


def _farthest(lamps, rm) -> float:
    return max(min(math.hypot(x + 0.5 - lx, y + 0.5 - ly)
                   for (lx, ly) in lamps)
               for y in range(rm.y, rm.y + rm.h)
               for x in range(rm.x, rm.x + rm.w))


def test_lamp_span_lights_a_long_corridor_and_mirrors():
    """PROPERTY (layout doc §11): with ``lamp_span = LAMP_SPAN`` a 50-tile
    corridor has no floor tile farther than the lamp range from one of its
    lamps (untagged, its one centre lamp leaves its ends dark: the property
    has teeth); a room shorter than the span keeps exactly the one centred
    lamp an untagged room has; a mirror twin's lamps are the reflection of
    its twin's. BREAKS IF lamps stop following the long axis, a short room
    grows extra lamps, the spans go lopsided so the twins disagree, or the
    untagged lamp moves (ship_demo would move with it)."""
    tagged = G.embed(_lamp_graph(G.LAMP_SPAN), RNG()).lights
    plain = G.embed(_lamp_graph(None), RNG()).lights
    g = _lamp_graph(G.LAMP_SPAN)
    corridor, short = g.rooms[2], g.rooms[3]
    assert _farthest(_lamps_in(tagged, corridor), corridor) <= G.LAMP_RANGE
    assert _farthest(_lamps_in(plain, corridor), corridor) > G.LAMP_RANGE
    assert _lamps_in(tagged, short) == _lamps_in(plain, short) \
        == [(short.x + short.w / 2.0, short.y + short.h / 2.0)]
    for a, b in ((g.rooms[0], g.rooms[1]), (g.rooms[3], g.rooms[4]),
                 (corridor, corridor)):
        ra = sorted((round(g.width - x, 9), round(y, 9))
                    for (x, y) in _lamps_in(tagged, a))
        rb = sorted((round(x, 9), round(y, 9))
                    for (x, y) in _lamps_in(tagged, b))
        assert ra == rb and len(ra) >= 1
    assert len(_lamps_in(tagged, g.rooms[0])) > 1    # the strip is lit along
