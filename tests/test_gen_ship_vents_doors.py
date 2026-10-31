"""Properties of gen_ship's stage-2 entity half (#133 patch 2;
docs/levelgen_zone_vocabulary_2026-10-29.md §0.4, §0.5, B.8, Part E.1 rows
5-6; docs/breach_levelgen_design_v0.1.md L5): the multi-duct vent builder,
vent_only edges, entity doors and airlock pairs.

Each test names the property it protects and the change that must break it.
The hand-built graphs are minimal fixtures, not layouts the generator owes.
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
import level_loader  # noqa: E402
from level_airtight import find_leaks  # noqa: E402
from simulation.materials import MAT_AIR, MAT_DOOR_CLOSED  # noqa: E402

try:
    import breach_physics as _bp  # noqa: E402
except Exception:                  # pragma: no cover - a box without the build
    _bp = None
needs_bp = pytest.mark.skipif(_bp is None,
                              reason="needs the compiled breach_physics")

RNG = lambda: np.random.Generator(np.random.PCG64(0))  # noqa: E731
BOTH = ["supply", "return"]


def _graph(kind: str = "door", *, tags: dict | None = None,
           ducts: dict | None = None) -> G.ShipGraph:
    """Spawn 0 | host section | spawn 1 joined by doors, and an ANNEX below
    the host joined to it by one ``kind`` edge. ``tags``: room id -> extra
    tags."""
    tags = tags or {}
    base = {0: {"spawn_team": 0}, 1: {"fuel": 1}, 2: {"spawn_team": 1},
            3: {}}
    boxes = {0: (4, 4, 10, 10, "spawn"), 1: (15, 4, 10, 10, "section"),
             2: (26, 4, 10, 10, "spawn"), 3: (15, 15, 10, 9, "room")}
    rooms = [G.Room(i, *boxes[i][:4], boxes[i][4],
                    {**base[i], **tags.get(i, {})}) for i in range(4)]
    # The side doors sit at the host's top rows, clear of a vestibule
    # carved at its bottom.
    edges = [G.Edge(0, 1, "door", at=4), G.Edge(1, 2, "door", at=4),
             G.Edge(1, 3, kind)]
    return G.ShipGraph(width=40, height=28, hull=(3, 3, 36, 24),
                       rooms=rooms, edges=edges, mirror_x=False,
                       max_per_team=4, zombies=0, ducts=dict(ducts or {}))


def _run(graph):
    emb = G.embed(graph, RNG())
    reasons = G.validate(emb.tilemap, emb.spawn_rooms, mirrored=False,
                         graph=graph)
    return emb, reasons


def _plan(graph) -> G.ShipPlan:
    emb, reasons = _run(graph)
    assert reasons == []
    return G.ShipPlan(seed=0, size="fixture", attempt=0, graph=graph,
                      embedding=emb)


def _write_load(graph, tmp_path, name="fixture"):
    plan = _plan(graph)
    G.write(plan, tmp_path / name)
    return plan, level_loader.load(name, levels_dir=str(tmp_path))


def _box_sum(plane, box) -> int:
    x0, y0, x1, y1 = box
    return int(plane[y0:y1 + 1, x0:x1 + 1].astype(np.int64).sum())


# ---------------------------------------------------------------------------
# The multi-duct vent builder
# ---------------------------------------------------------------------------

def test_two_ducts_make_two_duct_entities_with_their_own_filters():
    """PROPERTY: rooms on two named ducts produce one duct entity per name,
    each with the filter the graph's ``ducts`` annotation gives it; every
    vent id is unique and names an existing duct; a room carrying both roles
    gets a supply AND a return on two distinct wall sites; the written level
    loads through the loader door. BREAKS IF the builder goes back to one
    ``duct_main`` (the second duct's rooms land on the first), ids collide
    (``vent_<role>`` for two supplies), or the filter is not per duct."""
    g = _graph(tags={0: {"vent": {"duct": "port", "roles": BOTH}},
                     2: {"vent": {"duct": "stbd", "roles": BOTH}}},
               ducts={"port": {"filter": "derelict"},
                      "stbd": {"filter": "hepa_basic"}})
    emb, reasons = _run(g)
    assert reasons == []
    ducts = {e.id: e.fields["filter"] for e in emb.entities
             if e.class_name == "duct"}
    assert ducts == {"duct_port": "derelict", "duct_stbd": "hepa_basic"}
    vents = [e for e in emb.entities if e.class_name == "vent"]
    ids = [v.id for v in vents]
    assert len(ids) == len(set(ids)) == 4
    assert {v.fields["duct"] for v in vents} == set(ducts)
    sites = {(v.fields["x"], v.fields["y"]) for v in vents}
    assert len(sites) == 4
    for duct in ducts:
        assert sorted(v.fields["role"] for v in vents
                      if v.fields["duct"] == duct) == ["return", "supply"]


def test_the_v1_shorthand_keeps_the_main_duct_and_its_ids():
    """PROPERTY: v1's string tags mean the ``main`` duct on ``hepa_basic``
    with v1's ids (``duct_main``, ``vent_supply``, ``vent_return``), so a v1
    ship writes the file it always wrote. BREAKS IF the shorthand is mapped
    to the new id scheme or another filter (levels/ship_demo would move;
    the byte-identity test in test_gen_ship_edges also guards this)."""
    g = _graph(tags={0: {"vent": "supply"}, 2: {"vent": "return"}})
    emb, _ = _run(g)
    assert [(e.id, e.fields.get("filter")) for e in emb.entities
            if e.class_name == "duct"] == [("duct_main", G.FILTER)]
    assert sorted(e.id for e in emb.entities if e.class_name == "vent") == \
        ["vent_return", "vent_supply"]


def test_a_duct_spanning_two_cells_is_refused_by_name():
    """PROPERTY: a duct whose rooms lie in two pressure cells is refused
    when the graph is built, naming the duct and the cells (§0.5: a duct
    never crosses a cell boundary). BREAKS IF the cell check is dropped or
    stops treating untagged rooms as one implicit cell."""
    with pytest.raises(G.GraphSchemaError,
                       match=r"duct 'spine' spans 2 cells"):
        _graph("bulkhead",
               tags={1: {"vent": {"duct": "spine", "roles": BOTH}},
                     3: {"cell": 1,
                         "vent": {"duct": "spine", "roles": BOTH}}})


def test_a_sealed_room_with_a_lone_vent_role_is_refused():
    """PROPERTY: a room whose every edge bounds a compartment (a sealed
    box) carrying only a supply is refused by name (no pressure feedback:
    it would be pumped up until a membrane bursts); the same room joined by
    an ``open`` edge (its air returns through the doorway) is accepted.
    BREAKS IF the sealed-room rule is dropped, or applied to a room that
    breathes through an open edge."""
    lone = {"vent": {"duct": "q", "roles": ["supply"]}}
    ret = {"vent": {"duct": "q", "roles": ["return"]}}
    with pytest.raises(G.GraphSchemaError, match=r"room 3 is sealed"):
        _graph("door", tags={3: lone, 1: {"vent": {"duct": "q",
                                                    "roles": BOTH}}})
    _emb, reasons = _run(_graph("open", tags={3: lone, 1: ret}))
    assert reasons == []


# ---------------------------------------------------------------------------
# vent_only edges and the B.8 secret room
# ---------------------------------------------------------------------------

def test_a_vent_only_edge_off_one_duct_is_refused():
    """PROPERTY: a ``vent_only`` edge whose rooms are on different ducts
    (or not on a duct) is refused by name: stage 2 cuts nothing in the
    wall, so the duct IS the air path. BREAKS IF a vent_only edge is
    accepted without a shared duct (a sealed room with no path at all)."""
    with pytest.raises(G.GraphSchemaError, match=r"not on one duct"):
        _graph("vent_only",
               tags={1: {"vent": {"duct": "a", "roles": BOTH}},
                     3: {"secret": True,
                         "vent": {"duct": "b", "roles": BOTH}}})


def _b8_graph(filter_name: str) -> G.ShipGraph:
    """B.8: the hidden annex carries a supply AND a return on the HOST's
    duct; the host carries both too (it is sealed)."""
    vent = {"vent": {"duct": "host", "roles": BOTH}}
    return _graph("vent_only", tags={1: dict(vent),
                                     3: {"secret": True, **vent}},
                  ducts={"host": {"filter": filter_name}})


@needs_bp
def test_b8_smoke_in_the_host_reaches_the_hidden_room_only_through_derelict(
        tmp_path):
    """PROPERTY (B.8): smoke seeded in the host room reaches the sealed
    hidden room through the shared duct when its filter passes smoke
    (``derelict``), and NEVER when it scrubs smoke (``hepa_basic``); the
    hidden room has no other path (its wall is uncut, the level airtight).
    BREAKS IF the hidden room's vents land on another duct, the builder
    drops the per-duct filter, a vent_only edge starts cutting the wall, or
    the vents are wired the wrong way (critique finding 3)."""
    from simulation import Simulation
    from simulation.field_edit import EditMode, Falloff, FieldEdit, Region
    from simulation.gases import SMOKE
    hidden = {}
    for fname in ("derelict", "hepa_basic"):
        g = _b8_graph(fname)
        _plan_, lv = _write_load(g, tmp_path, f"b8_{fname}")
        sim = Simulation(lv, seed=1, breach_physics=_bp,
                         enable_recorder=False)
        host, annex = g.rooms[1], g.rooms[3]
        sim.edit_queue.enqueue(FieldEdit(
            field="smoke", region=Region.DISC,
            coords=(host.y + host.h // 2, host.x + host.w // 2, 4.0),
            amount=4.0, mode=EditMode.ADD, falloff=Falloff.FLAT,
            source_id=991))
        for _ in range(360):
            sim.set_paused(False)
            sim.step()
        smoke = sim.gmap.gas[SMOKE]
        assert _box_sum(smoke, host.box) > 0, "no smoke in the host: vacuous"
        hidden[fname] = _box_sum(smoke, annex.box)
    assert hidden["derelict"] > 0
    assert hidden["hepa_basic"] == 0


# ---------------------------------------------------------------------------
# Entity doors
# ---------------------------------------------------------------------------

def test_an_entity_door_edge_stamps_a_door_entity_and_bounds_its_rooms():
    """PROPERTY: an ``entity_door`` edge emits one A6 ``door`` entity whose
    canonical span (``door.base_span``) is exactly the edge's opening, in
    the state ``ENTITY_DOOR_INITIAL`` stamps (air while open), and the door
    still bounds a compartment: four rooms, four compartments. BREAKS IF
    the door's anchor / orientation / length miss the opening (the loader
    would stamp a different span), the CSV disagrees with the door's own
    stamp, or the labeller merges the rooms through the open door."""
    from simulation.entities.door import base_span
    g = _graph("entity_door")
    emb, reasons = _run(g)
    assert reasons == []
    doors = [e for e in emb.entities if e.class_name == "door"]
    assert len(doors) == 1
    opening = G._opening_tiles(g, g.edges[2])
    assert base_span(doors[0].fields, G.TILE_SIZE_M) == opening
    assert doors[0].fields["initial_state"] == G.ENTITY_DOOR_INITIAL
    stamp = MAT_AIR if G.ENTITY_DOOR_INITIAL == "open" else MAT_DOOR_CLOSED
    assert all(emb.tilemap[y, x] == stamp for (y, x) in opening)
    assert G.compartments(emb.tilemap, g)[1] == 4


@needs_bp
def test_the_gates_see_an_entity_door_as_the_sim_does(tmp_path):
    """PROPERTY: what the reachability gate walks on equals what the loaded
    sim's GameMap lets a unit enter, tile for tile (the gate's ``walk`` mask
    vs the GameMap's ``mobility > 0`` after the loader's door stamp), and a
    3x3 unit can stand in the door's span in the sim. The bots never open a
    door (no USE in control_bots) and a closed entity door has mobility 0,
    so stage 2 emits them open; this test is where a closed default would
    fail. BREAKS IF the CSV and the loader's stamp disagree, or entity
    doors go back to closed while no door-opening rule exists (#142)."""
    from simulation import Simulation
    g = _graph("entity_door")
    plan, lv = _write_load(g, tmp_path)
    sim = Simulation(lv, seed=1, breach_physics=_bp, enable_recorder=False)
    gm = sim.gmap
    _mat, vac, _solid, walk = G._masks(plan.tilemap)
    sim_walk = np.asarray(gm.materials.mobility)[gm.material] > 0
    assert np.array_equal(walk & ~vac, sim_walk & ~gm.is_vacuum)
    (y0, x0), (y1, x1) = G._opening_tiles(g, g.edges[2])[0], \
        G._opening_tiles(g, g.edges[2])[-1]
    assert y0 == y1                                   # a horizontal wall
    assert gm.is_passable_block(y0 - 1, x0, 3)        # a unit in the doorway


# ---------------------------------------------------------------------------
# Airlock pairs (L5)
# ---------------------------------------------------------------------------

def _airlock_graph() -> G.ShipGraph:
    """The annex is its own pressure cell, joined to the host by an
    airlock (the vestibule carved on the host's side)."""
    return _graph("airlock", tags={0: {"cell": 0}, 1: {"cell": 0},
                                   2: {"cell": 0}, 3: {"cell": 1}})


def test_an_airlock_edge_is_a_vestibule_two_doors_and_one_controller():
    """PROPERTY: an ``airlock`` edge yields a vestibule that is its own,
    airtight compartment (rooms + 1), two door entities (inner open, outer
    closed, their spans the vestibule's two faces), exactly one
    ``airlock_controller`` in the mode its edge calls for (two interior
    rooms: ``interior``, #133), every controller input that mode reads
    wired exactly once, every endpoint an emitted entity. The plan passes every gate. BREAKS
    IF the vestibule leaks into its host, a door misses its face, the
    controller is duplicated or unwired, or the vestibule count is dropped
    from the compartment gate."""
    g = _airlock_graph()
    emb, reasons = _run(g)
    assert reasons == []
    by_class: dict = {}
    for e in emb.entities:
        by_class.setdefault(e.class_name, []).append(e)
    assert len(by_class["airlock_controller"]) == 1
    doors = {e.fields["initial_state"]: e for e in by_class["door"]}
    assert len(by_class["door"]) == 2 and set(doors) == {"open", "closed"}
    (_i, al), = G._airlocks(g)
    from simulation.entities.door import base_span
    assert base_span(doors["open"].fields, G.TILE_SIZE_M) == list(al.inner)
    assert base_span(doors["closed"].fields, G.TILE_SIZE_M) == list(al.outer)
    ids = {e.id for e in emb.entities}
    ctrl, = by_class["airlock_controller"]
    assert ctrl.fields["mode"] == "interior"
    needs = ("presence", "inner_open", "outer_open", "inner_alive",
             "outer_alive", "chamber_p", "inner_side_p", "outer_side_p")
    into = [dst.split(".")[1] for _src, dst in emb.wires
            if dst.split(".")[0] == ctrl.id]
    assert sorted(into) == sorted(needs)
    for src, dst in emb.wires:
        assert src.split(".")[0] in ids and dst.split(".")[0] in ids
    clab, cn = G.compartments(emb.tilemap, g)
    assert cn == 4 + 1
    x0, y0, x1, y1 = al.chamber
    vest = clab == clab[y0, x0]
    assert vest.sum() == (x1 - x0 + 1) * (y1 - y0 + 1)   # exactly the chamber
    _m, vac, solid, _w = G._masks(emb.tilemap)
    _, vac_leak, edge_leak = find_leaks(solid, vac)
    assert not vac_leak and not edge_leak


@pytest.mark.parametrize("kind", ["door", "open", "entity_door", "window"])
def test_an_edge_across_two_cells_must_be_an_airlock_or_bulkhead(kind):
    """PROPERTY (L5): an edge whose rooms carry different ``cell`` tags is
    refused by name unless it is an airlock or a bulkhead; the airlock and
    the bulkhead versions of the same graph build. BREAKS IF the L5 check
    is dropped or lets a plain door join two cells."""
    tags = {0: {"cell": 0}, 1: {"cell": 0}, 2: {"cell": 0}, 3: {"cell": 1}}
    with pytest.raises(G.GraphSchemaError, match=r"\(L5\)"):
        _graph(kind, tags=tags)
    for ok in G.CELL_CROSSING_KINDS:
        _graph(ok, tags=tags)


_HEADLESS = r"""
import sys
from pathlib import Path
root = Path.cwd()
sys.path[:0] = [str(root), str(root / 'src'), str(root / 'tools'),
                str(root / 'cpp' / 'build' / 'Release'), str(root / 'tests')]
import level_loader, level_airtight
import breach_physics as bp
from simulation import Simulation
from simulation.entities.actuators import AIRLOCK_OPEN_FAR, AIRLOCK_STATE_NAMES
from simulation.entities.door import DOOR_OPEN
from simulation.logic_nodes import AirlockControllerRuntime
from simulation.unit import Unit
d, name, ux, uy = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
lv = level_loader.load(name, levels_dir=d)
assert level_airtight.check(name, levels_dir=d)
sim = Simulation(lv, seed=1, breach_physics=bp, enable_recorder=False)
ctrl = next(e for e in sim.entities if isinstance(e, AirlockControllerRuntime))
outer = next(x for x in sim._doors if x.id.endswith('_outer'))
for _ in range(48):                      # empty: the airlock idles
    sim.set_paused(False); sim.step()
idle = AIRLOCK_STATE_NAMES[ctrl.state]
sim.add_unit(Unit('Alpha', x=ux, y=uy, team=0, footprint=3))
ticks = None
for t in range(600):                     # a unit in the vestibule: it cycles
    sim.set_paused(False); sim.step()
    if ctrl.state == AIRLOCK_OPEN_FAR and outer.state == DOOR_OPEN:
        ticks = t + 1
        break
bad = [m for m in sys.modules
       if m in ('pyray', 'raylib') or m.split('.')[0] == 'renderer']
print('RESULT', idle, ticks, len(lv.wires), bad)
"""


@needs_bp
def test_an_airlock_level_loads_runs_headless_and_cycles_for_a_unit(
        tmp_path):
    """PROPERTY: a written level with an airlock loads through the loader
    door (every wire resolves), passes the GameMap airtight check, idles
    while its vestibule is empty, and -- when a unit stands in the vestibule
    -- seals, equalises (an interior pair, #133) and OPENS THE OUTER DOOR, all without importing the
    renderer. That is what the reachability gate's "the outer door is
    passable" rests on. BREAKS IF the recipe stops matching the airlock
    runtime (a wire, a mount tile, the plate off the chamber, the probe
    sampling a wall), or the gate's assumption stops being true."""
    import subprocess
    g = _airlock_graph()
    plan, _lv = _write_load(g, tmp_path)
    (_i, al), = G._airlocks(g)
    x0, y0, _x1, _y1 = al.chamber
    proc = subprocess.run(
        [sys.executable, "-c", _HEADLESS, str(tmp_path), "fixture",
         str(x0 + 1), str(y0 + 1)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
    assert line, proc.stdout[-2000:]
    _r, idle, ticks, n_wires, bad = line[0].split(" ", 4)
    assert idle == "IDLE"
    assert ticks != "None", "the outer door never opened for a unit inside"
    assert n_wires == str(len(plan.embedding.wires))
    assert bad == "[]"
