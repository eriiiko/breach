"""Properties of the CARGO DOOR IN THE HULL (#133; levelgen vocabulary
docs/levelgen_zone_vocabulary_2026-10-29.md B.4 and Part E.2 row 5): a room's
``hull_door`` tag makes stage 2 cut an A6 ``door`` entity into the room's
hull wall -- the cargo hold's depressurisation valve -- emitted CLOSED.

Closed it is a wall (the airtight gate passes it unchanged: the closed stamp
``MAT_DOOR_CLOSED`` has permeability 0); opened in play it is a breach the
engine already knows (``unseal_tiles`` joins the span to vacuum, the room's
air and smoke leave); closed again the room stops losing gas. Every test
names the property it protects and the change that must break it. The
hand-built graphs are minimal fixtures (tests/test_gen_ship_vents_doors.py's
host + annex), not layouts the generator owes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_ship as G  # noqa: E402
import level_loader  # noqa: E402
import test_gen_ship_vents_doors as V  # noqa: E402
from level_airtight import find_leaks  # noqa: E402
from simulation.materials import MAT_AIR, MAT_DOOR_CLOSED  # noqa: E402

try:
    import breach_physics as _bp  # noqa: E402
except Exception:                  # pragma: no cover - a box without the build
    _bp = None
needs_bp = pytest.mark.skipif(_bp is None,
                              reason="needs the compiled breach_physics")

FP_ONE = 65536


def _cargo_graph(tag=None) -> G.ShipGraph:
    """The airlock fixture with the annex as a CARGO BAY: its own pressure
    cell, joined to the host by an interior airlock, its bottom wall the
    hull, carrying a ``hull_door`` (the real-ship pattern, vocabulary
    D4/D6: a hold with a door to space, entered from the spine through an
    airlock)."""
    g = V._airlock_graph()
    g.rooms[3].tags["hull_door"] = {} if tag is None else tag
    g.check()
    return g


def _mirrored_graph(tags: dict) -> G.ShipGraph:
    """A mirrored fixture: wing rooms 0 / 2 (twins), axis rooms 1 / 3,
    every room touching the hull. ``tags``: room id -> extra tags."""
    base = {0: {"spawn_team": 0}, 1: {}, 2: {"spawn_team": 1}, 3: {}}
    boxes = {0: (4, 4, 10, 10, "spawn"), 1: (15, 4, 10, 10, "section"),
             2: (26, 4, 10, 10, "spawn"), 3: (15, 15, 10, 9, "room")}
    mirror = {0: 2, 1: 1, 2: 0, 3: 3}
    rooms = [G.Room(i, *boxes[i][:4], boxes[i][4],
                    {**base[i], **tags.get(i, {})}, mirror=mirror[i])
             for i in range(4)]
    edges = [G.Edge(0, 1, "door", at=4), G.Edge(1, 2, "door", at=4),
             G.Edge(1, 3, "door")]
    return G.ShipGraph(width=40, height=28, hull=(3, 3, 36, 24), rooms=rooms,
                       edges=edges, mirror_x=True, max_per_team=4, zombies=0)


def _hull_door(emb, rid=3):
    d, = [e for e in emb.entities if e.id == f"hull_door_r{rid}"]
    return d


# ---------------------------------------------------------------------------
# Stage 2 + the gates
# ---------------------------------------------------------------------------

def test_a_hull_door_is_a_closed_door_entity_in_the_hull_and_airtight():
    """PROPERTY: a ``hull_door`` room gets ONE A6 door entity whose span is
    hull wall with SPACE one step beyond and the room's floor within,
    initial state ``closed``, stamped ``MAT_DOOR_CLOSED`` in the CSV (the
    loader's own closed stamp), 3 tiles wide by default; and the plan passes
    every gate -- the airtight flood fill treats the closed door as solid --
    while the SAME tiles opened to air leak to vacuum (the gate is not
    vacuous: it is the door that seals the hull). BREAKS IF the door lands
    off the hull, opens by default, the CSV disagrees with the entity, or
    the closed stamp stops being gas-tight."""
    from simulation.entities.door import base_span
    g = _cargo_graph()
    emb, reasons = V._run(g)
    assert reasons == []
    d = _hull_door(emb)
    assert d.fields["initial_state"] == "closed"
    span = base_span(d.fields, G.TILE_SIZE_M)
    _o, tiles, beyond = G.hull_door_site(g, g.rooms[3])
    assert span == list(tiles) and len(span) == G.HULL_DOOR_W
    x0, y0, x1, y1 = g.rooms[3].box
    for (y, x), (by, bx) in zip(tiles, beyond):
        assert emb.tilemap[y, x] == MAT_DOOR_CLOSED
        assert emb.tilemap[by, bx] == level_loader.SPACE_CODE
        iy, ix = 2 * y - by, 2 * x - bx
        assert x0 <= ix <= x1 and y0 <= iy <= y1
        assert emb.tilemap[iy, ix] == MAT_AIR
    _mat, vac, solid, _walk = G._masks(emb.tilemap)
    assert all(solid[y, x] for (y, x) in tiles)
    opened = emb.tilemap.copy()
    for (y, x) in tiles:
        opened[y, x] = MAT_AIR
    _m, vac2, solid2, _w = G._masks(opened)
    _, vac_leak, _edge = find_leaks(solid2, vac2)
    assert vac_leak, "an open hull door must be a leak, or the test is vacuous"
    assert any("airtight" in r for r in
               G.validate(opened, emb.spawn_rooms, mirrored=False, graph=g))


def test_a_hull_door_keeps_the_generator_s_compartments():
    """PROPERTY: a closed hull door bounds nothing new -- the plan realises
    exactly the graph's compartments (four rooms + the vestibule), the
    hull door's span is structure. BREAKS IF the labeller counts the door
    span as room air (a hull door would then join its room to space)."""
    g = _cargo_graph()
    emb, reasons = V._run(g)
    assert reasons == []
    clab, cn = G.compartments(emb.tilemap, g)
    _o, tiles, _b = G.hull_door_site(g, g.rooms[3])
    assert all(clab[y, x] == 0 for (y, x) in tiles)
    plain_emb, _ = V._run(V._airlock_graph())
    assert cn == G.compartments(plain_emb.tilemap, V._airlock_graph())[1]


def test_an_open_hull_door_is_refused_naming_the_breach_archetype():
    """PROPERTY: v1 refuses ``state = "open"`` at the graph, by name,
    pointing at the future breach archetype (an open hull door is a room
    open to space, which the airtight gate would reject on every sub-seed).
    BREAKS IF an open hull door is accepted silently, or refused only by
    the airtight gate's anonymous leak count."""
    with pytest.raises(G.GraphSchemaError, match="BREACH archetype"):
        _cargo_graph({"state": "open"})


@pytest.mark.parametrize("bad", [{"width": 0}, {"state": "ajar"},
                                 {"wide": 3}, "closed"])
def test_a_malformed_hull_door_tag_is_refused(bad):
    """PROPERTY: the tag's shape is checked by the ROOM_TAGS door, naming
    the tag. BREAKS IF a malformed tag reaches stage 2."""
    with pytest.raises(G.GraphSchemaError, match="hull_door"):
        _cargo_graph(bad)


def test_a_room_without_a_hull_wall_or_too_short_a_run_is_refused():
    """PROPERTY: a ``hull_door`` on a room no side of which lies against
    the hull is refused by name, and so is a door wider than the room's
    longest hull run. BREAKS IF the door is cut into an interior wall (its
    far side would be another room, not space) or clipped silently."""
    g = V._airlock_graph()
    inner = G.ShipGraph(width=g.width, height=g.height, hull=(2, 2, 37, 25),
                        rooms=g.rooms, edges=g.edges, mirror_x=False,
                        max_per_team=g.max_per_team, zombies=0)
    inner.rooms[3].tags["hull_door"] = {}
    with pytest.raises(G.GraphSchemaError, match="no hull wall"):
        inner.check()
    with pytest.raises(G.GraphSchemaError, match="does not fit"):
        _cargo_graph({"width": 11})


def test_mirrored_hull_doors_keep_the_grid_symmetric():
    """PROPERTY: in a mirrored graph a wing room's hull door and its twin's
    are mirror images (the twin REFLECTS the lower id's site: with the
    default width on a 10-tile run, centring each independently would land
    them one tile apart), an axis room's door is centred on the axis, and
    the plan passes the symmetry gate; an axis door whose width cannot be
    centred on its run is refused by name. BREAKS IF twin doors are placed
    independently, or an axis room takes a side wall."""
    g = _mirrored_graph({0: {"hull_door": {}}, 2: {"hull_door": {}},
                         1: {"hull_door": {"width": 4}},
                         3: {"hull_door": {"width": 2}}})
    emb = G.embed(g, V.RNG())
    reasons = G.validate(emb.tilemap, emb.spawn_rooms, mirrored=True,
                         graph=g)
    assert reasons == []
    assert np.array_equal(emb.tilemap, emb.tilemap[:, ::-1])
    W = g.width
    _o, t0, _ = G.hull_door_site(g, g.rooms[0])
    _o, t2, _ = G.hull_door_site(g, g.rooms[2])
    assert sorted((y, W - 1 - x) for (y, x) in t0) == list(t2)
    for rid in (1, 3):
        orient, t, _ = G.hull_door_site(g, g.rooms[rid])
        assert orient == "h"
        assert sorted((y, W - 1 - x) for (y, x) in t) == list(t)
    assert len([e for e in emb.entities if e.class_name == "door"
                and e.id.startswith("hull_door_")]) == 4
    with pytest.raises(G.GraphSchemaError, match="mirror axis"):
        _mirrored_graph({1: {"hull_door": {}}})


# ---------------------------------------------------------------------------
# The loaded level: a valve the engine already knows how to open
# ---------------------------------------------------------------------------

@needs_bp
def test_the_cargo_bay_decompresses_through_its_hull_door_and_stops_when_closed(
        tmp_path):
    """PROPERTY (B.4, D7): the generated cargo bay -- its own cell behind an
    interior airlock, a closed hull door to space -- writes, loads and runs
    headless. Closed, the ship's bulk air is conserved to the LSB and the
    bay stays at 1 atm. Opened (the door's ``want_open`` latch: the hull
    door is unwired, so the Arc-A latch is its drive), the bay's air and
    the smoke seeded in it leave through the gap -- the engine's existing
    breach behaviour, no new physics. Closed again, the bay loses no more
    gas. BREAKS IF the hull door leaks while closed, does not open onto
    space (a door the loader or the sweep cannot open in the hull), or
    keeps venting after it closes."""
    from simulation import Simulation
    from simulation.field_edit import EditMode, Falloff, FieldEdit, Region
    from simulation.gamemap import INERT_N2, O2
    from simulation.gases import SMOKE
    from simulation.entities.door import DOOR_CLOSED, DOOR_OPEN
    g = _cargo_graph()
    plan, lv = V._write_load(g, tmp_path, "cargo")
    sim = Simulation(lv, seed=1, breach_physics=_bp, enable_recorder=False)
    gm = sim.gmap
    bay = g.rooms[3]
    x0, y0, x1, y1 = bay.box
    box = (slice(y0, y1 + 1), slice(x0, x1 + 1))

    def bulk(sel=Ellipsis):
        return int(gm.gas[O2][sel].astype(np.int64).sum()
                   + gm.gas[INERT_N2][sel].astype(np.int64).sum())

    def p_bay():
        return float(gm.atmosphere[box].astype(np.int64).mean()) / FP_ONE

    def smoke():
        return int(gm.gas[SMOKE][box].astype(np.int64).sum())

    def run(n):
        for _ in range(n):
            sim.set_paused(False)
            sim.step()

    door, = [d for d in sim._doors if d.id == f"hull_door_r{bay.id}"]
    assert door.state == DOOR_CLOSED
    sim.edit_queue.enqueue(FieldEdit(
        field="smoke", region=Region.DISC,
        coords=(y0 + bay.h // 2, x0 + bay.w // 2, 3.0), amount=4.0,
        mode=EditMode.ADD, falloff=Falloff.FLAT, source_id=991))
    run(1)
    n_ship0, n_bay0, smoke0 = bulk(), bulk(box), smoke()
    assert smoke0 > 0, "no smoke seeded: vacuous"
    run(60)                                     # closed: a wall
    assert bulk() == n_ship0 and bulk(box) == n_bay0
    assert abs(p_bay() - 1.0) < 0.01

    door.want_open = True                       # the valve opens
    run(60)
    assert door.state == DOOR_OPEN
    assert p_bay() < 0.1, p_bay()
    assert bulk(box) < n_bay0 // 10
    assert smoke() < smoke0 // 10
    vented = n_ship0 - bulk()
    assert vented > 0

    door.want_open = False                      # and closes
    run(2)
    assert door.state == DOOR_CLOSED
    n_ship1, n_bay1 = bulk(), bulk(box)
    run(100)
    assert bulk() == n_ship1 and bulk(box) == n_bay1
