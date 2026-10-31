"""Properties of gen_ship's stage-2 edge kinds, graph compartments, secret
reachability and room tags (#133; docs/levelgen_zone_vocabulary_2026-10-29.md
§0.4, B.8, Part E.1).

Each test names the property it protects and the change that must break it.
The hand-built graphs are minimal fixtures, not layouts the generator owes.
"""
from __future__ import annotations

import re
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
from level_airtight import find_leaks  # noqa: E402
from simulation.materials import MAT_STEEL, MAT_WOOD  # noqa: E402

RNG = lambda: np.random.Generator(np.random.PCG64(0))  # noqa: E731


def _graph(kind: str, annex_tags: dict | None = None, *,
           mid_tags: dict | None = None, cell: int | None = None
           ) -> G.ShipGraph:
    """Spawn 0 | zombie section | spawn 1 joined by doors, and an ANNEX room
    below the section joined to it by one ``kind`` edge (the annex shares a
    wall with the section only). A ``vent_only`` edge's two rooms ride one
    duct with both roles each (its air path IS the duct; #133 patch 2)."""
    duct = ({"vent": {"duct": "aux", "roles": ["supply", "return"]}}
            if kind == "vent_only" else {})
    rooms = [G.Room(0, 4, 4, 10, 10, "spawn", {"spawn_team": 0}),
             G.Room(1, 15, 4, 10, 10, "section",
                    {"zombie": True, "fuel": 1, **duct, **(mid_tags or {})}),
             G.Room(2, 26, 4, 10, 10, "spawn", {"spawn_team": 1}),
             G.Room(3, 15, 15, 10, 6, "room",
                    {**duct, **(annex_tags or {})})]
    if cell is not None:                  # one pressure cell for every room
        for rm in rooms:
            rm.tags["cell"] = cell
    edges = [G.Edge(0, 1, "door"), G.Edge(1, 2, "door", at=6),
             G.Edge(1, 3, kind)]
    return G.ShipGraph(width=40, height=25, hull=(3, 3, 36, 21),
                       rooms=rooms, edges=edges, mirror_x=False,
                       max_per_team=4, zombies=2)


def _run(graph):
    emb = G.embed(graph, RNG())
    reasons = G.validate(emb.tilemap, emb.spawn_rooms, mirrored=False,
                         graph=graph)
    return emb, reasons


def _airtight(tm) -> bool:
    _mat, vac, solid, _walk = G._masks(tm)
    _, vac_leak, edge_leak = find_leaks(solid, vac)
    return not vac_leak and not edge_leak


def test_an_open_edge_makes_one_compartment_a_door_two():
    """PROPERTY: an ``open`` edge (a MAT_AIR gap) joins its two rooms into
    ONE compartment, in the graph and in the tiles, and the room-count gate
    compares against the graph's COMPARTMENT count, so the plan passes; the
    same graph with a ``door`` edge has one compartment per room and also
    passes. BREAKS IF the gate goes back to comparing tile compartments
    against the graph's room count (the open plan is refused), or the open
    kind stops bounding nothing."""
    g_open = _graph("open")
    emb, reasons = _run(g_open)
    assert reasons == []
    assert max(g_open.compartment_of()) + 1 == 3
    assert G.compartments(emb.tilemap, g_open)[1] == 3
    g_door = _graph("door")
    emb, reasons = _run(g_door)
    assert reasons == []
    assert max(g_door.compartment_of()) + 1 == 4
    assert G.compartments(emb.tilemap, g_door)[1] == 4


@pytest.mark.parametrize("kind", ["breakable", "vent_only"])
def test_a_secret_room_behind_a_secret_edge_passes(kind):
    """PROPERTY: a room tagged ``secret`` whose only edge is a breakable
    (wood segment) or vent_only (nothing cut) edge is its OWN compartment
    (a breakable's wood is structure, not a crate), passes the reachability
    gate (reachable in principle: cut the wall / open the duct) and the
    airtight gate untouched; the SAME room without the tag is refused as
    unreachable. BREAKS IF compartments() counts the breakable wood as room
    contents again (host and annex merge), the reachability gate stops
    excusing a secret compartment, or starts excusing an untagged one."""
    g = _graph(kind, {"secret": True})
    emb, reasons = _run(g)
    assert reasons == []
    assert _airtight(emb.tilemap)
    assert G.compartments(emb.tilemap, g)[1] == 4
    wall = [emb.tilemap[y, x] for (y, x) in
            G._opening_tiles(g, g.edges[2])]
    if kind == "breakable":
        assert wall and all(m == MAT_WOOD for m in wall)
    else:
        assert wall == []
        assert emb.tilemap[14, 20] == MAT_STEEL      # the shared wall, uncut
    _emb, reasons = _run(_graph(kind))
    assert any(r.startswith("compartments: 1 not reachable") for r in reasons)


def test_a_bulkhead_does_not_excuse_a_secret_room():
    """PROPERTY: only a ``secret_path`` kind excuses an unreached secret
    room: behind a bulkhead (a cell boundary nothing crosses) it is refused.
    BREAKS IF the excuse ignores the edge kind."""
    _emb, reasons = _run(_graph("bulkhead", {"secret": True}))
    assert any(r.startswith("compartments: 1 not reachable") for r in reasons)


def test_unknown_kinds_and_tags_are_refused_by_name():
    """PROPERTY: a ShipGraph with an edge kind outside the table, a room tag
    key outside the schema, or a tag of the wrong shape is refused when
    BUILT, naming the offender. BREAKS IF the schema check is dropped, so a
    typo'd kind or tag would embed as a default."""
    with pytest.raises(G.GraphSchemaError, match="'hatch'"):
        _graph("hatch")
    with pytest.raises(G.GraphSchemaError, match="colour"):
        _graph("door", {"colour": 3})
    with pytest.raises(G.GraphSchemaError, match="'light'"):
        _graph("door", {"light": (1.0, 0.5, 0.2)})          # level missing
    with pytest.raises(G.GraphSchemaError, match="secret"):
        _graph("door", {"secret": "yes"})


def test_pending_kinds_are_accepted_then_refused_by_embed():
    """PROPERTY: a kind stage 2 cannot stamp yet (``locked``, blocked on
    #135's terminal) is part of the schema (a graph carrying one builds) but
    stage 2 refuses it naming what it waits for, rather than stamping a
    guess. BREAKS IF embed silently stamps something for a pending kind, or
    the schema refuses it."""
    pending = [k for k, spec in G.EDGE_KINDS.items() if spec.pending]
    assert "locked" in pending
    for kind in pending:
        g = _graph(kind)
        with pytest.raises(NotImplementedError, match="#135"):
            G.embed(g, RNG())


def test_a_light_tag_sets_its_rooms_lamp():
    """PROPERTY: a room's ``light`` tag ((r, g, b), level) becomes its lamp
    row's colour and intensity; untagged rooms keep the default lamp; the
    annotation tags ride Embedding.room_tags with the room's compartment.
    BREAKS IF embed ignores the tag or the annotations are dropped."""
    g = _graph("open", {"dept": "medbay", "role": "ward"},
               mid_tags={"light": ((0.2, 0.4, 1.0), 2.5)}, cell=2)
    emb, reasons = _run(g)
    assert reasons == []
    assert tuple(emb.lights[1].color) == (0.2, 0.4, 1.0)
    assert emb.lights[1].intensity == 2.5
    assert tuple(emb.lights[0].color) == G.LAMP_RGB
    assert emb.lights[0].intensity == G.LAMP_INTENSITY
    assert emb.room_tags[3]["dept"] == "medbay"
    assert emb.room_tags[3]["cell"] == 2
    assert emb.room_tags[3]["compartment"] == emb.room_tags[1]["compartment"]


def test_every_kind_the_schema_accepts_has_a_table_row():
    """PROPERTY: the schema accepts exactly the table's kinds, every row
    names an L9 strength, and every stamped (non-pending) kind embeds its
    row's material across its row's width (nothing for a kind that cuts
    nothing) -- the table and the code are one table. BREAKS IF a kind is
    accepted or stamped somewhere other than EDGE_KINDS."""
    assert {"door", "window", "open", "bulkhead", "vent_only", "breakable",
            "entity_door", "airlock", "locked"} <= set(G.EDGE_KINDS)
    for kind, spec in G.EDGE_KINDS.items():
        assert spec.l9 in G.L9_STRENGTHS, kind
        g = _graph(kind)
        if spec.pending:
            continue
        emb = G.embed(g, RNG())
        tiles = G._opening_tiles(g, g.edges[2])
        if spec.stamps is None:
            assert tiles == [], kind
        else:
            assert len(tiles) == spec.width, kind
            assert all(emb.tilemap[y, x] == spec.stamps for (y, x) in tiles)
    assert emb.edges_by_kind  # the per-kind count rides the embedding


def _generated_levels():
    """Every shipped level written by gen_ship, with the argv its header
    names."""
    out = []
    for toml in sorted((ROOT / "levels").glob("*/level.toml")):
        text = toml.read_text(encoding="utf-8")
        if "GENERATED by tools/gen_ship.py" not in text:
            continue
        m = re.search(r"tools/gen_ship\.py (--seed .*)$", text, re.M)
        assert m, toml
        out.append(pytest.param(toml.parent, m.group(1).split(),
                                id=toml.parent.name))
    return out


@pytest.mark.parametrize("folder, argv", _generated_levels())
def test_shipped_generated_levels_regenerate_byte_identical(folder, argv,
                                                            tmp_path):
    """PROPERTY: every shipped gen_ship level regenerates BYTE-IDENTICAL
    from the command its own header names: a stage-2 or gate change that
    leaves v1's door/window graphs alone moves no level. BREAKS IF a change
    alters what v1 stamps, draws, rejects or writes (then the level is
    regenerated and the change named, never silently)."""
    out = tmp_path / folder.name
    assert G.main(argv + ["--out", str(out)]) == 0
    committed = sorted(p.name for p in folder.iterdir() if p.is_file())
    assert set(G.DATA_FILES) <= set(committed)
    for name in committed:
        assert (out / name).read_bytes() == (folder / name).read_bytes(), name
