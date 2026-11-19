"""Properties of BROOD PLACEMENT on recipe ships (vocabulary Part G, arc #63
P5d; ``tools/ship_broods.py``, ``gen_ship._build_broods``): the department
``[brood]`` row, the archetype ``[infestation]`` weights drawn on the level
seed's own stream, the fairness rules, the record.

Each test names the property it protects and the change that must break it.
No ship, brood count or set of hosting departments is pinned: the numbers are
PLACEHOLDERS Erik tunes, so the tests read them from the recipes.
"""
from __future__ import annotations

import dataclasses
import math
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
import level_loader  # noqa: E402
import levelgen_recipes as R  # noqa: E402
import ship_broods as B  # noqa: E402
import ship_grammar as S  # noqa: E402
from simulation.materials import MAT_AIR, MAT_DOOR_CLOSED  # noqa: E402

SEEDS = range(8)


@pytest.fixture(scope="module")
def arch():
    return S.load_archetype("freighter")


@pytest.fixture(scope="module")
def clean_arch(arch):
    """The same archetype with no [infestation] table: every ship clean."""
    return dataclasses.replace(arch, infestation=())


@pytest.fixture(scope="module")
def plans(arch):
    return {(size, prof, seed): G.generate(seed, size, recipe=arch,
                                           profile=prof)
            for size in arch.sizes for prof in arch.profiles
            for seed in SEEDS}


@pytest.fixture(scope="module")
def clean_plans(clean_arch, plans):
    return {k: G.generate(k[2], k[0], recipe=clean_arch, profile=k[1])
            for k in plans}


def _seed_in_state(arch, state, start=0):
    return next(s for s in range(start, start + 10_000)
                if B.draw(arch, s).state == state)


def _brood_rooms(plan):
    return [rm for rm in plan.graph.rooms if "brood" in rm.tags]


def _broods(plan):
    return [e for e in plan.entities if e.class_name == "swarm_brood"]


# ---------------------------------------------------------------------------
# The stream: a clean ship is the ship it was
# ---------------------------------------------------------------------------

def test_the_brood_draw_moves_no_other_draw(plans, clean_plans):
    """PROPERTY (Part G.1): the brood draw has its own stream and touches the
    graph only where a brood goes -- every ship is the ship of the same
    archetype WITHOUT an [infestation] table (the pre-patch ship), tile for
    tile, but for its broods (appended entities + zones) and, infested, the
    opened corridor doors (door -> air under an open door entity). A clean
    ship is identical. BREAKS IF the brood draw takes a value from the
    grammar or layout generator, ``apply`` changes a graph with nothing to
    host, a brood changes a tile, or an opened door is not an existing open
    door row."""
    seen = set()
    for k, p in plans.items():
        c = clean_plans[k]
        assert (p.g, p.attempt) == (c.g, c.attempt), k
        seen.add(p.infestation)
        diff = np.argwhere(p.tilemap != c.tilemap)
        opened = B.opened_doors(c.graph, p.graph)
        if p.infestation != "infested":
            assert not opened and diff.size == 0, k
        door_tiles = {t for i in opened
                      for t in G._opening_tiles(p.graph, p.graph.edges[i])}
        for (y, x) in diff:
            assert (int(y), int(x)) in door_tiles, (k, y, x)
            assert c.tilemap[y, x] == MAT_DOOR_CLOSED
            assert p.tilemap[y, x] == MAT_AIR
        opened_ids = {f"{G.EDGE_DOOR_PREFIX}{i}" for i in opened}
        rest = [e for e in p.entities if e.class_name != "swarm_brood"]
        assert [(e.id, e.class_name, e.ordinal) for e in rest] == \
            [(e.id, e.class_name, e.ordinal) for e in c.entities], k
        for a, b in zip(rest, c.entities):
            fa = dict(a.fields)
            if a.id in opened_ids:
                assert fa.pop("initial_state") == "open"
                assert b.fields["initial_state"] == "closed"
                fa["initial_state"] = "closed"
            assert fa == b.fields, (k, a.id)
        assert p.embedding.wires == c.embedding.wires
        assert p.embedding.lights == c.embedding.lights
        if p.infestation == "clean":
            assert not _broods(p) and p.embedding.zones is None
    assert seen == set(S.INFESTATION_STATES), seen


def test_an_archetype_without_the_table_is_always_clean(clean_arch):
    """PROPERTY: no [infestation] table, no broods, no draw. BREAKS IF the
    default state changes or a missing table draws."""
    for s in range(50):
        inf = B.draw(clean_arch, s)
        assert inf.state == "clean" and inf.rolls == ()


# ---------------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------------

def _band(n, p, z=4.5):
    s = math.sqrt(max(p * (1 - p), 1e-12) / n)
    return p - z * s, p + z * s


def test_state_frequencies_follow_the_weights(arch):
    """PROPERTY (Part G.1): over many level seeds each state's frequency is
    within a 4.5-sigma binomial band of its normalised weight. BREAKS IF the
    weights are read in the wrong order, not normalised, or the draw is
    biased."""
    n = 4000
    counts = {s: 0 for s in S.INFESTATION_STATES}
    for s in range(n):
        counts[B.draw(arch, s).state] += 1
    for state, w in arch.infestation:
        lo, hi = _band(n, w)
        assert lo <= counts[state] / n <= hi, (state, counts, w)


def test_nest_rolls_follow_the_chances(arch):
    """PROPERTY (Part G.1): on a nest ship a department hosts at about its
    effective chance (plus the forced nest when every roll fails), and a
    ``with`` multiplier moves that frequency in its declared direction.
    BREAKS IF a roll ignores the chance, the multiplier is applied the other
    way, applied when that department is absent, or rolls shift with the
    departments present."""
    lib = arch.library
    names = B._brood_departments(arch)
    assert names, "the freighter carries [brood] rows"
    nests = [B.draw(arch, s) for s in range(20_000)]
    nests = [x for x in nests if x.state == "nest"]
    all_present = set(arch.departments())
    for n in names:
        for other, m in lib.departments[n].brood.with_:
            if m == 1.0 or other not in all_present:
                continue
            with_ = all_present
            without = all_present - {other}
            f_with = sum(n in B.hosts(arch, x, with_) for x in nests)
            f_without = sum(n in B.hosts(arch, x, without) for x in nests)
            if m > 1.0:
                assert f_with > f_without, (n, other, f_with, f_without)
            else:
                assert f_with < f_without, (n, other, f_with, f_without)
            # the roll itself sits at the effective chance (forced nests
            # only ADD hosts)
            p = B.effective_chance(arch, n, with_)
            rolled = sum(dict(x.rolls)[n] < p for x in nests) / len(nests)
            lo, hi = _band(len(nests), p)
            assert lo <= rolled <= hi
    # every nest ship has at least one host when a department can host
    for x in nests[:500]:
        assert B.hosts(arch, x, all_present)


def test_a_department_without_a_row_never_hosts(arch, plans):
    """PROPERTY: only departments with a [brood] row host, in every state.
    BREAKS IF the infested state hosts every department or the forced nest
    picks one without a row."""
    rows = set(B._brood_departments(arch))
    for p in plans.values():
        assert set(p.broods) <= rows
        for rm in _brood_rooms(p):
            assert rm.tags["dept"] in p.broods


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------

def test_broods_are_mirror_symmetric(plans):
    """PROPERTY (Part G.2): a wing brood is emitted in both wings with the
    same count, an axis brood once on the axis, and the zones paint a
    mirror-symmetric mask. BREAKS IF only one wing is tagged, the twin's
    count differs, or a zone is not the room's reflected floor."""
    for k, p in plans.items():
        rooms = {rm.id: rm for rm in _brood_rooms(p)}
        counts = {int(e.id[len(G.BROOD_PREFIX):]): e.fields["count"]
                  for e in _broods(p)}
        assert set(counts) == set(rooms), k
        for rid, rm in rooms.items():
            assert rm.mirror in rooms, (k, rid)
            assert counts[rm.mirror] == counts[rid], (k, rid)
        if rooms:
            z = p.embedding.zones > 0
            assert np.array_equal(z, z[:, ::-1]), k


def test_forbidden_rooms_never_host(plans):
    """PROPERTY (Part G.2): never a brood in a spawn slab, the passageway,
    the lobby, a main section, the zombies' room, a secret room or on a
    vestibule or spawn-box tile; always in a department room its row names.
    BREAKS IF the room choice ignores the row's ``rooms``, picks a secret,
    or a zone covers a carved or spawn tile."""
    lib = S.load_archetype("freighter").library
    for k, p in plans.items():
        carved = {t for ts in G._carved(p.graph).values() for t in ts}
        for rm in _brood_rooms(p):
            t = rm.tags
            assert t.get("dept") and not t.get("secret"), (k, rm.id)
            assert "spawn_team" not in t and not t.get("zombie"), (k, rm.id)
            assert t["role"] in R.brood_eligible(lib.departments[t["dept"]])
        if not _brood_rooms(p):
            continue
        z = p.embedding.zones
        for (y, x) in carved:
            assert z[y, x] == 0, (k, y, x)
        for r in p.spawn_rooms:
            assert not z[r.y:r.y + r.h, r.x:r.x + r.w].any(), (k, r.team)


def test_a_zone_never_covers_a_vestibule(plans):
    """PROPERTY (Part G.2): a brood zone leaves out the carved airlock
    vestibule of its room. No shipped [brood] row reaches a carved room (the
    vestibule is carved on a lobby or a promoted department's stub), so the
    input is hand-built: a promoted stub tagged directly. BREAKS IF
    ``_build_broods`` stops masking the carved tiles."""
    hits = 0
    tag = {"species": "larva", "density": 0.5, "max": 10}
    for p in plans.values():
        carved = G._carved(p.graph)
        rids = [rid for rid, ts in carved.items()
                if ts and p.graph.rooms[rid].tags.get("dept")]
        if not rids:
            continue
        rooms = [dataclasses.replace(rm, tags={**rm.tags, "brood": tag})
                 if rm.id in rids else rm for rm in p.graph.rooms]
        g = dataclasses.replace(p.graph, rooms=rooms)
        _ents, z = G._build_broods(p.tilemap, g, p.spawn_rooms)
        for rid in rids:
            for (y, x) in carved[rid]:
                assert z[y, x] == 0
                hits += 1
    assert hits, "no sampled ship carves a vestibule out of a department room"


@pytest.mark.parametrize("role", ["slab", "passageway", "lobby", "section",
                                  "secret"])
def test_stage_two_refuses_a_brood_on_a_forbidden_room(plans, role):
    """PROPERTY (Part G.2): the stage-2 builder refuses, by name, a brood tag
    on a skeleton, spawn, zombie or secret room -- whoever wrote it. BREAKS
    IF the refusal is dropped from ``_build_broods``."""
    p = next(p for p in plans.values()
             if any(rm.tags.get("role") == role for rm in p.graph.rooms))
    tag = {"species": "larva", "density": 0.5, "max": 10}
    rooms = [dataclasses.replace(rm, tags={**rm.tags, "brood": tag})
             if rm.tags.get("role") == role else rm for rm in p.graph.rooms]
    g = dataclasses.replace(p.graph, rooms=rooms)
    with pytest.raises(ValueError, match="a brood is never in"):
        G._build_broods(p.tilemap, g, p.spawn_rooms)


def test_brood_counts_follow_the_row(plans):
    """PROPERTY (Part G.1): a brood's count is ``density`` x the zone's free
    floor tiles, floored, within [1, max]. BREAKS IF the count ignores the
    cap, counts solid tiles, or the zone is not the room."""
    for k, p in plans.items():
        _m, _v, solid, _w = G._masks(p.tilemap)
        rooms = {rm.id: rm for rm in _brood_rooms(p)}
        for e in _broods(p):
            rm = rooms[int(e.id[len(G.BROOD_PREFIX):])]
            zone = p.embedding.zones == e.fields["zone_id"]
            free = int((zone & ~solid).sum())
            b = rm.tags["brood"]
            assert e.fields["count"] == min(b["max"], max(
                1, math.floor(b["density"] * free))), (k, e.id)
            x0, y0, x1, y1 = rm.box
            ys, xs = np.nonzero(zone)
            assert ys.min() >= y0 and ys.max() <= y1, (k, e.id)
            assert xs.min() >= x0 and xs.max() <= x1, (k, e.id)


# ---------------------------------------------------------------------------
# The infested ship's doors
# ---------------------------------------------------------------------------

def test_infested_ships_open_exactly_the_hosts_corridor_doors(plans,
                                                              clean_plans):
    """PROPERTY (Part G.1): on an infested ship every ``door`` edge between a
    hosting department's room and the passageway is an open entity door,
    and no other edge changed (airlocks, hull doors, vented doors and doors
    to the spawn slab untouched). BREAKS IF a door of a non-hosting
    department opens, a host's corridor door stays shut, or a non-corridor
    door changes."""
    n = 0
    for k, p in plans.items():
        if p.infestation != "infested":
            continue
        n += 1
        c = clean_plans[k]
        g = p.graph
        want = []
        for i, e in enumerate(c.graph.edges):
            ends = (g.rooms[e.a], g.rooms[e.b])
            if (e.kind == "door"
                    and any(r.tags.get("dept") in p.broods for r in ends)
                    and any(r.tags.get("role") == "passageway"
                            and not r.tags.get("dept") for r in ends)):
                want.append(i)
        assert want, k
        assert B.opened_doors(c.graph, g) == want, k
        for i in want:
            assert g.edges[i].kind == "entity_door"
    assert n, "the sample has no infested ship"


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def test_the_record_carries_the_state_and_regenerates_the_broods(
        arch, tmp_path):
    """PROPERTY (Part G.1 item 3): ``[levelgen]`` records the state and the
    hosting departments; the written level loads (its broods pass the
    spawn's own checks) and ``generate_from_spec`` gives the same ship with
    the same broods. BREAKS IF the record drops a key, the writer omits
    zones.npy, or the brood draw depends on anything outside the record."""
    from simulation.swarm_brood import brood_problems
    for state in ("nest", "infested"):
        seed = _seed_in_state(arch, state)
        p = G.generate(seed, "medium", recipe="freighter")
        assert p.infestation == state and p.broods
        out = G.write(p, tmp_path / p.name, version="test-build")
        lvl = level_loader.load(p.name, levels_dir=str(tmp_path))
        rec = lvl.levelgen
        assert rec.infestation == state and rec.broods == p.broods
        ents = [e for e in lvl.entities if e.class_name == "swarm_brood"]
        assert [(e.id, e.fields["count"]) for e in ents] == \
            [(e.id, e.fields["count"]) for e in _broods(p)]
        solid = np.asarray(G._masks(lvl.tilemap)[2])
        assert brood_problems(lvl.zone_grid, solid, lvl.entities) == []
        again = G.generate_from_spec(out)
        assert np.array_equal(again.tilemap, p.tilemap)
        assert np.array_equal(again.embedding.zones, p.embedding.zones)
        assert [(e.id, e.fields) for e in again.entities] == \
            [(e.id, e.fields) for e in p.entities]


def test_a_clean_ship_writes_no_zones(arch, tmp_path):
    """PROPERTY: a clean ship's folder carries no zones.npy, and a rewrite
    over a nest ship's folder removes it. BREAKS IF zones.npy leaves
    OPTIONAL_DATA_FILES or is written empty."""
    nest = G.generate(_seed_in_state(arch, "nest"), "medium",
                      recipe="freighter")
    clean = G.generate(_seed_in_state(arch, "clean"), "medium",
                       recipe="freighter")
    out = tmp_path / "ship"
    G.write(nest, out, version="t")
    assert (out / "zones.npy").exists()
    G.write(clean, out, version="t")
    assert not (out / "zones.npy").exists()


# ---------------------------------------------------------------------------
# The doors: refused by name
# ---------------------------------------------------------------------------

def _dept_copy(tmp_path, name, extra: str):
    import shutil
    root = tmp_path / "recipes"
    shutil.copytree(R.RECIPES_DIR, root)
    p = root / "departments" / f"{name}.toml"
    text = p.read_text(encoding="utf-8")
    head, _sep, _tail = text.partition("\n[brood]\n")
    p.write_bytes((head + "\n[brood]\n" + extra).encode())
    return p, root


@pytest.mark.parametrize("extra, why", [
    ("chance = 0.5\nnests = 3\n", "unknown field"),
    ("chance = 1.5\n", "brood.chance"),
    ('chance = 0.5\nrooms = ["stub"]\n', "brood.rooms[0]"),
    ('chance = 0.5\nrooms = ["galley_x"]\n', "brood.rooms[0]"),
    ("chance = 0.5\ndensity = 0\n", "brood.density"),
    ("chance = 0.5\nmax = 0\n", "brood.max"),
    ("chance = 0.5\nmax = 100000000\n", "brood.max"),
    ("chance = 0.5\nwith = { spaceport = 2.0 }\n", "brood.with.spaceport"),
    ("chance = 0.5\nwith = { medbay = -1 }\n", "brood.with.medbay"),
])
def test_the_department_door_refuses_a_bad_brood_row(tmp_path, extra, why):
    """PROPERTY (Part G.1): the [brood] row is refused by name -- an unknown
    key, a chance outside [0, 1], a room that is the entry or not the
    department's, a density <= 0, a cap outside [1, max_units], a ``with``
    key that is no department, a negative multiplier. BREAKS IF any check
    leaves ``levelgen_recipes._brood``."""
    p, root = _dept_copy(tmp_path, "crew_quarters", extra)
    with pytest.raises(R.RecipeError, match=__import__("re").escape(why)):
        R.load_department(p, root=root)


@pytest.mark.parametrize("table, why", [
    ("[infestation]\nclean = 1\nplague = 1\n", "unknown field"),
    ("[infestation]\nclean = -1\nnest = 1\n", "infestation.clean"),
    ("[infestation]\nclean = 0\n", "every weight is 0"),
])
def test_the_archetype_door_refuses_a_bad_infestation(tmp_path, table, why):
    """PROPERTY (Part G.1): the [infestation] table is refused by name: an
    unknown state, a negative weight, all weights zero. BREAKS IF a check
    leaves ``ship_grammar._infestation``."""
    import re
    import shutil
    root = tmp_path / "recipes"
    shutil.copytree(R.RECIPES_DIR, root)
    p = root / "archetypes" / "freighter.toml"
    text = p.read_text(encoding="utf-8")
    start = text.index("[infestation]")
    end = text.index("\n\n", start)
    p.write_bytes((text[:start] + table + text[end:]).encode())
    with pytest.raises(R.RecipeError, match=re.escape(why)):
        S.load_archetype(p)
