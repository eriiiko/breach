"""Properties of the recipe library loader and the resolver
(``tools/levelgen_recipes.py``, #136, #152;
docs/levelgen_cycle_types_and_recipes_2026-10-31.md format revision 2.1, the
layout doc's §13 and the archetype doc's §7.1).

Each test names the property it protects and the change that must break it.
Nothing here pins how many recipes, rooms or presets exist: the library is
designed to grow. Refusals are exercised on a COPY of the shipped recipes
with one line changed, so each test proves one rule.
"""
from __future__ import annotations

import re
import shutil
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

import gen_ship as G  # noqa: E402
import levelgen_recipes as R  # noqa: E402
from config import CFG  # noqa: E402

SEEDS = range(50)
CQ = "departments/crew_quarters.toml"
ENG = "departments/engineering.toml"
MED = "departments/medbay.toml"
CARGO = "departments/cargo_hold.toml"
# The layout doc's row depths (§6, §13 A6): a medium row is 24-27 deep, a
# large one 44-47.
MEDIUM_ROW_MIN, LARGE_ROW_MAX = 24, 47


def _rng(seed: int):
    return np.random.Generator(np.random.PCG64(seed))


def _copy(tmp_path: Path, rel: str | None = None, old: str | None = None,
          new: str | None = None) -> Path:
    """The shipped recipe tree copied to ``tmp_path``, with ``old`` replaced
    by ``new`` (exactly once) in ``rel``."""
    root = tmp_path / "recipes"
    shutil.copytree(R.RECIPES_DIR, root)
    if rel is not None:
        p = root / rel
        text = p.read_text(encoding="utf-8")
        assert text.count(old) == 1, (rel, old)
        p.write_text(text.replace(old, new), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# The shipped library
# ---------------------------------------------------------------------------

def test_the_shipped_recipes_load():
    """PROPERTY: every shipped recipe passes its door -- the six cycle types
    of §1.3 (the flank pair and the retreat loop as ONE-leg `second_leg`
    types, rev 2.1), every department and the fixtures (§3.3) -- and in each
    department the entry is required and attaches to nothing, every other
    room attaches to something (§3.2), every declared duct is used and every
    fixture exists. BREAKS IF a shipped file drifts from the door (a new key,
    a renamed preset, a config row removed) or the door stops reading a
    field."""
    lib = R.load_recipes()
    assert {"retreat_loop", "flank_pair", "airlock_bypass",
            "maintenance_ring", "duct_link", "pocket"} <= set(lib.cycles)
    for name in ("flank_pair", "retreat_loop"):
        cyc = lib.cycles[name]
        assert cyc.move == R.SECOND_LEG and len(cyc.legs) == 1
    assert "crew_quarters" in lib.departments
    for name, dept in lib.departments.items():
        entry = dept.room(dept.entry)
        assert entry.required and entry.attach is None, name
        assert all(r.attach for r in dept.rooms if r.id != dept.entry), name
        assert {v.duct for r in dept.rooms for v in r.vents} == \
            {d.name for d in dept.ducts}, name
        for r in dept.rooms:
            if r.fixture:
                assert r.fixture in lib.fixtures, (name, r.id)
    assert lib.cycles["maintenance_ring"].rooms[0] % 2 == 0


def test_wing_departments_are_depth_elastic():
    """PROPERTY (layout doc §13 A6): every shipped wing department with an
    end room spans the row depths -- stub ``along`` + end room ``h`` + 1
    reaches a medium row's 24 at its minima and a large row's 47 at its
    maxima. BREAKS IF a department's ranges are narrowed so it can no longer
    pack in a medium row or fill a large one (the layout patch retunes them
    against its own packing twin, which then replaces this bound)."""
    lib = R.load_recipes()
    wings = [d for d in lib.departments.values() if d.placement == "wing"]
    assert wings
    for dept in wings:
        ends = [r for r in dept.rooms if r.attach == "end"]
        if not ends:
            continue
        stub, end = dept.room(dept.entry), ends[0]
        assert stub.size.b[0] + end.size.b[0] + 1 <= MEDIUM_ROW_MIN, dept.name
        assert stub.size.b[1] + end.size.b[1] + 1 >= LARGE_ROW_MAX, dept.name


def test_the_loader_imports_nothing_heavy():
    """PROPERTY: importing the loader pulls in no numpy, gen_ship, config or
    renderer (it is stage-1 tooling; the door imports its vocabulary lazily).
    BREAKS IF a module-level import of gen_ship / config / numpy is added."""
    code = ("import sys; sys.path.insert(0, 'tools'); import levelgen_recipes;"
            "bad = [m for m in ('numpy', 'gen_ship', 'config', 'pyray') "
            "if m in sys.modules]; print('BAD', bad)")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "BAD []" in out.stdout, out.stdout


# ---------------------------------------------------------------------------
# The door: each refusal names the file and the field
# ---------------------------------------------------------------------------

REFUSALS = [
    # (id, file, old, new, the field named, the reason named)
    ("unknown_field", CQ, 'toward = "spawn"', 'toward = "spawn"\ncolour = 1',
     "[department]", "unknown field"),
    ("physics_number", CQ, 'size = { w = [9, 30], h = [9, 20] }\nfuel = '
     '"furniture"', 'size = { w = [9, 30], h = [9, 20] }\nfuel = 3',
     "room[3].fuel", "physics number"),
    ("missing_filter_row", CQ, 'name = "galley"\nfilter = "derelict"',
     'name = "galley"\nfilter = "nosuch"', "duct[1].filter",
     r"no \[filters.nosuch\] row"),
    ("missing_light_preset", CQ, 'light = "dim"', 'light = "gloomy"',
     "room[1].light", r"\[levelgen.lights.gloomy\]"),
    ("missing_heat_source_preset", "fixtures.toml", 'heat_source = "stove"',
     'heat_source = "furnace"', "[galley].heat_source",
     r"\[heat_sources.furnace\]"),
    ("missing_hazard_preset", "fixtures.toml", 'hazard = "magazine_cookoff"',
     'hazard = "boom"', "[magazine].hazard", r"\[hazards.boom\]"),
    ("missing_pressure_preset", CQ, 'pressure = "ambient"',
     'pressure = "high"', "physics.pressure", r"\[pressures"),
    ("edge_kind", CQ, 'b = "laundry"\nkind = "door"',
     'b = "laundry"\nkind = "portal"', "edge[4].kind", "EDGE_KINDS"),
    ("width_class", CQ, 'width = "wide"', 'width = "medium"',
     "room[0].width", "not one of"),
    ("narrow_is_one_room", "cycles/flank_pair.toml",
     "rooms = [1, 1]                  # inclusive", "rooms = [1, 2]  #",
     "leg[0].width", "exactly one room"),
    ("sealed_room_lone_role", CQ,
     'roles = ["return", "supply"] }   # a sealed box',
     'roles = ["return"] }   # a sealed box', "room 'heads'.vent", "sealed"),
    ("count_low", CQ, "count = [2, 4]", "count = [0, 4]", "room[1].count",
     "out of range"),
    ("count_reversed", CQ, "count = [2, 4]", "count = [4, 2]",
     "room[1].count", "out of range"),
    ("p_high", CQ, "p = 0.7", "p = 1.5", "room[4].p", "probability"),
    ("multiplicity_twice", CQ, "p = 0.3", "p = 0.3\nrequired = true",
     "room[5]", "exactly one"),
    ("unknown_cycle_type", CQ, "[[edge]]\na = \"stub\"\nb = \"mess\"",
     "[[cycle]]\ntype = \"spiral\"\na = \"stub\"\nb = \"mess\"\n\n[[edge]]\n"
     "a = \"stub\"\nb = \"mess\"", "cycle[0].type", "no cycle type"),
    ("cycle_phase_not_1", CQ, "[[edge]]\na = \"stub\"\nb = \"mess\"",
     "[[cycle]]\ntype = \"retreat_loop\"\na = \"stub\"\nb = \"mess\"\n\n"
     "[[edge]]\na = \"stub\"\nb = \"mess\"", "cycle[0].type", "phase"),
    ("unknown_fixture", CQ, 'fixture = "galley"', 'fixture = "kitchen"',
     "room[4].fixture", "no fixture"),
    ("face_move_unused", "cycles/retreat_loop.toml",
     'move = "second_leg"', 'move = "face"', "cycle.move",
     "unused in v1"),
    ("ring_is_even", "cycles/maintenance_ring.toml", "rooms = [4, 8]",
     "rooms = [4, 7]", "cycle.rooms", "EVEN"),
    ("non_walk_leg_passes_gas", "cycles/airlock_bypass.toml",
     'edges = ["breakable"]', 'edges = ["door"]', "leg[1].edges",
     "can pass gas"),
    ("vent_only_bypass_waits_for_49", "cycles/airlock_bypass.toml",
     'edges = ["breakable"]', 'edges = ["vent_only"]', "leg[1].edges", "#49"),
    ("crosses_cell_needs_airlock", "cycles/airlock_bypass.toml",
     'edges = ["airlock"]', 'edges = ["door"]', "cycle.crosses_cell",
     "airlock pairs"),
    ("contact_triangle", CQ, '[[duct]]\nname = "quarters"',
     '[[edge]]\na = "bunk"\nb = "mess"\nkind = "open"\n\n[[duct]]\n'
     'name = "quarters"', "edge", "3-cycle of contacts"),
    # the layout's loader twin: at its MINIMA the department cannot pack
    ("feasibility", CQ, "count = [2, 4]", "count = [12, 12]",
     "department", "cannot be laid out.*side rooms at minimum count"),
    ("narrow_never_with_fuel", CQ,
     'width = "wide"\nsize = { across = [6, 8], along = [10, 40] }',
     'width = "narrow"\nfuel = "furniture"\n'
     'size = { across = [3, 3], along = [10, 40] }', "room[0].fuel",
     "never carries fuel"),
    ("unknown_wish", CQ, 'wish_near = ["medbay"]', 'wish_near = ["spa"]',
     "department.wish_near[0]", "no department"),
    ("entry_must_exist", CQ, 'entry = "stub"', 'entry = "lobby"',
     "department.entry", "no room"),
    # second_leg (rev 2.1): one walkable leg of one room
    ("second_leg_one_room", "cycles/retreat_loop.toml",
     'rooms = [1, 1]\nwidth = "wide"', 'rooms = [1, 2]\nwidth = "wide"',
     ": leg:", "exactly ONE"),
    # attach / exit / hull_door (§3.2; layout §4, §13 A2; archetype §7.1)
    ("second_end", CQ, "p = 0.3", 'p = 0.3\nattach = "end"',
     "room 'laundry'.attach", "at most one"),
    ("fore_in_a_wing", CQ, "p = 0.3", 'p = 0.3\nattach = "fore"',
     "room 'laundry'.attach", "axis"),
    ("end_in_an_axis", ENG, 'attach = "aft"', 'attach = "end"',
     "room 'fuel_bunker'.attach", "wing department"),
    ("second_aft", ENG, '[[edge]]\na = "engine_room"',
     '[[room]]\nid = "pump_room"\nrequired = true\nattach = "aft"\n'
     'size = { w = [12, 22], h = [6, 10] }\n\n[[edge]]\na = "engine_room"\n'
     'b = "pump_room"\nkind = "door"\n\n[[edge]]\na = "engine_room"',
     "room 'pump_room'.attach", "at most one"),
    ("behind_unknown_parent", CQ, 'attach = "behind:mess"',
     'attach = "behind:pantry"', "room 'galley'.attach", "parent"),
    ("behind_a_counted_room", CQ, 'attach = "behind:mess"',
     'attach = "behind:bunk"', "room 'galley'.attach", "counted"),
    ("count_off_a_side_room", CQ, "p = 0.7", "count = [1, 2]",
     "room 'galley'.count", "side room"),
    ("exit_twice", CQ, "p = 0.3", "p = 0.3\nexit = true",
     "room 'laundry'.exit", "at most one"),
    ("exit_off_a_side_room", CARGO, "hull_door = true",
     "hull_door = true\nexit = true", "room 'main_hold'.exit", "side room"),
    ("hull_door_off_the_end", CQ, "p = 0.3", "p = 0.3\nhull_door = true",
     "room 'laundry'.hull_door", "end"),
    ("edge_not_to_its_anchor", CQ, 'a = "stub"\nb = "laundry"',
     'a = "heads"\nb = "laundry"', "edge[4]", "only its stub"),
    ("wing_entry_is_a_corridor", MED,
     'shape = "corridor"\nrequired = true\nwidth = "wide"\n'
     'size = { across = [6, 8], along = [10, 40] }',
     'required = true\nsize = { w = [6, 8], h = [10, 40] }',
     "room 'stub'.shape", "corridor"),
    ("secret_behind_the_exit", CQ, 'host = "bunk"', 'host = "heads"',
     "secret.host", "exit room"),
]


@pytest.mark.parametrize("rel, old, new, where, why",
                         [pytest.param(*r[1:], id=r[0]) for r in REFUSALS])
def test_the_door_refuses_by_file_and_field(tmp_path, rel, old, new, where,
                                            why):
    """PROPERTY: each rule of the door refuses its violation at load, and
    the refusal names the FILE and the FIELD (and why) -- for an unknown
    field, a physics number where a preset name belongs, a config row /
    preset config.toml lacks, an edge kind outside EDGE_KINDS, a width class,
    the narrow one-room rule, the sealed-room vent rule, count / p ranges and
    multiplicity, an unknown or wrong-phase cycle type, an unknown fixture,
    the unused face move, the even ring, a gas-blind non-walk leg, the #49
    wait, L5 on crosses_cell, a contact triangle, the layout's packing twin
    at the recipe's minima (a department whose minimum rooms cannot be dealt
    along its stub is refused, naming the layout's reason),
    fuel in a narrow corridor, an unknown wish, a missing entry, a
    `second_leg` type with other than one one-room leg, and every attach /
    exit / hull_door rule the layout's primitives need (one end, one fore,
    one aft; fore / aft only on an axis department and side / end / behind
    only on a wing one; a behind parent that exists, is uncounted and is not
    itself behind; counts on side rooms only; one exit, on a side room, never
    the secret's host; a hull door only on the end room; a declared edge only
    between a room and what it attaches to; a wing entry that is a
    corridor). BREAKS IF a rule is dropped, or a refusal stops naming where
    it came from."""
    root = _copy(tmp_path, rel, old, new)
    with pytest.raises(R.RecipeError) as ei:
        R.load_recipes(root)
    msg = str(ei.value)
    assert Path(rel).name in msg, msg
    assert where in msg, msg
    assert re.search(why, msg), msg


def test_the_galley_fuse_is_refused_when_the_stove_cannot_light(
        tmp_path, monkeypatch):
    """PROPERTY (Erik's FUSE ruling, §4): a fixture whose heat source caps
    at or below its fuel's ignition_temp is refused by name, pointing at the
    preset to raise; the shipped stove passes. BREAKS IF the fuse check is
    dropped or compares against the wrong material."""
    R.load_fixtures(R.RECIPES_DIR / "fixtures.toml")
    ign = float(CFG.materials.kindling.ignition_temp)
    monkeypatch.setattr(CFG.heat_sources.stove, "max_temp", ign - 1.0)
    with pytest.raises(R.RecipeError, match=r"\[galley\].heat_source.*FUSE"):
        R.load_fixtures(R.RECIPES_DIR / "fixtures.toml")


def test_an_unknown_department_is_refused_by_name():
    """PROPERTY: asking the library for a department it has no recipe for is
    refused naming it and the known ones. BREAKS IF a missing recipe falls
    through to a bare FileNotFoundError or an empty department."""
    with pytest.raises(R.RecipeError, match=r"'nosuch'.*crew_quarters"):
        R.department("nosuch")


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lib():
    return R.load_recipes()


def test_resolved_tags_are_the_presets_values(lib):
    """PROPERTY: the resolver turns each name into its preset's VALUES --
    the galley's heat source is [heat_sources.stove]'s four fields, a
    `dim` room's light is [levelgen.lights.dim], every room of a `main`
    department rides cell 0 and carries its department and recipe id --
    and each tag passes gen_ship's own ROOM_TAGS check. BREAKS IF a value is
    hard-coded in the resolver, a preset is read from the wrong table, or a
    tag shape drifts from ROOM_TAGS."""
    cq = lib.departments["crew_quarters"]
    rd = R.resolve_department(cq, lib.fixtures, _rng(0))
    stove = vars(CFG.heat_sources.stove)
    assert rd.rooms["galley"].tags["heat_source"] == {
        k: stove[k] for k in ("heat_per_s", "radius", "max_temp",
                              "death_payload")}
    dim = CFG.levelgen.lights.dim
    assert rd.rooms["bunk"].tags["light"] == (tuple(dim.rgb), dim.level)
    for rid, rr in rd.rooms.items():
        assert rr.tags["dept"] == "crew_quarters" and rr.tags["role"] == rid
        assert rr.tags["cell"] == R.MAIN_CELL
        for k, v in rr.tags.items():
            assert G.ROOM_TAGS[k][1](v), (rid, k, v)
    assert rd.ducts == {d.name: {"filter": d.filter} for d in cq.ducts}
    # A recipe flag resolves to gen_ship's own value, a department preset to
    # every room: the cargo hold's door to space is HULL_DOOR_DEFAULT, every
    # engineering room starts [temps.hot_engineering] warm.
    for name, dept in lib.departments.items():
        rd = R.resolve_department(dept, lib.fixtures, _rng(0),
                                  cell=None if dept.cell == "main" else 1)
        for r in dept.rooms:
            tags = rd.rooms[r.id].tags
            assert (tags.get("hull_door") == G.HULL_DOOR_DEFAULT) \
                == r.hull_door, (name, r.id)
            if dept.physics.temp != R.AMBIENT:
                assert tags["temp"] == getattr(
                    CFG.temps, dept.physics.temp).above_ambient


def test_a_key_without_its_mechanism_is_pending_never_dropped(tmp_path):
    """PROPERTY (§3.2 last paragraph): a recipe key whose stage-2 mechanism
    is not built (here a disabled heat source: the heat_source tag has no
    `enabled` yet, §5.2) is kept as PENDING with the resolved value and the
    extension it waits for, and ``stamp_tags`` refuses it by name; rooms
    with nothing pending stamp. BREAKS IF the resolver silently drops the
    key, writes it as an enabled heat source, or lets stage 2 stamp a room
    with a pending key."""
    root = _copy(tmp_path, "fixtures.toml",
                 "heat_source_enabled = true      # Erik",
                 "heat_source_enabled = false     # Erik")
    lib = R.load_recipes(root)
    rd = R.resolve_department(lib.departments["crew_quarters"],
                              lib.fixtures, _rng(0))
    galley = rd.rooms["galley"]
    assert "heat_source" not in galley.tags
    value, ext = galley.pending["heat_source"]
    assert value["enabled"] is False
    assert value["max_temp"] == CFG.heat_sources.stove.max_temp
    assert "§5.2" in ext
    with pytest.raises(R.PendingMechanism,
                       match=r"'galley'.*'heat_source'"):
        galley.stamp_tags()
    assert rd.rooms["heads"].stamp_tags() == rd.rooms["heads"].tags


def test_fuel_classes_and_vent_lists_resolve_to_gen_ships_forms(lib,
                                                               tmp_path):
    """PROPERTY (#133's stage-2 extensions, landed): a fuel class resolves to
    gen_ship's tag -- the int form exactly when its mix is gen_ship's v1 mix
    (so nothing v1 stamped moves), else ``{"clusters", "mix"}`` with the
    class's mix, which ``fuel_spec`` reads -- and a room on several ducts
    resolves to the LIST form of ``vent``, which ``vent_specs`` reads; neither
    is pending. BREAKS IF the resolver marks either pending again, writes the
    int form for another mix, or emits a shape gen_ship's readers refuse."""
    for dept in lib.departments.values():
        rd = R.resolve_department(dept, lib.fixtures, _rng(3),
                                  cell=None if dept.cell == "main" else 1)
        for r in dept.rooms:
            rr = rd.rooms[r.id]
            assert "fuel" not in rr.pending and "vent" not in rr.pending
            cls = r.fuel or (lib.fixtures[r.fixture].fuel if r.fixture
                             else None)
            if cls is None:
                assert "fuel" not in rr.tags
                continue
            mix = vars(getattr(CFG.levelgen.fuel, cls).mix)
            tag = rr.tags["fuel"]
            if mix == G.FUEL_MIX_V1:
                assert isinstance(tag, int)
            else:
                assert tag["mix"] == mix
            names = list(G._table().names)
            assert {names[m] for m, _f in G.fuel_spec(tag)[1]} == set(mix)
    root = _copy(tmp_path, CQ,
                 'vent = { duct = "quarters", roles = ["return", "supply"] }',
                 'vent = [{ duct = "quarters", roles = ["return", "supply"] '
                 '}, { duct = "galley", roles = ["supply", "return"] }]')
    two = R.load_recipes(root)
    rd = R.resolve_department(two.departments["crew_quarters"], two.fixtures,
                              _rng(0))
    heads = rd.rooms["heads"]
    assert isinstance(heads.tags["vent"], list) and not heads.pending
    assert [d for d, _r, _s in G.vent_specs(heads.tags["vent"])] == \
        ["quarters", "galley"]


def test_the_resolved_department_passes_the_ship_graph_schema(lib):
    """PROPERTY: for EVERY shipped department, over 50 seeds, one instance
    of every resolved room, joined by the recipe's edges (the entry's door
    to a stand-in spine room in the department's cell), builds a gen_ship
    ShipGraph -- tags (fuel classes, vent lists, hull doors included), the
    sealed-room rule per duct, ducts in one cell, L5 -- with the resolver's
    ducts. BREAKS IF the resolver writes a tag stage 2 would refuse or the
    loader's sealed-room rule drifts from ShipGraph's."""
    for dept in lib.departments.values():
        cell = R.MAIN_CELL if dept.cell == "main" else 1
        for seed in SEEDS:
            rd = R.resolve_department(dept, lib.fixtures, _rng(seed),
                                      cell=None if dept.cell == "main"
                                      else cell)
            ids = [r.id for r in dept.rooms]
            # each room's top wall on the hull, so a hull_door has its site
            rooms = [G.Room(i, 2 + 10 * i, 1, 6, 6, "room",
                            {k: v for k, v in rd.rooms[rid].tags.items()})
                     for i, rid in enumerate(ids)]
            spine = len(rooms)
            rooms.append(G.Room(spine, 0, 20, 6, 6, "room", {"cell": cell}))
            edges = [G.Edge(ids.index(e.a), ids.index(e.b), e.kind)
                     for e in dept.edges]
            edges.append(G.Edge(spine, ids.index(dept.entry), "door"))
            G.ShipGraph(width=200, height=40, hull=(0, 0, 199, 39),
                        rooms=rooms, edges=edges, mirror_x=False,
                        ducts=dict(rd.ducts))


def test_resolution_is_deterministic_and_tracks_the_presets(lib, monkeypatch):
    """PROPERTY: the same seed resolves to the same tags and digest; a
    fuel class's cluster count stays inside its preset's range over 50
    seeds and varies across them; retuning a preset changes the digest
    (§3.4: the (recipe hash, seed) pair hashes the RESOLVED values). BREAKS
    IF a draw escapes the passed generator (a global RNG), the count range is
    ignored, or the digest covers only the recipe text."""
    cq = lib.departments["crew_quarters"]
    a = R.resolve_department(cq, lib.fixtures, _rng(7))
    b = R.resolve_department(cq, lib.fixtures, _rng(7))
    assert a == b
    lo, hi = CFG.levelgen.fuel.furniture.clusters
    counts = set()
    for seed in SEEDS:
        rd = R.resolve_department(cq, lib.fixtures, _rng(seed))
        n = rd.rooms["bunk"].tags["fuel"]
        assert lo <= n <= hi
        counts.add(n)
    assert len(counts) > 1 or lo == hi
    monkeypatch.setattr(CFG.heat_sources.stove, "heat_per_s",
                        CFG.heat_sources.stove.heat_per_s * 2)
    c = R.resolve_department(cq, lib.fixtures, _rng(7))
    assert c.digest != a.digest


def test_an_own_cell_needs_the_grammars_number(tmp_path):
    """PROPERTY: an `own`-cell department resolves only with the cell number
    the grammar hands it (never MAIN_CELL), and then every room carries it.
    BREAKS IF the resolver invents a cell number or lets an own department
    ride the main cell."""
    root = _copy(tmp_path, CQ, 'cell = "main"', 'cell = "own"')
    lib = R.load_recipes(root)
    cq = lib.departments["crew_quarters"]
    with pytest.raises(R.RecipeError, match="department.cell"):
        R.resolve_department(cq, lib.fixtures, _rng(0))
    rd = R.resolve_department(cq, lib.fixtures, _rng(0), cell=2)
    assert {rr.tags["cell"] for rr in rd.rooms.values()} == {2}
