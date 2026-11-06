"""Properties of the ship grammar (``tools/ship_grammar.py`` + the skeleton's
graph side in ``tools/ship_layout.py``, #136;
docs/levelgen_archetype_recipe_2026-10-31.md with its §7.1 amendments,
docs/levelgen_layout_stage_2026-10-31.md §1, §2, §4, §7, §13).

Each test names the property it protects and the change that must break it.
Nothing here pins how many rooms, edges, cycles or departments a ship has:
the grammar and its recipes are designed to grow. Ships are drawn over many
seeds x both sizes x every profile; refusals are exercised on a COPY of the
shipped recipes with one line changed, so each row proves one rule.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_ship as G  # noqa: E402
import levelgen_recipes as R  # noqa: E402
import ship_grammar as S  # noqa: E402
import ship_layout as L  # noqa: E402
from config import CFG  # noqa: E402

SEEDS = range(30)
ARCH = "archetypes/freighter.toml"
P = ("test", 0)


@pytest.fixture(scope="module")
def arch():
    return S.load_archetype("freighter")


@pytest.fixture(scope="module")
def ships(arch):
    """One graph per (size, profile, seed); the zombies argument varies with
    the seed so the zombie-less case is drawn too."""
    return {(size, prof, seed): S.plan_quotient(arch, size, prof, seed,
                                                zombies=(0, 3, 6)[seed % 3])
            for size in arch.sizes for prof in arch.profiles for seed in SEEDS}


def _copy(tmp_path: Path, rel: str | None = None, old: str | None = None,
          new: str | None = None) -> Path:
    """The shipped recipe tree copied to ``tmp_path`` with ``old`` replaced
    by ``new`` (exactly once) in ``rel``; returns the archetype's path."""
    root = tmp_path / "recipes"
    shutil.copytree(R.RECIPES_DIR, root)
    if rel is not None:
        p = root / rel
        text = p.read_bytes().decode()
        assert text.count(old) == 1, (rel, old)
        p.write_bytes(text.replace(old, new).encode())
    return root / ARCH


def _block(q) -> list:
    return [x for x in q.band if q.room(x).role != L.SECTION]


# ---------------------------------------------------------------------------
# The archetype door
# ---------------------------------------------------------------------------

def test_the_freighter_loads_with_its_amendments(arch):
    """PROPERTY: the shipped freighter passes its door with the §7.1
    amendments (the lobby at least the airlock's 9 tiles, A1; the annex size
    a range of wide rooms), on sizes gen_ship knows; every (size, profile)
    has an effective range for every [[apply]] entry with lo <= hi, and a
    band whose section count holds the hub department's block plus one main
    section; a type the file keeps at [0, 0] on a size stays [0, 0] under
    every profile (no ring on medium). BREAKS IF the file drifts from the
    door, the lobby amendment is undone, or a profile shift starts raising
    a type onto a size the file excludes."""
    assert set(arch.sizes) <= set(G.SIZES)
    assert arch.lobby_x[0] >= L.LOBBY_X_MIN
    assert arch.secret_size[0] >= R.MIN_WIDE
    hub = arch.library.departments[arch.hub_department]
    block = 1 + sum(r.required for r in hub.rooms if r.attach in R.ATTACH_AXIS)
    for size in arch.sizes:
        lo, hi = arch.sections[size]
        assert block + 1 <= lo <= hi <= G.SIZES[size].sections[1]
        for prof in arch.profiles:
            rng = arch.ranges[(size, prof)]
            assert set(rng) == {e.index for e in arch.applies}
            for e in arch.applies:
                elo, ehi = rng[e.index]
                assert 0 <= elo <= ehi
                if e.count[size] == (0, 0):
                    assert (elo, ehi) == (0, 0), (size, prof, e.rule)


def test_the_profiles_apply_their_named_operators(arch):
    """PROPERTY (archetype doc §1.1, §4): ``cycles_shift`` moves BOTH ends of
    every phase-2 / phase-4 cycle range the file allows on a size (floored
    at 0; an uncapped class by exactly the shift, a capped one never above
    its cap) and leaves the departments' counts alone; ``secrets`` replaces
    the budget; ``zombies_scale`` rounds half up and keeps at least one
    zombie. BREAKS IF an operator is ignored, applied to the wrong entries,
    or the rounding changes."""
    skel = L.SKELETONS[arch.skeleton]
    hub = arch.library.departments[arch.hub_department]
    for size in arch.sizes:
        for prof, p in arch.profiles.items():
            got = arch.ranges[(size, prof)]
            for e in arch.applies:
                lo, hi = e.count[size]
                if e.department or hi == 0:
                    assert got[e.index] == (lo, hi)
                    continue
                cls = skel.anchor(e.on, hub.entry)
                want_hi = max(0, hi + p.cycles_shift)
                if cls.cap == "slack":
                    assert got[e.index] == (max(0, lo + p.cycles_shift),
                                            want_hi), (size, prof, e.rule)
                else:
                    assert got[e.index][1] == min(want_hi, S._cap(arch, cls,
                                                                  size))
            assert arch.secrets(prof) == (arch.secret_budget
                                          if p.secrets is None else p.secrets)
            for base in range(0, 9):
                z, exact = arch.scaled_zombies(prof, base), \
                    base * p.zombies_scale
                assert (z == 0) == (base == 0)
                if base:
                    assert z >= 1 and (z == 1 or -0.5 < z - exact <= 0.5)
    assert any(p.cycles_shift for p in arch.profiles.values())


ROWS = [
    ('name = "freighter"', 'name = "tanker"', r"archetype\.name"),
    ('zombies = "axis_section"', 'zombies = "axis_section"\nfoo = 1',
     r"\[archetype\].*unknown field"),
    ('skeleton = "freighter"', 'skeleton = "station"', r"archetype\.skeleton"),
    ('sizes = ["medium", "large"]', 'sizes = ["medium", "huge"]',
     r"archetype\.sizes\[1\]"),
    ('lobby = { x = [9, 11] }', 'lobby = { x = [8, 10] }',
     r"archetype\.lobby\.x"),
    ('band = { sections = [3, 5] }', 'band = { sections = [1, 2] }',
     r"archetype\.band\.sections"),
    ('spawn = { exits = [1, 3] }', 'spawn = { exits = [1, 4] }',
     r"archetype\.spawn\.exits"),
    ('zombies = "axis_section"', 'zombies = "lobby"', r"archetype\.zombies"),
    ('budget = 1,', 'budget = -1,', r"archetype\.secrets\.budget"),
    ('size = [6, 8] }', 'size = [4, 8] }', r"archetype\.secrets\.size"),
    ('allow_cross_cell_leak = false', 'allow_cross_cell_leak = true',
     r"allow_cross_cell_leak.*#49"),
    ('promote = []', 'promote = ["engineering"]',
     r"cells\.promote\[0\].*redundant"),
    ('promote = []', 'promote = ["galley"]', r"cells\.promote\[0\]"),
    ('department = "crew_quarters"\nplacement = "wing"',
     'department = "crew_quarters"\nplacement = "axis"',
     r"axiom\[1\]\.placement"),
    ('[[axiom]]\ndepartment = "engineering"', '[[axiom]]\ndepartment = '
     '"medbay"', r"axiom\[0\]\.placement"),
    ('department = "cargo_hold"\n', 'department = "cargo_hold"\ncycle = '
     '"pocket"\n', r"apply\[0\].*exactly one"),
    ('phase = 1\ndepartment = "cargo_hold"',
     'phase = 2\ndepartment = "cargo_hold"', r"apply\[0\]\.phase"),
    ('phase = 4\ncycle = "duct_link"', 'phase = 3\ncycle = "duct_link"',
     r"apply\[8\]\.phase.*phase 3"),
    ('on = "slab-lobby"', 'on = "slab-galley"', r"apply\[4\]\.on"),
    ('on = "slab-lobby"', 'on = "department-room"',
     r"apply\[4\]\.on.*'leaf'"),
    ('department = "armory"\ncount = { medium = [0, 0], large = [0, 1] }',
     'department = "armory"\ncount = { large = [0, 1] }',
     r"apply\[2\]\.count\.medium.*required"),
    ('on = "department-room"\ncount = { medium = [0, 1], large = [0, 2] }',
     'on = "department-room"\ncount = { medium = [0, 1], large = [2, 1] }',
     r"apply\[7\]\.count\.large"),
    ('on = "engineering.engine_room"\ncount = { medium = [0, 0], ',
     'on = "engineering.engine_room"\ncount = { medium = [0, 1], ',
     r"apply\[5\]\.count\.medium.*band"),
    ('department = "cargo_hold"\ncount = { medium = [1, 1], ',
     'department = "cargo_hold"\ncount = { medium = [0, 1], ',
     r"size 'medium'.*wing department"),
    ('cells = [2, 3]', 'cells = [2, 3]\nwalls = 1',
     r"\[validation\].*unknown field"),
    ('rooms = { medium = [14, 30], large = [28, 60] }',
     'rooms = { medium = [14, 30] }', r"validation\.rooms.*required"),
    ('zombies_scale = 1.5', 'zombies_mult = 1.5',
     r"profiles\.hard.*unknown field"),
    ('zombies_scale = 0.5', 'zombies_scale = 0', r"profiles\.easy\.zombies"),
]


@pytest.mark.parametrize("old, new, where", ROWS,
                         ids=[r[2] for r in ROWS])
def test_the_door_refuses_a_malformed_archetype(tmp_path, old, new, where):
    """PROPERTY: the archetype door refuses, at load and naming the file and
    the field, each malformed variant of the shipped freighter (archetype
    doc §1.1 with §7.1). BREAKS IF the door stops checking that field (the
    malformed file then loads) or stops naming it."""
    path = _copy(tmp_path, ARCH, old, new)
    with pytest.raises(R.RecipeError) as err:
        S.load_archetype(path)
    msg = str(err.value)
    assert "freighter.toml" in msg, msg
    assert re.search(where, msg, re.S), msg


@pytest.mark.parametrize("rel, old, new, where", [
    # a promoted stub that cannot take the airlock entry (>= 7)
    ("departments/armory.toml",
     "size = { across = [6, 8], along = [14, 34] }",
     "size = { across = [6, 6], along = [14, 34] }",
     r"profiles\.hard\.promote\[0\].*across"),
    # the closure walk: a disabled heat source is a pending mechanism
    ("fixtures.toml", "heat_source_enabled = true      # Erik",
     "heat_source_enabled = false      # Erik",
     r"crew_quarters.*galley.*heat_source.*enabled"),
])
def test_the_door_walks_the_closure(tmp_path, rel, old, new, where):
    """PROPERTY: the door reads THROUGH the archetype into the departments
    and fixtures it names: a promoted department whose stub cannot take an
    airlock, or a department whose resolved room still waits for an unbuilt
    stage-2 mechanism, is refused at load, by the archetype's file and the
    field, naming the extension (archetype doc §1.1). BREAKS IF the closure
    walk or the promote check is dropped (the ship would fail at stamp
    time instead)."""
    path = _copy(tmp_path, rel, old, new)
    with pytest.raises(R.RecipeError) as err:
        S.load_archetype(path)
    msg = str(err.value)
    assert "freighter.toml" in msg and re.search(where, msg, re.S), msg


# ---------------------------------------------------------------------------
# Determinism and the attempt loop
# ---------------------------------------------------------------------------

def test_same_inputs_give_the_same_graph(arch, ships):
    """PROPERTY: the quotient graph is a pure function of (archetype, size,
    profile, seed, max_per_team, zombies) -- a fresh load and a fresh run
    give the same digest -- and the seed matters: over many seeds the graphs
    are not all one (archetype doc §5.1). BREAKS IF a draw escapes the
    attempt's SeedSequence (set or dict-order iteration, a global rng), or
    the digest stops covering the graph."""
    fresh = S.load_archetype("freighter")
    for size in arch.sizes:
        for prof in arch.profiles:
            a = S.plan_quotient(fresh, size, prof, 7, zombies=3)
            assert a.digest() == ships[(size, prof, 7)].digest()
            assert a.digest() != S.plan_quotient(
                fresh, size, prof, 7, zombies=3, max_per_team=5).digest()
            digests = {ships[(size, prof, s)].digest() for s in SEEDS}
            assert len(digests) > 1, (size, prof)


def test_a_rejected_attempt_moves_to_the_next_stream(arch, monkeypatch):
    """PROPERTY: the layout's feasibility is consulted at the end of every
    grammar attempt (layout doc §13 A5); a refusal costs that attempt and
    the loop returns attempt g + 1's graph, drawn from its own stream
    ``SeedSequence([seed, 0, g + 1])``; a seed no attempt passes is
    :class:`GrammarExhausted` after G_MAX attempts. BREAKS IF the call site
    is removed, the loop stops at the first refusal, or attempts share a
    stream."""
    calls = []

    def refuse_g0(size, *, graph=None, archetype=None, profile=None):
        calls.append(graph)
        return "too wide" if graph is not None and graph.g == 0 else None
    monkeypatch.setattr(L, "layout_feasibility", refuse_g0)
    q = S.plan_quotient(arch, "large", "standard", 11)
    assert q.g == 1 and any(c is not None and c.g == 0 for c in calls)
    assert q.digest() == S.grammar_attempt(arch, "large", "standard", 11,
                                           1).digest()
    monkeypatch.setattr(L, "layout_feasibility",
                        lambda size, **kw: "never" if kw.get("graph")
                        else None)
    with pytest.raises(S.GrammarExhausted):
        S.plan_quotient(arch, "medium", "standard", 11)


# ---------------------------------------------------------------------------
# What every returned graph satisfies
# ---------------------------------------------------------------------------

def test_every_graph_keeps_every_invariant(ships):
    """PROPERTY: every returned graph satisfies all of the recipe doc's §1.5
    invariants (one edge per pair bar the bypass, no triangle, L5 crossings,
    reachability, duct-in-cell, the sealed-room rule), every edge joins
    rooms of the graph, every room's tags pass gen_ship's ROOM_TAGS (and its
    fuel / vent readers), every vent's duct is declared and a duct's rooms
    are ONE department instance unless it is a duct link; the ring and the
    bypass never coexist and a medium ship never carries a ring. BREAKS IF
    an application slips past its phase's invariants, a resolved tag goes
    off-schema, duct names lose their instance qualifier, or the exclusion
    is dropped."""
    for key, q in ships.items():
        S.check_invariants(S.Draft.of(q), 4)
        ids = {r.id for r in q.rooms}
        assert len(ids) == len(q.rooms), key
        ducts = dict(q.ducts)
        owners: dict = {}
        for r in q.rooms:
            for k, v in r.tags.items():
                assert G.ROOM_TAGS[k][1](v), (key, r.id, k, v)
            if "fuel" in r.tags:
                G.fuel_spec(r.tags["fuel"])
            for dn, _roles, short in (G.vent_specs(r.tags["vent"])
                                      if "vent" in r.tags else ()):
                assert not short and dn in ducts, (key, r.id, dn)
                owners.setdefault(dn, set()).add((r.dept, r.instance))
        for dn, own in owners.items():
            assert len(own) == 1 or (len(own) == 2 and dn.startswith("link_")),\
                (key, dn, own)
        for e in q.edges:
            assert e.a in ids and e.b in ids and e.kind in G.EDGE_KINDS, key
        # what the layout's `behind` primitive can realise (layout doc §3,
        # §7): one behind child per parent, a side or end room, never the
        # exit room (its far side is the slab) nor a hull-door room
        behind = [r for r in q.rooms if r.attach == "behind"]
        parents = [r.parent for r in behind]
        assert len(parents) == len(set(parents)), key
        for p in parents:
            par = q.room(p)
            assert par.attach in ("side", "end") and not par.exit \
                and "hull_door" not in par.tags, (key, p)
        types = {c.type for c in q.cycles}
        assert not {"maintenance_ring", "airlock_bypass"} <= types, key
        if q.size == "medium":
            assert "maintenance_ring" not in types, key


def test_every_count_lies_in_its_range(arch, ships):
    """PROPERTY: every [[apply]] entry placed a count inside its effective
    (profile-shifted, capped) range and the graph records it; the secrets
    are the profile's budget; the exits lie in the archetype's range and
    never exceed what the ship realises (one exit edge each); the band's
    section count lies in its range; the zombies are the profile's scaling
    of the argument (archetype doc §2, §4). BREAKS IF a count is drawn
    outside its range, a rejected candidate is counted, the shift or the
    cap is lost, or the scaling rule changes."""
    for (size, prof, seed), q in ships.items():
        placed = dict(q.placed)
        for e in arch.applies:
            lo, hi = arch.ranges[(size, prof)][e.index]
            if e.department:
                n = sum(1 for d in q.departments if d.prov[0] == e.rule)
            else:
                n = sum(1 for c in q.cycles if c.entry == e.index)
            assert lo <= n <= hi and placed[e.index] == n, (size, prof, seed,
                                                            e.rule, n)
        assert len(q.secrets) == arch.secrets(prof)
        assert all(q.room(s).tags.get("secret") for s in q.secrets)
        lo, hi = arch.exits
        assert lo <= q.exits <= hi and len(q.exit_edges) == q.exits
        assert all(L.SLAB in (e.a, e.b) for e in q.exit_edges)
        slo, shi = arch.sections[size]
        assert slo <= len(q.band) - sum(q.room(x).role in L.MADE_ROLES
                                        for x in q.band) <= shi
        assert q.zombies == arch.scaled_zombies(prof, q.zombies_base)


def test_the_skeleton_rows_cells_and_exits(arch, ships):
    """PROPERTY (layout doc §2, §13 A1 / A2 / A7, archetype doc §4): the band
    runs fore to aft through the main-cell sections (doors between them)
    then the hub department's block (a bulkhead between them); the lobby's
    door goes to the aft-most main section, which is the zombie room (and
    carries the tag exactly when there are zombies); the lobby enters the
    hub department through an airlock carved on the LOBBY's side; wing
    departments are dealt to rows A, B alternately in gradient order (spawn
    first), positions counting from the slab end; an extra exit joins the
    slab to the exit room of a row's first department; cells are main = 0,
    the hub department's own, and each promoted department its own, its
    entry an airlock carved on its stub. BREAKS IF the band order, the
    airlock's orientation, the row deal, the exit anchoring or the cell
    numbering changes."""
    for (size, prof, seed), q in ships.items():
        mains = [x for x in q.band if q.room(x).role == L.SECTION]
        block = _block(q)
        assert q.band == tuple(mains + block) and mains and block
        kinds = {frozenset((e.a, e.b)): e.kind for e in q.edges}
        for x, y in zip(mains, mains[1:]):
            assert kinds[frozenset((x, y))] == "door"
        assert kinds[frozenset((mains[-1], block[0]))] == "bulkhead"
        assert q.zombie_room == mains[-1]
        assert kinds[frozenset((L.LOBBY, mains[-1]))] == "door"
        assert bool(q.room(q.zombie_room).tags.get("zombie")) == \
            (q.zombies > 0)
        hub_cell = q.room(block[0]).cell
        assert hub_cell != R.MAIN_CELL
        locks = [e for e in q.edges if e.kind == "airlock"
                 and L.LOBBY in (e.a, e.b)]
        assert len(locks) == 1 and locks[0].a == L.LOBBY
        assert q.room(locks[0].b).cell == hub_cell
        wing = sorted((d for d in q.departments if d.placement == "wing"),
                      key=lambda d: d.order)
        assert [d.row for d in wing] == [S.ROWS[i % 2]
                                         for i in range(len(wing))]
        assert [d.position for d in wing] == [i // 2
                                              for i in range(len(wing))]
        towards = [d.toward for d in wing]
        assert towards == sorted(towards, key=lambda t: t != "spawn")
        firsts = {d.row: d for d in wing if d.position == 0}
        for e in q.exit_edges[1:]:
            room = q.room(e.b)
            assert room.exit and any(room.dept == d.name
                                     and room.instance == d.instance
                                     for d in firsts.values())
        # the draw is capped at what the ship realises, never below it
        avail = 1 + sum(any(r.exit and r.dept == d.name
                            and r.instance == d.instance
                            and r.cell == R.MAIN_CELL for r in q.rooms)
                        for d in firsts.values())
        (drew,) = [int(c.split()[1]) for _p, rule, c, _o in q.log
                   if rule == "exits"]
        assert q.exits == min(drew, avail), (size, prof, seed)
        cells = arch.cells(prof)
        for d in q.departments:
            assert d.cell == cells[d.name]
            assert d.promoted == (d.name in arch.promoted(prof))
            if d.promoted:
                ent = [e for e in q.edges if e.role == "entry"
                       and d.entry in (e.a, e.b)]
                assert ent[0].kind == "airlock" and ent[0].a == d.entry


def test_each_cycle_type_has_its_shape(ships):
    """PROPERTY (layout doc §4.1-4.6, archetype doc §3): a retreat loop adds
    one second stub from the passageway into the department's END room; a
    flank pair one narrow strip with a door into the slab and the lobby; a
    ring 4 or 6 rooms (two on the axis, the rest wing-class side rooms), the
    hub joined to exactly the odd-numbered ones, the lobby's airlock moved
    onto a side room; a bypass a breakable beside the lobby's airlock on the
    same pair; a pocket one room with one edge to a department room; a duct
    link a new duct carrying both roles in two rooms of one cell in
    different department instances. BREAKS IF a realisation changes shape
    (a triangle-making ring, a strip into the passageway, a second pocket
    door, a link across cells)."""
    seen = set()
    for key, q in ships.items():
        ids = {r.id: r for r in q.rooms}
        for c in q.cycles:
            seen.add(c.type)
            made = [ids[x] for x in c.rooms]
            if c.type == "retreat_loop":
                (s2,) = made
                stub, end = c.anchor
                assert s2.role == L.STUB2 and ids[end].attach == "end"
                made_e = {(e[0], e[1]) for e in c.edges}
                assert {(L.PASSAGEWAY, s2.id), (s2.id, end)} <= made_e
                # the rest: side rooms moved from the first stub to it
                for a, b in made_e - {(L.PASSAGEWAY, s2.id), (s2.id, end)}:
                    assert a == s2.id and ids[b].parent == s2.id
                    assert any(r[:2] == (stub, b) for r in c.removed)
            elif c.type == "flank_pair":
                (strip,) = made
                assert strip.width == "narrow"
                assert {frozenset(e[:2]) for e in c.edges} == {
                    frozenset((L.SLAB, strip.id)),
                    frozenset((strip.id, L.LOBBY))}
            elif c.type == "maintenance_ring":
                n = dict(c.params)["n"]
                assert n in (4, 6) and len(made) == 2 + (n - 2) // 2
                assert sum(r.side == S.AXIS for r in made) == 2
                hub = c.anchor[0]
                spokes = {e.b if e.a == hub else e.a for e in q.edges
                          if hub in (e.a, e.b) and (e.b if e.a == hub
                                                    else e.a) in c.rooms}
                assert spokes == {r.id for r in made if r.index % 2}
                lock = next(e for e in q.edges if e.kind == "airlock"
                            and e.a == L.LOBBY)
                assert ids[lock.b].role == L.RING_SIDE
            elif c.type == "airlock_bypass":
                pair = sorted(e.kind for e in q.edges
                              if {e.a, e.b} == set(c.anchor))
                assert pair == sorted(G.BYPASS_KINDS)
            elif c.type == "pocket":
                (p,) = made
                es = [e for e in q.edges if p.id in (e.a, e.b)]
                assert len(es) == 1 and p.parent in (es[0].a, es[0].b)
            elif c.type == "duct_link":
                a, b = (ids[x] for x in c.anchor)
                name = dict(c.params)["duct"]
                assert a.cell == b.cell and (a.dept, a.instance) != \
                    (b.dept, b.instance)
                for r in (a, b):
                    specs = {dn: ro for dn, ro, _ in G.vent_specs(
                        r.tags["vent"])}
                    assert specs[name] == G.VENT_ROLES
    assert seen == {"retreat_loop", "flank_pair", "maintenance_ring",
                    "airlock_bypass", "pocket", "duct_link"}


def test_the_ring_and_the_bypass_exclude_each_other_by_name(ships):
    """PROPERTY (archetype doc §3): when one of the maintenance ring and the
    airlock bypass is on a ship and the other's entry tries its candidate,
    the provenance log records the rejection BY NAME ("mutually
    exclusive"), and the attempt does not end. BREAKS IF the exclusion
    becomes silent (no log line) or fatal (a GrammarReject)."""
    hits = 0
    for q in ships.values():
        for _ph, rule, _cand, out in q.log:
            if rule.endswith(("maintenance_ring", "airlock_bypass")) \
                    and out.startswith("rejected"):
                assert "mutually exclusive" in out, out
                hits += 1
    assert hits


def test_two_instances_of_a_department_never_share_a_duct(tmp_path):
    """PROPERTY (the lead's decision 5): a department drawn twice in a wing
    gets its ducts twice, each instance's rooms on its own qualified duct
    names. BREAKS IF duct names lose their instance qualifier (two
    instances would breathe through one plenum)."""
    path = _copy(tmp_path, ARCH,
                 'department = "cargo_hold"\ncount = { medium = [1, 1], '
                 'large = [1, 1] }',
                 'department = "cargo_hold"\ncount = { medium = [1, 1], '
                 'large = [2, 2] }')
    q = S.plan_quotient(S.load_archetype(path), "large", "standard", 3)
    by_inst: dict = {}
    for r in q.rooms:
        if r.dept == "cargo_hold" and "vent" in r.tags:
            for dn, _ro, _sh in G.vent_specs(r.tags["vent"]):
                if not dn.startswith("link_"):
                    by_inst.setdefault(r.instance, set()).add(dn)
    assert set(by_inst) == {0, 1} and not (by_inst[0] & by_inst[1])


# ---------------------------------------------------------------------------
# The moves and the invariants, on small crafted graphs
# ---------------------------------------------------------------------------

def _small(*rooms) -> S.Draft:
    """slab - a - b (doors), plus ``rooms``."""
    d = S.Draft()
    d.rooms = {"slab": S.QRoom(id="slab", role=L.SLAB),
               "a": S.QRoom(id="a"),
               "b": S.QRoom(id="b", tags={"vent": {"duct": "x", "roles":
                                                   ["supply", "return"]}})}
    for r in rooms:
        d.rooms[r.id] = r
    d = S.join(d, "slab", "a", "door", P, "test")
    return S.join(d, "a", "b", "door", P, "test")


def _vent(*roles) -> dict:
    return {"vent": {"duct": "x", "roles": list(roles)}}


@pytest.mark.parametrize("name, bad, good, phase", [
    ("one_edge_per_pair",
     lambda d: S.parallel(d, "a", "b", "open", P),
     lambda d: S.parallel(S.leaf(d, "b", S.QRoom(id="c"), "airlock", P),
                          "b", "c", "breakable", P), 1),
    ("no_triangle",
     lambda d: S.join(d, "slab", "b", "door", P, "test"),
     lambda d: S.second_leg(d, "slab", "a", "b", S.QRoom(id="c"),
                            ("door", "door"), P), 1),
    ("cell_crossing",
     lambda d: S.leaf(d, "b", S.QRoom(id="c", cell=1), "door", P),
     lambda d: S.leaf(d, "b", S.QRoom(id="c", cell=1), "airlock", P), 1),
    ("reachable",
     lambda d: S.leaf(d, "b", S.QRoom(id="c"), "bulkhead", P),
     lambda d: S.leaf(d, "b", S.QRoom(id="c", tags={"secret": True}),
                      "breakable", P), 3),
    ("duct_in_cell",
     lambda d: S.leaf(d, "b", S.QRoom(id="c", cell=1,
                                      tags=_vent("supply", "return")),
                      "airlock", P),
     lambda d: S.leaf(d, "b", S.QRoom(id="c", tags=_vent("supply",
                                                         "return")),
                      "door", P), 4),
    ("sealed_room",
     lambda d: S.leaf(d, "b", S.QRoom(id="c", tags=_vent("supply")), "door",
                      P),
     lambda d: S.leaf(d, "b", S.QRoom(id="c", tags=_vent("supply")), "open",
                      P), 4),
])
def test_each_invariant_refuses_by_name(name, bad, good, phase):
    """PROPERTY (recipe doc §1.5, §5.3): an application that breaks an
    invariant is refused NAMING it, from the first phase that invariant
    runs in (contact rules from phase 1, reachability from 3, the air rules
    in 4) and not before; the same move applied legally passes. BREAKS IF an
    invariant is dropped, renamed, or run in the wrong phase."""
    d = bad(_small())
    with pytest.raises(S.InvariantViolation) as err:
        S.check_invariants(d, phase)
    assert err.value.rule == name
    if phase > 1:
        S.check_invariants(d, phase - 1)
    S.check_invariants(good(_small()), 4)


def test_a_skeleton_t_junction_is_not_a_triangle():
    """PROPERTY (ship_layout's docstring): a 3-cycle all of whose pairs are
    SKELETON contacts (the lobby against two stacked band sections) is a
    T-junction the skeleton draws, not a refused triangle -- and it stops
    being exempt as soon as one pair is a move's edge. BREAKS IF the
    exemption widens to moves or vanishes (the freighter could not exist)."""
    d = S.Draft()
    d.rooms = {x: S.QRoom(id=x, role=L.SLAB if x == "slab" else None)
               for x in ("slab", "s", "h")}
    for a, b in (("slab", "s"), ("s", "h"), ("slab", "h")):
        d = S.join(d, a, b, "door", P, "skeleton")
    S.check_invariants(d, 1)
    d.edges[-1] = S.QEdge("slab", "h", "door", "cycle", P)
    with pytest.raises(S.InvariantViolation, match="no_triangle"):
        S.check_invariants(d, 1)


@pytest.mark.parametrize("move, call", [
    ("subdivide", lambda d: S.subdivide(d, "slab", "b", S.QRoom(id="c"),
                                        ("door", "door"), P)),
    ("edge_to_cycle", lambda d: S.edge_to_cycle(
        d, "slab", "b", S.QRoom(id="c"), S.QRoom(id="e"),
        ("door",) * 4, P)),
    ("leaf", lambda d: S.leaf(d, "nowhere", S.QRoom(id="c"), "door", P)),
    ("ring", lambda d: S.ring(d, "a", S.QRoom(id="f"), S.QRoom(id="g"), (),
                              "door", "door", P)),
    ("parallel", lambda d: S.parallel(d, "a", "b", "door", P)),
    ("duct", lambda d: S.duct(S.leaf(d, "b", S.QRoom(id="c", cell=1),
                                     "airlock", P),
                              "a", "c", "y", "derelict", G.VENT_ROLES, P)),
    ("second_leg", lambda d: S.second_leg(d, "slab", "b", "a",
                                          S.QRoom(id="c"), ("door", "door"),
                                          P)),
])
def test_each_move_refuses_its_precondition_by_name(move, call):
    """PROPERTY (recipe doc §0.1): each of the seven moves checks its
    precondition and refuses naming itself, leaving its input untouched.
    BREAKS IF a move applies without its precondition (a subdivide of a
    missing edge, a ring without side rooms, a duct across cells)."""
    d = _small()
    before = (dict(d.rooms), list(d.edges))
    with pytest.raises(S.PreconditionFailed) as err:
        call(d)
    assert err.value.rule == move
    assert (d.rooms, d.edges) == before


# ---------------------------------------------------------------------------
# recipe_hash
# ---------------------------------------------------------------------------

def test_recipe_hash_tracks_content_not_comments(tmp_path, monkeypatch):
    """PROPERTY (archetype doc §5.1): the recipe hash ignores a comment edit
    and another profile's table, and moves with a resolved preset value and
    with a department file the archetype names. BREAKS IF it hashes raw
    text, hashes every profile, or misses the closure or the presets."""
    def h(path, prof="standard"):
        a = S.load_archetype(path)
        return S.recipe_hash(a, prof, a.library)
    base = _copy(tmp_path / "base")
    ref = h(base)
    assert h(_copy(tmp_path / "comment", ARCH,
                   "# ---- phase 1: grow (departments) ----",
                   "# ---- phase 1: grow ----")) == ref
    other = _copy(tmp_path / "easy", ARCH, "zombies_scale = 0.5",
                  "zombies_scale = 0.4")
    assert h(other) == ref and h(other, "easy") != h(base, "easy")
    assert h(_copy(tmp_path / "dept", "departments/cargo_hold.toml",
                   "count = [1, 8]", "count = [1, 7]")) != ref
    monkeypatch.setattr(CFG.levelgen.lights.dim, "level",
                        CFG.levelgen.lights.dim.level + 0.25)
    assert h(base) != ref
