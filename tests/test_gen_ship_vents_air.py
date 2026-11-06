"""The air-volume vent rule and the vent design flow (#124 / #151;
docs/levelgen_zone_vocabulary_2026-10-29.md §0.6, Erik's ruling 2026-11-06:
the ventilation is realistic, and a vent stands only where its duct's air
comes back).

The gate of §0.6 item 6: pure properties of the air graph, the rule and the
design flow (hand-built inputs and generated recipe ships), then ONE idle
recipe ship simulated -- no air volume's gas drifts, no room without a heat
source leaves a narrow pressure band, teargas put in the heads is found in the
bunk rooms and nowhere the heads' air cannot reach.

Each test names the property it protects and the change that must break it.
"""
from __future__ import annotations

import shutil
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
import ship_grammar as S  # noqa: E402
import ship_layout as L  # noqa: E402
from config import CFG  # noqa: E402

try:
    import breach_physics as _bp  # noqa: E402
except Exception:                  # pragma: no cover - a box without the build
    _bp = None
needs_bp = pytest.mark.skipif(_bp is None,
                              reason="needs the compiled breach_physics")

# §0.6 item 1, by name: these pass gas from tick 0 and nothing closes them;
# these do not (gas-tight, closable, or must break first).
AIR_KINDS = ("open", "grille", "vented_door")
NOT_AIR_KINDS = ("door", "entity_door", "window", "airlock", "breakable",
                 "bulkhead", "vent_only")
SUPPLY, RETURN = ["supply"], ["return"]
BOTH = ["supply", "return"]
PROV = ("test", 0)

# The pure sweep: a few seeds on every size and profile of the freighter.
SWEEP_SEEDS = (0, 1, 2)
# The simulated ship: the one #124's reproduction ran (medium, seed 28).
SIM_SEED, SIM_SIZE = 28, "medium"
IDLE_S = 30                        # idle sim seconds for (a) and (b)
TEAR_S = 5                         # sim seconds for (c)


def _v(roles, duct="q"):
    return ((duct, tuple(r for r in G.VENT_ROLES if r in roles), False),)


# ---------------------------------------------------------------------------
# The air graph (§0.6 item 1)
# ---------------------------------------------------------------------------

def test_air_volumes_join_exactly_through_the_air_kinds():
    """PROPERTY: two rooms joined by ``open``, ``grille`` or ``vented_door``
    are ONE air volume; joined by a door, an entity door, a window, an
    airlock, a breakable wall, a bulkhead or a vent_only edge they are two;
    a chain of air edges is one volume however long; and the ShipGraph twin
    (``air_volume_of``) differs from ``compartment_of`` exactly where a
    vented door bounds a compartment but not a volume. BREAKS IF the air
    column is set on a kind that can be shut or must break first, dropped
    from one of the three, or ``air_volume_of`` reads ``bounds``."""
    for kind in AIR_KINDS:
        assert G.EDGE_KINDS[kind].air, kind
        vol = G.air_volumes(["a", "b"], [("a", "b", kind)])
        assert vol["a"] == vol["b"], kind
    for kind in NOT_AIR_KINDS:
        assert not G.EDGE_KINDS[kind].air, kind
        vol = G.air_volumes(["a", "b"], [("a", "b", kind)])
        assert vol["a"] != vol["b"], kind
    chain = G.air_volumes("abcd", [("a", "b", "open"), ("b", "c", "grille"),
                                   ("c", "d", "vented_door")])
    assert len(set(chain.values())) == 1
    g = _ring_graph("vented_door")
    assert g.air_volume_of()[0] == g.air_volume_of()[1]
    assert g.compartment_of()[0] != g.compartment_of()[1]


def _ring_graph(kind: str, tags: dict | None = None) -> G.ShipGraph:
    """Two rooms side by side, joined by one ``kind`` edge."""
    tags = tags or {}
    rooms = [G.Room(0, 4, 4, 10, 10, "spawn", {"spawn_team": 0,
                                               **tags.get(0, {})}),
             G.Room(1, 15, 4, 10, 10, "spawn", {"spawn_team": 1,
                                                **tags.get(1, {})})]
    return G.ShipGraph(width=29, height=18, hull=(3, 3, 25, 14), rooms=rooms,
                       edges=[G.Edge(0, 1, kind)], mirror_x=False,
                       max_per_team=4, zombies=0)


# ---------------------------------------------------------------------------
# The rule (§0.6 item 2)
# ---------------------------------------------------------------------------

def test_the_rule_refuses_a_volume_with_a_lone_role_and_names_it():
    """PROPERTY: a duct's air volume holding only supplies (or only
    returns) is a violation naming the duct, the volume's rooms, the rooms
    with the role it has and the role it lacks; a return-only room that
    shares air with a supply room through a vented door is accepted, and the
    SAME room behind a plain door is refused (both rooms are then lone);
    a second duct is judged on its own; v1's string shorthand is exempt.
    BREAKS IF the rule judges rooms instead of volumes, stops reading the
    air column, merges ducts, or the shorthand exemption is lost (ship_demo
    would move)."""
    lone = G.vent_air_rule({"a": _v(SUPPLY), "b": ()}, [("a", "b", "door")])
    assert len(lone) == 1
    v = lone[0]
    assert (v.duct, v.rooms, v.have, v.have_rooms, v.missing) == \
        ("q", ("a",), "supply", ("a",), "return")
    assert "duct 'q'" in str(v) and "no 'return'" in str(v)
    lone_r = G.vent_air_rule({"a": (), "b": _v(RETURN)}, [("a", "b", "door")])
    assert [x.missing for x in lone_r] == ["supply"]
    shared = {"bunk": _v(SUPPLY), "heads": _v(RETURN)}
    assert G.vent_air_rule(shared, [("bunk", "heads", "vented_door")]) == []
    split = G.vent_air_rule(shared, [("bunk", "heads", "door")])
    assert sorted(x.missing for x in split) == ["return", "supply"]
    two = G.vent_air_rule({"a": _v(BOTH) + _v(SUPPLY, "x"), "b": ()},
                          [("a", "b", "open")])
    assert [(x.duct, x.missing) for x in two] == [("x", "return")]
    assert G.vent_air_rule({"a": G.vent_specs("supply"),
                            "b": G.vent_specs("return")},
                           [("a", "b", "door")]) == []


def test_the_three_sites_refuse_what_the_rule_refuses():
    """PROPERTY: ``ShipGraph.check`` refuses a lone-role volume and accepts
    the same vents across a vented door; the grammar's ``air_volume``
    invariant refuses the lone room and accepts it through a vented door.
    (The recipe door's refusal is in test_levelgen_recipes' REFUSALS.)
    BREAKS IF a site keeps its own copy of the rule (the old sealed-room
    test) instead of calling ``vent_air_rule``."""
    tags = {0: {"vent": {"duct": "q", "roles": SUPPLY}},
            1: {"vent": {"duct": "q", "roles": RETURN}}}
    with pytest.raises(G.GraphSchemaError, match=r"duct 'q'.*no 'return'"):
        _ring_graph("door", tags)
    _ring_graph("vented_door", tags)
    d = S.Draft()
    d.rooms = {"slab": S.QRoom(id="slab", role=L.SLAB),
               "a": S.QRoom(id="a", tags={"vent": {"duct": "q",
                                                   "roles": SUPPLY}}),
               "b": S.QRoom(id="b", tags={"vent": {"duct": "q",
                                                   "roles": RETURN}})}
    d = S.join(d, "slab", "a", "door", PROV, "test")
    with pytest.raises(S.InvariantViolation) as err:
        S.check_invariants(S.join(d, "a", "b", "door", PROV, "test"), 4)
    assert err.value.rule == "air_volume"
    S.check_invariants(S.join(d, "a", "b", "vented_door", PROV, "test"), 4)


def test_the_recipe_door_checks_every_pattern_of_optional_absence(tmp_path):
    """PROPERTY: the recipe door judges a department with its optional
    rooms present AND absent: crew quarters whose only return sits in the
    optional laundry (open to the stub) passes while the laundry is there
    and is refused, naming the absence, because it is not always there.
    BREAKS IF the door checks only the full department."""
    root = tmp_path / "recipes"
    shutil.copytree(R.RECIPES_DIR, root)
    p = root / "departments" / "crew_quarters.toml"
    text = p.read_text(encoding="utf-8")
    edits = (('vent = { duct = "quarters", roles = ["return"] }   # the',
              '# was the heads\' return   # the'),
             ('p = 0.3\n', 'p = 0.3\nvent = { duct = "quarters", roles = '
                           '["return"] }\n'),
             ('b = "laundry"\nkind = "door"', 'b = "laundry"\nkind = "open"'),
             ('b = "heads"\nkind = "vented_door"', 'b = "heads"\nkind = "door"'))
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    p.write_text(text, encoding="utf-8")
    with pytest.raises(R.RecipeError, match=r"no 'return'.*laundry'\] absent"):
        R.load_recipes(root)


def test_the_design_dial_is_refused_when_missing_or_not_positive(monkeypatch):
    """PROPERTY: ``[levelgen.vents] ach`` is read through a door that
    refuses a missing key and a value that is not a positive number, naming
    the key. BREAKS IF the door falls back to a default (a silent studio
    rate) or lets a zero / negative rate through (a vent that never moves
    air, or a reversed one)."""
    assert G.vent_ach() > 0
    for bad in (0, -1.0, "ten", True):
        monkeypatch.setattr(CFG.levelgen.vents, "ach", bad)
        with pytest.raises(ValueError, match=r"\[levelgen.vents\] ach"):
            G.vent_ach()
    monkeypatch.delattr(CFG.levelgen.vents, "ach")
    with pytest.raises(ValueError, match=r"\[levelgen.vents\] ach"):
        G.vent_ach()


# ---------------------------------------------------------------------------
# The design flow on generated ships (§0.6 item 3)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ships():
    arch = S.load_archetype("freighter")
    return {(size, prof, seed): G.generate(seed, size, recipe=arch,
                                           profile=prof)
            for size in arch.sizes for prof in arch.profiles
            for seed in SWEEP_SEEDS}


def _vents_by_room(plan) -> dict:
    """room id -> [(duct entity id, role, q_circ)] -- a wall vent serves the
    room whose air tile its aperture faces."""
    step = {"n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0)}
    out: dict = {}
    for e in plan.entities:
        if e.class_name != "vent":
            continue
        dx, dy = step[e.fields["facing"]]
        x, y = e.fields["x"] + dx, e.fields["y"] + dy
        rm = next(r for r in plan.graph.rooms
                  if r.x <= x < r.x + r.w and r.y <= y < r.y + r.h)
        out.setdefault(rm.id, []).append(
            (e.fields["duct"], e.fields["role"], e.fields["q_circ"]))
    return out


def _raw(q: float) -> int:
    r = q * G.RATE_ONE
    assert r == int(r), f"q_circ {q!r} is not a multiple of the rate quantum"
    return int(r)


def test_every_duct_balances_in_every_air_volume(ships):
    """PROPERTY (§0.6 item 3): on recipe ships of every size and profile,
    for every duct and every air volume it serves the supplies' rates and
    the returns' rates are the SAME integer in the runtime's quantum (raw
    Q16 bulk N per second), and that integer is ``ach / 3600`` x the served
    rooms' floor area (to the quantum, +1 for a mirror pair's parity); no
    vent carries the studio rate. BREAKS IF a rate is written off the
    quantum, the two sides are apportioned to different totals, the flow
    stops following the served area or ``ach``, or the recipe path falls
    back to ``Q_CIRC``."""
    per_tile = G.vent_ach() / 3600.0 * G.RATE_ONE
    for key, plan in ships.items():
        g = plan.graph
        vol = g.air_volume_of()
        sums: dict = {}
        served: dict = {}
        for rid, vents in _vents_by_room(plan).items():
            for duct, role, q in vents:
                assert q != G.Q_CIRC, (key, rid, duct)
                k = (duct, vol[rid])
                s = sums.setdefault(k, {"supply": 0, "return": 0})
                s[role] += _raw(q)
                served.setdefault(k, set()).add(rid)
        assert sums, key
        for k, s in sums.items():
            assert s["supply"] == s["return"] > 0, (key, k, s)
            area = sum(g.rooms[r].w * g.rooms[r].h for r in served[k])
            assert abs(s["supply"] - per_tile * area) <= 1.5, (key, k, s)


def test_mirror_twins_get_equal_rates(ships):
    """PROPERTY: a room and its mirror twin carry the same vents at the same
    rates (role by role; the twin's on its mirrored duct). BREAKS IF the
    apportionment's remainder goes to one wing (ties broken by something
    the mirror does not preserve)."""
    for key, plan in ships.items():
        by_room = _vents_by_room(plan)
        for rm in plan.graph.rooms:
            tw = rm.mirror
            if tw is None or tw == rm.id:
                continue
            mine = sorted((r, q) for _d, r, q in by_room.get(rm.id, ()))
            theirs = sorted((r, q) for _d, r, q in by_room.get(tw, ()))
            assert mine == theirs, (key, rm.id, tw)


# ---------------------------------------------------------------------------
# The gate, simulated (§0.6 item 6): one idle recipe ship
# ---------------------------------------------------------------------------

def _sim(plan, tmp_path):
    from level_loader import load as load_level
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    G.write(plan, tmp_path / plan.name, version="test")
    sim = Simulation(load_level(plan.name, levels_dir=str(tmp_path)), seed=1,
                     breach_physics=_bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _box(rm):
    return slice(rm.y, rm.y + rm.h), slice(rm.x, rm.x + rm.w)


@pytest.fixture(scope="module")
def idle(tmp_path_factory):
    """The simulated ship run idle for ``IDLE_S`` seconds (no units act):
    the plan, the sim after, the tile components, the bulk N per component
    at the start and the end, and every vented room's mean pressure each
    second."""
    if _bp is None:
        pytest.skip("needs the compiled breach_physics")
    from scipy import ndimage
    from simulation import atmosphere_fixed as afx
    from simulation.gases import INERT_N2, O2
    plan = G.generate(SIM_SEED, SIM_SIZE, recipe="freighter")
    sim = _sim(plan, tmp_path_factory.mktemp("idle"))
    g = sim.gmap
    gas_open = ~g.solid & ~g.is_vacuum & ~g.is_ambient
    labels, _n = ndimage.label(gas_open)
    vented = [rm for rm in plan.graph.rooms if "vent" in rm.tags]

    def bulk():
        n = g.gas[O2].astype(np.int64) + g.gas[INERT_N2].astype(np.int64)
        return {lab: int(n[labels == lab].sum())
                for lab in {int(labels[_box(rm)].max()) for rm in vented}}

    def pressures():
        out = {}
        for rm in vented:
            sl = _box(rm)
            out[rm.id] = float(afx.dequantize_f32(g.atmosphere[sl])
                               [~g.solid[sl]].mean())
        return out
    n0, p = bulk(), [pressures()]
    for _s in range(IDLE_S):
        for _t in range(int(sim._tps)):
            sim.step()
        p.append(pressures())
    return {"plan": plan, "sim": sim, "labels": labels, "n0": n0,
            "n1": bulk(), "p": p}


@needs_bp
def test_the_runtime_rates_balance_per_duct_and_volume(idle):
    """PROPERTY: what the vent RUNTIME runs (each vent's quantized
    ``rate_raw_per_s``) sums to the same integer for a duct's supplies and
    its returns in every air volume -- the generator's unit (one tile-volume
    of ambient air per second = 2^16 raw) is the runtime's. BREAKS IF
    vent_system.build_vents' conversion (q_circ / (C * T_amb), quantized)
    stops being the identity on the design's quanta."""
    plan, sim = idle["plan"], idle["sim"]
    vol = plan.graph.air_volume_of()
    sums: dict = {}
    for v in sim._vents:
        rm = next(r for r in plan.graph.rooms
                  if r.x <= v.aperture_x < r.x + r.w
                  and r.y <= v.aperture_y < r.y + r.h)
        s = sums.setdefault((v.inst.fields["duct"], vol[rm.id]),
                            {"supply": 0, "return": 0})
        s[v.role] += v.rate_raw_per_s
    assert sums
    for k, s in sums.items():
        assert s["supply"] == s["return"] > 0, (k, s)


@needs_bp
def test_no_air_volume_drifts_while_the_ship_idles(idle):
    """PROPERTY (§0.6 item 6a): on the idle recipe ship every body of air
    holding a vent (a gas-connected region of the loaded level) keeps its
    bulk gas amount within 0.1 % of its start -- gas amount, not pressure
    (the engine room's heat source legitimately raises its pressure). Every
    graph air volume lies inside one such region. Before #124 the heads lost
    72 % and the quarters 9 % in 60 s. BREAKS IF a duct's supply and return
    in a volume stop being equal, the rule lets a lone role through, or the
    duct link exchanges net air between its two volumes."""
    plan, labels = idle["plan"], idle["labels"]
    vol = plan.graph.air_volume_of()
    region: dict = {}
    for rm in plan.graph.rooms:
        labs = set(np.unique(labels[_box(rm)])) - {0}
        if labs:
            assert len(labs) == 1, rm.id
            assert region.setdefault(vol[rm.id], labs) == labs, rm.id
    for lab, n0 in idle["n0"].items():
        drift = (idle["n1"][lab] - n0) / n0
        assert abs(drift) <= 1e-3, (lab, drift)


@needs_bp
def test_no_vented_room_leaves_its_pressure_band(idle):
    """PROPERTY (§0.6 item 6b): on the idle recipe ship every vented room
    without a heat source -- the heads included, which sits a little below
    the corridor it draws from -- stays within PRESSURE_BAND of its starting
    mean pressure every second. Measured on this ship: at most ~0.5 % over
    30 s; before #124 the heads fell to 0.25 atm within 10 s. BREAKS IF the
    heads' return outruns what the volume supplies, or a supply pumps a
    volume up."""
    plan, series = idle["plan"], idle["p"]
    rooms = [rm for rm in plan.graph.rooms
             if "vent" in rm.tags and "heat_source" not in rm.tags]
    assert any(rm.tags.get("role") == "heads" for rm in rooms)
    for rm in rooms:
        p0 = series[0][rm.id]
        worst = max(abs(s[rm.id] - p0) / p0 for s in series)
        assert worst <= PRESSURE_BAND, (rm.id, rm.tags.get("role"), worst)


PRESSURE_BAND = 0.02


@needs_bp
def test_teargas_in_the_heads_reaches_the_bunks_and_nowhere_off_its_air(
        tmp_path):
    """PROPERTY (§0.6 item 6c): teargas put into the heads (the canonical
    payload path, ``tear_burst``) is found after a few seconds in every bunk
    room of its quarters (the heads' return takes it into the quarters
    duct, whose filter passes teargas, and the bunks' supplies hand it out;
    the vented door's jambs let it into the corridor too), and NONE is in
    any room the heads' air cannot reach -- a room outside every air volume
    joined to the heads' by a chain of volumes and the ducts with vents in
    them (the other wing's quarters, the cargo hold the duct link does not
    touch, engineering). BREAKS IF the heads' return lands on another duct,
    the quarters filter scrubs teargas, the heads is sealed off its quarters
    again, or a duct's air leaks into a volume it has no vent in."""
    from simulation.gases import TEARGAS
    from simulation.payloads import execute_payload
    plan = G.generate(SIM_SEED, SIM_SIZE, recipe="freighter")
    graph = plan.graph
    heads = min((rm for rm in graph.rooms if rm.tags.get("role") == "heads"),
                key=lambda rm: rm.id)
    sim = _sim(plan, tmp_path)
    g = sim.gmap
    execute_payload(g, sim.edit_queue, sim.units, heads.y + heads.h // 2,
                    heads.x + heads.w // 2,
                    sim.weapons_tables.payloads.by_name["tear_burst"],
                    sim.rng)
    for _t in range(TEAR_S * int(sim._tps)):
        sim.step()

    def tear(rm) -> int:
        return int(g.gas[TEARGAS][_box(rm)].astype(np.int64).sum())
    vol = graph.air_volume_of()
    ducts = {rm.id: {d for d, _r, _s in G.vent_specs(rm.tags["vent"])}
             for rm in graph.rooms if "vent" in rm.tags}
    reach, todo = {vol[heads.id]}, [vol[heads.id]]
    while todo:
        v = todo.pop()
        on = set().union(*(ducts.get(r.id, set()) for r in graph.rooms
                           if vol[r.id] == v))
        for r in graph.rooms:
            if vol[r.id] not in reach and ducts.get(r.id, set()) & on:
                reach.add(vol[r.id])
                todo.append(vol[r.id])
    bunks = [rm for rm in graph.rooms if rm.tags.get("role") == "bunk"
             and vol[rm.id] == vol[heads.id]]
    assert bunks
    assert all(tear(rm) > 0 for rm in bunks), [(rm.id, tear(rm))
                                               for rm in bunks]
    off = [rm for rm in graph.rooms if vol[rm.id] not in reach]
    assert any(rm.tags.get("role") == "bunk" for rm in off)
    assert all(tear(rm) == 0 for rm in off), [(rm.id, tear(rm))
                                              for rm in off if tear(rm)]
