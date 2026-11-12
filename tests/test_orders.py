"""#195 1f -- THE ORDER: four unit columns written only through the
commander's verb ``Simulation.order_intent`` (``simulation.unit_orders``:
the door, slot 1b', the expiry after the tick advance, the clears at death
and conversion), ``room_graph.place_at``, format v8, and the bots' team
layer as the commander (``control_bots.BotCommander``) for a team driven
through the door only (``control_source.DOOR_CONTROLLERS``).

Note: ``docs/rl_agent_rung1_note_1f_2026-11-12.md``. Every test's docstring
names the property it protects and the change that must break it.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import control_bots  # noqa: E402
import control_policy  # noqa: E402
import control_source  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
import match_replay as mr  # noqa: E402
from control_source import ControlSource  # noqa: E402
from field_ab_harness import capture_snapshot  # noqa: E402
from field_digest import field_digest  # noqa: E402
from level_loader import LevelData  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import unit_orders as O  # noqa: E402
from simulation import unit_schema as us  # noqa: E402
from simulation.ai_zombie import convert_marines_to_zombies  # noqa: E402
from simulation.damage import KINETIC, DamagePacket, apply_packet  # noqa: E402
from simulation.match import MatchSpec, MatchTeam, SpawnRoom  # noqa: E402
from simulation.match_session import MatchSession  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.room_graph import place_at  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.unit import Unit  # noqa: E402

ORDER_KEYS = set(O.ORDER_FIELDS)
H, W = 20, 40
WALL_X = 20
DOOR_Y = (8, 12)            # the opening's rows, y in [8, 12)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _graph_rows():
    """Two rooms split by a wall at x = 20 and one door opening through it
    (rows 8..11): room 0 the west half, room 1 the east, the opening's
    tiles and the walls -1."""
    rooms = [dict(id=0, x=1, y=1, w=WALL_X - 1, h=H - 2, kind="room"),
             dict(id=1, x=WALL_X + 1, y=1, w=W - WALL_X - 2, h=H - 2,
                  kind="room")]
    openings = [dict(id=0, a=0, b=1, kind="door", x=WALL_X, y=DOOR_Y[0],
                     w=1, h=DOOR_Y[1] - DOOR_Y[0])]
    grid = np.full((H, W), -1, dtype=np.int32)
    grid[1:H - 1, 1:WALL_X] = 0
    grid[1:H - 1, WALL_X + 1:W - 1] = 1
    return rooms, openings, grid


def _tilemap():
    tm = np.full((H, W), MAT_HULL, dtype=np.int32)
    tm[1:H - 1, 1:W - 1] = MAT_AIR
    tm[:, WALL_X] = MAT_HULL
    tm[DOOR_Y[0]:DOOR_Y[1], WALL_X] = MAT_AIR
    return tm


def _level(graph=True) -> LevelData:
    rooms, openings, grid = _graph_rows()
    rg = (level_loader.parse_room_graph(rooms, openings, grid, (H, W),
                                        "test") if graph else None)
    return LevelData(name="orderbox", version="1", path=Path("."),
                     tilemap=_tilemap(), tile_size_m=1.0,
                     diffuse_path=Path("."), rooms=rg)


def _sim(graph=True) -> Simulation:
    sim = Simulation(_level(graph), seed=5, breach_physics=None,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _marine(sim, cx, cy, team=0, name="M"):
    uid = sim.add_unit(Unit(name, x=cx - 1, y=cy - 1, team=team))
    return sim.get_unit(uid)


def _step(sim, n=1):
    for _ in range(n):
        sim.step()
        sim.set_paused(False)


def _rec(u) -> dict:
    return us.digest_record(u)


def _order_keys(u) -> set:
    return set(_rec(u)) & ORDER_KEYS


class _Tap:
    def __init__(self):
        self.calls = []

    def record(self, tick, kind, args):
        self.calls.append((tick, kind, args))


# ---------------------------------------------------------------------------
# A. the columns and their digest
# ---------------------------------------------------------------------------
def test_an_order_is_hashed_only_while_held():
    """Property: a fresh unit and a unit whose order was cleared add no
    ``order_*`` key to the unit record; a held order adds ``order_what`` and
    ``order_where``, and ``order_until`` / ``order_target`` only when they
    are not 'until told otherwise' / 'no target'. Breaks if a column hashes
    unconditionally or compares to the wrong default."""
    sim = _sim()
    a = _marine(sim, 5, 5)
    b = _marine(sim, 30, 5, team=1)
    assert _order_keys(a) == set() and _order_keys(b) == set()
    sim.order_intent(a.id, O.ORDER_HOLD, 0, 0, -1)
    _step(sim)
    assert _order_keys(a) == {"order_what", "order_where"}
    sim.order_intent(a.id, O.ORDER_GO_TO, 1, 50, b.id)
    _step(sim)
    assert _order_keys(a) == ORDER_KEYS
    assert _rec(a)["order_target"] == b.id
    sim.order_intent(a.id, 0, -1, 0, -1)
    _step(sim)
    assert _order_keys(a) == set()


# ---------------------------------------------------------------------------
# B. the verb: latched, applied at slot 1b'
# ---------------------------------------------------------------------------
def test_the_verb_changes_no_unit_record_until_the_tick_applies_it():
    """Property: ``order_intent`` at tick T changes no unit record and no
    field digest by itself (it latches on the sim); the order is on the
    unit after T's step. Breaks if the verb writes the unit directly (the
    ``set_flashlight`` shape the critique's M4 forbids)."""
    sim = Simulation(_level(), seed=5, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    a = _marine(sim, 5, 5)
    _step(sim, 2)
    snap = capture_snapshot(sim)
    before = (field_digest(snap), snap["__unit_state__"]["units"])
    sim.order_intent(a.id, O.ORDER_GO_TO, O.ORDER_OPENING_BASE + 0, 0, -1)
    snap = capture_snapshot(sim)
    assert (field_digest(snap), snap["__unit_state__"]["units"]) == before
    assert O.held(a) == O.NO_ORDER
    _step(sim)
    assert O.held(a) == (O.ORDER_GO_TO, O.ORDER_OPENING_BASE, O.NO_LIMIT,
                         O.NO_TARGET)
    assert sim._order_requests == {}


def test_an_order_holds_its_ticks_and_expires_after_the_advance():
    """Property: an order of ``ticks = d`` issued at tick T is in force for
    the d steps T .. T+d-1: it is held after the steps T .. T+d-2 (between
    ticks a held order always has ``order_until > sim.tick``) and gone after
    step T+d-1's; ``ticks = 0`` is still held after 500 ticks. Breaks on an
    off-by-one in ``order_until`` or the expiry compare (``<`` for ``<=``),
    or an expiry at slot 1b (an order with no time left seen between
    ticks)."""
    sim = _sim()
    a = _marine(sim, 5, 5)
    b = _marine(sim, 10, 5)
    _step(sim, 3)
    d = 7
    t0 = sim.tick
    sim.order_intent(a.id, O.ORDER_HOLD, 0, d, -1)
    sim.order_intent(b.id, O.ORDER_HOLD, 0, 0, -1)
    for k in range(d - 1):
        _step(sim)
        assert a.order_what == O.ORDER_HOLD, k
        assert a.order_until == t0 + d and a.order_until > sim.tick
    _step(sim)
    assert sim.tick == t0 + d
    assert O.held(a) == O.NO_ORDER
    d1 = 1
    sim.order_intent(a.id, O.ORDER_HOLD, 0, d1, -1)
    _step(sim)
    assert O.held(a) == O.NO_ORDER          # one step in force, none after
    _step(sim, 500)
    assert O.held(b) == (O.ORDER_HOLD, 0, O.NO_LIMIT, O.NO_TARGET)


def test_death_and_conversion_clear_the_order():
    """Property: a unit killed while holding an order holds none after that
    step (``damage.apply_packet``'s death); a dead marine that turns into a
    zombie holds none as a zombie (``convert_marines_to_zombies``), even
    when an order was still on the corpse. Breaks if the death clear or the
    conversion clear is missing."""
    sim = _sim()
    a = _marine(sim, 5, 5)
    b = _marine(sim, 10, 5)
    for u in (a, b):
        sim.order_intent(u.id, O.ORDER_GO_TO, 1, 0, -1)
    _step(sim)
    assert a.order_what and b.order_what
    apply_packet(a, DamagePacket(amount=1e9, dtype=KINETIC), source="bullet")
    assert not a.alive and O.held(a) == O.NO_ORDER
    apply_packet(b, DamagePacket(amount=1e9, dtype=KINETIC), source="melee",
                 mark_killed_by_zombie=True)
    _step(sim)                                  # on_tick_end converts it
    assert b.is_zombie and b.alive and O.held(b) == O.NO_ORDER
    # the conversion's own clear: a corpse with an order left on it
    c = _marine(sim, 15, 5)
    c.alive = False
    c.killed_by_zombie = True
    c.order_what, c.order_where = O.ORDER_HOLD, 0
    convert_marines_to_zombies([c])
    assert c.is_zombie and O.held(c) == O.NO_ORDER


def test_none_clears_and_the_last_call_of_a_tick_wins():
    """Property: ``what = 0`` clears a held order (and takes no place); two
    ``order_intent`` calls for one unit in one tick leave the second, whole
    (no field of the first merged in). Breaks if a request merges, or a
    clear needs a place."""
    sim = _sim()
    a = _marine(sim, 5, 5)
    b = _marine(sim, 30, 5, team=1)
    sim.order_intent(a.id, O.ORDER_GO_TO, 1, 40, b.id)
    sim.order_intent(a.id, O.ORDER_HOLD, 0, 0, -1)
    _step(sim)
    assert O.held(a) == (O.ORDER_HOLD, 0, O.NO_LIMIT, O.NO_TARGET)
    sim.order_intent(a.id, O.ORDER_NONE, -1, 0, -1)
    _step(sim)
    assert O.held(a) == O.NO_ORDER


def test_the_door_refuses_by_name_and_a_refused_call_leaves_no_trace():
    """Property: the door refuses, naming the cause, a level without a room
    graph, an unknown / dead / zombie unit, ``what`` 5, a room id past the
    rooms, an opening past the openings, ``ticks`` -1 or ``True``, an
    unknown / dead / teammate target and a field set with ``what = 0``; a
    refused call reaches no tap and latches nothing. Breaks if a door check
    is missing, or the tap records before the door."""
    sim = _sim()
    a = _marine(sim, 5, 5)
    mate = _marine(sim, 10, 5)
    foe = _marine(sim, 30, 5, team=1)
    dead = _marine(sim, 30, 12, team=1)
    dead.alive = False
    zed = _marine(sim, 15, 12)
    zed.is_zombie = True
    tap = _Tap()
    sim.attach_intent_log(tap)
    G = O.ORDER_GO_TO
    cases = [
        ((999, G, 0, 0, -1), ValueError, "does not exist"),
        ((dead.id, G, 0, 0, -1), ValueError, "is dead"),
        ((zed.id, G, 0, 0, -1), ValueError, "zombie"),
        ((a.id, 5, 0, 0, -1), ValueError, "not an order code"),
        ((a.id, G, 2, 0, -1), ValueError, "names no place"),
        ((a.id, G, O.ORDER_OPENING_BASE + 1, 0, -1), ValueError,
         "names no place"),
        ((a.id, G, -1, 0, -1), ValueError, "names no place"),
        ((a.id, G, 0, -1, -1), ValueError, "ticks = -1"),
        ((a.id, G, 0, True, -1), TypeError, "ticks must be an integer"),
        ((a.id, G, 0, 0, 999), ValueError, "target 999 does not exist"),
        ((a.id, G, 0, 0, dead.id), ValueError, "is dead"),
        ((a.id, G, 0, 0, mate.id), ValueError, "teammate"),
        ((a.id, 0, 0, 0, -1), ValueError, "takes no other field"),
        ((a.id, 0, -1, 3, -1), ValueError, "takes no other field"),
        ((a.id, 0, -1, 0, foe.id), ValueError, "takes no other field"),
    ]
    for args, exc, msg in cases:
        with pytest.raises(exc, match=re.escape(msg)):
            sim.order_intent(*args)
    assert tap.calls == [] and sim._order_requests == {}
    bare = _sim(graph=False)
    u = _marine(bare, 5, 5)
    with pytest.raises(ValueError, match="has none"):
        bare.order_intent(u.id, G, 0, 0, -1)
    # legal calls pass: a zombie (ZOMBIE_TEAM) is an enemy target
    zt = _marine(sim, 25, 12)
    zt.team = control_bots.ZOMBIE_TEAM
    zt.is_zombie = True
    sim.order_intent(a.id, G, O.ORDER_OPENING_BASE, 0, zt.id)
    assert [c[1] for c in tap.calls] == ["order_intent"]


# ---------------------------------------------------------------------------
# C. place_at: the one map from a tile to a place
# ---------------------------------------------------------------------------
def test_place_at_names_rooms_openings_and_nothing_on_a_wall():
    """Property: every floor tile of a room names that room; every tile of
    an opening (grid -1) names ``ORDER_OPENING_BASE`` + its id; a wall tile
    and a tile off the map name nothing. Breaks if the commander reads a
    second map, or the opening lookup is missing."""
    lvl = _level()
    g = lvl.rooms
    for y in range(H):
        for x in range(W):
            p = place_at(g, y, x)
            r = int(g.grid[y, x])
            if r >= 0:
                assert p == r
            elif x == WALL_X and DOOR_Y[0] <= y < DOOR_Y[1]:
                assert p == O.ORDER_OPENING_BASE + 0
            else:
                assert p == O.NO_PLACE, (y, x)
    assert place_at(g, -1, 3) == O.NO_PLACE == place_at(g, 3, W)
    assert O.place_name(g, O.ORDER_OPENING_BASE + 0) == "opening 0"
    assert O.place_name(g, 1) == "room 1"


# ---------------------------------------------------------------------------
# D. information only; the one writer
# ---------------------------------------------------------------------------
class _RandomOrders(ControlSource):
    """Every 10 ticks, a random legal order to every living marine (a
    seeded generator of its own: draws nothing from the sim)."""

    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.issued = 0

    def handle_frame(self, sim, renderer=None):
        if sim.tick % 10:
            return
        g = sim.level.rooms
        marines = [u for u in sim.units if u.alive and not u.is_zombie]
        for u in marines:
            what = int(self.rng.integers(0, len(O.ORDER_WHAT)))
            if what == 0:
                sim.order_intent(u.id, 0, -1, 0, -1)
                self.issued += 1
                continue
            n = len(g.rooms) + len(g.openings)
            k = int(self.rng.integers(0, n))
            where = (k if k < len(g.rooms)
                     else O.ORDER_OPENING_BASE + k - len(g.rooms))
            foes = [e.id for e in sim.units if e.alive and e.team != u.team]
            target = (int(self.rng.choice(foes))
                      if foes and self.rng.random() < 0.5 else -1)
            ticks = int(self.rng.integers(0, 30))
            sim.order_intent(u.id, what, where, ticks, target)
            self.issued += 1


def _ship_demo():
    return level_loader.load("ship_demo", str(ROOT / "levels"))


def test_orders_are_information_only_on_a_bots_match():
    """Property: ship_demo's bots match with a controller issuing random
    legal orders to every marine every 10 ticks plays the same match as
    without -- the same result, the same field digest after every tick, and
    every unit record equal bar the ``order_*`` keys, tick for tick. Breaks
    if any rule of the game reads an order."""
    lvl = _ship_demo()
    assert lvl.rooms is not None
    spec = replace(lvl.match, time_limit_s=12.0)
    a = MatchSession(lvl, spec, breach_physics=bp)
    b = MatchSession(lvl, spec, breach_physics=bp)
    rnd = _RandomOrders()
    held_ticks = 0

    def state(ses):
        snap = capture_snapshot(ses.sim)
        units = [{k: v for k, v in r.items() if k not in ORDER_KEYS}
                 for r in snap["__unit_state__"]["units"]]
        return field_digest(snap), units
    while not a.is_terminal():
        rnd.handle_frame(b.sim)
        a.tick()
        b.tick()
        assert state(a) == state(b), a.sim.tick
        held_ticks += any(u.order_what for u in b.sim.units)
        assert b.is_terminal() == a.is_terminal()
    assert a.result() == b.result()
    assert rnd.issued > 20 and held_ticks > 50


def test_only_the_order_modules_name_the_order_columns():
    """Property: no module under ``src/`` names ``order_what`` /
    ``order_where`` / ``order_until`` / ``order_target`` but
    ``unit_orders.py`` (the writer), ``unit_schema.py`` (the columns),
    ``damage.py`` / ``ai_zombie.py`` (the clears), ``simulation.py`` and
    ``control_bots.py`` (the commander's read of the held order) -- so
    nothing reads an order for a decision before rung 4. Breaks if a reader
    appears without a design line."""
    allowed = {"simulation/unit_orders.py", "simulation/unit_schema.py",
               "simulation/damage.py", "simulation/ai_zombie.py",
               "simulation/simulation.py", "control_bots.py"}
    pat = re.compile(r"\border_(what|where|until|target)\b")
    hits = set()
    for p in (ROOT / "src").rglob("*.py"):
        if pat.search(p.read_text(encoding="utf-8")):
            hits.add(p.relative_to(ROOT / "src").as_posix())
    assert hits <= allowed, sorted(hits - allowed)
    assert "simulation/unit_orders.py" in hits


def test_no_policy_row_carries_an_order():
    """Property: ``order_intent`` is in ``EXCLUDED_KINDS``, no action-row
    column belongs to it, and ``random_action_rows`` never yields one (a
    soldier's policy never orders itself). Breaks if the kind joins the
    policy's columns."""
    assert "order_intent" in control_policy.EXCLUDED_KINDS
    assert "order_intent" in mr.INTENT_KINDS
    assert "order_intent" not in control_policy.POLICY_KINDS
    schema = control_policy.action_schema()
    assert not any(c.name.startswith("order_intent")
                   for c in schema.columns)
    rows = control_policy.random_action_rows(np.random.default_rng(1), 64)
    assert rows.shape[1] == len(schema)


# ---------------------------------------------------------------------------
# E. recording: the kind replays; v7 is refused
# ---------------------------------------------------------------------------
def test_a_v7_recording_is_refused_by_its_header(tmp_path):
    """Property: the format is v8 and a recording written as v7 is refused
    by its header, naming the version. Breaks if the bump is missing."""
    assert mr.FORMAT_VERSION == 8
    team = MatchTeam(n=1, loadout="rifleman", env_profile="default",
                     controller="idle")
    rec = mr.Recording(level="box", spec=MatchSpec(teams=(team, team), seed=1),
                       seed=1, checkpoint_every_ticks=2, build="test")
    rec.intents = {0: [("order_intent", (0, 1, 3, 0, -1))]}
    rec.checkpoints = {0: ("a", "b")}
    rec.end_tick = 0
    data = mr.encode_recording(rec)
    assert mr.decode_recording(data).intents == rec.intents
    lines = gzip.decompress(data).decode("utf-8").splitlines()
    head = json.loads(lines[0])
    head["version"] = 7
    old = gzip.compress("\n".join([json.dumps(head)] + lines[1:])
                        .encode("utf-8"))
    with pytest.raises(mr.ReplayRefused, match="format version 7"):
        mr.decode_recording(old)


# ---------------------------------------------------------------------------
# F. the commander, for a door-driven team only
# ---------------------------------------------------------------------------
def _policy_spec(lvl, time_limit_s=10.0):
    """ship_demo with team 0 on ``policy`` (no rows handed: it stands) and
    team 1 on the layered bots, no horde."""
    t0, t1 = lvl.match.teams
    return replace(lvl.match, teams=(replace(t0, controller="policy"), t1),
                   zombies=0, time_limit_s=time_limit_s)


def test_the_commander_orders_a_door_driven_team_and_it_replays():
    """Property: on ship_demo (a room graph) with team 0 on ``policy`` and
    team 1 on ``bots``, the builder adds ONE commander, for team 0; from the
    first team update every living marine of team 0 holds the commander's
    order -- ``what`` = ``PLAY_ORDER[brain.play]``, ``where`` = ``place_at``
    of the objective's centre tile, the brain's target id -- and team 1
    holds none; the recording holds an ``order_intent`` only on a change
    (never two equal consecutive orders for a unit); the recorded match
    replays to every checkpoint and the same final digests. Breaks if the
    translation, the door-driven test, issue-on-change or the replay of the
    kind is wrong."""
    lvl = _ship_demo()
    spec = _policy_spec(lvl)
    ses = MatchSession(lvl, spec, breach_physics=bp)
    cmd = [c for c in ses.controllers
           if isinstance(c, control_bots.BotCommander)]
    assert len(cmd) == 1 and cmd[0].team == 0
    cmd = cmd[0]
    tap = _Tap()
    ses.sim.attach_intent_log(tap)
    checked = 0
    while not ses.is_terminal() and ses.sim.tick < 120:
        ses.tick()
        sim = ses.sim
        mine = [u for u in sim.units if u.team == 0 and u.alive
                and not u.is_zombie]
        if mine and cmd.brain.updated_at == sim.tick - 1:
            want = cmd.order_for(sim, mine)
            brain = cmd.brain
            if want != O.NO_ORDER:
                fp = mine[0].footprint
                ox, oy = brain.objective
                assert want[0] == O.ORDER_WHAT.index(
                    control_bots.PLAY_ORDER[brain.play])
                assert want[1] == place_at(sim.level.rooms, oy + fp // 2,
                                           ox + fp // 2)
                assert want[2] == O.NO_LIMIT
                assert want[3] == (brain.target.id if brain.target else -1)
            for u in mine:
                assert O.held(u) == want, (sim.tick, u.id)
            checked += 1
        for u in sim.units:
            if u.team == 1:
                assert O.held(u) == O.NO_ORDER
    assert checked >= 10
    orders = [(t, a) for t, k, a in tap.calls if k == "order_intent"]
    assert orders and all(a[0] in {u.id for u in ses.sim.units
                                   if u.team == 0} for _t, a in orders)
    last = {}
    for _t, a in orders:
        assert last.get(a[0]) != a[1:], a
        last[a[0]] = a[1:]
    # record and replay the whole match
    rec = run_match(lvl, spec, record=True, checkpoint_every_ticks=20)
    kinds = {k for v in rec.recording.intents.values() for k, _ in v}
    assert "order_intent" in kinds
    again = run_match(lvl, spec,
                      replay=mr.decode_recording(
                          mr.encode_recording(rec.recording)))
    assert again.checkpoints_verified == len(rec.recording.checkpoints)
    assert (again.result, again.field_digest, again.tick_digest) == \
        (rec.result, rec.field_digest, rec.tick_digest)


ROOMS_BOX = [
    SpawnRoom(team=0, x=2, y=2, w=8, h=8, max_per_team=4),
    SpawnRoom(team=1, x=30, y=14, w=8, h=8, max_per_team=4),
    SpawnRoom(team=255, x=12, y=2, w=8, h=8, max_per_team=4),
]


def test_no_commander_without_a_graph_or_a_door_driven_team(tmp_path):
    """Property: a ``policy`` team on a level without a room graph gets no
    commander and its match runs; on ship_demo no match of ``bots`` /
    ``bots_v1`` / ``idle`` teams and the ``zombies`` horde (both bot
    generations, with and without the horde) builds a commander or ever
    calls ``order_intent``. Breaks if the commander rule leaks to a team not
    on the door, or needs no graph."""
    lvl = tmp_path / "bare"
    lvl.mkdir()
    level_lib.write_level_header(lvl, name="bare", tile_size_m=1.0)
    tm = np.full((24, 40), MAT_HULL, dtype=np.int32)
    tm[1:23, 1:39] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (40, 24)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", ROOMS_BOX)
    pol = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                    controller="idle")
    level_lib.write_match(lvl, MatchSpec(teams=(pol, pol), seed=3,
                                         zombies=2, time_limit_s=2.0))
    bare = level_loader.load("bare", str(tmp_path))
    assert bare.rooms is None
    spec = replace(bare.match, teams=(replace(pol, controller="policy"),
                                      pol))
    ses = MatchSession(bare, spec, breach_physics=bp)
    assert not any(isinstance(c, control_bots.BotCommander)
                   for c in ses.controllers)
    while not ses.is_terminal():
        ses.tick()
    ship = _ship_demo()
    for a, b, z in (("bots", "bots_v1", 0), ("bots_v1", "bots", 4),
                    ("bots", "idle", 4), ("idle", "bots_v1", 0)):
        t0, t1 = ship.match.teams
        sp = replace(ship.match, teams=(replace(t0, controller=a),
                                        replace(t1, controller=b)),
                     zombies=z, time_limit_s=4.0)
        ses = MatchSession(ship, sp, breach_physics=bp)
        assert not any(isinstance(c, control_bots.BotCommander)
                       for c in ses.controllers)
        tap = _Tap()
        ses.sim.attach_intent_log(tap)
        while not ses.is_terminal():
            ses.tick()
        assert tap.calls and not any(k == "order_intent"
                                     for _t, k, _a in tap.calls), (a, b)


def test_every_play_names_its_order():
    """Property: ``PLAY_ORDER`` has a row for every ``PLAYS`` row and names
    order kinds only; a fourth play without a row is refused by name by the
    check the import runs. Breaks if a new play silently writes no order."""
    assert set(control_bots.PLAY_ORDER) == {p.name for p in
                                            control_bots.PLAYS}
    control_bots.check_play_order()
    fourth = control_bots.PLAYS + (control_bots.Play("breach",
                                                     lambda t, c: True),)
    with pytest.raises(ValueError, match="breach"):
        control_bots.check_play_order(plays=fourth)
    with pytest.raises(ValueError, match="no order kind"):
        control_bots.check_play_order(
            table=dict(control_bots.PLAY_ORDER, hold="none"))
