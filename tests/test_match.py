"""#119 P2 — the match foundation: match.toml, spawn rooms, MatchRules, the
controller registry and the headless runner (``tools/run_match.py``).

Every property here is checked on levels written through ``level_lib``'s
writers (never a hand-written toml), and every match is run through
``run_match`` (the one match loop).
"""
from __future__ import annotations

import os
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

import level_lib  # noqa: E402
import level_loader  # noqa: E402
import control_source  # noqa: E402
from control_source import ControlSource, quantize_stick_direction  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.match import (  # noqa: E402
    FORMATION_FOOTPRINT, MatchSpec, MatchTeam, SpawnRoom, formation_capacity,
    formation_cells, parse_match, spawn_room_from_dict,
)
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from run_match import run_match  # noqa: E402

TPS = 24  # [clock] ticks_per_second (asserted against CFG below)

IDLE = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                 controller="idle")


def _write_level(tmp_path, name, rooms, spec, *, w=40, h=24):
    """An open hull box written through the level_lib writers."""
    lvl = tmp_path / name
    lvl.mkdir()
    level_lib.write_level_header(lvl, name=name, tile_size_m=1.0)
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (w, h)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", rooms)
    if spec is not None:
        level_lib.write_match(lvl, spec)
    return level_loader.load(name, str(tmp_path))


# Two marine rooms far apart, a zombie room right next to team 0.
ROOMS = [
    SpawnRoom(team=0, x=2, y=2, w=8, h=8, max_per_team=4),
    SpawnRoom(team=1, x=30, y=14, w=8, h=8, max_per_team=4),
    SpawnRoom(team=ZOMBIE_TEAM, x=12, y=2, w=8, h=8, max_per_team=4),
]


def _spec(*, teams=(IDLE, IDLE), zombies=0, seed=1, time_limit_s=60.0):
    return MatchSpec(teams=tuple(teams), seed=seed, zombies=zombies,
                     time_limit_s=time_limit_s)


class _TriggerControl(ControlSource):
    """Test controller: every living unit of its team aims at the nearest
    living enemy and holds the trigger — the fire cone draws from sim.rng,
    so the match seed matters."""

    def __init__(self, team):
        self.team = team

    def handle_frame(self, sim, renderer=None):
        assert renderer is None
        foes = [u for u in sim.units if u.alive and u.team != self.team]
        for u in sim.units:
            if not (u.alive and u.team == self.team) or not foes:
                continue
            t = min(foes, key=lambda f: ((f.x - u.x) ** 2 + (f.y - u.y) ** 2,
                                         f.id))
            dx, dy, _ = quantize_stick_direction(t.x - u.x, t.y - u.y, 0.0)
            sim.set_aim(u.id, dx, dy)
            sim.set_trigger(u.id, True)


@pytest.fixture
def trigger_controller(monkeypatch):
    monkeypatch.setitem(control_source._MATCH_CONTROLLERS, "test_trigger",
                        _TriggerControl)
    return "test_trigger"


def _consistent(outcome, spec):
    """The result agrees with the living slots it reports."""
    r = outcome.result
    alive = sorted(t for t in range(len(spec.teams)) if outcome.living[t])
    if r.outcome == "win":
        assert alive == [r.team], (r, outcome.living)
    else:
        assert r.outcome == "draw" and r.team is None
        assert len(alive) == 0 or (
            len(alive) >= 2 and r.ticks == round(spec.time_limit_s * TPS)), \
            (r, outcome.living)


def test_tick_rate_matches_config():
    from config import CFG
    assert int(CFG.clock.ticks_per_second) == TPS


# ---------------------------------------------------------------------------
# (a) determinism — the P2 seed gate
# ---------------------------------------------------------------------------
def test_one_seed_one_digest_another_seed_differs(tmp_path,
                                                  trigger_controller):
    """Property: a match is a pure function of its config + seed — two runs
    of one seed end in the same digest, and the seed is LIVE (another seed
    moves the digest when shots draw from sim.rng). Breaks if anything but
    the sim's seeded rng draws randomness (an OS-seeded draw makes run 1 !=
    run 2), or if the match seed never reaches Simulation(seed=) (seed 1 ==
    seed 2)."""
    shooter = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                        controller=trigger_controller)
    lvl = _write_level(tmp_path, "duel", ROOMS,
                       _spec(teams=(shooter, shooter), time_limit_s=4.0))
    a = run_match(lvl)
    b = run_match(lvl)
    c = run_match(lvl, seed=2)
    assert a.seed == 1 and c.seed == 2
    assert (a.tick_digest, a.field_digest, a.result) == \
        (b.tick_digest, b.field_digest, b.result)
    assert c.tick_digest != a.tick_digest


# ---------------------------------------------------------------------------
# (b) + (c) MatchRules
# ---------------------------------------------------------------------------
def test_idle_teams_with_zombies_end_in_a_consistent_result(tmp_path):
    """Property: two idle teams beside a zombie room end in win or draw,
    and the result agrees with the living slots (a win names the only team
    with a living slot; a draw has none or hits the limit). Breaks if
    zombies count as a team, if a converted marine still counts for its old
    team, or if result() and is_terminal() disagree."""
    spec = _spec(zombies=4, time_limit_s=60.0)
    lvl = _write_level(tmp_path, "horde", ROOMS, spec)
    o = run_match(lvl)
    assert o.result.outcome in ("win", "draw")
    _consistent(o, spec)
    # Not vacuous: the horde beside team 0 reached it and converted marines
    # (who then left team 0 -- the match ended on team 0 having none).
    assert o.converted_alive > 0
    assert o.result.ticks < round(spec.time_limit_s * TPS)


def test_idle_teams_without_zombies_draw_exactly_at_the_limit(tmp_path):
    """Property: with nobody able to die, the match ends as a draw on the
    limit tick — not before, not after. Breaks if the limit is off by a tick
    (a `>` for `>=`, a float compare), or if the ruleset ends a two-team
    fight early (ContinuousRealtime's "no zombie alive" rule leaking in)."""
    spec = _spec(time_limit_s=2.0)
    lvl = _write_level(tmp_path, "standoff", ROOMS[:2], spec)
    o = run_match(lvl)
    assert o.result.outcome == "draw" and o.result.team is None
    assert o.result.ticks == 2 * TPS
    assert o.living == {0: [0, 1], 1: [0, 1]}


# ---------------------------------------------------------------------------
# (d) formation
# ---------------------------------------------------------------------------
def _footprint(x, y):
    return {(x + dx, y + dy) for dy in range(FORMATION_FOOTPRINT)
            for dx in range(FORMATION_FOOTPRINT)}


@pytest.mark.parametrize("w,h", [(5, 5), (8, 8), (13, 17), (14, 13),
                                 (23, 9)])
def test_formation_offsets_depend_on_the_room_shape_only(w, h):
    """Property: slot k's offset from its room's corner is the same for any
    team and any room position of that shape — two equal rooms give
    symmetric starts; footprints stay inside the room and never overlap.
    Breaks if placement reads the team, the room's position, or n (a slot
    must not move because a team grew), or packs cells closer than a
    footprint."""
    cap = formation_capacity(SpawnRoom(0, 0, 0, w, h, 1))
    assert cap >= 1
    rel = None
    for team, (x, y) in ((0, (3, 4)), (1, (40, 7)), (ZOMBIE_TEAM, (11, 30))):
        room = SpawnRoom(team, x, y, w, h, cap)
        cells = formation_cells(room, cap)
        offs = [(cx - x, cy - y) for cx, cy in cells]
        if rel is None:
            rel = offs
        assert offs == rel
        # a smaller team keeps the same cells for its slots
        assert formation_cells(room, 1) == cells[:1]
        tiles = set()
        for cx, cy in cells:
            fp = _footprint(cx, cy)
            assert all(x <= tx < x + w and y <= ty < y + h for tx, ty in fp)
            assert not (fp & tiles)
            tiles |= fp
    with pytest.raises(ValueError):
        formation_cells(SpawnRoom(0, 0, 0, w, h, 1), cap + 1)


def test_playground_match_spawns_symmetric_and_disjoint():
    """Property: on the playground, the two marine rooms (equal shape) put
    every slot at the same offset from its room, every unit stands inside
    its room, and no two footprints overlap. Breaks if a room is moved or
    resized asymmetrically, or spawn_match ignores the placement rule."""
    import breach_physics as bp
    from simulation import Simulation
    lvl = level_loader.load("playground")
    rooms = {r.team: r for r in lvl.spawn_rooms}
    assert rooms[0].w == rooms[1].w and rooms[0].h == rooms[1].h
    full = MatchTeam(n=rooms[0].max_per_team, loadout="rifleman",
                     env_profile="default", controller="idle")
    spec = MatchSpec(teams=(full, full), seed=1,
                     zombies=rooms[ZOMBIE_TEAM].max_per_team)
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    units = sim.spawn_match(spec)
    offsets = {}
    tiles = set()
    for u in units:
        r = rooms[u.team]
        offsets.setdefault(u.team, []).append((u.x - r.x, u.y - r.y))
        fp = set(map(tuple, u.occupied_tiles()))
        assert all(r.x <= tx < r.x + r.w and r.y <= ty < r.y + r.h
                   for tx, ty in fp), u.name
        assert not (fp & tiles), u.name
        tiles |= fp
    assert offsets[0] == offsets[1]
    assert [u.name for u in units[:2]] == ["T0-0", "T0-1"]
    assert [u.slot for u in sim.roster(1)] == list(range(full.n))
    assert all(u.is_zombie for u in sim.roster(ZOMBIE_TEAM))


def test_spawn_match_needs_a_fresh_sim(tmp_path):
    """Property: slots start at 0 — spawn_match refuses a sim that already
    holds units. Breaks if a second spawn shifts every slot."""
    import breach_physics as bp
    from simulation import Simulation
    lvl = _write_level(tmp_path, "fresh", ROOMS, _spec())
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    sim.spawn_match()
    with pytest.raises(ValueError):
        sim.spawn_match()


# ---------------------------------------------------------------------------
# (e) the door
# ---------------------------------------------------------------------------
def _raw(**over):
    raw = {"seed": 1, "zombies": 0, "time_limit_s": 10.0,
           "objective": "eliminate",
           "team": [dict(n=2, loadout="rifleman", env_profile="default",
                         controller="idle") for _ in range(2)]}
    raw.update(over)
    return raw


def _team(**over):
    row = dict(n=2, loadout="rifleman", env_profile="default",
               controller="idle")
    row.update(over)
    return [row, dict(n=2, loadout="rifleman", env_profile="default",
                      controller="idle")]


@pytest.mark.parametrize("raw,needle", [
    (_raw(objective="bomb_site"), "objective"),
    (_raw(team=_team(n=5)), "max_per_team"),
    (_raw(team=_team(controller="nobody")), "controller"),
    (_raw(team=_team(loadout="bazooka")), "loadout"),
    (_raw(team=_team(env_profile="spacesuit_x")), "env_profile"),
    (_raw(zombies=5), "max_per_team"),
    (_raw(team=_raw()["team"][:1]), "two"),
    (_raw(time_limit_s=0), "time_limit_s"),
    (_raw(colour="red"), "unknown"),
])
def test_match_door_refuses(raw, needle):
    """Property: the match door refuses every value it cannot run — an
    unknown objective, controller, loadout or profile, n or zombies over the
    room's max_per_team, fewer than two teams, a non-positive limit, an
    unknown key. Breaks if any of these is accepted silently (a typo'd
    loadout would arm the default; n > max would spawn into a wall)."""
    with pytest.raises(ValueError, match=needle):
        parse_match(raw, ROOMS, "test")
    # and the good one passes
    parse_match(_raw(), ROOMS, "test")


@pytest.mark.parametrize("row,needle", [
    (dict(team=0, x=1, y=1, w=8, h=8, max_per_team=4, placement="scatter"),
     "placement"),
    (dict(team=0, x=1, y=1, w=7, h=7, max_per_team=4), "too small"),
    (dict(team=0, x=1, y=1, w=8, h=8), "max_per_team"),
    (dict(team=0, x=1, y=1, w=8, h=8, max_per_team=4, colour=1), "unknown"),
])
def test_spawn_room_door_refuses(row, needle):
    """Property: a spawn room with an unknown placement rule, too small for
    its max_per_team, missing a key or carrying an unknown one is refused.
    Breaks if a second rule is accepted before it exists, or a room promises
    slots it cannot place."""
    with pytest.raises(ValueError, match=needle):
        spawn_room_from_dict(row, "test")


def test_loader_refuses_a_room_off_open_floor(tmp_path):
    """Property: the loader refuses a spawn room whose placement cells are
    not open floor (here: over the hull). Breaks if a match can spawn a unit
    inside a wall."""
    bad = [SpawnRoom(team=0, x=0, y=0, w=8, h=8, max_per_team=4)]
    with pytest.raises(ValueError, match="open interior floor"):
        _write_level(tmp_path, "wall", bad, None)


def test_writer_refuses_what_the_loader_refuses(tmp_path):
    """Property: write_match never writes a file its loader would refuse —
    no match.toml appears. Breaks if the writer skips the door."""
    lvl = _write_level(tmp_path, "w", ROOMS, None)
    bad = _spec(teams=(IDLE, MatchTeam(n=2, loadout="rifleman",
                                       env_profile="default",
                                       controller="nobody")))
    with pytest.raises(ValueError, match="controller"):
        level_lib.write_match(lvl.path, bad)
    assert not (lvl.path / level_loader.MATCH_FILENAME).exists()


# ---------------------------------------------------------------------------
# (f) round trip
# ---------------------------------------------------------------------------
def test_match_and_rooms_round_trip(tmp_path):
    """Property: what the writers write, the loader reads back equal, and
    re-formatting the loaded values is byte-identical. Breaks if a field is
    dropped, retyped (time_limit_s float, seed int), or defaulted on the way
    through."""
    spec = _spec(zombies=3, seed=17, time_limit_s=42.5,
                 teams=(IDLE, MatchTeam(n=3, loadout="rifleman",
                                        env_profile="default",
                                        controller="idle"), IDLE))
    rooms = ROOMS + [SpawnRoom(team=2, x=20, y=14, w=8, h=8, max_per_team=4)]
    lvl = _write_level(tmp_path, "rt", rooms, spec)
    assert lvl.spawn_rooms == rooms
    assert lvl.match == spec
    text = (lvl.path / level_loader.MATCH_FILENAME).read_text("utf-8")
    assert level_lib.format_match_text(lvl.match) == text
    assert level_lib.format_spawn_room_lines(lvl.spawn_rooms) == \
        level_lib.format_spawn_room_lines(rooms)
    # a level without a match file carries none
    assert _write_level(tmp_path, "nomatch", ROOMS, None).match is None


def test_spawn_rooms_leave_the_authored_spawns_alone():
    """Property: the playground's [[spawn]] squad and horde are what they
    were before spawn rooms existed (the game is unchanged), and the zombie
    room is the glass pen the horde already stands in. Breaks if writing the
    rooms rewrote or reordered the [[spawn]] block."""
    lvl = level_loader.load("playground")
    assert [(s.name, s.team, s.x, s.y) for s in lvl.spawns[:4]] == [
        ("Alpha", 0, 8.0, 42.0), ("Bravo", 0, 13.0, 42.0),
        ("Cobra", 0, 8.0, 48.0), ("Delta", 0, 13.0, 48.0)]
    pen = {r.team: r for r in lvl.spawn_rooms}[ZOMBIE_TEAM]
    for s in lvl.spawns:
        if s.team == ZOMBIE_TEAM:
            assert pen.x <= s.x and s.x + s.footprint <= pen.x + pen.w
            assert pen.y <= s.y and s.y + s.footprint <= pen.y + pen.h


# ---------------------------------------------------------------------------
# controllers
# ---------------------------------------------------------------------------
def test_controller_registry():
    """Property: `idle` is registered and builds a ControlSource that runs
    with renderer=None and emits nothing; a taken name is refused. Breaks if
    a registration silently overrides another controller."""
    assert "idle" in control_source.registered_controllers()
    c = control_source.create_controller("idle", 0)
    assert isinstance(c, ControlSource)
    assert c.handle_frame(object(), None) is None
    with pytest.raises(ValueError):
        control_source.register_controller("idle", lambda team: c)
    with pytest.raises(ValueError):
        control_source.create_controller("nobody", 0)


# ---------------------------------------------------------------------------
# (g) the CLI
# ---------------------------------------------------------------------------
_CLI = r"""
import runpy, sys
sys.argv = ["run_match.py", "--level", "playground", "--ticks-per-report",
            "720"]
try:
    runpy.run_path("tools/run_match.py", run_name="__main__")
    rc = 0
except SystemExit as e:
    rc = e.code
leaked = sorted(m for m in sys.modules
                if m == "pyray" or m == "raylib" or m.split(".")[0] == "renderer")
if leaked:
    print("RENDERER IMPORTED:", leaked)
    rc = 3
sys.exit(rc)
"""


def test_run_match_cli_on_the_playground():
    """Property: `tools/run_match.py --level playground` runs a full match
    to a result headless and exits 0, printing the result, the ticks, the
    living slots and the digests, without importing the renderer or raylib.
    Breaks if the runner needs a window, requests light through the
    renderer, or the playground's match config stops loading."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", _CLI], cwd=str(ROOT),
                          env=env, capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    for key in ("result: ", "ticks: ", "living team 0: ", "living team 1: ",
                "living zombies: ", "field_digest: ", "tick_digest: "):
        assert key in out, out
