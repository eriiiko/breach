"""#126 -- match recording and replay by re-simulation (``src/match_replay.py``,
the sim's intent tap, ``tools/run_match.py --record / --replay``).

Properties: a recorded match replays to the same checkpoints and the same
final field / tick digests; recording changes nothing about the match; a
recording is refused on another level, seed or spec by its header; a tampered
intent is caught at the next checkpoint, naming it; the format round-trips
every intent kind exactly; the tap records every entry point as issued and
refuses orders; ``replay`` is registered but no match file can name it.
(Attaching no log changes no digest: ``tests/test_zombie_control.py``'s
pre-#122 pins run with the tap dormant and must stay green.)

Levels are written through ``level_lib``; matches run through ``run_match``.
"""
from __future__ import annotations

import os
import subprocess
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

import control_source  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
import match_replay as mr  # noqa: E402
from control_source import ControlSource, quantize_stick_direction  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.match import MatchSpec, MatchTeam, SpawnRoom  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.orders import ORDER_MOVE, Order  # noqa: E402
from simulation.ruleset import MatchRules  # noqa: E402

BOTS = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                 controller="bots")
ROOMS = [
    SpawnRoom(team=0, x=2, y=2, w=8, h=8, max_per_team=4),
    SpawnRoom(team=1, x=30, y=14, w=8, h=8, max_per_team=4),
    SpawnRoom(team=ZOMBIE_TEAM, x=12, y=2, w=8, h=8, max_per_team=4),
]
BOX_SPEC = MatchSpec(teams=(BOTS, BOTS), seed=3, zombies=2,
                     time_limit_s=15.0)


def _box(tmp_path, name, spec=BOX_SPEC, *, w=40, h=24):
    lvl = tmp_path / name
    lvl.mkdir()
    level_lib.write_level_header(lvl, name=name, tile_size_m=1.0)
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (w, h)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", ROOMS)
    level_lib.write_match(lvl, spec)
    return level_loader.load(name, str(tmp_path))


def _copy(rec):
    return mr.decode_recording(mr.encode_recording(rec))


# ---------------------------------------------------------------------------
# (a) record a seeded ship_demo bots-vs-bots match, replay it headless
# ---------------------------------------------------------------------------
SHIP_TICKS_S = 10.0     # 240 ticks: checkpoints at 0, 60, 120, 180, 240
EVERY = 60


@pytest.fixture(scope="module")
def ship():
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    spec = replace(level.match, time_limit_s=SHIP_TICKS_S)
    rec_run = run_match(level, spec, record=True,
                        checkpoint_every_ticks=EVERY)
    return level, spec, rec_run


def test_a_recorded_ship_demo_match_replays_to_every_checkpoint(ship,
                                                                tmp_path):
    """Property: the ship_demo bots-vs-bots match (zombies in), recorded,
    written, read back and replayed headless, matches every checkpoint and
    ends with the original run's field_digest and tick_digest. Breaks if the
    tap misses an intent or its order, an argument is stored inexactly, the
    replay spawns through another path (unit ids), or anything in the match
    draws from a source the seed does not fix."""
    level, spec, rec_run = ship
    rec = rec_run.recording
    assert sorted(rec.checkpoints) == list(range(0, rec.end_tick + 1, EVERY))
    assert rec.n_intents() > 0
    path = tmp_path / "ship.replay.gz"
    nbytes = mr.write_recording(rec, path)
    assert nbytes == path.stat().st_size
    back = mr.read_recording(path)
    o = run_match(level, replay=back)
    assert o.checkpoints_verified == len(rec.checkpoints)
    assert (o.result, o.field_digest, o.tick_digest) == \
        (rec_run.result, rec_run.field_digest, rec_run.tick_digest)


def test_recording_changes_nothing_about_the_match(ship):
    """Property: the same match run without a recording ends at the same
    digests -- the tap and the checkpoints only read. Breaks if attaching
    the log or taking a checkpoint writes sim state."""
    level, spec, rec_run = ship
    plain = run_match(level, spec)
    assert plain.recording is None
    assert (plain.result, plain.field_digest, plain.tick_digest) == \
        (rec_run.result, rec_run.field_digest, rec_run.tick_digest)


def test_a_recording_is_refused_on_another_seed_level_or_spec(ship,
                                                              tmp_path):
    """Property: the header refuses a replay on another seed, another level
    or another spec before any tick runs. Breaks if replay silently
    re-simulates a different match (which would only fail at a checkpoint,
    or never, on a level with no units in it)."""
    level, spec, rec_run = ship
    rec = rec_run.recording
    with pytest.raises(mr.ReplayRefused, match="seed"):
        run_match(level, seed=rec.seed + 1, replay=rec)
    with pytest.raises(mr.ReplayRefused, match="level"):
        run_match(_box(tmp_path, "other"), replay=rec)
    with pytest.raises(mr.ReplayRefused, match="spec"):
        run_match(level, replace(spec, time_limit_s=11.0), replay=rec)


def test_a_tampered_move_direction_diverges_at_the_next_checkpoint(ship):
    """Property: flipping ONE recorded move direction mid-match raises
    ReplayDivergence at the first checkpoint after it, naming that tick.
    Breaks if the checkpoints do not cover unit state, or a replay keeps
    going past a mismatch."""
    level, spec, rec_run = ship
    rec = _copy(rec_run.recording)
    t = next(t for t in sorted(rec.intents) if t > EVERY
             and any(k == "set_move_dir" and (a[1] or a[2])
                     for k, a in rec.intents[t]))
    rows = rec.intents[t]
    i = next(i for i, (k, a) in enumerate(rows)
             if k == "set_move_dir" and (a[1] or a[2]))
    uid, dx, dy, mode = rows[i][1]
    rows[i] = ("set_move_dir", (uid, -dx, -dy, mode))
    nxt = min(c for c in rec.checkpoints if c > t)
    with pytest.raises(mr.ReplayDivergence) as e:
        run_match(level, replay=rec)
    assert e.value.tick == nxt


# ---------------------------------------------------------------------------
# (b) every intent kind, live: grenades, USE and the horde
# ---------------------------------------------------------------------------
class _ThrowerControl(ControlSource):
    """Test controller: aims at the nearest enemy, holds the trigger, lobs a
    grenade at it (fixed, awkward fuse; with and without a reach) and
    presses USE -- the intents the bots use rarely or never."""

    def __init__(self, team):
        self.team = team

    def handle_frame(self, sim, renderer=None):
        foes = [u for u in sim.units if u.alive and u.team != self.team]
        for u in sim.units:
            if not (u.alive and u.team == self.team) or not foes:
                continue
            f = min(foes, key=lambda f: ((f.x - u.x) ** 2 + (f.y - u.y) ** 2,
                                         f.id))
            dx, dy, _ = quantize_stick_direction(f.x - u.x, f.y - u.y, 0.0)
            sim.set_aim(u.id, dx, dy)
            sim.set_trigger(u.id, sim.tick > 20 and sim.tick % 7 < 4)
            if sim.tick % 40 == 5:
                sim.throw_grenade_intent(
                    u.id, dx, dy, 0.1 + 0.2,
                    None if u.id % 2 else 6 * 65536)
            if sim.tick % 25 == 3:
                sim.use_intent(u.id)
            if sim.tick % 9 == 0:
                sim.set_move_dir(u.id, dx, dy, 1)
            elif sim.tick % 9 == 5:
                sim.clear_move_dir(u.id)


@pytest.fixture
def thrower(monkeypatch):
    monkeypatch.setitem(control_source._MATCH_CONTROLLERS, "test_thrower",
                        _ThrowerControl)
    return "test_thrower"


def test_every_intent_kind_replays_live(tmp_path, thrower):
    """Property: a match issuing every recordable kind (move, clear, aim,
    trigger, throw with and without a reach, USE, the horde's ZOMBIE intent)
    replays to every checkpoint and the original final digests. Breaks if
    any kind is fed back with another argument, out of order, or dropped
    (checked: dropping the throws diverges at tick 15). USE has no effect in
    this doorless box, so only its recording and feeding are live here; its
    exact encoding is the format test's."""
    t = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                  controller=thrower)
    lvl = _box(tmp_path, "throwbox", replace(BOX_SPEC, teams=(t, t)))
    a = run_match(lvl, record=True, checkpoint_every_ticks=5)
    # the frags lobbed at tick 5 (fuse 0.3 s) end this match: it outlives them
    assert a.result.ticks > 5 + 8, a.result
    kinds = {k for v in a.recording.intents.values() for k, _ in v}
    assert kinds == set(mr.INTENT_KINDS), kinds
    b = run_match(lvl, replay=_copy(a.recording))
    assert b.checkpoints_verified == len(a.recording.checkpoints)
    assert (b.result, b.field_digest, b.tick_digest) == \
        (a.result, a.field_digest, a.tick_digest)


# ---------------------------------------------------------------------------
# (c) the format
# ---------------------------------------------------------------------------
def _synthetic():
    rec = mr.Recording(level="box", spec=BOX_SPEC, seed=3,
                       checkpoint_every_ticks=2, build="test")
    rec.intents = {
        0: [("set_move_dir", (0, 65536, -46341, 1)),
            ("clear_move_dir", (1,)),
            ("set_aim", (0, -2 ** 40, 7)),
            ("set_trigger", (0, True)),
            ("set_trigger", (1, False))],
        1: [("throw_grenade_intent", (0, 3, -4, 0.1 + 0.2, None)),
            ("throw_grenade_intent", (1, 3, -4, 2.5e-300, 393216)),
            ("use_intent", (1,)),
            ("drive_zombies", ([9, 4, 7],)),
            ("drive_zombies", ([],))],
    }
    rec.checkpoints = {0: ("f0", "k0"), 2: ("f2", "k2")}
    rec.end_tick = 2
    rec.result = {"outcome": "draw", "team": None, "ticks": 2}
    return rec


def test_the_format_round_trips_every_intent_kind_exactly():
    """Property: encode -> decode reproduces every intent kind, argument and
    order exactly (the float fuse bit for bit, ints beyond 2^53, bools as
    bools, a None reach), and the bytes are deterministic. Breaks if a value
    goes through a lossy type, or the file embeds a timestamp."""
    rec = _synthetic()
    data = mr.encode_recording(rec)
    assert data == mr.encode_recording(rec)
    back = mr.decode_recording(data)
    assert back == rec
    fuse = back.intents[1][0][1][3]
    assert type(fuse) is float and fuse.hex() == (0.1 + 0.2).hex()
    for kind, args in (x for v in back.intents.values() for x in v):
        for typ, a in zip(mr.INTENT_KINDS[kind], args):
            if typ == "bool":
                assert type(a) is bool
            elif typ == "int":
                assert type(a) is int


@pytest.mark.parametrize("mutate, needle", [
    (lambda h, rows: h.update(version=mr.FORMAT_VERSION + 1), "version"),
    (lambda h, rows: h.update(format="other"), "not a"),
    (lambda h, rows: rows[0]["i"].append(["teleport", 0]), "unknown intent"),
    (lambda h, rows: rows[0]["i"].append(["set_trigger", 0, 1]), "bool"),
    (lambda h, rows: rows[0]["i"].append(["use_intent", True]), "int"),
    (lambda h, rows: rows.pop(), "no end row"),
])
def test_the_reader_refuses_a_malformed_recording(mutate, needle):
    """Property: the reader refuses another format or version, an unknown
    kind, a mistyped argument and a truncated file, naming the problem.
    Breaks if a replay could run on input it misread."""
    import gzip
    import json
    lines = gzip.decompress(mr.encode_recording(_synthetic())).splitlines()
    head, rows = json.loads(lines[0]), [json.loads(x) for x in lines[1:]]
    mutate(head, rows)
    raw = b"".join(json.dumps(r).encode() + b"\n" for r in [head, *rows])
    with pytest.raises(mr.ReplayRefused, match=needle):
        mr.decode_recording(gzip.compress(raw))


# ---------------------------------------------------------------------------
# (d) the tap, the registry, the setting
# ---------------------------------------------------------------------------
def test_the_tap_records_every_entry_point_as_issued(tmp_path):
    """Property: with a log attached, each of the intent entry points is
    recorded once, at the tick it was issued, in issue order, with its
    arguments normalised as the sim normalises them -- even for an unknown
    unit (the call is what is recorded, not its effect); detached, nothing
    is recorded; apply_action orders are refused. Breaks if an entry point
    bypasses the tap, or the tap records after (or instead of) acting."""
    level = _box(tmp_path, "tapbox")
    spec = level.match
    sim = Simulation(level, seed=spec.seed, breach_physics=bp,
                     enable_recorder=False, ruleset=MatchRules.from_spec(spec))
    sim.spawn_match(spec)
    sim.hand_zombies_to_controller()
    rec = mr.Recording(level="tapbox", spec=spec, seed=spec.seed,
                       checkpoint_every_ticks=60)
    sim.attach_intent_log(mr.IntentLog(rec))
    sim.set_move_dir(np.int64(0), np.int32(65536), 0, 1)
    sim.clear_move_dir(1)
    sim.set_aim(0, 0, -65536)
    sim.set_trigger(0, np.bool_(True))
    sim.throw_grenade_intent(0, 65536, 0, 2.0)
    sim.use_intent(999)
    sim.drive_zombies((u.id for u in sim.zombies()))
    got = rec.intents[0]
    assert [k for k, _ in got] == ["set_move_dir", "clear_move_dir",
                                   "set_aim", "set_trigger",
                                   "throw_grenade_intent", "use_intent",
                                   "drive_zombies"]
    assert got[0][1] == (0, 65536, 0, 1) and type(got[0][1][0]) is int
    assert got[3][1] == (0, True) and type(got[3][1][1]) is bool
    assert got[4][1] == (0, 65536, 0, 2.0, None)
    assert got[6][1] == ([u.id for u in sim.zombies()],)
    with pytest.raises(mr.RecordingRefused, match="apply_action"):
        sim.apply_action(0, Order(ORDER_MOVE, 5, 5, 0))
    sim.attach_intent_log(None)
    sim.set_aim(0, 65536, 0)
    assert len(rec.intents[0]) == 7


def test_replay_is_registered_but_no_match_file_can_name_it(tmp_path):
    """Property: ``replay`` is a registered controller, built only from a
    recording, and the match door refuses it. Breaks if a match.toml could
    name a controller that cannot be built from a team alone."""
    assert "replay" in control_source.registered_controllers()
    assert "replay" not in control_source.match_file_controllers()
    with pytest.raises(ValueError, match="recording"):
        control_source.create_controller("replay", 0)
    c = control_source.create_controller("replay", None,
                                         recording=_synthetic())
    assert isinstance(c, mr.ReplayControl)
    t = replace(BOTS, controller="replay")
    with pytest.raises(ValueError, match="registered controller"):
        _box(tmp_path, "named", replace(BOX_SPEC, teams=(t, t)))


@pytest.mark.parametrize("section, needle", [
    (None, "no \\[replay\\]"),
    ({}, "missing"),
    ({"checkpoint_every_ticks": 0}, ">= 1"),
    ({"checkpoint_every_ticks": 60.0}, "int"),
    ({"checkpoint_every_ticks": True}, "int"),
])
def test_replay_settings_refuse_a_missing_or_mistyped_key(section, needle):
    """Property: ``[replay] checkpoint_every_ticks`` is required and an int
    >= 1; the shipped value is accepted. Breaks if a typo in config.toml
    fell back to a silent default."""
    from types import SimpleNamespace
    from config import CFG
    assert mr.replay_settings(CFG) >= 1
    cfg = SimpleNamespace() if section is None else \
        SimpleNamespace(replay=SimpleNamespace(**section))
    with pytest.raises(ValueError, match=needle):
        mr.replay_settings(cfg)


# ---------------------------------------------------------------------------
# (e) the CLI: record in one process, replay in another
# ---------------------------------------------------------------------------
def _cli(*args):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHASHSEED", None)
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_match.py"), *args],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)


def test_the_cli_records_and_replays_across_processes(tmp_path):
    """Property: ``run_match.py --record`` in one process and ``--replay``
    in another give the same final digests with every checkpoint matched
    (exit 0), and a recording of another seed is refused (exit 2). Breaks
    if anything in a match depends on per-process state (hash
    randomisation, object ids) or the CLI loses an argument."""
    _box(tmp_path, "clibox")
    path = tmp_path / "m.replay.gz"
    a = _cli("--level", "clibox", "--levels-dir", str(tmp_path),
             "--record", str(path))
    assert a.returncode == 0, a.stdout + a.stderr
    assert "recorded: " in a.stdout and path.exists()
    b = _cli("--replay", str(path), "--levels-dir", str(tmp_path))
    assert b.returncode == 0, b.stdout + b.stderr
    assert "replay: every checkpoint matched" in b.stdout

    def digests(out):
        return [ln for ln in out.splitlines()
                if ln.startswith(("field_digest: ", "tick_digest: ",
                                  "result: ", "ticks: "))]
    assert digests(a.stdout) == digests(b.stdout) and len(digests(a.stdout)) == 4
    c = _cli("--replay", str(path), "--levels-dir", str(tmp_path),
             "--seed", "99")
    assert c.returncode == 2 and "seed" in c.stderr, c.stdout + c.stderr
