"""#127 -- ``main.py --replay PATH [--seek TICK]``: the window plays a recorded
match by re-simulation (``src/control_match.py::ReplayObserver``).

Properties: the replay level is the recording's header (``--level`` only may
agree with it); a seek reaches its target by re-simulation, rebuilding from
tick 0 only when the target is behind; the observer, driven as main.py drives
it, ends on the recording's own final digests whether it played straight,
seeked forward or rewound; a divergence stops it without raising; a rebuilt
sim's changed tiles are exactly the ones the picture must repatch; main.py
refuses the flags a replay cannot honour before opening a window; and the
real main.py in a hidden window (``tools/e2e_drive.py``) ends a replay on the
recording's final digests.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import level_loader  # noqa: E402
import match_replay as mr  # noqa: E402
from control_match import ReplayObserver, replay_level_name, seek_plan  # noqa: E402
from renderer.tile_art import changed_between  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import Simulation  # noqa: E402

LEVELS = ROOT / "levels"
EVERY = 30
LIMIT_S = 5.0          # 120 ticks: checkpoints 0, 30, 60, 90, 120


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    level = level_loader.load("ship_demo", str(LEVELS))
    spec = replace(level.match, time_limit_s=LIMIT_S)
    o = run_match(level, spec, record=True, checkpoint_every_ticks=EVERY)
    path = tmp_path_factory.mktemp("rec") / "short.replay.gz"
    mr.write_recording(o.recording, path)
    return path, mr.read_recording(path)


def _observer(rec, seek_tick=0):
    level = level_loader.load(rec.level, str(LEVELS))

    def make_sim(seed, rules):
        return Simulation(level_loader.load(rec.level, str(LEVELS)),
                          seed=seed, breach_physics=bp,
                          enable_recorder=False, ruleset=rules)

    obs = ReplayObserver(level, rec, make_sim=make_sim, seek_tick=seek_tick)
    sim = make_sim(obs.initial_seed(42), obs.initial_ruleset())
    sim.set_paused(obs.starts_paused())
    obs.spawn_units(sim, level)
    return obs


def _play(obs, max_ticks=10_000):
    """main.py's tick loop body: on_tick, then step, until the sim pauses."""
    sim = obs.sim
    for _ in range(max_ticks):
        if sim.is_paused():
            break
        obs.on_tick(sim)
        sim.step()
    return sim


def _end_digests(rec):
    return tuple(rec.checkpoints[rec.end_tick])


# ---------------------------------------------------------------------------
# Pure parts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("current", [0, 1, 37, 60, 119, 120])
@pytest.mark.parametrize("target", [-5, 0, 1, 37, 60, 119, 120, 500])
def test_seek_plan_reaches_its_target_and_restarts_only_backward(current,
                                                                 target):
    """Property: the plan's target is the request clamped to [0, end]; it
    restarts exactly when that target is behind the sim, and the steps it
    takes land on the target from where it starts (tick 0 after a restart).
    Breaks if a forward seek rebuilds (wasted re-simulation), a backward one
    does not (impossible: checkpoints hold no state), or a target past the
    recording's end is stepped beyond it."""
    end = 120
    restart, t, n = seek_plan(current, target, end)
    assert t == max(0, min(target, end))
    assert restart == (t < current)
    assert (0 if restart else current) + n == t and n >= 0


def test_the_replay_level_is_the_recordings_and_level_may_only_agree(
        recorded):
    """Property: with no --level the header names the level folder; the same
    folder is accepted, any other refused naming both. Breaks if --level
    could load another map under a recording (every checkpoint would then
    diverge, or worse, a same-sized map replay silently)."""
    _, rec = recorded
    assert replay_level_name(rec) == rec.level == "ship_demo"
    assert replay_level_name(rec, "ship_demo") == "ship_demo"
    with pytest.raises(mr.ReplayRefused, match="playground"):
        replay_level_name(rec, "playground")


def test_a_rebuilt_maps_changed_tiles_are_exactly_the_repatched_ones():
    """Property: ``changed_between`` lists the tiles whose material or vacuum
    state differs, and nothing else; two maps of the same load differ
    nowhere. Breaks if a backward seek leaves the old sim's destroyed walls
    drawn destroyed (a tile missed) or repaints the grid (a tile too many)."""
    level = level_loader.load("ship_demo", str(LEVELS))
    a = Simulation(level_loader.load("ship_demo", str(LEVELS)), seed=1,
                   breach_physics=bp, enable_recorder=False)
    b = Simulation(level, seed=1, breach_physics=bp, enable_recorder=False)
    assert changed_between(a.gmap, b.gmap) == []
    solid = [(int(y), int(x)) for y, x in zip(*a.gmap.solid.nonzero())]
    wall = next(t for t in solid if 0 < t[0] < a.gmap.solid.shape[0] - 1
                and 0 < t[1] < a.gmap.solid.shape[1] - 1)
    a.gmap.destroy_wall(*wall)
    changed = changed_between(a.gmap, b.gmap)
    assert wall in changed
    assert all(a.gmap.material[t] != b.gmap.material[t]
               or a.gmap.is_vacuum[t] != b.gmap.is_vacuum[t]
               for t in changed)


# ---------------------------------------------------------------------------
# The observer, driven headless as main.py drives it
# ---------------------------------------------------------------------------
def test_a_straight_replay_ends_on_the_recordings_digests(recorded):
    """Property: played tick by tick through on_tick + step, the observer
    verifies every checkpoint, pauses at the recording's end and holds its
    final digests; the HUD names the last verified checkpoint. Breaks if the
    window path spawns or drives the match differently from run_match."""
    _, rec = recorded
    obs = _observer(rec)
    sim = _play(obs)
    assert obs.diverged is None and obs.ended
    assert sim.tick == rec.end_tick and sim.is_paused()
    assert obs.final_digests == _end_digests(rec)
    assert obs.controllers[0].checkpoints_verified == len(rec.checkpoints)
    assert ("checkpoint %d verified" % rec.end_tick, "ok") in \
        obs.hud_lines(sim)


def test_seeking_forward_then_back_re_simulates_the_same_match(recorded):
    """Property: a forward seek keeps the sim; a backward one rebuilds it
    at tick 0; either way the match then ends on the recording's digests,
    and a restart leaves the pause state as it was. Breaks if a seek skips
    a tick's intents, steps a tick twice, or a rebuild reuses mutated
    state (the level, the rules)."""
    _, rec = recorded
    obs = _observer(rec, seek_tick=45)
    first = obs.sim
    sim = obs.seek(70, progress=None)
    assert sim is first and sim.tick == 70 and obs.last_verified == 60
    sim.set_paused(True)
    sim = obs.seek(obs.seek_tick, progress=None)
    assert sim is not first and sim is obs.sim
    assert sim.tick == 45 and sim.is_paused()
    assert obs.generation >= 2
    sim.set_paused(False)
    _play(obs)
    assert obs.diverged is None and obs.final_digests == _end_digests(rec)
    # seeking past the end lands on the end and reports it
    obs.restart()
    sim = obs.seek(10_000, progress=None)
    assert sim.tick == rec.end_tick and obs.ended
    assert obs.final_digests == _end_digests(rec)


def test_a_divergence_stops_the_replay_without_raising(recorded):
    """Property: a recording whose intents were tampered with diverges at
    the next checkpoint: the observer records the tick, pauses the sim,
    keeps it paused on Space, and shows DIVERGED -- it never raises (the
    window stays open). Breaks if ReplayDivergence escapes the observer."""
    _, rec = recorded
    bad = mr.decode_recording(mr.encode_recording(rec))
    t = min(t for t, v in bad.intents.items()
            if t > 0 and any(k == "set_move_dir" for k, _ in v))
    v = bad.intents[t]
    i = next(n for n, (k, _) in enumerate(v) if k == "set_move_dir")
    k, a = v[i]
    v[i] = (k, (a[0], -a[1] if a[1] else 65536, a[2], a[3]))
    obs = _observer(bad)
    sim = _play(obs)
    assert obs.diverged is not None and obs.diverged.tick > t
    assert sim.is_paused() and obs.final_digests is None
    obs.toggle_pause(sim)
    assert sim.is_paused()
    assert (f"DIVERGED at tick {obs.diverged.tick}", "bad") in \
        obs.hud_lines(sim)


# ---------------------------------------------------------------------------
# main.py itself
# ---------------------------------------------------------------------------
def _run(args, timeout=300):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, *args], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=timeout)


@pytest.mark.parametrize("extra, needle", [
    (["--level", "playground"], "playground"),
    (["--control", "match"], "--control"),
    (["--warp", "2"], "--warp"),
    (["--res", "2"], "--res"),
])
def test_main_refuses_what_a_replay_cannot_honour(recorded, extra, needle):
    """Property: main.py --replay exits non-zero, naming the flag, before
    any window opens, when asked for another level, another controller, a
    warp or a resampled grid. Breaks if one of them is silently ignored and
    the window then shows a match that is not the recorded one."""
    path, _ = recorded
    r = _run(["main.py", "--replay", str(path), *extra], timeout=120)
    assert r.returncode != 0
    assert needle in r.stdout + r.stderr
    assert "Window:" not in r.stdout


def test_main_seek_without_replay_is_refused():
    """Property: --seek alone is refused (it has nothing to seek). Breaks if
    a stray --seek starts a normal game."""
    r = _run(["main.py", "--seek", "10"], timeout=120)
    assert r.returncode != 0 and "--seek" in r.stdout + r.stderr


def test_the_window_replay_ends_on_the_recordings_digests(recorded):
    """Property (E2E, the real main.py in a hidden window through
    tools/e2e_drive.py): --replay with --seek past the middle plays to the
    end and prints the recording's own final field and tick digests with
    every checkpoint matched, exit 0. Breaks if the windowed path (light
    requested, renderer reading the sim) changes the match in any way."""
    path, rec = recorded
    r = _run(["tools/e2e_drive.py", "--replay", str(path), "--seek",
              str(rec.end_tick - 10), "--frames", "90"], timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    f, k = _end_digests(rec)
    assert f"field_digest: {f}" in r.stdout
    assert f"tick_digest: {k}" in r.stdout
    assert f"replay: every checkpoint matched ({len(rec.checkpoints)})" \
        in r.stdout
