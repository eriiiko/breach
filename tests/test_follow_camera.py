"""#128 -- the follow camera (``renderer/follow_camera.py``) of a watched match
(``main.py --replay`` / ``--control match``).

Properties: the action target is a point inside the living units' convex
hull, pulled toward a cluster that fights (events weigh more than density)
and, with no fight, toward the biggest crowd; a dead unit is never a target;
the cycle walks action -> every living unit (team, slot) -> free -> action;
the smoothing approaches monotonically, never past the target, and holds
still inside its dead zone; a manual pan frees the camera; the settings door
refuses a bad value; and the real main.py (``tools/e2e_drive.py``) replays a
recording with the camera off (0, 0), on the action.
"""
from __future__ import annotations

import math
import os
import random
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from renderer.camera import Camera2D  # noqa: E402
from renderer.follow_camera import (  # noqa: E402
    ACTION, FREE, ActionTracker, FollowCamera, FollowSettings,
    action_target, cycle_order, follow_settings, smooth_step, unit_centre)
from simulation.events import UnitHitEvent, UnitKilledEvent  # noqa: E402

SETTINGS = FollowSettings(camera="action", time_constant_s=0.6,
                          dead_zone_tiles=1.5, action_memory_s=3.0,
                          action_event_weight=6.0, action_radius_tiles=10.0)
DT_TICK = 1.0 / 24.0


def _unit(uid, x, y, team=0, slot=0, alive=True, name=None):
    return SimpleNamespace(id=uid, x=float(x), y=float(y), footprint=3,
                           alive=alive, team=team, slot=slot,
                           name=name or f"T{team}-{slot}")


def _inside_hull(p, pts, eps=1e-9):
    """p lies in the convex hull of pts: no direction separates it (checked
    on 720 directions -- the hull's support function bounds p in each)."""
    for k in range(720):
        a = math.pi * k / 360.0
        c, s = math.cos(a), math.sin(a)
        proj = [x * c + y * s for x, y in pts]
        v = p[0] * c + p[1] * s
        if not (min(proj) - eps <= v <= max(proj) + eps):
            return False
    return True


def _target(units, heat=None):
    return action_target(units, heat or {},
                         event_weight=SETTINGS.action_event_weight,
                         radius_tiles=SETTINGS.action_radius_tiles)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------------------
# Where the action is
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(12))
def test_the_action_target_lies_inside_the_living_units_hull(seed):
    """Property: whatever the crowd and the fight heat, the target is inside
    the convex hull of the LIVING units' centres -- dead units far outside
    it change nothing. Breaks if the target is a raw event position (a
    corpse's tile, a grenade's landing), an extrapolation, or counts the
    dead."""
    rng = random.Random(seed)
    units = [_unit(i, rng.uniform(0, 90), rng.uniform(0, 60), slot=i)
             for i in range(rng.randint(1, 9))]
    dead = [_unit(100 + i, rng.uniform(300, 400), rng.uniform(300, 400),
                  alive=False) for i in range(3)]
    heat = {u.id: rng.choice([0.0, 0.0, rng.uniform(0, 5)])
            for u in units + dead}
    t = _target(units + dead, heat)
    assert _inside_hull(t, [unit_centre(u) for u in units])
    assert t == pytest.approx(_target(units, heat))


def test_with_no_fight_the_target_goes_to_the_biggest_crowd():
    """Property: with no events the target sits nearer the bigger of two
    clusters than the smaller. Breaks if the density weight is dropped (a
    plain mean sits in the empty corridor between them, or nearer neither)."""
    big = [_unit(i, 10 + i % 3, 10 + i // 3, slot=i) for i in range(6)]
    small = [_unit(10 + i, 70 + i, 50, team=1, slot=i) for i in range(2)]
    t = _target(big + small)
    assert _dist(t, unit_centre(big[0])) < _dist(t, unit_centre(small[0]))
    plain = (sum(unit_centre(u)[0] for u in big + small) / 8,
             sum(unit_centre(u)[1] for u in big + small) / 8)
    assert _dist(t, unit_centre(big[0])) < _dist(plain, unit_centre(big[0]))


def test_the_target_moves_to_a_smaller_cluster_that_fights():
    """Property: a fight in the SMALLER cluster pulls the target to it --
    after two seconds of hits there the target is nearer the fight than the
    idle crowd; events weigh more than density. And the heat decays: three
    memory spans after the last hit the target is back at the crowd.
    Breaks if events are ignored, under-weighted, or never forgotten."""
    crowd = [_unit(i, 10 + i % 3, 10 + i // 3, slot=i) for i in range(6)]
    a, b = _unit(20, 70, 50, team=1, slot=0), _unit(21, 74, 50, team=2,
                                                    slot=0)
    units = crowd + [a, b]
    tracker = ActionTracker(SETTINGS.action_memory_s)
    before = _target(units, tracker.heat)
    assert _dist(before, unit_centre(crowd[0])) < _dist(before,
                                                        unit_centre(a))
    for tick in range(48):
        ev = [UnitHitEvent(unit_id=b.id, damage=5, source="bullet",
                           attacker_id=a.id)] if tick % 6 == 0 else []
        tracker.observe(ev, DT_TICK)
    during = _target(units, tracker.heat)
    assert _dist(during, unit_centre(a)) < _dist(during, unit_centre(crowd[0]))
    for _ in range(int(3 * SETTINGS.action_memory_s / DT_TICK)):
        tracker.observe([], DT_TICK)
    after = _target(units, tracker.heat)
    assert _dist(after, unit_centre(crowd[0])) < _dist(after, unit_centre(a))


def test_a_kill_heats_its_killer_and_the_dead_carry_no_weight():
    """Property: a UnitKilledEvent heats the killer (attacker_id >= 0) and
    the victim; the environment (-1) is never a key; the dead victim's heat
    moves nothing. Breaks if kills are ignored or -1 becomes a unit."""
    tracker = ActionTracker(3.0)
    tracker.observe([UnitKilledEvent(unit_id=5, killed_by="bullet",
                                     attacker_id=2),
                     UnitHitEvent(unit_id=7, damage=1, source="heat")],
                    DT_TICK)
    assert set(tracker.heat) == {2, 5, 7}


# ---------------------------------------------------------------------------
# What is followed
# ---------------------------------------------------------------------------
def test_cycling_walks_action_every_living_unit_then_free():
    """Property: from action, C visits every LIVING unit once in (team,
    slot) order, then free, then action again; dead ones are skipped.
    Breaks if the order follows the sim's list order, a dead unit is
    visited, or free / action is skipped."""
    units = [_unit(1, 0, 0, team=1, slot=1), _unit(2, 5, 0, team=0, slot=2),
             _unit(3, 9, 0, team=0, slot=0),
             _unit(4, 9, 9, team=0, slot=1, alive=False),
             _unit(5, 1, 1, team=255, slot=0, name="Z-0"),
             _unit(6, 2, 2, team=1, slot=0)]
    fc = FollowCamera(SETTINGS)
    seen = [fc.label(units)]
    for _ in range(7):
        fc.cycle(units)
        seen.append(fc.label(units))
    assert seen == ["action", "T0-0", "T0-2", "T1-0", "T1-1", "Z-0", "free",
                    "action"]
    assert [u.id for u in cycle_order(units)] == [3, 2, 6, 1, 5]


def test_a_followed_unit_that_dies_falls_back_to_action():
    """Property: the moment the followed unit is dead the target is action
    (label, position and the next cycle step all agree) and the camera
    never moves toward the corpse. Breaks if a dead unit stays a target."""
    units = [_unit(1, 10, 10), _unit(2, 50, 40, team=1)]
    fc = FollowCamera(SETTINGS)
    fc.cycle(units)
    assert fc.label(units) == "T0-0"
    units[0].alive = False
    assert fc.label(units) == "action" and fc.target == ACTION
    assert fc.target_pos(units) == pytest.approx(unit_centre(units[1]))
    fc.cycle(units)
    assert fc.label(units) == "T1-0"


def test_a_manual_pan_frees_the_camera_until_cycled():
    """Property: after a manual pan the target is free and apply() leaves
    the camera exactly where the player put it, frame after frame; the cycle
    key takes it back to action. Breaks if the follow fights the player."""
    units = [_unit(1, 60, 40), _unit(2, 64, 40, team=1)]
    cam = Camera2D(0.0, 0.0, 20.0, 400, 300, 100, 70)
    fc = FollowCamera(SETTINGS)
    fc.apply(cam, units, 1 / 60)
    assert (cam.pos_tile_x, cam.pos_tile_y) != (0.0, 0.0)
    fc.manual_pan()
    cam.pan(-30, -30)
    held = (cam.pos_tile_x, cam.pos_tile_y)
    for _ in range(30):
        fc.apply(cam, units, 1 / 60)
    assert fc.label(units) == "free"
    assert (cam.pos_tile_x, cam.pos_tile_y) == held
    fc.cycle(units)
    assert fc.label(units) == "action"
    fc.apply(cam, units, 1 / 60)
    assert (cam.pos_tile_x, cam.pos_tile_y) != held


def test_the_follow_stays_in_the_world_and_never_zooms():
    """Property: following a unit in a corner keeps the camera inside the
    world (Camera2D's clamp) and its zoom unchanged. Breaks if the follow
    writes the position past the edge or touches the zoom."""
    units = [_unit(1, 97, 67)]
    cam = Camera2D(0.0, 0.0, 20.0, 400, 300, 100, 70)
    fc = FollowCamera(SETTINGS)
    fc.cycle(units)
    for _ in range(200):
        fc.apply(cam, units, 1 / 30)
    w, h = cam.visible_tiles()
    assert cam.pos_tile_x == pytest.approx(100 - w)
    assert cam.pos_tile_y == pytest.approx(70 - h)
    assert cam.zoom_px_per_tile == 20.0


# ---------------------------------------------------------------------------
# How it moves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_smoothing_converges_monotonically_without_overshoot(seed):
    """Property: at any frame rate, each step shortens the distance to the
    target (never lengthens it), stays on the near side of it (no
    overshoot), and within ten time constants ends inside the dead zone.
    Breaks if the gain is >= 1, frame-rate dependent beyond the exp law, or
    the dead zone is ignored."""
    rng = random.Random(seed)
    target = (rng.uniform(0, 90), rng.uniform(0, 60))
    pos = (rng.uniform(0, 90), rng.uniform(0, 60))
    d0 = (target[0] - pos[0], target[1] - pos[1])
    t = 0.0
    while t < 10 * SETTINGS.time_constant_s:
        dt = rng.choice([1 / 144, 1 / 60, 1 / 30, 0.1, 0.25])
        new = smooth_step(pos, target, dt, SETTINGS)
        assert _dist(new, target) <= _dist(pos, target) + 1e-12
        assert (target[0] - new[0]) * d0[0] + (target[1] - new[1]) * d0[1] \
            >= -1e-9
        pos, t = new, t + dt
    assert _dist(pos, target) <= SETTINGS.dead_zone_tiles + 0.05


def test_smoothing_holds_still_inside_the_dead_zone():
    """Property: a target within the dead zone moves the camera not at all,
    however long the frame. Breaks if a standing target makes it jitter."""
    pos = (40.0, 30.0)
    for dx, dy in [(0, 0), (1.0, 0), (0.6, -1.0), (-1.06, 1.06)]:
        target = (pos[0] + dx, pos[1] + dy)
        assert smooth_step(pos, target, 0.5, SETTINGS) == pos


# ---------------------------------------------------------------------------
# Settings and HUD
# ---------------------------------------------------------------------------
def _cfg(**over):
    base = dict(checkpoint_every_ticks=60, camera="action",
                time_constant_s=0.6, dead_zone_tiles=1.5,
                action_memory_s=3.0, action_event_weight=6.0,
                action_radius_tiles=10.0)
    base.update(over)
    return SimpleNamespace(replay=SimpleNamespace(
        **{k: v for k, v in base.items() if v is not None}))


@pytest.mark.parametrize("over, needle", [
    (dict(camera="unit"), "camera must be one of"),
    (dict(camera=None), "missing 'camera'"),
    (dict(camera=1), "camera must be one of"),
    (dict(time_constant_s=0), "time_constant_s must be a number > 0"),
    (dict(time_constant_s="1"), "time_constant_s"),
    (dict(dead_zone_tiles=-1.0), "dead_zone_tiles must be a number >= 0"),
    (dict(action_memory_s=None), "missing 'action_memory_s'"),
    (dict(action_event_weight=True), "action_event_weight"),
    (dict(action_radius_tiles=0.0), "action_radius_tiles"),
])
def test_the_settings_door_refuses_a_bad_value(over, needle):
    """Property: every follow key is required and typed; ``camera`` is
    exactly "action" or "free"; the shipped config passes. Breaks if a typo
    in config.toml fell back to a silent default."""
    from config import CFG
    s = follow_settings(CFG)
    assert s.camera in ("action", "free")
    assert follow_settings(_cfg()).camera == "action"
    assert follow_settings(_cfg(camera="free")).camera == "free"
    with pytest.raises(ValueError, match=needle):
        follow_settings(_cfg(**over))


def test_a_free_start_follows_nothing():
    """Property: ``camera = "free"`` starts on free (the camera stays where
    main.py put it). Breaks if the setting is ignored."""
    fc = FollowCamera(replace(SETTINGS, camera="free"))
    cam = Camera2D(0.0, 0.0, 20.0, 400, 300, 100, 70)
    fc.apply(cam, [_unit(1, 60, 40)], 1 / 60)
    assert fc.label([]) == FREE and (cam.pos_tile_x, cam.pos_tile_y) == (0, 0)


# ---------------------------------------------------------------------------
# The window: the real main.py replays with the camera on the action
# ---------------------------------------------------------------------------
LIMIT_S = 30.0         # 720 ticks of ship_demo: the fight starts ~tick 600


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    import level_loader
    import match_replay as mr
    from run_match import run_match
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    spec = replace(level.match, time_limit_s=LIMIT_S)
    o = run_match(level, spec, record=True, checkpoint_every_ticks=60)
    path = tmp_path_factory.mktemp("rec") / "follow.replay.gz"
    mr.write_recording(o.recording, path)
    return path


def test_the_replay_hud_names_what_the_camera_follows(recorded):
    """Property: the replay HUD carries the ``cam: ...`` line and the cycle
    key in its hint when a camera line is given, and neither without.
    Breaks if the HUD stops telling the player what is followed."""
    import level_loader
    import match_replay as mr
    import breach_physics as bp
    from control_match import ReplayObserver
    from simulation import Simulation
    rec = mr.read_recording(recorded)
    level = level_loader.load(rec.level, str(ROOT / "levels"))
    obs = ReplayObserver(level, rec)
    sim = Simulation(level, seed=obs.initial_seed(0), breach_physics=bp,
                     enable_recorder=False, ruleset=obs.initial_ruleset())
    obs.spawn_units(sim, level)
    with_cam = [t for t, _ in obs.hud_lines(sim, cam="cam: T0-3")]
    assert "cam: T0-3" in with_cam and "C camera" in with_cam[-1]
    plain = [t for t, _ in obs.hud_lines(sim)]
    assert not any("cam:" in t or "C camera" in t for t in plain)


def test_the_window_replay_follows_the_action(recorded, tmp_path):
    """Property: the real main.py (hidden window, tools/e2e_drive.py)
    replays a ship_demo recording from tick 600 with ``camera = "action"``:
    it exits 0, saves the mid-match shot, and the camera it reports is off
    the (0, 0) corner it starts in, following the action; one C press
    moves it to a unit (the first living one, by team and slot). Breaks if the follow is not
    wired into the window's frame (the camera stays at 0, 0) or the cycle
    key is not."""
    shot = tmp_path / "follow.png"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(
        [sys.executable, "tools/e2e_drive.py", "--replay", str(recorded),
         "--seek", "600", "--frames", "41", "--shot", f"{shot}@40"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert shot.exists()
    m = re.search(r"camera at centre \(([\d.]+), ([\d.]+)\) pos "
                  r"\(([\d.]+), ([\d.]+)\) cam: (\S+)", r.stdout)
    assert m, r.stdout[-3000:]
    assert (float(m.group(3)), float(m.group(4))) != (0.0, 0.0)
    assert m.group(5) == "action"

    r2 = subprocess.run(
        [sys.executable, "tools/e2e_drive.py", "--replay", str(recorded),
         "--seek", "600", "--frames", "12", "--press", "C@5"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=600)
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    m2 = re.search(r"camera: (T\d+-\d+)\s*$", r2.stdout, re.M)
    assert m2, r2.stdout[-3000:]
    assert f"cam: {m2.group(1)}" in r2.stdout
