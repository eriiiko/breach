"""#143 -- ``tools/e2e_drive.py --profile`` (``tools/frame_profile.py``).

The frame profiler is the driver's own flag set, not a second launcher of
``main.py``. These tests hold its pure parts (argv splitting, the frame
bookkeeping, the report); the window run itself is the driver's job and is
not repeated here (a hidden-window run is slow, and ``e2e_drive``'s own
tests cover that the driver still runs frames).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "tools", ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import frame_profile as fp  # noqa: E402


def test_profile_flags_are_consumed_and_the_rest_passes_through():
    """PROPERTY: ``--profile``, ``--profile-skip``, ``--glfinish`` and
    ``--profile-inner`` are the driver's own and never reach main.py's argv;
    everything else keeps its order. Breaks if a profiler flag leaks through
    (main.py would see an unknown argument) or an ordinary one is eaten."""
    opts, rest = fp.split_profile_args(
        ["--level", "x", "--profile", "--profile-skip", "7", "--control", "match",
         "--glfinish", "--profile-inner", "a.b:C.d;e:F.g", "--profile-inner", "h:I.j"])
    assert rest == ["--level", "x", "--control", "match"]
    assert opts == {"skip": 7, "glfinish": True,
                    "inner": ["a.b:C.d", "e:F.g", "h:I.j"]}


def test_without_profile_nothing_is_consumed_and_a_stray_subflag_is_refused():
    """PROPERTY: with no profiler flag the argv comes back untouched and the
    options are None (the driver behaves as before); a profiler sub-flag
    without ``--profile`` is refused by name instead of silently doing
    nothing. Breaks if a sub-flag is accepted alone."""
    argv = ["--level", "x", "--press", "M@3"]
    assert fp.split_profile_args(argv) == (None, argv)
    for stray in (["--glfinish"], ["--profile-skip", "3"], ["--profile-inner", "a:B.c"]):
        with pytest.raises(SystemExit):
            fp.split_profile_args(argv + stray)
    with pytest.raises(SystemExit):
        fp.split_profile_args(["--profile", "--profile-skip", "-1"])
    with pytest.raises(SystemExit):
        fp.split_profile_args(["--profile", "--profile-skip", "soon"])


def test_driver_import_does_not_load_the_profiler():
    """PROPERTY: importing ``e2e_drive`` does not import ``frame_profile`` --
    the profiler is loaded only when a profiler flag is given, so a plain
    driver run is byte for byte what it was. Breaks if ``e2e_drive`` imports it
    at module level."""
    code = ("import sys; sys.path.insert(0, 'tools'); import e2e_drive; "
            "print('frame_profile' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                         capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "False", out.stderr


def test_parse_inner_wants_module_class_method():
    """PROPERTY: an inner-timer spec is ``module:Class.method``; anything else
    is refused by name. Breaks if a malformed spec is accepted."""
    assert fp.parse_inner("renderer.lighting:LightingPass.upload") == (
        "renderer.lighting", "LightingPass", "upload")
    for bad in ("nomodule", "m:NoMethod", "m:.x", ":C.m", "m:C."):
        with pytest.raises(ValueError):
            fp.parse_inner(bad)


def test_frames_inside_the_skip_window_are_not_measured(monkeypatch):
    """PROPERTY: the first ``skip`` frames contribute to no number -- neither a
    frame time, nor a step count -- and each later frame is measured from one
    hook call to the next. Breaks if the skip is off by one or the frame time
    is taken from the wrong pair of calls."""
    clock = [0.0]
    monkeypatch.setattr(fp.time, "perf_counter", lambda: clock[0])
    prof = fp.FrameProfiler(skip=2)
    for n in range(1, 7):                       # frames 1..6 begin
        prof.on_frame(n)
        prof._cur_steps = n                     # steps seen during frame n
        clock[0] += 0.010 * n                   # frame n lasts 10 ms * n
    prof.on_frame(7)                            # closes frame 6
    # frames 3..6 measured: 30, 40, 50, 60 ms
    assert [round(t, 6) for t in prof.frame_times] == [0.03, 0.04, 0.05, 0.06]
    assert prof.steps_per_frame == [3, 4, 5, 6]


def test_report_has_the_scratch_format_and_adds_up():
    """PROPERTY: the report keeps the layout of the session script that
    measured #143 (header, frame ms line, steps line, a table sorted by cost,
    ``(everything else in the loop)``), its fps and shares follow from the
    inputs, and a nested ``--profile-inner`` timer is shown but not
    subtracted from "everything else". Breaks if a column or line moves, the
    fps is mis-derived, or an inner label is double-counted."""
    frame_times = [0.1] * 10                    # 10 frames of 100 ms -> 10 fps
    acc = {"sim.step": 0.5, "r.compose_world": 0.2, "  > Inner.m": 0.1}
    calls = {"sim.step": 20, "r.compose_world": 10, "  > Inner.m": 10}
    text = fp.format_report(["--level", "x"], True, 40, acc, calls, frame_times,
                            [2] * 10, inner_labels={"  > Inner.m"}, tps=20.0)
    lines = text.splitlines()
    assert lines[1] == "=== frame_profile: --level x  glfinish=True"
    assert lines[2] == "measured frames 10  (skipped 40);  sim steps 20"
    assert "-> 10.0 fps" in lines[3] and "mean 100.0" in lines[3]
    assert lines[4].startswith("sim steps per frame: mean 2.00  max 2;")
    assert "sim pace = 1.00x real time" in lines[4]   # 20 steps / 20 tps / 1.0 s
    assert lines[5].split() == ["part", "ms/frame", "%", "frame", "calls", "ms/call"]
    names = [ln[:34].strip() for ln in lines[6:-1]]
    assert names == ["sim.step", "r.compose_world", "> Inner.m"]
    assert lines[6].split()[1:] == ["50.00", "50.0", "20", "25.00"]
    # everything else = 1.0 s total - 0.5 - 0.2 (the inner 0.1 is inside them)
    assert lines[-1].split()[-2:] == ["30.00", "30.0"]


def test_report_without_measured_frames_says_so():
    """PROPERTY: a run shorter than the skip window reports that nothing was
    measured instead of raising. Breaks if an empty sample divides by zero."""
    assert "no measured frames" in fp.format_report([], False, 40, {}, {}, [], [])
