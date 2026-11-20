"""tools/frame_profile.py -- where does a watched frame's time go? (#143)

The timer and report half of ``tools/e2e_drive.py --profile``: NOT a launcher.
``e2e_drive`` is THE hidden-window driver of the real ``main.py``; this module
only (a) wraps wall-clock timers around the loop's entry points at CLASS level
-- so ``main.py`` itself is untouched -- and (b) turns what they measured into
the report. ``e2e_drive`` imports it only when ``--profile`` is given.

    C:/Users/steen/anaconda3/envs/data/python.exe tools/e2e_drive.py \\
        --level ship_freighter_standard_s28_medium --control match \\
        --frames 160 --profile --profile-skip 40 --glfinish

Flags (parsed by :func:`split_profile_args`, consumed before main.py sees argv):
    --profile            time the loop and print the report after the run
    --profile-skip K     leave the first K frames out of every number
                         (default 40: shader compiles, level warm-up)
    --glfinish           ``glFinish`` after each render stage, so a stage's
                         time includes its GPU work (without it a stage shows
                         only the CPU time of issuing its draws)
    --profile-inner "module:Class.method;..."
                         extra timers, nested INSIDE the stages above and
                         listed under them as ``  > Class.method`` (the
                         ``renderer`` modules are glFinish'd too); a spec that
                         does not resolve is reported and skipped

What is timed (every label is a wall-clock sum over the measured frames):
``Simulation.step`` / ``set_light`` / ``relight``, the ``GameRenderer``
frame stages (``upload_state``, ``compose_world``, ...), and the match
observers' ``on_tick`` / ``handle_frame``. The report adds the frame time
(mean / median / p95 / max, fps), the sim steps per frame, the sim's pace
against real time, and "everything else in the loop".

Born from issue #143: a watched bot match ran at 4 fps because one thread runs
ticks and drawing and a tick cost 39 ms of its 41.7 ms period; this showed it.
"""
from __future__ import annotations

import ctypes
import importlib
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

DEFAULT_SKIP = 40

RENDER_STAGES = ("upload_state", "compose_world", "draw_background_to_screen",
                 "blit_world_to_screen", "draw_debug_hud", "draw_panel",
                 "consume_events", "end_frame")


# ---------------------------------------------------------------------------
# argv (pure)
# ---------------------------------------------------------------------------
def split_profile_args(argv):
    """Split the profiler's own flags off ``argv``. Returns
    ``(options, rest)``: ``options`` is None without ``--profile``, else
    ``{"skip": int, "glfinish": bool, "inner": [spec, ...]}``. A profile
    sub-flag without ``--profile`` is refused (it would silently do nothing)."""
    rest = []
    on = False
    skip = None
    glfinish = False
    inner = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--profile":
            on = True
            i += 1
        elif a == "--profile-skip":
            try:
                skip = int(argv[i + 1])
            except (IndexError, ValueError):
                raise SystemExit("--profile-skip wants a frame count")
            if skip < 0:
                raise SystemExit(f"--profile-skip must be >= 0, got {skip}")
            i += 2
        elif a == "--glfinish":
            glfinish = True
            i += 1
        elif a == "--profile-inner":
            if i + 1 >= len(argv):
                raise SystemExit("--profile-inner wants \"module:Class.method;...\"")
            inner = (inner or []) + [s for s in argv[i + 1].split(";") if s.strip()]
            i += 2
        else:
            rest.append(a)
            i += 1
    if not on:
        if skip is not None or glfinish or inner is not None:
            raise SystemExit("--profile-skip / --glfinish / --profile-inner "
                             "only mean something with --profile")
        return None, rest
    return {"skip": DEFAULT_SKIP if skip is None else skip,
            "glfinish": glfinish, "inner": inner or []}, rest


def parse_inner(spec: str):
    """``"module:Class.method"`` -> ``(module, Class, method)``; refused by
    name when it does not have that shape."""
    mod, colon, path = spec.strip().partition(":")
    cname, dot, meth = path.partition(".")
    if not (mod and colon and cname and dot and meth):
        raise ValueError(f"--profile-inner wants module:Class.method, got {spec!r}")
    return mod, cname, meth


# ---------------------------------------------------------------------------
# the report (pure)
# ---------------------------------------------------------------------------
def format_report(passthrough, glfinish, skip, acc, calls, frame_times,
                  steps_per_frame, inner_labels=(), tps=24.0):
    """The report text. ``acc`` is label -> seconds and ``calls`` label ->
    count over the measured frames, ``frame_times`` the seconds of each
    measured frame, ``steps_per_frame`` the sim steps in each, ``inner_labels``
    the labels that nest inside other timers (so they are not subtracted from
    "everything else in the loop"). Pure: numbers in, text out."""
    import numpy as np
    ft = np.array(frame_times, dtype=np.float64)
    n = len(ft)
    if n == 0:
        return "frame_profile: no measured frames (--frames <= --profile-skip?)"
    tot = float(ft.sum())
    steps = np.array(steps_per_frame, dtype=np.float64)
    lines = [
        "",
        f"=== frame_profile: {' '.join(passthrough)}  glfinish={glfinish}",
        f"measured frames {n}  (skipped {skip});  sim steps {int(steps.sum())}",
        f"frame ms: mean {1e3 * ft.mean():.1f}  median {1e3 * np.median(ft):.1f}  "
        f"p95 {1e3 * np.percentile(ft, 95):.1f}  max {1e3 * ft.max():.1f}  "
        f"-> {n / tot:.1f} fps",
        f"sim steps per frame: mean {steps.mean():.2f}  max {int(steps.max())};  "
        f"sim pace = {steps.sum() / tps / tot:.2f}x real time",
        f"{'part':34s} {'ms/frame':>9s} {'% frame':>8s} {'calls':>6s} {'ms/call':>8s}",
    ]
    inner = set(inner_labels)
    rest = tot
    for label, s in sorted(acc.items(), key=lambda kv: -kv[1]):
        if label not in inner:
            rest -= s
        lines.append(f"{label:34s} {1e3 * s / n:9.2f} {100 * s / tot:8.1f} "
                     f"{calls[label]:6d} {1e3 * s / max(1, calls[label]):8.2f}")
    lines.append(f"{'(everything else in the loop)':34s} {1e3 * rest / n:9.2f} "
                 f"{100 * rest / tot:8.1f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the timers
# ---------------------------------------------------------------------------
def prepare_imports(root: Path, passthrough) -> None:
    """Put the game's modules on ``sys.path`` as ``main.py`` does, BEFORE the
    timers import them (the CUDA build first when ``--cuda`` is among the
    arguments: ``main.py`` hands off to ``tools/run_on_cuda`` only while
    ``breach_physics`` is not yet imported, and the timers import it)."""
    if "--cuda" in passthrough and "breach_physics" not in sys.modules:
        sys.path.insert(0, str(root / "tools"))
        import run_on_cuda
        run_on_cuda.setup_cuda_import()
        import breach_physics as _bp_cuda
        run_on_cuda.enable_all_backends(_bp_cuda)
    for p in (root / "cpp" / "build" / "Release", root, root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


class FrameProfiler:
    """Wall-clock timers wrapped around the loop's entry points at class level.

    ``install(rl)`` patches the classes; ``on_frame(n)`` is called by the
    driver once per frame (``n`` counts from 1, after the driver's own
    increment) from the hook ``main.py``'s loop already calls; ``report()``
    returns the text. Timers do not nest except through ``--profile-inner``.
    """

    def __init__(self, skip=DEFAULT_SKIP, glfinish=False, inner=(),
                 passthrough=(), tps=None):
        self.skip = skip
        self.glfinish = glfinish
        self.inner_specs = list(inner)
        self.passthrough = list(passthrough)
        self.tps = tps
        self.frame = 0
        self.acc = defaultdict(float)
        self.calls = defaultdict(int)
        self.frame_times = []
        self.steps_per_frame = []
        self.inner_labels = set()
        self._cur_steps = 0
        self._t_frame = None
        self._gl_finish = None
        self._rl = None

    # -- wrapping ----------------------------------------------------------
    def _timed(self, cls, name, label, gl=False, count_step=False):
        orig = getattr(cls, name)
        prof = self

        def wrapper(*a, **k):
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                if gl and prof._gl_finish is not None:
                    prof._rl.rl_draw_render_batch_active()
                    prof._gl_finish()
                if prof.frame > prof.skip:
                    prof.acc[label] += time.perf_counter() - t0
                    prof.calls[label] += 1
                if count_step:
                    prof._cur_steps += 1
        setattr(cls, name, wrapper)

    def install(self, rl):
        """Patch the timers in. Call after ``prepare_imports`` and before the
        game's loop runs."""
        self._rl = rl
        if self.glfinish:
            self._gl_finish = ctypes.windll.opengl32.glFinish
        if self.tps is None:
            try:
                from config import CFG
                self.tps = float(CFG.clock.ticks_per_second)
            except Exception:                              # pragma: no cover
                self.tps = 24.0
        from simulation import Simulation
        from renderer import GameRenderer
        self._timed(Simulation, "step", "sim.step", count_step=True)
        self._timed(Simulation, "set_light", "sim.set_light")
        self._timed(Simulation, "relight", "sim.relight")
        for n in RENDER_STAGES:
            self._timed(GameRenderer, n, f"r.{n}", gl=True)
        try:
            import control_match
            for cname in ("MatchObserver", "ReplayObserver"):
                c = getattr(control_match, cname, None)
                if c is None:
                    continue
                for meth in ("on_tick", "handle_frame"):
                    if meth in c.__dict__:
                        self._timed(c, meth, f"{cname}.{meth}")
        except Exception as exc:                           # pragma: no cover
            print("[frame_profile] no control_match timers:", exc, flush=True)
        for spec in self.inner_specs:
            try:
                mod, cname, meth = parse_inner(spec)
                c = getattr(importlib.import_module(mod), cname)
                label = f"  > {cname}.{meth}"
                self._timed(c, meth, label, gl=mod.startswith("renderer"))
                self.inner_labels.add(label)
            except Exception as exc:
                print(f"[frame_profile] inner timer skipped: {spec} ({exc})", flush=True)

    # -- per frame ---------------------------------------------------------
    def on_frame(self, n: int) -> None:
        """The driver's frame hook: ``n`` is the frame about to run (1-based).
        Closes the previous frame's measurement and opens this one."""
        now = time.perf_counter()
        if self._t_frame is not None and self.frame > self.skip:
            self.frame_times.append(now - self._t_frame)
            self.steps_per_frame.append(self._cur_steps)
        self._t_frame = now
        self._cur_steps = 0
        self.frame = n

    # -- the end -----------------------------------------------------------
    def report(self) -> str:
        return format_report(self.passthrough, self.glfinish, self.skip,
                             self.acc, self.calls, self.frame_times,
                             self.steps_per_frame, self.inner_labels,
                             tps=self.tps or 24.0)
