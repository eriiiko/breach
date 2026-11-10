"""tools/e2e_drive.py -- THE headless end-to-end reproduction.

The real ``main.py`` in a HIDDEN raylib window, driven by scripted keys for a
fixed number of frames. Bug fixes start here (master rules: an E2E
reproduction as an end user would see it) -- never a parallel harness.

    C:/Users/steen/anaconda3/python.exe tools/e2e_drive.py --level playground \\
        --press F4@40 --frames 120
    C:/Users/steen/anaconda3/python.exe tools/e2e_drive.py --level playground \\
        --press F4@40 --frames 120 --cuda
    ... --control onephase --debug --press F4@10 --press F4@20 --frames 60

Flags (this tool's own):
    --frames N        run N frames, then close the window (default 120)
    --press KEY@F     report KEY as pressed on frame F (repeatable; KEY is a
                      raylib KeyboardKey name without the ``KEY_`` prefix:
                      M, F11, SPACE, I, ...)
    --shot PATH@F     save the frame-F picture to PATH (.png; repeatable),
                      read back just before that frame's end_drawing
    --hold KEY@A-B    report KEY as held DOWN on frames A..B inclusive (and
                      pressed on frame A; repeatable). KEY is a KeyboardKey
                      name as for --press, or MOUSE_LEFT / MOUSE_RIGHT /
                      MOUSE_MIDDLE (a mouse button). Direct control's
                      movement reads held keys (gameplay pack G1p)
    --mouse X,Y@F     from frame F on, the mouse is at screen pixel (X, Y)
                      (repeatable; the last one at or before a frame wins)
    --level NAME      the level (passed through to main.py)
    --profile         time the loop and print a report after the run: frame ms
                      and fps, sim steps per frame, the sim's pace, and the
                      wall-clock cost of Simulation.step, the GameRenderer
                      frame stages and the match observers' per-tick work
                      (tools/frame_profile.py, imported only with this flag;
                      without it nothing here changes). #143
    --profile-skip K  leave the first K frames out of the numbers (default 40)
    --glfinish        glFinish after each render stage, so a stage's time
                      includes its GPU work
    --profile-inner "module:Class.method;..."
                      extra timers listed under the stages ("  > Class.method")
Every other argument passes through to main.py unchanged (--cuda,
--control NAME, --debug, --res N, ...); ``--windowed`` is always added (a fixed
1280x720 hidden window, never the monitor-sized borderless one).

How it drives: ``pyray.window_should_close`` becomes a frame counter (main.py's
loop calls it once per frame, through ``GameRenderer.should_close``) and
``pyray.is_key_pressed`` reports the scripted keys on their frames, else the
real (idle) keyboard. Nothing in main.py is modified: everything it calls
resolves ``rl.<name>`` at call time, so the patch reaches every module.

Exit status: 0 when main.py ran all N frames and returned cleanly; 1 on ANY
exception (the traceback is printed); 2 when main.py returned before frame N
without an exception (a premature close is a finding, not a pass); main.py's
own SystemExit code otherwise.

Born from issue #80 (ray-engine-v2 P6c): the 3D-marines crash
(``anim.keyframeCount`` on raylib 5.5) reproduced on the first try by exactly
this: ``--level playground --press M@40 --frames 120`` (M toggled the 3D
units then; they are always on since #33, and no key toggles them).
"""
from __future__ import annotations

import faulthandler
import os
import runpy
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FRAMES = 120

# the profiler's own flags (tools/frame_profile.py); any of them loads it
_PROFILE_FLAGS = ("--profile", "--profile-skip", "--glfinish", "--profile-inner")


def parse_args(argv):
    """Split argv into (frames, {frame: [key names]}, {frame: [shot paths]},
    main.py argv). Pure -- the tool's own flags are consumed, everything else
    passes through. (The holds and mouse moves: :func:`parse_args_full`.)"""
    frames, presses, shots, _holds, _mouse, passthrough = parse_args_full(argv)
    return frames, presses, shots, passthrough


def parse_args_full(argv):
    """:func:`parse_args` plus the scripted held keys ``[(key name, first,
    last)]`` and mouse moves ``[(frame, x, y)]`` (G1p): (frames, presses,
    shots, holds, mouse, main.py argv)."""
    frames = DEFAULT_FRAMES
    presses = {}
    shots = {}
    holds = []
    mouse = []
    passthrough = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--frames":
            frames = int(argv[i + 1])
            i += 2
            continue
        if a == "--press":
            spec = argv[i + 1]
            key, _, at = spec.partition("@")
            if not key or not at:
                raise SystemExit(f"--press wants KEY@FRAME, got {spec!r}")
            presses.setdefault(int(at), []).append(key.upper())
            i += 2
            continue
        if a == "--hold":
            spec = argv[i + 1]
            key, _, span = spec.partition("@")
            lo, _, hi = span.partition("-")
            if not key or not lo or not hi:
                raise SystemExit(f"--hold wants KEY@FIRST-LAST, got {spec!r}")
            holds.append((key.upper(), int(lo), int(hi)))
            i += 2
            continue
        if a == "--mouse":
            spec = argv[i + 1]
            xy, _, at = spec.partition("@")
            x, _, y = xy.partition(",")
            if not x or not y or not at:
                raise SystemExit(f"--mouse wants X,Y@FRAME, got {spec!r}")
            mouse.append((int(at), int(x), int(y)))
            i += 2
            continue
        if a == "--shot":
            spec = argv[i + 1]
            path, _, at = spec.rpartition("@")
            if not path or not at:
                raise SystemExit(f"--shot wants PATH@FRAME, got {spec!r}")
            shots.setdefault(int(at), []).append(path)
            i += 2
            continue
        passthrough.append(a)
        i += 1
    if frames < 1:
        raise SystemExit(f"--frames must be >= 1, got {frames}")
    if "--windowed" not in passthrough:
        passthrough.append("--windowed")
    return frames, presses, shots, holds, sorted(mouse), passthrough


def main(argv=None) -> int:
    faulthandler.enable()
    argv = list(sys.argv[1:] if argv is None else argv)
    profile_opts = None
    if any(a in _PROFILE_FLAGS for a in argv):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import frame_profile
        profile_opts, argv = frame_profile.split_profile_args(argv)
    frames, presses, shots, holds, mouse, passthrough = parse_args_full(argv)

    os.chdir(ROOT)
    import pyray as rl
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)

    # Resolve every scripted key BEFORE the run: a typo fails here, by name.
    scripted = {}
    for at, names in presses.items():
        for name in names:
            try:
                code = int(getattr(rl.KeyboardKey, f"KEY_{name}"))
            except AttributeError:
                raise SystemExit(f"--press: unknown key {name!r} (a raylib "
                                 f"KeyboardKey name without KEY_, e.g. M, F11)")
            scripted.setdefault(at, set()).add(code)
    # --hold: keyboard keys -> held_keys, mouse buttons -> held_buttons, each
    # {code: [(first, last)]}; a hold's first frame is also a press.
    mouse_buttons = {"MOUSE_LEFT": 0, "MOUSE_RIGHT": 1, "MOUSE_MIDDLE": 2}
    held_keys, held_buttons = {}, {}
    for name, lo, hi in holds:
        if name in mouse_buttons:
            held_buttons.setdefault(mouse_buttons[name], []).append((lo, hi))
            continue
        try:
            code = int(getattr(rl.KeyboardKey, f"KEY_{name}"))
        except AttributeError:
            raise SystemExit(f"--hold: unknown key {name!r} (a raylib "
                             f"KeyboardKey name without KEY_, or MOUSE_LEFT / "
                             f"MOUSE_RIGHT / MOUSE_MIDDLE)")
        held_keys.setdefault(code, []).append((lo, hi))

    prof = None
    if profile_opts is not None:
        # the timers import the game's modules, so main.py's own path setup
        # (and its --cuda hand-off) has to be done first
        frame_profile.prepare_imports(ROOT, passthrough)
        prof = frame_profile.FrameProfiler(
            skip=profile_opts["skip"], glfinish=profile_opts["glfinish"],
            inner=profile_opts["inner"], passthrough=passthrough)
        prof.install(rl)

    frame = [0]
    orig_close = rl.window_should_close
    orig_pressed = rl.is_key_pressed

    def _should_close():
        frame[0] += 1
        if prof is not None:
            prof.on_frame(frame[0])
        if frame[0] % 20 == 0:
            print(f"[e2e_drive] frame {frame[0]}", flush=True)
        return frame[0] > frames or orig_close()

    def _in(spans, f):
        return any(lo <= f <= hi for lo, hi in spans)

    def _is_key_pressed(key):
        if int(key) in scripted.get(frame[0], ()):
            print(f"[e2e_drive] frame {frame[0]}: pressing "
                  f"{rl.KeyboardKey(int(key)).name}", flush=True)
            return True
        if any(lo == frame[0] for lo, _ in held_keys.get(int(key), ())):
            return True
        return orig_pressed(key)

    orig_down = rl.is_key_down
    orig_btn_down = rl.is_mouse_button_down
    orig_btn_pressed = rl.is_mouse_button_pressed
    orig_mx, orig_my = rl.get_mouse_x, rl.get_mouse_y

    def _is_key_down(key):
        if _in(held_keys.get(int(key), ()), frame[0]):
            return True
        return orig_down(key)

    def _is_mouse_button_down(button):
        if _in(held_buttons.get(int(button), ()), frame[0]):
            return True
        return orig_btn_down(button)

    def _is_mouse_button_pressed(button):
        if any(lo == frame[0] for lo, _ in held_buttons.get(int(button), ())):
            return True
        return orig_btn_pressed(button)

    def _mouse_at():
        at = None
        for f, x, y in mouse:
            if f <= frame[0]:
                at = (x, y)
        return at

    def _get_mouse_x():
        at = _mouse_at()
        return orig_mx() if at is None else at[0]

    def _get_mouse_y():
        at = _mouse_at()
        return orig_my() if at is None else at[1]

    orig_end_drawing = rl.end_drawing

    def _end_drawing():
        if shots.get(frame[0]):
            # Flush raylib's pending batch first: the panel, the HUD and any
            # other batched 2D draw is only drawn at end_drawing, so a
            # read-back before it showed the world without them (#127).
            rl.rl_draw_render_batch_active()
        for path in shots.get(frame[0], ()):
            img = rl.load_image_from_screen()
            rl.export_image(img, str(Path(path).resolve()))
            rl.unload_image(img)
            print(f"[e2e_drive] frame {frame[0]}: saved {path}", flush=True)
        orig_end_drawing()

    rl.window_should_close = _should_close
    rl.is_key_pressed = _is_key_pressed
    rl.end_drawing = _end_drawing
    if held_keys:
        rl.is_key_down = _is_key_down
    if held_buttons:
        rl.is_mouse_button_down = _is_mouse_button_down
        rl.is_mouse_button_pressed = _is_mouse_button_pressed
    if mouse:
        rl.get_mouse_x = _get_mouse_x
        rl.get_mouse_y = _get_mouse_y

    sys.argv = ["main.py"] + passthrough
    print(f"[e2e_drive] main.py {' '.join(passthrough)} -- {frames} frames, "
          f"presses {dict(sorted(presses.items()))}"
          + (f", holds {holds}" if holds else "")
          + (f", mouse {mouse}" if mouse else ""), flush=True)
    try:
        runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if code != 0:
            print(f"[e2e_drive] FAIL: main.py exited {exc.code!r} at frame {frame[0]}",
                  flush=True)
            return code
    except BaseException:
        traceback.print_exc()
        print(f"[e2e_drive] FAIL: exception at frame {frame[0]}", flush=True)
        return 1
    finally:
        if prof is not None:
            print(prof.report(), flush=True)
    if frame[0] <= frames:
        print(f"[e2e_drive] FAIL: main.py returned at frame {frame[0]} "
              f"(< {frames}) without an exception", flush=True)
        return 2
    print(f"[e2e_drive] OK: {frames} frames, clean exit", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
