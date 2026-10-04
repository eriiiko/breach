"""tools/e2e_drive.py -- THE headless end-to-end reproduction.

The real ``main.py`` in a HIDDEN raylib window, driven by scripted keys for a
fixed number of frames. Bug fixes start here (master rules: an E2E
reproduction as an end user would see it) -- never a parallel harness.

    C:/Users/steen/anaconda3/python.exe tools/e2e_drive.py --level playground \\
        --press M@40 --frames 120
    C:/Users/steen/anaconda3/python.exe tools/e2e_drive.py --level playground \\
        --press M@40 --frames 120 --cuda
    ... --control onephase --debug --press F4@10 --press F4@20 --frames 60

Flags (this tool's own):
    --frames N        run N frames, then close the window (default 120)
    --press KEY@F     report KEY as pressed on frame F (repeatable; KEY is a
                      raylib KeyboardKey name without the ``KEY_`` prefix:
                      M, F11, SPACE, I, ...)
    --shot PATH@F     save the frame-F picture to PATH (.png; repeatable),
                      read back just before that frame's end_drawing
    --level NAME      the level (passed through to main.py)
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
this: ``--level playground --press M@40 --frames 120``.
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


def parse_args(argv):
    """Split argv into (frames, {frame: [key names]}, main.py argv). Pure --
    the tool's own flags are consumed, everything else passes through."""
    frames = DEFAULT_FRAMES
    presses = {}
    shots = {}
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
    return frames, presses, shots, passthrough


def main(argv=None) -> int:
    faulthandler.enable()
    frames, presses, shots, passthrough = parse_args(list(sys.argv[1:] if argv is None else argv))

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

    frame = [0]
    orig_close = rl.window_should_close
    orig_pressed = rl.is_key_pressed

    def _should_close():
        frame[0] += 1
        if frame[0] % 20 == 0:
            print(f"[e2e_drive] frame {frame[0]}", flush=True)
        return frame[0] > frames or orig_close()

    def _is_key_pressed(key):
        if int(key) in scripted.get(frame[0], ()):
            print(f"[e2e_drive] frame {frame[0]}: pressing "
                  f"{rl.KeyboardKey(int(key)).name}", flush=True)
            return True
        return orig_pressed(key)

    orig_end_drawing = rl.end_drawing

    def _end_drawing():
        for path in shots.get(frame[0], ()):
            img = rl.load_image_from_screen()
            rl.export_image(img, str(Path(path).resolve()))
            rl.unload_image(img)
            print(f"[e2e_drive] frame {frame[0]}: saved {path}", flush=True)
        orig_end_drawing()

    rl.window_should_close = _should_close
    rl.is_key_pressed = _is_key_pressed
    rl.end_drawing = _end_drawing

    sys.argv = ["main.py"] + passthrough
    print(f"[e2e_drive] main.py {' '.join(passthrough)} -- {frames} frames, "
          f"presses {dict(sorted(presses.items()))}", flush=True)
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
    if frame[0] <= frames:
        print(f"[e2e_drive] FAIL: main.py returned at frame {frame[0]} "
              f"(< {frames}) without an exception", flush=True)
        return 2
    print(f"[e2e_drive] OK: {frames} frames, clean exit", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
