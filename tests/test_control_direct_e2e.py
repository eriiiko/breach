"""G1p (gameplay pack v1, #183) -- direct control in the REAL window.

``main.py --control direct`` on ``levels/door_test`` in the hidden window
(``tools/e2e_drive.py``, the one E2E driver), with scripted held keys and a
scripted mouse position: what Erik does at the keyboard, as he would see it.
The level's first [[spawn]] marine (Alpha, at (24, 20)) is possessed; main.py
prints its final position and facing (the ``direct:`` line).
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPAWN = (24.0, 20.0)

# door_test in e2e_drive's 1280x720 window: the map viewport is 1000 x 720 px
# at the start zoom 25 px / tile, clamped at the world's top-left, so the
# view starts at tile (0, 1.2) and Alpha's centre (25.5, 21.5) is at screen
# pixel (637.5, 507.5). A cursor at (900, 508) starts due east of it.
MOUSE_EAST = "900,508"
# ... and a cursor at (638, 100) starts (very nearly) due north of it.
MOUSE_NORTH = "638,100"


def _drive(*args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(
        [sys.executable, "tools/e2e_drive.py", "--level", "door_test",
         "--control", "direct", "--frames", "80", *args],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    m = re.search(r"direct: device (\w+), unit (\S+) at \(([-\d.]+), "
                  r"([-\d.]+)\) facing ([-\d.]+)", r.stdout)
    assert m, r.stdout[-3000:]
    return (m.group(1), m.group(2), float(m.group(3)), float(m.group(4)),
            float(m.group(5)))


def test_the_rdfg_keys_move_the_possessed_marine_and_the_mouse_aims_it():
    """Property: in the real window, holding R moves the possessed marine
    up the screen (forward, -y) and nothing sideways, holding G moves it
    right (+x) and nothing vertically, holding A with R moves it farther than
    R alone (full speed) when the cursor is ahead of it and no farther when
    the cursor is to its side (G1p-c: the sprint is forward only), and a
    cursor due east turns it to face east. Breaks if the key table is not
    wired to the movement intents, a key moves the wrong way, A is not the
    full speed, the full speed ignores the facing, or the mouse does not
    aim."""
    dev, name, x, y, facing = _drive("--hold", "R@10-70",
                                     "--mouse", f"{MOUSE_EAST}@1")
    assert (dev, name) == ("keyboard", "Alpha")
    assert x == SPAWN[0] and y < SPAWN[1] - 0.5
    # turned to face the cursor (its world point: view origin + px / zoom),
    # from where the marine ended up
    cx, cy = 0.0 + 900 / 25.0, 1.2 + 508 / 25.0
    want = math.atan2(-(cy - (y + 1.5)), cx - (x + 1.5))
    assert abs(facing - want) < 0.01, (facing, want)
    walked = SPAWN[1] - y

    _, _, x, y, _ = _drive("--hold", "G@10-70", "--mouse", f"{MOUSE_EAST}@1")
    assert y == SPAWN[1] and x > SPAWN[0] + 0.5

    # G1p-c: the throttle's extra speed is FORWARD ONLY (Erik's ruling,
    # 2026-11-10). With the cursor east, R moves the marine at 90 deg to its
    # aim, so A adds nothing: it strafes at the same speed.
    _, _, x, y, _ = _drive("--hold", "R@10-70", "--hold", "A@10-70",
                           "--mouse", f"{MOUSE_EAST}@1")
    assert x == SPAWN[0] and SPAWN[1] - y < 1.2 * walked
    # With the cursor north (along the move), A runs: farther than R alone.
    _, _, x, y, _ = _drive("--hold", "R@10-30", "--mouse", f"{MOUSE_NORTH}@1")
    walked_north = SPAWN[1] - y
    _, _, x, y, _ = _drive("--hold", "R@10-30", "--hold", "A@10-30",
                           "--mouse", f"{MOUSE_NORTH}@1")
    assert x == SPAWN[0] and SPAWN[1] - y > 1.5 * walked_north
