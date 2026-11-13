"""The sound layer in the REAL window (#202 S1): `main.py --control direct`
on `levels/door_test` through the one E2E driver, hidden and therefore
muted. main.py prints how many sounds the layer started (the `sound:` line).

Properties held:
  - holding the trigger starts carbine sounds, about one per round fired --
    breaks if the layer is not fed every sim step's events, the carbine is
    unbound, or the possessed unit is out of its own shot's reach;
  - a thrown frag starts exactly one sound, its explosion's -- breaks if the
    explosion event stops naming its payload row or the frag is unbound;
  - doing nothing starts none.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOUSE_EAST = "900,508"          # due east of Alpha (test_control_direct_e2e.py)


def _sounds(frames: int, *args) -> int:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(
        [sys.executable, "tools/e2e_drive.py", "--level", "door_test",
         "--control", "direct", "--frames", str(frames),
         "--mouse", f"{MOUSE_EAST}@1", *args],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    m = re.search(r"sound: (\d+) sounds started", r.stdout)
    assert m, r.stdout[-3000:]
    return int(m.group(1))


def test_holding_the_trigger_starts_carbine_sounds():
    assert _sounds(150, "--hold", "MOUSE_LEFT@20-90") >= 5


def test_a_thrown_frag_starts_one_sound_and_idling_none():
    assert _sounds(420, "--press", "W@20") == 1
    assert _sounds(60) == 0
