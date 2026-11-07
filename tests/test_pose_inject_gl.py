"""#162 gate (pytest) -- the pose layer's CPU-inject skinner on this machine's
raylib binding (design §4.5, property 12).

Runs ``tests/gl_pose_inject_check.py`` in an ISOLATED subprocess with a hidden
raylib window (a GL context never enters the pytest process -- the
``gl_*_check.py`` convention) and asserts its result line. What each check
protects, and what breaks it, is in that script's header. Run it on EVERY
machine: the 5.x / 6.x binding split is the risk it guards (#80).

SKIPS when ``BREACH_NO_GL=1`` (a machine without a GL 3.3 context).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(os.environ.get("BREACH_NO_GL") == "1",
                                reason="no GL context (BREACH_NO_GL=1)")


def test_pose_inject_gl_checks():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "gl_pose_inject_check.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    out = proc.stdout + "\n" + proc.stderr
    assert "POSE_INJECT_GL_RESULT: PASS" in out, (
        f"pose inject GL checks did not pass (returncode {proc.returncode}):\n"
        + "\n".join(ln for ln in out.splitlines()
                    if ln.startswith("[check]") or "Error" in ln
                    or "POSE_INJECT" in ln))
    assert proc.returncode == 0, out
