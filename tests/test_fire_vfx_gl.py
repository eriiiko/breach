"""G3 F1 gate (pytest) -- the flame VFX substrate's GL checks.

Runs ``tests/gl_fire_vfx_check.py`` in an ISOLATED subprocess with a hidden
raylib window (a GL context never enters the pytest process -- the
``tests/cuda_harness.py`` convention, for GL) and asserts its result line.
What each check protects, and what breaks it, is in that script's header.

SKIPS when ``BREACH_NO_GL=1`` (a machine without a GL 3.3 context) or when
``breach_physics`` is not importable (its last check runs the real renderer
on a real sim).
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    os.environ.get("BREACH_NO_GL") == "1"
    or importlib.util.find_spec("breach_physics") is None,
    reason="no GL context (BREACH_NO_GL=1) or no breach_physics build",
)


def test_fire_vfx_gl_checks():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "gl_fire_vfx_check.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900)
    out = proc.stdout + "\n" + proc.stderr
    assert "FIRE_VFX_GL_RESULT: PASS" in out, (
        f"flame VFX GL checks did not pass (returncode {proc.returncode}):\n"
        + "\n".join(ln for ln in out.splitlines()
                    if ln.startswith("[check]") or "Error" in ln
                    or "FIRE_VFX" in ln))
    assert proc.returncode == 0, out
