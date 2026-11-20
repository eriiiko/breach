"""Arc #63 gate (pytest) -- the larvae's bodies' GL checks (one instanced
draw whatever the count, drawn where it lies, the pose a function of the
phase, lit by the light field, a faded body leaves the floor untouched, off
costs nothing).

Runs ``tests/gl_larva_bodies_check.py`` in an ISOLATED subprocess with a
hidden raylib window (a GL context never enters the pytest process -- the
``tests/cuda_harness.py`` convention, for GL) and asserts its result line.
What each check protects, and what breaks it, is in that script's header.

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


def test_larva_bodies_gl_checks():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "gl_larva_bodies_check.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    out = proc.stdout + "\n" + proc.stderr
    assert "LARVA_GL_RESULT: PASS" in out, (
        f"larva body GL checks did not pass (returncode {proc.returncode}):\n"
        + "\n".join(ln for ln in out.splitlines()
                    if ln.startswith("[check]") or "Error" in ln
                    or "LARVA" in ln))
    assert proc.returncode == 0, out
