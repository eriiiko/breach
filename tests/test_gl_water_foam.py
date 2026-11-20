"""Water W2 gate (pytest) -- the foam's GL checks (#97: a dry map owns no
foam image and runs no pass; the step equals the numpy twin per cell; the
draw's amount and lace cover equal the twin per pixel; frames between ticks
allocate nothing; the cells that leave the wet box are cleared in both
ping-pong buffers; a backward tick empties the foam).

Runs ``tests/gl_water_foam_check.py`` in an ISOLATED subprocess with a
hidden raylib window (a GL context never enters the pytest process -- the
``tests/cuda_harness.py`` convention, for GL) and asserts its result line.
What each check protects, and what breaks it, is in that script's header.

SKIPS when ``BREACH_NO_GL=1`` (a machine without a GL 3.3 context) or when
``breach_physics`` is not importable (the check imports the sim's
``water_fixed`` through the pass).
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_BUILD = ROOT / "cpp" / "build" / "Release"
if _BUILD.is_dir() and str(_BUILD) not in sys.path:
    sys.path.append(str(_BUILD))

pytestmark = pytest.mark.skipif(
    os.environ.get("BREACH_NO_GL") == "1"
    or importlib.util.find_spec("breach_physics") is None,
    reason="no GL context (BREACH_NO_GL=1) or no breach_physics build",
)


def test_water_foam_gl_checks():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "gl_water_foam_check.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    out = proc.stdout + "\n" + proc.stderr
    assert "WATER_FOAM_GL_RESULT: PASS" in out, (
        f"water foam GL checks did not pass (returncode {proc.returncode}):\n"
        + "\n".join(ln for ln in out.splitlines()
                    if ln.startswith("[check]") or "Error" in ln
                    or "WATER_FOAM" in ln))
    assert proc.returncode == 0, out
