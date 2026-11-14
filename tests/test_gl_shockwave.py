"""G4 P2 gate (pytest) -- the shock front's GL checks (the resolve with no
ring / off the band is bit-identical; one ring over a checker equals the
twin's resample within 1/255, which also catches a y flip).

Runs ``tests/gl_shockwave_check.py`` in an ISOLATED subprocess with a hidden
raylib window (a GL context never enters the pytest process -- the
``tests/cuda_harness.py`` convention, for GL) and asserts its result line.
What each check protects, and what breaks it, is in that script's header.

SKIPS when ``BREACH_NO_GL=1`` (a machine without a GL 3.3 context) or when
``breach_physics`` is not importable (the walk's module imports the sim).
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# the build the check script itself puts on its path: found here too, so the
# skip below is about a missing build, not a missing sys.path entry
_BUILD = ROOT / "cpp" / "build" / "Release"
if _BUILD.is_dir() and str(_BUILD) not in sys.path:
    sys.path.append(str(_BUILD))

pytestmark = pytest.mark.skipif(
    os.environ.get("BREACH_NO_GL") == "1"
    or importlib.util.find_spec("breach_physics") is None,
    reason="no GL context (BREACH_NO_GL=1) or no breach_physics build",
)


def test_shockwave_gl_checks():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "gl_shockwave_check.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    out = proc.stdout + "\n" + proc.stderr
    assert "SHOCKWAVE_GL_RESULT: PASS" in out, (
        f"shock front GL checks did not pass (returncode {proc.returncode}):\n"
        + "\n".join(ln for ln in out.splitlines()
                    if ln.startswith("[check]") or "Error" in ln
                    or "SHOCKWAVE" in ln))
    assert proc.returncode == 0, out
