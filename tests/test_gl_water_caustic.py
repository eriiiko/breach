"""Water W4 gate (pytest) -- the water pass's caustic GL checks (#97: the
fine surface's Hessian and the caustic's gain equal the numpy twin per pixel;
a flat surface and an unlit floor leave the picture bit-identical to the
caustic off; under light the caustic darkens and brightens within its cap).

Runs ``tests/gl_water_caustic_check.py`` in an ISOLATED subprocess with a hidden
raylib window (a GL context never enters the pytest process -- the
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


def test_water_caustic_gl_checks():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "gl_water_caustic_check.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    out = proc.stdout + "\n" + proc.stderr
    assert "WATER_CAUSTIC_GL_RESULT: PASS" in out, (
        f"water caustic GL checks did not pass (returncode {proc.returncode}):\n"
        + "\n".join(ln for ln in out.splitlines()
                    if ln.startswith("[check]") or "Error" in ln
                    or "WATER_CAUSTIC" in ln))
    assert proc.returncode == 0, out
