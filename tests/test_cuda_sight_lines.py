"""#195 1c gate (pytest wrapper): sight for many tiles, CUDA module == CPU module.

SKIPS cleanly without a CUDA build / device. When the GPU build is present, runs
tests/cuda_sight_lines_check.py in an isolated subprocess (cuda_harness) and
compares the digest it prints with this process's CPU module's own.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src", _ROOT / "cpp" / "build" / "Release",
           _ROOT / "tools", _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("sight_lines"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_the_cuda_module_answers_every_line_as_the_cpu_module_does():
    """PROPERTY: the CUDA module's ``sight_rays`` / ``sight_window`` equal the
    Python reference and ``has_los`` (checked in the subprocess), and the sha256
    of all its answers on ship_demo, a freighter and a screen-scattered box
    equals the CPU module's.

    BREAKS IF: the routine's translation unit is left out of the CUDA module,
    compiled under another floating-point or integer-width rule there, or the
    two modules' walks diverge.
    """
    proc = cuda_harness.run_cuda_script(
        "import cuda_sight_lines_check, sys; "
        "sys.exit(cuda_sight_lines_check.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "SIGHT_LINES_RESULT: PASS" in out, (
        f"the sight_lines gate did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
    gpu = re.search(r"SIGHT_LINES_DIGEST: ([0-9a-f]{64})", out)
    assert gpu, out
    import cuda_sight_lines_check            # this process: the CPU module
    assert "build_cuda" not in str(cuda_sight_lines_check.bp.__file__)
    assert cuda_sight_lines_check.digest() == gpu.group(1)
