"""CUDA-S1 gate (pytest) — the GPU temperature solver bit-identity proof.

SKIPS cleanly without a CUDA build / device. When the GPU build is present, runs
the S1 check in an isolated anaconda-3.11 subprocess (cuda_harness): the isolated
GPU-vs-CPU solver comparison over rich synthetic inputs (saturation, negatives,
NO_FACE, vacuum exposure) AND the full-engine backend-switch integration. A
non-zero exit or a missing PASS marker fails the test.

This is the first REAL physics kernel's gate — GPU-int == CPU-int in the engine.
"""
from __future__ import annotations

import pytest

import cuda_harness


pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_s1_temperature_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_s1_check, sys; sys.exit(cuda_s1_check.main())", timeout=240,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "S1_RESULT: PASS" in out, (
        f"CUDA-S1 temperature did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
