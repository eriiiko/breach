"""CUDA-S3 gate (pytest) — the GPU water solver bit-identity proof.

SKIPS cleanly without a CUDA build / device. When the GPU build is present, runs
the S3 check in an isolated anaconda-3.11 subprocess (cuda_harness): the isolated
GPU-vs-CPU pipe-model solver comparison over rich synthetic inputs (the EOS-P3
integer head term, tilt poly + per-tile double tilt, donor-cell flux, the OUTFLOW LIMITER,
scale_mag, the dry/solid/eps clamps) on water_depth + flow_vx + flow_vy AND the
full-engine backend-switch integration. A non-zero exit or a missing PASS marker
fails the test.

This is the water solver's cross-GPU determinism gate — GPU-int == CPU-int.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_s3_water_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_s3_check, sys; sys.exit(cuda_s3_check.main())", timeout=300,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "S3_RESULT: PASS" in out, (
        f"CUDA-S3 water did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
