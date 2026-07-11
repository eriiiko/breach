"""EOS P6.1 gate (pytest) — the GPU bulk donor-cell flux bit-identity proof.

The first per-kernel P6 digest gate (docs/eos_p6_gpu_alignment_review.md §4,
P6.1 row). Gated by ``cuda_available(kernel="bulk_flux")`` — the P6.0 pending-
set contract: this test SKIPS without a CUDA build / runtime, and RUNS (never
pinned-skips) now that P6.1 has removed the "bulk_flux" key from
``EOS_P6_PENDING_KERNELS``. When it runs, it executes the bulk-flux check in an
isolated GPU subprocess (cuda_harness): the isolated all-branch GPU-vs-CPU
comparison (donor upwinding both signs, sealed/partial faces, the OUTFLOW
LIMITER, scale_mag, solid/vacuum/negative clamps, trace + all-zero-plane
skips) AND the 200-tick closed-loop breach-venting + blast digest trajectory,
per-plane byte-compare every tick. A non-zero exit or a missing PASS marker
fails the test.

This is the bulk donor-cell flux's cross-GPU determinism gate — GPU-int ==
CPU-int (engine dispatch itself lands in P6.5).
"""
from __future__ import annotations

import pytest

import cuda_harness


pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(kernel="bulk_flux"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_bulk_flux_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_bulk_flux_check, sys; sys.exit(cuda_bulk_flux_check.main())",
        timeout=600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "BULK_FLUX_RESULT: PASS" in out, (
        f"EOS P6.1 bulk flux did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
