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

# ---------------------------------------------------------------------------
# CUDA PARITY SUSPENDED, P-G1a -> P-G2 (gas-energy conservation arc #54,
# design §5 "CUDA parity is suspended from P-G1a until P-G2 -- named,
# time-boxed"). P-G1a rewrites the CPU EOS energy chain (the per-stage KE
# brackets, the face-flux energy step, the once-per-tick recovery, and the
# transport's energy half) and DELIBERATELY leaves the .cu twins on the old
# step-4c kernels, so every CPU-vs-GPU bit-identity check in this family is
# expected to diverge until P-G2 lands K1's brackets and the new K3 flux
# kernel. `strict=False` so a check that happens to still agree does not
# fail the suite -- P-G2 removes these marks and re-arms the gate.
pytestmark = pytest.mark.xfail(
    reason="P-G2 pending: CUDA twins of the energy step", strict=False)



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
