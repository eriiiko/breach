"""CUDA-S4a gate (pytest) — the GPU smoke solver bit-identity proof.

SKIPS cleanly without a CUDA build / device. When the GPU build is present, runs
the S4a check in an isolated anaconda-3.11 subprocess (cuda_harness): the isolated
GPU-vs-CPU smoke comparison over rich synthetic inputs (the permeability-weighted
diffusion Laplacian, the wind^2 diffusion fold, the INTEGER semi-Lagrangian
advection with NEGATIVE-displacement back-traces + DDA wall-clip, the WSUM-near-
floor reciprocal renorm, wall/vacuum zeroing) on the gas plane AND the full-engine
smoke-backend-switch integration over 30 ticks. A non-zero exit or a missing PASS
marker fails the test.

This is the smoke solver's cross-GPU determinism gate — GPU-int == CPU-int. The
semi-Lagrangian advection is the hardest kernel of the CUDA arc so far.
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
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_s4a_smoke_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_s4a_check, sys; sys.exit(cuda_s4a_check.main())", timeout=300,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "S4A_RESULT: PASS" in out, (
        f"CUDA-S4a smoke did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
