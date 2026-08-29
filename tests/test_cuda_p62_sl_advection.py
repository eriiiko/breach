"""EOS P6.2 gate (pytest) — the GPU fused 3-field SL advection bit-identity proof.

SKIPS cleanly without a CUDA build / device (per-kernel key: "sl_advection" —
this is the FIRST gate on the P6.0 per-kernel pending-set contract). When the
GPU build is present, runs the P6.2 check in an isolated subprocess
(cuda_harness): the isolated GPU-vs-CPU-reference comparison over rich
synthetic inputs (multi-cell negative-displacement DDA marches, sealed/breach/
live cmask corners, the WSUM-near-floor Newton renorm, n_sub up to the cap,
degenerate grids) AND the blast + venting trajectory gate — 80 real engine
ticks with the per-tick digest_advect chain asserted CPU-solver == CPU-ref ==
GPU, byte-identical fields throughout — AND the committed default-scenario
golden on the CUDA build's CPU path. A non-zero exit or a missing PASS marker
fails the test.

SL advection is a pure gather (each destination reads its backtraced source
from a frozen snapshot — docs/eos_p6_gpu_alignment_review.md §1.4), so
bit-identity needs no restructuring; this gate is the proof.
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
    not cuda_harness.cuda_available("sl_advection"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_p62_sl_advection_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_p62_check, sys; sys.exit(cuda_p62_check.main())", timeout=600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "P62_RESULT: PASS" in out, (
        f"EOS P6.2 SL advection did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
