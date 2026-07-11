"""EOS P6.4 gate (pytest) — the GPU kick + compression-work bit-identity proof.

SKIPS cleanly without a CUDA build / device (per-kernel key:
"kick_compression" on the P6.0 per-kernel pending-set contract). When the GPU
build is present, runs the P6.4 check in an isolated subprocess
(cuda_harness): the isolated GPU-vs-CPU-reference comparison over synthetic
inputs that FORCE every v2.4 rail (the counted min(c_LOCAL, U_MAX) magnitude
clamp in both cap regimes, the ±2^30 RAD_SAFE component guard, the ±T_WORK_CLAMP
factor rail, the T_MIN energy floor, the T_MAX_PHYS ceiling — counters asserted
bit-equal, not just fields) AND a 120-tick blast + venting trajectory gate —
real engine ticks with the per-tick digest_velocity / digest_compression chain
asserted CPU-solver == CPU-ref == GPU, byte-identical fields and rail-counter
deltas throughout — AND the committed default-scenario golden on the CUDA
build's CPU path. A non-zero exit or a missing PASS marker fails the test.

Both passes are pure per-cell gathers (each writes only its own u or T —
docs/eos_p6_gpu_alignment_review.md §1.5), so bit-identity needs no
restructuring; the counters are order-free +1-per-engaging-cell sums, exact
under device atomics. This gate is the proof.
"""
from __future__ import annotations

import pytest

import cuda_harness


pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("kick_compression"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_p64_kick_compression_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_kick_check, sys; sys.exit(cuda_kick_check.main())", timeout=600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "P64_RESULT: PASS" in out, (
        f"EOS P6.4 kick+compression did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
