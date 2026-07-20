"""CUDA-S8a Path B residency gate (pytest wrapper).

SKIPS cleanly without a CUDA build / device. When the GPU build is present,
runs tests/cuda_s8a_check.py in an isolated CUDA subprocess (cuda_harness):

  PART 1 — a >=30-tick full-engine A/B (detonations / water / fire / a scripted
  structural edit) driven through PhysicsRunner.step, residency ON (water
  substeps + smoke traces resident; EOS/combustion/tail bracketed) vs the CPU
  path, asserting byte-for-byte identity (tol 0) of every synced field including
  host-path heat/ripple/ripple_v.

  PART 2 — a CPU vs per-call-GPU vs resident benchmark at two grid sizes proving
  the substep-/plane-MULTIPLIED transfer tax is gone (resident beats per-call GPU).

A non-zero exit or a missing PASS marker fails the test.
"""
from __future__ import annotations

import pytest

import cuda_harness


pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("s8a_residency"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_s8a_residency_bit_identity_and_payoff():
    proc = cuda_harness.run_cuda_script(
        "import cuda_s8a_check, sys; sys.exit(cuda_s8a_check.main())",
        timeout=1800,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "S8A_RESULT: PASS" in out, (
        f"S8a residency gate did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
