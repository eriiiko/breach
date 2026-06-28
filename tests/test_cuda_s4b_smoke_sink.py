"""CUDA-S4b gate (pytest) — the GPU smoke sink_hop bit-identity proof.

SKIPS cleanly without a CUDA build / device. When the GPU build is present, runs
the S4b check in an isolated anaconda-3.11 subprocess (cuda_harness): the isolated
GPU-vs-CPU sink_hop comparison over rich synthetic inputs (the sealed-room identity
hop, the breach-ward ±/diagonal back-traces, the new sink float bridge with the
host-side min(sink_strength,1) capped & uncapped, the DDA wall-clip, the WSUM-near-
floor reciprocal renorm, wall/vacuum zeroing) on the gas plane AND the full-engine
breach-ACTIVE smoke-backend-switch integration over 30 ticks (step + sink_hop both
on the GPU). A non-zero exit or a missing PASS marker fails the test.

This is the breach sink-pull's cross-GPU determinism gate — GPU-int == CPU-int. It
reuses S4a's verified back-trace + clamp machinery; only the sink displacement is
new, so the proof is small but the bit-identity bar (tol 0) is unchanged.
"""
from __future__ import annotations

import pytest

import cuda_harness


pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_s4b_smoke_sink_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_s4b_check, sys; sys.exit(cuda_s4b_check.main())", timeout=300,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "S4B_RESULT: PASS" in out, (
        f"CUDA-S4b sink_hop did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
