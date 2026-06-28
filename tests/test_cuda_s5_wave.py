"""CUDA-S5 gate (pytest) — the GPU wave_substep bit-identity proof.

SKIPS cleanly without a CUDA build / device. When the GPU build is present, runs
the S5 check in an isolated anaconda-3.11 subprocess (cuda_harness): the isolated
GPU-vs-CPU wave_substep comparison over rich synthetic inputs (the rate-limited
source feed, the permeability-weighted Laplacian gather, the int64 velocity kick,
the pressure update, the scale_mag per-cell absorption, the wall/vacuum BCs, the
sign-symmetric anomaly transfer) AND — the determinism crux — the **mean_wp int64
reduction** (an order-free atomicAdd; the gate reconstructs the int64 sum + mean
and proves the GPU sum == the CPU sum to the LSB across varied interior masks +
+/- wave_p), plus the full-engine wave-backend-switch integration over 30 ticks.
A non-zero exit or a missing PASS marker fails the test.

This is the wave solver's cross-GPU determinism gate — GPU-int == CPU-int. The
int64 atomicAdd reduction is the arc's FIRST GPU reduction; integer + is order-free
so the sum is bit-identical regardless of thread/scheduler order.
"""
from __future__ import annotations

import pytest

import cuda_harness


pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_s5_wave_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_s5_check, sys; sys.exit(cuda_s5_check.main())", timeout=300,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "S5_RESULT: PASS" in out, (
        f"CUDA-S5 wave did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
