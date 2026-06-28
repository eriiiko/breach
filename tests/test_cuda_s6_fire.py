"""CUDA-S6 gate (pytest) — the GPU FireSimulation::step bit-identity proof.

SKIPS cleanly without a CUDA build / device. When the GPU build is present, runs
the S6 check in an isolated anaconda-3.11 subprocess (cuda_harness): the isolated
GPU-vs-CPU fire-step comparison over rich synthetic inputs (the P1 host max
early-exit, the P2 signed-logistic feedback incl. sqrt_q16 wind + the 4-neighbour
atmosphere mean + the snap-extinguish, the P3 own-tile plume deposit, the P4 smoke
emission SCATTER — incl. a dedicated OVERLAPPING-neighbour scenario proving the
integer atomicAdd is order-free, the P5 wall burn-through with the device-collected
destroyed list checked for SET equality + no drops/dupes, the P6 clamp), plus the
full-engine fire-backend-switch integration over 30 ticks. A non-zero exit or a
missing PASS marker fails the test.

This is the fire solver's cross-GPU determinism gate — GPU-int == CPU-int. The P4
smoke scatter is the arc's first GPU non-saturating atomic scatter; integer + is
order-free so the per-neighbour deposit sum is bit-identical regardless of
thread/scheduler order. The destroyed list is collected via a device atomicAdd
counter; set-equality + length proves no slot was dropped or duplicated.
"""
from __future__ import annotations

import pytest

import cuda_harness


pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_s6_fire_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_s6_check, sys; sys.exit(cuda_s6_check.main())", timeout=300,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "S6_RESULT: PASS" in out, (
        f"CUDA-S6 fire did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
