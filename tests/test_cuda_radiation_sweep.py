"""P4 gate (pytest) — the radiation sweep's CUDA twin, CPU/GPU tol 0
(ray-engine-v2, design v3 §2.9 gate 7).

PROPERTY: the GPU sweep (cpp/src/cuda_radiation_sweep.cu — the per-call path
PhysicsEngine.step_tail dispatches, and its (N, h, w) launch core) computes the
SAME integers as RadiationSweep::run on every input the engine can hand it:
the four output planes, the Fleck plane and the stream telemetry, on gate 0's
scene matrix, the derived and the per-cell ambient (the vacuum ring included),
thin rows, a burning tile on the live calibration, stamped bodies, degenerate to
full-size grids, ingress rejections, an N = 3 batch, and the live conductor
(the step path with only the radiation backend flipping, a cold sky, and the
resident tick with every backend on but combustion). Gate 0 holds the CPU to
the integer reference, so this chains the GPU to the reference.

BREAKS IF: the .cu transcribes any term of design §2.3 differently, a wavefront
launch stops being a topological order of an ordinate's dependency DAG, the
per-cell books stop being order-free integer sums, or the (N, h, w) indexing
leaks one env into another.

Runs in an isolated subprocess (tests/cuda_harness.py — the CUDA .pyd is never
imported into pytest); SKIPS cleanly without a CUDA build / device. See
tests/cuda_radiation_sweep_check.py for the nine parts and their non-vacuity
assertions.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_radiation_sweep_cpu_cuda_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_radiation_sweep_check as c, sys; sys.exit(c.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "RS_RESULT: PASS" in out, (
        f"the radiation sweep's CPU/CUDA parity did not pass.\n"
        f"returncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
