"""BC B4 gate (pytest) — the planetside AMBIENT ring CUDA-vs-CPU lockstep.

SKIPS cleanly without a CUDA build / device (the cuda_harness contract). When
the GPU build is present, runs cuda_ambient_check in an isolated subprocess: a
full-tick PhysicsRunner.step trajectory on an ambient map (SPACE ring + open-air
interior + an active u-damping band) run twice — flags off (CPU) vs the four EOS
kernel flags + the temperature flag on (the GPU chain) — asserting per-tick
bit-identity of every EOS field, all six digests, the five rail counters, AND
the per-plane boundary_flux rail, with a scripted ring-adjacent destroy_wall
(joins-ambient twin + rail vent). Proves the B4 .cu mirror of the whole BC arc
(B3a/b/c) is tol-0 identical to the CPU reference.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("eos_step"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_bc_ambient_ring_cuda_cpu_lockstep():
    proc = cuda_harness.run_cuda_script(
        "import cuda_ambient_check, sys; sys.exit(cuda_ambient_check.main())",
        timeout=1800,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "AMBIENT_RESULT: PASS" in out, (
        f"BC ambient CUDA/CPU lockstep did not pass.\n"
        f"returncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
