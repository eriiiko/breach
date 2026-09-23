"""Issue #7 gate (pytest) — the O2 law's pressure factor, CPU <-> CUDA at tol 0.

PROPERTY (brief §4.4): both reads of the O2 law — the fire ODE's `o2f` and
combustion's claim gate `o2f_j` — compute the SAME pressure factor g(p) on the
CPU and the GPU, bit for bit, in every regime (below p_ext, on the ramp, at
and above p_full, exactly on each edge), and the LIVE fire backend
(`main.py --cuda`) receives the bound edges and the `atmosphere` plane.
BREAKS IF: a CUDA twin drops or mis-transcribes the factor, the pressure
gather, the plane upload, or the edges' plumbing (physics_engine.cpp ->
cuda_fire_step, physics_runner.py -> cuda_combustion_step).

Runs tests/cuda_o2_pressure_check.py in an isolated GPU subprocess
(tests/cuda_harness.py); skips cleanly without a CUDA build / device. A
non-zero exit or a missing PASS marker fails the test.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_o2_pressure_factor_cpu_gpu_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_o2_pressure_check, sys; "
        "sys.exit(cuda_o2_pressure_check.main())",
        timeout=600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "O2P_RESULT: PASS" in out, (
        f"#7 pressure factor CPU/GPU gate did not pass.\n"
        f"returncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
