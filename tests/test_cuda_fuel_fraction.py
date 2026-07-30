"""FUEL-FRACTION AXIS gate (pytest) — the CPU<->CUDA lockstep proof.

Gate (c) of the fuel-fraction axis (2026-07-30): the fire logistic's fuel term
now divides by the tile's OWN material hp via the per-tile ``GameMap.fuel_recip``
reciprocal plane, and the GPU fire kernel (``cuda_fire.cu``) must read the same
plane and produce the same bits — at TOLERANCE ZERO, on a map whose fuel is
genuinely NON-UNIFORM (wood hp 60 beside crates hp 30), for BOTH the step path
and the resident path.

SKIPS cleanly without a CUDA build / device; otherwise runs the check in an
isolated subprocess (cuda_harness), because a single interpreter can import
``breach_physics`` only once and the rest of the suite imports the CPU build.
"""
from __future__ import annotations

import pytest

import cuda_harness


pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("fire"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_fuel_fraction_axis_cpu_cuda_lockstep():
    proc = cuda_harness.run_cuda_script(
        "import cuda_fuel_fraction_check, sys; "
        "sys.exit(cuda_fuel_fraction_check.main())",
        timeout=1200,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "FUEL_FRACTION_RESULT: PASS" in out, (
        f"fuel-fraction axis lockstep did not pass.\n"
        f"returncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
