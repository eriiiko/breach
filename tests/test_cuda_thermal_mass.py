"""THERMAL-MASS AXIS P2 gate (pytest) — the CPU<->CUDA lockstep proof.

Gate (d) of docs/thermal_mass_axis_design_2026-07-25.md §3 (build addendum
2026-07-30 §3, "P2 (CUDA)"): the GPU mirror of P1's six MEDIUM tests
(`solid` -> `thermal_solid`) is bit-identical to the CPU solver at TOLERANCE
ZERO on a FURNITURE-BURN scenario — the case where the two masks differ and the
pre-P2 kernel was wrong — for BOTH the step path and the GPU-resident path.

SKIPS cleanly without a CUDA build / device; otherwise runs the check in an
isolated subprocess (cuda_harness), because a single interpreter can import
`breach_physics` only once and the rest of the suite imports the CPU build.
"""
from __future__ import annotations

import pytest

import cuda_harness


pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("conduction"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_thermal_mass_axis_cpu_cuda_lockstep():
    proc = cuda_harness.run_cuda_script(
        "import cuda_thermal_mass_check, sys; sys.exit(cuda_thermal_mass_check.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "THERMAL_MASS_RESULT: PASS" in out, (
        f"thermal-mass axis P2 lockstep did not pass.\n"
        f"returncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
