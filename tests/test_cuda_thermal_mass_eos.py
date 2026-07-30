"""THERMAL-MASS AXIS **P-EOS** gate (pytest) — the CPU<->CUDA lockstep proof.

Gate (d) of docs/thermal_mass_eos_ruling_2026-07-30.md §3: the GPU mirror of the
ruling's EOS/combustion edits (step-1b's skipped T write + its T-only occluder
mask, step-4c's skipped write, the combustion deposit's OBJECT conversion) is
bit-identical to the CPU at TOLERANCE ZERO on a FURNITURE-BURN scenario — the
case where `thermal_solid != solid` and the pre-patch kernels were wrong — for
BOTH the step path and the GPU-resident path, with the non-vacuousness controls
P2 established (the same kernels driven WITHOUT the mask must DIVERGE).

Sibling of test_cuda_thermal_mass.py (P2, the TemperatureSolver mirror).

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


def test_thermal_mass_eos_cpu_cuda_lockstep():
    proc = cuda_harness.run_cuda_script(
        "import cuda_thermal_mass_eos_check, sys; "
        "sys.exit(cuda_thermal_mass_eos_check.main())",
        timeout=1200,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "THERMAL_MASS_EOS_RESULT: PASS" in out, (
        f"thermal-mass axis P-EOS lockstep did not pass.\n"
        f"returncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
