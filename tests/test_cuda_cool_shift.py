"""COOL-SHIFT AXIS gate (pytest) — the CPU<->CUDA lockstep proof.

Gate (d) of the cool-shift axis (2026-07-30): the per-tile ambient-decay shift
(`GameMap.cool_shift`) and the vacuum OFFSET rule derived from it are mirrored
bit-exactly on the GPU (`cuda_temperature.cu` `temp_cool`, MEDIUM-TEST SITE
6/6), at TOLERANCE ZERO, on a map whose per-tile shifts are NON-UNIFORM — the
case a stale kernel would miss — for BOTH the step path and the resident path.

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


def test_cool_shift_axis_cpu_cuda_lockstep():
    proc = cuda_harness.run_cuda_script(
        "import cuda_cool_shift_check, sys; sys.exit(cuda_cool_shift_check.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "COOL_SHIFT_RESULT: PASS" in out, (
        f"cool-shift axis lockstep did not pass.\n"
        f"returncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
