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

# ---------------------------------------------------------------------------
# CUDA PARITY SUSPENDED, P-G1a -> P-G2 (gas-energy conservation arc #54,
# design §5 "CUDA parity is suspended from P-G1a until P-G2 -- named,
# time-boxed"). P-G1a rewrites the CPU EOS energy chain (the per-stage KE
# brackets, the face-flux energy step, the once-per-tick recovery, and the
# transport's energy half) and DELIBERATELY leaves the .cu twins on the old
# step-4c kernels, so every CPU-vs-GPU bit-identity check in this family is
# expected to diverge until P-G2 lands K1's brackets and the new K3 flux
# kernel. `strict=False` so a check that happens to still agree does not
# fail the suite -- P-G2 removes these marks and re-arms the gate.
pytestmark = pytest.mark.xfail(
    reason="P-G2 pending: CUDA twins of the energy step", strict=False)



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
