"""#29 P5 -- the device pre-stage == the host pre-stage at N = 1 (pytest wrapper).

PROPERTY: the resident EOS stage's DEVICE pre-stage (the per-env reductions,
the per-env n_sub fold into a fold-table row, the device-built cap² / absorb /
coeffE / coeffS / p_prev planes) equals the host pre-stage the per-call path
runs, bit for bit, on the same tick-entry state -- over a blast room, a sweep
that reaches every substep count the cap allows, and ship_demo's match. A
change that must break it: any device re-derivation of the fold that differs
from eos_host_prestage / eos_prestage_finish (a dropped count epsilon, a
missing t_amb floor, a cap² branch without the thermal-solid rule, a coeff
kernel reading another env's dt). Validated by breaking the fold once (the
run log in docs/residency_benchmark_2026-11-07.md, P5 section).

SKIPS cleanly without a CUDA build / device; runs
tests/cuda_prestage_parity_check.py in the cuda_harness subprocess.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("prestage_parity"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_device_prestage_equals_host_prestage():
    proc = cuda_harness.run_cuda_script(
        "import cuda_prestage_parity_check as m, sys; sys.exit(m.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "PRESTAGE_PARITY_RESULT: PASS" in out, (
        f"device pre-stage parity failed.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
