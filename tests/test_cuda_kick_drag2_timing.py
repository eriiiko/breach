"""Drag-law v2 (docs/drag_law_v2_design_2026-08-23.md §7/§8 gate 5) — the
k_drag2 timing-budget gate (pytest wrapper). SKIPS cleanly without a CUDA
build/device. Runs cuda_kick_drag2_timing_check in an isolated subprocess
(cuda_harness): the CPU-tick leg AND the isolated-CUDA-call leg, both
armed-and-windy per design §7, both asserted under a 3% tick-budget bound.
A non-zero exit or a missing PASS marker fails the test.
"""
from __future__ import annotations

import pytest

import cuda_harness


pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("kick_compression"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_drag2_timing_budget():
    proc = cuda_harness.run_cuda_script(
        "import cuda_kick_drag2_timing_check as m, sys; sys.exit(m.main())",
        timeout=300,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "DRAG2_TIMING_RESULT: PASS" in out, (
        f"drag-law v2 k_drag2 timing budget not met.\n"
        f"returncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
