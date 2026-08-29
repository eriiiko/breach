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
