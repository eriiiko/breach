"""#31 Patch B gate (pytest wrapper): the blast's divergence source, GPU == CPU.

SKIPS cleanly without a CUDA build / device. When the GPU build is present, runs
tests/cuda_div_source_check.py in an isolated subprocess (cuda_harness).
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("div_source"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_the_blast_source_is_bit_identical_per_call_and_resident():
    """PROPERTY: on a blast tick, a saturated source and a blast on a wall tile, the
    per-call and the resident EOS match the CPU at tol 0 on every synced field.

    BREAKS IF: either GPU path drops the source, applies it before the mask, or
    subtracts it without the widening clamp the CPU uses.
    """
    proc = cuda_harness.run_cuda_script(
        "import cuda_div_source_check, sys; "
        "sys.exit(cuda_div_source_check.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "DIV_SOURCE_RESULT: PASS" in out, (
        f"the div_source gate did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
