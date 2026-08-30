"""CUDA-S0 gate (pytest) — the GPU hello-world bit-identity proof.

SKIPS cleanly when there is no CUDA build / no device (so the suite stays green
on a CPU-only checkout or CI box). When the GPU build IS present, it runs the
bit-identity check in an ISOLATED anaconda-3.11 subprocess (cuda_harness) — never
importing the CUDA module into this pytest process, which has already imported
the CPU build. A non-zero exit or a missing PASS marker fails the test.

This is the committed S0 gate: it asserts the integer toolkit, compiled for the
device, is byte-identical to the CPU on real hardware.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_s0_hello_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_s0_check, sys; sys.exit(cuda_s0_check.main())"
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "S0_RESULT: PASS" in out, (
        f"CUDA-S0 hello-world did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
