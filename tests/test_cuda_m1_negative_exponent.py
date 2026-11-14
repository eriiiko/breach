"""M1 gate (pytest) — CPU/CUDA bit-identity at a NEGATIVE `heat_inv_shift`.

PROPERTY: the two temperature backends compute the SAME integer at every
exponent M1 makes expressible, including the ones no shipped material row
reaches yet. The determinism contract is tol 0, and M1 is the patch that makes
`heat_inv_shift` signed everywhere — so the exponents the shipped table cannot
produce are exactly the ones no existing gate covers.

BREAKS IF: one backend keeps the old `>> heat_inv_shift` (undefined behaviour at
a negative exponent — on x86 a shift by `s & 63`, which silently yields 0) while
the other takes the kit's signed twin; or the CPU deposit is widened to int64
and the CUDA twin is not, which is the asymmetry the design's section 5 row 4
names as the one real risk.

SKIPS cleanly without a CUDA build / device. See
`tests/cuda_m1_negative_exponent_check.py` for the four parts and their
non-vacuity assertions.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_m1_negative_exponent_cpu_cuda_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_m1_negative_exponent_check as c, sys; sys.exit(c.main())",
        timeout=600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "M1_RESULT: PASS" in out, (
        f"M1 CPU/CUDA parity did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
