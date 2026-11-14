"""#29 P6.1 -- the device unit store gate (pytest wrapper of
tests/cuda_unit_store_check.py).

Property: the (N, U, W) device unit store returns any rows bit for bit at
N = 1, 2 and 8, and its pilot kernel (slot 1c, exposure decay) equals the
reference ``exposure.decay_units`` row for row at tol 0 on every tick of a
live ship_demo bots match (its coverage counter equal to the rows the host
changed and > 0 on part of the window) and of batched focus-fire arenas at
N = 2 and 8 (env i of the batch == env i alone), while the hooked match
plays exactly as an unhooked one. Breaks if the kernel's decay differs from
the law, it touches a padding row or another env's rows, a copy loses or
reorders bytes, or hooking a slot changes the game.

SKIPS cleanly without a CUDA build / device; runs the check in an isolated
CUDA subprocess (cuda_harness).
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_cuda_unit_store_round_trip_and_exposure_lockstep():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_store_check, sys; sys.exit(cuda_unit_store_check.main())",
        timeout=1800,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_STORE_RESULT: PASS" in out, (
        f"the device unit store gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
