"""#29 P4a — the bus gate (pytest wrapper of tests/cuda_resident_bus_check.py).

Property: in slot 7 of a resident tick of the real match (ship_demo seed 1,
bots + horde, headless), the GameMap moves across the bus exactly twice --
ONE ``from_host`` before every device stage and ONE ``to_host`` after the last
of them -- with zero GameMap transfers between (the residency design's P4a
row). Breaks if a resident stage regains a host bracket, if the entry or the
exit splits into several calls, or if a device stage moves outside the
upload / download window.

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


def test_cuda_resident_slot7_moves_planes_only_at_its_ends():
    proc = cuda_harness.run_cuda_script(
        "import cuda_resident_bus_check, sys; "
        "sys.exit(cuda_resident_bus_check.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "RESIDENT_BUS_RESULT: PASS" in out, (
        f"the resident bus gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
