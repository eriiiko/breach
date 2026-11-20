"""#29 P7e.1 -- planes and latches resident (pytest wrapper of
tests/cuda_unit_resident_check.py).

Property: the device unit slots read and write the batch's RESIDENT planes in
place (``ResidentBatch.buffers``: sight_screen, heat_atten, light_atten,
ignition_armed, div_source refreshed at every entry upload of the played
resident tick and equal to the host mirror there; slot 9's burn-through, the
seam's tile probe and slot 9d writing the batch's own buffers equal to the
host law), the FLIGHT plane the grenade slots read is built on the device
equal to ``combat.flight_plane``, every unit kernel reads the per-env TICK
ROW (never per-launch host ticks) whose tile-change count the topology seam
bumps on the device and the HORDE stage then reads, and slots 9c / 9c3 read
the device CLAIMS LIST in place, first match first. Breaks if a face stages a
copy of a plane, a unit plane leaves the resident set or its entry upload,
the flight builder drops the blocks_thrown term, the seam's bump is lost, the
HORDE stage reads its serial elsewhere, or 9c ignores the claims list's
order.

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


def test_cuda_unit_slots_read_the_resident_planes_and_latches():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_resident_check, sys; sys.exit(cuda_unit_resident_check.main())",
        timeout=1800,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_RESIDENT_RESULT: PASS" in out, (
        f"the resident planes and latches gate did not pass.\nreturncode={proc.returncode}\n"
        f"{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
