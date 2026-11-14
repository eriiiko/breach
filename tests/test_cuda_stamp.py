"""#29 P4b — the stamp kernel gate (pytest wrapper of tests/cuda_stamp_check.py).

Property: the device stamp (``stamp_launch_resident``) produces the five stamp
planes -- obstacles, dyn_permeability, dyn_wave_absorb, dyn_heat_atten_q,
dyn_light_atten_q -- byte for byte equal to the CPU stamp's, from the host's
rows alone, on every tick of an empty map, a crowd of overlapping units of
different size / permeability / absorption / extinction moving over walls and
the map edge, the same with walls destroyed mid-run, and a dense contention
scene -- with the device planes poisoned before each run so a kernel that
writes nothing cannot pass. Breaks if a combine direction flips, the last row
stops winning the permeability write, a reset is skipped, a footprint is off
by one, or a light channel is dropped.

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


def test_cuda_stamp_planes_equal_the_cpu_stamp_on_every_tick():
    proc = cuda_harness.run_cuda_script(
        "import cuda_stamp_check, sys; sys.exit(cuda_stamp_check.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "STAMP_RESULT: PASS" in out, (
        f"the stamp kernel gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
