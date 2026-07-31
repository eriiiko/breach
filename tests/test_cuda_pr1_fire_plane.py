"""P-R1 gate (a), CUDA half (pytest) — cuda_raycaster_cast_from_fire_plane's
``heat`` byte-identity vs the pre-patch per-tile loop
(docs/radiation_raycaster_extinction_ruling_2026-07-31.md A4.1-A4.2).

SKIPS cleanly without a CUDA build / device. When the GPU build is present,
runs cuda_pr1_fire_plane_check in an isolated anaconda/data-env subprocess
(cuda_harness): a 600-fire synthetic firestorm AND the real "playground"
level over several ticks of evolving fire, each cast four ways (old
per-tile-loop-CPU, old per-tile-loop-CUDA via cuda_raycaster_cast_batch, new
cast_from_fire_plane CPU, new cuda_raycaster_cast_from_fire_plane) and
asserted byte-for-byte equal (tol 0). A non-zero exit or a missing PASS
marker fails the test.
"""
from __future__ import annotations

import pytest

import cuda_harness


pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_pr1_fire_plane_cast_heat_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_pr1_fire_plane_check as c, sys; sys.exit(c.main())",
        timeout=300,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "PR1_FIRE_PLANE_RESULT: PASS" in out, (
        "P-R1 cast_from_fire_plane did not pass its CUDA byte-identity gate.\n"
        f"returncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
