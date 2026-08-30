"""CUDA-S2 LIVE gate (pytest) — the raycaster wired into the live fire->heat cast.

SKIPS cleanly without a CUDA build / device. When the GPU build is present, runs
cuda_s2b_raycaster_live_check in an isolated anaconda-3.11 subprocess (cuda_harness):

  PART 1 — the production PhysicsRunner.cast_fire_heat run with the raycaster
  backend OFF vs ON on a multi-source firestorm; the resulting gmap.heat must be
  byte-for-byte equal (the live wiring preserves the per-tick clear + per-source
  saturating accumulate the S2 march gate proved in isolation).

  PART 2 — the default scenario stepped 30 ticks with ALL backends ON vs the
  pure CPU path; EVERY synced field (incl. heat + temperature, which the raycaster
  feeds) bit-identical (tol 0), and the CPU path still reproduces the golden. This
  is the proof that `--cuda` is a full 7/7.

A non-zero exit or a missing PASS marker fails the test.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_s2_raycaster_live_heat_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_s2b_raycaster_live_check as c, sys; sys.exit(c.main())",
        timeout=420,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "RAYCASTER_LIVE_RESULT: PASS" in out, (
        "CUDA-S2 live raycaster did not pass (heat bit-identity / 7-of-7 "
        f"integration).\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
