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
