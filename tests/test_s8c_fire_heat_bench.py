"""S8c item 1 — fire-heat batched-cast PAYOFF gate (pytest).

SKIPS cleanly without a CUDA build / device. When the GPU build is present, runs
bench_s8c_fire_heat_check in an isolated interpreter (cuda_harness) and asserts
the batched device cast (cuda_raycaster_cast_batch) is both byte-identical on
`heat` to the per-source loop AND removes the per-tick round-trip tax that made
hundreds of burning tiles run at ~3 fps (2026-07-20 B5 feel-test):

  * batched > 3x faster than the per-source loop (throttle-robust: measured
    back-to-back at the same GPU clock/thermal state);
  * batched best-of-3 < 100 ms on a 600-fire firestorm (the ~3-fps scenario),
    with 30 fps / 33 ms named as the target.

The observed margins are ~55-280x (batched ~1 ms), so this gate has wide headroom
under laptop throttle; a FAIL means the per-source malloc/H2D/D2H tax is back.
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


def test_s8c_fire_heat_batched_cast_payoff():
    proc = cuda_harness.run_cuda_script(
        "import bench_s8c_fire_heat_check as c, sys; sys.exit(c.main())",
        timeout=600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "S8C_FIRE_BENCH_RESULT: PASS" in out, (
        "S8c fire-heat batched cast failed its payoff/identity gate "
        f"(ratio<3x, floor>100ms, or heat mismatch).\n"
        f"returncode={proc.returncode}\n{out}"
    )
