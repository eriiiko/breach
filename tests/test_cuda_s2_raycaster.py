"""CUDA-S2 gate (pytest) — the GPU directional raycaster HEAT bit-identity proof.

SKIPS cleanly without a CUDA build / device. When the GPU build is present, runs
the S2 check in an isolated anaconda-3.11 subprocess (cuda_harness): a firestorm
+ smoke scenario cast on BOTH the CPU (Raycaster.cast_source_directional) and the
GPU (cuda_raycaster_cast), asserting the Q16.16 `heat` buffer is byte-for-byte
equal (tol 0). The render channels are deterministic-exempt and NOT gated. A
non-zero exit or a missing PASS marker fails the test.

This is the raycaster's gameplay output (heat -> temperature -> unit damage) made
bit-identical GPU == CPU, mirroring the CUDA-S1 temperature gate.
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


def test_s2_raycaster_heat_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_s2_check, sys; sys.exit(cuda_s2_check.main())", timeout=300,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "S2_RESULT: PASS" in out, (
        f"CUDA-S2 raycaster heat did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
