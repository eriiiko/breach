"""Sky-exchange gate (pytest) — resident vs CPU bit-identity (design gate e).

WRAPPER ADDED BY audit Patch A / A4 (2026-08-04). ``cuda_sky_exchange_check.py``
has been complete and correct since the sky-exchange arc landed, but nothing
referenced it, so ``pytest tests -q`` never collected it and it had NEVER RUN
under the suite.

The check proves a full GPU-RESIDENT tick with the sky pass ACTIVE stays
bit-identical to a full CPU tick — the pass runs host-side on the numpy mirror
in both paths, and this is what pins that it introduces no CPU<->GPU
divergence. Scenario is a planetside ambient map (sky_tau_s = 60 s) with a fire
seed and a hand-depleted O2 patch, A/B over N ticks on two independently built
worlds, tol 0 on every synced field each tick — plus the suite's
non-vacuousness controls (sky_flux went non-zero; residency actually live).

Skips cleanly without a CUDA build / device.
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


def test_sky_exchange_resident_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_sky_exchange_check, sys; sys.exit(cuda_sky_exchange_check.main())",
        timeout=420,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "SKY_RESULT: PASS" in out, (
        f"CUDA sky-exchange lockstep did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
