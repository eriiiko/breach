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
