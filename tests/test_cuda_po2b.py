"""P-O2b gate (pytest) — the extended oxygen draw, CPU<->CUDA bit-identity.

WRAPPER ADDED BY audit Patch A / A4 (2026-08-04). ``cuda_po2b_check.py`` has
been complete and correct since P-O2b landed, but nothing referenced it, so
``pytest tests -q`` never collected it and it had NEVER RUN under the suite.
That left the SHIPPED ``draw_r = 2`` combustion path with zero collected GPU
parity coverage — ``cuda_combustion_check.py`` predates the extended draw and
calls the pass with its pre-P-O2b defaults (draw_r == 1, no dem_acc, no heat),
so it covers only the byte-identical legacy path.

The check proves, in one isolated subprocess: the R = 1 identity against the
pre-P-O2b default call; CPU == GPU at draw_r 1, 2 and 3 over a multi-tick
trajectory built to hit every branch (contested claimants, attenuating
permeable crates, blocking walls, a vacuum pocket, a burning source tile,
out-of-bounds edge slots); and order-freedom against the GPU's arbitrary
grid-stride execution order — all at tol 0 including the persistent
``dem_acc`` plane.

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


def test_po2b_extended_draw_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_po2b_check, sys; sys.exit(cuda_po2b_check.main())",
        timeout=420,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "PO2B_RESULT: PASS" in out, (
        f"CUDA P-O2b extended draw did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
