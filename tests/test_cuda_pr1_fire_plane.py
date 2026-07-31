"""P-R4 gates (d) + (g), CUDA half (pytest) — the radiation-law lockstep.

REWRITTEN AT P-R4 (documented re-anchor; see tests/cuda_pr1_fire_plane_check.py
for the full note). This wrapper used to assert the P-R1 ``heat`` byte-identity
gate, whose oracle was the retired PAINTER
(docs/radiation_raycaster_extinction_ruling_2026-07-31.md A1). It now runs the
radiation witness:

  * gate (d) — ``rad_net`` bit-identical CPU vs CUDA at TOLERANCE ZERO on three
    scenarios: a 600-emitter firestorm, the EQUAL-T pair (exactly 0 on both
    backends), and a hot/cold pair with the FLUX LIMITER engaged;
  * gate (g) — the 600-emitter batched device cast inside its 3.0 ms budget
    (2x the 1.5 ms S8c painter baseline).

SKIPS cleanly without a CUDA build / device. A non-zero exit or a missing PASS
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
    """P-R4: rad_net CPU<->CUDA at tol 0, + the 600-emitter cost budget.

    (Test NAME kept so the suite's failure-set comparison stays anchored across
    the re-anchor; the ORACLE inside is the radiation law, not the painter.)"""
    proc = cuda_harness.run_cuda_script(
        "import cuda_pr1_fire_plane_check as c, sys; sys.exit(c.main())",
        timeout=300,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "PR4_RADIATION_RESULT: PASS" in out, (
        "P-R4 radiation exchange did not pass its CUDA lockstep/cost gate.\n"
        f"returncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
