"""EOS P6.6 gate (pytest) — the GPU unified-temperature bit-identity proof.

SKIPS cleanly without a CUDA build / device (per-kernel key: "conduction" on the
P6.0 per-kernel pending-set contract). When the GPU build is present, runs the
P6.6 check in an isolated subprocess (cuda_harness): the isolated
GPU-vs-CPU-reference comparison over synthetic + edge configs that drive every
pass (Pass 0 zero-vacuum + SL advection, Pass 1 solid/gas radiant deposit across
the three N regimes with the T_MAX_PHYS rail FORCED in both branches, Pass 2
conduction, Pass 3 cooling — rail-hit count asserted bit-equal), AND a 120-tick
hot-core-vs-cold-hull thin-gas trajectory with per-tick byte-identity on
`temperature` and rail-hit parity. A non-zero exit or a missing PASS marker fails.

Every pass is a per-cell / gather single-writer kernel over frozen inputs (each
cell reads neighbours/snapshots, writes only its own T —
docs/eos_p6_gpu_alignment_review.md §1.5), so bit-identity needs no restructuring;
the only cross-pass read-after-write is at the pass boundaries, reproduced by the
kernel launch barriers. This gate is the proof.
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
    not cuda_harness.cuda_available("conduction"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_p66_conduction_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_conduction_check, sys; sys.exit(cuda_conduction_check.main())",
        timeout=600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "P66_RESULT: PASS" in out, (
        f"EOS P6.6 conduction did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
