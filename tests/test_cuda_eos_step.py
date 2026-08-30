"""EOS P6.5 gate (pytest) — the chained full-eos.step GPU dispatch proof.

SKIPS cleanly without a CUDA build / device (per-kernel key: "eos_step" on
the P6.0 per-kernel pending-set contract). When the GPU build is present,
runs the P6.5 check in an isolated subprocess (cuda_harness): a 120-tick
breach-to-vacuum + blast trajectory through the REAL engine
(PhysicsEngine::run_substeps) run twice — flags off (CPU) vs all four EOS
kernel-surface flags on (every tick dispatched to the chained GPU
orchestration, cuda_eos_step.cu; dispatch-fired proven via
eos_step_cuda_calls) — asserting per-tick bit-identity of every EOS-owned
field (wind_x/wind_y/temperature/atmosphere/p_prev/all gas planes), all six
solver digests (advect/bulk_flux/pstar/helmholtz/velocity/compression), the
five rail counters, and the schedule telemetry — AND the committed
default-scenario golden on the CUDA build's CPU path (the dispatch is
strictly additive). A non-zero exit or a missing PASS marker fails the test.

P6.1–P6.4 proved each kernel surface in isolation; this gate proves the
CHAINED composition (the substep-loop interleave with device-resident
intermediates, the solve on the chain's own p*/div_u, the kick on the
chain's own solved P) is the CPU tick, bit for bit.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("eos_step"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_p65_eos_step_chained_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_eos_step_check, sys; sys.exit(cuda_eos_step_check.main())",
        timeout=1800,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "P65_RESULT: PASS" in out, (
        f"EOS P6.5 chained eos.step did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
