"""EOS P6.7 gate (pytest) — the GPU trace-smoke advection bit-identity proof.

The P6.7 per-kernel P6 digest gate (docs/eos_p6_gpu_alignment_review.md §4, P6.7
row: trace-smoke re-port at the new once-per-tick cadence; resolves the P3
``physics_engine.cpp`` cadence assert). Gated by
``cuda_available(kernel="trace_smoke")`` — the P6.0 pending-set contract: this
test SKIPS without a CUDA build / runtime, and RUNS (never pinned-skips) now that
P6.7 has removed the "trace_smoke" key from ``EOS_P6_PENDING_KERNELS``.

When it runs, it executes the trace-smoke check in an isolated GPU subprocess
(cuda_harness): the isolated all-branch GPU-vs-CPU synthetic A/B (the diffusion
Laplacian + wind^2 fold + the INTEGER semi-Lagrangian back-trace with NEGATIVE-
displacement + DDA wall-clip + WSUM-near-floor renorm, plus 1xN/Nx1, all-solid,
all-vacuum, near-empty edge configs) AND the 120-tick blast+venting multi-room
REAL-engine trajectory (CPU smoke backend vs GPU smoke backend, per-tick byte-
compare on every gas plane + wind + T) AND the CPU-path golden. A non-zero exit
or a missing PASS marker fails the test.

THE RE-DERIVATION FINDING: the EOS refactor changed only the trace CADENCE (once
per tick on the corrected wind, not n_smoke-substepped); SmokeDynamics::step's
per-pass arithmetic is unchanged, so cuda_smoke.cu's smoke_step is bit-identical
at the new cadence — P6.7 wires the dispatch and re-proves it. This is the
trace-smoke path's cross-GPU determinism gate: GPU-int == CPU-int.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(kernel="trace_smoke"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_trace_smoke_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_trace_smoke_check, sys; sys.exit(cuda_trace_smoke_check.main())",
        timeout=600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "TRACE_SMOKE_RESULT: PASS" in out, (
        f"EOS P6.7 trace-smoke did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
