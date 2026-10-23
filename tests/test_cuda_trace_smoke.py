"""Smoke transport v2 (#12) V6 gate (pytest): both GPU paths -- the chained GPU
EOS (V6-chained, P2a) and the device-RESIDENT tick (V6-resident, P2b) -- carry
the trace planes exactly as the CPU does.

SKIPS cleanly without a CUDA build / device. When the GPU build is present, runs
``cuda_trace_smoke_check`` in an isolated GPU subprocess (cuda_harness): three
real-Simulation scenarios (a sealed blast with an air-less pocket, a breach to
space with a stranded deposit, a crate wall in a blast), each run on two
independently built worlds -- CPU vs the chained GPU orchestration (PART 1),
then CPU vs the resident tick with the resident trace tail (PART 3) --
asserting per-tick bit-identity of every gas plane (bulk AND trace), the gas
energy, wind, temperature, pressure, and every trace book (vent in
``boundary_flux()``'s trace slots, wipe, sink, decay; the EOS bulk-flux digest on
the chained leg); plus the CPU-path golden (PART 2). A non-zero exit or a
missing PASS marker fails the test.

PROPERTY: GPU-int == CPU-int on the trace law (bulk_transport stages 3b/3c +
trace_tail, and their device twins in cuda_bulk_transport.cu). BREAKS IF: a
device stage-3b/3c or tail arithmetic, face order or participation predicate
differs from the CPU's; stage 3b/3c move across stage 4's d_nb
re-accumulation; the trace D2H lands after digest_bulk_flux; a trace counter is
not reset per tick; the resident tail is dropped, reordered, or books onto the
wrong channel.

(This file once gated the per-call GPU semi-Lagrangian smoke step of P6.7; that
step and its law are gone -- smoke transport v2, #12.)
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
        f"V6-chained trace transport did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
