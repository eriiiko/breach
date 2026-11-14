"""#88 P2 -- V6 / V9 on CUDA (pytest wrapper): the counted int32 saturation and
its books at every device site -- stage 3b chained, stage 3b resident, the
soot mint (isolated and live through the runner) -- bit-identical to the CPU,
with the §5.1 identity closing on the GPU world and the saturation binding.

Runs ``cuda_trace_saturation_check`` in an isolated GPU subprocess
(cuda_harness). SKIPS only without a CUDA build (the C4 rule: the report shows
zero CUDA skips). A non-zero exit or a missing PASS marker fails the test.

PROPERTY: GPU-int == CPU-int on the saturation and its books. BREAKS IF: a
device site wraps, clamps unbooked, saturates per face, books into a slot
that is not appended / reset / added, or the GPU combustion loses ``eos=``.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(kernel="trace_saturation"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_trace_saturation_cpu_equals_gpu():
    proc = cuda_harness.run_cuda_script(
        "import cuda_trace_saturation_check, sys; "
        "sys.exit(cuda_trace_saturation_check.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "TRACE_SAT_RESULT: PASS" in out, (
        f"the CUDA saturation gate did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
