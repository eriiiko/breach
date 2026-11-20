"""#29 P5 -- the resident tick's device chain captures in global mode (pytest wrapper).

PROPERTY: every launch, copy and memset of the tick's device chain runs on the
one tick stream (no legacy-stream operation, no synchronous copy, no
allocation in a steady-state tick), and the chain's ONLY host syncs are the
two named reads -- the EOS stage's max(n_sub) and the tail's books D2H;
capturing the chain changes nothing (captured tick == uncaptured twin, bit
for bit). A change that must break it: a kernel on the default stream, a
stream-less cudaMemcpy / cudaMemset, a cudaMalloc in steady state, a new host
sync in the chain. Validated by launching K_store_atm on the default stream:
the capture failed ("operation would make the legacy stream depend on a
capturing blocking stream").

SKIPS cleanly without a CUDA build / device; runs
tests/cuda_tick_capture_check.py in the cuda_harness subprocess.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("tick_capture"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_resident_tick_captures_in_global_mode():
    proc = cuda_harness.run_cuda_script(
        "import cuda_tick_capture_check as m, sys; sys.exit(m.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "TICK_CAPTURE_RESULT: PASS" in out, (
        f"the tick capture failed.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
