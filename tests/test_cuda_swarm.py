"""arc #63 P3 -- the larva kernel's 3-part CUDA gate (pytest wrapper).

PROPERTY (docs/swarm_P3_impl.md §11, v4.7): the larva law's three loops --
the CPU twin (THE reference), the per-call CUDA wrapper and the resident
launch core -- are byte-identical on a forced-branch fuzz with per-env fields
and live-looking padding (PART 1), over a 130-tick facade-driven lockstep
(PART 2), and on the played sim paths CPU / per-call / resident with scripted
spawn, wall and reclaim (PART 2b); the canonical golden holds on the CUDA
build's CPU path (PART 3). Changes that must break it: a kernel, wrapper or
transfer that differs from the CPU law; a loop bound other than hw[e]; a
dropped valid check; records copied beyond hw or after the books sync; a
stale device store after a facade write or a host-path tick.

SKIPS cleanly without a CUDA build / device; runs tests/cuda_swarm_check.py
in the cuda_harness subprocess.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("swarm"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_larva_kernel_three_loops_are_byte_identical():
    proc = cuda_harness.run_cuda_script(
        "import cuda_swarm_check as m, sys; sys.exit(m.main())",
        timeout=600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "SWARM_P3_RESULT: PASS" in out, (
        f"larva CUDA gate failed.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
