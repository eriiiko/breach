"""arc #63 -- the swarm store rides ResidentBatch's one batch axis (pytest wrapper).

PROPERTY (Erik's P6 ruling, docs/unit_design_2026-11-14.md §6.3): the swarm
arrays are per map and ResidentBatch supplies N -- a batch of N >= 2 real
Simulations stacks every swarm resident array to (N,) + the map's own shape,
slice e holds env e's store, a spawn into env 1 uploaded with env 1's own
from_host moves no other env's slice or carrier, and env 0's tick_digest is
the same whether or not env 1 carries a swarm. Changes that must break it: a
leading env axis back in the store (a second N), a bind that seeds slices
from the wrong map, an env-stride bug in a swarm upload, swarm state leaking
into another env's hashed state.

SKIPS cleanly without a CUDA build / device; runs
tests/cuda_swarm_batch_check.py in the cuda_harness subprocess.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("swarm_batch"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_swarm_slices_ride_the_one_batch_axis_independently():
    proc = cuda_harness.run_cuda_script(
        "import cuda_swarm_batch_check as m, sys; sys.exit(m.main())",
        timeout=1200,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "SWARM_BATCH_RESULT: PASS" in out, (
        f"swarm batch axis failed.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
