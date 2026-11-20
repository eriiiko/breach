"""#29 P5 -- batch invariance: env i of N == env i alone (pytest wrapper).

PROPERTY: an env's evolution in the batched resident engine is independent of
the batch it rides in -- every GameMap array and every engine book of env i,
after every tick, equals the same env stepped alone on the played resident
path, for N in {1, 2, 8, 64}, with envs on different templates, seeds and
substep counts (the run proves the counts differ on the same tick), one env
with topology events (incl. the withheld-stamp-rows path) and one env reset
mid-batch; and for real ship_demo Simulations with their entities on. Changes
that must break it: an env-stride bug in any (N, h, w) kernel, an unmasked
substep in an env whose n_sub is below the batch's max, a book folded into
the wrong env's engine. Validated by breaking it twice (an ignored substep
mask in sl_advect3; an un-rebased temperature pointer in K_pstar): both
caught (docs/residency_benchmark_2026-11-07.md, P5 section).

SKIPS cleanly without a CUDA build / device; runs
tests/cuda_batch_invariance_check.py in the cuda_harness subprocess.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("batch_invariance"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_env_in_batch_equals_env_alone():
    proc = cuda_harness.run_cuda_script(
        "import cuda_batch_invariance_check as m, sys; sys.exit(m.main())",
        timeout=1800,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "BATCH_INVARIANCE_RESULT: PASS" in out, (
        f"batch invariance failed.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
