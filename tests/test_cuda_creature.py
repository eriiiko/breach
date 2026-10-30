"""#106 P6d1 -- a limp octopus is backend-blind (pytest wrapper).

SKIPS cleanly without a CUDA build / device, or when the CUDA build predates
the arm chain solver (stale: rebuild cpp/build_cuda). Otherwise runs
tests/cuda_creature_check.py in an isolated CUDA subprocess (cuda_harness --
never the CUDA .pyd in this process): a flooded room with one limp octopus,
48 ticks on the CPU, ``--cuda`` and ``--resident`` paths in lockstep, the field
digest, the tick digest and the ``__creature__`` section identical at tol 0
every tick. Breaks if a GPU path changes anything the creature reads or writes.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("creature"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_limp_octopus_is_identical_on_cpu_cuda_and_resident():
    proc = cuda_harness.run_cuda_script(
        "import cuda_creature_check, sys; sys.exit(cuda_creature_check.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    if "CREATURE_RESULT: SKIP" in out:
        pytest.skip(out.split("CREATURE_RESULT: SKIP", 1)[1].strip().splitlines()[0])
    assert "CREATURE_RESULT: PASS" in out, (
        f"creature backend gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
