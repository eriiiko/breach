"""#29 P7a.3 -- the sprays gate (pytest wrapper of tests/cuda_unit_sprays_check.py).

Property: slot 4b (``combat.process_sprays``) on the device unit store is the
reference integer for integer -- the cone (``spray_cone_tiles`` over the
PHYSICAL sight: its tile set, order and divisors on synthetic cones over four
real maps), the own-row writes, one SPRAY_CELLS record per depositing sprayer
appended to the effect list after slot 4's and 4a's, its cells in THE CELL
BUFFER -- and the SPRAY_CELLS effect at the resolve is ``_enqueue_spray_cells``:
the door's rows equal ``pack_for_device`` of the host's edits (absolute seq,
the sort key appended) and the claims list holds every claim in order; equal
in a batch of differing envs as alone, padding untouched, every capacity
refused by name, the WEGO burst and an interleave refused, and the chained
twin (4b + 4c on the device, the rows replayed through the host's enqueue)
equal to the untouched reference every tick. Breaks if the cone's root,
angle test, range, nozzle rule, sight form or tile order, a spray row, the
countdown, the record's fields or span, an edit's amount / field / slice /
source / key, the claims' order or channel, a capacity check or the batch
indexing differs from the law.

SKIPS cleanly without a CUDA build / device; runs the check in an isolated
CUDA subprocess (cuda_harness).
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_cuda_sprays_and_their_cells_equal_the_reference():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_sprays_check, sys; sys.exit(cuda_unit_sprays_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_SPRAYS_RESULT: PASS" in out, (
        f"the sprays gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
