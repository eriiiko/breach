"""#29 P7d -- the horde gate (pytest wrapper of tests/cuda_unit_horde_check.py).

Property: slot 5 on the device (the applier's HORDE stage) is
ai_zombie.update_zombies_tick integer for integer -- the noise list drained
at its head, the request latch and the no-prey early returns leaving the
field cache standing, the cache's refresh rule (a refresh tick, a topology
change) and its reuse, the trigger and the chain wide, the hearing with its
noise_fields_max keep, pass 3 serial in row order with the bite (its events,
its credit, the conversion flag), the walk on the cached fields, the fear of
fire, the door bash through the wall-chew seam with its noise, the sprint, the
parity tie order and the seedless wander -- and its block-wide flood equals
the C++ reference flood cell for cell; on the horde scenes, ship_demo's and
playground's matches, in a batch of differing envs as alone, padding
untouched, every capacity refused by name, and the chained twin (alone and
beside the shooting, melee, resolve and door slots) equal to the untouched
reference every tick. Breaks if a branch of the law, the flood, the cache's
refresh or its life across ticks, the drain, the latch, the bite, the bash,
a capacity or the batch indexing differs from the reference.

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


def test_cuda_horde_on_the_device_equals_the_reference():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_horde_check, sys; sys.exit(cuda_unit_horde_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_HORDE_RESULT: PASS" in out, (
        f"the horde gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
