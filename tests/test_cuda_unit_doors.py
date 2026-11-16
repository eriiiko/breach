"""#29 P6b.1 -- the doors gate (pytest wrapper of tests/cuda_unit_doors_check.py).

Property: the device door table IS the sim's doors (unpack(pack) door for
door, its entity rows the serializer's); the device seal_tiles /
unseal_tiles / can_seal_tiles, inject / extract / transfer_gas_n and
_aperture_blocked are GameMap's integer for integer on real maps with real
gas -- every plane, every book, the closure -- and every refusal the Python
raises before it mutates is refused on the device with nothing written; slot
9e(d)'s device sweep is door_system.sweep_doors on the table (ordinal order,
the occupancy test, external destruction by the whole-door rule, the hp copy
at the open and the restamp at the close) and slot 1b's USE is the latch's
flip; in a batch of differing envs as alone, padding rows untouched, every
capacity refused by name, and the chained twin (the tables landed back on the
door runtimes) equal to the untouched reference every tick. Breaks if a seal
or unseal plane family, the trace saturation, an energy move, the close-T
mean, a refusal, a primitive's split, the occupancy test, the ordinal order,
the whole-door rule, the hp copy or restamp, the use write or the batch
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


def test_cuda_doors_on_their_table_equal_the_reference():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_doors_check, sys; sys.exit(cuda_unit_doors_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_DOORS_RESULT: PASS" in out, (
        f"the doors gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
