"""#29 P6b.2 -- the vents gate (pytest wrapper of tests/cuda_unit_vents_check.py).

Property: the device duct and vent tables ARE the sim's ducts and vents
(unpack(pack) entity for entity, their entity rows the serializer's); the
device inject_gas_n_vec is GameMap's integer for integer (each rail, both
sides of every floor, the trace saturation, every refusal with nothing
written) and the kit's floor division / floor product are Python's; slot
9e(d)'s device vent sweep is vent_system.sweep_vents on the tables (ducts in
ordinal order, returns then supplies, the guards, the Bresenham accrual, the
filter rows, the near-empty wipe, the split, the deposit with its rails, the
debit); a duct past the proved ledger range is refused by name with nothing
written, one step inside it is not; in a batch of differing envs as alone,
padding rows untouched, every capacity refused by name, and the chained twins
(the vent scene, the door scene, ship_demo's bots match) equal to an
untouched reference every tick. Breaks if any of those differs from the law.

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


def test_cuda_vents_on_their_tables_equal_the_reference():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_vents_check, sys; sys.exit(cuda_unit_vents_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_VENTS_RESULT: PASS" in out, (
        f"the vents gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
