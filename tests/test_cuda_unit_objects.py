"""#29 P7b -- the grenades and charges gate (pytest wrapper of
tests/cuda_unit_objects_check.py).

Property: on the device object tables, the grenades in flight ARE the host's
``Simulation.projectiles`` (unpack(pack) attribute for attribute, a round
trip bit for bit); slot 1b's throws and accepted plants are appended to the
grenade and charge tables in the reference's order; slot 2's flight is
``Projectile.update_position`` integer for integer and each fuse-out is ONE
PAYLOAD record equal to the host's ``execute_payload`` call, in list order;
``charges.tick``'s fires are PAYLOAD records in list order and its prune, like
slot 10's grenade prune, is the stable compaction; every slot equal in a
batch of differing envs as alone, padding rows untouched, every capacity
refused by name, the WEGO forms refused at the device door, and the chained
twin (the tables landed back, the records replayed on the host) equal to the
untouched reference every tick. Breaks if a grenade column's encode /
decode, the flight's distance, landing test or rounding, the fuse test, a
PAYLOAD field (tile, payload row, owner, approach, kind), the record order,
an append's fields or order, a compaction, the batch indexing, a capacity
check or a device-door refusal differs from the law.

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


def test_cuda_grenades_and_charges_on_their_tables_equal_the_reference():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_objects_check, sys; sys.exit(cuda_unit_objects_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_OBJECTS_RESULT: PASS" in out, (
        f"the grenades and charges gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
