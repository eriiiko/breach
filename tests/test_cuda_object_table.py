"""#29 P7.0b -- the object tables' device gate (pytest wrapper of
tests/cuda_object_table_check.py).

Property: the device object table (the planted charges, the pilot) holds
exactly the host list -- a round trip bit for bit; an append keeps its call
order after what the table holds; the compaction is stable (dead rows in the
middle, at the ends, all, none dropped, the rest in order, rows past the
count never read); past the capacity nothing is written, the overflow is
counted and refused naming the table and the env; on live scenes that plant
and detonate charges (the verb scene, a full table, a charge scene with dead
rows in every position) every tick unpack(pack(list)) is the list attribute
for attribute, the device append equals the host list at charges.tick's
entry, the device compaction equals the host's prune, and env i of a batch
of differing envs equals env i alone. Breaks if a column's encode / decode
pair, the append order, the compaction's stability or its live test, the
capacity check, the overflow count or the per-env layout differs, or a
compiled module with a stale layout is accepted.

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


def test_cuda_object_table_identity_append_compaction_overflow():
    proc = cuda_harness.run_cuda_script(
        "import cuda_object_table_check, sys; sys.exit(cuda_object_table_check.main())",
        timeout=1800,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "OBJECT_TABLE_RESULT: PASS" in out, (
        f"the object table gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
