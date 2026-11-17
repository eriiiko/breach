"""#29 P6b.4 -- the hazards gate (pytest wrapper of tests/cuda_unit_hazards_check.py).

Property: the device timed-charge and heat-source tables ARE the sim's; slot
9e(d')'s timed-charge sweep (every trigger kind, the period, the heat latch
and fuse, a tile an earlier charge destroyed in the same sweep) and 9e(d'')'s
heat-source sweep (the growing blast log read into the sources, the chain of
death payloads within one tick, the deposits, the cap) on the device are
sweep_timed_charges / sweep_heat_sources integer for integer, shadowed every
tick (tables, blast log, PAYLOAD records, planes, books, edits, claims, ring)
on the hazard scene and fire_smoke_studio with a census of both laws'
branches; a chain at its proved bound equals the host and one past it is
refused by name with nothing written; N = 2 / 8 batches equal each env alone;
padding untouched; every capacity refused by name; slot 9e WHOLE chained on
the device equals an untouched reference every tick with the entity and
signal sections exported from the device tables equal to the serializer's,
the host laws run 0 times in the twin. Breaks if any of those differs from
the law.

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


def test_cuda_hazards_on_their_tables_equal_the_reference():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_hazards_check, sys; sys.exit(cuda_unit_hazards_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_HAZARDS_RESULT: PASS" in out, (
        f"the hazards gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
