"""#29 P6b.3 -- the signals gate (pytest wrapper of tests/cuda_unit_signals_check.py).

Property: the device signal tables (the bus, the sensors and their discs, the
logic nodes and their inputs, the pumps and their inputs, the door drives)
ARE the sim's (unpack(records) row for row, the digest and entity rows the
serializer's); slot 9e's (a) emit + sensors, (b) logic, (c) door inputs, (d)
pumps and (e) swap on the device are Simulation._signal_emit /
sample_sensors / sweep_logic_nodes / _resolve_door_inputs / sweep_pumps /
SignalBus.swap_node_signals on those tables integer for integer (every sensor
channel over its whole range with the trace floor, motion plates through
walls and grilles, every predicate code at its threshold, the filter's floor
and its proved range refused by name one past it, every airlock transition,
the pumps' seam calls and latch), shadowed every tick on the signal scene
and airlock_demo, in a batch with an env without a bus as alone, padding
untouched, every capacity refused by name, and chained twins equal to an
untouched reference with the host laws run 0 times. Breaks if any of those
differs from the law.

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


def test_cuda_signals_on_their_tables_equal_the_reference():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_signals_check, sys; sys.exit(cuda_unit_signals_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_SIGNALS_RESULT: PASS" in out, (
        f"the signals gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
