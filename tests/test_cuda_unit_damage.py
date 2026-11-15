"""#29 P6.3 -- the damage gate (pytest wrapper of tests/cuda_unit_damage_check.py).

Property: the device damage pipeline (``damage.apply_packet``: mitigation, the
exposure scale, the hp, the kill exactly once, every death clear, the exposure
raise) equals the Python law on synthetic rows, packets and tables over their
range; the device tables of P6.3 equal the reference objects; and the three
slots on it -- 2b (statuses), 9c (heat), 9c3 (teargas, the poison dose, the
poison law) -- equal their Python laws row for row AND event for event, in the
reference's emission order, at tol 0 on every tick of live scenes that make
every branch act (deaths with every clear to make, credited and uncredited
heat and poison, radiation the larger exposure, the poison flush to exactly
0, the dose clipped at the body's store, a zombie in gas, damage over time of
every kind, a stack at its cap), with non-neutral HP / TOX hooks, in batches
of N = 2 and 8 with padding rows (env i == env i alone, rows and rings); a
ring too small for a tick's events is counted, never overrun, and refused by
name; hooking a match leaves it equal to an unhooked one; and 2b alone and
9c + 9c2 + 9c3 chained in place of the host's slots keep a twin match on the
reference's path every tick. Breaks if any of those laws' arithmetic, a death
clear, the ring's order, an event field, the credit's channel, the heat
plane's currency, the poison flush, a TOX hook or the zombie's immunity
differs from the reference, or a padding row or another env is touched.

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


def test_cuda_unit_damage_shadow_lockstep_events_and_chained():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_damage_check, sys; sys.exit(cuda_unit_damage_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_DAMAGE_RESULT: PASS" in out, (
        f"the damage gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
