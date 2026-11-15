"""#29 P6.2m -- the slot-3 (movement) gate (pytest wrapper of
tests/cuda_unit_movement_check.py).

Property: the device slot-3 kernel is the continuous path's
``Simulation._update_player_movement`` row for row at tol 0 -- the MOVE_DIR
walk (the speed table's costs reduced over the footprint's mobility with the
WALK / SPRINT hooks and the effort scale, the direction rule and the throttle
rounded once, the walk ban, the statuses that take the walk away), the walk
and the pushed velocity as one displacement through the slide rule (the axis
into a wall zeroes pv there, the substeps), a held-no-aim walker's facing
target, pv alone for every other row (a zombie included) -- on every tick of
live scenes where each branch acts and is counted (ship_demo's bots match, a
movement box with walls, furniture, pushes, statuses, an idle horde, a dead
unit; the doorway jet), in batches of N = 2 and 8 whose envs differ in map
and tables (env i == env i alone, padding rows untouched), its coverage
counter equal to the rows the host changed; the walk's library functions
equal their Python on synthetic rows over the proved range; hooking leaves a
match equal to a harness-free twin; and slot 3 on the device keeps a twin
match on the reference's path (CHAINED). Breaks if the kernel's arithmetic
differs from the law, a hook is ignored, a padding row or another env (its
rows or its material plane) is touched, a table value is not the
reference's, or the chained twin diverges.

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


def test_cuda_unit_movement_shadow_lockstep_and_chained():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_movement_check, sys; sys.exit(cuda_unit_movement_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_MOVEMENT_RESULT: PASS" in out, (
        f"the slot-3 movement gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
