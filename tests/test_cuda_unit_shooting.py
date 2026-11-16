"""#29 P7a.2 -- the shooting gate (pytest wrapper of
tests/cuda_unit_shooting_check.py).

Property: slot 4's shooting (``combat.process_shooting`` on the continuous
path), slot 2''s rounds (``Simulation._march_rounds``) and THE MARCH they share
(``combat.BulletInFlight.advance``) on the device equal the reference integer
for integer: the shooters' rows (the magazine, the cadence, a spray's burst),
the effect records record for record in the reference's order (the recoil
PUSH, the march's CHEW / HIT / PUSH / PAYLOAD), the round table (the rounds
still flying, compacted) and the noise list (the shots' noises in emission
order) -- on synthetic rounds over random maps of every material (every stop,
both outcomes of each draw), on the shooting scenes (kills, two rounds on one
target, a wall chewed to zero, rounds landing ticks later, a shooter dead or
turned while its rounds fly, payload stops, sprays, every gate refusal, the
census of the law's branches), in batches of N = 2 / 8 different envs (env i
equals env i alone), with padding rows never touched, each list's capacity
counted and refused, the beam / WEGO orders / physical cover refused by name,
slots 2' + landing and 4 + 4a + 4c chained in place of the host's keeping the
scene on the reference's path every tick, ship_demo's bots shadowed.
Breaks if a step, the stop, the list order, the shooter's exclusion, a draw,
a record's field, the recoil, the noise order, the table's append or
compaction, the batch rebase or a table row differs from the reference.

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


def test_cuda_unit_shooting_march_rounds_and_chained():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_shooting_check, sys; sys.exit(cuda_unit_shooting_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_SHOOT_RESULT: PASS" in out, (
        f"the shooting gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
