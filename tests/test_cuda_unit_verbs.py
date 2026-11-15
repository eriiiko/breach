"""#29 P6.4a -- the verbs gate (pytest wrapper of tests/cuda_unit_verbs_check.py).

Property: slot 1b's hand verbs on the device unit store
(``Simulation._consume_direct_intents``, the continuous path) equal the
Python law row for row, and what the slot does that is not a row write -- a
grenade released, a charge offered to the charge table (placed or refused at
its capacity, in id order), a door flipped -- equals what the host did, record
for record in the host's order, at tol 0 on every tick of live scenes that
make every branch act (ship_demo's bots match with HANDLING / PUSH hooks; a
scripted verb scene whose census of the host's own decisions counts each verb
acting and every refusal of every gate, the drop branch on a dead and a
zombie row, every grenade kind, every clamp, two planters at a full charge
table on one tick, the shove's hit / miss / two shovers on one target / a
zombie target / a dead unit skipped, the use's door in reach / none / a dead
door / a hull door; the class wind-up and plant at 0); the device gates equal
verb_gates.py on synthetic rows over their range; the new device tables equal
the reference objects; in batches of N = 2 and 8 whose envs differ in layout,
seed, unit count, tick and live charges (env i == env i alone); padding rows
holding live latches untouched and silent; a list too small refused by name;
slot 1b alone and 1a + 1b + 1b' + 1c chained in place of the host's keep a
twin match on the reference's path every tick; the done kernel equals
MatchRules on every tick of matches ending by elimination, at the limit in a
draw, by a fallen team, and in a batch ending on different ticks. Breaks if
any gate term, hook, clamp, rounding, the capacity's id order, the door scan,
the reload's settle, the shove's target / lever / angular term, the drop
branch, the request order or the done kernel's team count, limit or outcome
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


def test_cuda_unit_verbs_shadow_lockstep_requests_chained_and_done():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_verbs_check, sys; sys.exit(cuda_unit_verbs_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_VERBS_RESULT: PASS" in out, (
        f"the verbs gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
