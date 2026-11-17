"""#29 P7a.1 -- the effect-list gate (pytest wrapper of
tests/cuda_unit_melee_check.py).

Property: the per-env serial applier's EFFECT LIST on the device (the record
widened to HIT / STATUS / PUSH / CHEW and the PAYLOAD record beside
DESTROY_WALL; a record list the producers append to; one walker with a STAGE)
equals the reference: slot 4a (``combat.process_melee_intents``, the melee
verb, its first device producer) gives the host's rows and, record for record
in the host's order, the host's ``SimultaneousEffects`` records; the RESOLVE
stage (slot 4c) fed the host's list gives the host's rows, ring, planes and
books, with the EXPOSURE LANDING (raises held, a corpse's raise skipped, the
dead zeroed at the walk's end) as a mode of the device ``apply_packet`` --
on a melee scene that takes every branch of the Python law (two units killing
each other, two blows on one target, a target dead mid-walk with later
effects aimed at it, a later blow reading a slot-killed unit's unzeroed
exposure, a tie, coincident centres, every latch refusal, STRIKE / PUSH
hooks), on synthetic lists (the landing's result different from applying at
once; a HIT / CHEW list whose two orders differ, both equal on the device; a
PAYLOAD recorded and refused by name; FACE refused), on the gates' synthetic
rows (``can_melee`` in both moments, the arc target, the adjacency), in
batches of N = 2 / 8 different envs (env i equals env i alone), with padding
rows never touched, the list's and a row's capacity counted and refused, and
4a + 4c chained in place of the host's keeping a melee match on the
reference's path every tick; ship_demo's bots shadowed 500 ticks.
Breaks if the landing lands at once, zeroes before the raises or raises a
corpse; the list's order, the gather's row order or its continuation; a
kind's function; the arc target's tie or the adjacency; a gate term; a melee
table row; or the batch rebase differs from the reference.

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


def test_cuda_unit_melee_effect_list_resolve_and_chained():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_melee_check, sys; sys.exit(cuda_unit_melee_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_MELEE_RESULT: PASS" in out, (
        f"the effect-list gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
