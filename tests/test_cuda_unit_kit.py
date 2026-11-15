"""#29 P7.0a -- the kit and slot 9d gate (pytest wrapper of
tests/cuda_unit_kit_check.py).

Property: the device draws (``cpp/src/cuda_unit_draws.cuh``) are
``MatchDraws.at`` / ``.next``, ``cone_offset_q16`` and ``roll_q16`` word for
word, per env of a batch whose keys and ticks differ, over every salt and the
range ends of id, index, cone and probability, each ``.next`` site counting on
its own; the device ``has_los`` (``cpp/src/cuda_unit_sight.cuh``) is
``GameMap.has_los`` in both forms on random pairs over four real maps with
grilles, in one batch of different maps, both answers often; the falloff
twins (``fixed_point.h``) are ``unit_fixed.falloff_floor`` /
``falloff_round`` over the callers' ranges and the range ends, refusing
outside the proved range; and slot 9d (``cpp/src/cuda_unit_ignition.cu``)
equals ``combat.apply_temperature_ignition`` on ``fire`` and
``ignition_armed`` with no other write (the write-set gate) on a scene where
ignition happens on many ticks with every branch of the law taken, on
ship_demo where it never happens, in batches of N = 2 / 8 different envs (env
i == env i alone), and chained after 9c / 9c2 / 9c3 in a twin match that
stays on the reference's path. Breaks if a counter word, the id word, the
``.next`` count's locality, the cone's or the roll's map, the walk's step
rule or its start / end test, the screen in either form, the falloff's
rounding or division, the re-arm, the fuel gate, the open-neighbour mask, the
fraction's floor or division, the seed or the disarm differs from the
reference, or the slot writes a plane it does not declare.

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


def test_cuda_unit_kit_draws_sight_falloffs_and_slot_9d():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_kit_check, sys; sys.exit(cuda_unit_kit_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_KIT_RESULT: PASS" in out, (
        f"the kit gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
