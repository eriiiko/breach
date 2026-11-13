"""#29 P4c — the device edit queue gate (pytest wrapper of
tests/cuda_edit_queue_check.py).

Property: slot 6b's FieldEdit flush applied on the device (field_edit.
pack_for_device -> cpp/src/cuda_edit_queue.cu) leaves every GameMap array and
every integer book where the CPU flush leaves them, tol 0, every tick, in one
scene per FieldEdit policy row (trace gases, atmosphere, wave_source, fire,
heat, gas_energy, TILES with weight_den and the blast's scale, an ambient
ring), each proven to land (the edited world differs from a no-edit world on
the touched tiles); a water_depth tick falls back to the host flush and says
so; a heat source's previous-tick 9e deposit applies before the tick's own
edit of the same (field, source, region) on both backends (the cross-tick
key); a conservative-slice refusal stays the host's; and the device's keyed
noise and rounding equal draws.noise_mult_q16 / div_round on every packed
record, while moving every key by one changes them. Breaks if a combine,
clamp, skip mask, rounding, noise counter, seam booking, mirror refresh or
the order differs.

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


def test_cuda_edit_queue_equals_the_cpu_flush_on_every_policy_row():
    proc = cuda_harness.run_cuda_script(
        "import cuda_edit_queue_check, sys; sys.exit(cuda_edit_queue_check.main())",
        timeout=1800,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "EDIT_QUEUE_RESULT: PASS" in out, (
        f"the device edit queue gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
