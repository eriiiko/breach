"""#29 P7c.1 -- the payloads gate, the disc path (pytest wrapper of
tests/cuda_unit_payloads_check.py).

Property: a PAYLOAD record of a [payloads.*] row with strength = 0, no bulk
gas and no explosive is ``payloads.execute_payload`` integer for integer on
the device (cpp/src/cuda_unit_payload.cuh, an effect function of the per-env
serial applier) -- the disc walked serially with its wall damage and
destroy_wall at once, the wave stencil on the live planes, the atmosphere and
smoke-clear DISC edits, the immediate blast packets, the flash over sight and
facing (never shortened), emit_gas / ignite_ring / deposit_heat through THE
DEVICE EDIT DOOR's DISC form with their claims, the noise and the blast log --
in the DETONATE stage (slot 2's grenades, slot 9e(d')'s charges) and the
RESOLVE stage (slot 4c, slot 2''s landing, where a blast-killed unit's
pending raise is skipped); equal on synthetic lists, on the payload scene
(every shipped disc-path row through every call site), in a batch of
differing envs as alone, with padding untouched, every list's capacity
refused by name, every refused row refused by name writing nothing, and in
chained twins whose host execute_payload never runs. Breaks if a payload
row, the disc's order, material filter, wall damage or live stencil read, a
DISC edit's field / radius / amount / mode / falloff / key / source, the
blast damage's falloff / threshold / row order / immediate form, the flash's
sight / facing / duration, a claim, the noise, the blast log, the event
order, the corpse skip, the call order within a tick or the batch indexing
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


def test_cuda_payloads_on_the_disc_path_equal_the_reference():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_payloads_check, sys; sys.exit(cuda_unit_payloads_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_PAYLOADS_RESULT: PASS" in out, (
        f"the payloads gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
