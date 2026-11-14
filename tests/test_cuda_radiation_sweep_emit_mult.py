"""G3 F4b gate (pytest) -- THE FIRE'S LIGHT FLICKER on the CUDA twin, CPU/GPU
tol 0, heat untouched (docs/g3_fire_design_2026-10-23.md section 6.3, RULED D).

PROPERTY: with emission-multiplier rows on the light group, the GPU twin -- the
per-call path PhysicsEngine.step_tail dispatches, its (N, h, w) launch core
with a per-env multiplier plane, and the live conductor with light requested
and the runner's flicker rows set -- computes the SAME integers as
RadiationSweep::run on every light plane, the twelve light books and the light
telemetry; heat stays bit-identical to the flicker-off sweep on both backends;
the door refuses an illegal row on both without touching a plane; and in the
live conductor the flicker moves light_q and no other GameMap array.

BREAKS IF: the .cu scales the emission at a different point of the chain than
run(), scales a cone, books the unscaled emission, indexes an env's multiplier
plane wrong, or a second scatter of the rows appears in the .cu.

Runs in an isolated subprocess (tests/cuda_harness.py -- the CUDA .pyd is never
imported into pytest); SKIPS cleanly without a CUDA build / device.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_flicker_cpu_cuda_bit_identity_and_heat_untouched():
    proc = cuda_harness.run_cuda_script(
        "import cuda_radiation_sweep_emit_mult_check as c, sys; sys.exit(c.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "LM_RESULT: PASS" in out, (
        f"the flicker CPU/CUDA parity did not pass.\n"
        f"returncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
