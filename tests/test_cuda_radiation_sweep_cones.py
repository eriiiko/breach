"""P6b gate (pytest) -- the CONE EMITTERS and THE SKY on the CUDA twin, CPU/GPU
tol 0, heat untouched (ray-engine-v2; docs/ray_engine_v2_p6b_lit_world_brief_
2026-09-25.md section 3.2).

PROPERTY: with cone emitters and a sky on the light group, the GPU twin -- the
per-call path PhysicsEngine.step_tail dispatches, its (N, h, w) launch core
with per-env skies and cone slots, and the live conductor with light requested
and the runner's cones and sky set -- computes the SAME integers as
RadiationSweep::run on every light plane, the twelve light books and the light
telemetry; heat stays bit-identical to the light-off sweep on both backends;
the door refuses an illegal cone or sky on both without touching a plane; and
PhysicsRunner.relight() gives the same planes on both backends while moving no
other GameMap array.

BREAKS IF: the .cu reads the sky for one upwind share only, injects a cone after
the cell's absorption, books the injection outside the emission slot, indexes a
slot's ordinate or an env's sky wrong, or a second projection appears in the
.cu. Validated by breaking the .cu once (the report names the break).

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


def test_cones_and_sky_cpu_cuda_bit_identity_and_heat_untouched():
    proc = cuda_harness.run_cuda_script(
        "import cuda_radiation_sweep_cones_check as c, sys; sys.exit(c.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "LC_RESULT: PASS" in out, (
        f"the cones/sky CPU/CUDA parity did not pass.\n"
        f"returncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
