"""P6d gate G2 (pytest) -- the light transport setting on the LIVE conductor,
CPU/GPU tol 0 (ray-engine-v2; docs/ray_engine_v2_p6d_light_transport_setting_
brief_2026-09-29.md section 3).

PROPERTY: with PhysicsEngine.light_transport set to RadiationSweep.SHEAR, the
GPU twin (the per-call path PhysicsEngine.step_tail dispatches, the resident
tick, and relight()) computes the SAME integers as the CPU sweep on a real
Simulation -- the playground level with its own emergency lamps, a burning
wood tile and a smoke puff, light requested through the Simulation facade --
every synced GameMap array (the three light planes included), the engine's
light books, per tick, tol 0.

BREAKS IF: run_sweep_ stops reading PhysicsEngine.light_transport on either
backend, or the CUDA twin's light walk is not in fact topological for
TRANSPORT_SHEAR. Validated by breaking cpp/src/physics_engine.cpp's
`lc.transport = this->light_transport` back to a literal (see tests/
cuda_p6d_light_transport_check.py).

Runs in an isolated subprocess (tests/cuda_harness.py -- the CUDA .pyd is
never imported into pytest); SKIPS cleanly without a CUDA build / device.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_light_transport_shear_cpu_cuda_bit_identity_on_the_live_conductor():
    proc = cuda_harness.run_cuda_script(
        "import cuda_p6d_light_transport_check as c, sys; sys.exit(c.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "P6D_RESULT: PASS" in out, (
        f"the light transport setting's CPU/CUDA parity (transport=SHEAR) "
        f"did not pass on the live conductor.\n"
        f"returncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
