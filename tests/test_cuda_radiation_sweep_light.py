"""P6a gate (pytest) — the LIGHT CHANNELS' CUDA twin, CPU/GPU tol 0, and heat
untouched by light on the GPU (ray-engine-v2; docs/ray_engine_v2_p6a_light_
channels_brief_2026-09-25.md section 3, items 2 and 3).

PROPERTY: with the light group riding the sweep, the GPU twin (the per-call path
PhysicsEngine.step_tail dispatches, and its (N, h, w) launch core) computes the
SAME integers as RadiationSweep::run on every light plane (light_q,
light_flux_q, light_glow), the twelve light books and the light-stream
telemetry -- and the heat planes, the Fleck plane and the heat telemetry it
computes WITH light on equal, integer for integer, the ones it computes with
light off (its wavefronts become the step anti-diagonals for every ordinate;
the gather form must make that invisible to heat). Over a scene matrix (tints,
glass, bodies, smoke, both transports each, S16/S12, the live fine table and the
resolving one), degenerate to full-size grids, the ingress rejections, an N = 3
batch with one illegal env, and the live conductor with light requested.

BREAKS IF: the .cu transcribes a light term differently, an anti-diagonal launch
stops being topological for shear, a light book stops being an order-free sum,
the union packing of gas planes feeds either channel a wrong coefficient, or an
env's light planes leak into another's. Validated by breaking the .cu once
(see tests/cuda_radiation_sweep_light_check.py).

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


def test_light_channels_cpu_cuda_bit_identity_and_heat_untouched():
    proc = cuda_harness.run_cuda_script(
        "import cuda_radiation_sweep_light_check as c, sys; sys.exit(c.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "LS_RESULT: PASS" in out, (
        f"the light channels' CPU/CUDA parity did not pass.\n"
        f"returncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
