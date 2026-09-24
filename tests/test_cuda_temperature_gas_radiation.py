"""P5c gate (pytest) -- the temperature twin's GAS radiation branch, CPU vs CUDA
at tol 0 (ray-engine-v2 issue #12, design v3 §2.8 / §6.3).

PROPERTY: the fold's gas branch -- an accountable gas cell's rad_net through the
staged chain, the clamp in its energy form, the railed seam deposit, the
withheld energy in TEMPERATURE_ENERGY_SLOTS slot 12 -- computes the SAME
integers on both temperature backends: on random isolated scenes (every kind of
cell the fold distinguishes, clamp on and off, both tables) and on the live
conductor with the temperature backend flipping every tick (a sealed room of
hot smoke; a smoke shield under a held 1263-game emitter).

BREAKS IF: cuda_temperature.cu's temp_convert_unified omits or reorders the gas
branch, converts through a device reciprocal the CPU does not use, clamps in a
different form, books slot 12 elsewhere, or physics_engine.cpp stops folding
slot 12 into e_rad_clamp_drop_sum.

SKIPS cleanly without a CUDA build / device. See
tests/cuda_temperature_gas_radiation_check.py for the two parts and their
non-vacuity assertions.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_temperature_twin_gas_radiation_cpu_cuda_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_temperature_gas_radiation_check as c, sys; sys.exit(c.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "TGR_RESULT: PASS" in out, (
        f"the temperature twin's gas branch diverged from the CPU.\n"
        f"returncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
