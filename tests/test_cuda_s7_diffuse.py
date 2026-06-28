"""CUDA-S7 gate (pytest) — the GPU diffuse_solve bit-identity proof.

diffuse_solve is the LAST + hardest solver of the CUDA arc: the once-per-tick
IMPLICIT atmosphere step — the Red-Black Gauss-Seidel pressure relaxation (residual
form, per-cell Dinv), the vacuum BFS + sponge boundary pass, and the wind gradient.

SKIPS cleanly without a CUDA build / device. When the GPU build is present, runs
the S7 check in an isolated anaconda-3.11 subprocess (cuda_harness):
  * PART 1 isolated GPU-vs-CPU over rich synthetic inputs — the Red-Black GS
    convergence (8 sweeps, residual-form increment + per-cell Dinv), the mu-gate
    SKIP path (mu<=MU_EPS -> the diffusion operator is the identity), the vacuum BFS
    layers (0/1/2/255, wall-blocked), the sponge tiers (mul_q16 atmosphere sink +
    scale_mag wave_v shrink, both signs), and the +/- wind gradients (shr_round0) —
    asserting byte-for-byte equality on ALL SIX fields (atmosphere, wave_p, wave_v,
    wave_source, wind_x, wind_y) at tol 0;
  * a DRIFT-FREE check: a uniform atmosphere stays exactly uniform after the GS
    (the residual is 0 at the fixed point -> the round-to-nearest increment is 0; a
    toward-(-inf) truncating increment would shave -1 LSB/sweep = a DC mass sink);
  * PART 2 full-engine atmos-backend-switch integration over 30 ticks on a
    gradient+breach scenario (the GS, sponge, AND wind all engage), plus the
    default-scenario golden re-confirm.
A non-zero exit or a missing PASS marker fails the test.

This is the atmosphere solver's cross-GPU determinism gate — GPU-int == CPU-int.
The GS increment uses round_nearest_q_dev (sign-symmetric); the Red-Black colour
schedule is two separate launches per sweep (order-free by colour); the BFS is
double-buffered. With S5 (wave) + this, the whole atmosphere/wave system is GPU.
"""
from __future__ import annotations

import pytest

import cuda_harness


pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_s7_diffuse_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_s7_check, sys; sys.exit(cuda_s7_check.main())", timeout=420,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "S7_RESULT: PASS" in out, (
        f"CUDA-S7 diffuse_solve did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
