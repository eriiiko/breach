"""EOS P6.8 gate (pytest) — the re-derived GPU fire-step bit-identity proof.

SKIPS cleanly without a CUDA build / device (per-kernel key: "fire" on the P6.0
per-kernel pending-set contract). When the GPU build is present, runs the P6.8
check in an isolated subprocess (cuda_harness):

  * PART 1 — isolated CPU FireSimulation.step vs GPU cuda_fire_step over random
    fuzz + deterministic forcers that exercise EVERY branch of the re-derived
    kernel: the n_o2 O2 gate (no-O2 starve / full-O2 grow / empty neighbour
    mean on all-solid + all-vacuum), BOTH plume->T self-limiter paths (the
    sat-clamp-to-zero above the ceiling AND the headroom HARD-CAP pinned exactly
    at T_FLAME_MAX), below-ambient sat-clamp, wind fan/strip, wall burn-through
    (SET-equal destroyed, no drops/dupes), snap-extinguish, degenerate 1xN/Nx1,
    P_degenerate smoothstep, non-identity temp_scale, overlapping-neighbour
    smoke atomicAdd, and the host max early-exit (fields untouched).
  * PART 2 — a 130-tick O2-rich-room ignition trajectory (plume heating to the
    T_FLAME_MAX ceiling, O2-depletion self-starving, wall burn-through) with the
    CPU-backend and GPU-backend states stepped in LOCKSTEP, per-tick byte-identity
    on fire/temperature/smoke/wall_hp + SET-equal destroyed.
  * PART 3 — the committed default-scenario golden on the CUDA build's CPU path.

A non-zero exit or a missing PASS marker fails the test.

The port is a pure per-cell gather chain (every pass writes only its own cell,
with read-only neighbour reads / order-free atomic scatter / order-free counter —
NO combustion-style Gauss-Seidel coupling), so bit-identity needs no
restructuring. This gate is the proof.
"""
from __future__ import annotations

import pytest

import cuda_harness


pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("fire"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_p68_fire_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_fire_check, sys; sys.exit(cuda_fire_check.main())",
        timeout=600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "P68_RESULT: PASS" in out, (
        f"EOS P6.8 fire did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
