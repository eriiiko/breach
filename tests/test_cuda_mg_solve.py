"""EOS P6.3 gate (pytest) — the GPU multigrid pressure solve bit-identity proof.

SKIPS cleanly without a CUDA build / device (per-kernel key: "mg_solve" on the
P6.0 per-kernel pending-set contract). When the GPU build is present, runs the
P6.3 check in an isolated subprocess (cuda_harness): the isolated
GPU-vs-CPU-reference comparison over synthetic/edge/overflow-stress configs
(full 9-level 160x160 pyramid with the fused coarse-tail kernel, flat RB-GS
path, 1x1/1xN/Nx1 degenerates, solid-ring 1-cell rooms, all-vacuum/all-solid,
near-N_FLOOR fields, blast-scale |P| against floored-N̂ faces) AND the
breach-to-vacuum + blast trajectory gate — 120 real engine ticks with the
per-tick digest_helmholtz chain asserted CPU-solver == CPU-ref == GPU,
byte-identical solved P (== the engine's own P_new, the velocity kick's
input) throughout — AND the committed default-scenario golden on the CUDA
build's CPU path. A non-zero exit or a missing PASS marker fails the test.

The smoother is two-color order-free and the MG transfers are single-writer
gathers (docs/eos_p6_gpu_alignment_review.md §1.1/§1.2), and the fused coarse
tail preserves every CPU pass boundary behind a block-wide barrier (§2.2), so
bit-identity needs no restructuring; this gate is the proof.
"""
from __future__ import annotations

import pytest

import cuda_harness


pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available("mg_solve"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_p63_mg_solve_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_mg_solve_check, sys; sys.exit(cuda_mg_solve_check.main())",
        timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "P63_RESULT: PASS" in out, (
        f"EOS P6.3 MG solve did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
