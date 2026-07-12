"""EOS P6.9b gate (pytest) — the GPU combustion solver bit-identity proof.

Gated by ``cuda_available(kernel="combustion")`` — the P6.0 pending-set contract:
this key is removed the moment the P6.9b port re-proves bit-identical, which (as
the LAST key) empties EOS_P6_PENDING_KERNELS and closes the P6 arc. SKIPS cleanly
without a CUDA build / device. When present, runs cuda_combustion_check in an
isolated subprocess (cuda_harness): the isolated edge-config + fuzz A/B, a hard
120-tick fire trajectory, and the golden re-check — all byte-for-byte GPU==CPU on
the three mutated gas planes + temperature + wall_hp + both rail counters.

This is the combustion solver's cross-GPU determinism gate — GPU-int == CPU-int —
and the capstone of the EOS GPU migration (docs/eos_p6_9_combustion_design.md §7).
"""
from __future__ import annotations

import pytest

import cuda_harness


pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(kernel="combustion"),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present, or the "
           "combustion kernel is still pinned in EOS_P6_PENDING_KERNELS",
)


def test_p69_combustion_bit_identity():
    proc = cuda_harness.run_cuda_script(
        "import cuda_combustion_check, sys; sys.exit(cuda_combustion_check.main())",
        timeout=420,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "P69_RESULT: PASS" in out, (
        f"CUDA-P6.9b combustion did not pass.\nreturncode={proc.returncode}\n{out}"
    )
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
