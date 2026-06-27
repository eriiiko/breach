"""CUDA-S0 hello-world bit-identity check (runs inside the GPU subprocess).

Proves, on the real device, that the integer toolkit compiled for the GPU
produces results BYTE-IDENTICAL to the CPU reference — the whole point of S0.
Imports the CUDA build (must be on sys.path via cuda_harness bootstrap), runs
``cuda_map_mul_q16`` over a wide, sign-mixed, edge-heavy input set for several
factors, and compares to an exact NumPy mirror of ``fixedpoint::mul_q16``.

Run directly (after add_dll_directory + sys.path), or via
``cuda_harness.run_cuda_script("import cuda_s0_check; cuda_s0_check.main()")``.
Prints ``S0_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

FP_ONE = 65536


def _cpu_mul_q16(a: np.ndarray, b: np.int32) -> np.ndarray:
    """Exact mirror of fixedpoint::mul_q16: (int64(a)*int64(b)) >> 16, truncating
    toward -inf (arithmetic shift), narrowed to int32."""
    return np.int32((a.astype(np.int64) * np.int64(b)) >> 16)


def _inputs() -> np.ndarray:
    rng = np.random.default_rng(20260627)
    return np.concatenate([
        rng.integers(-(1 << 24), 1 << 24, size=200_000, dtype=np.int64),
        rng.integers(-(1 << 30), 1 << 30, size=50_000, dtype=np.int64),
        # edge / boundary values
        np.array([0, 1, -1, FP_ONE, -FP_ONE, 32767 * FP_ONE, -32768 * FP_ONE,
                  (1 << 30), -(1 << 30), (1 << 31) - 1, -(1 << 31)], dtype=np.int64),
    ]).astype(np.int32)


def main() -> int:
    import breach_physics as bp
    if not getattr(bp, "HAS_CUDA", False):
        print("S0_RESULT: FAIL (module built without CUDA)")
        return 1
    if not bp.cuda_available():
        print("S0_RESULT: FAIL (no usable CUDA device)")
        return 1
    print("device:", bp.cuda_device_info())

    vals = _inputs()
    factors = [0.0, 1.0, -1.0, 0.37, 2.5, -0.8, 1.0 / 3.0, 1000.0, -1234.5]
    worst = 0
    for fr in factors:
        factor = np.int32(int(round(fr * FP_ONE)))
        gpu = np.asarray(bp.cuda_map_mul_q16(vals.tolist(), int(factor)), dtype=np.int32)
        cpu = _cpu_mul_q16(vals, factor)
        mism = int(np.count_nonzero(gpu != cpu))
        worst = max(worst, mism)
        tag = "ok" if mism == 0 else "DESYNC"
        print(f"  factor={fr:>10}  n={vals.size}  mismatches={mism}  {tag}")

    # Empty-input edge case must not crash.
    assert bp.cuda_map_mul_q16([], FP_ONE) == []

    if worst == 0:
        print(f"S0_RESULT: PASS (n={vals.size} x {len(factors)} factors, 0 mismatches)")
        return 0
    print(f"S0_RESULT: FAIL (worst mismatch count {worst})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
