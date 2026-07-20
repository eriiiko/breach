"""CUDA-S8a residency spike: prove a breach kernel mutates CuPy-owned memory.

The foundation de-risk for the whole S8a residency patch
(docs/cuda_s8a_residency_spec_2026-07-19.md §"Proven foundation"). It allocates
a CuPy int32 array, hands the breach `.pyd` its raw device address
(`int(arr.data.ptr)`) via `cuda_spike_add1`, and asserts the in-place +1 landed
ON the CuPy array. That is the exact contract every STEP B–F launch core relies
on: CuPy and the breach `.pyd` share the one CUDA primary context, so a device
pointer allocated by CuPy is directly launchable from breach kernels — no
cudaMalloc, no H2D/D2H.

Runs the check in the CUDA-owning interpreter via tests/cuda_harness (which
prepends the DLL dir + sys.path so `import breach_physics` resolves to the GPU
build). REQUIRES cupy in that interpreter's env (the conda `data` env on the
Lenovo/Ada). This file is written for the Berlin run — do NOT expect it to pass
until the CUDA build exists AND cupy is importable there.

Run:  C:/Users/steen/miniconda3/envs/data/python.exe tests/_spike_s8.py
      (or set BREACH_CUDA_PYTHON to the interpreter that owns the .pyd)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuda_harness import run_cuda_script, cuda_available  # noqa: E402


# This body runs INSIDE the CUDA-owning subprocess (cuda_harness._bootstrap
# prepends the DLL dir + sys.path). It prints "RESULT: ..." the parent asserts on.
_BODY = r"""
import numpy as np
import breach_physics as bp

print("HAS_CUDA", bool(getattr(bp, "HAS_CUDA", False)))
assert getattr(bp, "HAS_CUDA", False), "imported the CPU build, not build_cuda"
assert bp.cuda_available(), "no usable CUDA device"
print("DEVICE", bp.cuda_device_info())

try:
    import cupy as cp
except Exception as e:
    print("RESULT: SKIP-NO-CUPY", repr(e))
    raise SystemExit(0)

# Allocate a CuPy-owned int32 array. Its .data.ptr is a raw device address in
# the ONE CUDA primary context both CuPy and the breach .pyd attach to.
n = 1024
host_before = np.arange(n, dtype=np.int32)
arr = cp.asarray(host_before)          # device copy, CuPy-owned
assert arr.dtype == cp.int32 and arr.flags["C_CONTIGUOUS"]

dev_ptr = int(arr.data.ptr)            # uintptr_t through pybind
print("DEV_PTR_NONZERO", dev_ptr != 0)

# The breach kernel mutates CuPy's memory IN PLACE — no malloc, no transfer.
bp.cuda_spike_add1(dev_ptr, n)
cp.cuda.Stream.null.synchronize()

# Read back FROM the CuPy array (same allocation) and confirm every element +1.
host_after = cp.asnumpy(arr)
expected = host_before + 1
ok = bool(np.array_equal(host_after, expected))
print("FIRST5_BEFORE", host_before[:5].tolist())
print("FIRST5_AFTER ", host_after[:5].tolist())
print("ALL_PLUS_ONE", ok)

# Partial-length safety: only the first `n` are touched; a spike over a prefix
# of a larger array must leave the tail untouched (pointer arithmetic sanity).
big = cp.zeros(64, dtype=np.int32)
bp.cuda_spike_add1(int(big.data.ptr), 16)
cp.cuda.Stream.null.synchronize()
big_h = cp.asnumpy(big)
prefix_ok = bool(np.all(big_h[:16] == 1))
tail_ok = bool(np.all(big_h[16:] == 0))
print("PREFIX_OK", prefix_ok, "TAIL_OK", tail_ok)

print("RESULT:", "PASS" if (ok and prefix_ok and tail_ok) else "FAIL")
"""


def main() -> int:
    if not cuda_available():
        print("SKIP: CUDA build / runtime not present. "
              "Build it with cpp/build_cuda_lenovo.bat.")
        return 2
    proc = run_cuda_script(_BODY, timeout=120.0)
    sys.stdout.write(proc.stdout)
    if proc.stderr.strip():
        sys.stderr.write("\n--- subprocess stderr ---\n")
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(f"\nFAIL: subprocess exited {proc.returncode}")
        return 1
    if "RESULT: SKIP-NO-CUPY" in proc.stdout:
        print("\nSKIP: cupy not importable in the CUDA interpreter env.")
        return 2
    if "RESULT: PASS" not in proc.stdout:
        print("\nFAIL: did not see RESULT: PASS")
        return 1
    print("\nOK: breach kernel mutated CuPy-owned device memory in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
