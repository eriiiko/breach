"""CUDA build loader + isolated-subprocess helper (CUDA-S0).

Two jobs:

1. **Locate + load the CUDA build.** The GPU `.pyd` is built into
   ``cpp/build_cuda/`` (Ninja, single-config) separately from the canonical CPU
   build in ``cpp/build/Release/`` — so the running game and the existing suite
   are untouched. Loading it needs the CUDA runtime DLL dir registered via
   ``os.add_dll_directory`` (since Python 3.8, PATH is NOT searched for an
   extension module's dependent DLLs).

2. **Run GPU checks in an ISOLATED subprocess.** A single Python process can
   import ``breach_physics`` only once, and the rest of the suite imports the CPU
   build. So a GPU test must run in its own interpreter (the anaconda 3.11 that
   matches the cp311 ``.pyd``) with sys.path pointed at ``cpp/build_cuda`` — never
   importing the CUDA module into the pytest process itself. ``run_cuda_script``
   does exactly that; the X-ARCH digest runner reuses the same pattern.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CUDA_BUILD_DIR = ROOT / "cpp" / "build_cuda"
# The interpreter that owns the .pyd — it MUST match the .pyd's cpXXX ABI. The
# default is the Work Desktop's anaconda 3.11 (cp311); override with the env var
# BREACH_CUDA_PYTHON on any machine whose CUDA build was produced by a different
# interpreter (e.g. the Lenovo/Ada, where the `data` miniconda env is cp312). The
# X-ARCH digest runner reuses this same interpreter.
# P6.2: when the anaconda default does not exist on this machine (the Lenovo has
# only miniconda), fall back to the Lenovo `data` env (cp312 — matches its
# build_cuda_lenovo.bat build) so the now-ACTIVE per-kernel gates run without a
# per-shell env var. BREACH_CUDA_PYTHON still overrides everything.
_CUDA_PYTHON_DEFAULT = r"C:\Users\steen\anaconda3\python.exe"
if not Path(_CUDA_PYTHON_DEFAULT).exists():
    _CUDA_PYTHON_DEFAULT = r"C:\Users\steen\miniconda3\envs\data\python.exe"
CUDA_PYTHON = Path(os.environ.get("BREACH_CUDA_PYTHON", _CUDA_PYTHON_DEFAULT))


def cuda_pyd() -> Path | None:
    """The built CUDA module, or None if the GPU build has not been produced."""
    hits = list(CUDA_BUILD_DIR.glob("breach_physics*.pyd"))
    return hits[0] if hits else None


def cuda_dll_dir() -> Path | None:
    """The CUDA runtime ``bin`` dir holding cudart64_*.dll, from CUDA_PATH or the
    pinned v12.4 install. None if not found."""
    cands = []
    cp = os.environ.get("CUDA_PATH")
    if cp:
        cands.append(Path(cp) / "bin")
    cands.append(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"))
    for c in cands:
        if c.is_dir() and list(c.glob("cudart64_*.dll")):
            return c
    return None


# EOS P6 GPU migration — COMPLETE (closed 2026-07-11, branch eos-p6-close).
# During the migration the CPU solvers ran ahead of the CUDA kernels, so the GPU
# gates were pinned to SKIP via a pending SET (EOS_P6_PENDING_KERNELS) with one
# key per P6 sub-patch surface (docs/eos_p6_gpu_alignment_review.md §2.1). The
# contract: each sub-patch removed EXACTLY its own key once its kernel was
# re-proved bit-identical (per-kernel A/B digest + cross-machine per-field
# digest). combustion — the LAST key — landed as P6.9b, emptying the set and
# closing the arc (design doc §7). Per that contract ("when the set is empty this
# machinery is deleted outright"), the pending set AND the _P6_KERNEL_KEYS
# typo-guard are now GONE: with every kernel ported, cuda_available() is a plain
# "is the GPU build present?" check for both the whole-suite cuda_s* gates and
# the per-kernel P6 gates.
def cuda_available(kernel: str | None = None) -> bool:
    """True iff both the CUDA build and its runtime DLLs are present on disk.
    (Whether a *device* is actually usable is checked inside the subprocess.)

    The EOS P6 GPU migration is COMPLETE — every kernel is ported and each has a
    passing bit-identity gate — so this is now a simple presence check used by
    ALL gates. The optional ``kernel`` arg is a vestige of the migration's
    pending-set pin (the per-kernel P6 gates call ``cuda_available(kernel="...")``);
    it is accepted-and-IGNORED so those call sites need no edit now that no kernel
    can be pending. The old per-key typo-guard (ValueError on an unknown key) is
    retired with the pending set: with nothing pinnable, an unknown key can no
    longer silently unpin anything, so there is nothing to guard."""
    return cuda_pyd() is not None and cuda_dll_dir() is not None


# A bootstrap snippet prepended to every in-subprocess GPU script: register the
# CUDA DLL dir, then put the CUDA build + src + repo root on sys.path FIRST so
# `import breach_physics` resolves to the GPU build.
def _bootstrap() -> str:
    dll = cuda_dll_dir()
    return (
        "import os, sys\n"
        f"os.add_dll_directory(r'{dll}')\n"
        f"sys.path.insert(0, r'{CUDA_BUILD_DIR}')\n"
        f"sys.path.insert(0, r'{ROOT / 'src'}')\n"
        f"sys.path.insert(0, r'{ROOT}')\n"
        f"sys.path.insert(0, r'{ROOT / 'tests'}')\n"
    )


def run_cuda_script(body: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh anaconda-3.11 interpreter with the CUDA build
    importable. Returns the CompletedProcess (caller asserts on returncode/stdout)."""
    script = _bootstrap() + body
    return subprocess.run(
        [str(CUDA_PYTHON), "-c", script],
        capture_output=True, text=True, timeout=timeout, cwd=str(ROOT),
    )
