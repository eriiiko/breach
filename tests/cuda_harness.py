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


# EOS refactor migration window (design doc §7 / D7, merged P1+P2 2026-07-10):
# the CPU solvers have moved ahead of the CUDA kernels (new gas species, unified
# temperature), so the GPU mirrors are STALE until the P6 port re-proves each
# kernel bit-identical. During this window the CUDA gates must SKIP — a stale
# kernel comparing against the new CPU path is a guaranteed, meaningless red.
#
# P6.0 (docs/eos_p6_gpu_alignment_review.md §2.1): the old single global bool
# `EOS_P6_PENDING = True` had no partial-unpinning mechanism, so it is now a
# pending SET with one string key per P6 sub-patch surface (the review's §4
# sub-patch table). THE CONTRACT for P6.1+: each sub-patch, once its kernel is
# re-proven bit-identical (per-kernel A/B digest + cross-machine per-field
# digest), removes EXACTLY its own key from this set — nothing else. When the
# set is empty (P6.9 landed), this machinery is deleted outright.
#
# key              unpinned by  surface
# ---------------  -----------  ------------------------------------------------
# bulk_flux        P6.1         donor-cell bulk flux (water K3–K8 pattern)
# sl_advection     P6.2         fused 3-field SL advection + cmask + zero-solid
# mg_solve         P6.3         MG pressure solve (smoother/transfers/fused tail)
# kick_compression P6.4         momentum kick + absorption + clamp; compression work
# eos_step         P6.5         EOS orchestration: full eos.step per-call dispatch
# conduction       P6.6         unified conduction + T Pass 1/Pass 3
# trace_smoke      P6.7         trace-smoke re-port at once-per-tick cadence
# fire             P6.8         fire re-derivation (plume→T shim + n_o2 signature)
# combustion       P6.9         combustion face-buffer split (gated on §3.1 CPU change)
#
# (The retired wave/atmosphere kernels have no key: P6.0 DELETED cuda_wave.cu /
# cuda_atmosphere.cu and their gates instead of unpinning them.)
# P6.2 (eos-p6-2-sl-advection): "sl_advection" REMOVED — the fused 3-field SL
# advection (cuda_sl_advection.cu) re-proved bit-identical via the P6.2 gate
# (tests/cuda_p62_check.py: isolated synthetic A/B + full blast+venting
# per-tick digest_advect trajectory vs the CPU solver).
# P6.1 (eos-p6-1-bulk-flux): "bulk_flux" REMOVED — cuda_bulk_transport.cu
# re-proven bit-identical (tests/cuda_bulk_flux_check.py — isolated all-branch
# A/B + closed-loop breach-venting/blast trajectory, per-plane byte-compare).
EOS_P6_PENDING_KERNELS = {
    "mg_solve",
    "kick_compression",
    "eos_step",
    "conduction",
    "trace_smoke",
    "fire",
    "combustion",
}

# The full P6 key universe (NEVER shrinks — used to reject typo'd kernel names,
# which would otherwise silently read as "already unpinned"). Spelled out as an
# EXPLICIT literal, not frozenset(EOS_P6_PENDING_KERNELS) — the derived form
# silently shrank with the pending set, so the FIRST key removal would have
# turned every unpinned kernel's own gate into a ValueError. (Both P6.1 and
# P6.2 hit and fixed this same latent P6.0 bug independently.)
_P6_KERNEL_KEYS = frozenset({
    "bulk_flux",
    "sl_advection",
    "mg_solve",
    "kick_compression",
    "eos_step",
    "conduction",
    "trace_smoke",
    "fire",
    "combustion",
})


def cuda_available(kernel: str | None = None) -> bool:
    """True iff both the CUDA build and its runtime DLLs are present on disk.
    (Whether a *device* is actually usable is checked inside the subprocess.)

    D7 rule (design doc §7; docs/eos_p6_gpu_alignment_review.md §2.1) — stale
    GPU kernels must be UNREACHABLE during the EOS migration window:

    * ``cuda_available()`` (no argument) — the whole-suite pin, exactly the old
      ``EOS_P6_PENDING`` bool semantics: returns False while ANY key is still
      in ``EOS_P6_PENDING_KERNELS``. Pre-P6 gates (the cuda_s* checks and any
      all-backends-on integration) stay skipped until the ENTIRE P6 arc is done,
      because they exercise the full stale surface.
    * ``cuda_available(kernel=<key>)`` — the per-kernel gate for P6 sub-patch
      tests: True iff the hardware is present AND that key has been removed
      from the pending set (i.e. its port re-proved bit-identical). Unknown
      keys raise ValueError so a typo cannot silently unpin anything."""
    if kernel is not None and kernel not in _P6_KERNEL_KEYS:
        raise ValueError(
            f"unknown P6 kernel key {kernel!r}; valid keys: "
            f"{sorted(_P6_KERNEL_KEYS)}")
    if kernel is None:
        if EOS_P6_PENDING_KERNELS:
            return False
    elif kernel in EOS_P6_PENDING_KERNELS:
        return False
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
