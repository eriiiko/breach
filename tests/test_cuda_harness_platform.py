"""The CUDA harness finds and loads the GPU build on this platform (#29 P10).

Pure-Python checks of ``tests/cuda_harness.py``'s platform half; they import
no breach_physics and need no GPU, so they run on every machine.

    <breach py> -m pytest tests/test_cuda_harness_platform.py -q
"""
from __future__ import annotations

import importlib.machinery
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import cuda_harness  # noqa: E402


def test_module_glob_matches_what_this_python_imports():
    """Property: the harness's module pattern names a file this interpreter
    can import as ``breach_physics`` (one of its EXTENSION_SUFFIXES). Breaks
    if the pattern is hard-coded to one platform's extension again (the
    pre-P10 ``*.pyd``, which made every CUDA gate skip on Linux)."""
    ext = cuda_harness.MODULE_GLOB.rsplit("*", 1)[1]
    assert cuda_harness.MODULE_GLOB.startswith("breach_physics")
    assert any(sfx.endswith(ext)
               for sfx in importlib.machinery.EXTENSION_SUFFIXES)


def test_bootstrap_registers_dlls_only_where_the_call_exists():
    """Property: the subprocess bootstrap calls ``os.add_dll_directory``
    exactly when this platform has it (Windows), so the GPU script does not
    die with AttributeError before its first line elsewhere. Breaks if the
    registration is made unconditional again."""
    import os
    boot = cuda_harness._bootstrap()
    assert ("add_dll_directory" in boot) == hasattr(os, "add_dll_directory")


def test_runtime_dir_holds_this_platforms_cudart():
    """Property: when the harness reports a CUDA runtime folder, it holds
    this platform's runtime library (cudart64_*.dll / libcudart.so*). Breaks
    if the lookup returns a folder of the other platform's layout."""
    d = cuda_harness.cuda_dll_dir()
    if d is None:
        return  # no toolkit on this machine: nothing to check
    pattern = "cudart64_*.dll" if sys.platform == "win32" else "libcudart.so*"
    assert list(d.glob(pattern))
