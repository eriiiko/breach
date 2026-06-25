"""S2a — the deterministic integer mean reduction (fixed_point.h mean_sum /
mean_round) unit gate.

Compiles + runs the standalone C++ proof ``_s2a_mean_reduction_check.cpp``
(no pybind needed) and asserts it exits 0. The proof checks the three
determinism-critical properties of the ``mean_wp`` primitive:

  (1) ORDER-INDEPENDENCE — ``mean_sum`` over a masked Q16.16 array is bit-
      identical under an arbitrary element shuffle (integer add is associative);
      this is the cross-GPU property the float sum lacked (spike-0a).
  (2) MATCHES A REFERENCE — ``mean_round(sum, count)`` equals an independent
      round-half-away-from-zero reference over a wide sweep (incl. negatives,
      exact .5 ties, both count parities).
  (3) SIGN SYMMETRY — ``mean_round(-s, c) == -mean_round(s, c)`` exactly (the
      DC-bias guard: a biased mean leaks mass into the conserved atmosphere).

SKIPS gracefully if no MSVC/gcc/clang toolchain is found (e.g. a CI box without
a compiler) — the proof is a developer gate, not a runtime dependency.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_s2a_mean_reduction.py -q
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tests" / "_s2a_mean_reduction_check.cpp"
INC = ROOT / "cpp" / "src"
EXE = ROOT / "tests" / "_s2a_mean_reduction_check.exe"

_VCVARS_CANDIDATES = (
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
)


def _try_build_msvc() -> bool:
    vcvars = next((p for p in _VCVARS_CANDIDATES if os.path.exists(p)), None)
    if vcvars is None:
        return False
    bat = (
        f'@echo off\r\n'
        f'call "{vcvars}" >nul 2>&1\r\n'
        f'cl /std:c++20 /O2 /EHsc /nologo /I "{INC}" "{SRC}" '
        f'/Fe:"{EXE}" /Fo:"{EXE.with_suffix(".obj")}"\r\n'
    )
    bat_path = ROOT / "tests" / "_s2a_build_tmp.bat"
    bat_path.write_text(bat, encoding="ascii")
    try:
        r = subprocess.run(["cmd.exe", "/c", str(bat_path)],
                           capture_output=True, text=True)
        return r.returncode == 0 and EXE.exists()
    finally:
        bat_path.unlink(missing_ok=True)


def _try_build_unix() -> bool:
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        return False
    out = EXE.with_suffix("")
    r = subprocess.run([cxx, "-std=c++20", "-O2", "-I", str(INC), str(SRC),
                        "-o", str(out)], capture_output=True, text=True)
    return r.returncode == 0 and out.exists()


def test_mean_reduction_arithmetic_proof():
    built = _try_build_msvc() if sys.platform == "win32" else False
    exe = EXE
    if not built:
        if _try_build_unix():
            exe = EXE.with_suffix("")
        else:
            pytest.skip("no C++ toolchain (MSVC/gcc/clang) found to build the "
                        "mean-reduction proof")
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, (
        "mean-reduction proof failed:\n" + r.stdout + r.stderr)
    assert "ALL PASS" in r.stdout, r.stdout
