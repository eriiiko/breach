"""S3b — the per-cell integer sqrt (fixed_point.h sqrt_q16) unit gate.

Compiles + runs the standalone C++ proof ``_s3b_sqrt_check.cpp`` (no pybind
needed) and asserts it exits 0. ``sqrt_q16`` is the FIRST per-cell transcendental
of the fixed-point arc — fire's ``W = sqrt(wind_x^2 + wind_y^2)`` — so it must be
EXACT-floor and deterministic (a wrong/non-deterministic isqrt is a silent
lockstep desync). The proof checks:

  (1) FLOOR — r = sqrt_q16(x) satisfies r^2 <= x < (r+1)^2 over a sampled int64
      domain (dense low range, exact squares +/-1, random, near the clamp edge).
  (2) SCALE FOLD — sqrt of a Q.32 radicand yields Q16.16 directly (no rescale):
      sqrt_q16(m^2 * 2^32) == m * 2^16.
  (3) DETERMINISM — pure integer ops -> bit-identical run-to-run.
  (4) CLAMP SELF-GUARD — a radicand >= 2^62 (true root >= 2^31) clamps to
      INT32_MAX deterministically (DEAD on the real fire call site), neg -> 0.
  (5) narrow_round / narrow_round_signed — the shared round-to-nearest deposit
      narrows (the fire plume/smoke/wall deposits) vs a double reference.

SKIPS gracefully if no MSVC/gcc/clang toolchain is found — a developer gate, not
a runtime dependency. Mirrors tests/test_s2b_reciprocal.py exactly.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_s3b_sqrt.py -q
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tests" / "_s3b_sqrt_check.cpp"
INC = ROOT / "cpp" / "src"
EXE = ROOT / "tests" / "_s3b_sqrt_check.exe"

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
    bat_path = ROOT / "tests" / "_s3b_build_tmp.bat"
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


def test_sqrt_q16_arithmetic_proof():
    built = _try_build_msvc() if sys.platform == "win32" else False
    exe = EXE
    if not built:
        if _try_build_unix():
            exe = EXE.with_suffix("")
        else:
            pytest.skip("no C++ toolchain (MSVC/gcc/clang) found to build the "
                        "sqrt_q16 proof")
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, (
        "sqrt_q16 proof failed:\n" + r.stdout + r.stderr)
    assert "ALL PASS" in r.stdout, r.stdout
