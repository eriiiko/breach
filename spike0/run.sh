#!/usr/bin/env bash
# =============================================================================
# Spike-0 run script for Git Bash on Windows. Run this UNCHANGED on each
# machine (Ampere RTX 30xx / Turing RTX 20xx / Ada RTX 40xx).
#
# On Windows, nvcc needs the MSVC host compiler (cl.exe) on PATH, which is set
# up by Visual Studio's vcvars64.bat. Since that is a cmd batch file, this
# script compiles via a short `cmd //c` that sources vcvars first, then runs
# the built exes directly from bash.
#
# If you are on a real Unix box (gcc host), this indirection is unnecessary --
# just compile by hand:
#     nvcc -O2 -arch=native -o spike0a_reduction.exe spike0a_reduction.cu
#     nvcc -O2 -arch=native -o spike0b_gs.exe       spike0b_gs.cu
#
# ARCH: defaults to native (CUDA 12.x auto-detects). For an older toolkit:
#     ARCH=sm_75 ./run.sh    (Turing=sm_75, Ampere=sm_86, Ada=sm_89)
# =============================================================================
set -u
cd "$(dirname "$0")"

ARCH="${ARCH:-native}"

echo "##############################################################"
echo "# SPIKE-0 GPU DE-RISK"
echo "# host    : $(hostname 2>/dev/null || echo unknown)"
echo "# date    : $(date 2>/dev/null || echo unknown)"
echo "# arch    : -arch=$ARCH"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "# gpu     : $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -n1)"
fi
echo "##############################################################"
echo
echo ">>> Compiling both spikes (via vcvars + nvcc) ..."

# Delegate the MSVC-env + nvcc compile to run.bat's own logic by calling a
# tiny cmd shell. run.bat does the heavy lifting (find vcvars, set up MSVC,
# compile, run, print everything). ARCH is exported so run.bat inherits it.
# MSYS_NO_PATHCONV stops Git Bash from mangling the cmd arguments.
export ARCH
# `.\run.bat` (not bare `run.bat`) so cmd finds it in the current directory.
MSYS_NO_PATHCONV=1 cmd /c ".\\run.bat"
RC=$?

if [ $RC -ne 0 ]; then
  echo "run.bat exited with code $RC"
  exit $RC
fi
echo "(run.sh) ALL DONE (rc=$RC)"
