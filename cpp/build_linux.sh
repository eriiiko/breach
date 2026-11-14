#!/usr/bin/env bash
# ===========================================================================
# Linux build (#29 P10, residency design §8): the CPU module AND the CUDA
# module, the way the Windows scripts build them -- CMake + Ninja, the same
# CMakeLists.txt, so the same determinism floor (the BREACH_FP_STRICT list:
# -fno-fast-math -ffp-contract=off on every sim TU and, through -Xcompiler,
# on the host side of every .cu; --fmad=false -prec-div/-prec-sqrt -ftz=false
# on the device side) and the same build id (the breach_build_id target).
#
#   cpp/build_linux.sh            # both modules
#   cpp/build_linux.sh cpu        # cpp/build (staged into cpp/build/Release)
#   cpp/build_linux.sh cuda       # cpp/build_cuda
#   cpp/build_linux.sh --fresh    # wipe the build dir(s) first
#
# Outputs, where the suite and the harness look for them:
#   cpp/build/Release/breach_physics*.so   (Ninja is single-config: built in
#                                           cpp/build, copied here -- the
#                                           sys.path target, like
#                                           build_cpu_data.bat)
#   cpp/build_cuda/breach_physics*.so      (tests/cuda_harness.py)
#
# The interpreter: BREACH_PYTHON, default /opt/micromamba/envs/breach/bin/python
# (the breach-ubuntu WSL distro's env and the container's, docs/dev_setup.md
# "Linux"). nvcc: CUDACXX, default /usr/local/cuda/bin/nvcc. CUDA archs: the
# CMakeLists default (75 80 86 89 90); override with BREACH_CUDA_ARCHS="86".
# ===========================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${BREACH_PYTHON:-/opt/micromamba/envs/breach/bin/python}"
NVCC="${CUDACXX:-/usr/local/cuda/bin/nvcc}"

fresh=0
targets=()
for a in "$@"; do
    case "$a" in
        --fresh) fresh=1 ;;
        cpu|cuda) targets+=("$a") ;;
        *) echo "build_linux.sh: unknown argument '$a' (cpu | cuda | --fresh)" >&2; exit 2 ;;
    esac
done
[ ${#targets[@]} -eq 0 ] && targets=(cpu cuda)

[ -x "$PY" ] || { echo "build_linux.sh: no python at $PY (set BREACH_PYTHON)" >&2; exit 2; }
PYBIND="$("$PY" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
common=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE="$PY" -DPYTHON_EXECUTABLE="$PY"
        -Dpybind11_DIR="$PYBIND")
echo "=== toolchain ==="
echo "python: $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"
echo "c++:    $(c++ --version | head -1)"
echo "cmake:  $(cmake --version | head -1)   ninja: $(ninja --version)"

for t in "${targets[@]}"; do
    start=$(date +%s)
    if [ "$t" = cpu ]; then
        dir=cpp/build
        [ $fresh -eq 1 ] && rm -rf "$dir"
        echo "=== CONFIGURE ($dir, Ninja, CPU) ==="
        cmake -S cpp -B "$dir" "${common[@]}"
        echo "=== BUILD ($dir) ==="
        cmake --build "$dir"
        mkdir -p "$dir/Release"
        cp -f "$dir"/breach_physics*.so "$dir/Release/"
        echo "staged: $(ls "$dir"/Release/breach_physics*.so)"
    else
        dir=cpp/build_cuda
        [ $fresh -eq 1 ] && rm -rf "$dir"
        [ -x "$NVCC" ] || { echo "build_linux.sh: no nvcc at $NVCC (set CUDACXX)" >&2; exit 2; }
        echo "nvcc:   $("$NVCC" --version | tail -1)"
        archs=()
        [ -n "${BREACH_CUDA_ARCHS:-}" ] && archs=(-DCMAKE_CUDA_ARCHITECTURES="$BREACH_CUDA_ARCHS")
        echo "=== CONFIGURE ($dir, Ninja, BREACH_CUDA=ON) ==="
        cmake -S cpp -B "$dir" "${common[@]}" -DBREACH_CUDA=ON \
            -DCMAKE_CUDA_COMPILER="$NVCC" "${archs[@]}"
        echo "=== BUILD ($dir) ==="
        cmake --build "$dir"
        echo "built: $(ls "$dir"/breach_physics*.so)"
    fi
    echo "=== $t done in $(( $(date +%s) - start )) s ==="
done
