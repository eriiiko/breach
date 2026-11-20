#!/usr/bin/env bash
# ===========================================================================
# The Linux breach Python env (#29 P10), the ONE recipe for the breach-ubuntu
# WSL distro and the container (containers/breach.def):
#
#   containers/breach_env.sh <ml-stack dir> [--no-gpu]
#
# 1. micromamba into $MAMBA_ROOT_PREFIX (default /opt/micromamba) if absent;
# 2. the env `breach` from conda-forge: Python 3.12 and the non-ML packages
#    the suite imports, at the Windows `data` env's versions where a compiled
#    module depends on them (numpy, pybind11), plus raylib from pip;
# 3. the ML stack (torch, CuPy, scipy, ...) through the ml-stack skill's
#    ml_stack.py -- ONE version set (ml_stack.txt), never installed by hand,
#    never an environment.yml. `--no-gpu` (a container build: no device yet)
#    skips the gates that need one; run `ml_stack.py verify` on a GPU later.
#
# The env's python is then $MAMBA_ROOT_PREFIX/envs/breach/bin/python, which
# cpp/build_linux.sh takes by default.
# ===========================================================================
set -euo pipefail

ML_STACK_DIR="${1:?usage: breach_env.sh <ml-stack dir> [--no-gpu]}"
shift
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/opt/micromamba}"
MM="$MAMBA_ROOT_PREFIX/bin/micromamba"
ENV="$MAMBA_ROOT_PREFIX/envs/breach"

if [ ! -x "$MM" ]; then
    mkdir -p "$MAMBA_ROOT_PREFIX"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C "$MAMBA_ROOT_PREFIX" bin/micromamba
fi
"$MM" --version

if [ ! -x "$ENV/bin/python" ]; then
    "$MM" create -y -q -n breach -c conda-forge \
        python=3.12 numpy=2.5.3 pybind11=3.0.2 pytest=9.1.1 \
        pillow matplotlib psutil pip
fi
"$ENV/bin/python" -m pip install -q --root-user-action=ignore raylib==5.5.0.4

"$ENV/bin/python" "$ML_STACK_DIR/ml_stack.py" apply "$@"
