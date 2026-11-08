# Breach — Developer Setup Guide

*Everything needed to build and run Breach on a fresh Windows machine.*
*Repeat these steps on each dev PC (Home Desktop, Work Desktop, Lenovo laptop).*

*Last updated: 2026-11-08 (the Linux section, #29 P10)*

---

## Prerequisites

- **Python** via Anaconda or Miniconda (already installed on all machines) — which interpreter
  each machine uses is in the next section
- **Git** + Git Bash (already installed)
- **VSCode** (already installed)

---

## Which Python each machine uses

Breach runs on a different interpreter on each machine. Call it by full path: bare `python` may be
another install, and it fails breach imports with a misleading ModuleNotFoundError. Below,
`<breach-py>` is this machine's interpreter and `<breach-py-dir>` the folder that holds it. Machine
specs and tool installs live in the ClaudeSync `environment.md`.

| Machine | `<breach-py>` | Build scripts (`cpp\`) |
|---|---|---|
| Home Desktop (`DESKTOP-0E98HUV`) | `C:/Users/steen/anaconda3/envs/data/python.exe` — the `data` env, 3.12 | CPU `build_cpu_home.bat`; CUDA `build_cuda.bat` |
| Home Desktop, Linux: WSL2 distro `breach-ubuntu` | `/opt/micromamba/envs/breach/bin/python` — the micromamba `breach` env, 3.12 (section "Linux" below) | both: `build_linux.sh` |
| Work Desktop | `C:/Users/steen/anaconda3/envs/data/python.exe` — the `data` env, 3.12 | CPU: Step 6; CUDA `build_cuda.bat` |
| Lenovo laptop (`ERIK_LENOVO`) | `C:/Users/steen/miniconda3/envs/data/python.exe` — the miniconda `data` env, 3.12 | CPU `build_cpu_data.bat`; CUDA `build_cuda_lenovo.bat` |

Every machine runs breach on its `data` env. Scripts' usage lines that still name
`anaconda3/python.exe` predate that: read `<breach-py>`.

The CUDA `.pyd` is importable only by the interpreter its build script pins — `<cuda-py>` below:
`<breach-py>` on every machine.
`tests/cuda_harness.py` runs it in a subprocess under that interpreter; `BREACH_CUDA_PYTHON`
overrides it.

---

## Step 1: Install MSVC Build Tools (C++ compiler)

This is the compiler only — NOT the full Visual Studio IDE. ~3-4 GB.

1. Download **Visual Studio Build Tools 2022** from:
   https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer
3. Check **"Desktop development with C++"**
4. Click Install
5. Verify after install — open a **new** Git Bash terminal:
   ```bash
   # Find the compiler
   "/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvarsall.bat" x64
   cl.exe
   # Should print "Microsoft (R) C/C++ Optimizing Compiler Version ..."
   ```

---

## Step 2: Install CMake

```bash
<breach-py> -m pip install cmake
```

Verify:
```bash
<breach-py-dir>/Scripts/cmake.exe --version
# Should print "cmake version 3.x.x"
```

---

## Step 3: Install pybind11

```bash
<breach-py> -m pip install pybind11
```

Verify:
```bash
<breach-py> -c "import pybind11; print(pybind11.get_cmake_dir())"
```

---

## Step 4: VSCode Extensions

Install these from the Extensions panel (Ctrl+Shift+X):

- **C/C++** (publisher: Microsoft) — IntelliSense, debugging, syntax
- **CMake Tools** (publisher: Microsoft) — build/configure integration

---

## Step 5: Install Python dependencies

```bash
<breach-py> -m pip install raylib pytest
```

- **raylib** provides the `pyray` module — the renderer (replaces the old pygame prototype).
- **pytest** for the test suite. (numpy ships with Anaconda.)

---

## Step 6: Build the C++ physics module

```bash
cd C:/Users/steen/projects/breach/cpp
cmake -B build
cmake --build build --config Release
```

This produces `breach_physics.pyd` in `cpp/build/Release/`. Where the table above names a CPU build
script for this machine, use it instead — it pins the right interpreter and toolchain.

---

## Step 7: Run the game

```bash
cd C:/Users/steen/projects/breach
<breach-py> main.py
```

Run the tests (scope to `tests/` — a bare `pytest` tries to collect the vendored third-party
`tools/` and fails on import):
```bash
<breach-py> -m pytest tests/ -q
```

Lighting / visual tuning tool: `<breach-py> tools/lighting_demo.py`

---

## Linux (WSL2 on the Home Desktop; Berzelius through the container)

Added 2026-11-08 (#29 P10). The same CMakeLists.txt builds both modules on gcc + nvcc, and the
pinned match gives the Windows digests bit for bit on CPU, CUDA and CUDA resident
(`tools/match_pin_check.py`). This machine's Linux breach env is the WSL2 distro
**`breach-ubuntu`** (Ubuntu 24.04, gcc 13.3, cmake 3.28, ninja 1.11, CUDA toolkit 12.4.131 — the
Windows build's version), its disk at `C:\wsl\breach-ubuntu`, the repo at `~/code/breach` (root's
home, inside the Linux filesystem: never build on `/mnt/c`), Python
`/opt/micromamba/envs/breach/bin/python`. Times are this machine's on 2026-11-08.

**1. The distro, no prompts, no admin** (PowerShell; ~1 min):

```powershell
New-Item -ItemType Directory -Force C:\wsl\images
Invoke-WebRequest https://cloud-images.ubuntu.com/wsl/releases/noble/current/ubuntu-noble-wsl-amd64-24.04lts.rootfs.tar.gz -OutFile C:\wsl\images\ubuntu-noble-wsl-amd64-24.04lts.rootfs.tar.gz
# check it against SHA256SUMS in the same folder, then:
wsl --import breach-ubuntu C:\wsl\breach-ubuntu C:\wsl\images\ubuntu-noble-wsl-amd64-24.04lts.rootfs.tar.gz --version 2
wsl -d breach-ubuntu -- nvidia-smi        # the Windows driver's GPU, nothing installed
```

You work as root inside. Run long scripts as files (`wsl -d breach-ubuntu -- bash /mnt/c/.../x.sh`),
and from Git Bash prefix `MSYS_NO_PATHCONV=1` or it rewrites `/mnt/c` paths.

**2. DNS and IPv6** (required on this machine). WSL's generated resolver (the host's NAT DNS
proxy) times out here, and IPv6 does not route, so apt stalls at `Temporary failure resolving`:

```bash
printf '[boot]\nsystemd=true\n\n[network]\ngenerateResolvConf = false\n' > /etc/wsl.conf
rm -f /etc/resolv.conf; printf 'nameserver 1.1.1.1\nnameserver 8.8.8.8\n' > /etc/resolv.conf
printf 'Acquire::ForceIPv4 "true";\nAcquire::Retries "5";\n' > /etc/apt/apt.conf.d/99force-ipv4
# then from Windows: wsl --terminate breach-ubuntu
```

**3. Build tools and the CUDA toolkit** (apt 23 s; toolkit 103 s). The toolkit comes from NVIDIA's
`wsl-ubuntu` repository and is the TOOLKIT ONLY: never install a Linux NVIDIA driver in WSL (the
Windows driver provides `libcuda` under `/usr/lib/wsl/lib`). The `cuda-toolkit-12-4` metapackage
does not install on 24.04 (its `nsight-systems-2023.4.4` needs `libtinfo5`, gone from noble), so
take the compiler, the libraries and the command-line tools without Nsight:

```bash
apt-get update && apt-get install -y build-essential cmake ninja-build git pkg-config wget curl ca-certificates bzip2
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update
apt-get install -y cuda-compiler-12-4 cuda-libraries-dev-12-4 cuda-command-line-tools-12-4
/usr/local/cuda/bin/nvcc --version        # release 12.4, V12.4.131
```

**4. The Python env** (~3 min): `containers/breach_env.sh <ml-stack dir>` — the ONE recipe, also
the container's: micromamba into `/opt/micromamba`, the env `breach` from conda-forge (Python
3.12, numpy 2.5.3 and pybind11 3.0.2 as in Windows' `data`, pytest, pillow, matplotlib, psutil;
raylib 5.5.0.4 from pip), then the ML stack through the ml-stack skill's `ml_stack.py apply` (ONE
version set; the skill's folder is not reachable from WSL through the Google Drive link, so copy
`ml_stack.py` + `ml_stack.txt` to a folder on `C:` first). On Linux, torch's wheel pins
`cuda-toolkit==13.2.2`, which `ml_stack.txt` carries as Linux-marked pins.

**5. The repo, in the Linux filesystem** (clone 86 s): from the main checkout (a worktree's `.git`
file holds a Windows path git on Linux cannot follow):

```bash
git config --global --add safe.directory '*'
git clone --single-branch --branch <branch> /mnt/c/Users/steen/projects/breach ~/code/breach
```

Code is written in a Windows worktree and committed there; the Linux clone only pulls
(`git pull --ff-only origin <branch>`) to build and test.

**6. Build** (fresh: CPU 26 s, CUDA 67 s for archs 75 80 86 89 90):

```bash
cd ~/code/breach && cpp/build_linux.sh [--fresh] [cpu|cuda]
```

The CPU module is built in `cpp/build` and copied to `cpp/build/Release` (the suite's
`sys.path` target), the CUDA module in `cpp/build_cuda`; both carry `bp.build_id()` = HEAD. The CUDA
`.so` finds `libcudart.so.12` through its RUNPATH; `tests/cuda_harness.py` runs the GPU checks
under the suite's own interpreter on Linux (`BREACH_CUDA_PYTHON` overrides). The flags are the
CMakeLists' determinism floor: every sim TU gets `-fno-fast-math -ffp-contract=off` after the
global `-ffast-math`, every `.cu` `--fmad=false` and `-Xcompiler=-fno-fast-math,-ffp-contract=off`
(check `cpp/build*/compile_commands.json`).

**7. The suite and the digests:**

```bash
export PATH=/usr/lib/wsl/lib:$PATH        # nvidia-smi for the ML stack's gates
/opt/micromamba/envs/breach/bin/python -m pytest tests -q
for b in cpu cuda resident; do /opt/micromamba/envs/breach/bin/python tools/match_pin_check.py --backend $b; done
```

**The container** (`containers/breach.def`, Apptainer; apptainer 1.5.5 from `ppa:apptainer/ppa`
here): build and test commands are in the file's header. It is CUDA 12.4.1 devel on Ubuntu 22.04
(gcc 11), runs `breach_env.sh` with `--no-gpu` (no device at build time), builds the repo at one
commit with `build_linux.sh`, and `apptainer test --nv` plays the pinned match on all three
backends.

## Type checking and code intelligence (optional, recommended)

pyright checks the Python side, clangd the C++ and the `.cu` kernels. Both are what Claude Code's
`pyright-lsp` / `clangd-lsp` plugins and VS Code talk to; the machine-side install (binaries, PATH,
plugins) is described in the ClaudeSync `environment.md`. Repo-side:

```bash
pyright          # whole Python tree; config in pyrightconfig.json
```

clangd needs `compile_commands.json`, which the Visual Studio generator never writes, so configure
once more with Ninja from a shell that has run `vcvars64.bat`:

```bash
cmake -S cpp -B cpp/build-clangd -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON       -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=<breach-py>       -Dpybind11_DIR=<breach-py-dir>/Lib/site-packages/pybind11/share/cmake/pybind11
```

`cpp/.clangd` points clangd at that folder (gitignored). It is a real build dir, not just a database,
so `cmake --build cpp/build-clangd` produces a `breach_physics` .pyd without touching `cpp/build`.
CUDA `.cu` files need this compile database and no additional tool.

### The `breach_physics` stub

`stubs/breach_physics.pyi` is what lets pyright see into the pybind11 module. It is GENERATED —
regenerate it after any change to `cpp/src/bindings.cpp`, never hand-edit. Generate it from the
**CUDA build** (`cpp/build_cuda`): that module is the CPU one plus the `#ifdef BREACH_HAS_CUDA` block
(the `set_*_backend` family and the `cuda_*` entry points), so the `tests/cuda_*.py` harnesses
type-check too. Importing it needs the CUDA runtime's DLL folder registered first (Python ≥ 3.8
ignores PATH for an extension's DLLs — `tests/cuda_harness.py` does the same), under the interpreter
the CUDA build was made for (`<cuda-py>`, defined under the table at the top). From the repo root:

```bash
<cuda-py> -m pip install pybind11-stubgen          # once
<cuda-py> -c "import os, sys; os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin')); sys.path.insert(0, 'cpp/build_cuda'); import pybind11_stubgen; pybind11_stubgen.main(['breach_physics', '-o', 'stubs'])"
```

On a machine with no CUDA build, `PYTHONPATH=cpp/build/Release <breach-py> -m pybind11_stubgen
breach_physics -o stubs` writes the CPU subset instead, and pyright then reports every CUDA-only
symbol as an unknown attribute (144 errors on 2026-09-23, when this recipe replaced that one).

Two things in the output are expected: `HAS_CUDA: bool = True` (pyright uses the declared `bool`,
not the CUDA build's value), and three signatures — `WaterSolver.step`, `WaterSolver.step_ripple`,
`cuda_water_step` — with a defaulted parameter before required ones, which is the order those
bindings declare. Neither is a stub bug to hand-fix.

## Troubleshooting

### CMake can't find compiler
Make sure Build Tools are installed with "Desktop development with C++".
CMake should auto-detect MSVC. If not, run from a "Developer Command Prompt".

### pybind11 not found by CMake
Ensure pybind11 is installed in the same Python environment CMake uses.
The CMakeLists.txt uses `find_package(pybind11)` which checks the active Python.

### Wrong Python / missing modules (raylib, breach_physics)
Use this machine's `<breach-py>` from the table at the top — that's where `raylib` and the compiled
`breach_physics` module live — not a bare system Python. A `breach_physics` AttributeError right
after a pull usually means a stale `.pyd`: rebuild before judging the suite.

### pytest errors on collection
Always scope to the project tests: `<breach-py> -m pytest tests/`. A bare `pytest` from the repo root
tries to import the vendored third-party `tools/` (ControlAR, IP-Adapter, …) and fails before it
reaches the real tests.
