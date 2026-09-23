# Breach — Developer Setup Guide

*Everything needed to build and run Breach on a fresh Windows machine.*
*Repeat these steps on each dev PC (Home Desktop, Work Desktop, Lenovo laptop).*

*Last updated: 2026-09-23*

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
| Home Desktop (`DESKTOP-0E98HUV`) | `C:/Users/steen/anaconda3/python.exe` — anaconda **base**, 3.11. This machine has **no `data` env** | CPU `build_cpu_home.bat`; CUDA `build_cuda.bat` |
| Work Desktop | `C:/Users/steen/anaconda3/envs/data/python.exe` — the `data` env, 3.12 | CPU: Step 6; CUDA `build_cuda.bat` (its paths are this machine's) |
| Lenovo laptop (`ERIK_LENOVO`) | `C:/Users/steen/miniconda3/envs/data/python.exe` — the miniconda `data` env, 3.12 | CPU `build_cpu_data.bat`; CUDA `build_cuda_lenovo.bat` |

The CUDA `.pyd` is importable only by the interpreter its build script pins — `<cuda-py>` below:
`C:/Users/steen/anaconda3/python.exe` (base, 3.11) on both desktops, `<breach-py>` on the Lenovo.
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
