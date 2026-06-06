# Breach — Developer Setup Guide

*Everything needed to build and run Breach on a fresh Windows machine.*
*Repeat these steps on each dev PC (Home Desktop, Work Desktop, Work Laptop).*

*Last updated: 2026-06-06*

---

## Prerequisites

- **Python 3.11** via Anaconda or Miniconda (already installed on all machines)
- **Git** + Git Bash (already installed)
- **VSCode** (already installed)

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
C:/Users/steen/anaconda3/python.exe -m pip install cmake
```

Verify:
```bash
C:/Users/steen/anaconda3/Scripts/cmake.exe --version
# Should print "cmake version 3.x.x"
```

---

## Step 3: Install pybind11

```bash
C:/Users/steen/anaconda3/python.exe -m pip install pybind11
```

Verify:
```bash
C:/Users/steen/anaconda3/python.exe -c "import pybind11; print(pybind11.get_cmake_dir())"
```

---

## Step 4: VSCode Extensions

Install these from the Extensions panel (Ctrl+Shift+X):

- **C/C++** (publisher: Microsoft) — IntelliSense, debugging, syntax
- **CMake Tools** (publisher: Microsoft) — build/configure integration

---

## Step 5: Install Python dependencies

```bash
C:/Users/steen/anaconda3/python.exe -m pip install raylib pytest
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

This produces `breach_physics.pyd` in `cpp/build/Release/`.

---

## Step 7: Run the game

```bash
cd C:/Users/steen/projects/breach
C:/Users/steen/anaconda3/python.exe main.py
```

Run the tests (scope to `tests/` — a bare `pytest` tries to collect the vendored third-party
`tools/` and fails on import):
```bash
C:/Users/steen/anaconda3/python.exe -m pytest tests/ -q
```

Lighting / visual tuning tool: `C:/Users/steen/anaconda3/python.exe tools/lighting_demo.py`

---

## Troubleshooting

### CMake can't find compiler
Make sure Build Tools are installed with "Desktop development with C++".
CMake should auto-detect MSVC. If not, run from a "Developer Command Prompt".

### pybind11 not found by CMake
Ensure pybind11 is installed in the same Python environment CMake uses.
The CMakeLists.txt uses `find_package(pybind11)` which checks the active Python.

### Wrong Python / missing modules (raylib, breach_physics)
Use Anaconda's Python 3.11 — that's where `raylib` and the compiled `breach_physics` module live —
not a bare system Python.
On Home/Work Desktop: `C:/Users/steen/anaconda3/python.exe`
On Laptop: `C:/Users/steen/miniconda3/python.exe`

### pytest errors on collection
Always scope to the project tests: `python -m pytest tests/`. A bare `pytest` from the repo root
tries to import the vendored third-party `tools/` (ControlAR, IP-Adapter, …) and fails before it
reaches the real tests.
