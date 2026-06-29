@echo off
REM ===========================================================================
REM CUDA build — the GPU backend (cpp/build_cuda/) — LENOVO (Ada) variant.
REM
REM Mirrors build_cuda.bat (Work Desktop) but with this machine's paths:
REM   - VS 2022 BUILD TOOLS (not Community)        -> vcvars64 below
REM   - miniconda `data` env (py3.12, has torch)   -> PYEXE / PYBIND below
REM   - CUDA 12.9 (accepts MSVC 14.44; CMakeLists already passes
REM     -allow-unsupported-compiler so nvcc's VS-version #error is suppressed)
REM   - cmake + ninja from the `data` env (pip-installed)
REM
REM WHY Ninja + direct nvcc (not the VS generator): no CUDA MSBuild integration
REM is assumed; Ninja invokes nvcc directly (the spike's proven path).
REM
REM Driver 596.47 (CUDA 13.2-capable) runs 12.9-built code fine (forward compat).
REM ===========================================================================
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
echo === toolchain ===
where cl
where nvcc
set "CMAKE=C:\Users\steen\miniconda3\envs\data\Scripts\cmake.exe"
set "NINJA=C:\Users\steen\miniconda3\envs\data\Scripts\ninja.exe"
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
set "PYEXE=C:/Users/steen/miniconda3/envs/data/python.exe"
set "PYBIND=C:/Users/steen/miniconda3/envs/data/Lib/site-packages/pybind11/share/cmake/pybind11"
cd /d "%~dp0\.."
echo === CONFIGURE (cpp/build_cuda, Ninja, BREACH_CUDA=ON) ===
"%CMAKE%" -S cpp -B cpp/build_cuda -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DCMAKE_CUDA_COMPILER="%NVCC%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBREACH_CUDA=ON ^
  -DPython_EXECUTABLE=%PYEXE% ^
  -DPYTHON_EXECUTABLE=%PYEXE% ^
  -Dpybind11_DIR=%PYBIND% ^
  -DCMAKE_CUDA_ARCHITECTURES="75;86;89"
echo CONFIGURE_EXIT=%errorlevel%
echo === BUILD ===
"%CMAKE%" --build cpp/build_cuda
echo BUILD_EXIT=%errorlevel%
endlocal
