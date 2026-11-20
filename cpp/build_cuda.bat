@echo off
REM ===========================================================================
REM CUDA-S0 build — the GPU backend (cpp/build_cuda/), separate from the CPU
REM build in cpp/build/Release used by the game + the existing suite.
REM
REM WHY Ninja + direct nvcc (not the VS generator): this box has no CUDA MSBuild
REM integration installed, so `-G "Visual Studio 17 2022"` fails with
REM "No CUDA toolset found". Ninja invokes nvcc directly (the spike's proven
REM path) and needs no VS CUDA integration.
REM
REM PATHS below are both desktops' (VS2022 Community, the anaconda `data` env,
REM cmake + ninja from that env). The Lenovo has build_cuda_lenovo.bat. CUDA_PATH
REM is read from the environment when set by the installer.
REM ===========================================================================
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
echo === toolchain ===
where cl
where nvcc
set "CMAKE=C:/Users/steen/anaconda3/envs/data/Scripts/cmake.exe"
set "NINJA=C:/Users/steen/anaconda3/envs/data/Scripts/ninja.exe"
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
set "PYEXE=C:/Users/steen/anaconda3/envs/data/python.exe"
set "PYBIND=C:/Users/steen/anaconda3/envs/data/Lib/site-packages/pybind11/share/cmake/pybind11"
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
