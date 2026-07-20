@echo off
REM ===========================================================================
REM CPU build against the conda `data` env — the interpreter the test suite
REM runs under (project CLAUDE.md: "Python: always the conda env `data`"). The
REM canonical CPU build the game + `pytest tests` load from cpp/build/Release.
REM Ninja generator + MSVC (vcvars64), BREACH_CUDA OFF (the .cu kernels are the
REM separate cpp/build_cuda via build_cuda*.bat). Sibling of build_cuda.bat.
REM
REM PER-MACHINE PATHS (like build_cuda.bat): the VS edition + the conda env
REM prefix below are this box's; adjust on another PC. The pyd is copied into
REM cpp/build/Release so the test-suite sys.path insert (cpp/build/Release)
REM finds it (Ninja is single-config and emits into cpp/build directly).
REM ===========================================================================
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
if not exist "%VCINSTALLDIR%" call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
where cl
set "PYEXE=C:/Users/steen/miniconda3/envs/data/python.exe"
set "PYBIND=C:/Users/steen/miniconda3/envs/data/Lib/site-packages/pybind11/share/cmake/pybind11"
set "CMAKE=C:/Users/steen/miniconda3/envs/data/Scripts/cmake.exe"
set "NINJA=C:/Users/steen/miniconda3/envs/data/Scripts/ninja.exe"
cd /d "%~dp0\.."
echo === CONFIGURE (cpp/build, Ninja, CPU) ===
"%CMAKE%" -S cpp -B cpp/build -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DPython_EXECUTABLE=%PYEXE% ^
  -DPYTHON_EXECUTABLE=%PYEXE% ^
  -Dpybind11_DIR=%PYBIND%
echo CONFIGURE_EXIT=%errorlevel%
echo === BUILD ===
"%CMAKE%" --build cpp/build
echo BUILD_EXIT=%errorlevel%
echo === STAGE into cpp/build/Release (the suite's sys.path target) ===
if not exist "cpp\build\Release" mkdir "cpp\build\Release"
copy /Y "cpp\build\breach_physics*.pyd" "cpp\build\Release\" >nul
endlocal
