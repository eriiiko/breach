@echo off
REM ===========================================================================
REM CPU build — Home Desktop (DESKTOP-0E98HUV). Sibling of build_cpu_data.bat
REM (Lenovo), adapted 2026-08-22: this box has NO conda `data` env — breach
REM runs on anaconda BASE (Python 3.11.7, cp311). No ninja here either, so the
REM Visual Studio multi-config generator is used; it emits straight into
REM cpp/build/Release, the suite's sys.path target. Python is PINNED to the
REM anaconda base exe because a stray Python 3.14 exists on this system and
REM CMake finds it first otherwise (docs: environment.md, Home Desktop).
REM ===========================================================================
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
where cl
set "PYEXE=C:/Users/steen/anaconda3/python.exe"
set "PYBIND=C:/Users/steen/anaconda3/Lib/site-packages/pybind11/share/cmake/pybind11"
set "CMAKE=C:/Users/steen/anaconda3/Scripts/cmake.exe"
cd /d "%~dp0\.."
echo === CONFIGURE (cpp/build, Visual Studio 2022, CPU) ===
"%CMAKE%" -S cpp -B cpp/build --fresh -G "Visual Studio 17 2022" -A x64 ^
  -DPYBIND11_FINDPYTHON=ON ^
  -DPython_EXECUTABLE=%PYEXE% ^
  -DPYTHON_EXECUTABLE=%PYEXE% ^
  -Dpybind11_DIR=%PYBIND%
echo CONFIGURE_EXIT=%errorlevel%
echo === BUILD (Release) ===
"%CMAKE%" --build cpp/build --config Release
echo BUILD_EXIT=%errorlevel%
endlocal
