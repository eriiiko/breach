@echo off
REM ===========================================================================
REM Spike-0 run script (Windows). Run UNCHANGED on each machine
REM (Ampere RTX 30xx / Turing RTX 20xx / Ada RTX 40xx).
REM
REM It (1) sets up the MSVC host compiler env that nvcc needs, (2) compiles
REM both spikes, (3) runs them, printing all digests in a greppable block.
REM
REM ARCH: defaults to -arch=native (CUDA 12.x auto-detects the local GPU).
REM   If your CUDA toolkit is older and rejects -arch=native, run with an
REM   explicit gencode, e.g.:   set ARCH=sm_75  &  run.bat
REM   Turing RTX 20xx = sm_75 | Ampere RTX 30xx = sm_86 | Ada RTX 40xx = sm_89
REM
REM MSVC: CUDA 12.4 requires VS 2017-2022 (NOT newer). We use vswhere with a
REM   version constraint [17.0,18.0) so a too-new VS install is not picked.
REM ===========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

if "%ARCH%"=="" set ARCH=native

REM ---- locate a SUPPORTED (VS2022) vcvars64.bat --------------------------
set "VCVARS="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -products * -version "[17.0,18.0)" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VSINSTALL=%%i"
  )
  if defined VSINSTALL if exist "!VSINSTALL!\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=!VSINSTALL!\VC\Auxiliary\Build\vcvars64.bat"
  )
)
REM fallback common paths if vswhere missed
if not defined VCVARS (
  for %%P in (
    "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
  ) do (
    if not defined VCVARS if exist %%P set "VCVARS=%%~P"
  )
)
if not defined VCVARS (
  echo ERROR: could not find a VS2017-2022 vcvars64.bat for the MSVC host compiler.
  echo        Open a "x64 Native Tools Command Prompt" and run nvcc there, or set
  echo        VCVARS manually at the top of this script.
  exit /b 1
)
echo # vcvars  : %VCVARS%
REM vcvars64.bat sometimes emits harmless "'C:\Program' / 'vswhere.exe' is not
REM recognized" noise depending on the parent PATH; swallow both streams.
call "%VCVARS%" >nul 2>nul

echo ##############################################################
echo # SPIKE-0 GPU DE-RISK
echo # host : %COMPUTERNAME%
echo # arch : -arch=%ARCH%
for /f "tokens=*" %%v in ('nvcc --version ^| findstr release') do echo # nvcc : %%v
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>nul
echo ##############################################################

echo.
echo ^>^>^> Compiling spike0a_reduction.cu ...
nvcc -O2 -arch=%ARCH% -o spike0a_reduction.exe spike0a_reduction.cu
if errorlevel 1 ( echo BUILD FAILED ^(0a^) & exit /b 1 )

echo ^>^>^> Compiling spike0b_gs.cu ...
nvcc -O2 -arch=%ARCH% -o spike0b_gs.exe spike0b_gs.cu
if errorlevel 1 ( echo BUILD FAILED ^(0b^) & exit /b 1 )

echo.
echo ================ SPIKE-0a OUTPUT ================
".\spike0a_reduction.exe"

echo.
echo ================ SPIKE-0b OUTPUT ================
".\spike0b_gs.exe"
set B_RC=%errorlevel%

echo.
echo ================ SUMMARY (digests to copy into README table) ================
echo host : %COMPUTERNAME%
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>nul
".\spike0a_reduction.exe" 2>nul | findstr /C:"RESULT method" /C:"DIGEST 0a"
".\spike0b_gs.exe" 2>nul | findstr /C:"DIGEST 0b" /C:"RESULT 0b"
echo ============================================================================

if not "%B_RC%"=="0" ( echo SPIKE-0b ASSERTION FAILED rc=%B_RC% -- GPU integer != CPU integer & exit /b %B_RC% )
echo ALL DONE
endlocal
