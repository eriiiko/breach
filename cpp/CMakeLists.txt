cmake_minimum_required(VERSION 3.18)
project(breach_physics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA-S0: optional GPU backend. OFF by default, so the canonical CPU build
# (cpp/build/Release, used by main.py) has ZERO CUDA dependency and the running
# game is untouched. Configure a SEPARATE dir with -DBREACH_CUDA=ON to bring in
# the .cu kernels + cudart. See docs/cuda_migration_plan.md §4.3 / §7.4.
option(BREACH_CUDA "Build the CUDA GPU backend (S0+ kernels)" OFF)

# Release by default — we want speed
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# MSVC: enable fast math + AVX2 SIMD. Guarded to the CXX language so these cl.exe
# flags are NEVER passed to nvcc (which rejects them) when BREACH_CUDA is ON.
# When CUDA is OFF every TU is CXX, so the guard is behaviourally a no-op — the
# existing CPU build is byte-for-byte unchanged.
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
        $<$<COMPILE_LANGUAGE:CXX>:/fp:fast>
        $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
        $<$<COMPILE_LANGUAGE:CXX>:/W3>)
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
        $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
endif()

# CUDA-S0: enable the CUDA language + toolkit only when requested.
if(BREACH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 20)            # signed-shift (SAR) portability floor
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # 86 Ampere (desktop 3070), 89 Ada (Lenovo), 75 Turing opportunistic.
        # One fatbin runs on all three machines.
        set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
    endif()
    message(STATUS "BREACH_CUDA=ON: archs ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Find pybind11 (installed via pip)
find_package(pybind11 REQUIRED)

# The physics module
set(BREACH_SOURCES
    src/atmosphere_solver.cpp
    src/smoke_dynamics.cpp
    src/fire_simulation.cpp
    src/temperature_solver.cpp
    src/raycaster.cpp
    src/water_solver.cpp
    src/bulk_transport.cpp
    src/eos_solver.cpp
    src/combustion.cpp
    src/physics_engine.cpp
    src/bindings.cpp
)
if(BREACH_CUDA)
    list(APPEND BREACH_SOURCES
        src/cuda_hello.cu          # S0 hello-world kernel
        src/cuda_temperature.cu    # S1 temperature solver
        src/cuda_raycaster.cu      # S2 raycaster march (heat bit-identical)
        src/cuda_water.cu          # S3 water solver (pipe model, bit-identical)
        src/cuda_smoke.cu          # S4a smoke solver (diffusion + SL advection)
        src/cuda_fire.cu           # S6 fire solver (logistic + plume + smoke scatter + burn)
        src/cuda_sl_advection.cu   # EOS P6.2 fused 3-field SL advection (u self + gas T)
        src/cuda_bulk_transport.cu) # EOS P6.1 bulk donor-cell flux (water K3-K8 pattern)
    # S5 cuda_wave.cu / S7 cuda_atmosphere.cu RETIRED in EOS P6.0 — their CPU
    # solvers (wave_substep / diffuse_solve) were replaced by the compressible
    # EOS solve in P3 (docs/eos_p6_gpu_alignment_review.md §1.11).
endif()

pybind11_add_module(breach_physics ${BREACH_SOURCES})

if(BREACH_CUDA)
    set_target_properties(breach_physics PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_link_libraries(breach_physics PRIVATE CUDA::cudart)
    target_compile_definitions(breach_physics PRIVATE BREACH_HAS_CUDA)
    # Determinism-correct DEVICE flags (the cross-GPU bit-identity contract,
    # plan §4.3.4): no fused-multiply-add contraction, precise div/sqrt, no
    # flush-to-zero, NEVER --use_fast_math. -allow-unsupported-compiler
    # suppresses nvcc's MSVC-version #error (VS2022 17.14 is newer than CUDA
    # 12.4 lists; harmless for pure-integer kernels — the bit-identity gate
    # would catch any codegen difference anyway). The host side of every .cu
    # gets /fp:strict (the determinism floor), matching the .cpp sim TUs.
    target_compile_options(breach_physics PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
        $<$<COMPILE_LANGUAGE:CUDA>:-prec-div=true>
        $<$<COMPILE_LANGUAGE:CUDA>:-prec-sqrt=true>
        $<$<COMPILE_LANGUAGE:CUDA>:-ftz=false>
        $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/fp:strict>)
endif()

# S0 part 3 (fixed-point prereq gate) — fast-math kill on the SIM solver TUs.
#
# The fixed-point migration needs the float PRECOMPUTE passes that feed integer
# state to be REPRODUCIBLE: no fast-math reordering, no inconsistent FMA
# contraction. So every simulation solver TU is compiled /fp:strict, which on
# MSVC disables BOTH expression reordering AND FMA contraction (stricter than the
# /fp:precise physics_engine.cpp carried before — it is upgraded to /fp:strict
# here for consistency). The wave_solver.cpp and the pure-glue bindings.cpp are
# NOT sim state and keep the global /fp:fast for speed.
#
# raycaster.cpp WAS in that render-only /fp:fast set, but the pure-density
# redesign (engine/08, CUDA-S2) makes its directional march deposit `heat` — a
# Q16.16 SYNCED gameplay field (feeds temperature -> unit damage). For the CUDA
# heat bit-identity gate (CPU reference must match the GPU's --fmad=false /fp:strict
# device math) the CPU heat path needs the SAME determinism floor: no FMA
# contraction, no reordering. So raycaster.cpp moves onto /fp:strict here. The
# render channels (light_rgb/dir/smoke_glow) ride along — they are float and a
# sub-ULP shift is cosmetically irrelevant (feel-gated). The CPU render-cast loses
# some fast-math speed, but the raycaster's perf answer is the GPU port itself.
#
# These are PER-SOURCE COMPILE_OPTIONS, which CMake appends AFTER the global
# /fp:fast from add_compile_options() above; MSVC honours the LAST /fp: switch on
# the command line, so /fp:strict wins. (Verified via a verbose build: the sim
# TUs compile with /fp:strict, /fp:fast is overridden.)
#
# This is a DELIBERATE behavior change (removing fast-math) — feel-gated, not
# 0-ULP; the /fp:fast goldens were regenerated when this landed.
if(MSVC)
    set_source_files_properties(
        src/atmosphere_solver.cpp
        src/smoke_dynamics.cpp
        src/fire_simulation.cpp
        src/water_solver.cpp
        src/bulk_transport.cpp
        src/eos_solver.cpp
        src/combustion.cpp
        src/temperature_solver.cpp
        src/physics_engine.cpp
        src/raycaster.cpp
        PROPERTIES COMPILE_OPTIONS "/fp:strict")
endif()

target_include_directories(breach_physics PRIVATE src)
