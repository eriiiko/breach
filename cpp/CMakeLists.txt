cmake_minimum_required(VERSION 3.18)
project(breach_physics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release by default — we want speed
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# MSVC: enable fast math + AVX2 SIMD
if(MSVC)
    add_compile_options(/O2 /fp:fast /arch:AVX2 /W3)
else()
    add_compile_options(-O3 -ffast-math -mavx2 -Wall)
endif()

# Find pybind11 (installed via pip)
find_package(pybind11 REQUIRED)

# The physics module
pybind11_add_module(breach_physics
    src/atmosphere_solver.cpp
    src/smoke_dynamics.cpp
    src/fire_simulation.cpp
    src/temperature_solver.cpp
    src/raycaster.cpp
    src/water_solver.cpp
    src/physics_engine.cpp
    src/bindings.cpp
)

# S0 part 3 (fixed-point prereq gate) — fast-math kill on the SIM solver TUs.
#
# The fixed-point migration needs the float PRECOMPUTE passes that feed integer
# state to be REPRODUCIBLE: no fast-math reordering, no inconsistent FMA
# contraction. So every simulation solver TU is compiled /fp:strict, which on
# MSVC disables BOTH expression reordering AND FMA contraction (stricter than the
# /fp:precise physics_engine.cpp carried before — it is upgraded to /fp:strict
# here for consistency). The render-only raycaster.cpp / wave_solver.cpp and the
# pure-glue bindings.cpp are NOT sim state and keep the global /fp:fast for speed.
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
        src/temperature_solver.cpp
        src/physics_engine.cpp
        PROPERTIES COMPILE_OPTIONS "/fp:strict")
endif()

target_include_directories(breach_physics PRIVATE src)
