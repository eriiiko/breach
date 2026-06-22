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

# Patch 1 S4: the per-tick orchestration glue (physics_engine.cpp) is compiled
# /fp:precise, overriding the global /fp:fast above for JUST this TU. The glue
# carries the float math that must match numpy's strict-IEEE rounding bit-for-bit
# (the IMEX substep loop + the W3/W5 water accounting, ported in later S4 sub-
# steps). S4a moves only pure solver CALLS (no new arithmetic), so /fp:precise is
# 0-ULP-trivial here; setting it up now means the later glue inherits it. If a
# future port traces a non-0-ULP A/B mismatch to FP reordering, escalate this one
# file to /fp:strict.
if(MSVC)
    set_source_files_properties(src/physics_engine.cpp
        PROPERTIES COMPILE_OPTIONS "/fp:precise")
endif()

target_include_directories(breach_physics PRIVATE src)
