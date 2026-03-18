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
    src/wave_solver.cpp
    src/atmo_diffusion.cpp
    src/smoke_dynamics.cpp
    src/fire_simulation.cpp
    src/raycaster.cpp
    src/physics_engine.cpp
    src/bindings.cpp
)

target_include_directories(breach_physics PRIVATE src)
