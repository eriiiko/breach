cmake_minimum_required(VERSION 3.14)
project(breach_raylib_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Download raylib automatically
include(FetchContent)
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        5.5
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(raylib)

# Explosion test
add_executable(breach_raylib_test main.cpp)
target_link_libraries(breach_raylib_test PRIVATE raylib)

# Flashlight test
add_executable(flashlight_test flashlight.cpp)
target_link_libraries(flashlight_test PRIVATE raylib)

# Copy executables after build
add_custom_command(TARGET breach_raylib_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:breach_raylib_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/breach_raylib_test.exe
)
add_custom_command(TARGET flashlight_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:flashlight_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/flashlight_test.exe
)
