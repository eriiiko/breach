# cpp/gen_build_id.cmake -- the build id, generated at BUILD time (#29 P1b,
# residency design §6 / §13 b.16).
#
# Run by the `breach_build_id` custom target on EVERY build (cmake -P, never at
# configure): it asks git for HEAD's SHA, appends "-dirty" when a TRACKED file
# differs from HEAD (`git status --porcelain --untracked-files=no` non-empty),
# and writes OUT, one C++ TU defining `breach_build_id()`. OUT is rewritten
# only when its text changes, so a build at an unchanged commit recompiles
# nothing and a new commit recompiles this one TU plus the link -- never the
# CUDA kernels. A configure-time id would go stale after an incremental
# `cmake --build`; this one cannot (critique 4 M2).
#
# Inputs (-D): GIT_EXECUTABLE, REPO_DIR (the checkout), OUT (the TU's path).
# No git, or not a checkout: the id is "unknown" (require_snapshot refuses it).
if(NOT GIT_EXECUTABLE)
    set(GIT_EXECUTABLE git)
endif()
execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${REPO_DIR}"
    OUTPUT_VARIABLE sha RESULT_VARIABLE rc
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT rc EQUAL 0 OR sha STREQUAL "")
    set(id "unknown")
else()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=no
        WORKING_DIRECTORY "${REPO_DIR}"
        OUTPUT_VARIABLE st RESULT_VARIABLE rc2
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT rc2 EQUAL 0)
        set(id "${sha}-unknown")
    elseif(NOT st STREQUAL "")
        set(id "${sha}-dirty")
    else()
        set(id "${sha}")
    endif()
endif()
set(content "// GENERATED at build time by cpp/gen_build_id.cmake (#29 P1b): never edit,\n// never commit. The git commit this module was built from; bindings.cpp\n// exposes it as breach_physics.build_id().\nconst char* breach_build_id() { return \"${id}\"; }\n")
set(old "")
if(EXISTS "${OUT}")
    file(READ "${OUT}" old)
endif()
if(NOT old STREQUAL content)
    file(WRITE "${OUT}" "${content}")
    message(STATUS "breach build id: ${id}")
endif()
