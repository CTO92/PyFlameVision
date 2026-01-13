# PyFlameVision: Cerebras-native Computer Vision Library
# Root CMakeLists.txt

cmake_minimum_required(VERSION 3.18)

# Version information
set(PYFLAME_VISION_VERSION_MAJOR 1)
set(PYFLAME_VISION_VERSION_MINOR 0)
set(PYFLAME_VISION_VERSION_PATCH 0)
set(PYFLAME_VISION_VERSION_SUFFIX "-alpha")
set(PYFLAME_VISION_VERSION_FULL "${PYFLAME_VISION_VERSION_MAJOR}.${PYFLAME_VISION_VERSION_MINOR}.${PYFLAME_VISION_VERSION_PATCH}${PYFLAME_VISION_VERSION_SUFFIX}")

project(PyFlameVision
    VERSION ${PYFLAME_VISION_VERSION_MAJOR}.${PYFLAME_VISION_VERSION_MINOR}.${PYFLAME_VISION_VERSION_PATCH}
    LANGUAGES CXX
    DESCRIPTION "Cerebras-native computer vision library for PyFlame"
)

# ============================================================================
# Options
# ============================================================================

option(PYFLAME_VISION_BUILD_PYTHON "Build Python bindings" ON)
option(PYFLAME_VISION_BUILD_TESTS "Build unit tests" ON)
option(PYFLAME_VISION_BUILD_EXAMPLES "Build example programs" ON)
option(PYFLAME_VISION_STANDALONE "Build without PyFlame dependency (limited functionality)" OFF)

# ============================================================================
# Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Export compile commands for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ============================================================================
# Paths
# ============================================================================

set(PYFLAME_VISION_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PYFLAME_VISION_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(PYFLAME_VISION_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ============================================================================
# Compiler flags
# ============================================================================

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3 -DNDEBUG)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /permissive- /Zc:__cplusplus)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi)
    else()
        add_compile_options(/O2 /DNDEBUG)
    endif()
endif()

# ============================================================================
# Find PyFlame (optional for standalone build)
# ============================================================================

if(NOT PYFLAME_VISION_STANDALONE)
    # Try to find installed PyFlame
    find_package(PyFlame CONFIG QUIET)

    # Try adjacent source directory
    if(NOT PyFlame_FOUND)
        set(PYFLAME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../PyFlame" CACHE PATH "PyFlame source directory")
        if(EXISTS "${PYFLAME_DIR}/CMakeLists.txt")
            message(STATUS "Found PyFlame source at: ${PYFLAME_DIR}")
            set(PyFlame_FOUND TRUE)
            set(PYFLAME_INCLUDE_DIR "${PYFLAME_DIR}/include")
        endif()
    endif()

    if(PyFlame_FOUND)
        message(STATUS "PyFlame integration: ENABLED")
    else()
        message(STATUS "PyFlame not found - building in standalone mode")
        set(PYFLAME_VISION_STANDALONE ON)
    endif()
endif()

if(PYFLAME_VISION_STANDALONE)
    message(STATUS "PyFlame integration: DISABLED (standalone mode)")
endif()

# ============================================================================
# Python bindings (pybind11)
# ============================================================================

if(PYFLAME_VISION_BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

    # Try to find pybind11
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# ============================================================================
# Google Test
# ============================================================================

if(PYFLAME_VISION_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    include(GoogleTest)
endif()

# ============================================================================
# Subdirectories
# ============================================================================

add_subdirectory(src)

if(PYFLAME_VISION_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(PYFLAME_VISION_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(PYFLAME_VISION_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== PyFlameVision ${PYFLAME_VISION_VERSION_FULL} Configuration ===")
message(STATUS "")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "  Build Python:      ${PYFLAME_VISION_BUILD_PYTHON}")
message(STATUS "  Build tests:       ${PYFLAME_VISION_BUILD_TESTS}")
message(STATUS "  Build examples:    ${PYFLAME_VISION_BUILD_EXAMPLES}")
message(STATUS "  Standalone mode:   ${PYFLAME_VISION_STANDALONE}")
if(NOT PYFLAME_VISION_STANDALONE AND PYFLAME_DIR)
    message(STATUS "  PyFlame dir:       ${PYFLAME_DIR}")
endif()
message(STATUS "")
