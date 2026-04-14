cmake_minimum_required(VERSION 3.18)
project(neural_codec_csdf LANGUAGES CXX)

option(WITH_CUDA "Build CUDA rasterizer backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ─── Dependencies ─────────────────────────────────────────────────────────────

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# pybind11 — try system package first, fall back to pip-installed
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _PB11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _PB11_RESULT)
    if(_PB11_RESULT EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${_PB11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR
            "pybind11 not found. Install it with:  pip install pybind11")
    endif()
endif()

find_package(OpenMP)

# ─── CPU extension (csdf_ext) ─────────────────────────────────────────────────

pybind11_add_module(csdf_ext csdf/rasterizer.cpp)

target_include_directories(csdf_ext PRIVATE "${CMAKE_SOURCE_DIR}/csdf")

target_compile_options(csdf_ext PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -ffast-math>
    $<$<CXX_COMPILER_ID:MSVC>:/O2>)

if(OpenMP_CXX_FOUND)
    target_link_libraries(csdf_ext PRIVATE OpenMP::OpenMP_CXX)
    # Note: _OPENMP is already defined by -fopenmp; no need to redefine it.
    message(STATUS "OpenMP found — parallel rasterization enabled")
else()
    message(WARNING "OpenMP not found — rasterizer will run single-threaded")
endif()

# Install the .so directly into the csdf/ Python package directory so that
# `from csdf import csdf_ext` works without any PYTHONPATH manipulation.
install(TARGETS csdf_ext
        LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/csdf"
        RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/csdf")

# ─── CUDA extension (stub — not yet implemented) ──────────────────────────────

if(WITH_CUDA)
    message(FATAL_ERROR "CUDA rasterizer backend is not yet implemented.")
endif()
