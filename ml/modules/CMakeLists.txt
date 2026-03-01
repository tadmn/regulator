cmake_minimum_required(VERSION 3.18)

include(../../shared/cmake/CPM.cmake)

project(audio_features LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

CPMAddPackage("gh:tadmn/tad-bits#3e79ba0eee59fccf7c74ea7b48de056313f86e0b")
include(${tad-bits_SOURCE_DIR}/cmake/compile-options.cmake)

CPMAddPackage("gh:tadmn/FastFourier#453106d9d2995aaca8cf53ea556749c1e07bdb05")

# Use local python from .venv
set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/../.venv/bin/python" CACHE FILEPATH "Python from venv" FORCE)
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development)

# Use local pybind11 from .venv
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_EXIT
)
if(NOT pybind11_EXIT EQUAL 0)
    message(FATAL_ERROR
        "Could not import pybind11 from ${Python_EXECUTABLE}\n"
        "Run:  uv sync  (from the repo root) then retry.")
endif()
message(STATUS "pybind11 cmake dir: ${pybind11_DIR}")
find_package(pybind11 CONFIG REQUIRED)

find_package(SndFile REQUIRED)

# C++ module
pybind11_add_module(audio_features src/AudioFeatures.cpp)
target_link_libraries(audio_features PRIVATE tad-bits FastFourier SndFile::sndfile)
target_include_directories(audio_features PRIVATE ../../shared)

# Drop the .so into python/ so `import audio_features` works from the repo root
set_target_properties(audio_features PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python")
