cmake_minimum_required(VERSION 3.10)
project(python_ffi)

include_directories(pybind11/include)
add_subdirectory(pybind11)

file(GLOB_RECURSE PYFFI_SRC src/*.cc src/*.cpp)
pybind11_add_module(python_ffi SHARED ${PYFFI_SRC})
target_link_libraries(python_ffi PRIVATE onnx)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
# target_compile_definitions(python_ffi
#                            PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
