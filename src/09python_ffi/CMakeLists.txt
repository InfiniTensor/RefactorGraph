cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(python_ffi VERSION 0.0.0 LANGUAGES CXX)
message(STATUS "Project " ${PROJECT_NAME} " version " ${PROJECT_VERSION})

include_directories(pybind11/include)
add_subdirectory(pybind11)

file(GLOB_RECURSE PYFFI_SRC src/*.cc src/*.cpp)
pybind11_add_module(python_ffi SHARED ${PYFFI_SRC})
target_link_libraries(python_ffi PRIVATE onnx llm communication)
target_include_directories(python_ffi PRIVATE include)

if(USE_BANG)
    target_include_directories(python_ffi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../04kernel/src/utilities/bang)
endif()

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
# target_compile_definitions(python_ffi
#                            PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
