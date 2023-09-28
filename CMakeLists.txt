cmake_minimum_required(VERSION 3.10)
project(refactor_graph VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Werror)
add_compile_options(-fPIC)
# add_compile_options(-march=native) # this will cause error in some machine
add_compile_options(-mtune=native)

option(BUILD_SHARED "Build shared library" OFF)
option(ABSL_PROPAGATE_CXX_STD "Abseil need this option" ON)

if(BUILD_SHARED)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(LIBRARY_TYPE SHARED)
else()
    set(LIBRARY_TYPE STATIC)
endif()

add_subdirectory(3rd-party/backward-cpp)

include_directories(3rd-party/fmt/include)
add_subdirectory(3rd-party/fmt)

include_directories(3rd-party/googletest/googletest/include)
add_subdirectory(3rd-party/googletest)

include_directories(3rd-party/fmtlog)
add_definitions(-D FMTLOG_HEADER_ONLY)

# disable warning in 3rd-party/abseil-cpp
# see <https://github.com/abseil/abseil-cpp/pull/1287>
add_compile_options(-w)
include_directories(3rd-party/abseil-cpp)
add_subdirectory(3rd-party/abseil-cpp)
add_compile_options(-Wall)

include_directories(3rd-party/result)

include_directories(src/00common/include)
add_subdirectory(src/00common)

include_directories(src/01graph_topo/include)
add_subdirectory(src/01graph_topo)

include_directories(src/02mem_manager/include)
add_subdirectory(src/02mem_manager)

include_directories(src/03computation/include)
add_subdirectory(src/03computation)

include_directories(src/04frontend/include)
add_subdirectory(src/04frontend)

include_directories(src/05onnx/include)
add_subdirectory(src/05onnx)

include_directories(src/06communication/include)
add_subdirectory(src/06communication)

add_subdirectory(src/07python_ffi)
