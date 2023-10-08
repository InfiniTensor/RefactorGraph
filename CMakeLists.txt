cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(refactor_graph VERSION 0.0.0 LANGUAGES CXX)

option(ABSL_PROPAGATE_CXX_STD "Abseil need this option" ON)
option(USE_CUDA "Build with cuda" OFF)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()
    message(STATUS "CMAKE_CUDA_HOST_COMPILER set to " ${CMAKE_CUDA_HOST_COMPILER})
    message(STATUS "CMAKE_CUDA_ARCHITECTURES set to " ${CMAKE_CUDA_ARCHITECTURES})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Werror)
add_compile_options(-fPIC)
# add_compile_options(-march=native) # this will cause error in some machine
add_compile_options(-mtune=native)

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

include_directories(src/03kernel/include)
add_subdirectory(src/03kernel)

include_directories(src/04computation/include)
add_subdirectory(src/04computation)

include_directories(src/05frontend/include)
add_subdirectory(src/05frontend)

include_directories(src/06onnx/include)
add_subdirectory(src/06onnx)

include_directories(src/07communication/include)
add_subdirectory(src/07communication)

add_subdirectory(src/08python_ffi)
