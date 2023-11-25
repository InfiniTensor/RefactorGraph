cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(refactor_graph VERSION 0.0.0 LANGUAGES CXX)

option(ABSL_PROPAGATE_CXX_STD "Abseil need this option" ON)
option(USE_CUDA "Build with cuda" OFF)
option(USE_KUNLUN "Support Kunlun XPU" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(USE_CUDA)
    add_compile_definitions(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_CXX_EXTENSIONS OFF)
    endif()
    message(STATUS "CMAKE_CUDA_HOST_COMPILER set to " ${CMAKE_CUDA_HOST_COMPILER})
    message(STATUS "CMAKE_CUDA_ARCHITECTURES set to " ${CMAKE_CUDA_ARCHITECTURES})
endif()

if(USE_KUNLUN)
    add_compile_definitions(USE_KUNLUN)
    if (DEFINED KUNLUN_HOME)
        set(KUNLUN_HOME ${KUNLUN_HOME} CACHE STRING "KUNLUN_HOME directory for Kunlun development")
    elseif(DEFINED ENV{KUNLUN_HOME})
        set(KUNLUN_HOME $ENV{KUNLUN_HOME} CACHE STRING "KUNLUN_HOME directory for Kunlun development")
    else()
        message(FATAL_ERROR "KUNLUN_HOME is not defined from cmake or env")
    endif()
    message(STATUS "KUNLUN_HOME: ${KUNLUN_HOME}")
endif()

# add_compile_options(-march=native) # this will cause error in some machine
add_compile_options(-mtune=native)
add_compile_options(-fPIC)

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

enable_testing()

add_subdirectory(src/00common)
add_subdirectory(src/01graph_topo)
add_subdirectory(src/02mem_manager)
add_subdirectory(src/03runtime)
add_subdirectory(src/04kernel)
add_subdirectory(src/05computation)
add_subdirectory(src/06frontend)
add_subdirectory(src/07onnx)
add_subdirectory(src/08communication)
add_subdirectory(src/09python_ffi)
