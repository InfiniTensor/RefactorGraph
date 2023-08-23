cmake_minimum_required(VERSION 3.10)
project(refactor_graph VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHARED "Build shared library" ON)
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

include_directories(3rd-party/abseil-cpp)
add_subdirectory(3rd-party/abseil-cpp)

include_directories(3rd-party/result)

include_directories(src/iterator/include)
add_subdirectory(src/iterator)

include_directories(src/00common/include)
add_subdirectory(src/00common)

include_directories(src/01graph_topo/include)
add_subdirectory(src/01graph_topo)

include_directories(src/02kernel_library/include)
add_subdirectory(src/02kernel_library)

include_directories(src/03graph/include)
add_subdirectory(src/03graph)

include_directories(src/04transformation/include)
add_subdirectory(src/04transformation)

add_executable(refactor_graph src/main.cpp)
target_link_libraries(refactor_graph common kernel_library graph ${BACKWARD_ENABLE})
add_backward(refactor_graph)
