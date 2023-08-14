cmake_minimum_required(VERSION 3.10)
project(try_cpp_ai)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(3rd-party/backward-cpp)

add_subdirectory(3rd-party/fmt)
include_directories(3rd-party/fmt/include)

add_subdirectory(3rd-party/googletest)
include_directories(3rd-party/googletest/googletest/include)

include_directories(3rd-party/fmtlog)
add_definitions(-D FMTLOG_HEADER_ONLY)

add_subdirectory("src/iterator")
include_directories("src/iterator/include")

add_subdirectory("src/00common")
include_directories("src/00common/include")

add_subdirectory("src/01graph_topo")
include_directories("src/01graph_topo/include")

add_subdirectory("src/02kernel_library")
include_directories("src/02kernel_library/include")

add_subdirectory("src/03graph")
include_directories("src/03graph/include")

add_executable(refactor_graph src/main.cpp)
target_link_libraries(refactor_graph common kernel_library graph ${BACKWARD_ENABLE})
add_backward(refactor_graph)

add_executable(test test/test.cpp)
