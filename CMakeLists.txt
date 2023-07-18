cmake_minimum_required(VERSION 3.10)
project(try_cpp_ai)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(3rd-party/fmt)
include_directories(3rd-party/fmt/include)

add_subdirectory(src/core)
include_directories(src/core/include)

add_subdirectory(src/kernel_list)
include_directories(src/kernel_list/include)

add_executable(try_cpp_ai src/main.cpp)
target_link_libraries(try_cpp_ai core kernel_list)

add_executable(test test/test.cpp)
