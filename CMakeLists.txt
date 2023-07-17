cmake_minimum_required(VERSION 3.10)
project(try_cpp_ai)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(3rd-party/fmt)
include_directories(3rd-party/fmt/include)

add_executable(project_x src/main.cpp)
target_link_libraries(try_cpp_ai fmt)

add_executable(test test/test.cpp)
