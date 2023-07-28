cmake_minimum_required(VERSION 3.10)
project(core)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)
add_library(core src/core.cc)
target_link_libraries(core fmt)

add_executable(core_test test/test.cpp)
target_link_libraries(core_test fmt)
