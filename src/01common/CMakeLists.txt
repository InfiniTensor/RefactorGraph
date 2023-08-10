cmake_minimum_required(VERSION 3.10)
project(common)

include_directories(include)
add_library(common src/core.cc)
target_link_libraries(common fmt)

add_executable(common_test test/test.cpp)
target_link_libraries(common_test fmt)
