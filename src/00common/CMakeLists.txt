cmake_minimum_required(VERSION 3.10)
project(common)

include_directories(include)
file(GLOB_RECURSE COMMON_SRC src/*.cc)
add_library(common ${COMMON_SRC})
target_link_libraries(common fmt)

add_executable(common_test test/test.cpp)
target_link_libraries(common_test fmt)
