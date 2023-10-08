cmake_minimum_required(VERSION 3.10)
project(common)

file(GLOB_RECURSE COMMON_SRC src/*.cc)
add_library(common STATIC ${COMMON_SRC})
target_link_libraries(common PUBLIC fmt)

file(GLOB_RECURSE COMMON_TEST test/*.cpp)
add_executable(common_test ${COMMON_TEST})
target_link_libraries(common_test common GTest::gtest_main)
