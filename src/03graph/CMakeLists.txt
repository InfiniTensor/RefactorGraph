cmake_minimum_required(VERSION 3.10)
project(graph)

include_directories(include)
file(GLOB_RECURSE GRAPH_SRC src/*.cc src/*.cpp)
add_library(graph ${GRAPH_SRC})
target_link_libraries(graph common fmt)

file(GLOB_RECURSE GRAPH_TEST test/*.cpp)
add_executable(graph_test ${GRAPH_TEST})
target_link_libraries(graph_test fmt GTest::gtest_main)
