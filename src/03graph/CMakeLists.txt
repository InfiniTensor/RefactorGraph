cmake_minimum_required(VERSION 3.10)
project(graph)

include_directories(include)
file(GLOB_RECURSE GRAPH_SRC src/*.cc src/*.cpp)

add_library(graph ${LIBRARY_TYPE} ${GRAPH_SRC})
target_link_libraries(graph PUBLIC common)

file(GLOB_RECURSE GRAPH_TEST test/*.cpp)
add_executable(graph_test ${GRAPH_TEST})
target_link_libraries(graph_test graph GTest::gtest_main ${BACKWARD_ENABLE})
add_backward(graph_test)
