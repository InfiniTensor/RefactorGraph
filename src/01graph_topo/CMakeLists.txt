cmake_minimum_required(VERSION 3.10)
project(graph_topo)

# add_library(graph_topo src/core.cc)
# target_link_libraries(graph_topo fmt)

file(GLOB_RECURSE GRAPH_TEST test/*.cpp)
add_executable(graph_topo_test ${GRAPH_TEST})
target_link_libraries(graph_topo_test graph GTest::gtest_main ${BACKWARD_ENABLE})
add_backward(graph_topo_test)
