cmake_minimum_required(VERSION 3.10)
project(graph_topo)

file(GLOB_RECURSE GRAPH_TOPO_SRC src/*.cc src/*.cpp)
add_library(graph_topo STATIC ${GRAPH_TOPO_SRC})
target_link_libraries(graph_topo PUBLIC common)

file(GLOB_RECURSE GRAPH_TEST test/*.cpp)
add_executable(graph_topo_test ${GRAPH_TEST})
target_link_libraries(graph_topo_test graph_topo GTest::gtest_main ${BACKWARD_ENABLE})
add_backward(graph_topo_test)
