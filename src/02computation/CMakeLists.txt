cmake_minimum_required(VERSION 3.10)
project(graph)

file(GLOB_RECURSE COMPUTATION_SRC src/*.cc src/*.cpp)
add_library(computation ${LIBRARY_TYPE} ${COMPUTATION_SRC})
target_link_libraries(computation PUBLIC common graph_topo)

# file(GLOB_RECURSE GRAPH_TEST test/*.cpp)
# add_executable(graph_test ${GRAPH_TEST})
# target_link_libraries(graph_test graph GTest::gtest_main ${BACKWARD_ENABLE})
# add_backward(graph_test)
