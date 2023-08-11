cmake_minimum_required(VERSION 3.10)
project(graph_topo)

include_directories(include)
# add_library(graph_topo src/core.cc)
# target_link_libraries(graph_topo fmt)

add_executable(graph_topo_test test/test.cpp)
target_link_libraries(graph_topo_test fmt)
