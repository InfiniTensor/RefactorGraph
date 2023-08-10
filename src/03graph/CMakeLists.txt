cmake_minimum_required(VERSION 3.10)
project(graph)

include_directories(include)
add_library(graph src/graph.cc)
target_link_libraries(graph fmt common)
