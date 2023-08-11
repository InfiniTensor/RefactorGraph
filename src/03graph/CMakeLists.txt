cmake_minimum_required(VERSION 3.10)
project(graph)

include_directories(include)
file(GLOB_RECURSE GRAPH_SRC src/*.cc src/*.cpp)
add_library(graph ${GRAPH_SRC})
target_link_libraries(graph common fmt)
