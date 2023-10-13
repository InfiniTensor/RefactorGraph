cmake_minimum_required(VERSION 3.10)
project(runtime)

file(GLOB_RECURSE RUNTIME_SRC src/*.cc src/*.cpp)
add_library(runtime STATIC ${RUNTIME_SRC})
target_link_libraries(runtime PUBLIC graph_topo mem_manager)
target_include_directories(runtime PUBLIC include)
