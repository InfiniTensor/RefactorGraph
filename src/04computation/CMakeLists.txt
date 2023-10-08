cmake_minimum_required(VERSION 3.10)
project(computation)

file(GLOB_RECURSE COMPUTATION_SRC src/*.cc src/*.cpp)
add_library(computation STATIC ${COMPUTATION_SRC})
target_link_libraries(computation PUBLIC kernel)
