cmake_minimum_required(VERSION 3.10)
project(kernel)

file(GLOB_RECURSE KERNEL_SRC src/*.cc src/*.cpp)
add_library(kernel STATIC ${KERNEL_SRC})
target_link_libraries(kernel PUBLIC mem_manager)
