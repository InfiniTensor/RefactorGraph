cmake_minimum_required(VERSION 3.10)
project(kernel_library)

include_directories(include)
add_library(kernel_library src/kernel_list.cc)
target_link_libraries(kernel_library fmt common)
