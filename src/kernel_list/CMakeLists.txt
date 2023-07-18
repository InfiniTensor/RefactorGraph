cmake_minimum_required(VERSION 3.10)
project(kernel_list)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)
add_library(kernel_list src/kernel_list.cc)
target_link_libraries(kernel_list fmt core)
