cmake_minimum_required(VERSION 3.10)
project(mem_manager)

file(GLOB_RECURSE MEM_MANAGER_SRC src/*.cc src/*.cpp)
add_library(mem_manager STATIC ${MEM_MANAGER_SRC})
target_link_libraries(mem_manager PUBLIC common)
target_include_directories(mem_manager PUBLIC include)
