cmake_minimum_required(VERSION 3.10)
project(kernel)

file(GLOB_RECURSE KERNEL_SRC src/*.cc src/*.cpp)

if(USE_CUDA)
    file(GLOB_RECURSE KERNEL_CUDA_SRC src/*.cu)
endif()

add_library(kernel STATIC ${KERNEL_SRC} ${KERNEL_CUDA_SRC})
target_link_libraries(kernel PUBLIC graph_topo mem_manager)

if(USE_CUDA)
set_target_properties(kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
