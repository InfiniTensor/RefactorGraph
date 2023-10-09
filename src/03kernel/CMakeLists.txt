cmake_minimum_required(VERSION 3.10)
project(kernel)

file(GLOB_RECURSE KERNEL_SRC src/*.cc src/*.cpp)

if(USE_CUDA)
    file(GLOB_RECURSE KERNEL_CUDA_SRC src/*.cu)
endif()

add_library(kernel STATIC ${KERNEL_SRC} ${KERNEL_CUDA_SRC})
target_link_libraries(kernel PUBLIC graph_topo mem_manager)

find_library(TBB_LIB NAMES tbb)
# link tbb for some compiler to enable `std::execution::par` and `std::execution::par_unseq` in <execution> if found
if (TBB_LIB)
    message(STATUS "Found tbb: " ${TBB_LIB})
    target_link_libraries(kernel PUBLIC ${TBB_LIB})
else()
    message(STATUS "Could NOT found tbb")
endif()

if(USE_CUDA)
set_property(TARGET kernel PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()
