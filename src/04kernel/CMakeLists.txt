cmake_minimum_required(VERSION 3.10)
project(kernel)

file(GLOB_RECURSE KERNEL_SRC src/*.cc src/*.cpp)

add_library(kernel STATIC ${KERNEL_SRC})
target_link_libraries(kernel PUBLIC graph_topo runtime)
target_include_directories(kernel PUBLIC include)

# link tbb for some compiler to enable `std::execution::par` and `std::execution::par_unseq` in <execution> if found
find_library(TBB_LIB NAMES tbb)
if (TBB_LIB)
    message(STATUS "Found tbb: " ${TBB_LIB})
    target_link_libraries(kernel PUBLIC ${TBB_LIB})
else()
    message(STATUS "Could NOT found tbb")
endif()

if(USE_CUDA)
    file(GLOB_RECURSE KERNEL_CUDA_SRC src/*.cu)
    add_library(cuda_kernel STATIC ${KERNEL_CUDA_SRC})
    set_property(TARGET cuda_kernel PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(cuda_kernel PUBLIC include)
    target_link_libraries(cuda_kernel PUBLIC cudnn runtime)
    target_link_libraries(kernel PUBLIC cuda_kernel)
endif()
