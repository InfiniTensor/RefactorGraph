cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(kernel_cuda)

file(GLOB_RECURSE KERNEL_CUDA_SUB_SRC src/*.cu)

add_library(kernel_cuda STATIC ${KERNEL_CUDA_SUB_SRC})
target_link_libraries(kernel_cuda PUBLIC common)
target_include_directories(kernel_cuda PUBLIC include)

file(GLOB_RECURSE KERNEL_CUDA_TEST test/*.cu)
if(KERNEL_CUDA_TEST)
    add_executable(kernel_cuda_test ${KERNEL_CUDA_TEST})
    add_test(kernel_cuda_test kernel_cuda_test)
    target_link_libraries(kernel_cuda_test kernel_cuda GTest::gtest_main Backward::Object)
endif()
