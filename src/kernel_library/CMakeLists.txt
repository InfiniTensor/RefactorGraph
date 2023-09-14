cmake_minimum_required(VERSION 3.10)
project(kernel_library)

file(GLOB_RECURSE KERNEL_LIBRAEY_SRC src/*.cc src/*.cpp)

add_library(kernel_library ${LIBRARY_TYPE} ${KERNEL_LIBRAEY_SRC})
target_link_libraries(kernel_library PUBLIC common)

# file(GLOB_RECURSE KERNEL_LIBRARY_TEST test/*.cpp)
# add_executable(kernel_library_test ${KERNEL_LIBRARY_TEST})
# target_link_libraries(kernel_library_test kernel_library GTest::gtest_main)
