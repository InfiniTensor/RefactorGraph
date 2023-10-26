cmake_minimum_required(VERSION 3.10)
project(kernel)

file(GLOB_RECURSE KERNEL_SRC src/*.cc src/*.cpp)
if(USE_CUDA)
    file(GLOB_RECURSE KERNEL_CUDA_SRC src/*.cu)
endif()

add_library(kernel STATIC ${KERNEL_SRC} ${KERNEL_CUDA_SRC})
target_link_libraries(kernel PUBLIC runtime)
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
    target_link_libraries(kernel PUBLIC cudnn)
endif()

file(GLOB_RECURSE KERNEL_TEST test/*.cpp)
if(KERNEL_TEST)
    add_executable(kernel_test ${KERNEL_TEST})
    target_link_libraries(kernel_test kernel GTest::gtest_main ${BACKWARD_ENABLE})
    add_backward(kernel_test)
endif()
