cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(kernel VERSION 0.0.0 LANGUAGES CXX)
message(STATUS "Project " ${PROJECT_NAME} " version " ${PROJECT_VERSION})

file(GLOB_RECURSE KERNEL_SRC src/*.cc src/*.cpp)
if(USE_CUDA)
    file(GLOB_RECURSE KERNEL_CUDA_SRC src/*.cu)
    add_subdirectory(cuda)
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
    # cuda   for cuda driver
    # nvrtc  for cuda kernel compile
    # cublas for matmul
    # cudnn  for conv and others
    target_link_libraries(kernel PUBLIC cuda nvrtc cublas cudnn kernel_cuda)
    target_include_directories(kernel PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
if(USE_KUNLUN)
    include_directories(${KUNLUN_HOME}/XTDK/include/)
    find_library(KUNLUN_RT libxpurt.so ${KUNLUN_HOME}/lib64)
    find_library(KUNLUN_DNN libxpuapi.so ${KUNLUN_HOME}/XTDK/shlib)
    target_link_libraries(kernel PUBLIC ${KUNLUN_RT} ${KUNLUN_DNN})
endif()
if(USE_BANG)
    target_link_libraries(kernel PUBLIC ${CAMBRICON_CNNL} ${CAMBRICON_CNRT} ${CAMBRICON_CNDRV})
endif()

file(GLOB_RECURSE KERNEL_TEST test/*.cpp)
if(KERNEL_TEST)
    add_executable(kernel_test ${KERNEL_TEST})
    add_test(kernel_test kernel_test)
    target_link_libraries(kernel_test kernel GTest::gtest_main Backward::Object)
endif()
