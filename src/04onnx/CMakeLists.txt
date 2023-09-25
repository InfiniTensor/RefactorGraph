cmake_minimum_required(VERSION 3.10)
project(onnx)

file(GLOB_RECURSE ONNX_SRC src/*.cc src/*.cpp)
add_library(onnx ${LIBRARY_TYPE} ${ONNX_SRC})
target_link_libraries(onnx PUBLIC frontend)
find_library(TBB_LIB NAMES tbb)
if (TBB_LIB) # link tbb for some compiler to enable <execution> if found
    message(STATUS "Found tbb: " ${TBB_LIB})
    target_link_libraries(onnx PUBLIC ${TBB_LIB})
else()
    message(STATUS "Could NOT found tbb")
endif()

file(GLOB_RECURSE ONNX_TEST test/*.cpp)
add_executable(onnx_test ${ONNX_TEST})
target_link_libraries(onnx_test onnx GTest::gtest_main ${BACKWARD_ENABLE})
add_backward(onnx_test)
