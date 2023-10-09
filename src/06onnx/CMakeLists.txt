cmake_minimum_required(VERSION 3.10)
project(onnx)

file(GLOB_RECURSE ONNX_SRC src/*.cc src/*.cpp)
add_library(onnx STATIC ${ONNX_SRC})
target_link_libraries(onnx PUBLIC frontend)

# file(GLOB_RECURSE ONNX_TEST test/*.cpp)
# add_executable(onnx_test ${ONNX_TEST})
# target_link_libraries(onnx_test onnx GTest::gtest_main ${BACKWARD_ENABLE})
# add_backward(onnx_test)
