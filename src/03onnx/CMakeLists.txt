cmake_minimum_required(VERSION 3.10)
project(graph)

file(GLOB_RECURSE ONNX_SRC src/*.cc src/*.cpp)
add_library(onnx ${LIBRARY_TYPE} ${ONNX_SRC})
target_link_libraries(onnx PUBLIC graph)
