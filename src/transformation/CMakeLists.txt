cmake_minimum_required(VERSION 3.10)
project(transformation)

file(GLOB_RECURSE GRAPH_SRC src/*.cc src/*.cpp)
add_library(transformation ${LIBRARY_TYPE} ${GRAPH_SRC})
target_link_libraries(transformation PUBLIC frontend)

file(GLOB_RECURSE TRANSFORMATION_TEST test/*.cpp)
add_executable(transformation_test ${TRANSFORMATION_TEST})
target_link_libraries(transformation_test transformation GTest::gtest_main ${BACKWARD_ENABLE})
add_backward(transformation_test)
