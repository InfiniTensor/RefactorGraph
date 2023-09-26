cmake_minimum_required(VERSION 3.10)
project(frontend)

file(GLOB_RECURSE FRONTEND_SRC src/*.cc src/*.cpp)
add_library(frontend ${LIBRARY_TYPE} ${FRONTEND_SRC})
target_link_libraries(frontend PUBLIC computation mem_manager)

file(GLOB_RECURSE FRONTEND_TEST test/*.cpp)
add_executable(frontend_test ${FRONTEND_TEST})
target_link_libraries(frontend_test frontend GTest::gtest_main ${BACKWARD_ENABLE})
add_backward(frontend_test)
