cmake_minimum_required(VERSION 3.10)
project(computation)

file(GLOB_RECURSE COMPUTATION_SRC src/*.cc src/*.cpp)
add_library(computation STATIC ${COMPUTATION_SRC})
target_link_libraries(computation PUBLIC kernel)
target_include_directories(computation PUBLIC include)

file(GLOB_RECURSE COMPUTATION_TEST test/*.cpp)
if(COMPUTATION_TEST)
    add_executable(computation_test ${COMPUTATION_TEST})
    target_link_libraries(computation_test computation GTest::gtest_main ${BACKWARD_ENABLE})
    add_backward(computation_test)
endif()
