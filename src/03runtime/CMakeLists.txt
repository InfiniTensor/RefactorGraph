cmake_minimum_required(VERSION 3.10)
project(runtime)

file(GLOB_RECURSE RUNTIME_SRC src/*.cc src/*.cpp)
add_library(runtime STATIC ${RUNTIME_SRC})
target_link_libraries(runtime PUBLIC graph_topo mem_manager)
target_include_directories(runtime PUBLIC include)

file(GLOB_RECURSE RUNTIME_TEST test/*.cpp)
if(RUNTIME_TEST)
    add_executable(runtime_test ${RUNTIME_TEST})
    target_link_libraries(runtime_test runtime GTest::gtest_main ${BACKWARD_ENABLE})
    add_backward(runtime_test)
endif()
