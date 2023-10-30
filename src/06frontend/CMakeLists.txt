cmake_minimum_required(VERSION 3.10)
project(frontend)

file(GLOB_RECURSE FRONTEND_SRC src/*.cc src/*.cpp)
add_library(frontend STATIC ${FRONTEND_SRC})
target_link_libraries(frontend PUBLIC computation)
target_include_directories(frontend PUBLIC include)

file(GLOB_RECURSE FRONTEND_TEST test/*.cpp)
if(FRONTEND_TEST)
    add_executable(frontend_test ${FRONTEND_TEST})
    target_link_libraries(frontend_test frontend GTest::gtest_main ${BACKWARD_ENABLE})
    add_backward(frontend_test)
endif()
