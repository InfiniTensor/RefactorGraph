cmake_minimum_required(VERSION 3.10)
project(mem_manager)

file(GLOB_RECURSE MEM_MANAGER_SRC src/*.cc src/*.cpp)
add_library(mem_manager STATIC ${MEM_MANAGER_SRC})
target_link_libraries(mem_manager PUBLIC common)
target_include_directories(mem_manager PUBLIC include)

file(GLOB_RECURSE MEM_MANAGER_TEST test/*.cpp)
if(MEM_MANAGER_TEST)
    add_executable(mem_manager_test ${MEM_MANAGER_TEST})
    target_link_libraries(mem_manager_test mem_manager GTest::gtest_main ${BACKWARD_ENABLE})
    add_backward(mem_manager_test)
endif()
