cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(hardware VERSION 0.0.0 LANGUAGES CXX)
message(STATUS "Project " ${PROJECT_NAME} " version " ${PROJECT_VERSION})

file(GLOB_RECURSE HARDWARE_SRC src/*.cc src/*.cpp)
add_library(hardware STATIC ${HARDWARE_SRC} ${HARDWARE_CUDA_SRC})
target_link_libraries(hardware PUBLIC common)
target_include_directories(hardware PUBLIC include)

if(USE_CUDA)
    target_include_directories(hardware PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

file(GLOB_RECURSE HARDWARE_TEST test/*.cpp)
if(HARDWARE_TEST)
    add_executable(hardware_test ${HARDWARE_TEST})
    add_test(hardware_test hardware_test)
    target_link_libraries(hardware_test hardware GTest::gtest_main Backward::Object)
endif()
