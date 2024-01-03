cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(hardware VERSION 0.0.0 LANGUAGES CXX)
message(STATUS "Project " ${PROJECT_NAME} " version " ${PROJECT_VERSION})

# Source files
file(GLOB HARDWARE_SRC src/*.cc src/*.cpp src/devices/cpu/*.cc)

if(USE_CUDA)
    file(GLOB_RECURSE HARDWARE_CUDA_SRC src/devices/nvidia/*.cu src/devices/nvidia/*.cc)
endif()

if(USE_BANG)
    file(GLOB_RECURSE HARDWARE_BANG_SRC src/devices/mlu/*.cc)
endif()

add_library(hardware STATIC ${HARDWARE_SRC} ${HARDWARE_CUDA_SRC} ${HARDWARE_BANG_SRC})
target_link_libraries(hardware PUBLIC common)
target_include_directories(hardware PUBLIC include)

file(GLOB_RECURSE HARDWARE_TEST test/*.cpp)
if(HARDWARE_TEST)
    add_executable(hardware_test ${HARDWARE_TEST})
    add_test(hardware_test hardware_test)
    target_link_libraries(hardware_test hardware GTest::gtest_main ${BACKWARD_ENABLE})
    add_backward(hardware_test)
endif()
