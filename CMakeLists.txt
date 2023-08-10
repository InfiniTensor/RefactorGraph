cmake_minimum_required(VERSION 3.10)
project(try_cpp_ai)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(3rd-party/fmt)
include_directories(3rd-party/fmt/include)

add_subdirectory("src/01common")
include_directories("src/01common/include")

add_subdirectory("src/02kernel_library")
include_directories("src/02kernel_library/include")

add_subdirectory("src/03graph")
include_directories("src/03graph/include")

add_executable(try_cpp_ai src/main.cpp)
target_link_libraries(try_cpp_ai common kernel_library)

add_executable(test test/test.cpp)
