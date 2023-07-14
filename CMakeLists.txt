cmake_minimum_required(VERSION 3.10)
project(project_x)

set(CMAKE_CXX_STANDARD 20)

add_executable(project_x src/main.cpp)
add_executable(test test/test.cpp)
