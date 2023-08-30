cmake_minimum_required(VERSION 3.10)
project(iterator)

include_directories(include)
file(GLOB_RECURSE ITERATOR_SRC src/*.cc)
# add_library(iterator ${ITERATOR_SRC})
# target_link_libraries(iterator fmt)

add_executable(iterator_test test/test.cpp)
target_link_libraries(iterator_test fmt)
