#ifndef PYTHON_FFI_SEARCH_H
#define PYTHON_FFI_SEARCH_H

#include <pybind11/numpy.h>

namespace refactor::python_ffi {

    pybind11::array randomSearch(pybind11::array, int topK, float topP, float temperature);

}// namespace refactor::python_ffi

#endif// PYTHON_FFI_SEARCH_H
