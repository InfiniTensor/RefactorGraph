#ifndef PYTHON_FFI_FUNCTIONS_H
#define PYTHON_FFI_FUNCTIONS_H

#include "common/data_type.h"
#include "frontend/tensor.h"
#include <pybind11/numpy.h>

namespace refactor::python_ffi {
    using DimVec = std::vector<std::variant<std::string, int64_t>>;

    common::DataType parseNumpyDType(pybind11::dtype const &);
    pybind11::dtype buildNumpyDType(common::DataType);
    frontend::Shape dimVec2Shape(DimVec const &);

}// namespace refactor::python_ffi

#endif// PYTHON_FFI_FUNCTIONS_H
