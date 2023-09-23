#ifndef PYTHON_FFI_FUNCTIONS_H
#define PYTHON_FFI_FUNCTIONS_H

#include "common/data_type.h"
#include "frontend/tensor.h"

namespace refactor::python_ffi {
    using DimVec = std::vector<std::variant<std::string, int64_t>>;

    std::string getFormat(common::DataType);
    frontend::Shape dimVec2Shape(DimVec const &);

}// namespace refactor::python_ffi

#endif// PYTHON_FFI_FUNCTIONS_H
