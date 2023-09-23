#include "functions.h"
#include "common/error_handler.h"
#include <pybind11/numpy.h>

namespace refactor::python_ffi {
    using namespace common;
    using namespace frontend;
    namespace py = pybind11;

    // A helper function that converts DataType to python format string
    std::string getFormat(DataType type) {

#define CASE(T)       \
    case DataType::T: \
        return py::format_descriptor<primitive_t<DataType::T>::type>::format();

        switch (type.internal) {
            CASE(F32);
            CASE(F64);
            CASE(I32);
            CASE(I64);
            CASE(I8);
            CASE(I16);
            CASE(U8);
            CASE(U16);
            CASE(U32);
            CASE(U64);
            case DataType::FP16:
            case DataType::BF16:
                // Python uses "e" for half precision float type code.
                // Check the following link for more information.
                // https://docs.python.org/3/library/struct.html#format-characters
                return "e";
            default:
                RUNTIME_ERROR("unsupported data type.");
        }
    }

    Shape dimVec2Shape(DimVec const &dims) {
        Shape shape(dims.size(), DimExpr(1));
        std::transform(dims.begin(), dims.end(), shape.begin(),
                       [](auto const &d) {
                           return std::holds_alternative<int64_t>(d)
                                      ? DimExpr(std::get<int64_t>(d))
                                      : DimExpr(std::get<std::string>(d));
                       });
        return shape;
    }

}// namespace refactor::python_ffi
