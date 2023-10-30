#include "functions.h"
#include "common.h"
#include <execution>

namespace refactor::python_ffi {
    using namespace frontend;
    namespace py = pybind11;

    DataType parseNumpyDType(py::dtype const &dt) {

#define CASE(T)                                                   \
    if (dt.is(py::dtype::of<primitive_t<DataType::T>::type>())) { \
        return DataType::T;                                       \
    }

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
        CASE(Bool);

#undef CASE
        RUNTIME_ERROR("unsupported data type.");
    }

    // A helper function that converts DataType to python format string
    pybind11::dtype buildNumpyDType(DataType dt) {

#define CASE(T)       \
    case DataType::T: \
        return py::dtype::of<primitive_t<DataType::T>::type>();

        switch (dt.internal) {

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
            CASE(Bool);

            case DataType::FP16:
            case DataType::BF16:
                // Python uses "e" for half precision float type code.
                // Check the following link for more information.
                // https://docs.python.org/3/library/struct.html#format-characters
                return py::dtype("e");

            default:
                RUNTIME_ERROR("unsupported data type.");
        }

#undef CASE
    }

    Shape dimVec2Shape(DimVec const &dims) {
        Shape shape(dims.size(), DimExpr(1));
        std::transform(std::execution::unseq,
                       dims.begin(), dims.end(), shape.begin(),
                       [](auto const &d) {
                           return std::holds_alternative<int64_t>(d)
                                      ? DimExpr(std::get<int64_t>(d))
                                      : DimExpr(std::get<std::string>(d));
                       });
        return shape;
    }

}// namespace refactor::python_ffi
