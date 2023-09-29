#ifndef COMPUTATION_SIMPLE_UNARY_H
#define COMPUTATION_SIMPLE_UNARY_H

#include "../operator.h"

namespace refactor::computation {

    enum class SimpleUnaryType {
        Abs,
        Acos,
        Acosh,
        Asin,
        Asinh,
        Atan,
        Atanh,
        Cos,
        Cosh,
        Sin,
        Sinh,
        Tan,
        Tanh,
        Relu,
        Sqrt,
        Sigmoid,
        Erf,
        Not,
    };

    struct SimpleUnary : public Operator {
        SimpleUnaryType type;

        constexpr explicit SimpleUnary(SimpleUnaryType type_)
            : Operator(), type(type_) {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_SIMPLE_UNARY_H
