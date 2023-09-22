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
    };

    struct SimpleUnary : public Operator {
        SimpleUnaryType type;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SIMPLE_UNARY_H
