#ifndef COMPUTATION_SIMPLE_UNARY_H
#define COMPUTATION_SIMPLE_UNARY_H

#include "../operator.h"

namespace refactor::computation {

    enum class SimpleUnaryType {
        Relu,
        Sqrt,
        Tanh,
    };

    struct SimpleUnary : public Operator {
        SimpleUnaryType type;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SIMPLE_UNARY_H
