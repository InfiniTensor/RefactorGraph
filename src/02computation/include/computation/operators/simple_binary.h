#ifndef COMPUTATION_SIMPLE_BINARY_H
#define COMPUTATION_SIMPLE_BINARY_H

#include "../operator.h"

namespace refactor::computation {

    enum class SimpleBinaryType {
        Add,
        Sub,
        Mul,
        Div,
    };

    struct SimpleBinary : public Operator {
        SimpleBinaryType type;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SIMPLE_BINARY_H
