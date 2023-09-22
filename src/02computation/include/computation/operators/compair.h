#ifndef COMPUTATION_COMPAIR_H
#define COMPUTATION_COMPAIR_H

#include "../operator.h"

namespace refactor::computation {

    enum class CompairType {
        EQ,
        NE,
        LT,
        LE,
        GT,
        GE,
    };

    struct Compair : public Operator {
        CompairType type;
    };

}// namespace refactor::computation

#endif// COMPUTATION_COMPAIR_H
