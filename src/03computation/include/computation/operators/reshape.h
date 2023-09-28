#ifndef COMPUTATION_RESHAPE_H
#define COMPUTATION_RESHAPE_H

#include "../operator.h"

namespace refactor::computation {

    struct Reshape : public Operator {
        constexpr Reshape() : Operator() {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_RESHAPE_H
