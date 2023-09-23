#ifndef COMPUTATION_RANGE_H
#define COMPUTATION_RANGE_H

#include "../operator.h"

namespace refactor::computation {

    struct Range : public Operator {
        constexpr Range() : Operator() {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_RANGE_H
