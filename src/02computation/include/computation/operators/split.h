#ifndef COMPUTATION_SPLIT_H
#define COMPUTATION_SPLIT_H

#include "../operator.h"

namespace refactor::computation {

    struct Split : public Operator {
        size_t axis;

        constexpr Split(size_t axis_) : Operator(), axis(axis_) {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_SPLIT_H
