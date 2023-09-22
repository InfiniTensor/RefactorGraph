#ifndef COMPUTATION_SPLIT_H
#define COMPUTATION_SPLIT_H

#include "../operator.h"

namespace refactor::computation {

    struct Split : public Operator {
        size_t axis, numOutputs;

        constexpr Split(size_t axis_, size_t numOutputs_)
            : Operator(), axis(axis_), numOutputs(numOutputs_) {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_SPLIT_H
