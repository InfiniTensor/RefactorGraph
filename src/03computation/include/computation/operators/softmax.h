#ifndef COMPUTATION_SOFTMAX_H
#define COMPUTATION_SOFTMAX_H

#include "../operator.h"

namespace refactor::computation {

    struct Softmax : public Operator {
        size_t axis;

        constexpr explicit Softmax(size_t axis_)
            : Operator(), axis(axis_) {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_SOFTMAX_H
