#ifndef COMPUTATION_GATHER_H
#define COMPUTATION_GATHER_H

#include "../operator.h"

namespace refactor::computation {

    struct Gather : public Operator {
        size_t axis;

        constexpr explicit Gather(size_t axis_)
            : Operator(), axis(axis_) {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_GATHER_H
