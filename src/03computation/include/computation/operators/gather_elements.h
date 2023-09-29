#ifndef COMPUTATION_GATHER_ELEMENTS_H
#define COMPUTATION_GATHER_ELEMENTS_H

#include "../operator.h"

namespace refactor::computation {

    struct GatherElements : public Operator {
        size_t axis;

        constexpr explicit GatherElements(size_t axis_)
            : Operator(), axis(axis_) {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_GATHER_ELEMENTS_H
