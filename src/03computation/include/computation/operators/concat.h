#ifndef COMPUTATION_CONCAT_H
#define COMPUTATION_CONCAT_H

#include "../operator.h"

namespace refactor::computation {

    struct Concat : public Operator {
        size_t axis;

        constexpr explicit Concat(size_t axis_)
            : Operator(), axis(axis_) {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_CONCAT_H
