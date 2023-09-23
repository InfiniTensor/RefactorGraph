#ifndef COMPUTATION_RESHAPE_H
#define COMPUTATION_RESHAPE_H

#include "../operator.h"

namespace refactor::computation {

    struct Reshape : public Operator {
        bool allowzero;

        constexpr explicit Reshape(bool allowzero_)
            : Operator(), allowzero(allowzero_) {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_RESHAPE_H
