#ifndef COMPUTATION_IDENTITY_H
#define COMPUTATION_IDENTITY_H

#include "../operator.h"

namespace refactor::computation {

    struct Identity : public Operator {
        constexpr Identity() : Operator() {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_IDENTITY_H
