#ifndef COMPUTATION_EXPAND_H
#define COMPUTATION_EXPAND_H

#include "../operator.h"

namespace refactor::computation {

    struct Expand : public Operator {
        constexpr Expand() : Operator() {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_EXPAND_H
