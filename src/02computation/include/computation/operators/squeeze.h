#ifndef COMPUTATION_SQUEEZE_H
#define COMPUTATION_SQUEEZE_H

#include "../operator.h"

namespace refactor::computation {

    struct Squeeze : public Operator {
        constexpr Squeeze() : Operator() {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_SQUEEZE_H
