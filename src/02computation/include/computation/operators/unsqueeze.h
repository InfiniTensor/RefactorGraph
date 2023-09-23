#ifndef COMPUTATION_UNSQUEEZE_H
#define COMPUTATION_UNSQUEEZE_H

#include "../operator.h"

namespace refactor::computation {

    struct Unsqueeze : public Operator {
        constexpr Unsqueeze() : Operator() {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_UNSQUEEZE_H
