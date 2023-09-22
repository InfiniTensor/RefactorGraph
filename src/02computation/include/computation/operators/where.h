#ifndef COMPUTATION_WHERE_H
#define COMPUTATION_WHERE_H

#include "../operator.h"

namespace refactor::computation {

    struct Where : public Operator {
        constexpr Where() : Operator() {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_WHERE_H
