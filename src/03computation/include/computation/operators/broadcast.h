#ifndef COMPUTATION_BROADCAST_H
#define COMPUTATION_BROADCAST_H

#include "../operator.h"

namespace refactor::computation {

    struct Broadcast : public Operator {
        constexpr Broadcast() : Operator() {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_BROADCAST_H
