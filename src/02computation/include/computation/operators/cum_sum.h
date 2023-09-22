#ifndef COMPUTATION_CUM_SUM_H
#define COMPUTATION_CUM_SUM_H

#include "../operator.h"
#include "common/data_type.h"

namespace refactor::computation {

    struct CumSum : public Operator {
        bool exclusive, reverse;
    };

}// namespace refactor::computation

#endif// COMPUTATION_CUM_SUM_H
