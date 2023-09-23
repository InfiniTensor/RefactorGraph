#ifndef COMPUTATION_REDUCE_H
#define COMPUTATION_REDUCE_H

#include "../operator.h"

namespace refactor::computation {

    enum class ReduceType {
        Mean,
        L1,
        L2,
        LogSum,
        LogSumExp,
        Max,
        Min,
        Prod,
        Sum,
        SumSquare,
    };

    struct Reduce : public Operator {
        ReduceType type;
        bool keepDims, noopWithEmptyAxes;

        constexpr Reduce(ReduceType type_, bool keepDims_, bool noopWithEmptyAxes_)
            : Operator(), type(type_), keepDims(keepDims_), noopWithEmptyAxes(noopWithEmptyAxes_) {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_REDUCE_H
