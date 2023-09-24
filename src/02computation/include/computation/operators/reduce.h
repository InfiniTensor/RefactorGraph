#ifndef COMPUTATION_REDUCE_H
#define COMPUTATION_REDUCE_H

#include "../operator.h"
#include <absl/container/inlined_vector.h>

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
        absl::InlinedVector<size_t, 4> axes;
        bool keepDims, noopWithEmptyAxes;

        Reduce(ReduceType type_,
               absl::InlinedVector<size_t, 4> axes_,
               bool keepDims_,
               bool noopWithEmptyAxes_)
            : Operator(),
              type(type_),
              axes(std::move(axes_)),
              keepDims(keepDims_),
              noopWithEmptyAxes(noopWithEmptyAxes_) {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_REDUCE_H
