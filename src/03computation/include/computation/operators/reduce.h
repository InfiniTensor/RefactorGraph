#ifndef COMPUTATION_REDUCE_H
#define COMPUTATION_REDUCE_H

#include "../operator.h"
#include "common/error_handler.h"
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

    struct Reduce final : public Operator {
        ReduceType type;
        absl::InlinedVector<size_t, 4> axes;// empty means reduce all axes
        bool keepDims;

        Reduce(ReduceType type_,
               absl::InlinedVector<size_t, 4> axes_,
               bool keepDims_)
            : Operator(),
              type(type_),
              axes(std::move(axes_)),
              keepDims(keepDims_) {}

        static size_t typeId(ReduceType);
        size_t opTypeId() const override;
        std::string_view name() const override;
    };

}// namespace refactor::computation

#endif// COMPUTATION_REDUCE_H
