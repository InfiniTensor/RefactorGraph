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
        absl::InlinedVector<uint32_t, 4> axes;// empty means reduce all axes
        uint32_t rank;
        bool keepDims;

        Reduce(ReduceType,
               absl::InlinedVector<uint32_t, 4> axes,
               uint32_t rank,
               bool keepDims);

        static size_t typeId(ReduceType);
        size_t opTypeId() const final;
        std::string_view name() const final;
        bool isLayoutDependent() const final;
        void transposeTo(LayoutType target) final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_REDUCE_H
