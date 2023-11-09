#ifndef COMPUTATION_REDUCE_H
#define COMPUTATION_REDUCE_H

#include "../operator.h"
#include "kernel/collectors/reduce.h"
#include <absl/container/inlined_vector.h>

namespace refactor::computation {
    using kernel::ReduceType;

    struct Reduce final : public Operator {
        ReduceType type;
        kernel::Axes axes;// empty means reduce all axes
        uint32_t rank;
        bool keepDims;

        Reduce(ReduceType,
               kernel::Axes axes,
               uint32_t rank,
               bool keepDims) noexcept;

        static size_t typeId(ReduceType) noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        bool isLayoutDependent() const noexcept final;
        void transposeTo(LayoutType target) noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_REDUCE_H
