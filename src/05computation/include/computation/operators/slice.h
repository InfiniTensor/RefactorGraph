#ifndef COMPUTATION_SLICE_H
#define COMPUTATION_SLICE_H

#include "../operator.h"
#include "kernel/collectors/slice.h"

namespace refactor::computation {
    using Dimensions = kernel::Dimensions;

    struct Slice final : public LayoutDependentOperator {
        Dimensions dims;

        explicit Slice(Dimensions) noexcept;

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SLICE_H
