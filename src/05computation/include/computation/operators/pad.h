#ifndef COMPUTATION_PAD_H
#define COMPUTATION_PAD_H

#include "../operator.h"
#include "kernel/collectors/pad.h"

namespace refactor::computation {
    using kernel::PadType;
    using Dimensions = kernel::PadDimension;

    struct Pad final : public LayoutDependentOperator {
        Dimensions dims;
        PadType mode;

        Pad(decltype(dims), PadType) noexcept;

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_PAD_H
