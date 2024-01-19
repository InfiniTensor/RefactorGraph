#ifndef COMPUTATION_PAD_H
#define COMPUTATION_PAD_H

#include "../operator.h"
#include "kernel/collectors/pad.h"

namespace refactor::computation {
    using kernel::PadsShape;
    using kernel::PadType;

    struct Pad final : public LayoutDependentOperator {
        PadsShape pads;
        PadType mode;

        Pad(decltype(pads), PadType) noexcept;

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_PAD_H
