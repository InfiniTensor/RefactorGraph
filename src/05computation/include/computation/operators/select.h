#ifndef COMPUTATION_SELECT_H
#define COMPUTATION_SELECT_H

#include "../operator.h"
#include "kernel/collectors/select.h"

namespace refactor::computation {
    using kernel::SelectType;

    struct Select final : public Operator {
        SelectType type;

        constexpr explicit Select(SelectType type_) noexcept
            : Operator(), type(type_) {}

        static size_t typeId(SelectType) noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SELECT_H
