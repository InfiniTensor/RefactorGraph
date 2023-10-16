#ifndef COMPUTATION_SIMPLE_BINARY_H
#define COMPUTATION_SIMPLE_BINARY_H

#include "../operator.h"
#include "kernel/collectors/simple_binary.h"
#include "refactor/common.h"

namespace refactor::computation {
    using kernel::SimpleBinaryType;

    struct SimpleBinary final : public Operator {
        SimpleBinaryType type;

        constexpr explicit SimpleBinary(SimpleBinaryType type_) noexcept
            : Operator(), type(type_) {}

        static size_t typeId(SimpleBinaryType) noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SIMPLE_BINARY_H
