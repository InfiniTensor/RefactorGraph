#ifndef COMPUTATION_SIMPLE_UNARY_H
#define COMPUTATION_SIMPLE_UNARY_H

#include "../operator.h"
#include "kernel/collectors/simple_unary.h"

namespace refactor::computation {
    using kernel::SimpleUnaryType;

    struct SimpleUnary final : public Operator {
        SimpleUnaryType type;

        constexpr explicit SimpleUnary(SimpleUnaryType type_) noexcept
            : Operator(), type(type_) {}

        static size_t typeId(SimpleUnaryType) noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SIMPLE_UNARY_H
