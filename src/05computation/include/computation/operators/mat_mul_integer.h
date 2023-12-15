#ifndef COMPUTATION_MAT_MUL_INTEGER_H
#define COMPUTATION_MAT_MUL_INTEGER_H

#include "../operator.h"

namespace refactor::computation {

    struct MatMulInteger final : public LayoutDependentOperator {

        constexpr MatMulInteger() noexcept = default;

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// #ifndef COMPUTATION_MAT_MUL_INTEGER_H
