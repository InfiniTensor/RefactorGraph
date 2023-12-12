#ifndef COMPUTATION_CAST_H
#define COMPUTATION_CAST_H

#include "../operator.h"

namespace refactor::computation {

    struct Cast final : public Operator {

        constexpr explicit Cast() noexcept : Operator() {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_CAST_H
