#ifndef COMPUTATION_ATTENTION_H
#define COMPUTATION_ATTENTION_H

#include "../operator.h"

namespace refactor::computation {

    struct Attention final : public Operator {

        constexpr Attention() noexcept = default;

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_ATTENTION_H
