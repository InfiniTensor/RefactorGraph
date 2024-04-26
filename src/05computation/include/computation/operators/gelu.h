#ifndef COMPUTATION_GELU_H
#define COMPUTATION_GELU_H

#include "../operator.h"

namespace refactor::computation {

    struct Gelu final : public Operator {

        constexpr explicit Gelu() noexcept
            : Operator() {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        // kernel::CollectorBox candidateKernels(Target) const final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation
#endif//COMPUTATION_GELU_H