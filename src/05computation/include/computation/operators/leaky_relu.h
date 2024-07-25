#ifndef COMPUTATION_LEAKY_RELU_H
#define COMPUTATION_LEAKY_RELU_H

#include "../operator.h"

namespace refactor::computation {

    struct LeakyRelu final : public Operator {
        float alpha;

        constexpr LeakyRelu(float alpha_) noexcept
            : Operator(), alpha(alpha_){};

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_LEAKY_RELU_H
