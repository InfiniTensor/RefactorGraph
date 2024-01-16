#ifndef COMPUTATION_HARD_SIGMOID_H
#define COMPUTATION_HARD_SIGMOID_H

#include "../operator.h"

namespace refactor::computation {

    struct HardSigmoid final : public Operator {
        float alpha, beta;

        constexpr HardSigmoid(float alpha_, float beta_) noexcept
            : Operator(), alpha(alpha_), beta(beta_){};

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_HARD_SIGMOID_H
