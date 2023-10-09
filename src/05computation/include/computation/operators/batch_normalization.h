#ifndef COMPUTATION_BATCH_NORMALIZATION_H
#define COMPUTATION_BATCH_NORMALIZATION_H

#include "../operator.h"

namespace refactor::computation {

    struct BatchNormalization final : public Operator {
        float epsilon;

        constexpr explicit BatchNormalization(float epsilon_) noexcept
            : Operator(), epsilon(epsilon_) {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_BATCH_NORMALIZATION_H
