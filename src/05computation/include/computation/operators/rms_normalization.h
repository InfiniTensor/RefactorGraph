#ifndef COMPUTATION_RMS_NORMALIZATION_H
#define COMPUTATION_RMS_NORMALIZATION_H

#include "../operator.h"

namespace refactor::computation {

    struct RmsNormalization final : public Operator {
        float epsilon;

        constexpr explicit RmsNormalization(float epsilon_) noexcept
            : Operator(), epsilon(epsilon_) {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_RMS_NORMALIZATION_H
