#ifndef COMPUTATION_LAYER_NORMALIZATION_H
#define COMPUTATION_LAYER_NORMALIZATION_H

#include "../operator.h"

namespace refactor::computation {

    struct LayerNormalization final : public Operator {
        float epsilon;
        int axis;

        constexpr explicit LayerNormalization(float epsilon_, int axis_) noexcept
            : Operator(), epsilon(epsilon_), axis(axis_) {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        // kernel::CollectorBox candidateKernels(Target) const final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_LAYER_NORMALIZATION_H
