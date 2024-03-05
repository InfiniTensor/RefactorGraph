#ifndef COMPUTATION_ROPE_H
#define COMPUTATION_ROPE_H

#include "../operator.h"

namespace refactor::computation {

    struct RotaryPositionEmbedding final : public Operator {
        float theta;

        constexpr RotaryPositionEmbedding(float _theta) noexcept
            : Operator(), theta(_theta) {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif
