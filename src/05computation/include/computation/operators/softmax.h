#ifndef COMPUTATION_SOFTMAX_H
#define COMPUTATION_SOFTMAX_H

#include "../operator.h"

namespace refactor::computation {

    struct Softmax final : public AxisRankOperator {
        constexpr Softmax(uint32_t axis, uint32_t rank) noexcept
            : AxisRankOperator(axis, rank) {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SOFTMAX_H
