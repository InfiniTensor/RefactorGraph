#ifndef COMPUTATION_SPLIT_H
#define COMPUTATION_SPLIT_H

#include "../operator.h"

namespace refactor::computation {

    struct Split final : public AxisRankOperator {
        constexpr Split(uint32_t axis, uint32_t rank) noexcept
            : AxisRankOperator(axis, rank) {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SPLIT_H
