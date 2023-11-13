#ifndef COMPUTATION_CONCAT_H
#define COMPUTATION_CONCAT_H

#include "../operator.h"

namespace refactor::computation {

    struct Concat final : public AxisRankOperator {
        constexpr Concat(uint32_t axis, uint32_t rank) noexcept
            : AxisRankOperator(axis, rank) {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_CONCAT_H
