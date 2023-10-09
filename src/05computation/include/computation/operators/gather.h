#ifndef COMPUTATION_GATHER_H
#define COMPUTATION_GATHER_H

#include "../operator.h"

namespace refactor::computation {

    struct Gather final : public AxisRankOperator {
        constexpr Gather(uint32_t axis, uint32_t rank) noexcept
            : AxisRankOperator(axis, rank) {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_GATHER_H
