#ifndef COMPUTATION_GATHER_ELEMENTS_H
#define COMPUTATION_GATHER_ELEMENTS_H

#include "../operator.h"

namespace refactor::computation {

    struct GatherElements final : public AxisRankOperator {
        constexpr GatherElements(uint32_t axis, uint32_t rank) noexcept
            : AxisRankOperator(axis, rank) {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_GATHER_ELEMENTS_H
