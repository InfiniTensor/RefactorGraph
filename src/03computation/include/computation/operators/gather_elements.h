#ifndef COMPUTATION_GATHER_ELEMENTS_H
#define COMPUTATION_GATHER_ELEMENTS_H

#include "../operator.h"

namespace refactor::computation {

    struct GatherElements final : public AxisRankOperator {
        constexpr GatherElements(uint32_t axis, uint32_t rank)
            : AxisRankOperator(axis, rank) {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_GATHER_ELEMENTS_H
