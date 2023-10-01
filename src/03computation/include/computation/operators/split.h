#ifndef COMPUTATION_SPLIT_H
#define COMPUTATION_SPLIT_H

#include "../operator.h"

namespace refactor::computation {

    struct Split final : public AxisRankOperator {
        constexpr Split(uint32_t axis, uint32_t rank)
            : AxisRankOperator(axis, rank) {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SPLIT_H
