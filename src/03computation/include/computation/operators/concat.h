#ifndef COMPUTATION_CONCAT_H
#define COMPUTATION_CONCAT_H

#include "../operator.h"

namespace refactor::computation {

    struct Concat final : public AxisRankOperator {
        constexpr Concat(uint32_t axis, uint32_t rank)
            : AxisRankOperator(axis, rank) {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_CONCAT_H
