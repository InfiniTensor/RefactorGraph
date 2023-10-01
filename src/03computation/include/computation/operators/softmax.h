#ifndef COMPUTATION_SOFTMAX_H
#define COMPUTATION_SOFTMAX_H

#include "../operator.h"

namespace refactor::computation {

    struct Softmax final : public AxisRankOperator {
        constexpr Softmax(uint32_t axis, uint32_t rank)
            : AxisRankOperator(axis, rank) {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SOFTMAX_H
