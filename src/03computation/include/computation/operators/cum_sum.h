#ifndef COMPUTATION_CUM_SUM_H
#define COMPUTATION_CUM_SUM_H

#include "../operator.h"
#include "common/data_type.h"

namespace refactor::computation {

    struct CumSum final : public Operator {
        bool exclusive, reverse;

        constexpr CumSum(bool exclusive_, bool reverse_)
            : Operator(), exclusive(exclusive_), reverse(reverse_) {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_CUM_SUM_H
