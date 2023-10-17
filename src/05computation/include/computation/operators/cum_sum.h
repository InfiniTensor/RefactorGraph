#ifndef COMPUTATION_CUM_SUM_H
#define COMPUTATION_CUM_SUM_H

#include "../operator.h"

namespace refactor::computation {

    struct CumSum final : public Operator {
        bool exclusive, reverse;

        constexpr CumSum(bool exclusive_, bool reverse_) noexcept
            : Operator(), exclusive(exclusive_), reverse(reverse_) {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_CUM_SUM_H
