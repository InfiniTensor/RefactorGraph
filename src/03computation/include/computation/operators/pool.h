#ifndef COMPUTATION_POOL_H
#define COMPUTATION_POOL_H

#include "../operator.h"

namespace refactor::computation {

    enum class PoolType {
        Average,
        Lp,
        Max,
    };

    struct Pool final : public Operator {
        PoolType type;

        constexpr Pool(PoolType type_) noexcept
            : Operator(), type(type_) {}

        static size_t typeId(PoolType) noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_POOL_H
