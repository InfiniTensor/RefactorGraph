#ifndef COMPUTATION_GLOBAL_POOL_H
#define COMPUTATION_GLOBAL_POOL_H

#include "../operator.h"
#include "pool.h"

namespace refactor::computation {

    struct GlobalPool final : public Operator {
        PoolType type;

        constexpr GlobalPool(PoolType type_) noexcept
            : Operator(), type(type_) {}

        static size_t typeId(PoolType) noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_GLOBAL_POOL_H
